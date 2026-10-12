import functools
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import m2t2 as m2t2_package
import numpy as np
import numpy.typing as npt
import open3d as o3d
import torch
from m2t2.dataset import collate
from m2t2.dataset_utils import sample_points
from m2t2.m2t2 import M2T2
from m2t2.train_utils import to_cpu, to_gpu
from omegaconf import OmegaConf
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy.spatial.transform import Rotation

from ladder.components.inference.api.client.hlc.vlm.base import VLM
from ladder.components.inference.models.tiptop import runfile
from ladder.components.inference.models.tiptop.config import TiptopPolicyConfig
from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)
_GROUNDING_PROMPT = """Perform two tasks on this image based on the task instruction: "{task_instruction}".

TASK 1 - OBJECT DETECTION:
Detect and return bounding boxes for objects in the image.
- DO NOT include the robot, robot gripper, or table surface
- DO NOT include objects or surfaces irrelevant to the task or too far away to matter (e.g. walls, things on the wall that are far away, things on the floor below the table, people who might be in the scene, etc.)
- Limit to 25 objects
- If an object appears multiple times, name them by unique characteristics (color, size, position, etc.). If they seem the same, then just use numbers (e.g. 'soda_can1' and 'soda_can2', ... for identical-looking soda cans)
- An object **cannot have the same name** as another under any circumstance.
- Format: normalized coordinates 0-1000 as integers

Be very careful to identify objects and name them in a way that's relevant to the task. If the task involves picking up a red apple, make sure that 'red' appears in the name of the apple.


TASK 2 - TASK TRANSLATION:
Translate this natural language instruction into some conjunction of formal predicates:

AVAILABLE PREDICATES:
- on(movable, surface): Object A is placed on top of object B
- holding(movable): The robot is holding the named object

It is very important that the goal is exact: use your visual recognition, common-sense and reasoning abilities to make sure the goal expression is perfectly accurate.

For instance - for the task "throw away the trash in the bin" when there is a bin, an open empty chips packet, an empty soda can, a closed and full soda bottle, and several full candy bars on the table, the goal should be:

"predicates": [
        {{"name": "on", "args": ["chips_packet", "bin"]}},
        {{"name": "on", "args": ["soda_can", "bin"]}},
]

This is because only the chips and soda are empty and clearly trash. Everything else is still usable!

Another example - for the task "pick up the apple" when there is a red apple, a banana, and a mug on the table, the goal should be:

"predicates": [
        {{"name": "holding", "args": ["red_apple"]}},
]

This is because the task is to pick up/grab/hold an object, not to place it somewhere.

The first argument of `on` must be something the robot can pick up. A surface that things are placed
ON - a plate, tray, towel, bin, box - can only ever be the SECOND argument. When an instruction
stages objects somewhere and then moves them on, a pronoun like "it" or "them" refers to those
OBJECTS and never to the surface they were staged on, and the goal states only where they END UP.

Another example - for the task "first put everything on the plate, then move it to the bin" with a
banana, a lemon and a lime on the table plus a plate and a bin, the goal should be:

"predicates": [
        {{"name": "on", "args": ["banana", "bin"]}},
        {{"name": "on", "args": ["lemon", "bin"]}},
        {{"name": "on", "args": ["lime", "bin"]}},
]

This is because "it" is the fruit, not the plate: the plate is where they pass through and the bin is
where they finish. `on(plate, bin)` would ask the robot to pick up the plate, which is not the task
and is not even expressible - the plate is a surface, not a movable.


Return a single JSON object with this structure (no code fencing):
{{
    "bboxes": [
        {{"box_2d": [ymin, xmin, ymax, xmax], "label": "object name"}},
        ...
    ],
    "predicates": [
        {{"name": "predicate_name", "args": ["object1", "object2"]}},
        {{"name": "predicate_name", "args": ["object1"]}},
        ...
    ]
}}

Use the object labels you detect in Task 1 when creating predicates in Task 2.
Only reference objects that you actually detected in the image."""


def _parse_grounding(reply: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    text = reply.strip()
    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "")
    elif text.startswith("```"):
        text = text.replace("```", "")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"the VLM returned a non-JSON grounding reply: {text[:400]}") from error
    bboxes = [box for box in parsed.get("bboxes", []) if len(box.get("box_2d", [])) == 4]
    atoms = [
        {"predicate": spec["name"], "args": [str(arg).replace(" ", "_") for arg in spec["args"]]}
        for spec in parsed.get("predicates", [])
        if spec.get("name") and spec.get("args")
    ]
    for box in bboxes:
        box["label"] = str(box["label"]).replace(" ", "_")
    return (bboxes, atoms)


def ground(
    vlm: VLM, rgb: npt.NDArray[np.uint8], task: str, config: TiptopPolicyConfig
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    image = Image.fromarray(rgb)
    scaled = image.resize((800, int(800 * image.size[1] / image.size[0])), Image.Resampling.LANCZOS)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": np.asarray(scaled)},
                {"type": "text", "text": _GROUNDING_PROMPT.format(task_instruction=task)},
            ],
        }
    ]
    for attempt in range(config.vlm_attempts):
        reply = vlm.complete(
            messages, config.vlm_max_tokens, config.vlm_temperature, config.vlm_reasoning_effort
        )
        if reply:
            return _parse_grounding(reply)
        if attempt + 1 < config.vlm_attempts:
            time.sleep(config.vlm_retry_base_s * 2**attempt)
            LOGGER.warn(f"empty grounding reply, retrying ({attempt + 2}/{config.vlm_attempts})")
    raise ValueError(f"the VLM returned an empty grounding reply {config.vlm_attempts} times")


@functools.lru_cache(maxsize=1)
def _sam2_predictor(model: str, config_name: str):
    checkpoint = runfile("weights", model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SAM2ImagePredictor(build_sam2(config_name, str(checkpoint), device=device))


def segment(
    rgb: npt.NDArray[np.uint8], bboxes: List[Dict[str, Any]], config: TiptopPolicyConfig
) -> npt.NDArray[np.bool_]:
    height, width = rgb.shape[:2]
    boxes = np.array(
        [
            [
                box["box_2d"][1] / 1000.0 * width,
                box["box_2d"][0] / 1000.0 * height,
                box["box_2d"][3] / 1000.0 * width,
                box["box_2d"][2] / 1000.0 * height,
            ]
            for box in bboxes
        ]
    )
    predictor = _sam2_predictor(config.sam2_model, config.sam2_config)
    predictor.set_image(Image.fromarray(rgb))
    masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=boxes, multimask_output=False)
    masks = np.asarray(masks)
    if masks.ndim == 3:
        masks = masks[None]
    return masks


def depth_to_xyz(
    depth_m: npt.NDArray[np.float32], intrinsic: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    fx, fy = (intrinsic[0, 0], intrinsic[1, 1])
    cx, cy = (intrinsic[0, 2], intrinsic[1, 2])
    height, width = depth_m.shape
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return np.stack([(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m], axis=-1)


def point_cloud(
    xyz_map: npt.NDArray[np.float32], rgb_map: npt.NDArray[np.float32], voxel_size: Optional[float]
) -> o3d.geometry.PointCloud:
    xyz = xyz_map.reshape(-1, 3)
    rgb = rgb_map.reshape(-1, 3)
    valid = ~np.isnan(xyz).any(axis=1)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz[valid])
    cloud.colors = o3d.utility.Vector3dVector(rgb[valid])
    return cloud if voxel_size is None else cloud.voxel_down_sample(voxel_size=voxel_size)


@functools.lru_cache(maxsize=4)
def m2t2_to_tool_transform(
    depth_offset_m: float, rotation_deg: Tuple[float, float, float]
) -> npt.NDArray[np.float64]:
    base_to_tcp = np.eye(4)
    base_to_tcp[2, 3] = 0.1034 + depth_offset_m
    to_tool = np.eye(4)
    to_tool[:3, :3] = Rotation.from_euler("xyz", rotation_deg, degrees=True).as_matrix()
    return base_to_tcp @ to_tool


class GraspModel:
    def __init__(self) -> None:
        weights = runfile("weights", "m2t2.pth")
        self._cfg = OmegaConf.load(Path(m2t2_package.__file__).absolute().parents[1] / "config.yaml")
        model = M2T2.from_config(self._cfg.m2t2)
        model.load_state_dict(torch.load(weights, map_location="cpu")["model"])
        self._model = model.cuda().eval()
        LOGGER.info(f"M2T2 loaded from {weights}")

    def predict(
        self, xyz: npt.NDArray[np.float64], rgb: npt.NDArray[np.float64], config: TiptopPolicyConfig
    ) -> Dict[str, Dict[str, npt.NDArray[np.float64]]]:
        points = torch.from_numpy(np.asarray(xyz, dtype=np.float32))
        colors = torch.from_numpy(np.asarray(rgb, dtype=np.float32))
        if config.m2t2_apply_bounds:
            inside = (
                (points[:, 0] >= 0.0)
                & (points[:, 0] <= 1.0)
                & (points[:, 1] >= -0.3)
                & (points[:, 1] <= 0.3)
                & (points[:, 2] >= -0.2)
                & (points[:, 2] <= 0.5)
            )
            points, colors = (points[inside], colors[inside])
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        inputs = torch.cat([points - points.mean(dim=0), (colors - mean) / std], dim=1)
        data = {
            "inputs": inputs,
            "points": points,
            "seg": torch.ones(points.shape[0], dtype=torch.long),
            "cam_pose": torch.eye(4).float(),
            "task": "pick",
            "object_inputs": torch.rand(1024, 6),
            "ee_pose": torch.eye(4).float(),
            "bottom_center": torch.zeros(3),
            "object_center": torch.zeros(3),
        }
        aggregated: Dict[str, List[Any]] = {"grasps": [], "grasp_confidence": [], "grasp_contacts": []}
        for _ in range(config.m2t2_num_runs):
            sampled = dict(data)
            indices = sample_points(data["points"], config.m2t2_num_points)
            sampled["inputs"] = data["inputs"][indices]
            sampled["points"] = data["points"][indices]
            sampled["seg"] = data["seg"][indices]
            sampled["object_inputs"] = data["object_inputs"][
                sample_points(data["object_inputs"], self._cfg.data.num_object_points)
            ]
            batch = collate([sampled])
            to_gpu(batch)
            with torch.no_grad():
                self._cfg.eval.mask_thresh = config.m2t2_mask_threshold
                self._cfg.eval.object_thresh = config.m2t2_mask_threshold
                outputs = self._model.infer(batch, self._cfg.eval)
            to_cpu(outputs)
            for key in aggregated:
                aggregated[key].extend(outputs[key][0])
        return _group_grasps(aggregated, config.m2t2_max_grasps)


def _group_grasps(outputs: Dict[str, List[Any]], max_grasps: int) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    triples = zip(outputs["grasps"], outputs["grasp_confidence"], outputs["grasp_contacts"])
    for index, (poses, confidences, contacts) in enumerate(triples):
        poses = np.asarray(poses)
        confidences = np.asarray(confidences)
        contacts = np.asarray(contacts)
        if len(poses) == 0:
            grouped[f"object_{index}"] = {
                "poses": np.zeros((0, 4, 4)),
                "confidences": np.zeros(0),
                "contacts": np.zeros((0, 3)),
            }
            continue
        if len(poses) > max_grasps:
            top = np.argsort(confidences)[-max_grasps:]
            poses, confidences, contacts = (poses[top], confidences[top], contacts[top])
        grouped[f"object_{index}"] = {"poses": poses, "confidences": confidences, "contacts": contacts}
    return grouped
