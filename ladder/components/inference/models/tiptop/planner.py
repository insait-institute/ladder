import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import torch

from ladder.components.inference.api.client.hlc.vlm.base import VLM
from ladder.components.inference.models.tiptop.config import TiptopPolicyConfig
from ladder.components.inference.models.tiptop.perception import (
    GraspModel,
    depth_to_xyz,
    ground,
    point_cloud,
    segment,
)
from ladder.components.inference.models.tiptop.scene import build_scene, goal_already_holds, label_key
from ladder.components.inference.models.tiptop.tamp import (
    build_solvers,
    build_tamp_config,
    create_environment,
    run_planning,
    serialize_plan,
)
from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)


def _dump(rgb, depth_m, intrinsic, world_from_cam, task, q_init, goal_atoms) -> None:
    directory = os.environ.get("TIPTOP_DUMP_DIR")
    if not directory:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    name = hashlib.blake2b(
        b"".join((str(task).encode(), np.ascontiguousarray(depth_m, dtype=np.float32).tobytes())),
        digest_size=8,
    ).hexdigest()
    target = path / f"{name}.npz"
    if target.exists():
        return
    np.savez_compressed(
        target,
        rgb=rgb,
        depth_m=depth_m,
        intrinsic=intrinsic,
        world_from_cam=world_from_cam,
        task=np.array(task),
        q_init=q_init,
        goal_atoms=np.array(json.dumps(goal_atoms or [])),
    )


class TiptopPlanner:
    def __init__(self, config: TiptopPolicyConfig, vlm: VLM) -> None:
        self._grasp_model = GraspModel()
        self._ik_solver, self._motion_gen, self._initial_world = build_solvers(config)
        self._config = config
        self._vlm = vlm
        self._lock = threading.Lock()
        self._tamp_config = build_tamp_config(config)
        LOGGER.info("TiPToP planning stack ready")

    def plan(
        self,
        rgb: npt.NDArray[np.uint8],
        depth_m: npt.NDArray[np.float32],
        intrinsic: npt.NDArray[np.float32],
        world_from_cam: npt.NDArray[np.float32],
        task: str,
        q_init: npt.NDArray[np.float32],
        goal_atoms: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            _dump(rgb, depth_m, intrinsic, world_from_cam, task, q_init, goal_atoms)
            try:
                return self._plan(rgb, depth_m, intrinsic, world_from_cam, task, q_init, goal_atoms)
            except Exception as error:  # noqa: BLE001
                LOGGER.exception("TiPToP planning failed", error)  # noqa: PLE1205
                return {
                    "success": False,
                    "plan": {},
                    "error": str(error) or repr(error),
                    "remaining_atoms": [],
                    "planned_atoms": [],
                }

    def _plan(
        self,
        rgb: npt.NDArray[np.uint8],
        depth_m: npt.NDArray[np.float32],
        intrinsic: npt.NDArray[np.float32],
        world_from_cam: npt.NDArray[np.float32],
        task: str,
        q_init: npt.NDArray[np.float32],
        goal_atoms: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        config = self._config
        seed = _request_seed(task, q_init, depth_m)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self._reset_world()
        depth = np.nan_to_num(depth_m.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth < 0] = 0.0
        depth[depth > config.depth_trunc_m] = 0.0
        xyz_map = depth_to_xyz(depth, intrinsic)
        xyz_map = xyz_map @ world_from_cam[:3, :3].T + world_from_cam[:3, 3]
        rgb_map = rgb.astype(np.float32) / 255.0
        bboxes, detected_atoms = ground(self._vlm, rgb, task, config)
        if not bboxes:
            raise ValueError("the VLM detected no objects in the scene")
        masks = segment(rgb, bboxes, config)
        cloud = point_cloud(xyz_map, rgb_map, config.voxel_downsample_size_m)
        grasps = self._grasp_model.predict(np.asarray(cloud.points), np.asarray(cloud.colors), config)
        scene = build_scene(xyz_map, rgb_map, masks, bboxes, grasps, config)
        atoms = list(detected_atoms) if goal_atoms is None else list(goal_atoms)
        if goal_atoms is not None:
            atoms = _relabel(atoms, scene)
            atoms = [
                atom
                for atom in atoms
                if not goal_already_holds(atom, scene.object_meshes, scene.table_cuboid)
            ]
        per_plan = config.goals_per_plan
        planned = atoms if per_plan is None else atoms[: int(per_plan)]
        remaining = [] if per_plan is None else atoms[int(per_plan) :]
        if goal_atoms is not None and (not planned) and (not remaining):
            return {
                "success": True,
                "plan": {"steps": []},
                "error": None,
                "remaining_atoms": [],
                "planned_atoms": [],
            }
        env, all_surfaces = create_environment(scene.object_meshes, scene.table_cuboid, planned, atoms)
        plan, _, failure = run_planning(
            env,
            self._tamp_config,
            config,
            q_init,
            self._ik_solver,
            scene.grasps,
            self._motion_gen,
            all_surfaces,
        )
        if plan is None:
            return {
                "success": False,
                "plan": {},
                "error": f"cuTAMP found no plan: {failure}",
                "remaining_atoms": remaining,
                "planned_atoms": planned,
            }
        return {
            "success": True,
            "plan": serialize_plan(plan, q_init, kin_model=self._motion_gen.kinematics),
            "error": None,
            "remaining_atoms": remaining,
            "planned_atoms": planned,
        }

    def _reset_world(self) -> None:
        self._ik_solver.update_world(self._initial_world)
        self._motion_gen.update_world(self._initial_world)
        self._motion_gen.reset(reset_seed=False)


def _relabel(atoms: List[Dict[str, Any]], scene: Any) -> List[Dict[str, Any]]:
    known = set(scene.object_meshes) | {scene.table_cuboid.name}
    by_key = {label_key(name): name for name in known}
    resolved = []
    for atom in atoms:
        args = [by_key.get(label_key(arg)) for arg in atom.get("args", ())]
        if all(args):
            resolved.append({**atom, "args": args})
        else:
            LOGGER.warn(f"dropping carried-over goal {atom}: not among this frame's detections")
    if not resolved:
        raise ValueError(f"none of the {len(atoms)} carried-over goals name a detected object")
    return resolved


def _request_seed(task: str, q_init: npt.NDArray[np.float32], depth_m: npt.NDArray[np.float32]) -> int:
    digest = hashlib.blake2b(
        b"".join(
            (
                str(task).encode(),
                np.ascontiguousarray(q_init, dtype=np.float32).tobytes(),
                np.ascontiguousarray(depth_m, dtype=np.float32).tobytes(),
            )
        ),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") % 2**31
