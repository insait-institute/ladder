import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ladder.components.inference.api.client.hlc.vlm.base import VLM
from ladder.components.inference.api.core.observation import CameraFrame, RobotObservation
from ladder.components.inference.models.harness import prompts
from ladder.components.inference.models.harness.arm import Arm
from ladder.components.inference.models.harness.perception.boxes import Box, best_match, parse_boxes
from ladder.components.inference.models.harness.perception.detection import MIN_COLUMN_POINTS, Detection
from ladder.components.inference.models.harness.perception.fitting import MIN_POINTS, fit_box, nearest_blob
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import Config, Configurable

LOGGER = get_logger(__name__)


class ObjectNotFoundError(RuntimeError):
    ...


_DEPTH_RANGE_M = (0.05, 2.5)
_SUPPORT_BAND_M = 0.006
_OBJECT_DEPTH_BAND_M = 0.08
_SELF_FILTER_RADIUS_M = 0.12
_SELF_FILTER_MARGIN_M = 0.02
_CARRY_DEPTH_M = 0.05
_MAX_TABLE_OBSTACLE_HEIGHT_M = 0.25
_CLOUD_STRIDE = 4
_MAX_TOKENS = 1024


class SceneConfig(Config):
    camera: str = "main"
    support_z_m: Optional[float] = None
    dump_dir: Optional[str] = None
    dump_limit: int = 40
    # Ground a destination every time instead of re-fitting the box it first had. Containers need
    # it; open surfaces do not.
    reground_destinations: bool = False
    # Where the hand waits while a snapshot is taken, for a real camera the arm would block: depth
    # estimated from RGB places the arm itself wrongly, and it hides what is behind it.
    clear_view_position_m: Optional[List[float]] = None
    # Drop the rivals' pixels before picking the nearest surface, so a box overlapping a taller
    # neighbour is not located on its rim.
    rivals_before_nearest: bool = False


class Scene(Configurable[SceneConfig]):
    def __init__(self, config: SceneConfig, vlm: VLM, arm: Arm) -> None:
        super().__init__(config)
        self._vlm = vlm
        self._arm = arm
        self._observation: Optional[RobotObservation] = None
        self._eef_m = np.zeros(3)
        self._carrying = False
        self._detections: Dict[Tuple[str, bool], Detection] = {}
        self._destinations: List[Box] = []
        self._plane: Optional[float] = None
        self._planes: List[float] = []
        self._cloud: Optional[np.ndarray] = None
        self._dumped = 0
        # One Scene per episode: the start time keeps each episode's dumps apart.
        self._dump_prefix = f"{os.getpid()}_{time.strftime('%Y%m%d_%H%M%S')}"
        self._parked_at_step: Optional[int] = None

    def refresh(self) -> None:
        # Parked and not sent anywhere since: a sagging arm never reads as arrived, so parking again
        # would only repeat the move.
        if self.config.clear_view_position_m is not None and self._parked_at_step != self._arm.steps:
            clear_view = np.asarray(self.config.clear_view_position_m, dtype=np.float64)
            self._arm.transit_to(clear_view, max(float(self._arm.position_m[2]), float(clear_view[2])))
            self._parked_at_step = self._arm.steps
        self._observation = self._arm.observation
        self._eef_m = self._arm.position_m
        self._carrying = self._arm.openness() <= 0.5
        self._detections = {}
        self._plane = None
        self._cloud = None

    def frame(self) -> CameraFrame:
        name = self.config.camera
        frames = self._snapshot().cameras
        if name not in frames:
            raise ObjectNotFoundError(f"camera {name!r} publishes no depth; available: {sorted(frames)}")
        return frames[name]

    def image(self) -> np.ndarray:
        name = self.config.camera
        images = self._snapshot().images or {}
        if name not in images:
            raise ObjectNotFoundError(f"camera {name!r} has no image; have {sorted(images)}")
        return images[name]

    def cloud(self) -> np.ndarray:
        if self._cloud is None:
            frame = self.frame()
            columns, rows = np.meshgrid(
                np.arange(0, frame.shape[1], _CLOUD_STRIDE), np.arange(0, frame.shape[0], _CLOUD_STRIDE)
            )
            points = frame.unproject(np.stack([columns.ravel(), rows.ravel()], axis=1))
            points = points[self._in_range_mask(points)]
            self._cloud = points[self._not_the_arm(points)] if points.size else points
        return self._cloud

    def support_z(self) -> float:
        if self.config.support_z_m is not None:
            return float(self.config.support_z_m)
        if self._plane is None:
            z = self.cloud()[:, 2]
            if z.size < MIN_POINTS:
                raise ObjectNotFoundError("no usable depth, so the support plane cannot be estimated")
            counts, edges = np.histogram(z, bins=max(10, int(np.ptp(z) / 0.005) + 1))
            peak = int(np.argmax(counts))
            self._plane = float(np.median(z[(z >= edges[peak]) & (z <= edges[peak + 1])]))
            self._planes.append(self._plane)
        return float(np.median(self._planes))

    def locate(self, name: str, surface: bool = False) -> Detection:
        if (name, surface) in self._detections:
            return self._detections[(name, surface)]
        # Strict: a remembered box may only answer the name it was grounded for. It serves twice
        # below, as the cache and as the not-found fallback.
        seen = best_match(self._destinations, name, strict=True) if surface and self._destinations else None
        # A destination seen before is re-fitted against current depth but not asked about again:
        # an open surface does not move. A container does, which `reground_destinations` covers.
        if seen is not None and not self.config.reground_destinations:
            LOGGER.info(f"{name!r} was seen before; re-fitting its box against this frame")
            return self._accept(name, seen, [seen], surface)
        reply = self._ask(prompts.BOX.format(name=name))
        self._dump(name, surface, reply)
        boxes = parse_boxes(reply)
        target = best_match(boxes, name) if boxes else None
        if target is None and seen is not None:
            LOGGER.info(f"{name!r} was not pointed at; re-using the box it had when it was seen")
            target = seen
        if target is None:
            raise ObjectNotFoundError(
                f"{name!r} was not found in the scene"
                + (
                    "; what IS there is " + ", ".join(sorted({b.label for b in boxes if b.label})[:6])
                    if boxes
                    else ""
                )
            )
        return self._accept(name, target, boxes, surface)

    def _accept(self, name: str, target: Box, boxes: List[Box], surface: bool) -> Detection:
        found = self._fit(name, target, [box for box in boxes if box is not target], surface)
        if surface and target not in self._destinations:
            self._destinations.append(target)
        LOGGER.info(
            f"located {name!r}{' (surface)' if surface else ''}: centre "
            f"{np.round(found.center_m, 3).tolist()} extent {np.round(found.extent_m, 3).tolist()} "
            f"from {len(found.points_m)} points, {len(boxes)} box(es) "
            f"(eef {np.round(self._eef_m, 3).tolist()})"
        )
        self._detections[(name, surface)] = found
        return found

    def _ask(self, text: str) -> str:
        return self._vlm.complete(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": self.image()},
                        {"type": "text", "text": text},
                    ],
                }
            ],
            max_tokens=_MAX_TOKENS,
            temperature=0.0,
        )

    # One snapshot plus the model's answer is everything `Scene` reads, so the pair replays a
    # grounding exactly. Off unless `dump_dir` is set.
    def _dump(self, name: str, surface: bool, reply: str) -> None:
        self._write_dump(self._snapshot(), self._eef_m, name, surface, reply)

    def _write_dump(
        self, observation: RobotObservation, eef_m: np.ndarray, name: str, surface: bool, reply: str
    ) -> None:
        if not self.config.dump_dir or self._dumped >= self.config.dump_limit:
            return
        payload: Dict[str, Any] = {
            "name": name,
            "surface": surface,
            "reply": reply,
            "camera": self.config.camera,
            "eef_m": eef_m,
            "eef_pose": np.asarray(observation.state.eef_pose, dtype=np.float64),
            "gripper_state": np.asarray(observation.state.gripper_state, dtype=np.float64).reshape(-1),
        }
        for camera, image in (observation.images or {}).items():
            payload[f"image__{camera}"] = np.asarray(image)
        for camera, frame in observation.cameras.items():
            payload[f"depth__{camera}"] = frame.depth_m
            payload[f"intrinsics__{camera}"] = frame.intrinsics
            payload[f"base_from_camera__{camera}"] = frame.base_from_camera
        os.makedirs(self.config.dump_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(
                self.config.dump_dir,
                f"{self._dump_prefix}_{threading.current_thread().name}_{self._dumped:04d}.npz",
            ),
            **payload,
        )
        self._dumped += 1

    def column_top(
        self, centres_xy: np.ndarray, radius_m: float, unmeasured_m: Optional[float] = None
    ) -> np.ndarray:
        obstacles, support = self._obstacles()
        tops = np.full(len(centres_xy), support if unmeasured_m is None else unmeasured_m, dtype=np.float64)
        if obstacles.size == 0:
            return tops
        for index, centre in enumerate(np.asarray(centres_xy, dtype=np.float64).reshape(-1, 2)):
            near = obstacles[np.linalg.norm(obstacles[:, :2] - centre, axis=1) < radius_m]
            if near.shape[0] >= MIN_COLUMN_POINTS:
                tops[index] = float(np.percentile(near[:, 2], 90))
        return tops

    def clearance_z(self, start_m: np.ndarray, goal_m: np.ndarray, radius_m: float) -> float:
        if self.cloud().size == 0:
            return float(max(start_m[2], goal_m[2]))
        obstacles, support = self._obstacles()
        if obstacles.size == 0:
            return float(support)
        segment = (goal_m - start_m)[:2]
        length_sq = float(segment @ segment)
        offset = obstacles[:, :2] - start_m[:2]
        t = np.clip(offset @ segment / length_sq, 0.0, 1.0) if length_sq > 1e-9 else np.zeros(len(obstacles))
        nearest = start_m[:2] + t[:, None] * segment
        inside = np.linalg.norm(obstacles[:, :2] - nearest, axis=1) < radius_m
        return float(obstacles[inside, 2].max()) if inside.any() else float(support)

    def _snapshot(self) -> RobotObservation:
        if self._observation is None:
            self.refresh()
        return self._observation

    def _obstacles(self) -> Tuple[np.ndarray, float]:
        support = self.support_z()
        points = self.cloud()
        if points.size == 0:
            return points, support
        return points[points[:, 2] <= support + _MAX_TABLE_OBSTACLE_HEIGHT_M], support

    def _not_the_arm(self, points: np.ndarray) -> np.ndarray:
        beside = np.linalg.norm(points[:, :2] - self._eef_m[:2], axis=1) < _SELF_FILTER_RADIUS_M
        drop = _CARRY_DEPTH_M if self._carrying else _SELF_FILTER_MARGIN_M
        return ~(beside & (points[:, 2] > self._eef_m[2] - drop))

    def _in_range_mask(self, points: np.ndarray) -> np.ndarray:
        camera_m = self.frame().base_from_camera[:3, 3]
        finite = np.isfinite(points).all(axis=1)
        distance = np.linalg.norm(np.where(finite[:, None], points, 0.0) - camera_m, axis=1)
        low, high = _DEPTH_RANGE_M
        return finite & (distance > low) & (distance < high)

    def _fit(self, name: str, box: Box, rivals: List[Box], surface: bool) -> Detection:
        y0, x0, y1, x1 = box.corners
        frame = self.frame()
        rows = np.arange(int(y0 / 1000 * frame.shape[0]), int(np.ceil(y1 / 1000 * frame.shape[0])) + 1)
        columns = np.arange(int(x0 / 1000 * frame.shape[1]), int(np.ceil(x1 / 1000 * frame.shape[1])) + 1)
        rows = rows[(rows >= 0) & (rows < frame.shape[0])]
        columns = columns[(columns >= 0) & (columns < frame.shape[1])]
        if columns.size == 0 or rows.size == 0:
            raise ObjectNotFoundError(f"the box for {name!r} falls outside the depth image")
        grid = np.meshgrid(columns, rows, indexing="xy")
        pixels = np.stack([grid[0].ravel(), grid[1].ravel()], axis=1)
        points = frame.unproject(pixels)
        pixels, points = _survivors(pixels, points, self._in_range_mask(points))
        if points.size:
            pixels, points = _survivors(pixels, points, self._not_the_arm(points))
        support = self.support_z()
        if not surface:
            pixels, points = _survivors(pixels, points, points[:, 2] > support + _SUPPORT_BAND_M)
        if points.shape[0] < MIN_POINTS:
            raise ObjectNotFoundError(f"{name!r} has no usable depth")
        if self.config.rivals_before_nearest:
            pixels, points = _without_rivals(pixels, points, rivals, frame.shape)
        axis = frame.base_from_camera[:3, 2]
        depth = (points - frame.base_from_camera[:3, 3]) @ axis
        pixels, points = _survivors(pixels, points, depth < depth.min() + _OBJECT_DEPTH_BAND_M)
        if not self.config.rivals_before_nearest:
            pixels, points = _without_rivals(pixels, points, rivals, frame.shape)
        if rivals and points.shape[0] < MIN_POINTS:
            raise ObjectNotFoundError(f"{name!r} is hidden by the other things pointed at")
        seed = np.array([(x0 + x1) / 2000 * frame.shape[1], (y0 + y1) / 2000 * frame.shape[0]])
        middle = frame.unproject(seed[None, :])[0]
        if np.isfinite(middle).all():
            points = nearest_blob(points, middle[:2])
        centre, extent, yaw = fit_box(points, support)
        # A destination's cloud is one oblique face, so its centroid leans toward the camera; the
        # ray through the middle of the box does not. Trust the ray only where it lands on the
        # destination, or a shallow view releases over the table instead.
        if surface and np.isfinite(middle).all():
            if (np.abs(middle[:2] - centre[:2]) <= 0.5 * extent[:2]).all():
                centre = np.array([middle[0], middle[1], centre[2]])
        return Detection(
            name=name,
            center_m=centre,
            extent_m=extent,
            yaw_rad=yaw,
            top_m=float(np.percentile(points[:, 2], 90)),
            points_m=points,
        )


def _survivors(pixels: np.ndarray, points: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return pixels[mask], points[mask]


def _without_rivals(
    pixels: np.ndarray, points: np.ndarray, rivals: List[Box], shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    for rival in rivals:
        ry0, rx0, ry1, rx1 = rival.corners
        inside = (
            (pixels[:, 0] >= rx0 / 1000 * shape[1])
            & (pixels[:, 0] <= rx1 / 1000 * shape[1])
            & (pixels[:, 1] >= ry0 / 1000 * shape[0])
            & (pixels[:, 1] <= ry1 / 1000 * shape[0])
        )
        if inside.all():
            continue
        pixels, points = _survivors(pixels, points, ~inside)
    return pixels, points
