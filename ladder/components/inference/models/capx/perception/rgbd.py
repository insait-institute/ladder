import math
from typing import Dict, List, Optional, Protocol

import numpy as np

from ladder.components.inference.api.core.observation import CameraFrame
from ladder.components.inference.api.core.utils import rotmat_to_quat_wxyz
from ladder.components.inference.models.capx.perception import camera as camera_payload
from ladder.components.inference.models.capx.perception.scene import Detection
from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)

_SUPPORT_BIN_M = 0.005
_CLUSTER_BIN_M = 0.01
_CLUSTER_MIN_SHARE = 0.02
_CLUSTER_SPECKLE_SHARE = 0.002
_OUTLIER_NEIGHBOURS = 20
_OUTLIER_STD_RATIO = 2.0


class Grounder(Protocol):
    def box(self, image: np.ndarray, object_name: str) -> np.ndarray:
        ...


class Segmenter(Protocol):
    def mask(self, image: np.ndarray, box_xyxy: np.ndarray) -> Optional[np.ndarray]:
        ...


def mask_pixels(mask: np.ndarray, stride: int = 1) -> np.ndarray:
    rows, columns = np.nonzero(np.asarray(mask, dtype=bool))
    step = max(1, stride)
    return np.stack([columns[::step], rows[::step]], axis=1).astype(np.float64)


class RgbdScene:
    def __init__(
        self,
        observe,
        grounder: Grounder,
        camera: str,
        segmenter: Segmenter,
        support_margin_m: float = 0.008,
        min_points: int = 24,
        stride: int = 2,
        support_z_m: Optional[float] = None,
        box_fit: str = "obb",
        depth_range_m: Optional[List[float]] = None,
    ) -> None:
        self._observe = observe
        self._grounder = grounder
        self._camera = camera
        self._segmenter = segmenter
        self._support_margin_m = support_margin_m
        self._min_points = min_points
        self._stride = stride
        self._support_z_m: Optional[float] = support_z_m
        self._box_fit = box_fit
        self._depth_range_m = list(depth_range_m) if depth_range_m else None
        self._seen: Dict[str, Detection] = {}
        self._seen_for: Optional[int] = None

    def locate(self, object_name: str) -> Detection:
        observation = self._observe()
        if self._seen_for != id(observation):
            self._seen_for, self._seen = id(observation), {}
        elif object_name in self._seen:
            return self._seen[object_name]
        points = self._body(self._object_points(object_name), object_name)
        detection = (
            _fit_oriented_box(points)
            if self._box_fit == "obb"
            else _fit_upright_box(points, self._support_z_m or 0.0)
        )
        self._seen[object_name] = detection
        return detection

    def _object_points(self, object_name: str) -> np.ndarray:
        observation = self._observe()
        frames = observation.cameras
        images = observation.images or {}
        frame, image = self._frame(frames, images, self._camera, observation)
        self._support_z(frame)
        points, _ = self._view(frame, image, object_name)
        return points

    def _frame(self, frames, images, camera: str, observation):
        frame, image = frames.get(camera), (images or {}).get(camera)
        if frame is None or image is None:
            raise RuntimeError(
                f"no depth calibration and image for camera {camera!r} (published: "
                f"{camera_payload.names(observation.metadata) or 'none'}); the server must publish "
                "depth and intrinsics for the perception API to see anything"
            )
        return frame, image

    def _view(self, frame, image: np.ndarray, object_name: str):
        image = np.asarray(image)
        height, width = image.shape[:2]
        box = self._grounder.box(image, object_name)
        box = np.asarray(box, dtype=np.float64) * np.array([width, height, width, height])
        mask = self._segmenter.mask(image, box)
        if mask is not None:
            pixels = mask_pixels(mask, self._stride)
        else:
            pixels = camera_payload.box_pixels(box, (height, width), self._stride)
        rows, columns = frame.shape
        scaled = pixels * np.array([columns / width, rows / height])
        if self._depth_range_m is not None:
            camera = frame.unproject_camera(scaled)
            near, far = self._depth_range_m[0], self._depth_range_m[1]
            inside = np.isfinite(camera[:, 2]) & (camera[:, 2] >= near) & (camera[:, 2] <= far)
            scaled = scaled[inside]
        points = frame.unproject(scaled)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < self._min_points:
            raise ValueError(f"only {len(points)} depth samples fell on it")
        return points, scaled

    def _body(self, points: np.ndarray, object_name: str) -> np.ndarray:
        if len(points) < self._min_points:
            raise ValueError(
                f"only {len(points)} depth samples fell on the {object_name!r} the grounder found; "
                "it may be occluded or outside the depth range"
            )
        heights = points[:, 2] - (self._support_z_m or 0.0)
        standing = heights > self._support_margin_m
        if standing.sum() >= max(self._min_points, 0.15 * len(points)):
            points, heights = points[standing], heights[standing]
        return _drop_outliers(_lowest_body(points, heights, self._min_points))

    def _support_z(self, frame: CameraFrame) -> float:
        if self._support_z_m is None:
            height, width = frame.shape
            step = max(1, min(height, width) // 64)
            pixels = camera_payload.box_pixels([0, 0, width - 1, height - 1], frame.shape, step)
            z = frame.unproject(pixels)[:, 2]
            z = z[np.isfinite(z)]
            if not len(z):
                raise RuntimeError("the depth frame is entirely invalid; check the camera payload")
            low, high = np.percentile(z, [0.5, 99.0])
            if high - low < _SUPPORT_BIN_M:
                self._support_z_m = float(np.median(z))
            else:
                bins = int(np.ceil((high - low) / _SUPPORT_BIN_M))
                counts, edges = np.histogram(z, bins=bins, range=(low, high))
                floor = max(4.0, _CLUSTER_MIN_SHARE * counts.sum())
                occupied = [i for i, c in enumerate(counts) if c >= floor]
                peak = occupied[0] if occupied else int(np.argmax(counts))
                self._support_z_m = float(0.5 * (edges[peak] + edges[peak + 1]))
            LOGGER.info(f"RgbdScene: support surface at z={self._support_z_m:.3f} m")
        return self._support_z_m


def fit_obb(points: np.ndarray) -> Dict[str, np.ndarray]:
    kept = _statistical_outliers(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    centre = kept.mean(axis=0)
    centred = kept - centre
    values, vectors = np.linalg.eigh(centred.T @ centred / max(1, len(centred)))
    axes = vectors[:, np.argsort(values)[::-1]]
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] = -axes[:, -1]
    local = centred @ axes
    low, high = local.min(axis=0), local.max(axis=0)
    return {"center": centre + axes @ (0.5 * (low + high)), "extent": high - low, "R": axes}


def _fit_oriented_box(points: np.ndarray) -> Detection:
    box = fit_obb(points)
    return Detection(
        center_m=box["center"], quaternion_wxyz=rotmat_to_quat_wxyz(box["R"]), extent_m=box["extent"]
    )


def _statistical_outliers(points: np.ndarray) -> np.ndarray:
    if len(points) <= _OUTLIER_NEIGHBOURS:
        return points
    gaps = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    nearest = np.sort(gaps, axis=1)[:, 1 : _OUTLIER_NEIGHBOURS + 1].mean(axis=1)
    keep = nearest <= nearest.mean() + _OUTLIER_STD_RATIO * nearest.std()
    return points[keep] if keep.sum() >= 4 else points


def _lowest_body(points: np.ndarray, heights: np.ndarray, min_points: int) -> np.ndarray:
    if len(points) < min_points:
        return points
    span = float(heights.max() - heights.min())
    if span <= _CLUSTER_BIN_M:
        return points
    counts, edges = np.histogram(heights, bins=int(np.ceil(span / _CLUSTER_BIN_M)))
    floor = max(float(min_points), _CLUSTER_SPECKLE_SHARE * len(points))
    occupied = [i for i, c in enumerate(counts) if c >= floor]
    if not occupied:
        return points
    start = end = occupied[0]
    while end + 1 < len(counts) and counts[end + 1] > 0:
        end += 1
    keep = (heights >= edges[start]) & (heights <= edges[end + 1])
    return points[keep] if keep.sum() >= min_points else points


def _drop_outliers(points: np.ndarray, std_ratio: float = 2.0) -> np.ndarray:
    if len(points) < 8:
        return points
    distance = np.linalg.norm(points - points.mean(axis=0), axis=1)
    keep = distance <= distance.mean() + std_ratio * distance.std()
    return points[keep] if keep.any() else points


def _fit_upright_box(points: np.ndarray, support_z_m: float) -> Detection:
    xy = points[:, :2]
    centred = xy - xy.mean(axis=0)
    values, vectors = np.linalg.eigh(centred.T @ centred / max(1, len(centred)))
    axis = vectors[:, int(np.argmax(values))]
    yaw = (float(np.arctan2(axis[1], axis[0])) + 0.5 * math.pi) % math.pi - 0.5 * math.pi
    cos, sin = math.cos(-yaw), math.sin(-yaw)
    local = centred @ np.array([[cos, -sin], [sin, cos]]).T
    low, high = np.percentile(local, 2.0, axis=0), np.percentile(local, 98.0, axis=0)
    centre_local = 0.5 * (low + high)
    cos, sin = math.cos(yaw), math.sin(yaw)
    centre_xy = xy.mean(axis=0) + centre_local @ np.array([[cos, -sin], [sin, cos]]).T
    top_z = float(np.percentile(points[:, 2], 98.0))
    extent = np.array([high[0] - low[0], high[1] - low[1], max(top_z - support_z_m, 1e-3)])
    centre = np.array([centre_xy[0], centre_xy[1], 0.5 * (support_z_m + top_z)])
    return Detection(
        center_m=centre,
        quaternion_wxyz=np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]),
        extent_m=extent,
    )
