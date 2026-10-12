import dataclasses
import math
from typing import List, Tuple

import numpy as np

MIN_COLUMN_POINTS = 3


@dataclasses.dataclass(frozen=True)
class Detection:
    name: str
    center_m: np.ndarray
    extent_m: np.ndarray
    yaw_rad: float
    top_m: float
    points_m: np.ndarray

    def grip_point_m(self, max_width_m: float, samples: int = 5) -> Tuple[np.ndarray, float]:
        along = np.array([math.cos(self.yaw_rad), math.sin(self.yaw_rad)])
        sections = self.cross_section_m(samples)
        step = float(self.extent_m[0]) / samples
        middle = sections[len(sections) // 2][0]
        best, best_width = 0.0, -1.0
        for index, (width, _) in enumerate(sections):
            if width <= 0.0 or width > max_width_m:
                continue
            if width > best_width:
                best_width = width
                best = -0.5 * float(self.extent_m[0]) + (index + 0.5) * step
        if best_width <= 0.0:
            return self.center_m[:2], 0.0
        if 0.0 < middle <= max_width_m:
            return self.center_m[:2], middle
        return self.center_m[:2] + best * along, best_width

    def cross_section_m(self, samples: int = 5) -> List[Tuple[float, float]]:
        along = np.array([math.cos(self.yaw_rad), math.sin(self.yaw_rad)])
        across = np.array([-along[1], along[0]])
        t = (self.points_m[:, :2] - self.center_m[:2]) @ along
        step = float(self.extent_m[0]) / samples
        out: List[Tuple[float, float]] = []
        for i in range(samples):
            low = -0.5 * float(self.extent_m[0]) + i * step
            slab = self.points_m[(t >= low) & (t < low + step)]
            if slab.shape[0] < MIN_COLUMN_POINTS:
                out.append((0.0, 0.0))
                continue
            w = float(np.ptp((slab[:, :2] - self.center_m[:2]) @ across))
            out.append(
                (w, float(np.percentile(slab[:, 2], 90)) - self.center_m[2] + 0.5 * float(self.extent_m[2]))
            )
        return out

    def foot_near(self, point_xy: np.ndarray, radius_m: float) -> float:
        near = self._near(point_xy, radius_m)
        points = near if near.shape[0] >= MIN_COLUMN_POINTS else self.points_m
        return float(np.percentile(points[:, 2], 2))

    def top_near(self, point_xy: np.ndarray, radius_m: float) -> float:
        near = self._near(point_xy, radius_m)
        return float(np.percentile(near[:, 2], 90)) if near.shape[0] >= MIN_COLUMN_POINTS else self.top_m

    def _near(self, point_xy: np.ndarray, radius_m: float) -> np.ndarray:
        return self.points_m[np.linalg.norm(self.points_m[:, :2] - point_xy, axis=1) < radius_m]
