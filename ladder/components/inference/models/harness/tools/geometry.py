from typing import Tuple

import numpy as np

from ladder.components.inference.models.harness.perception import Detection, Scene

COLUMN_FLOOR_M = 0.02
LIFT_M = 0.10
MIN_ABOVE_SUPPORT_M = 0.004
PLACE_CLEARANCE_M = 0.02

_GRASP_HEIGHT_FRACTION = 0.5
_MIN_REACH_PRECISION_M = 0.003
_REACH_FRACTION = 0.2
_GRASP_BAND_FRACTION = 0.2
_RELEASE_GRID = 5


def column_radius(width_m: float) -> float:
    return max(0.5 * width_m, COLUMN_FLOOR_M)


def grasp_height(detection: Detection, grasp_xy: np.ndarray, width_m: float, support_z: float) -> float:
    radius = column_radius(width_m)
    local_top = detection.top_near(grasp_xy, radius)
    return max(
        support_z + MIN_ABOVE_SUPPORT_M,
        support_z + _GRASP_HEIGHT_FRACTION * (local_top - support_z),
        min(detection.foot_near(grasp_xy, radius), local_top - MIN_ABOVE_SUPPORT_M),
    )


def reach_windows(width_m: float, local_top_m: float, support_z: float) -> Tuple[float, float]:
    return (
        max(_MIN_REACH_PRECISION_M, _REACH_FRACTION * width_m),
        max(_MIN_REACH_PRECISION_M, _GRASP_BAND_FRACTION * (local_top_m - support_z)),
    )


def release_point(scene: Scene, destination: Detection, held_xy_m: np.ndarray) -> np.ndarray:
    radius = 0.5 * float(np.min(held_xy_m))
    reach = np.maximum(0.5 * destination.extent_m[:2] - radius, 0.0)
    steps = np.linspace(-1.0, 1.0, _RELEASE_GRID)
    offsets = np.stack(np.meshgrid(steps, steps, indexing="xy"), axis=-1).reshape(-1, 2)
    candidates = destination.center_m[:2] + offsets * reach
    measured = scene.column_top(candidates, max(radius, PLACE_CLEARANCE_M), float("nan"))
    tops = np.where(np.isnan(measured), destination.top_m, measured)
    # Rank by what is PILED above the destination's own median column, not by absolute height: an
    # unseen column has no right height, and filling to the destination top breaks every bin.
    level = float(np.nanmedian(measured)) if np.isfinite(measured).any() else destination.top_m
    piled = np.maximum(measured - level, 0.0)
    rung = np.round(np.where(np.isnan(piled), 0.0, piled) / PLACE_CLEARANCE_M)
    centred = np.linalg.norm(candidates - destination.center_m[:2], axis=1)
    best = int(np.lexsort((centred, rung))[0])
    return np.array([candidates[best][0], candidates[best][1], tops[best]])
