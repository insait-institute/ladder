from typing import Dict, List, Tuple

import numpy as np

MIN_POINTS = 12
_EXTENT_PERCENTILE = 2.0
_BLOB_CELL_M = 0.012


def nearest_blob(points_m: np.ndarray, seed_xy: np.ndarray) -> np.ndarray:
    if points_m.shape[0] < MIN_POINTS:
        return points_m
    cells = np.floor(points_m[:, :2] / _BLOB_CELL_M).astype(np.int64)
    members: Dict[Tuple[int, int], List[int]] = {}
    for index, cell in enumerate(map(tuple, cells)):
        members.setdefault(cell, []).append(index)

    def _distance_to_seed(cell: Tuple[int, int]) -> float:
        centre = (np.array(cell) + 0.5) * _BLOB_CELL_M
        return float(np.hypot(centre[0] - seed_xy[0], centre[1] - seed_xy[1]))

    seed_cell = min(members, key=_distance_to_seed)
    seen = {seed_cell}
    frontier = [seed_cell]
    while frontier:
        cx, cy = frontier.pop()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbour = (cx + dx, cy + dy)
                if neighbour in members and neighbour not in seen:
                    seen.add(neighbour)
                    frontier.append(neighbour)
    return points_m[np.concatenate([members[c] for c in seen])]


def fit_box(points_m: np.ndarray, support_z: float) -> Tuple[np.ndarray, np.ndarray, float]:
    xy = points_m[:, :2]
    centred = xy - xy.mean(axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    yaw = float(np.arctan2(axes[0, 1], axes[0, 0]))
    rotated = centred @ np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]).T
    low_xy = np.percentile(rotated, _EXTENT_PERCENTILE, axis=0)
    high_xy = np.percentile(rotated, 100.0 - _EXTENT_PERCENTILE, axis=0)
    top = float(np.percentile(points_m[:, 2], 90))
    centre_xy = xy.mean(axis=0) + np.array(
        [
            np.cos(yaw) * (low_xy[0] + high_xy[0]) / 2 - np.sin(yaw) * (low_xy[1] + high_xy[1]) / 2,
            np.sin(yaw) * (low_xy[0] + high_xy[0]) / 2 + np.cos(yaw) * (low_xy[1] + high_xy[1]) / 2,
        ]
    )
    extent = np.array([high_xy[0] - low_xy[0], high_xy[1] - low_xy[1], max(top - support_z, 1e-3)])
    if extent[1] > extent[0]:
        extent = np.array([extent[1], extent[0], extent[2]])
        yaw += np.pi / 2
    yaw = float((yaw + np.pi / 2) % np.pi - np.pi / 2)
    return np.array([centre_xy[0], centre_xy[1], 0.5 * (top + support_z)]), extent, yaw
