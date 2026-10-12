from typing import Optional, Tuple

import numpy as np
import torch

from ladder.components.inference.robots.robot_arena.success_checker.types import SuccessThresholds


def _compute_camera_axes_xy(
    camera_extrinsics: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Camera image right/down axes projected onto world XY (for directional language)."""
    if camera_extrinsics is None:
        return np.array([1.0, 0.0]), np.array([0.0, -1.0])

    rotation = camera_extrinsics[:3, :3]
    image_right_world = rotation[0, :]
    image_down_world = rotation[1, :]

    right_xy = image_right_world[:2].copy()
    right_norm = np.linalg.norm(right_xy)
    right_xy = right_xy / right_norm if right_norm > 1e-6 else np.array([1.0, 0.0])

    down_xy = image_down_world[:2].copy()
    down_norm = np.linalg.norm(down_xy)
    down_xy = down_xy / down_norm if down_norm > 1e-6 else np.array([0.0, -1.0])

    return right_xy, down_xy


def _axis_to_tensor(axis_xy: Optional[np.ndarray], default: tuple, ref: torch.Tensor) -> torch.Tensor:
    if axis_xy is None:
        axis_xy = np.asarray(default, dtype=np.float64)
    return torch.as_tensor(np.asarray(axis_xy).reshape(2), dtype=ref.dtype, device=ref.device)


def is_on_top_of(
    target_pos: torch.Tensor,
    target_min: torch.Tensor,
    target_max: torch.Tensor,
    dest_min: torch.Tensor,
    dest_max: torch.Tensor,
    thresholds: SuccessThresholds,
) -> torch.Tensor:
    """Target resting on the destination top: height band + footprint overlap."""
    height_above = target_pos[:, 2] - dest_max[:, 2]
    h_ok = (height_above >= thresholds.on_top_height_min_m) & (height_above <= thresholds.on_top_height_max_m)
    overlap_x = torch.clamp(
        torch.minimum(target_max[:, 0], dest_max[:, 0]) - torch.maximum(target_min[:, 0], dest_min[:, 0]),
        min=0.0,
    )
    overlap_y = torch.clamp(
        torch.minimum(target_max[:, 1], dest_max[:, 1]) - torch.maximum(target_min[:, 1], dest_min[:, 1]),
        min=0.0,
    )
    overlap_area = overlap_x * overlap_y
    target_area = torch.clamp(
        (target_max[:, 0] - target_min[:, 0]) * (target_max[:, 1] - target_min[:, 1]), min=1e-9
    )
    dest_area = torch.clamp((dest_max[:, 0] - dest_min[:, 0]) * (dest_max[:, 1] - dest_min[:, 1]), min=1e-9)
    overlap_frac = overlap_area / torch.minimum(target_area, dest_area)
    return h_ok & (overlap_frac >= thresholds.on_top_overlap_min_frac)


def is_inside(
    target_pos: torch.Tensor,
    dest_min: torch.Tensor,
    dest_max: torch.Tensor,
    thresholds: SuccessThresholds,
) -> torch.Tensor:
    """Target center within the destination's inscribed-circle opening and Z band."""
    center_x = (dest_min[:, 0] + dest_max[:, 0]) / 2.0
    center_y = (dest_min[:, 1] + dest_max[:, 1]) / 2.0
    half_x = (dest_max[:, 0] - dest_min[:, 0]) / 2.0
    half_y = (dest_max[:, 1] - dest_min[:, 1]) / 2.0
    opening_radius = torch.minimum(half_x, half_y)
    dx = target_pos[:, 0] - center_x
    dy = target_pos[:, 1] - center_y
    xy_ok = (dx * dx + dy * dy) <= (opening_radius * opening_radius)
    z_ok = (target_pos[:, 2] >= dest_min[:, 2] - thresholds.inside_z_tolerance_m) & (
        target_pos[:, 2] <= dest_max[:, 2] + thresholds.inside_z_tolerance_m
    )
    return xy_ok & z_ok


def is_picked_up(
    target_pos: torch.Tensor, initial_z: torch.Tensor, thresholds: SuccessThresholds
) -> torch.Tensor:
    """Target lifted above its initial Z by more than pickup_height_m."""
    return target_pos[:, 2] > (initial_z + thresholds.pickup_height_m)


def is_relative_to(
    target_pos: torch.Tensor,
    ref_pos: torch.Tensor,
    direction: str,
    initial_target_pos: torch.Tensor,
    thresholds: SuccessThresholds,
    image_right_xy: Optional[np.ndarray] = None,
    image_down_xy: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Target in the camera-frame direction relative to a reference, displaced and near it."""
    right = _axis_to_tensor(image_right_xy, (1.0, 0.0), target_pos)
    down = _axis_to_tensor(image_down_xy, (0.0, -1.0), target_pos)
    displacement_xy = torch.linalg.vector_norm(target_pos[:, :2] - initial_target_pos[:, :2], dim=1)
    disp_ok = displacement_xy >= thresholds.displacement_min_m
    distance_to_ref = torch.linalg.vector_norm(target_pos[:, :2] - ref_pos[:, :2], dim=1)
    near_ok = distance_to_ref <= thresholds.relative_max_distance_m
    delta_xy = target_pos[:, :2] - ref_pos[:, :2]
    proj_right = delta_xy @ right
    proj_down = delta_xy @ down
    margin = thresholds.relative_margin_m
    if direction == "left":
        dir_ok = proj_right < -margin
    elif direction == "right":
        dir_ok = proj_right > margin
    elif direction in ("upper", "back"):
        dir_ok = proj_down < -margin
    elif direction in ("lower", "front"):
        dir_ok = proj_down > margin
    else:
        dir_ok = torch.zeros_like(disp_ok)
    return disp_ok & near_ok & dir_ok


def is_in_table_region(
    target_pos: torch.Tensor,
    region: str,
    table_bounds: tuple,
    initial_target_pos: torch.Tensor,
    thresholds: SuccessThresholds,
    image_right_xy: Optional[np.ndarray] = None,
    image_down_xy: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Target in the camera-frame table region, displaced from start, within table proximity."""
    right = _axis_to_tensor(image_right_xy, (1.0, 0.0), target_pos)
    down = _axis_to_tensor(image_down_xy, (0.0, -1.0), target_pos)
    displacement = torch.linalg.vector_norm(target_pos[:, :2] - initial_target_pos[:, :2], dim=1)
    disp_ok = displacement >= thresholds.displacement_min_m

    min_x, max_x, min_y, max_y = table_bounds
    table_center = torch.as_tensor(
        [(min_x + max_x) / 2.0, (min_y + max_y) / 2.0], dtype=target_pos.dtype, device=target_pos.device
    )
    delta = target_pos[:, :2] - table_center
    proj_right = delta @ right
    proj_down = delta @ down
    table_radius = float(np.linalg.norm(np.array([max_x - min_x, max_y - min_y])) / 2.0)
    distance_from_center = torch.linalg.vector_norm(delta, dim=1)
    prox_ok = distance_from_center <= (table_radius + thresholds.region_max_distance_m)

    sig = (table_radius / 2.0) * 0.3
    if region == "upper_left":
        region_ok = (proj_right < -sig) & (proj_down < -sig)
    elif region == "upper":
        region_ok = proj_down < -sig
    elif region == "upper_right":
        region_ok = (proj_right > sig) & (proj_down < -sig)
    elif region == "left":
        region_ok = proj_right < -sig
    elif region == "center":
        region_ok = (torch.abs(proj_right) < sig) & (torch.abs(proj_down) < sig)
    elif region == "right":
        region_ok = proj_right > sig
    elif region == "lower_left":
        region_ok = (proj_right < -sig) & (proj_down > sig)
    elif region == "lower":
        region_ok = proj_down > sig
    elif region == "lower_right":
        region_ok = (proj_right > sig) & (proj_down > sig)
    else:
        region_ok = torch.zeros_like(disp_ok)
    return disp_ok & prox_ok & region_ok


def is_between(
    target_pos: torch.Tensor, ref1_pos: torch.Tensor, ref2_pos: torch.Tensor, thresholds: SuccessThresholds
) -> torch.Tensor:
    """Target within between_tolerance_m of the XY midpoint of two references."""
    midpoint_xy = (ref1_pos[:, :2] + ref2_pos[:, :2]) / 2.0
    distance_to_midpoint = torch.linalg.vector_norm(target_pos[:, :2] - midpoint_xy, dim=1)
    return distance_to_midpoint < thresholds.between_tolerance_m


def is_near(target_pos: torch.Tensor, ref_pos: torch.Tensor, thresholds: SuccessThresholds) -> torch.Tensor:
    """Target within reach_distance_m of a reference object (XY)."""
    distance = torch.linalg.vector_norm(target_pos[:, :2] - ref_pos[:, :2], dim=1)
    return distance < thresholds.reach_distance_m
