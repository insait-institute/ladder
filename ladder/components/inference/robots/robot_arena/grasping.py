"""
Force-based gripper control and weld-constraint grasping for Genesis simulation.

Follows the reference RobotArena implementation: when the gripper closes, per-substep
force control is applied via ``apply_safety_limits`` and contact forces are checked via
``is_grasping_two_finger``.  On successful grasp detection a weld constraint attaches the
object to the end-effector link so it moves with the robot.
"""

from typing import Optional, Tuple

import genesis
import numpy as np
import torch

from ladder.core.common.logger.logger import get_logger

LOGGER = get_logger(__name__)


def apply_safety_limits(
    command: torch.Tensor,
    q_measured: torch.Tensor,
    v_measured: torch.Tensor,
    motors_soft_position_lower: float,
    motors_soft_position_upper: float,
    motors_effort_limit: float,
    motors_velocity_limit: float,
    kp: float,
    kd: float,
) -> torch.Tensor:
    """Clip torque commands using soft position/velocity limits with kp/kd gains.

    Mirrors the reference RobotArena ``apply_safety_limits`` exactly.
    """
    device = command.device
    pos_lower = torch.tensor(motors_soft_position_lower, device=device)
    pos_upper = torch.tensor(motors_soft_position_upper, device=device)
    effort_lim = torch.tensor(motors_effort_limit, device=device)
    vel_lim = torch.tensor(motors_velocity_limit, device=device)
    kp_t = torch.tensor(kp, device=device)
    kd_t = torch.tensor(kd, device=device)

    safe_velocity_lower = vel_lim * torch.clip(-kp_t * (q_measured - pos_lower), -1.0, 1.0)
    safe_velocity_upper = vel_lim * torch.clip(-kp_t * (q_measured - pos_upper), -1.0, 1.0)

    safe_effort_lower = effort_lim * torch.clip(-kd_t * (v_measured - safe_velocity_lower), -1.0, 1.0)
    safe_effort_upper = effort_lim * torch.clip(-kd_t * (v_measured - safe_velocity_upper), -1.0, 1.0)

    return torch.clip(command, safe_effort_lower, safe_effort_upper)


def _quat_to_unit_vec(quat: torch.Tensor) -> torch.Tensor:
    """Convert a wxyz quaternion to a unit direction vector (local Z-axis)."""
    qw, qx, qy, qz = quat.unbind(-1)
    # Z-axis of the rotation matrix from quaternion
    z_x = 2 * (qx * qz + qw * qy)
    z_y = 2 * (qy * qz - qw * qx)
    z_z = 1 - 2 * (qx * qx + qy * qy)
    return torch.stack([z_x, z_y, z_z], dim=-1)


def is_grasping_two_finger(
    robot_entity: "genesis.Entity",
    obj_entity: "genesis.Entity",
    left_finger_idx: int,
    right_finger_idx: int,
    n_envs: int = 1,
    max_angle: float = 10.0,
    min_force: float = 0.05,
    max_plane_angle: float = 30.0,
) -> "torch.Tensor":
    """Per-env grasp detection. Returns a ``(n_envs,)`` bool tensor on the contacts'
    device (the batched checker keeps it on-GPU; the GripperController calls ``.cpu()``
    for its per-env Python loop).

    Mirrors the reference RobotArena ``is_grasping_two_finger``:
    1. Sum contact forces between each finger and the object.
    2. Verify forces are roughly in the gripper plane (not axial).
    3. Verify each finger has sufficient opposing force.

    Genesis returns contacts in two shapes depending on how the scene was
    built: ``(M, ...)`` (pre-filtered, no ``valid_mask``) for non-parallel
    scenes and ``(K, M_max, ...)`` (with ``valid_mask``) for parallel scenes.
    We lift the non-parallel shape to ``(1, M, ...)`` with an all-true mask
    so the body below runs once for both modes.
    """
    contacts = robot_entity.get_contacts(with_entity=obj_entity)
    if "valid_mask" in contacts:
        link_a, link_b = contacts["link_a"], contacts["link_b"]
        force_a, force_b = contacts["force_a"], contacts["force_b"]
        valid = contacts["valid_mask"]
        link_pos = robot_entity.get_links_pos()
        link_quats = robot_entity.get_links_quat()
    else:
        link_a, link_b = contacts["link_a"][None], contacts["link_b"][None]
        force_a, force_b = contacts["force_a"][None], contacts["force_b"][None]
        valid = torch.ones_like(link_a, dtype=torch.bool)
        link_pos = robot_entity.get_links_pos()[None]
        link_quats = robot_entity.get_links_quat()[None]

    if not bool(valid.any()):
        return torch.zeros(n_envs, dtype=torch.bool, device=valid.device)

    left_mask_a = ((link_a == left_finger_idx) & valid)[..., None]  # (K, M, 1)
    left_mask_b = ((link_b == left_finger_idx) & valid)[..., None]
    right_mask_a = ((link_a == right_finger_idx) & valid)[..., None]
    right_mask_b = ((link_b == right_finger_idx) & valid)[..., None]

    # Sum across the contact dim → per-env net forces (K, 3).
    left_forces = torch.sum(force_a * left_mask_a, dim=1) + torch.sum(force_b * left_mask_b, dim=1)
    right_forces = torch.sum(force_a * right_mask_a, dim=1) + torch.sum(force_b * right_mask_b, dim=1)

    left_local = left_finger_idx - robot_entity.link_start
    right_local = right_finger_idx - robot_entity.link_start
    gripper_vec = _quat_to_unit_vec(link_quats[:, left_local])  # (K, 3)
    plane_normal = torch.cross(gripper_vec, link_pos[:, left_local] - link_pos[:, right_local], dim=-1)

    left_cos_plane = torch.nn.functional.cosine_similarity(plane_normal, left_forces, dim=-1)  # (K,)
    right_cos_plane = torch.nn.functional.cosine_similarity(plane_normal, right_forces, dim=-1)
    cos_threshold = float(np.cos(np.deg2rad(90 - max_plane_angle)))
    forces_in_plane = (
        (-cos_threshold < left_cos_plane)
        & (left_cos_plane < cos_threshold)
        & (-cos_threshold < right_cos_plane)
        & (right_cos_plane < cos_threshold)
    )

    left_ang = torch.nn.functional.cosine_similarity(gripper_vec, left_forces, dim=-1)
    right_ang = torch.nn.functional.cosine_similarity(gripper_vec, right_forces, dim=-1)
    angle_cos = float(np.cos(np.deg2rad(max_angle)))
    has_left = (-angle_cos < left_ang) & (left_ang < angle_cos) & (left_forces.norm(dim=-1) > min_force)
    has_right = (-angle_cos < right_ang) & (right_ang < angle_cos) & (right_forces.norm(dim=-1) > min_force)

    return has_left & has_right & forces_in_plane


def find_finger_link_indices(
    robot_entity: "genesis.Entity",
    left_pattern: str = "left_finger",
    right_pattern: str = "right_finger",
) -> Tuple[int, int]:
    """Return global link indices for the left and right finger links."""
    left_idx: Optional[int] = None
    right_idx: Optional[int] = None
    for link in robot_entity.links:
        if left_pattern in link.name:
            left_idx = link.idx
        if right_pattern in link.name:
            right_idx = link.idx
    if left_idx is None or right_idx is None:
        raise ValueError(
            f"Could not find finger links matching '{left_pattern}'/'{right_pattern}' "
            f"in robot links: {[link.name for link in robot_entity.links]}"
        )
    return left_idx, right_idx


def single_env_view(arr: np.ndarray | torch.Tensor, expected_ndim: int = 1) -> np.ndarray:
    """Normalize a Genesis getter return to its single-env numpy view.

    Genesis getters return the per-env shape at K=1 and a K-prefixed shape
    at K>=2:
        ``get_pos``  → ``(3,)`` / ``(K, 3)``       (``expected_ndim=1``)
        ``get_AABB`` → ``(2, 3)`` / ``(K, 2, 3)``  (``expected_ndim=2``)
    At K>=2 envs are identical at build time (replication), so env 0 is
    canonical for overlap-resolution math.
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    arr = np.asarray(arr, dtype=np.float64)
    return arr[0] if arr.ndim > expected_ndim else arr


def resolve_overlap(
    entity1: "genesis.Entity",
    entity2: "genesis.Entity",
    buffer: float = 0.005,
) -> bool:
    """Lift the upper entity to clear the lower one if their AABBs overlap.

    K=1 and K>=2 share this path; at K>=2 we read env 0's AABB and write a
    1D ``set_pos`` that Genesis broadcasts to all envs (envs are identical
    at build time, so the correction is uniform).
    """
    aabb1_min, aabb1_max = single_env_view(entity1.get_AABB(), expected_ndim=2)
    aabb2_min, aabb2_max = single_env_view(entity2.get_AABB(), expected_ndim=2)

    overlap = (
        aabb1_min[0] <= aabb2_max[0]
        and aabb1_max[0] >= aabb2_min[0]
        and aabb1_min[1] <= aabb2_max[1]
        and aabb1_max[1] >= aabb2_min[1]
        and aabb1_min[2] <= aabb2_max[2]
        and aabb1_max[2] >= aabb2_min[2]
    )
    if not overlap:
        return False

    center_z_1 = (aabb1_min[2] + aabb1_max[2]) / 2.0
    center_z_2 = (aabb2_min[2] + aabb2_max[2]) / 2.0

    if center_z_1 >= center_z_2:
        upper_entity, upper_aabb_min, lower_aabb_max = entity1, aabb1_min, aabb2_max
    else:
        upper_entity, upper_aabb_min, lower_aabb_max = entity2, aabb2_min, aabb1_max

    lift_amount = (lower_aabb_max[2] + buffer) - upper_aabb_min[2]
    if lift_amount > 0:
        current_pos = single_env_view(upper_entity.get_pos())
        upper_entity.set_pos(
            [float(current_pos[0]), float(current_pos[1]), float(current_pos[2]) + float(lift_amount)]
        )
        return True

    return False
