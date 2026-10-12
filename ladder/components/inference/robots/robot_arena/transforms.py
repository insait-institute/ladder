"""
Coordinate transforms between ladder's canonical (Franka) frame and Genesis robot frames.

The canonical convention used by this codebase matches the Franka EEF frame orientation.
Each robot in Genesis may report EEF poses in a different frame, so we need correction
rotation matrices to convert between them.

The WidowX correction rotation is verified to match the ManiSkill2 convention in
ladder/components/inference/robots/simpler/transforms.py.
"""

from enum import Enum
from typing import Dict, Tuple

import numpy as np
import scipy.spatial.transform

from ladder.core.common.logger.logger import get_logger

LOGGER = get_logger(__name__)


class RobotName(str, Enum):
    WIDOWX = "widowx"
    FRANKA = "franka"
    UR5 = "ur5"


def quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion from Genesis (w, x, y, z) to scipy (x, y, z, w) order."""
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=quat_wxyz.dtype)


def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion from scipy (x, y, z, w) to Genesis (w, x, y, z) order."""
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=quat_xyzw.dtype)


# Rotation from canonical (Franka) EEF frame to WidowX EEF frame in Genesis.
# Verified to match CANONICAL_TO_WIDOWX_ROTATION in simpler/transforms.py (ManiSkill2 WidowX).
CANONICAL_TO_WIDOWX_GENESIS_ROTATION = np.array(
    [
        [0, 0, 1],
        [0, -1, 0],
        [1, 0, 0],
    ],
    dtype=np.float32,
)

# Franka in Genesis should be near-identity since canonical IS Franka convention.
CANONICAL_TO_FRANKA_GENESIS_ROTATION = np.eye(3, dtype=np.float32)

# UR5 correction matrix: must be determined empirically from Genesis neutral pose.
# Placeholder identity until validated.
CANONICAL_TO_UR5_GENESIS_ROTATION = np.eye(3, dtype=np.float32)


_CANONICAL_TO_GENESIS_ROTATIONS: Dict[RobotName, np.ndarray] = {
    RobotName.WIDOWX: CANONICAL_TO_WIDOWX_GENESIS_ROTATION,
    RobotName.FRANKA: CANONICAL_TO_FRANKA_GENESIS_ROTATION,
    RobotName.UR5: CANONICAL_TO_UR5_GENESIS_ROTATION,
}


def _get_correction_rotation(robot_name: RobotName) -> np.ndarray:
    if robot_name not in _CANONICAL_TO_GENESIS_ROTATIONS:
        raise NotImplementedError(f"No correction rotation defined for robot '{robot_name}'")
    return _CANONICAL_TO_GENESIS_ROTATIONS[robot_name]


def rotation_control_canonical_to_genesis(robot_name: RobotName, rotmat: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix from canonical (Franka) frame to Genesis robot frame.

    Used when sending EE pose commands to Genesis: the policy outputs rotations in canonical
    frame, and this converts them to what Genesis expects for the specific robot.

    Args:
        robot_name: Robot identifier ("widowx", "franka", "ur5").
        rotmat: Rotation matrix [3, 3] in canonical frame.

    Returns:
        Rotation matrix [3, 3] in Genesis robot frame.
    """
    assert rotmat.shape == (3, 3), f"Expected shape (3, 3), got {rotmat.shape}"
    correction = _get_correction_rotation(robot_name)
    return rotmat @ correction


def rotation_genesis_to_canonical(robot_name: RobotName, rotmat: np.ndarray) -> np.ndarray:
    """Genesis robot-frame rotation -> canonical (Franka) frame. Accepts ``(3, 3)`` or ``(K, 3, 3)``."""
    return rotmat @ _get_correction_rotation(robot_name).T


def eef_poses_world_to_canonical(
    world_to_base: np.ndarray,
    robot_name: RobotName,
    ee_pos_world: np.ndarray,
    ee_quat_world_wxyz: np.ndarray,
) -> np.ndarray:
    """Batched world-frame EEF poses -> canonical. ``ee_pos_world`` ``(K, 3)`` +
    ``ee_quat_world_wxyz`` ``(K, 4)`` (wxyz) -> ``(K, 4, 4)`` float32."""
    k = ee_pos_world.shape[0]
    quat_xyzw = ee_quat_world_wxyz[:, [1, 2, 3, 0]]
    ee_world = np.tile(np.eye(4, dtype=np.float64), (k, 1, 1))
    ee_world[:, :3, :3] = scipy.spatial.transform.Rotation.from_quat(quat_xyzw).as_matrix()
    ee_world[:, :3, 3] = ee_pos_world
    ee_base = world_to_base @ ee_world
    poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    poses[:, :3, :3] = rotation_genesis_to_canonical(robot_name, ee_base[:, :3, :3].astype(np.float32))
    poses[:, :3, 3] = ee_base[:, :3, 3].astype(np.float32)
    return poses


def gripper_control_canonical_to_genesis(robot_name: RobotName, gripper: float) -> float:
    """
    Convert gripper command from canonical [1=open, 0=closed] to Genesis convention.

    Canonical gripper state is proportional to the distance between the gripper
    fingers: 1.0 means fully open, 0.0 means fully closed.

    Args:
        robot_name: Robot identifier.
        gripper: Gripper command in [0, 1] (1=open, 0=closed).

    Returns:
        Gripper command in Genesis convention (0=close, 1=open).
    """
    if robot_name in (RobotName.WIDOWX, RobotName.FRANKA, RobotName.UR5):
        # Genesis convention matches canonical: higher value = more open
        return float(gripper > 0.5)
    raise NotImplementedError(f"Gripper mapping not defined for robot '{robot_name}'")


def gripper_genesis_to_canonical(robot_name: RobotName, gripper_state: float) -> float:
    """
    Convert gripper state from Genesis to canonical [1=open, 0=closed].

    Canonical gripper state is proportional to the distance between the gripper
    fingers: 1.0 means fully open, 0.0 means fully closed.

    Args:
        robot_name: Robot identifier.
        gripper_state: Raw gripper state from Genesis.

    Returns:
        Gripper state in [0, 1] (1=open, 0=closed).
    """
    if robot_name in (RobotName.WIDOWX, RobotName.FRANKA, RobotName.UR5):
        return np.clip(float(gripper_state), 0.0, 1.0)
    raise NotImplementedError(f"Gripper mapping not defined for robot '{robot_name}'")


def canonical_pose_to_genesis_target(
    robot_name: RobotName, translation: np.ndarray, rotmat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a canonical EE pose action to Genesis IK target (position + quaternion).

    Args:
        robot_name: Robot identifier.
        translation: Target position [3] in canonical frame.
        rotmat: Target rotation [3, 3] in canonical frame.

    Returns:
        Tuple of (position [3], quaternion [4] in wxyz order) for Genesis IK.
    """
    rotmat_genesis = rotation_control_canonical_to_genesis(robot_name, rotmat)
    quat_xyzw = scipy.spatial.transform.Rotation.from_matrix(rotmat_genesis).as_quat()
    quat_wxyz = quat_xyzw_to_wxyz(quat_xyzw)
    return translation.astype(np.float32), quat_wxyz.astype(np.float32)
