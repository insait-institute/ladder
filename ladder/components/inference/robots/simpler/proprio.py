import numpy as np
import scipy.spatial.transform

from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)

CANONICAL_TO_WIDOWX_ROTATION = np.array(
    [
        [0, 0, 1],
        [0, -1, 0],
        [1, 0, 0],
    ],
    dtype=np.float32,
)


def rotation_control_canonical_to_env(env_id: str, rotmat: np.ndarray) -> np.ndarray:
    if "widowx" in env_id:
        return rotmat @ CANONICAL_TO_WIDOWX_ROTATION
    raise NotImplementedError(env_id)


def gripper_control_canonical_to_env(env_id: str, gripper: float) -> np.ndarray:
    if "widowx" in env_id:
        return binary_to_sign(gripper)

    raise NotImplementedError(env_id)


def binary_to_sign(value: np.ndarray) -> np.ndarray:
    return 2.0 * (value > 0.5) - 1.0


def widowx_eef_pose_to_canonical(eef_pose: np.ndarray) -> np.ndarray:
    assert eef_pose.shape == (7,), eef_pose.shape
    ee_pose_translation: np.ndarray = eef_pose[:3]  # [3]
    ee_pose_rotation: np.ndarray = eef_pose[3:7]  # NOTE: wxyz

    ee_pose_rotation: np.ndarray = scipy.spatial.transform.Rotation.from_quat(
        ee_pose_rotation, scalar_first=True
    ).as_matrix()

    # When the robot is in neutral pose, with the gripper point down, the end effector frame
    # orientation w.r.t. the robot base frame reported by the WidowX SDK is different from the
    # CANONICAL convention used by this repo (CANONICAL matches Franka convention).
    # Here we convert the WidowX observation to the CANONICAL convention.

    # R_BE = R_BE' @ (R_EE')^-1
    ee_pose_rotation = ee_pose_rotation @ CANONICAL_TO_WIDOWX_ROTATION.T  # [3, 3]

    # NOTE: Additionally to above, keep in mind that the Bridge dataset was collected with EEF frame
    # aligned with robot base frame. To get compatible observations, you need to post-multiply the
    # observed rotation (returned by this function) by
    # np.array(
    #     [
    #         [1, 0, 0],
    #         [0, np.cos(np.pi), -np.sin(np.pi)],
    #         [0, np.sin(np.pi), np.cos(np.pi)],
    #     ],
    # )

    # For bridge dataset collection code see:
    # https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py#L203

    ee_pose = np.eye(4, dtype=np.float32)
    ee_pose[:3, 3] = ee_pose_translation
    ee_pose[:3, :3] = ee_pose_rotation

    return ee_pose


def quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.concatenate([quat[3:4], quat[:3]], axis=-1)


def quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.concatenate([quat[1:4], quat[0:1]], axis=-1)
