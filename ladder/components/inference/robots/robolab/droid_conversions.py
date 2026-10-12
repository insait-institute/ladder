import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openpi_client import image_tools

from ladder.components.inference.api.core.action import EndEffectorPoseAction, JointAnglesAction, RobotAction
from ladder.components.inference.api.core.observation import RobotStateType
from ladder.components.inference.robots.robolab.franka_kinematics import FrankaKinematics
from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)


# Every Franka joint limit is within +-3.1 rad, so anything past this is a diverged articulation
# that merely has not reached NaN yet. Checking only for non-finite values would let huge joint
# readings through IK and feed the runaway back into the sim.
_JOINT_SANE_RAD = 6.5


# Per-env maintained EE target + FK/IK between ladder EE-pose actions and 8-DoF jointpos.
class DroidJointposConverter:
    def __init__(
        self,
        num_envs: int,
        max_joint_step_rad: float = 0.5,
        image_size: Tuple[int, int] = (224, 224),
    ) -> None:
        self._kinematics = FrankaKinematics()
        self._max_joint_step_rad = max_joint_step_rad
        self._image_width, self._image_height = image_size
        # Per-env maintained EE target (4x4); None => seed from the observed pose.
        self._target_poses: List[Optional[np.ndarray]] = [None] * num_envs
        self._degenerate: List[bool] = [False] * num_envs  # logged once per env per episode
        # Last finite in-range joint vector per env: the freeze target for a diverged env.
        # Commanding zeros instead would be a full-authority PD slam toward the upright pose,
        # which can turn a marginal trip into a runaway that carries into the next episode.
        self._last_sane: List[Optional[np.ndarray]] = [None] * num_envs

    def observation_dict(
        self,
        env_i: int,
        main_image: np.ndarray,
        wrist_image: np.ndarray,
        joints: np.ndarray,
        gripper_position: np.ndarray,
        metadata: Dict[str, Any],
        scene_image: Optional[np.ndarray] = None,
        right_image: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        # RobotObservation field dict (what RobotEnv reconstructs client-side).
        joints = np.asarray(joints, dtype=np.float32)
        eef_pose = self._kinematics.forward(joints).astype(np.float32)
        if self._target_poses[env_i] is None:
            self._target_poses[env_i] = eef_pose
        return {
            # Native-path parity: RoboLab's client resizes with the same call before serving.
            # The size is the consumer's, not the sim's.
            # "scene" is the depth camera's own framing, unpadded: "main" is letterboxed to a
            # square, so image coordinates on it do not map onto the depth frame. A consumer that
            # reads depth through the image needs "scene"; the padded inputs are unchanged.
            "images": {
                "main": image_tools.resize_with_pad(main_image, self._image_height, self._image_width),
                "wrist": image_tools.resize_with_pad(wrist_image, self._image_height, self._image_width),
                **({"scene": scene_image} if scene_image is not None else {}),
                **(
                    {}
                    if right_image is None
                    else {
                        "right": image_tools.resize_with_pad(
                            right_image, self._image_height, self._image_width
                        )
                    }
                ),
            },
            "state": {
                "type": RobotStateType.SINGLE_MANIPULATOR.value,
                "eef_pose": eef_pose,
                "eef_pose_from_target": self._target_poses[env_i],
                "gripper_state": 1.0
                - np.atleast_1d(np.asarray(gripper_position, dtype=np.float32)),  # RoboLab 0=open
                "joints": joints,
            },
            "timestamps": np.array([time.time()], dtype=np.float64),
            "metadata": metadata,
        }

    # Ladder action -> [7 clamped joint targets, binary gripper] (RoboLab wire). An end-effector
    # pose is solved with inverse kinematics; joint angles are already the commanded quantity. Both
    # go through the same divergence guard and per-step clamp below.
    def actions_to_jointpos(self, actions: List[RobotAction], current_joints: np.ndarray) -> np.ndarray:
        current = np.asarray(current_joints, dtype=np.float32)
        solve = [
            env_i
            for env_i, action in enumerate(actions)
            if isinstance(action.arm_action, EndEffectorPoseAction)
            and np.all(np.isfinite(current[env_i]))
            and np.abs(current[env_i]).max() <= _JOINT_SANE_RAD
        ]
        inverses = {}
        if solve:
            try:
                poses = np.stack([actions[env_i].arm_action.to_homogeneous_matrix() for env_i in solve])
                inverses = dict(zip(solve, self._kinematics.inverse(poses, current[solve]), strict=True))
            except Exception as error:  # noqa: BLE001
                LOGGER.warn(
                    f"envs {solve}: IK solve failed ({error!r}); freezing at current joints this step"
                )
        return np.stack(
            [
                self.action_to_jointpos(env_i, action, current[env_i], inverses.get(env_i, current[env_i]))
                for env_i, action in enumerate(actions)
            ]
        )

    def action_to_jointpos(
        self, env_i: int, action: RobotAction, current_joints: np.ndarray, inverse: np.ndarray
    ) -> np.ndarray:
        arm_action = action.arm_action
        if isinstance(arm_action, JointAnglesAction):
            target_pose = None
        elif isinstance(arm_action, EndEffectorPoseAction):
            target_pose = np.asarray(arm_action.to_homogeneous_matrix(), dtype=np.float64)
            self._target_poses[env_i] = target_pose.astype(np.float32)
        else:
            raise TypeError(f"unsupported arm action: {type(arm_action).__name__}")
        current = np.asarray(current_joints, dtype=np.float32)
        # An exploded env feeds NaN joints -> singular Jacobian -> LinAlgError that would kill the
        # whole multi-env server; command a constant for just that env instead (its episode times
        # out on its own), and freeze it at the current pose on any solver failure.
        if not np.all(np.isfinite(current)) or np.abs(current).max() > _JOINT_SANE_RAD:
            if not self._degenerate[env_i]:
                self._degenerate[env_i] = True
                LOGGER.warn(
                    f"env {env_i}: diverged joints (max |q| = {np.nanmax(np.abs(current)):.3g} rad); "
                    "freezing at last sane pose until reset"
                )
                LOGGER.warn(
                    f"env {env_i}: joint vector at trip: "
                    f"{np.array2string(current, precision=2, max_line_width=300)}"
                )
            frozen = self._last_sane[env_i]
            if frozen is None:
                frozen = np.clip(np.nan_to_num(current), -_JOINT_SANE_RAD, _JOINT_SANE_RAD)
            return np.concatenate([frozen, [0.0]]).astype(np.float32)
        self._degenerate[env_i] = False
        self._last_sane[env_i] = current.copy()
        if target_pose is None:
            inverse = np.asarray(arm_action.angles, dtype=np.float32)
        if not np.all(np.isfinite(inverse)):
            LOGGER.warn(f"env {env_i}: IK returned non-finite solution; freezing at current joints this step")
            inverse = current
        joint_step = np.clip(
            inverse - current,
            -self._max_joint_step_rad,
            self._max_joint_step_rad,
        )
        # Native-path parity: the serve flips (ladder 1.0=open -> RoboLab 0=open), then RoboLab's
        # client binarizes the chunk's gripper at 0.5 (Pi0DroidJointposClient._postprocess_chunk).
        gripper = 1.0 if (1.0 - float(action.gripper_action)) > 0.5 else 0.0
        return np.concatenate([current + joint_step, [gripper]]).astype(np.float32)
