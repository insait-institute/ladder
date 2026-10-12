from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from ladder.components.inference.api.core.observation import EFFECTOR_NAMES


class ArmActionType(str, Enum):
    NONE = "none"
    JOINT_ANGLES = "joint_angles"
    JOINT_VELOCITIES = "joint_velocities"
    END_EFFECTOR_POSE = "end_effector_pose"


class EndEffectorPoseDeltaMode(str, Enum):
    ROT_EEF_TRANS_BASE = "rot_eef_trans_base"
    ROT_EEF_TRANS_EEF = "rot_eef_trans_eef"
    ROT_EEF_BASE_ALIGNED_TRANS_BASE = "rot_eef_base_aligned_trans_base"


@dataclass(kw_only=True)
class ArmAction:
    type: ArmActionType = ArmActionType.NONE

    def __post_init__(self):
        assert self.type != ArmActionType.NONE

    @staticmethod
    def from_dict(arm_action_dict: Dict[str, Any]) -> "ArmAction":
        try:
            action_type = ArmActionType(arm_action_dict["type"])
        except KeyError:
            raise ValueError("Action type not specified.")
        except ValueError:
            raise ValueError(f"Invalid action type. {arm_action_dict['type']}")

        try:
            action_cls = {
                ArmActionType.JOINT_ANGLES: JointAnglesAction,
                ArmActionType.JOINT_VELOCITIES: JointVelocitiesAction,
                ArmActionType.END_EFFECTOR_POSE: EndEffectorPoseAction,
            }.get(action_type)
        except KeyError:
            raise ValueError(f"No action class found for action type {action_type}")
        return action_cls(**arm_action_dict)


@dataclass(kw_only=True)
class JointAnglesAction(ArmAction):
    type: ArmActionType = ArmActionType.JOINT_ANGLES
    angles: npt.NDArray[np.float32]  # [num_dofs]


@dataclass(kw_only=True)
class JointVelocitiesAction(ArmAction):
    type: ArmActionType = ArmActionType.JOINT_VELOCITIES
    velocities: npt.NDArray[np.float32]  # [num_dofs]


@dataclass(kw_only=True)
class EndEffectorPoseAction(ArmAction):
    """
    The end effector absolute pose is always with respect
    to the robot base frame.
    """

    type: ArmActionType = ArmActionType.END_EFFECTOR_POSE
    translation: npt.NDArray[np.float32]  # [3]
    rotation: npt.NDArray[np.float32]  # [3, 3]

    def __post_init__(self):
        super().__post_init__()
        assert self.rotation.shape == (3, 3)
        assert self.translation.shape == (3,)

    def to_homogeneous_matrix(self) -> npt.NDArray[np.float32]:
        return np.block([[self.rotation, self.translation.reshape(3, 1)], [0, 0, 0, 1]])

    @staticmethod
    def from_homogeneous_matrix(matrix: npt.NDArray[np.float32]) -> "EndEffectorPoseAction":
        return EndEffectorPoseAction(translation=matrix[:3, 3], rotation=matrix[:3, :3])

    def apply_delta(
        self,
        delta_rotation: npt.NDArray[np.float32],
        delta_translation: npt.NDArray[np.float32],
        delta_mode: str | EndEffectorPoseDeltaMode,
    ) -> "EndEffectorPoseAction":
        """
        Constructs a new pose action by applying a delta rotation and a delta translation
        (in the specified reference frame) to the current end effector pose.

        In the following we refer to `self.rotation` and `self.translation` as R_be and t_be.

        Let [R_be, t_be] be the current rotation and translation of the end effector frame with respect
        to the robot base frame, i.e. the arguments passed `apply_delta`.

        Case 1: delta_mode == EndEffectorPoseDeltaMode.ROT_EEF_TRANS_BASE

        The delta pose is [DR, Dt], where DR represents a rotations in the current end effector frame and
        Dt represents a translation in the robot base frame.
        Then, `apply_delta` will return:

            [R, t] = [R_be @ DR, t_be + Dt]

        Case 2: translation_reference_frame == EndEffectorPoseDeltaMode.ROT_EEF_TRANS_EEF

        The delta pose is [R_e_e1, t_e], where t_e represents a translation in the end effector frame.
        Then, `apply_delta` will return:

            [R, t] = [R_be @ R_e_e1, t_be + R_be @ t_e]

        Case 3: delta_mode == EndEffectorPoseDeltaMode.ROT_EEF_BASE_ALIGNED_TRANS_BASE

        The delta pose is [DR, Dt], where DR represents a rotation in the axis aligned end effector
        frame and Dt represents a translation in the robot base frame.

        Then, `apply_delta` will return:

            [R, t] = [DR @ R_be, t_be + Dt]

        Args:
            delta_rotation: The delta rotation of the end effector.
            delta_translation: The delta translation of the end effector.
            delta_mode: The mode of the delta pose.

        Returns:
            A new pose action representing the end effector pose after applying the delta pose.
        """
        delta_mode = EndEffectorPoseDeltaMode(delta_mode)

        if delta_mode == EndEffectorPoseDeltaMode.ROT_EEF_TRANS_BASE:
            rotation = self.rotation @ delta_rotation
            translation = delta_translation + self.translation
        elif delta_mode == EndEffectorPoseDeltaMode.ROT_EEF_TRANS_EEF:
            rotation = self.rotation @ delta_rotation
            translation = self.rotation @ delta_translation + self.translation
        elif delta_mode == EndEffectorPoseDeltaMode.ROT_EEF_BASE_ALIGNED_TRANS_BASE:
            rotation = delta_rotation @ self.rotation
            translation = delta_translation + self.translation
        else:
            raise ValueError(f"Invalid delta mode: {delta_mode}")
        return EndEffectorPoseAction(translation=translation, rotation=rotation)


class RobotActionType(str, Enum):
    NONE = "none"
    SINGLE_MANIPULATOR = "single_manipulator"
    DUAL_MANIPULATOR = "dual_manipulator"


@dataclass(kw_only=True)
class RobotAction:
    type: RobotActionType = RobotActionType.NONE

    def __post_init__(self):
        assert self.type != RobotActionType.NONE

    @staticmethod
    def from_dict(action_dict: Dict[str, Any]) -> "RobotAction":
        try:
            action_type = RobotActionType(action_dict["type"])
        except KeyError:
            raise ValueError("Robot action type not specified.")
        except ValueError:
            raise ValueError(f"Invalid robot action type. {action_dict['type']}")
        action_cls = {
            RobotActionType.SINGLE_MANIPULATOR: ManipulatorAction,
            RobotActionType.DUAL_MANIPULATOR: DualManipulatorAction,
        }.get(action_type)
        if action_cls is None:
            raise ValueError(f"No action class found for robot action type {action_type}")
        return action_cls.from_dict(action_dict)


@dataclass(kw_only=True)
class ManipulatorAction(RobotAction):
    """A single arm + its gripper (the former RobotAction)."""

    type: RobotActionType = RobotActionType.SINGLE_MANIPULATOR
    arm_action: Optional[ArmAction] = None
    gripper_action: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        if self.arm_action is None and self.gripper_action is None:
            raise ValueError("At least arm_action or gripper_action must be specified.")

    @staticmethod
    def from_dict(action_dict: Dict[str, Any]) -> "ManipulatorAction":
        arm_action = None
        if arm_action_dict := action_dict.get("arm_action", None):
            arm_action = ArmAction.from_dict(arm_action_dict)
        return ManipulatorAction(
            arm_action=arm_action,
            gripper_action=action_dict.get("gripper_action", None),
        )


@dataclass(kw_only=True)
class DualManipulatorAction(RobotAction):
    type: RobotActionType = RobotActionType.DUAL_MANIPULATOR
    right: ManipulatorAction
    left: ManipulatorAction

    @staticmethod
    def from_dict(action_dict: Dict[str, Any]) -> "DualManipulatorAction":
        return DualManipulatorAction(
            right=ManipulatorAction.from_dict(action_dict["right"]),
            left=ManipulatorAction.from_dict(action_dict["left"]),
        )


def effector_actions(action: RobotAction) -> List[ManipulatorAction]:
    """One entry per effector, right before left, mirroring
    :func:`ladder.components.inference.api.core.observation.effector_states`."""
    if isinstance(action, DualManipulatorAction):
        return [action.right, action.left]
    assert isinstance(action, ManipulatorAction), f"{type(action).__name__} commands no manipulator"
    return [action]


def named_effector_actions(action: RobotAction) -> List[Tuple[str, ManipulatorAction]]:
    """:func:`effector_actions` paired with the effector names."""
    return list(zip(EFFECTOR_NAMES, effector_actions(action), strict=False))
