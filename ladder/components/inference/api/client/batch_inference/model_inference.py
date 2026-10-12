import abc
import dataclasses
from typing import Any, Dict, List, Optional, TypeVar

import numpy as np

from ladder.components.inference.api.client.batch_inference.batch_protocol import PolicyMetadataPayload
from ladder.core.databib.config import Config, Configurable
from ladder.core.databib.template import Template


def _as_array(value: Optional[Any]) -> Optional[np.ndarray]:
    return None if value is None else np.asarray(value)


@dataclasses.dataclass
class ArmsInferenceObservation:
    """Arm(s) observation for one env. Mirrors the training ``ArmsDataInput`` without its batch axis"""

    ee_pose_translation: Optional[np.ndarray]  # shape: [H, num_effectors, 3]; np.float32
    ee_pose_rotation: Optional[np.ndarray]  # shape: [H, num_effectors, 3 | 4 | 9 | 3 x 3]; np.float32
    joint_positions: Optional[np.ndarray]  # shape: [H, num_effectors, 6 | 7]; np.float32

    @property
    def num_effectors(self) -> int:
        for value in (self.ee_pose_translation, self.ee_pose_rotation, self.joint_positions):
            if value is not None:
                return value.shape[1]
        raise ValueError("Arms observation carries no scalar column to take the effector count from")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ee_pose_translation": self.ee_pose_translation,
            "ee_pose_rotation": self.ee_pose_rotation,
            "joint_positions": self.joint_positions,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "ArmsInferenceObservation":
        return ArmsInferenceObservation(
            ee_pose_translation=_as_array(payload["ee_pose_translation"]),
            ee_pose_rotation=_as_array(payload["ee_pose_rotation"]),
            joint_positions=_as_array(payload["joint_positions"]),
        )


@dataclasses.dataclass
class HandsInferenceObservation:
    """Gripper(s) observation for one env. Mirrors the training ``HandsDataInput``"""

    gripper_width: Optional[np.ndarray]  # shape: [H, num_effectors, 1]; np.float32

    @property
    def num_effectors(self) -> int:
        return self.gripper_width.shape[1]

    def to_dict(self) -> Dict[str, Any]:
        return {"gripper_width": self.gripper_width}

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "HandsInferenceObservation":
        return HandsInferenceObservation(gripper_width=_as_array(payload["gripper_width"]))


@dataclasses.dataclass
class RobotInferenceObservation:
    """What one policy forward is shown for one env. Distinct from the api/core ``RobotObservation``
    a robot server publishes: this packs the last ``H`` of those into the history window the model
    was trained to read, in the layout its processor expects"""

    images: Dict[str, np.ndarray]  # Keys: cameras, Values: shape [H, height, width, channels]; np.uint8
    instruction: Optional[str]
    arms: ArmsInferenceObservation
    hands: HandsInferenceObservation
    timestamps: Optional[np.ndarray]  # shape: [H]; np.float64
    dataset_name: Optional[str]

    @property
    def num_effectors(self) -> int:
        return self.arms.num_effectors

    def to_dict(self) -> Dict[str, Any]:
        # Arrays go in by reference: dataclasses.asdict would deep-copy the whole image stack, once
        # per observation per step
        return {
            "images": self.images,
            "instruction": self.instruction,
            "arms": self.arms.to_dict(),
            "hands": self.hands.to_dict(),
            "timestamps": self.timestamps,
            "dataset_name": self.dataset_name,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "RobotInferenceObservation":
        return RobotInferenceObservation(
            images=payload["images"],
            instruction=payload["instruction"],
            arms=ArmsInferenceObservation.from_dict(payload["arms"]),
            hands=HandsInferenceObservation.from_dict(payload["hands"]),
            timestamps=_as_array(payload["timestamps"]),
            dataset_name=payload["dataset_name"],
        )


@dataclasses.dataclass
class ArmsInferenceControl:
    """Arm(s) half of an action chunk. Mirrors the training ``ArmsDataTarget``"""

    translation: np.ndarray  # [num_future_control_steps, num_effectors, 3]; np.float32
    rotation: np.ndarray  # [num_future_control_steps, num_effectors, 3, 3]; np.float32

    def to_dict(self) -> Dict[str, Any]:
        return {"translation": self.translation, "rotation": self.rotation}

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "ArmsInferenceControl":
        return ArmsInferenceControl(
            translation=np.asarray(payload["translation"]),
            rotation=np.asarray(payload["rotation"]),
        )


@dataclasses.dataclass
class HandsInferenceControl:
    """Gripper(s) half of an action chunk. The training ``HandsDataTarget`` carries a width, while a
    policy emits the probability that the gripper is closed, thresholded at the robot boundary"""

    gripper_prob: np.ndarray  # [num_future_control_steps, num_effectors, 1]; np.float32

    def to_dict(self) -> Dict[str, Any]:
        return {"gripper_prob": self.gripper_prob}

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "HandsInferenceControl":
        return HandsInferenceControl(gripper_prob=np.asarray(payload["gripper_prob"]))


@dataclasses.dataclass
class RobotInferenceControl:
    """The action chunk one forward returns per observation: a pose delta and a gripper probability
    per step and per effector, as ``[num_future_control_steps, num_effectors, ...]`` arrays.

    Timesteps stay at dimension 0 and the effector axis sits inside, so ``ActionChunkingController``
    keeps indexing a step the same way at any effector count."""

    arms: ArmsInferenceControl
    hands: HandsInferenceControl

    @property
    def num_effectors(self) -> int:
        return self.arms.translation.shape[1]

    def at_step(self, index: int) -> "RobotInferenceControl":
        """One step out of the chunk, as the same type with the time axis dropped: ``[num_effectors, ...]``"""
        return RobotInferenceControl(
            arms=ArmsInferenceControl(
                translation=self.arms.translation[index],
                rotation=self.arms.rotation[index],
            ),
            hands=HandsInferenceControl(gripper_prob=self.hands.gripper_prob[index]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"arms": self.arms.to_dict(), "hands": self.hands.to_dict()}

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "RobotInferenceControl":
        return RobotInferenceControl(
            arms=ArmsInferenceControl.from_dict(payload["arms"]),
            hands=HandsInferenceControl.from_dict(payload["hands"]),
        )


PolicyConfigT = TypeVar("PolicyConfigT", bound=Config)


class ModelInference(Configurable[PolicyConfigT], Template[PolicyConfigT]):
    """The model itself, stateless and batched: what ``BatchPolicyServer`` serves, whether that is a
    ladder checkpoint or an off-the-shelf VLA, selected by the server config's single
    ``policy_config``.

    Not api/client's ``Policy``, which is the other seam: a ``Policy`` is what an episode runner
    drives -- robot observations in, robot actions out, per-env rollout state inside -- and it
    reaches this one over the wire, owning the history window and the action chunk
    (``VLAMRobotInference`` over a ``RemoteInferenceClient``)."""

    @abc.abstractmethod
    def metadata(self) -> PolicyMetadataPayload:
        """What remote clients need to know to talk to this policy (see ``PolicyMetadataPayload``)."""

    @abc.abstractmethod
    def prepare_observation(self, observation: RobotInferenceObservation) -> RobotInferenceObservation:
        """Server-side normalization of one wire observation — drop cameras the policy does not
        read, resize the rest — so every client sends full frames and stays model-agnostic."""

    @abc.abstractmethod
    def low_level_step(self, observations: List[RobotInferenceObservation]) -> List[RobotInferenceControl]:
        """One batched forward: one action chunk per observation."""

    def high_level_step(
        self, observations: List[RobotInferenceObservation], max_new_tokens: int
    ) -> List[str]:
        """One generated subtask string per observation, for policies with a language head."""
        raise NotImplementedError(f"{type(self).__name__} has no high-level (subtask) head")
