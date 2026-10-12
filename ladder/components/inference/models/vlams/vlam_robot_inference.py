from typing import List, Optional, Union

import numpy as np
import torch

from ladder.components.inference.api.client.batch_inference.model_inference import (
    ArmsInferenceObservation,
    HandsInferenceObservation,
    RobotInferenceControl,
    RobotInferenceObservation,
)
from ladder.components.inference.api.client.batch_inference.remote_inference_client import (
    RemoteInferenceClient,
)
from ladder.components.inference.api.core.action import (
    DualManipulatorAction,
    EndEffectorPoseAction,
    ManipulatorAction,
    RobotAction,
)
from ladder.components.inference.api.core.observation import RobotObservation, effector_states
from ladder.core.databib.config import AutoConfig, AutoConfigurable, Config, Configurable

from .history_buffer import HistoryBuffer
from .vlam_inference import VLAMInference
from .vlam_utils import euler_to_rotmat, rotmat_to_euler


class VLAMRobotInferenceConfig(Config):
    env_name: str
    # Selects the inference backend. Use VLAMInference for an in-process model or
    # RemoteInferenceClient to delegate to a remote BatchPolicyServer.
    inference_config: AutoConfig
    translation_scale: float = 1.0
    rotation_scale: float = 1.0
    past_observations_stride: float | int = 1
    controller_config: AutoConfig
    controls_on_targets: bool = True  # If True, apply controls on target pose, otherwise on observed pose

    def __post_init__(self):
        super().__post_init__()
        if self.past_observations_stride != 1:
            raise NotImplementedError("Past observations stride should be implemented in the robot server")

    @property
    def dataset_name(self) -> str:
        if 'google' in self.env_name or 'fractal' in self.env_name:
            return 'fractal20220817_data'
        if 'widowx' in self.env_name or 'bridge' in self.env_name:
            return 'bridge_orig'
        return self.env_name


class VLAMRobotInference(Configurable[VLAMRobotInferenceConfig]):
    """Drive N parallel envs through a shared inference engine with per-env state."""

    inference: Union[VLAMInference, RemoteInferenceClient]

    def __init__(self, config: VLAMRobotInferenceConfig) -> None:
        super().__init__(config)
        self.inference = AutoConfigurable(self.config.inference_config)
        self.history_buffers: List[HistoryBuffer] = []
        self.controllers: List[Configurable] = []

    def _create_history_buffer(self) -> HistoryBuffer:
        io = self.inference.processor.config.control_io_config
        if io.past_scalars_sequence_length != io.past_frames_sequence_length:
            raise NotImplementedError()
        return HistoryBuffer(
            past_observations_len=io.past_scalars_sequence_length,
            past_observations_stride=self.config.past_observations_stride,
        )

    def reset(self) -> None:
        self.history_buffers = []
        self.controllers = []

    def predict_actions(
        self,
        observations: List[RobotObservation],
        instructions: List[str],
        env_indices: Optional[List[int]] = None,
    ) -> List[RobotAction]:
        """Return one action per env. Envs whose action-chunk cache is exhausted contribute to a
        single batched ``inference.step`` call. Per-env state is kept per ``env_indices[k]`` (default
        ``0..N-1``); pass ``env_indices`` to advance a subset of envs and leave the rest frozen."""
        n = len(observations)
        assert len(instructions) == n
        if not observations:
            return []
        if env_indices is None:
            env_indices = list(range(n))
        assert len(env_indices) == n
        # Grow per-env state to cover the largest env index requested.
        while len(self.history_buffers) <= max(env_indices):
            self.controllers.append(AutoConfigurable(self.config.controller_config))
            self.history_buffers.append(self._create_history_buffer())

        processor = self.inference.processor
        eef_poses: List[Optional[np.ndarray]] = [None] * n
        control_poses: List[Optional[np.ndarray]] = [None] * n
        steps: List[Optional[RobotInferenceControl]] = [None] * n
        pending_obs: List[RobotInferenceObservation] = []
        pending_positions: List[int] = []

        for i, (obs, instr, env_index) in enumerate(
            zip(observations, instructions, env_indices, strict=True)
        ):
            # Drop images that are not used by the model as it interferes with inference
            # We replace in-place to avoid unnecessary videos for unused cameras
            for cam in list(obs.images.keys()):
                if cam not in processor.image_sizes:
                    obs.images.pop(cam)

            # Resize images using the processor's resize_image method
            for cam, img in obs.images.items():
                obs.images[cam] = processor.resize_image(cam, img)

            effectors = effector_states(obs.state)
            assert len(effectors) == self.inference.num_effectors, (
                f"Robot publishes {len(effectors)} effectors but the model serves "
                f"{self.inference.num_effectors}"
            )

            # TODO(START): Deprecate Franka frame
            # Rotate 180 degrees around X axis -> make it compatible with bridge V2 dataset
            # eef_pose is now w.r.t. robot base frame (instead of canonical franka EEF frame)
            eef_pose = np.stack([_canonical_to_bridge_frame(effector.eef_pose) for effector in effectors])

            if self.config.controls_on_targets:
                control_poses[i] = np.stack(
                    [_canonical_to_bridge_frame(effector.eef_pose_from_target) for effector in effectors]
                )
            # TODO(END): Deprecate Franka frame

            eef_poses[i] = eef_pose  # [num_effectors, 4, 4]

            self.history_buffers[env_index].append(
                {
                    'images': obs.images,
                    # NOTE: We feed in the model the actually observed EEF pose, not the target pose.
                    'ee_pose_translation': eef_pose[:, :3, -1],
                    'ee_pose_rotation': eef_pose[:, :3, :3],
                    'gripper_width': np.stack([effector.gripper_state for effector in effectors]),
                    'joint_positions': np.stack([effector.joints for effector in effectors]),
                }
            )

            steps[i] = self.controllers[env_index].step()
            if steps[i] is None:
                h = self.history_buffers[env_index].get_obs()
                pending_obs.append(
                    RobotInferenceObservation(
                        images=h['images'],
                        instruction=instr,
                        arms=ArmsInferenceObservation(
                            ee_pose_translation=h['ee_pose_translation'],
                            ee_pose_rotation=h['ee_pose_rotation'],
                            joint_positions=h['joint_positions'],
                        ),
                        hands=HandsInferenceObservation(gripper_width=h['gripper_width']),
                        timestamps=None,
                        dataset_name=self.config.dataset_name,
                    )
                )
                pending_positions.append(i)

        # One batched inference call for the envs whose chunk just ran out.
        if pending_obs:
            for i, control in zip(pending_positions, self.inference.low_level_step(pending_obs), strict=True):
                env_index = env_indices[i]
                self.controllers[env_index].update_control(control)
                steps[i] = self.controllers[env_index].step()

        actions: List[RobotAction] = []
        for i in range(n):
            cs = steps[i]
            translation = cs.arms.translation * self.config.translation_scale
            if self.config.rotation_scale != 1.0:
                # TODO: Maybe scale in rotvec?
                rotmat = euler_to_rotmat(
                    torch.as_tensor(
                        rotmat_to_euler(torch.as_tensor(cs.arms.rotation)).numpy()
                        * self.config.rotation_scale
                    )
                ).numpy()
            else:
                rotmat = cs.arms.rotation

            # NOTE: For control we usually use the target EEF frame, computed based on the previous target pose
            # rather than the actually observed pose. Otherwise, the gripper might grasp too early. However,
            # when using targets, the internal robot controller might deviate intermittently
            poses = control_poses[i] if self.config.controls_on_targets else eef_poses[i]

            # `cs` is one step out of the chunk, so its leading axis is the effector, not time. Take
            # the count off the robot's own poses [num_effectors, 4, 4] rather than off the control
            effector_actions: List[ManipulatorAction] = []
            for effector in range(len(poses)):
                gripper = float(cs.hands.gripper_prob[effector].item() > 0.5)

                arm = EndEffectorPoseAction.from_homogeneous_matrix(np.array(poses[effector])).apply_delta(
                    delta_rotation=rotmat[effector],
                    delta_translation=translation[effector],
                    delta_mode=self.delta_mode,
                )

                # TODO(START): Deprecate Franka frame
                # Convert from Bridge frame (robot base) back to canonical Franka EEF frame
                arm = EndEffectorPoseAction(
                    translation=arm.translation,
                    rotation=arm.rotation @ CANONICAL_TO_BRIDGE_ROTATION.T,
                )
                # TODO(END): Deprecate Franka frame

                effector_actions.append(ManipulatorAction(arm_action=arm, gripper_action=gripper))

            actions.append(_robot_action(effector_actions))
        return actions

    @property
    def delta_mode(self) -> str:
        """The delta_mode passed to EndEffectorPoseAction.apply_delta."""
        return self.inference.delta_mode


CANONICAL_TO_BRIDGE_ROTATION = np.array(
    [
        [1, 0, 0],
        [0, np.cos(np.pi), -np.sin(np.pi)],
        [0, np.sin(np.pi), np.cos(np.pi)],
    ],
    dtype=np.float32,
)


def _canonical_to_bridge_frame(eef_pose: np.ndarray) -> np.ndarray:
    pose = np.array(eef_pose)
    pose[:3, :3] = pose[:3, :3] @ CANONICAL_TO_BRIDGE_ROTATION
    return pose


def _robot_action(effector_actions: List[ManipulatorAction]) -> RobotAction:
    """The core types name the arms rather than indexing them,
    so the effector axis is unpacked back into right/left here, at the robot boundary."""
    if len(effector_actions) == 1:
        return effector_actions[0]
    if len(effector_actions) == 2:
        right, left = effector_actions
        return DualManipulatorAction(right=right, left=left)
    raise NotImplementedError(f"No robot action for {len(effector_actions)} effectors")
