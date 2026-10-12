from typing import List, Optional, Union

import numpy as np
import numpy.typing

from ladder.components.inference.api.client.batch_inference.model_inference import (
    ArmsInferenceObservation,
    HandsInferenceObservation,
    RobotInferenceObservation,
)
from ladder.components.inference.api.client.batch_inference.remote_inference_client import (
    RemoteInferenceClient,
)
from ladder.components.inference.api.client.hlc.base import Decomposition, SubtaskSource
from ladder.components.inference.api.client.policy import Policy
from ladder.components.inference.api.core.observation import RobotObservation, effector_states
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import AutoConfig, AutoConfigurable, Config

from .vlam_inference import VLAMInference
from .vlam_robot_inference import VLAMRobotInference

LOGGER = get_logger(__name__)


class VLAMSubtaskSourceConfig(Config):
    """Config for the unified-checkpoint subtask source, supporting BOTH HLC modes.

    The HL backend is selected by ``inference_config``:

    * **self-HLC** (``inference_config = None``, the default): the runner calls
      :meth:`VLAMSubtaskSource.bind_policy` so the HL reuses the LL policy's own
      inference client: one on-GPU model copy is both planner and action expert, and
      HL/LL forwards coalesce on the shared :class:`BatchPolicyServer`.
    * **external HLC** (``inference_config`` set): a SEPARATE backend, e.g. a
      :class:`RemoteInferenceClient` pointed at its own batch server, or an
      in-process :class:`VLAMInference` for K=1 debugging. ``bind_policy`` is a no-op once
      an external backend is configured.

    The per-env task (from the controller's ``reset``) is substituted into the processor's
    ``subtask_prompt_template`` at every generation, so the HL conditions on (image + task).
    """

    # None => self-HLC: the runner binds the LL policy's client via bind_policy().
    # Set => external HLC backend (its own remote/in-process inference).
    inference_config: Optional[AutoConfig] = None
    dataset_name: str = "bridge"
    image_camera: str = "main"
    max_new_tokens: int = 64


class VLAMSubtaskSource(SubtaskSource[VLAMSubtaskSourceConfig]):
    """On-the-fly subtasks from a trained VLAM checkpoint's own VLM backbone (its trained
    subtask prompt, not generic chat, hence a dedicated source rather than a :class:`VLM`
    backend). Proprio is filled with zeros (only image + per-env task drives generation); the
    envs whose index moved collapse into one ``high_level_step`` batch on the shared server."""

    def __init__(self, config: VLAMSubtaskSourceConfig):
        super().__init__(config)
        # External HLC: build the configured backend eagerly. self-HLC: leave None and let
        # the runner supply the LL policy's client via bind_policy().
        self._inference: Optional[Union[VLAMInference, RemoteInferenceClient]] = (
            AutoConfigurable(self.config.inference_config)
            if self.config.inference_config is not None
            else None
        )

    def bind_policy(self, policy: Policy) -> None:
        """self-HLC fallback: when no external ``inference_config`` was given, reuse the LL
        policy's client (one on-GPU model copy is both planner and action expert). No-op once
        an external backend is already configured."""
        if self._inference is not None:
            return
        assert isinstance(policy, VLAMRobotInference), (
            "VLAMSubtaskSource must be paired with a VLAMRobotInference policy to reuse "
            f"its client; got {type(policy).__name__}"
        )
        self._inference = policy.inference
        LOGGER.info(f"VLAMSubtaskSource bound to shared {type(policy.inference).__name__}")

    def commands(
        self,
        observations: List[RobotObservation],
        tasks: List[str],
        env_indices: List[int],
        indices: List[int],
        step: int,
        decomposition: Optional[Decomposition],
    ) -> List[str]:
        del indices, decomposition
        assert self._inference is not None, "VLAMSubtaskSource needs inference_config OR bind_policy()"
        vlam_obs_list = [
            self._build_vlam_observation(observation=obs, task=task)
            for obs, task in zip(observations, tasks, strict=True)
        ]
        commands = self._inference.high_level_step(
            vlam_obs_list,
            max_new_tokens=self.config.max_new_tokens,
        )
        for env_idx, task, command in zip(env_indices, tasks, commands, strict=True):
            LOGGER.info(f"VLA HL command at step {step} env_idx={env_idx} (task={task!r}): {command!r}")
        return commands

    def _build_vlam_observation(self, observation: RobotObservation, task: str) -> RobotInferenceObservation:
        camera = self.config.image_camera
        if observation.images is None or camera not in observation.images:
            raise ValueError(f"Camera '{camera}' not found in observation images")

        assert self._inference is not None
        # Pass-through for the remote client (server resizes before the forward);
        # in-process VLAMInference resizes locally. Both share the processor surface.
        resized: numpy.typing.NDArray[np.uint8] = self._inference.processor.resize_image(
            camera, observation.images[camera]
        )
        images = {camera: np.stack([resized], axis=0)}

        past_steps = self._inference.processor.config.control_io_config.past_scalars_sequence_length
        num_effectors = len(effector_states(observation.state))

        return RobotInferenceObservation(
            images=images,
            instruction=task,
            arms=ArmsInferenceObservation(
                ee_pose_translation=np.zeros((past_steps, num_effectors, 3), dtype=np.float32),
                ee_pose_rotation=np.zeros((past_steps, num_effectors, 3), dtype=np.float32),
                joint_positions=np.zeros((past_steps, num_effectors, 7), dtype=np.float32),
            ),
            hands=HandsInferenceObservation(
                gripper_width=np.zeros((past_steps, num_effectors, 1), dtype=np.float32)
            ),
            timestamps=None,
            dataset_name=self.config.dataset_name,
        )
