import os
import pathlib
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import download as _download, nnx_utils as _nnx_utils, normalize as _normalize
from openpi.training import config as _train_config
from openpi_client import image_tools

from ladder.components.inference.api.core.action import (
    ArmAction,
    JointAnglesAction,
    JointVelocitiesAction,
    ManipulatorAction,
    RobotAction,
)
from ladder.components.inference.api.core.observation import ManipulatorState, RobotObservation
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import Config, Configurable

LOGGER = get_logger(__name__)

OPENPI_CHECKPOINTS = "gs://openpi-assets/checkpoints"

JOINT_POSITIONS = "joint_positions"
JOINT_VELOCITIES = "joint_velocities"


def _enable_jax_compilation_cache(cache_dir: str) -> None:
    if not cache_dir or "JAX_COMPILATION_CACHE_DIR" in os.environ:
        return
    jax.config.update("jax_compilation_cache_dir", str(pathlib.Path(cache_dir).expanduser()))


class OpenpiModelConfig(Config):
    model_id: str = "pi05_droid"
    checkpoint_dir: str = ""
    jax_compilation_cache_dir: str = "~/.cache/jax"  # "" disables the cache
    main_camera: str = "main"
    wrist_camera: str = "wrist"
    exterior_key: str = "observation/exterior_image_1_left"
    wrist_key: str = "observation/wrist_image_left"
    joint_key: str = "observation/joint_position"
    gripper_key: str = "observation/gripper_position"
    image_resolution: int = 224
    arm_joints: int = 7
    flip_gripper: bool = True

    def checkpoint(self) -> str:
        return self.checkpoint_dir or f"{OPENPI_CHECKPOINTS}/{self.model_id}"


class OpenpiModel(Configurable[OpenpiModelConfig]):
    def __init__(self, config: OpenpiModelConfig) -> None:
        super().__init__(config)
        _enable_jax_compilation_cache(config.jax_compilation_cache_dir)
        train_config = _train_config.get_config(config.model_id)
        anon = {"token": "anon"} if config.checkpoint().startswith(OPENPI_CHECKPOINTS) else {}
        checkpoint_dir = pathlib.Path(_download.maybe_download(config.checkpoint(), **anon))
        LOGGER.info(f"Loading {config.model_id} from {checkpoint_dir} ...")

        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        norm_stats = _normalize.load(checkpoint_dir / "assets" / data_config.asset_id)
        self._input_transform = _transforms.compose(
            [
                _transforms.InjectDefaultPrompt(None),
                *data_config.data_transforms.inputs,
                _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ]
        )
        self._output_transform = _transforms.compose(
            [
                *data_config.model_transforms.outputs,
                _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.data_transforms.outputs,
            ]
        )
        self._sample_actions = _nnx_utils.module_jit(
            train_config.model.load(
                _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
            ).sample_actions
        )
        self._rng = jax.random.key(0)
        LOGGER.info(f"{config.model_id} loaded")

    def _example(self, observation: RobotObservation, instruction: str) -> Dict[str, Any]:
        state = observation.state
        assert isinstance(state, ManipulatorState), f"expected a single-arm observation, got {type(state)}"
        if state.joints is None or state.gripper_state is None:
            raise ValueError("an openpi checkpoint needs joints and gripper_state on the observation")
        gripper = np.atleast_1d(np.asarray(state.gripper_state, dtype=np.float32))
        return {
            self.config.exterior_key: self._image(observation, self.config.main_camera),
            self.config.wrist_key: self._image(observation, self.config.wrist_camera),
            self.config.joint_key: np.asarray(state.joints, dtype=np.float32)[: self.config.arm_joints],
            self.config.gripper_key: 1.0 - gripper if self.config.flip_gripper else gripper,
            "prompt": instruction,
        }

    def _image(self, observation: RobotObservation, camera: str) -> np.ndarray:
        images = observation.images or {}
        if images.get(camera) is None:
            raise KeyError(f"observation is missing camera {camera!r} (has {sorted(images)})")
        resolution = self.config.image_resolution
        return image_tools.resize_with_pad(images[camera], resolution, resolution)

    def infer(self, observations: List[RobotObservation], instructions: List[str]) -> np.ndarray:
        examples = [
            self._example(observation, instruction)
            for observation, instruction in zip(observations, instructions, strict=True)
        ]
        transformed = [self._input_transform(dict(example)) for example in examples]
        batch = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *transformed)
        observation = _model.Observation.from_dict(batch)
        self._rng, key = jax.random.split(self._rng)
        actions = np.asarray(self._sample_actions(key, observation))
        states = np.asarray(batch["state"])
        chunks = []
        for b in range(len(examples)):
            result = self._output_transform({"state": states[b], "actions": actions[b]})
            chunks.append(np.asarray(result["actions"], dtype=np.float32))
        return np.stack(chunks)


class OpenpiPolicyConfig(Config):
    model_config: OpenpiModelConfig

    # Must match the checkpoint: pi05_droid emits joint velocities, pi05_droid_jointpos targets.
    action_space: str = JOINT_VELOCITIES
    open_loop_horizon: int = 8

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.action_space in (
            JOINT_POSITIONS,
            JOINT_VELOCITIES,
        ), f"unknown action_space {self.action_space!r}"


class OpenpiPolicy(Configurable[OpenpiPolicyConfig]):
    def __init__(self, config: OpenpiPolicyConfig) -> None:
        super().__init__(config)
        self._model = OpenpiModel(config.model_config)
        self._chunks: Dict[int, Deque[RobotAction]] = {}
        self._instructions: Dict[int, str] = {}

    def reset(self) -> None:
        self._chunks.clear()
        self._instructions.clear()

    def predict_action(self, observation: RobotObservation, instruction: str) -> RobotAction:
        return self.predict_actions([observation], [instruction])[0]

    def predict_actions(
        self,
        observations: List[RobotObservation],
        instructions: List[str],
        env_indices: Optional[List[int]] = None,
    ) -> List[RobotAction]:
        if env_indices is None:
            env_indices = list(range(len(observations)))
        assert len(instructions) == len(observations) == len(env_indices)

        missing = []
        for observation, instruction, env_index in zip(observations, instructions, env_indices, strict=True):
            chunk = self._chunks.get(env_index)
            if not chunk or self._instructions.get(env_index) != instruction:
                missing.append((observation, instruction, env_index))
        if missing:
            vectors = self._model.infer(
                [observation for observation, _, _ in missing],
                [instruction for _, instruction, _ in missing],
            )
            for (_, instruction, env_index), chunk_vectors in zip(missing, vectors, strict=True):
                chunk = deque(
                    self._action(vector) for vector in chunk_vectors[: self.config.open_loop_horizon]
                )
                self._chunks[env_index] = chunk
                self._instructions[env_index] = instruction
        return [self._chunks[env_index].popleft() for env_index in env_indices]

    def _action(self, vector: np.ndarray) -> RobotAction:
        arm = np.asarray(vector[: self.config.model_config.arm_joints], dtype=np.float32)
        arm_action: ArmAction
        if self.config.action_space == JOINT_VELOCITIES:
            arm_action = JointVelocitiesAction(velocities=np.clip(arm, -1.0, 1.0))
        else:
            arm_action = JointAnglesAction(angles=arm)
        gripper = float(vector[-1] > 0.5)
        return ManipulatorAction(
            arm_action=arm_action,
            gripper_action=1.0 - gripper if self.config.model_config.flip_gripper else gripper,
        )
