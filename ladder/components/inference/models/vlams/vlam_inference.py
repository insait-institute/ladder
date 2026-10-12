import dataclasses
import importlib
import os
import pathlib
import sys
from typing import Dict, List, Protocol

import numpy as np
import torch
import yaml

from ladder.components.inference.api.client.batch_inference.batch_protocol import PolicyMetadataPayload
from ladder.components.inference.api.client.batch_inference.model_inference import (
    ArmsInferenceControl,
    HandsInferenceControl,
    ModelInference,
    RobotInferenceControl,
    RobotInferenceObservation,
)
from ladder.core.compute.collate import collate_and_pad_elements, concat_and_pad_elements
from ladder.core.databib.config import Config
from ladder.core.databib.dataclasses import Dataclass

from .vlam_utils import rotmat_as_3x3
from .vlam_versions import (
    env_version_from_session,
    has_dataclass_processor_api,
    has_effector_axis_processor_api,
    has_unbatched_processor_api,
)


class _ControlFrame(Protocol):
    def to_core(self) -> object:
        ...


class _ArmsIOConfig(Protocol):
    translation_control_frame: _ControlFrame
    rotation_control_frame: _ControlFrame


class _VLAMProcessorConfig(Protocol):
    # Env versions before EFFECTOR_AXIS_ENV_VERSION - arms_io_config not present
    # Env versions after EFFECTOR_AXIS_ENV_VERSION - ONLY arms_io_config is present
    translation_control_frame: _ControlFrame
    rotation_control_frame: _ControlFrame
    arms_io_config: _ArmsIOConfig


class _VLAMModelConfig(Protocol):
    processor_config: _VLAMProcessorConfig


class VLAMInferenceConfig(Config):
    ckpt_path: str
    dtype: str = "bfloat16"
    device: str = "cuda"

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)


class VLAMInference(ModelInference[VLAMInferenceConfig]):
    """
    General module that performs VLAM inference. In general, you would likely need a wrapper around
    it to adapt it to different APIs for closed loop control. Implements the batch_inference
    ``ModelInference`` interface, so a ladder checkpoint is served exactly like any other policy.
    """

    def __init__(self, config: VLAMInferenceConfig) -> None:
        super().__init__(config)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.dtype = config.torch_dtype
        self.device = config.device

        # Environment version info
        self.env_version: str = env_version_from_session(pathlib.Path(config.ckpt_path).parent.parent)
        self.unbatched_processor_api: bool = has_unbatched_processor_api(self.env_version)
        self.effector_axis_processor_api: bool = has_effector_axis_processor_api(self.env_version)
        self.dataclass_processor_api: bool = has_dataclass_processor_api(self.env_version)

        self.model = self._create_model(ckpt_path=config.ckpt_path)
        self.processor = self.model.processor
        self.delta_mode: str = compute_delta_mode(
            self.model.config, nested_arm_frames=self.effector_axis_processor_api
        )

    @property
    def num_effectors(self) -> int:
        return int(getattr(self.processor.config, "num_effectors", 1))

    def metadata(self) -> PolicyMetadataPayload:
        control_io_config = self.processor.config.control_io_config
        return {
            "image_sizes": {
                name: [int(size.height), int(size.width)] for name, size in self.processor.image_sizes.items()
            },
            "past_scalars_sequence_length": int(control_io_config.past_scalars_sequence_length),
            "past_frames_sequence_length": int(control_io_config.past_frames_sequence_length),
            "delta_mode": self.delta_mode,
            "num_effectors": self.num_effectors,
        }

    def prepare_observation(self, observation: RobotInferenceObservation) -> RobotInferenceObservation:
        """Drop cameras the model doesn't expect and resize each remaining camera's frames
        through the real processor."""
        resized_images: Dict[str, np.ndarray] = {}
        for camera_name, frames in observation.images.items():
            if camera_name not in self.processor.image_sizes:
                continue
            resized_images[camera_name] = np.stack(
                [self.processor.resize_image(camera_name, frame) for frame in frames],
                axis=0,
            )
        return dataclasses.replace(observation, images=resized_images)

    def _create_model(self, ckpt_path: str) -> torch.nn.Module:
        model = make_model(ckpt_path)
        model = model.to(dtype=self.dtype, device=self.device)
        model = model.eval()
        return model

    def _create_model_input(self, robot_obs: RobotInferenceObservation) -> Dataclass:
        """Preprocess one observation into a model input"""
        chat = [robot_obs.instruction, ""]

        # Convert from numpy [history, height, width, channels] to List of [height, width, channels]
        images = {camera_name: list(camera_images) for camera_name, camera_images in robot_obs.images.items()}

        if self.dataclass_processor_api:
            return self.processor.create_input(self._create_data_input(robot_obs, chat=chat, images=images))

        scalars = {
            'ee_pose_translation': robot_obs.arms.ee_pose_translation,
            'ee_pose_rotation': robot_obs.arms.ee_pose_rotation,
            'gripper': robot_obs.hands.gripper_width,
            'joints': robot_obs.arms.joint_positions,
        }
        legacy_kwargs = {}

        # DATACLASS_IO_ENV_VERSION bundles returned above, so only v0-v4 reach here and this branch is
        # exactly "is v4". TODO: if the v4 env is removed, drop the branch and keep the else body
        if self.effector_axis_processor_api:
            # Processor API from EFFECTOR_AXIS_ENV_VERSION onwards packs the effectors of an embodiment in
            # a dedicated axis, which the observation already carries. The same env renamed the keyword,
            # so the two move together
            scalars['joint_positions'] = scalars.pop('joints')
        else:
            # Older processor APIs are single-arm: [num_past_scalars, D], with no effector axis to
            # unpack a second arm into
            assert robot_obs.num_effectors == 1, (
                f"A bi-manual observation ({robot_obs.num_effectors} effectors) cannot be expressed for "
                f"the {self.env_version} processor API"
            )
            scalars = {name: value[:, 0] for name, value in scalars.items()}

        if self.unbatched_processor_api:
            # Processor API before BATCHED_IO_ENV_VERSION expects scalar np.ndarray unbatched.
            # Additionally, inference_mode arg is still required
            legacy_kwargs['inference_mode'] = True
        else:
            # Processor API after BATCHED_IO_ENV_VERSION expects scalar np.ndarray with batch dimension.
            # inference_mode arg is removed
            scalars = {name: value[np.newaxis] for name, value in scalars.items()}

        return self.processor.create_input(
            chat=chat,
            images=images,
            **scalars,
            dataset_name=np.array([robot_obs.dataset_name if robot_obs.dataset_name is not None else 'ANY']),
            **legacy_kwargs,
        )

    def _create_data_input(
        self, robot_obs: RobotInferenceObservation, chat: List[str], images: Dict[str, List[np.ndarray]]
    ) -> Dataclass:
        """Build the `RoboticsDataInput` a DATACLASS_IO_ENV_VERSION model bundle expects"""

        # The bundle defines its own ReferenceFrame, and the dataclass type-checks exactly, so take the
        # enum off a value the bundle's own config carries rather than importing it
        ReferenceFrame = type(self.processor.arms_io_config.translation_obs_frame)

        # [1, num_past_scalars, num_effectors, D] -- the observation carries the effector axis
        scalars = {
            f'{group}.{name}': value[np.newaxis]
            for group, fields in (('arms', robot_obs.arms.to_dict()), ('hands', robot_obs.hands.to_dict()))
            for name, value in fields.items()
            if value is not None
        }

        return self.processor.DataInputT.make_empty().replace(
            images=images,
            language=np.asarray([chat]),
            dataset_name=np.array([robot_obs.dataset_name if robot_obs.dataset_name is not None else 'ANY']),
            **scalars,
            **{
                # Observations are assumed to be in the ROBOT_BASE frame.
                # The processor converts to the frame needed by the model
                'arms.translation_frame': ReferenceFrame.ROBOT_BASE,
                'arms.rotation_frame': ReferenceFrame.ROBOT_BASE,
            },
        )

    def low_level_step(self, obs_list: List[RobotInferenceObservation]) -> List[RobotInferenceControl]:
        """One batched forward pass over N observations → N RobotControls."""
        n = len(obs_list)
        model_inputs = [self._create_model_input(obs) for obs in obs_list]
        dataset_names = [obs.dataset_name for obs in obs_list]

        batched_input = self._collate_and_pad(model_inputs).to(device=self.device)
        with torch.no_grad(), torch.autocast(self.device, dtype=self.dtype):
            batched_output = self.model.generate(batched_input)

        control_plan = self.processor.policy_control_plan_from_model_output(
            model_output=batched_output,
            dataset_name=np.array([dn if dn is not None else 'ANY' for dn in dataset_names]),
            valid_mask=None,
        )

        translation_m = control_plan.translation_m.to(dtype=torch.float32, device='cpu').numpy()
        rotmat = rotmat_as_3x3(control_plan.rotmat.to(dtype=torch.float32, device='cpu')).numpy()
        gripper_prob = control_plan.gripper_prob.to(dtype=torch.float32, device='cpu').numpy()

        if not self.effector_axis_processor_api:
            # Control plans from before EFFECTOR_AXIS_ENV_VERSION are single-arm and flat. Add the axis
            # here, so a chunk has the same layout whatever bundle produced it
            translation_m = translation_m[:, :, np.newaxis]  # [B, num_timesteps, 1, 3]
            rotmat = rotmat[:, :, np.newaxis]  # [B, num_timesteps, 1, 3, 3]
            gripper_prob = gripper_prob[:, :, np.newaxis]  # [B, num_timesteps, 1, 1]

        return [
            RobotInferenceControl(
                arms=ArmsInferenceControl(translation=translation_m[i], rotation=rotmat[i]),
                hands=HandsInferenceControl(gripper_prob=gripper_prob[i]),
            )
            for i in range(n)
        ]

    def high_level_step(
        self,
        obs_list: List[RobotInferenceObservation],
        max_new_tokens: int = 64,
    ) -> List[str]:
        """One batched forward → a subtask per observation from the VLM backbone (needs lm_head=True)."""
        if not obs_list:
            return []
        template = self.processor.config.subtask_prompt_template
        prompts = [
            template.format(task=obs.instruction if obs.instruction is not None else "") for obs in obs_list
        ]
        model_inputs = [
            self._create_model_input(dataclasses.replace(obs, instruction=prompt))
            for obs, prompt in zip(obs_list, prompts, strict=True)
        ]
        batched_input = self._collate_and_pad(model_inputs).to(device=self.device)
        # Left-padding makes every prompt end at the same column, so the generated suffix
        # starts at `prefix_len` uniformly across the batch.
        prefix_len = int(batched_input.input_ids.shape[-1])
        with torch.no_grad(), torch.autocast(self.device, dtype=self.dtype):
            output_ids = self.model.generate_text(batched_input, max_new_tokens=max_new_tokens)
        output_ids = output_ids.to(device='cpu')
        return [
            self.processor.tokenizer.decode(
                output_ids[i, prefix_len:].tolist(), skip_special_tokens=True
            ).strip()
            for i in range(len(obs_list))
        ]

    def _collate_and_pad(self, model_inputs: List[Dataclass]) -> Dataclass:
        """Batch ``model_inputs``. Use left-padding for ``input_ids`` and ``attn_mask``"""
        pad_token_id = self.processor.tokenizer.pad_token_id or 0
        # Older versions expect scalars unbatched, newer ones expect them batched
        batch_fn = collate_and_pad_elements if self.unbatched_processor_api else concat_and_pad_elements
        return batch_fn(
            model_inputs,
            pads={
                "input_ids": {"dim": -1, "pad_value": pad_token_id, "side": "left"},
                "attn_mask": {"dim": -1, "pad_value": 0, "side": "left"},
            },
        )


def import_py_module(file_path: str | pathlib.Path, module_name: str):
    """
    Args:
        path: Full path to the python file, including the '.py' extension
        module_name: The name that will be given to the base python module

    Ex:
        path = /path/to/models/my_session/hf_export/src/modeling_paligemma_dit_pizero.py
        module_name: my_session
        root_path -> /path/to/models
        py_module_path -> my_session.hf_export.src.modeling_paligemma_dit_pizero

    The equivalent would be
        sys.path.append('/path/to/models')
        from my_session.hf_export.src.modeling_paligemma_dit_pizero import MainModel
    """
    file_path = str(file_path)

    if '.' in module_name:
        raise ValueError(f"module_name '{module_name}' cannot contain dots")

    if f'/{module_name}/' not in str(file_path):
        raise ValueError(f"module_name '{module_name}' has to be part of the path to the file {file_path}")

    root_path, _, submodule_path = file_path.rpartition(f'/{module_name}/')
    py_module_path = f'{module_name}.{os.path.splitext(submodule_path)[0].replace(os.sep, ".").strip(".")}'

    # Add the directory containing `module_name` to sys.path so that relative imports work
    if root_path not in sys.path:
        sys.path.append(str(root_path))

    # Create the spec from the file path
    spec = importlib.util.spec_from_file_location(py_module_path, file_path)

    # Create the module from the spec
    py_module = importlib.util.module_from_spec(spec)

    # Return the already-loaded module if it exists
    if py_module_path in sys.modules:
        return sys.modules[py_module_path]

    # Add the module to sys.modules
    sys.modules[py_module_path] = py_module

    # Execute the module
    spec.loader.exec_module(py_module)

    return py_module


# Keys are ``ReferenceFrame.{ROBOT_BASE,EEF}.value``. Comparing against the StrEnum
# string lets ``compute_delta_mode`` work without importing the dynamically-loaded HF
# model config's enum class.
_TRANSLATION_LABELS = {"robot_base": "trans_base", "eef_relative": "trans_eef"}
_ROTATION_LABELS = {"robot_base": "rot_eef_base_aligned", "eef_relative": "rot_eef"}


def compute_delta_mode(model_config: _VLAMModelConfig, nested_arm_frames: bool) -> str:
    """
    Derive the `delta_mode` string passed to ``EndEffectorPoseAction.apply_delta`` from a VLAM model
    config.

    Args:
        model_config: the dynamically loaded config of an exported bundle
        nested_arm_frames: True for bundles from EFFECTOR_AXIS_ENV_VERSION onwards, which carry the arm
            control frames on `processor_config.arms_io_config`. Older bundles carry them flat on
            `processor_config`
    """
    cfg = model_config.processor_config
    arm_frames = cfg.arms_io_config if nested_arm_frames else cfg
    rotation_label = _ROTATION_LABELS[arm_frames.rotation_control_frame.to_core()]
    translation_label = _TRANSLATION_LABELS[arm_frames.translation_control_frame.to_core()]
    return f"{rotation_label}_{translation_label}"


def make_model(ckpt_path: str | pathlib.Path) -> torch.nn.Module:
    """
    Create a model from a session path.

    Requires the following directory structure:
        {session_path}
        ├── checkpoints
        │   └── {ckpt_filename}
        └── hf_export
            └── {session_id}
                └── src
                    ├── configuration_{model_name}.py
                    ├── model_config.yaml
                    ├── modeling_{model_name}.py
                    └── processing_{model_name}.py

    Args:
        ckpt_path: Absolute path to a checkpoint file.
    Returns:
        Instance of the model on CPU with checkpoint loaded
    """

    session_path = pathlib.Path(ckpt_path).parent.parent

    if len(session_ids := list((session_path / 'hf_export').glob('*'))) != 1:
        raise ValueError(
            f"There should be exactly one session_id present on disk, but found {len(session_ids)}. "
            f"Check that the checkpoint path is correct: {ckpt_path}"
        )

    session_id: str = str(session_ids[0].name)
    src_path: pathlib.Path = session_path / 'hf_export' / session_id / 'src'

    # Get the paths we need to load from
    modeling_path = str(list(src_path.glob('modeling_*.py'))[0])
    configuration_path = str(list(src_path.glob('configuration_*.py'))[0])
    model_config_path = str(src_path / 'model_config.yaml')
    model_ckpt_path = str(ckpt_path)

    # Load model_config.yaml
    with open(model_config_path) as file:
        config_dict = yaml.safe_load(file)

    # Load the files configuration_{model_name}.py and modeling_{model_name}.py as python modules
    py_configuration = import_py_module(configuration_path, session_id)
    py_modeling = import_py_module(modeling_path, session_id)

    # Instantiate the config and the model
    config = py_configuration.MainModelConfig(**config_dict)
    model = py_modeling.MainModel(config)

    assert os.path.exists(model_ckpt_path), f"{model_ckpt_path} does not exist"

    # Load the model state dict
    state_dict = torch.load(model_ckpt_path, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(state_dict, strict=True)

    return model
