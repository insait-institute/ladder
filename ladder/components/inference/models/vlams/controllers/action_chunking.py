import dataclasses

import roma
import torch

from ladder.components.inference.api.client.batch_inference.model_inference import (
    ArmsInferenceControl,
    RobotInferenceControl,
)
from ladder.core.databib.config import Config, Configurable


class ActionChunkingControllerConfig(Config):
    chunk_size: int
    start_index: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.start_index >= self.chunk_size:
            raise ValueError(f"start_index {self.start_index} must be less than chunk_size {self.chunk_size}")


class ActionChunkingController(Configurable[ActionChunkingControllerConfig]):
    def __init__(self, config: ActionChunkingControllerConfig):
        super().__init__(config)
        self._chunk_index = 0
        self._chunk: RobotInferenceControl = None

    def step(self) -> RobotInferenceControl | None:
        if self._chunk_index >= (self.config.chunk_size - self.config.start_index) or self._chunk is None:
            return None

        output = self._chunk.at_step(self._chunk_index)
        self._chunk_index += 1

        return output

    def update_control(self, robot_control: RobotInferenceControl) -> None:
        # Convert controls from relative w.r.t. current pose to deltas w.r.t. to previous control
        # NOTE: We skip all controls before the start_index
        # The chunk arrives as numpy (it crossed the wire); the delta math below is roma's, so
        # borrow it as tensors and hand it back as arrays -- both directions are views, not copies.
        # The chunk is [num_timesteps, num_effectors, ...] but the delta helpers walk the sequence on
        # dim -2, so swap the effector axis to the front and back again.
        translation = torch.as_tensor(robot_control.arms.translation[self.config.start_index :, ...])
        rotmat = torch.as_tensor(robot_control.arms.rotation[self.config.start_index :, ...])
        delta_control = dataclasses.replace(
            robot_control,
            arms=ArmsInferenceControl(
                translation=relative_to_delta_translations(translation.transpose(0, 1))
                .transpose(0, 1)
                .numpy(),
                rotation=relative_to_delta_rotations(rotmat.transpose(0, 1)).transpose(0, 1).numpy(),
            ),
        )
        self._chunk = delta_control
        self._chunk_index = 0

    def reset(self) -> None:
        self._chunk_index = 0
        self._chunk = None


def relative_to_delta_translations(translation_sequence: torch.Tensor) -> torch.Tensor:
    """
    Transform a sequence of translation vectors encoded w.r.t. the same reference frame to delta translation
    vectors where each value is encoded w.r.t. the PREVIOUS frame in the sequence. The first element in
    the sequence remains the same.
    Ex:
        Sequence of points: T1, T2, T3, T4
        `translation_sequence` contains the vectors: RT1, RT2, RT3, RT4, where R is the reference frame
        Output: RT1, T1T2, T2T3, T3T4

    Args:
        translation_sequence: torch.Tensor of shape [..., S, 3], containing the translation vectors, where S
            corresponds to the sequence dimension
    Returns:
        torch.Tensor of the same shape as translation_sequence, containing delta translations
    """
    assert translation_sequence.ndim >= 3, translation_sequence.shape

    reference_frames = torch.roll(translation_sequence, 1, dims=-2).clone()
    reference_frames[..., 0, :] = 0
    delta_translations = translation_sequence - reference_frames

    return delta_translations


def relative_to_delta_rotations(rotation_sequence: torch.Tensor) -> torch.Tensor:
    """
    Transform a sequence of rotation representations encoded w.r.t. the same reference frame to delta
    rotations where each element is encoded w.r.t. the PREVIOUS rotation frame in the sequence.
    The first element in the sequence remains the same.

    Ex:
        Sequence of points (rotations): R_1, R_2, R_3, R_4
        `rotation_sequence` contains the rotations: R_W1, R_W2, R_W3, R_W4, where W is the reference frame
            and R_W1 is the pose of R1 frame in the reference frame W, i.e. R_1W converts from reference
            frame to R1 frame
        Output: R_W1, R_12, R_23, R_34, i.e. the rotation poses of R_1 in W frame, of R_2 in R1 frame, etc

            Args:
        rotation_sequence: torch.Tensor of shape [..., S, 3, 3], containing either rotation matrices
            (R_01, R_12, R_23, R_34, ...) or quaternions, where S corresponds to the sequence dimension

    Returns:
        torch.Tensor of the same shape as rotation_sequence, containing delta rotations
    """
    assert rotation_sequence.ndim >= 3, rotation_sequence.shape

    # rotation_format: RotationFormat = rotation_format_from_tensor(rotation_sequence)

    rotation_sequence = roma.rotmat_to_unitquat(rotation_sequence)  # [..., S, 4]

    # Compute a sequence I, R_1W, R_2W, R_3W, ...
    reference_sequence = torch.roll(rotation_sequence, 1, dims=-2).clone()  # [..., S, 4]
    reference_sequence[..., 0, :] = roma.identity_quat()  # [..., S, 4]
    reference_sequence = roma.quat_inverse(reference_sequence)  # [..., S, 4]

    delta_rotations = roma.quat_product(reference_sequence, rotation_sequence)  # [..., S, 4]

    delta_rotations = roma.unitquat_to_rotmat(delta_rotations)  # [..., S, 3, 3]

    return delta_rotations
