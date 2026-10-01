import os
from typing import Optional

import numpy as np
import numpy.typing as npt
import rerun as rr

from ladder.components.inference.api.core.action import (
    EndEffectorPoseAction,
    ManipulatorAction,
    RobotAction,
    named_effector_actions,
)
from ladder.components.inference.api.core.observation import (
    ManipulatorState,
    RobotObservation,
    named_effector_states,
)
from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)


class RerunWriter:
    """Rerun recording of one episode. Every proprioception and control entity is per arm
    (``observation/right/eef_pose``, ``gripper/left/action``), so a bi-manual episode plots two of
    everything and a single-arm one keeps its single ``right`` set."""

    def __init__(self, recording_id: str, fps: int):
        self.fps = fps
        self.frames: int = 0
        # make_default=False keeps each writer's stream private, so K batched envs don't cross-log.
        self._stream = rr.RecordingStream("rerun_replay", recording_id=recording_id, make_default=False)
        # Which arms already have their static series declared: the robot's arms are only known once
        # the first observation arrives.
        self._series_arms: set[str] = set()

    def _init_series(self, arm: str) -> None:
        if arm in self._series_arms:
            return
        self._series_arms.add(arm)
        self._stream.log(
            f"gripper/{arm}/observation",
            rr.SeriesPoints(colors=[255, 0, 0], markers="circle", marker_sizes=3),
            static=True,
        )
        self._stream.log(
            f"gripper/{arm}/action",
            rr.SeriesPoints(colors=[0, 255, 0], markers="cross", marker_sizes=3),
            static=True,
        )

    def capture_frame(self, observation: RobotObservation, action: RobotAction):
        self.frames += 1
        timestamp = observation.timestamps[0]

        axis_length: float = 0.3

        self._stream.set_time("timestamp", timestamp=timestamp)
        self._stream.set_time("frame_index", sequence=self.frames)

        # Observation camera images
        for camera_name, image in observation.images.items():
            self._stream.log(f"camera/{camera_name}", rr.Image(image))

        # Robot base pose
        self._stream.log(
            "robot/base", rr.Transform3D(translation=np.zeros(3), mat3x3=np.eye(3), axis_length=axis_length)
        )

        # The arms come off the robot's own state; an action that commands fewer of them logs
        # observation only, rather than mispairing the arms it does command.
        commanded = dict(named_effector_actions(action))
        for arm, state in named_effector_states(observation.state):
            self._log_arm(arm, state, commanded.get(arm), axis_length)

    def _log_arm(
        self,
        arm: str,
        state: ManipulatorState,
        action: Optional[ManipulatorAction],
        axis_length: float,
    ) -> None:
        self._init_series(arm)

        translation_obs = None
        if state.eef_pose is not None:
            translation_obs, rotation_obs = pose_to_components(state.eef_pose)
            self._log_pose(f"observation/{arm}/eef_pose", translation_obs, rotation_obs, axis_length)

        if action is not None and isinstance(action.arm_action, EndEffectorPoseAction):
            # EEF control
            self._log_pose(
                f"action/{arm}/eef_pose",
                action.arm_action.translation,
                action.arm_action.rotation,
                axis_length,
            )

            if translation_obs is not None:
                # tracking error wrt to action, per arm: one arm tracking badly is invisible in a
                # figure that averages the two
                translation_tracking_error = np.linalg.norm(translation_obs - action.arm_action.translation)
                self._stream.log(f"tracking/{arm}/translation_error", rr.Scalars(translation_tracking_error))

        # EEF pose gripper
        if state.gripper_state is not None:
            self._stream.log(
                f"gripper/{arm}/observation", rr.Scalars(float(np.ravel(state.gripper_state)[0]))
            )

        # EEF control gripper
        if action is not None and action.gripper_action is not None:
            self._stream.log(f"gripper/{arm}/action", rr.Scalars(action.gripper_action))

    def _log_pose(self, name: str, translation: np.ndarray, rotation: np.ndarray, axis_length: float):
        # Full pose
        self._stream.log(
            f"{name}/transform",
            rr.Transform3D(translation=translation, mat3x3=rotation, axis_length=axis_length),
        )

        # Translation component only
        self._stream.log(
            f"{name}/translation",
            rr.Transform3D(translation=translation, mat3x3=np.eye(3), axis_length=axis_length),
        )

        # Rotation component only
        self._stream.log(
            f"{name}/rotation",
            rr.Transform3D(translation=np.zeros(3), mat3x3=rotation, axis_length=axis_length),
        )

    def close(self, output_path: str):
        if self.frames == 0:
            return

        # Save the current recording to disk
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._stream.save(output_path)

        LOGGER.info(f"Rerun data saved to {output_path}")


def pose_to_components(
    pose: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    assert pose.shape == (4, 4), pose.shape
    translation = pose[:3, 3]
    rotation = pose[:3, :3]
    return translation, rotation
