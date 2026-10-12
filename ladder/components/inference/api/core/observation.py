import dataclasses
import functools
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt


class RobotStateType(str, Enum):
    NONE = "none"
    SINGLE_MANIPULATOR = "single_manipulator"
    DUAL_MANIPULATOR = "dual_manipulator"


@dataclasses.dataclass(kw_only=True, frozen=True)
class RobotState:
    type: RobotStateType = RobotStateType.NONE

    def __post_init__(self):
        assert self.type != RobotStateType.NONE

    @staticmethod
    def from_dict(state_dict: Dict[str, Any]) -> "RobotState":
        try:
            state_type = RobotStateType(state_dict["type"])
        except KeyError:
            raise ValueError("Robot state type not specified.")
        except ValueError:
            raise ValueError(f"Invalid robot state type. {state_dict['type']}")
        state_cls = {
            RobotStateType.SINGLE_MANIPULATOR: ManipulatorState,
            RobotStateType.DUAL_MANIPULATOR: DualManipulatorState,
        }.get(state_type)
        if state_cls is None:
            raise ValueError(f"No state class found for robot state type {state_type}")
        return state_cls.from_dict(state_dict)


@dataclasses.dataclass(kw_only=True, frozen=True)
class ManipulatorState(RobotState):
    """State of a single arm + its gripper (the former flat RobotObservation fields)."""

    type: RobotStateType = RobotStateType.SINGLE_MANIPULATOR
    eef_pose: Optional[npt.NDArray[np.float32]] = None  # [4, 4]
    eef_pose_from_target: Optional[npt.NDArray[np.float32]] = None  # [4, 4]
    gripper_state: Optional[npt.NDArray[np.float32]] = None  # [1]
    joints: Optional[npt.NDArray[np.float32]] = None  # [num_dofs]

    @staticmethod
    def from_dict(state_dict: Dict[str, Any]) -> "ManipulatorState":
        return ManipulatorState(
            eef_pose=state_dict.get("eef_pose"),
            eef_pose_from_target=state_dict.get("eef_pose_from_target"),
            gripper_state=state_dict.get("gripper_state"),
            joints=state_dict.get("joints"),
        )


@dataclasses.dataclass(kw_only=True, frozen=True)
class DualManipulatorState(RobotState):
    type: RobotStateType = RobotStateType.DUAL_MANIPULATOR
    right: ManipulatorState
    left: ManipulatorState

    @staticmethod
    def from_dict(state_dict: Dict[str, Any]) -> "DualManipulatorState":
        return DualManipulatorState(
            right=ManipulatorState.from_dict(state_dict["right"]),
            left=ManipulatorState.from_dict(state_dict["left"]),
        )


EFFECTOR_NAMES: Tuple[str, ...] = ("right", "left")


def effector_states(state: RobotState) -> List[ManipulatorState]:
    """One entry per effector. A single manipulator is one effector"""
    if isinstance(state, DualManipulatorState):
        return [state.right, state.left]
    assert isinstance(
        state, ManipulatorState
    ), f"{type(state).__name__} carries no manipulator to drive a VLAM with"
    return [state]


def named_effector_states(state: RobotState) -> List[Tuple[str, ManipulatorState]]:
    """:func:`effector_states` paired with :data:`EFFECTOR_NAMES`, so a single arm is ``right``."""
    return list(zip(EFFECTOR_NAMES, effector_states(state), strict=False))


@dataclasses.dataclass(kw_only=True, frozen=True)
class RobotObservation:
    images: Optional[Dict[str, npt.NDArray[np.uint8]]]  # images are [h, w, c]
    state: RobotState
    timestamps: Optional[npt.NDArray[np.float64]]  # [1]
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @staticmethod
    def from_dict(obs_dict: Dict[str, Any]) -> "RobotObservation":
        return RobotObservation(
            images=obs_dict.get("images"),
            state=RobotState.from_dict(obs_dict["state"]),
            timestamps=obs_dict.get("timestamps"),
            metadata=obs_dict.get("metadata", {}),
        )

    @functools.cached_property
    def cameras(self) -> Dict[str, "CameraFrame"]:
        """The depth cameras, decoded from the plain dicts that crossed the msgpack wire."""
        payload = self.metadata.get(METADATA_CAMERAS_KEY) or {}
        return {
            str(name): CameraFrame(
                depth_m=np.asarray(raw["depth_m"], dtype=np.float32),
                intrinsics=np.asarray(raw["intrinsics"], dtype=np.float64).reshape(3, 3),
                base_from_camera=np.asarray(raw["base_from_camera"], dtype=np.float64).reshape(4, 4),
            )
            for name, raw in payload.items()
        }


# Camera frames in ``RobotObservation.metadata``, keyed by camera name.
METADATA_CAMERAS_KEY = "cameras"


def encode_camera(depth_m: Any, intrinsics: Any, base_from_camera: Any) -> Dict[str, Any]:
    """One depth camera's frame and calibration, for ``METADATA_CAMERAS_KEY``.

    ``depth_m`` is metres along the optical axis (not ray length), NaN where the sensor returned
    nothing. ``intrinsics`` is a 3x3 pinhole matrix in pixels, OpenCV convention (+Z forward, +Y
    down). ``base_from_camera`` takes camera points into the frame the policy commands in.
    """
    return {
        "depth_m": np.asarray(depth_m, dtype=np.float32),
        "intrinsics": np.asarray(intrinsics, dtype=np.float32).reshape(3, 3),
        "base_from_camera": np.asarray(base_from_camera, dtype=np.float32).reshape(4, 4),
    }


@dataclasses.dataclass(frozen=True)
class CameraFrame:
    depth_m: np.ndarray  # [H, W] float32 metres, NaN where invalid
    intrinsics: np.ndarray  # [3, 3] pinhole, pixels
    base_from_camera: np.ndarray  # [4, 4] camera frame into the canonical robot base frame

    @property
    def shape(self) -> tuple:
        return tuple(np.asarray(self.depth_m).shape[:2])

    def unproject_camera(self, pixels_xy: np.ndarray) -> np.ndarray:
        """Points in the camera's own frame, which is the frame a grasp sampler works in."""
        pixels = np.asarray(pixels_xy, dtype=np.float64).reshape(-1, 2)
        depth = np.asarray(self.depth_m, dtype=np.float64)
        columns = np.clip(np.round(pixels[:, 0]).astype(int), 0, depth.shape[1] - 1)
        rows = np.clip(np.round(pixels[:, 1]).astype(int), 0, depth.shape[0] - 1)
        z = depth[rows, columns]
        intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        camera = np.stack([(pixels[:, 0] - cx) * z / fx, (pixels[:, 1] - cy) * z / fy, z], axis=1)
        return np.where((np.isfinite(z) & (z > 0.0))[:, None], camera, np.nan)

    def unproject(self, pixels_xy: np.ndarray) -> np.ndarray:
        camera = self.unproject_camera(pixels_xy)
        transform = np.asarray(self.base_from_camera, dtype=np.float64)
        return camera @ transform[:3, :3].T + transform[:3, 3]
