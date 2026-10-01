import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import gpt_policy.geometry.poses
import numpy as np

from ladder.components.inference.api.core.action import EndEffectorPoseAction, ManipulatorAction
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.core.databib.config import Config, Configurable

_STILL_STEPS = 3
_GRIPPER_STILL = 0.005


class HostConfig(Config):
    tool_offset_m: float = 0.0
    reset_tool_quat_xyzw: List[float] = [1.0, 0.0, 0.0, 0.0]
    max_translation_per_step_m: float = 0.02
    max_rotation_per_step_rad: float = 0.2
    settle_steps: int = 12
    settle_still_m: float = 0.002
    gripper_steps: int = 12
    max_steps_per_segment: int = 60
    workspace_min_m: Optional[List[float]] = None
    workspace_max_m: Optional[List[float]] = None
    control_hz: float = 5.0


class ToolRejectedError(ValueError):
    pass


class SimRobot(Configurable[HostConfig]):
    def __init__(
        self,
        config: HostConfig,
        observation: Callable[[], RobotObservation],
        act: Callable[[ManipulatorAction], None],
    ) -> None:
        super().__init__(config)
        self._observation = observation
        self._act = act
        pose = self._flange()
        reset_tool = _matrix_from_quat(np.asarray(config.reset_tool_quat_xyzw, dtype=np.float64))
        self._flange_to_tool = pose[:3, :3].T @ reset_tool
        self._gripper = 1.0 if self._openness() > 0.5 else 0.0
        self._commanded: Optional[np.ndarray] = None

    def state(self) -> Dict[str, Any]:
        observation = self._observation()
        joints = np.asarray(
            observation.state.joints if observation.state.joints is not None else [], dtype=np.float64
        ).reshape(-1)
        tcp = self.tcp_pose()
        zeros = [0.0] * len(joints)
        return {
            "joint_positions_rad": joints.tolist(),
            "joint_velocities_rad_s": zeros,
            "joint_torques_nm": zeros,
            "tcp_xyzrpy": gpt_policy.geometry.poses.matrix_to_pose(tcp).tolist(),
            "tcp_xyzquat": [
                *tcp[:3, 3].tolist(),
                *gpt_policy.geometry.poses.rpy_to_quaternion(
                    gpt_policy.geometry.poses.matrix_to_pose(tcp)[3:]
                ),
            ],
            "gripper_position_m": self._openness(),
            "gripper_normalized": self._openness(),
            "gripper_command_normalized": self._gripper,
            "gripper_velocity_m_s": 0.0,
            "gripper_torque_nm": 0.0,
            "gravity_compensation": False,
            "timestamp_s": time.time(),
        }

    def tcp_pose(self) -> np.ndarray:
        return self._tool_pose(self._flange())

    def execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name == "move_to":
            return self._move([_target_pose(arguments["target"])])
        if name == "move_eef_chunk":
            poses = arguments["poses"]
            if not isinstance(poses, list) or not poses:
                raise ToolRejectedError("poses must be a non-empty array of waypoints")
            return self._move([_target_pose(pose) for pose in poses])
        if name == "set_gripper":
            return self._set_gripper(arguments)
        raise ToolRejectedError(f"tool is not available on this host: {name}")

    def return_home(self) -> Dict[str, Any]:
        return {"home": {"settle": {"settled": True}}, **self.state()}

    def _tool_pose(self, flange: np.ndarray) -> np.ndarray:
        tool = flange[:3, :3] @ self._flange_to_tool
        transform = np.eye(4)
        transform[:3, :3] = tool
        transform[:3, 3] = flange[:3, 3] + tool[:, 2] * self.config.tool_offset_m
        return transform

    def _move(self, targets: List[np.ndarray]) -> Dict[str, Any]:
        planned = [gpt_policy.geometry.poses.matrix_to_pose(self.tcp_pose()).tolist()]
        budget = self.config.max_steps_per_segment
        for index, target in enumerate(targets):
            clamped = self._clamp(target)
            planned.append(gpt_policy.geometry.poses.matrix_to_pose(clamped).tolist())
            settle = self.config.settle_steps if index == len(targets) - 1 else 0
            budget -= self._drive_to(clamped, settle, budget)
            if budget <= 0:
                break
        return self._result(self._clamp(targets[-1]), planned)

    def _set_gripper(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        requested = arguments.get("gripper")
        if not isinstance(requested, (int, float)) or not 0.0 <= float(requested) <= 1.0:
            raise ToolRejectedError("gripper must be a number between 0 and 1")
        self._gripper = float(requested)
        travel = [self._openness()]
        for _ in range(self.config.gripper_steps):
            self._issue(self._commanded if self._commanded is not None else self.tcp_pose())
            travel.append(self._openness())
            if len(travel) > _STILL_STEPS and abs(travel[-1] - travel[-1 - _STILL_STEPS]) < _GRIPPER_STILL:
                break
        result = self.state()
        result["execution_feedback"] = {
            "gripper_requested_normalized": self._gripper,
            "gripper_submitted_normalized": self._gripper,
            "gripper_measured_normalized": result["gripper_normalized"],
            "gripper_residual_normalized": abs(self._gripper - result["gripper_normalized"]),
            "steps_used": len(travel) - 1,
        }
        return result

    def _drive_to(self, target: np.ndarray, settle_steps: int, cap: int) -> int:
        start = self.tcp_pose()
        samples = self._sample_count(start, target)
        used = 0
        start_pose = gpt_policy.geometry.poses.matrix_to_pose(start)
        target_pose = gpt_policy.geometry.poses.matrix_to_pose(target)
        for index in range(1, samples + 1):
            if used >= cap:
                break
            self._issue(
                _matrix_of(
                    gpt_policy.geometry.poses.interpolate_pose(start_pose, target_pose, index / samples)
                )
            )
            used += 1
        history: List[float] = []
        for _ in range(settle_steps):
            if used >= cap:
                break
            self._issue(target)
            used += 1
            history.append(float(np.linalg.norm(self.tcp_pose()[:3, 3] - target[:3, 3])))
            if len(history) > _STILL_STEPS and (
                abs(history[-1] - history[-1 - _STILL_STEPS]) < self.config.settle_still_m
            ):
                break
        return used

    def _sample_count(self, start: np.ndarray, target: np.ndarray) -> int:
        distance = float(np.linalg.norm(target[:3, 3] - start[:3, 3]))
        angle = gpt_policy.geometry.poses.rotation_distance(
            gpt_policy.geometry.poses.matrix_to_pose(start)[3:],
            gpt_policy.geometry.poses.matrix_to_pose(target)[3:],
        )
        return max(
            1,
            math.ceil(distance / self.config.max_translation_per_step_m),
            math.ceil(angle / self.config.max_rotation_per_step_rad),
        )

    def _result(self, target: np.ndarray, planned: List[List[float]]) -> Dict[str, Any]:
        result = self.state()
        measured = np.asarray(result["tcp_xyzrpy"], dtype=np.float64)
        target_pose = gpt_policy.geometry.poses.matrix_to_pose(target)
        error = target_pose[:3] - measured[:3]
        result["trajectory"] = {
            "planned_duration_s": (len(planned) - 1) / max(self.config.control_hz, 1e-6),
            "_trace": {"model_tcp_points_xyzrpy": planned},
        }
        result["execution_feedback"] = {
            "target_tcp_xyzrpy": target_pose.tolist(),
            "measured_tcp_xyzrpy": measured.tolist(),
            "tcp_error_xyz_m": error.tolist(),
            "tcp_translation_error_m": float(np.linalg.norm(error)),
            "tcp_rotation_error_rad": gpt_policy.geometry.poses.rotation_distance(
                target_pose[3:], measured[3:]
            ),
            "gripper_measured_normalized": result["gripper_normalized"],
            "settle": {"settled": bool(np.linalg.norm(error) <= 0.02)},
        }
        return result

    def _clamp(self, target: np.ndarray) -> np.ndarray:
        low, high = self.config.workspace_min_m, self.config.workspace_max_m
        if low is None or high is None:
            return target
        clamped = target.copy()
        clamped[:3, 3] = np.clip(target[:3, 3], np.asarray(low), np.asarray(high))
        return clamped

    def _issue(self, tcp: np.ndarray) -> None:
        self._commanded = tcp
        tool = tcp[:3, :3]
        flange_rotation = tool @ self._flange_to_tool.T
        translation = tcp[:3, 3] - tool[:, 2] * self.config.tool_offset_m
        self._act(
            ManipulatorAction(
                arm_action=EndEffectorPoseAction(
                    translation=translation.astype(np.float32),
                    rotation=flange_rotation.astype(np.float32),
                ),
                gripper_action=self._gripper,
            )
        )

    def _flange(self) -> np.ndarray:
        pose = self._observation().state.eef_pose
        if pose is None:
            raise RuntimeError("the environment published no end-effector pose")
        return np.asarray(pose, dtype=np.float64)

    def _openness(self) -> float:
        gripper = self._observation().state.gripper_state
        return 0.0 if gripper is None else float(np.asarray(gripper).reshape(-1)[0])


def _target_pose(target: Any) -> np.ndarray:
    if not isinstance(target, dict) or "pose_xyzquat" not in target:
        raise ToolRejectedError("each waypoint must be an object with pose_xyzquat")
    values = np.asarray(target["pose_xyzquat"], dtype=np.float64).reshape(-1)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ToolRejectedError("pose_xyzquat must be seven finite numbers [x,y,z,qx,qy,qz,qw]")
    if float(np.linalg.norm(values[3:])) < 1e-6:
        raise ToolRejectedError("pose_xyzquat orientation cannot be a zero quaternion")
    transform = np.eye(4)
    transform[:3, :3] = _matrix_from_quat(values[3:])
    transform[:3, 3] = values[:3]
    return transform


def _matrix_from_quat(xyzw: np.ndarray) -> np.ndarray:
    return gpt_policy.geometry.poses.rpy_to_matrix(gpt_policy.geometry.poses.quaternion_to_rpy(xyzw))


def _matrix_of(pose: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = gpt_policy.geometry.poses.rpy_to_matrix(pose[3:])
    transform[:3, 3] = pose[:3]
    return transform


class Localizer:
    def __init__(
        self,
        cameras: Dict[str, str],
        observation: Callable[[], RobotObservation],
        shown_shape: Callable[[str], Optional[Tuple[int, int]]],
    ) -> None:
        self._cameras = cameras
        self._observation = observation
        self._shown_shape = shown_shape

    def locate(
        self, arguments: Dict[str, Any], state: Dict[str, Any], history: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        del state, history
        requested = str(arguments.get("camera", ""))
        name = self._cameras.get(requested)
        if name is None:
            raise ToolRejectedError(f"no such camera: {requested}; available: {sorted(self._cameras)}")
        pixels = np.asarray(arguments.get("pixels_xy", []), dtype=np.float64).reshape(-1, 2)
        if pixels.shape[0] == 0 or not np.isfinite(pixels).all():
            raise ToolRejectedError("pixels_xy must be a non-empty list of finite [x,y] pairs")

        frame = self._observation().cameras.get(name)
        if frame is None:
            return {
                "camera": requested,
                "points": [{"pixel_xy": p.tolist(), "metric_position_available": False} for p in pixels],
                "explanation": f"camera {requested} publishes no depth, so a pixel gives no position",
            }

        height, width = frame.shape
        shown = self._shown_shape(requested)
        scale = np.ones(2) if shown is None else np.array([width / shown[1], height / shown[0]])
        scaled = pixels * scale
        inside = (scaled[:, 0] >= 0) & (scaled[:, 0] < width) & (scaled[:, 1] >= 0) & (scaled[:, 1] < height)
        positions = frame.unproject(scaled)
        metric = inside & np.isfinite(positions).all(axis=1)
        points = [
            _point(pixel, position, is_inside, is_metric, width, height)
            for pixel, position, is_inside, is_metric in zip(pixels, positions, inside, metric, strict=True)
        ]
        return {
            "camera": requested,
            "ray_origin_base_xyz": frame.base_from_camera[:3, 3].tolist(),
            "points": points,
            "explanation": _explain(int((~inside).sum()), int((~metric).sum()), len(points), width, height),
        }


def _point(
    pixel: np.ndarray, position: np.ndarray, inside: bool, metric: bool, width: int, height: int
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"pixel_xy": pixel.tolist(), "metric_position_available": bool(metric)}
    if not inside:
        entry["explanation"] = f"outside the {width}x{height} image, so there is nothing to read"
    elif metric:
        entry["metric_position_base_xyz"] = position.tolist()
    return entry


def _explain(outside: int, missing: int, total: int, width: int, height: int) -> str:
    if missing == 0:
        return "depth was valid at every pixel"
    if outside > 0:
        return (
            f"{outside} of {total} pixels fell outside the {width}x{height} image; "
            f"x must be 0..{width - 1} and y 0..{height - 1}"
        )
    if missing == total:
        return (
            "this frame carried no depth at all, which happens intermittently and is not about the "
            "pixels you chose; act on your last good reading or move the arm and look again, but do "
            "not simply repeat this call"
        )
    return f"the depth sensor returned nothing at {missing} of {total} pixels; pick another point on the same object"
