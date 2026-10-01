import dataclasses
import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ladder.components.inference.api.client.program import Robot
from ladder.components.inference.api.core.action import EndEffectorPoseAction, ManipulatorAction
from ladder.components.inference.api.core.utils import quat_wxyz_to_rotmat, rotmat_to_quat_wxyz
from ladder.components.inference.models.capx.perception.scene import Scene
from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)

_GRIPPER_DOWN_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])
_FEED_FORWARD_CAP_M = 0.04
_SQUARE_ENOUGH_M = 0.005
_STALLED_M = 0.005
_FALLBACK_TILTS = (("top-down", 0, 0.0), ("45-tilt", 0, 0.25 * math.pi), ("side-approach", 1, 0.5 * math.pi))


@dataclasses.dataclass(frozen=True)
class Motion:
    tool_offset_m: float
    max_translation_per_step_m: float
    move_steps: int
    arrive_tolerance_m: float
    arrive_tolerance_rad: float
    open_gripper_steps: int
    close_gripper_steps: int
    align_grasp_yaw: bool
    workspace_min_m: Optional[List[float]] = None
    workspace_max_m: Optional[List[float]] = None
    orientation_fallbacks: bool = True
    settle_tolerance_m: float = 0.02
    blocking_waypoint: bool = True
    reachability_clamp: bool = False


class CapxApi:
    def __init__(self, robot: Robot, motion: Motion, scene: Scene, name: str = "") -> None:
        self._robot = robot
        self._motion = motion
        self._scene = scene
        self._name = f"{name}: " if name else ""
        state = robot.observation.state
        self._tool_to_env = quat_wxyz_to_rotmat(_GRIPPER_DOWN_WXYZ).T @ np.asarray(
            state.eef_pose[:3, :3], dtype=np.float64
        )
        self._arm = EndEffectorPoseAction(
            translation=np.asarray(state.eef_pose[:3, 3], dtype=np.float32).copy(),
            rotation=np.asarray(state.eef_pose[:3, :3], dtype=np.float32).copy(),
        )
        self._gripper = 1.0 if self._openness() > 0.5 else 0.0
        self._last_arm = self._arm
        self._home = np.asarray(state.eef_pose, dtype=np.float64).copy()
        self.calls: List[str] = []
        self.reports: List[str] = []

    def functions(self) -> Dict[str, Callable[..., Any]]:
        return {
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "home_pose": self.home_pose,
        }

    def combined_doc(self) -> str:
        return combined_doc(self.functions())

    def get_object_pose(
        self, object_name: str, return_bbox_extent: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Get the pose of an object in the environment from a natural language description.
        The quaternion from get_object_pose may be unreliable, so disregard it and use the grasp pose quaternion OR (0, 0, 1, 0) wxyz as the gripper down orientation if using this for placement position.

        Args:
            object_name: The name of the object to get the pose of.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            bbox_extent: (3,) object extent in meters of x, y, z axes respectively in the world frame (full side length, not half-length extent). If return_bbox_extent is False, returns None.
        """
        self.calls.append(f"get_object_pose({object_name!r})")
        found = self._scene.locate(object_name)
        extent = found.extent_m.copy() if return_bbox_extent else None
        return found.center_m.copy(), found.quaternion_wxyz.copy(), extent

    def sample_grasp_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object in the environment from a natural language description.
        Do use the grasp sample quaternion from sample_grasp_pose.

        Args:
            object_name: The name of the object to sample a grasp pose for.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        self.calls.append(f"sample_grasp_pose({object_name!r})")
        found = self._scene.locate(object_name)
        yaw = _grasp_yaw(found) if self._motion.align_grasp_yaw else 0.0
        return found.center_m.copy(), rotmat_to_quat_wxyz(_yaw(yaw) @ quat_wxyz_to_rotmat(_GRIPPER_DOWN_WXYZ))

    def goto_pose(self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0) -> None:
        """Go to pose using Inverse Kinematics.
        There is no need to call a second goto_pose with the same position and quaternion_wxyz after calling it with z_approach.
        Args:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            z_approach: (float) Z-axis distance offset for goto_pose insertion approach motion. Will first arrive at position + z_approach meters in Z-axis before moving to the requested pose. Useful for more precise grasp approaches. Default is 0.0.
        Returns:
            None
        """
        pos = self._reachable(np.asarray(position, dtype=np.float64).reshape(3))
        wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        self.calls.append(f"goto_pose({np.round(pos, 3).tolist()}, z_approach={float(z_approach):g})")
        offset_pos = pos
        for label, rot in self._orientations(wxyz):
            offset_pos = pos - rot[:, 2] * self._motion.tool_offset_m
            env_rot = (rot @ self._tool_to_env).astype(np.float32)
            start = self._eef()
            arrived = True
            if z_approach != 0.0:
                arrived = self._move(offset_pos + rot @ np.array([0.0, 0.0, -float(z_approach)]), env_rot)
            arrived = self._move(offset_pos, env_rot) and arrived
            if arrived:
                if label != "requested":
                    self._say(f"goto_pose: reached the pose with the {label} orientation")
                return
            if float(np.linalg.norm(self._eef() - start)) > _STALLED_M:
                break
        eef = self._eef()
        self._say(
            f"goto_pose: did not reach the pose, stopped {np.linalg.norm(offset_pos - eef):.3f} m away",
            warn=True,
        )

    def _orientations(self, wxyz: np.ndarray):
        yield "requested", quat_wxyz_to_rotmat(wxyz)
        if self._motion.orientation_fallbacks:
            down = quat_wxyz_to_rotmat(_GRIPPER_DOWN_WXYZ)
            for label, axis, angle in _FALLBACK_TILTS:
                yield label, down @ _about(axis, angle)

    def _reachable(self, position: np.ndarray) -> np.ndarray:
        low, high = self._motion.workspace_min_m, self._motion.workspace_max_m
        if low is None or high is None or not self._motion.reachability_clamp:
            return position
        clamped = np.clip(position, np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64))
        if not np.allclose(clamped, position):
            self._say(f"goto_pose: clamped {np.round(position, 3).tolist()} into the workspace")
        return clamped

    def open_gripper(self) -> None:
        """Open gripper fully.

        Args:
            None
        """
        self.calls.append("open_gripper()")
        self._hold_gripper(1.0, self._motion.open_gripper_steps)

    def close_gripper(self) -> None:
        """Close gripper fully.

        Args:
            None
        """
        self.calls.append("close_gripper()")
        self._hold_gripper(0.0, self._motion.close_gripper_steps)

    def home_pose(self) -> None:
        """Move the robot to a safe home pose.

        Args:
            None
        """
        self.calls.append("home_pose()")
        self._move(self._home[:3, 3], self._home[:3, :3].astype(np.float32))

    def _say(self, message: str, *, warn: bool = False) -> None:
        self.reports.append(message)
        (LOGGER.warn if warn else LOGGER.info)(f"{self._name}{message}")

    def _eef(self) -> np.ndarray:
        return np.asarray(self._robot.observation.state.eef_pose, dtype=np.float64)[:3, 3].copy()

    def _openness(self) -> float:
        return float(np.asarray(self._robot.observation.state.gripper_state).reshape(-1)[0])

    def _hold_gripper(self, gripper: float, steps: int) -> None:
        if gripper == self._gripper:
            return
        self._gripper = gripper
        for _ in range(steps):
            self._hold()

    def _move(self, target_m: np.ndarray, rotation: np.ndarray) -> bool:
        rotation = np.asarray(rotation, dtype=np.float32)
        history: List[Tuple[float, float]] = []
        distance = angle = float("inf")
        reached = False
        for _ in range(self._motion.move_steps):
            pose = np.asarray(self._robot.observation.state.eef_pose, dtype=np.float64)
            residual = target_m - pose[:3, 3]
            distance, angle = float(np.linalg.norm(residual)), _angle(pose[:3, :3], rotation)
            history.append((distance, angle))
            if distance <= self._motion.arrive_tolerance_m and angle <= self._motion.arrive_tolerance_rad:
                reached = True
                break
            stalled = len(history) > 5 and (
                history[-6][0] - distance < 0.002 and history[-6][1] - angle < 0.01
            )
            if stalled and distance <= self._motion.settle_tolerance_m:
                LOGGER.info(f"{self._name}goto_pose settled {distance:.3f} m, {angle:.2f} rad off the target")
                reached = angle <= self._motion.arrive_tolerance_rad
                break
            if self._motion.blocking_waypoint:
                self._command(target_m, rotation)
                continue
            bias = np.zeros(3)
            if np.linalg.norm(residual[:2]) <= _FEED_FORWARD_CAP_M:
                bias[:2] = np.clip(residual[:2], -_FEED_FORWARD_CAP_M, _FEED_FORWARD_CAP_M)
            self._command(
                _toward(pose[:3, 3], target_m + bias, self._motion.max_translation_per_step_m), rotation
            )
        if not reached:
            LOGGER.warn(
                f"{self._name}goto_pose did not arrive in {self._motion.move_steps} steps: "
                f"{distance:.3f} m, {angle:.2f} rad off"
            )
        self._arm = EndEffectorPoseAction(translation=self._eef().astype(np.float32), rotation=rotation)
        return reached

    def _command(self, translation: np.ndarray, rotation: np.ndarray) -> None:
        self._arm = EndEffectorPoseAction(
            translation=np.asarray(translation, dtype=np.float32).copy(),
            rotation=np.asarray(rotation, dtype=np.float32).copy(),
        )
        self._issue(self._arm)

    def _issue(self, arm_action) -> None:
        self._last_arm = arm_action
        self._robot.act(ManipulatorAction(arm_action=arm_action, gripper_action=self._gripper))

    def _hold(self) -> None:
        self._robot.act(ManipulatorAction(arm_action=self._last_arm, gripper_action=self._gripper))


def _grasp_yaw(found) -> float:
    extent = np.asarray(found.extent_m, dtype=np.float64)
    wxyz = np.asarray(found.quaternion_wxyz, dtype=np.float64)
    box_yaw = 2.0 * math.atan2(wxyz[3], wxyz[0])
    if extent[1] > extent[0] + _SQUARE_ENOUGH_M:
        return box_yaw + math.pi / 2
    if extent[0] > extent[1] + _SQUARE_ENOUGH_M:
        return box_yaw
    return 0.0


def combined_doc(functions: Dict[str, Callable[..., Any]], extra: Optional[Dict[str, str]] = None) -> str:
    lines: List[str] = []
    for name, fn in functions.items():
        lines.append(f"{name}{inspect.signature(fn)}")
        doc = inspect.getdoc(fn) or ""
        appended = (extra or {}).get(name)
        if appended:
            doc = f"{doc}\n\n{appended}" if doc else appended
        if doc:
            lines.append("  Doc:")
            lines.extend(f"    {ln}" for ln in doc.splitlines())
        lines.append("")
    return "\n".join(lines).strip()


def _toward(eef_m: np.ndarray, target_m: np.ndarray, max_step_m: float) -> np.ndarray:
    delta = target_m - eef_m
    dist = float(np.linalg.norm(delta))
    if dist <= max_step_m or dist == 0.0:
        return target_m.copy()
    return eef_m + delta * (max_step_m / dist)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    cos = (np.trace(np.asarray(a, dtype=np.float64).T @ np.asarray(b, dtype=np.float64)) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def _about(axis: int, angle: float) -> np.ndarray:
    cos, sin = math.cos(angle), math.sin(angle)
    rotation = np.eye(3)
    other = [i for i in range(3) if i != axis]
    rotation[np.ix_(other, other)] = [[cos, -sin], [sin, cos]]
    return rotation


def _yaw(angle: float) -> np.ndarray:
    cos, sin = math.cos(angle), math.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
