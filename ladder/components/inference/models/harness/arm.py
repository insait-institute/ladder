import math
from typing import List, Optional, Tuple

import numpy as np

from ladder.components.inference.api.client.program import Robot
from ladder.components.inference.api.core.action import EndEffectorPoseAction, ManipulatorAction
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.components.inference.api.core.utils import quat_wxyz_to_rotmat
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import Config, Configurable

LOGGER = get_logger(__name__)

GRIPPER_DOWN_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])
_GRIPPER_STILL = 0.005
_GRIPPER_SETTLE_STEPS = 3
_BIAS_GAIN = 0.5
_BIAS_CAP_M = 0.02
_DESCENT_M = 0.01
_STILL_M = 0.002  # the stall check's threshold


def yaw_matrix(angle: float) -> np.ndarray:
    cos, sin = math.cos(angle), math.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    cos = (np.trace(np.asarray(a, dtype=np.float64).T @ np.asarray(b, dtype=np.float64)) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def _turn_toward(a: np.ndarray, b: np.ndarray, max_angle: float) -> np.ndarray:
    """Rotation ``a`` turned toward ``b`` by at most ``max_angle``. A half turn has no unique axis, so
    it is taken whole."""
    angle = _angle_between(a, b)
    if angle <= max_angle or math.sin(angle) < 1e-6:
        return np.asarray(b, dtype=np.float64)
    relative = np.asarray(a, dtype=np.float64).T @ np.asarray(b, dtype=np.float64)
    axis = np.array(
        [relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0], relative[1, 0] - relative[0, 1]]
    ) / (2.0 * math.sin(angle))
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    turn = np.eye(3) + math.sin(max_angle) * skew + (1.0 - math.cos(max_angle)) * skew @ skew
    return np.asarray(a, dtype=np.float64) @ turn


class ArmConfig(Config):
    tool_offset_m: float = 0.0
    blocking_waypoint: bool = True
    max_translation_per_step_m: float = 0.04
    move_steps: int = 25
    arrive_tolerance_m: float = 0.01
    arrive_tolerance_rad: float = 0.15
    settle_tolerance_m: float = 0.02
    # A move has stalled once it gained under 2 mm and 0.01 rad over this many steps.
    stall_window_steps: int = 5
    # End a move that stalls outside settle_tolerance_m instead of re-sending its target until
    # move_steps. Re-sending does not bring a stalled arm closer, and a real WidowX holding a lift
    # that way overloads its shoulder servos within seconds.
    end_move_on_stall: bool = False
    # Advance a ramp from where the arm was last sent instead of where it is measured. An arm that
    # sags below its command (a loaded WidowX, by 1 to 4 cm) would otherwise never climb.
    ramp_from_command: bool = False
    # With ramp_from_command, hold the ramp while the arm stands still more than this behind it, so a
    # command the robot dropped does not turn into a jump once it takes the next one.
    max_ramp_lead_m: Optional[float] = None
    max_rotation_per_step_rad: Optional[float] = None
    transit_clearance_m: float = 0.05
    min_grip_openness: float = 0.0
    grip_from_openness: bool = False
    max_span_m: Optional[float] = None
    verify_place: bool = False
    # Release just above the destination's top: jaws opened inside a bowl catch its wall.
    release_above_rim: bool = False
    open_gripper_steps: int = 10
    close_gripper_steps: int = 15
    workspace_min_m: Optional[List[float]] = None
    workspace_max_m: Optional[List[float]] = None
    # Raw gripper_state of the empty gripper closed and fully open, for a robot that does not publish
    # a normalised [0, 1] openness (the real WidowX publishes the interbotix gripper position).
    gripper_state_closed: Optional[float] = None
    gripper_state_open: Optional[float] = None


class Arm(Configurable[ArmConfig]):
    def __init__(self, config: ArmConfig, robot: Robot) -> None:
        super().__init__(config)
        self._robot = robot
        pose = np.asarray(robot.observation.state.eef_pose, dtype=np.float64)
        self._tool_to_env = quat_wxyz_to_rotmat(GRIPPER_DOWN_WXYZ).T @ pose[:3, :3]
        self._home = pose.copy()
        self._gripper = 1.0 if self.openness() > 0.5 else 0.0
        self._last: Optional[ManipulatorAction] = None
        self._issued_at: Optional[np.ndarray] = None
        self._bias_z = 0.0
        self.steps = 0

    @property
    def pose(self) -> np.ndarray:
        return np.asarray(self._robot.observation.state.eef_pose, dtype=np.float64)

    @property
    def observation(self) -> RobotObservation:
        return self._robot.observation

    @property
    def position_m(self) -> np.ndarray:
        pose = self.pose
        return pose[:3, 3] + (pose[:3, :3] @ self._tool_to_env.T)[:, 2] * self.config.tool_offset_m

    @property
    def bias_z_m(self) -> float:
        return self._bias_z

    @property
    def home_m(self) -> np.ndarray:
        return self._home[:3, 3].copy()

    @property
    def holding(self) -> bool:
        return self._gripper < 0.5

    def openness(self) -> float:
        state = float(np.asarray(self._robot.observation.state.gripper_state).reshape(-1)[0])
        if self.config.gripper_state_closed is None:
            return state
        closed, fully_open = self.config.gripper_state_closed, self.config.gripper_state_open
        return (state - closed) / (fully_open - closed)

    def clamp(self, position_m: np.ndarray) -> np.ndarray:
        low, high = self.config.workspace_min_m, self.config.workspace_max_m
        if low is None or high is None:
            return position_m
        return np.clip(position_m, np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64))

    def move_to(
        self,
        position_m: np.ndarray,
        yaw_rad: float = 0.0,
        tolerance_m: Optional[float] = None,
        vertical_m: Optional[float] = None,
        ramped: bool = False,
    ) -> bool:
        target = self.clamp(np.asarray(position_m, dtype=np.float64).reshape(3))
        tool = yaw_matrix(yaw_rad) @ quat_wxyz_to_rotmat(GRIPPER_DOWN_WXYZ)
        flange = target - tool[:, 2] * self.config.tool_offset_m
        start = self.position_m
        reached = self._drive(
            flange, (tool @ self._tool_to_env).astype(np.float32), tolerance_m, vertical_m, ramped
        )
        if target[2] < start[2] - _DESCENT_M:
            residual = float(self.position_m[2] - target[2])
            self._bias_z = float(
                np.clip(self._bias_z + _BIAS_GAIN * (residual - self._bias_z), -_BIAS_CAP_M, _BIAS_CAP_M)
            )
        return reached

    def transit_to(
        self,
        position_m: np.ndarray,
        clearance_z: float,
        yaw_rad: float = 0.0,
        tolerance_m: Optional[float] = None,
        ramped: bool = False,
    ) -> bool:
        target = self.clamp(np.asarray(position_m, dtype=np.float64).reshape(3))
        over = np.array([target[0], target[1], max(clearance_z, target[2])])
        here = self.position_m
        if here[2] < clearance_z - 1e-4:
            self.move_to(np.array([here[0], here[1], clearance_z]), yaw_rad, ramped=ramped)
        self.move_to(over, yaw_rad, ramped=ramped)
        return self.move_to(target, yaw_rad, tolerance_m, ramped=ramped)

    def open_gripper(self, wait: bool = True) -> None:
        self._hold_gripper(1.0, self.config.open_gripper_steps if wait else 0)

    def close_gripper(self) -> None:
        self._hold_gripper(0.0, self.config.close_gripper_steps)

    def go_home(self) -> bool:
        return self._drive(self._home[:3, 3], self._home[:3, :3].astype(np.float32))

    def _hold_gripper(self, gripper: float, steps: int) -> None:
        if gripper == self._gripper:
            return
        self._gripper = gripper
        travel = [self.openness()]
        for _ in range(steps):
            self._issue(self._last.arm_action if self._last else self._pose_action(self.pose))
            travel.append(self.openness())
            if (
                len(travel) > _GRIPPER_SETTLE_STEPS
                and abs(travel[-1] - travel[-1 - _GRIPPER_SETTLE_STEPS]) < _GRIPPER_STILL
            ):
                return

    @staticmethod
    def _pose_action(pose: np.ndarray) -> EndEffectorPoseAction:
        return EndEffectorPoseAction(
            translation=pose[:3, 3].astype(np.float32), rotation=pose[:3, :3].astype(np.float32)
        )

    def _drive(
        self,
        target_m: np.ndarray,
        rotation: np.ndarray,
        tolerance_m: Optional[float] = None,
        vertical_m: Optional[float] = None,
        ramped: bool = False,
    ) -> bool:
        history: List[Tuple[float, float]] = []
        arrive = self.config.arrive_tolerance_m if tolerance_m is None else tolerance_m
        rise = arrive if vertical_m is None else vertical_m
        settled = self.config.settle_tolerance_m
        distance = angle = float("inf")
        last_arm_action = self._last.arm_action if self._last is not None else None
        if isinstance(last_arm_action, EndEffectorPoseAction):
            commanded = np.asarray(last_arm_action.translation, dtype=np.float64)
            commanded_rotation = np.asarray(last_arm_action.rotation, dtype=np.float64)
        else:
            commanded = self.position_m
            commanded_rotation = self.pose[:3, :3]
        for _ in range(self.config.move_steps):
            pose = self.pose
            offset = target_m - pose[:3, 3]
            distance = float(np.linalg.norm(offset))
            angle = _angle_between(pose[:3, :3], rotation)
            history.append((distance, angle))
            aligned = angle <= self.config.arrive_tolerance_rad
            sideways = float(np.linalg.norm(offset[:2]))
            if sideways <= arrive and abs(float(offset[2])) <= rise and aligned:
                return True
            window = self.config.stall_window_steps
            stalled = len(history) > window and (
                history[-1 - window][0] - distance < 0.002 and history[-1 - window][1] - angle < 0.01
            )
            if stalled:
                settled_here = sideways <= settled and abs(float(offset[2])) <= (
                    settled if vertical_m is None else rise
                )
                if settled_here:
                    return aligned
                if self.config.end_move_on_stall:
                    LOGGER.info(f"move stalled {distance:.3f} m, {angle:.2f} rad short of its target")
                    return False
            if self.config.blocking_waypoint and not ramped:
                self._issue(EndEffectorPoseAction(translation=target_m.astype(np.float32), rotation=rotation))
                continue
            cap = self.config.max_translation_per_step_m
            held = self._left_behind(commanded, pose[:3, 3])
            if self.config.ramp_from_command:
                ahead = target_m - commanded
                step = float(np.linalg.norm(ahead))
                if not held:
                    commanded = target_m if step <= cap else commanded + ahead * (cap / step)
                waypoint = commanded
            else:
                step = distance
                waypoint = target_m if step <= cap or step == 0.0 else pose[:3, 3] + offset * (cap / step)
            if self.config.max_rotation_per_step_rad is not None and not held:
                commanded_rotation = _turn_toward(
                    commanded_rotation, rotation, self.config.max_rotation_per_step_rad
                )
            turned = (
                rotation
                if self.config.max_rotation_per_step_rad is None
                else commanded_rotation.astype(np.float32)
            )
            self._issue(EndEffectorPoseAction(translation=waypoint.astype(np.float32), rotation=turned))
        LOGGER.info(
            f"move did not arrive in {self.config.move_steps} steps (tol {arrive:.3f}): "
            f"{distance:.3f} m, {angle:.2f} rad off"
        )
        return False

    def _left_behind(self, commanded_m: np.ndarray, measured_m: np.ndarray) -> bool:
        if not self.config.ramp_from_command or self._issued_at is None:
            return False
        still = float(np.linalg.norm(measured_m - self._issued_at)) < _STILL_M
        behind = float(np.linalg.norm(commanded_m - measured_m)) > self.config.max_ramp_lead_m
        if still and behind:
            LOGGER.info(
                f"the arm stands {np.linalg.norm(commanded_m - measured_m):.3f} m behind; holding the ramp"
            )
        return still and behind

    def _issue(self, arm_action: EndEffectorPoseAction) -> None:
        if self.config.max_ramp_lead_m is not None:
            self._issued_at = self.pose[:3, 3].copy()
        self._last = ManipulatorAction(arm_action=arm_action, gripper_action=self._gripper)
        self.steps += 1
        self._robot.act(self._last)
