from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt

from ladder.components.inference.api.core.action import (
    EndEffectorPoseAction,
    JointAnglesAction,
    ManipulatorAction,
)
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = 0.0


def euler_xyz_deg_to_matrix(rpy_deg: List[float]) -> npt.NDArray[np.float64]:
    x, y, z = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64))
    cx, sx, cy, sy, cz, sz = (np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z))
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def quaternion_wxyz_to_matrix(quaternion: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64) / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


class PlanCursor:
    def __init__(
        self,
        steps: List[Dict[str, Any]],
        execution: str,
        control_hz: float,
        gripper_hold_steps: int,
        arrive_tolerance_m: float,
        arrive_tolerance_rad: float,
        arrive_budget_steps: int,
        ee_rotation: Optional[npt.NDArray[np.float64]] = None,
        waypoint_spacing: Optional[float] = None,
        approach_refine: float = 3.0,
    ) -> None:
        if execution not in ("joint_angles", "end_effector"):
            raise ValueError(f"Unknown execution mode: {execution}")
        self._execution = execution
        self._control_period = 1.0 / control_hz
        self._arrive_tolerance_m = arrive_tolerance_m
        self._arrive_tolerance_rad = arrive_tolerance_rad
        self._arrive_budget_steps = arrive_budget_steps
        self._spacing = waypoint_spacing
        self._approach_refine = approach_refine
        self._ee_rotation = np.eye(3) if ee_rotation is None else np.asarray(ee_rotation, dtype=np.float64)
        self._targets: List[ManipulatorAction] = []
        self._gated: List[bool] = []
        gripper = GRIPPER_OPEN
        last_arm = None
        leaves_gripper = False
        for index, step in enumerate(steps):
            if step["type"] == "trajectory":
                enters_gripper = index + 1 < len(steps) and steps[index + 1]["type"] == "gripper"
                for arm_action in self._trajectory_actions(step, leaves_gripper, enters_gripper):
                    last_arm = arm_action
                    self._targets.append(ManipulatorAction(arm_action=arm_action, gripper_action=gripper))
                    self._gated.append(False)
                leaves_gripper = False
                if enters_gripper and self._gated:
                    self._gated[-1] = True
            elif step["type"] == "gripper":
                gripper = GRIPPER_CLOSE if step["action"] == "close" else GRIPPER_OPEN
                leaves_gripper = True
                for _ in range(gripper_hold_steps):
                    self._targets.append(ManipulatorAction(arm_action=last_arm, gripper_action=gripper))
                    self._gated.append(False)
            else:
                LOGGER.warn(f"Ignoring unknown plan step type: {step['type']}")
        self._index = 0
        self._gate_steps = 0
        self.gates_arrived = 0
        self.gates_exhausted = 0
        self.gate_residuals: List[float] = []

    def _trajectory_actions(self, step: Dict[str, Any], fine_head: bool, fine_tail: bool) -> List[Any]:
        spacing = self._spacing
        stride = max(1, int(round(self._control_period / float(step.get("dt") or self._control_period))))
        if self._execution == "joint_angles":
            waypoints = np.asarray(step["positions"], dtype=np.float32)
            if spacing is not None:
                travel = np.abs(np.diff(waypoints, axis=0)).max(axis=1) if len(waypoints) > 1 else np.zeros(0)
                waypoints = waypoints[
                    _resample_indices(travel, self._spacings(travel, spacing, fine_head, fine_tail))
                ]
            else:
                waypoints = _subsample(waypoints, stride)
            return [JointAnglesAction(angles=waypoint) for waypoint in waypoints]
        ee_poses = step.get("ee_poses")
        if ee_poses is None:
            raise ValueError(
                "Plan step carries no ee_poses; end_effector execution needs a server that serializes them"
            )
        positions = np.asarray(ee_poses["position"], dtype=np.float64)
        quaternions = np.asarray(ee_poses["quaternion_wxyz"], dtype=np.float64)
        if spacing is not None:
            travel = np.zeros(0)
            if len(positions) > 1:
                travel = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                dots = np.clip(np.abs(np.sum(quaternions[1:] * quaternions[:-1], axis=1)), 0.0, 1.0)
                travel = travel + 0.15 * 2.0 * np.arccos(dots)
            indices = _resample_indices(travel, self._spacings(travel, spacing, fine_head, fine_tail))
            positions, quaternions = (positions[indices], quaternions[indices])
        else:
            positions, quaternions = (_subsample(positions, stride), _subsample(quaternions, stride))
        return [
            EndEffectorPoseAction(
                translation=position.astype(np.float32),
                rotation=(quaternion_wxyz_to_matrix(quaternion) @ self._ee_rotation).astype(np.float32),
            )
            for position, quaternion in zip(positions, quaternions, strict=True)
        ]

    def _spacings(self, travel: npt.NDArray, spacing: float, fine_head: bool, fine_tail: bool) -> npt.NDArray:
        spacings = np.full(len(travel), float(spacing), dtype=np.float64)
        if not (fine_head or fine_tail) or self._approach_refine <= 1.0:
            return spacings
        covered = np.cumsum(travel)
        if fine_head:
            spacings[covered - travel < spacing] = spacing / self._approach_refine
        if fine_tail:
            spacings[covered > (covered[-1] if len(covered) else 0.0) - spacing] = (
                spacing / self._approach_refine
            )
        return spacings

    @property
    def done(self) -> bool:
        return self._index >= len(self._targets)

    @property
    def total_steps(self) -> int:
        return len(self._targets)

    def next(self, observation: RobotObservation) -> Optional[ManipulatorAction]:
        if self.done:
            return None
        while self._index < len(self._targets):
            action = self._targets[self._index]
            if self._gated[self._index]:
                error = self._arrival_error(action, observation)
                if error <= self._tolerance_for(action):
                    self.gates_arrived += 1
                elif self._gate_steps <= self._arrive_budget_steps:
                    self._gate_steps += 1
                    return action
                else:
                    self.gates_exhausted += 1
                self.gate_residuals.append(error)
                self._gate_steps = 0
                self._index += 1
                continue
            self._index += 1
            return action
        return None

    def _tolerance_for(self, action: ManipulatorAction) -> float:
        return (
            self._arrive_tolerance_rad
            if isinstance(action.arm_action, JointAnglesAction)
            else self._arrive_tolerance_m
        )

    def _arrival_error(self, action: ManipulatorAction, observation: RobotObservation) -> float:
        arm = action.arm_action
        if isinstance(arm, JointAnglesAction):
            joints = np.asarray(observation.state.joints, dtype=np.float64).reshape(-1)
            return float(np.abs(joints[: len(arm.angles)] - arm.angles).max())
        pose = np.asarray(observation.state.eef_pose, dtype=np.float64).reshape(4, 4)
        cos_angle = (np.trace(arm.rotation.T @ pose[:3, :3]) - 1.0) / 2.0
        if np.arccos(np.clip(cos_angle, -1.0, 1.0)) > self._arrive_tolerance_rad:
            return float("inf")
        return float(np.linalg.norm(pose[:3, 3] - arm.translation))


def _subsample(waypoints: npt.NDArray, stride: int) -> npt.NDArray:
    if stride <= 1 or len(waypoints) == 0:
        return waypoints
    indices = np.arange(0, len(waypoints), stride)
    if indices[-1] != len(waypoints) - 1:
        indices = np.append(indices, len(waypoints) - 1)
    return waypoints[indices]


def _resample_indices(travel: npt.NDArray, spacing: npt.NDArray) -> npt.NDArray:
    cumulative = np.concatenate([[0.0], np.cumsum(travel)])
    picked: List[int] = []
    anchor = 0.0
    for index in range(1, len(cumulative)):
        if cumulative[index] - anchor >= spacing[index - 1]:
            picked.append(index)
            anchor = float(cumulative[index])
    last = len(cumulative) - 1
    if not picked or picked[-1] != last:
        picked.append(last)
    return np.asarray(picked)
