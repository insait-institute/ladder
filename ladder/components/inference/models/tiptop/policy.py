import dataclasses
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from ladder.components.inference.api.client.hlc.vlm.base import VLM
from ladder.components.inference.api.core.action import (
    EndEffectorPoseAction,
    JointAnglesAction,
    ManipulatorAction,
    RobotAction,
)
from ladder.components.inference.api.core.observation import METADATA_CAMERAS_KEY, RobotObservation
from ladder.components.inference.models.tiptop.config import TiptopPolicyConfig
from ladder.components.inference.models.tiptop.plan import GRIPPER_OPEN, PlanCursor, euler_xyz_deg_to_matrix
from ladder.components.inference.models.tiptop.planner import TiptopPlanner
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import AutoConfigurable, Configurable

LOGGER = get_logger(__name__)
_PLANNER: Optional[Any] = None
_PLANNER_LOCK = threading.Lock()


@dataclasses.dataclass
class _Lane:
    steps: int = 0
    instruction: Optional[str] = None
    cursor: Optional[PlanCursor] = None
    last_action: Optional[ManipulatorAction] = None
    reported: bool = False
    remaining_atoms: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    events: List[Dict[str, object]] = dataclasses.field(default_factory=list)


def _shared_planner(config: TiptopPolicyConfig):
    global _PLANNER
    with _PLANNER_LOCK:
        if _PLANNER is None:
            vlm: VLM = AutoConfigurable(config.vlm)
            vlm.warmup()
            _PLANNER = TiptopPlanner(config, vlm)
    return _PLANNER


class TiptopPolicy(Configurable[TiptopPolicyConfig]):
    def __init__(self, config: TiptopPolicyConfig) -> None:
        super().__init__(config)
        self._ee_rotation = euler_xyz_deg_to_matrix(list(config.plan_ee_rotation_deg))
        self._pool = ThreadPoolExecutor(max_workers=config.max_workers)
        self._lanes: Dict[int, _Lane] = {}

    def reset(self) -> None:
        self._lanes = {}

    def predict_actions(
        self,
        observations: List[RobotObservation],
        instructions: List[str],
        env_indices: Optional[List[int]] = None,
    ) -> List[RobotAction]:
        env_indices = list(range(len(observations))) if env_indices is None else list(env_indices)
        lanes = [self._lanes.setdefault(env_idx, _Lane()) for env_idx in env_indices]
        planning: List[Tuple[int, _Lane, RobotObservation, str]] = []
        for env_idx, lane, observation, instruction in zip(
            env_indices, lanes, observations, instructions, strict=True
        ):
            lane.steps += 1
            if lane.steps > self.config.settle_steps and self._wants_plan(lane, instruction):
                planning.append((env_idx, lane, observation, instruction))
        list(self._pool.map(lambda item: self._plan(*item), planning))
        return [self._act(lane, observation) for lane, observation in zip(lanes, observations, strict=True)]

    def _wants_plan(self, lane: _Lane, instruction: str) -> bool:
        if lane.cursor is not None and (not lane.cursor.done):
            return False
        return instruction != lane.instruction or bool(lane.remaining_atoms)

    def _plan(self, env_idx: int, lane: _Lane, observation: RobotObservation, instruction: str) -> None:
        try:
            self._plan_once(env_idx, lane, observation, instruction)
        except Exception as error:  # noqa: BLE001
            LOGGER.exception(f"env {env_idx}: planning failed", error)  # noqa: PLE1205
            lane.events.append({"kind": "plan_failed", "error": str(error) or repr(error)})

    def _plan_once(self, env_idx: int, lane: _Lane, observation: RobotObservation, instruction: str) -> None:
        goal_atoms = (
            lane.remaining_atoms if instruction == lane.instruction and lane.remaining_atoms else None
        )
        lane.instruction = instruction
        lane.cursor = None
        lane.remaining_atoms = []
        rgb, depth_m, intrinsic, world_from_cam, q_init = planning_inputs(observation, self.config.arm_dof)
        started = time.monotonic()
        response = _shared_planner(self.config).plan(
            rgb=rgb,
            depth_m=depth_m,
            intrinsic=intrinsic,
            world_from_cam=world_from_cam,
            task=instruction,
            q_init=q_init,
            goal_atoms=goal_atoms,
        )
        steps = list((response.get("plan") or {}).get("steps", []))
        planned_atoms = list(response.get("planned_atoms") or [])
        lane.remaining_atoms = list(response.get("remaining_atoms") or [])
        if not response.get("success"):
            lane.remaining_atoms = []
            LOGGER.warn(f"env {env_idx}: no plan for {instruction!r} ({response.get('error')}); holding")
            lane.events.append({"kind": "plan_failed", "error": str(response.get("error"))})
            return
        lane.events.append({"kind": "plan", "subtasks": [_goal_key(atom) for atom in planned_atoms]})
        lane.events.append(
            {
                "kind": "code",
                "code": json.dumps(response.get("plan") or {}, default=str),
            }
        )
        lane.reported = False
        lane.cursor = PlanCursor(
            steps,
            execution=self.config.execution,
            control_hz=self.config.control_hz,
            gripper_hold_steps=self.config.gripper_hold_steps,
            arrive_tolerance_m=self.config.arrive_tolerance_m,
            arrive_tolerance_rad=self.config.arrive_tolerance_rad,
            arrive_budget_steps=self.config.arrive_budget_steps,
            ee_rotation=self._ee_rotation,
            waypoint_spacing=self.config.waypoint_spacing,
            approach_refine=self.config.approach_refine,
        )
        LOGGER.info(
            f"env {env_idx} plan for {instruction!r}: {len(steps)} steps, {lane.cursor.total_steps} control steps, planned in {time.monotonic() - started:.1f}s"
        )

    def pop_events(self) -> Dict[int, List[Dict[str, object]]]:
        pending = {env_idx: lane.events for env_idx, lane in self._lanes.items() if lane.events}
        for env_idx in pending:
            self._lanes[env_idx].events = []
        return pending

    def _act(self, lane: _Lane, observation: RobotObservation) -> ManipulatorAction:
        action = lane.cursor.next(observation) if lane.cursor is not None else None
        if lane.cursor is not None and lane.cursor.done and (not lane.reported):
            lane.reported = True
            residuals = lane.cursor.gate_residuals
            unit = "rad" if self.config.execution == "joint_angles" else "m"
            worst = (
                f", residual mean {np.mean(residuals):.4f} max {np.max(residuals):.4f} {unit}"
                if residuals
                else ""
            )
            report = (
                f"{lane.cursor.gates_arrived} gates arrived, {lane.cursor.gates_exhausted} exhausted{worst}"
            )
            LOGGER.info(f"plan executed: {report}")
            lane.events.append(
                {
                    "kind": "tool",
                    "tool": "execute_plan",
                    "report": report,
                    "ok": lane.cursor.gates_exhausted == 0,
                }
            )
        if action is None:
            action = lane.last_action or self._current_pose(observation)
        lane.last_action = action
        return action

    def _current_pose(self, observation: RobotObservation) -> ManipulatorAction:
        if self.config.execution == "joint_angles":
            joints = np.asarray(observation.state.joints, dtype=np.float32).reshape(-1)
            arm = JointAnglesAction(angles=joints[: self.config.arm_dof])
        else:
            target = observation.state.eef_pose_from_target
            pose = np.asarray(observation.state.eef_pose if target is None else target, dtype=np.float32)
            arm = EndEffectorPoseAction(translation=pose[:3, 3].copy(), rotation=pose[:3, :3].copy())
        return ManipulatorAction(arm_action=arm, gripper_action=GRIPPER_OPEN)


def _goal_key(atom: Dict[str, Any]) -> str:
    return f"{atom.get('predicate')}({','.join((str(a) for a in atom.get('args', ())))})"


def planning_inputs(
    observation: RobotObservation, arm_dof: int
) -> Tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    metadata = observation.metadata or {}
    cameras = metadata.get(METADATA_CAMERAS_KEY) or {}
    name, camera = next(iter(cameras.items())) if len(cameras) == 1 else ("main", cameras.get("main"))
    if camera is None:
        raise ValueError(
            f"TiPToP needs a depth camera in metadata[{METADATA_CAMERAS_KEY!r}]; the sim server was started without depth publishing (cameras present: {sorted(cameras)})"
        )
    if observation.images is None:
        raise ValueError("TiPToP needs images in the observation; the runner published none")
    rgb = metadata.get("rgb")
    if rgb is None:
        rgb = observation.images.get(name)
    if rgb is None:
        rgb = observation.images["main"]
    depth_m = np.asarray(camera["depth_m"], dtype=np.float32)
    intrinsic = np.asarray(camera["intrinsics"], dtype=np.float32)
    world_from_cam = np.asarray(camera["base_from_camera"], dtype=np.float32)
    joints = np.asarray(observation.state.joints, dtype=np.float32).reshape(-1)[:arm_dof]
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if rgb.shape[:2] != depth_m.shape[:2]:
        rgb_h, rgb_w = rgb.shape[:2]
        depth_h, depth_w = depth_m.shape[:2]
        if abs(rgb_w / rgb_h - depth_w / depth_h) > 0.001:
            raise ValueError(
                f"TiPToP needs RGB and depth in the same framing, but camera {name!r} published depth {depth_h}x{depth_w} against RGB {rgb_h}x{rgb_w}: different aspect ratios, so the RGB is padded and cannot be registered (images: {sorted(observation.images)})"
            )
        rgb = np.ascontiguousarray(
            cv2.resize(rgb, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR), dtype=np.uint8
        )
    return (rgb, depth_m, intrinsic, world_from_cam, joints)
