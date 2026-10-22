import dataclasses
import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ladder.components.inference.api.client.scripted.grounding import ground_subtask
from ladder.components.inference.api.client.scripted.primitives import (
    EnvState,
    MotionParams,
    Phase,
    rim_direction,
    step,
)
from ladder.components.inference.api.core.action import (
    EndEffectorPoseAction,
    JointAnglesAction,
    ManipulatorAction,
    RobotAction,
)
from ladder.components.inference.api.core.observation import ManipulatorState, RobotObservation
from ladder.components.inference.api.core.privileged_state import (
    PrivilegedCondition,
    PrivilegedObject,
    PrivilegedState,
    from_metadata,
)
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import Config, Configurable

LOGGER = get_logger(__name__)


# Heights (pad z, base frame) at which the wall limits were measured: wall_max_y_m and
# wall_hover_max_y_m.
_WALL_LOW_Z_M = 0.04
_WALL_HOVER_Z_M = 0.10
_CLEAR_END_M = 0.02  # a grip slid along an object stops this far from its end
_CLEAR_MARGIN_M = 0.01  # a finger this close to a neighbour's footprint counts as on it
_JOINT_HOME_TOL_RAD = 0.05
_JOINT_HOME_MAX_STEPS = 40


def _rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix by ``angle`` (radians) about ``axis`` (Rodrigues)."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9 or angle == 0.0:
        return np.eye(3, dtype=np.float32)
    x, y, z = axis / norm
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)).astype(np.float32)


def _jaw_yaw(angle: float) -> float:
    """``angle`` wrapped into [-pi/2, pi/2).

    Two parallel jaws grip the same way turned by a half turn, but the arm does not reach the two the
    same way: a wrist asked to turn 180 or 270 degrees swings the whole arm through its IK instead.
    """
    return (angle + math.pi / 2) % math.pi - math.pi / 2


class ScriptedPolicyConfig(Config):
    """Config for :class:`ScriptedPolicy`.

    The geometry / timing fields mirror :class:`MotionParams`; they are exposed so a
    sweep can retune the baseline (e.g. descent depth, lift height) without code
    changes. ``require_privileged`` fails loudly when the server did not emit the
    privileged payload, rather than silently no-op (a misconfiguration guard).
    """

    max_translation_per_step_m: float = 0.025
    transit_translation_per_step_m: Optional[float] = None
    approach_height_m: float = 0.10
    grasp_z_offset_m: float = 0.0
    grasp_from_top: bool = False
    # See MotionParams.rim_grasp_min_span_m / rim_inset_m / close_above_m.
    rim_grasp_min_span_m: float = 0.0
    rim_inset_m: float = 0.012
    rim_along_jaw: bool = False
    close_above_m: float = 0.0
    rim_z_offset_m: Optional[float] = None
    close_descent_step_m: float = 0.002
    # Rim grasps tilt the jaw axis this far from level, raising the finger outside the wall: a bowl's
    # wall flares, so a level pinch pushes about 45 degrees up out of the jaw plane, and RobotArena's
    # two-finger grasp test only accepts forces within 30 degrees of that plane.
    rim_roll_deg: float = 0.0
    lift_height_m: float = 0.12
    carry_height_m: float = 0.12
    place_clearance_m: float = 0.03
    xy_tol_m: float = 0.012
    z_tol_m: float = 0.015
    grasp_close_steps: int = 3
    max_grasp_steps: int = 20
    release_steps: int = 3
    # See MotionParams.shed_pulses.
    shed_pulses: int = 0
    # See MotionParams.wide_grasp_depth_m.
    wide_grasp_depth_m: Optional[float] = None
    wide_object_min_m: float = 0.09
    wide_object_min_height_m: float = 0.0
    max_phase_steps: int = 25
    require_privileged: bool = True
    # Restarts allowed on one goal before moving to the next. Sticking forever lets a single
    # unreachable object hide whether the rest of the scene is reachable.
    max_goal_attempts: int = 3
    # Yaw the wrist so a long object is gripped across its short axis. Only needed where objects
    # rest at free yaw (RobotArena).
    align_grasp_yaw: bool = False
    # Wrist orientations to grasp with, as [yaw, pitch] degrees about the base z and y axes applied to
    # the grasp rotation, tried in order. The arm's IK solves locally, so a pose it cannot reach at one
    # orientation is often reachable at another (SimplerEnv's +y edge: see the suite config). The next
    # one is tried when the reach check below fires; None keeps the single grasp rotation.
    grasp_orientations_deg: Optional[List[List[float]]] = None
    # See MotionParams.reach_stall_steps.
    reach_stall_steps: int = 0
    # See MotionParams.grasp_confirm_steps / grasp_lost_steps.
    grasp_confirm_steps: int = 1
    grasp_lost_steps: int = 0
    # See MotionParams.jaw_stall_steps / descent_high_tol_m.
    jaw_stall_steps: int = 0
    descent_high_tol_m: float = 0.0
    # Move to the next grasp orientation after ANY failed chain, not only a reach stall: a grip that
    # slipped or never formed at one orientation is repeated identically otherwise.
    rotate_on_failure: bool = False
    # The arm's reach, as a sphere the WRIST must stay inside: its centre (the shoulder pivot, base
    # frame), its radius, and how far the wrist sits behind the commanded point along the gripper's
    # approach axis. A waypoint past it (carrying at transit height to a far plate) straightens the
    # elbow, and the controller's one-step IK cannot bend it back from that singularity, so the arm
    # freezes for the rest of the episode. Waypoints are pulled inside, giving up height first, or
    # with reach_keep_height giving up reach: a transit lowered to a far goal sweeps through whatever
    # already stands on it (SimplerEnv's piles), while a lower carry drops a credited object from less
    # height (RobotArena's cloth). None disables the clamp.
    reach_center_m: Optional[List[float]] = None
    reach_radius_m: float = 0.48
    wrist_offset_m: float = 0.158
    reach_keep_height: bool = False
    # A wall the gripper may meet, as a base-frame y: at the benchmark's 5 m env spacing a neighbouring
    # env's stage wall stands at y=0.185 in 16 of SimplerEnv's 24 envs. An env counts as walled once a
    # chain there ends out of reach with its target past ``wall_turn_y_m``. From then on, jaws open
    # level across y (which reach only 0.12) are avoided for such targets, and grasp and goal points
    # are pulled back to ``wall_max_y_m``, where the gripper still fits. None disables.
    wall_max_y_m: Optional[float] = None
    wall_turn_y_m: float = 0.10
    # Past wall_level_max_y_m, the reach of level jaws across y at table height, a target takes a
    # wall pose instead of the usual orientation retries: jaws across y roll this far about base x
    # (the outer finger rises over the object, the fingertips lean toward the wall). A rolled
    # gripper cannot get as low, so nothing short of that limit rolls.
    wall_roll_deg: float = 30.0
    wall_level_max_y_m: float = 0.15
    # Higher up, the wrist meets the wall first: at hover height the gripper gets only this far, so
    # the allowed y narrows from wall_max_y_m near the table to this at hover height.
    wall_hover_max_y_m: float = 0.12
    # A gripper held back by the wall stops between wall_hover_max_y_m and wall_stop_max_y_m; a stop
    # anywhere else is some other obstacle (a pile, the object itself) and says nothing about a wall.
    # An env where the gripper has been past wall_free_y_m, further out than any grasp pose gets
    # next to the wall, has none.
    wall_stop_max_y_m: float = 0.14
    wall_free_y_m: float = 0.175
    # Goals whose object has fallen below this height (base frame) are off the table and cannot be
    # completed; they are skipped instead of spending the rest of the budget on them. None disables.
    fallen_z_m: Optional[float] = None
    # The arm's horizontal reach from the base (base frame): an object resting farther out cannot be
    # grasped from above, so its goal is skipped like a fallen one, and a goal point farther out is
    # pulled back toward the base inside its destination. A place past it drops the arm to the table.
    # None disables.
    max_reach_radius_m: Optional[float] = None
    # Drive the arm back to its first-observed joints, in joint space, before a new pick whenever the
    # base, upper-arm or forearm twist (joints 1, 3, 5) has drifted this far from them. The server's
    # IK has no rest-pose pull, so over an episode the redundant joints wander until the base joint
    # sits at its limit and far targets become unreachable. None disables.
    joint_rehome_drift_rad: Optional[float] = None
    # On a task the env scores as a sequence (metadata["oracle"]["ordered"]), stay on the active goal
    # until it holds instead of giving up on it (max_goal_attempts, fallen, out of reach): a later
    # goal completed first fails the task anyway, and moving on is what breaks the order.
    respect_order: bool = False
    # Grip a curved object (a banana) on its body rather than its AABB centre, which lies in the gap
    # the curve encloses: attempts alternate the grasp point to either side along the jaw axis, by the
    # AABB's half width less its half height (the body's radius), then try the centre.
    grasp_body_offset: bool = False
    # Finish a delivery under way even once it is credited, onto a destination that is not a
    # container (relation "in") and while nothing else is credited on it yet. A success test that
    # scores the object over its destination fires while it still hangs in the jaws, and moving on
    # opens them at carry height: a dropped bowl bounces off RobotArena's cloth. Onto objects already
    # delivered the drop is the gentler of the two (a placed arrival presses them apart), and on
    # SimplerEnv's piles it is too, so this stays off there.
    finish_credited_delivery: bool = False
    # Finish the release and climb straight out of the destination before the next object, also when
    # the delivery is credited on the way down (it drops there): setting off for the next object from
    # down there sweeps the open jaws through what the destination holds (RobotArena's scene45 bowl,
    # RoboLab's bins). SimplerEnv's budgets are too tight for the climb, so it stays off there.
    climb_out_after_delivery: bool = False
    # An object lying in a container (a spoon in a pot) is gripped near the end of its long axis that
    # reaches furthest out past the container's wall, this far in from that end, when that end sticks
    # out at least this far: at its centre the jaws straddle the rim, and it jams them open.
    grasp_outside_container_m: Optional[float] = None
    # Carry into, release in and retreat from a container with the open jaws along its longer side:
    # across a bin's short side the open fingers span as much as the bin does and shove it.
    container_jaw_along_long_side: bool = False
    # Carry an object longer than a container's shorter side into it lengthwise, with the jaws across
    # the container: set in crosswise it bridges both rims, and a regrasp there closes on the rim
    # (RobotArena's 14 cm spoon over its 10 cm-wide pot).
    long_object_along_container: bool = False
    # Keep a container's goal this far in from its walls along the jaw axis, so the open jaws that
    # set an object down stay inside it: an off-centre slot in RoboLab's 10 cm-wide bin put a finger
    # on the wall, which shoved the bin and wedged the gripper under its lip. None disables it.
    container_jaw_margin_m: Optional[float] = None
    # Half the open jaws' span. When a finger would come down inside a neighbouring object's footprint
    # (RobotArena's spoon lying against the pot wall), slide the grasp point along the object's long
    # axis until both fingers are clear. None disables it.
    grasp_clear_half_span_m: Optional[float] = None
    # See MotionParams.approach_timeout_max_xy_m.
    approach_timeout_max_xy_m: Optional[float] = None
    # See MotionParams.max_container_insertion_m.
    max_container_insertion_m: Optional[float] = None
    # Identical movers (carrot#0, carrot#1) are credited by count, not by name: the goal naming
    # carrot#1 is met by any two carrots on the plate. When the named one already rests there, move
    # another instance instead of lifting it back off, which undoes the goal it was chasing.
    retarget_identical: bool = False
    # Gap from the reported end-effector frame to the finger pads. The policy reasons in pad space,
    # so a suite whose kinematics report the flange (RoboLab: panda_link8) declares it here.
    tool_offset_m: float = 0.0
    # "structured" -> drive from the task's privileged goals.
    # "subtask"    -> follow an orchestrator's per-step subtask, grounded on ground-truth poses,
    #                 which isolates the orchestrator's choices from action-expert error.
    mode: str = "structured"
    # Subtask mode source: "instruction_arg" is the runner's per-step instruction (a learned HLC),
    # "env_instruction" is the env's own oracle decomposition.
    subtask_source: str = "instruction_arg"

    def __post_init__(self):
        super().__post_init__()
        if self.mode not in ("structured", "subtask"):
            raise ValueError(f"mode must be 'structured' or 'subtask', got {self.mode!r}")
        if self.subtask_source not in ("instruction_arg", "env_instruction"):
            raise ValueError(
                f"subtask_source must be 'instruction_arg' or 'env_instruction', got {self.subtask_source!r}"
            )


class ScriptedPolicy(Configurable[ScriptedPolicyConfig]):
    """Privileged-state pick-and-place policy. Per-env state keyed by list index."""

    def __init__(self, config: ScriptedPolicyConfig) -> None:
        super().__init__(config)
        self._params = MotionParams(
            max_translation_per_step_m=config.max_translation_per_step_m,
            transit_translation_per_step_m=config.transit_translation_per_step_m,
            approach_height_m=config.approach_height_m,
            grasp_z_offset_m=config.grasp_z_offset_m,
            grasp_from_top=config.grasp_from_top,
            rim_grasp_min_span_m=config.rim_grasp_min_span_m,
            rim_inset_m=config.rim_inset_m,
            rim_along_jaw=config.rim_along_jaw,
            close_above_m=config.close_above_m,
            rim_z_offset_m=config.rim_z_offset_m,
            close_descent_step_m=config.close_descent_step_m,
            lift_height_m=config.lift_height_m,
            carry_height_m=config.carry_height_m,
            place_clearance_m=config.place_clearance_m,
            xy_tol_m=config.xy_tol_m,
            z_tol_m=config.z_tol_m,
            grasp_close_steps=config.grasp_close_steps,
            max_grasp_steps=config.max_grasp_steps,
            release_steps=config.release_steps,
            shed_pulses=config.shed_pulses,
            wide_grasp_depth_m=config.wide_grasp_depth_m,
            wide_object_min_m=config.wide_object_min_m,
            wide_object_min_height_m=config.wide_object_min_height_m,
            max_phase_steps=config.max_phase_steps,
            reach_stall_steps=config.reach_stall_steps,
            grasp_confirm_steps=config.grasp_confirm_steps,
            grasp_lost_steps=config.grasp_lost_steps,
            jaw_stall_steps=config.jaw_stall_steps,
            approach_timeout_max_xy_m=config.approach_timeout_max_xy_m,
            max_container_insertion_m=config.max_container_insertion_m,
            descent_high_tol_m=config.descent_high_tol_m,
        )
        self._states: Dict[int, EnvState] = {}
        # Condition each env is executing; a fresh EnvState starts when it advances.
        self._active_cond: Dict[int, int] = {}
        # subtask mode: the grounded subtask key each env is executing.
        self._active_subtask: Dict[int, tuple] = {}
        # Reset EEF orientation; the home pose is already grasp-capable on both suites.
        self._home_rot: Dict[int, np.ndarray] = {}
        self._home_pos: Dict[int, np.ndarray] = {}
        # Held for the whole chain: the resting AABB it was read from stops meaning anything once
        # the object is in the gripper.
        self._grasp_rot: Dict[int, np.ndarray] = {}
        # How many times the chain for the current goal has been restarted. An object's AABB says
        # which way it lies only when it lies along an axis; a spoon at 45 degrees has a nearly
        # square footprint and the reading is a coin flip, so each retry turns the wrist by a further
        # quarter turn and the alternative grip gets tried rather than the same failure repeated.
        self._attempts: Dict[int, int] = {}
        # Index into grasp_orientations_deg each env is grasping with for its current goal.
        self._orientation: Dict[int, int] = {}
        self._walled_envs: Set[int] = set()
        self._wall_free_envs: Set[int] = set()
        self._home_joints: Dict[int, np.ndarray] = {}
        # Steps spent so far on a joint-space rehome, per env currently doing one.
        self._joint_rehoming: Dict[int, int] = {}
        self._ordered: Set[int] = set()
        self._subtasks: List[Optional[str]] = []

    def reset(self) -> None:
        self._states = {}
        self._active_cond = {}
        self._active_subtask = {}
        self._home_rot = {}
        self._home_pos = {}
        self._grasp_rot = {}
        self._attempts = {}
        self._orientation = {}
        self._walled_envs = set()
        self._wall_free_envs = set()
        self._home_joints = {}
        self._joint_rehoming = {}
        self._ordered = set()
        self._subtasks = []

    @property
    def current_subtasks(self) -> List[Optional[str]]:
        """Per-env phase:object the policy executed at the last ``predict_actions``,
        recorded by the runner into the viewer's subtask overlay."""
        return self._subtasks

    def predict_actions(
        self,
        observations: List[RobotObservation],
        instructions: List[str],
        env_indices: Optional[List[int]] = None,
    ) -> List[RobotAction]:
        """``env_indices`` names which env each observation belongs to.

        The runner drops an env from the batch as soon as its episode ends, so after the first
        termination a list position no longer identifies an env. Per-env state is keyed by the real
        index, or by position when the caller does not supply one (an unbatched runner).
        """
        actions: List[RobotAction] = []
        self._subtasks = []
        indices = list(env_indices) if env_indices is not None else list(range(len(observations)))
        for position, observation in enumerate(observations):
            env_idx = indices[position]
            if not isinstance(observation.state, ManipulatorState):
                raise NotImplementedError(
                    f"ScriptedPolicy scripts a single manipulator; got " f"{type(observation.state).__name__}"
                )
            priv = from_metadata(observation.metadata)
            oracle = observation.metadata.get("oracle")
            if self.config.respect_order and isinstance(oracle, dict) and oracle.get("ordered"):
                self._ordered.add(env_idx)
            if priv is None:
                if self.config.require_privileged:
                    raise RuntimeError(
                        "ScriptedPolicy requires privileged state in observation.metadata"
                        f"['privileged'] but none was found (env {env_idx}). The simulator"
                        " server must be configured to emit it."
                    )
                actions.append(self._hold(observation, env_idx))
                self._subtasks.append(None)
                continue
            if self.config.mode == "subtask":
                if self.config.subtask_source == "env_instruction":
                    instruction = observation.metadata.get("instruction", "")
                else:
                    instruction = instructions[position] if position < len(instructions) else ""
                actions.append(self._step_subtask(observation, priv, str(instruction), env_idx))
            else:
                actions.append(self._step_env(observation, priv, env_idx))
        return actions

    def _step_subtask(
        self, observation: RobotObservation, priv: PrivilegedState, instruction: str, env_idx: int
    ) -> RobotAction:
        """Execute the orchestrator's current subtask, grounded onto ground-truth poses."""
        if env_idx not in self._home_rot:
            self._home_rot[env_idx] = observation.state.eef_pose[:3, :3].astype(np.float32).copy()

        cond = ground_subtask(instruction, priv)
        if cond is None:
            # Unparseable subtask -> hold, surfacing the orchestrator's failure as no progress.
            self._subtasks.append(f"ungrounded:{instruction}")
            return self._hold(observation, env_idx)

        eef_m = self._pads(observation)
        # Restart on a subtask switch, but start at CARRY when the object is already held so the
        # chain does not re-open the gripper and drop it.
        key = (cond.predicate, cond.target.name, instruction)
        if self._active_subtask.get(env_idx) != key:
            self._active_subtask[env_idx] = key
            self._grasp_rot.pop(env_idx, None)
            state = EnvState()
            if cond.goal_point_m is not None and cond.grasped:
                state.phase = Phase.CARRY
                state.grasp_eef_m = eef_m.copy()
            self._states[env_idx] = state
        state = self._states.setdefault(env_idx, EnvState())

        gripper_open = float(np.asarray(observation.state.gripper_state).reshape(-1)[0]) > 0.5
        target_m, want_open = step(state, cond, eef_m, gripper_open, self._params)

        self._subtasks.append(f"{state.phase.value}:{cond.target.display_name or cond.target.name}")
        return ManipulatorAction(
            arm_action=EndEffectorPoseAction(
                translation=self._command(target_m, env_idx, self._grasp_rotation(env_idx, cond)),
                rotation=self._grasp_rotation(env_idx, cond),
            ),
            gripper_action=1.0 if want_open else 0.0,
        )

    def _step_env(  # noqa: PLR0915 -- the per-env pipeline: target rewrites, retries, then one step
        self, observation: RobotObservation, priv: PrivilegedState, env_idx: int
    ) -> RobotAction:
        if env_idx not in self._home_rot:
            self._home_rot[env_idx] = observation.state.eef_pose[:3, :3].astype(np.float32).copy()
            self._home_pos[env_idx] = self._pads(observation).copy()

        pads_y = float(self._pads(observation)[1])
        if pads_y > self.config.wall_free_y_m:
            self._wall_free_envs.add(env_idx)
            self._walled_envs.discard(env_idx)

        selected = self._select_condition(priv, env_idx)
        if selected is None:
            # Every condition satisfied -> hold open, mark done.
            self._subtasks.append("done")
            return self._hold(observation, env_idx)
        cond_idx, cond = selected
        if self.config.retarget_identical and not cond.grasped:
            cond = self._retargeted(priv, cond)
        if self.config.grasp_body_offset:
            cond = self._body_shifted(env_idx, cond)
        if self.config.grasp_outside_container_m is not None and not cond.grasped:
            cond = self._outside_shifted(priv, cond)
        if self.config.grasp_clear_half_span_m is not None and not cond.grasped:
            cond = self._clear_shifted(priv, cond, env_idx)
        if env_idx in self._walled_envs:
            cond = self._walled(cond)

        # Restart when the active condition advances, to run the chain for the next object.
        if self._active_cond.get(env_idx) != cond_idx:
            self._active_cond[env_idx] = cond_idx
            self._states[env_idx] = EnvState()
            self._grasp_rot.pop(env_idx, None)
            self._attempts[env_idx] = 0
            self._orientation[env_idx] = 0
        state = self._states.setdefault(env_idx, EnvState())
        # Out of reach near the wall: the wall stands in this env (see wall_max_y_m).
        if state.phase == Phase.DONE and state.unreachable and self.config.wall_max_y_m is not None:
            stopped_at_wall = self.config.wall_hover_max_y_m <= pads_y < self.config.wall_stop_max_y_m
            if self._wall_side(cond) and stopped_at_wall and env_idx not in self._wall_free_envs:
                self._walled_envs.add(env_idx)
        # Out of reach at this wrist orientation: try the next one; a full cycle counts as an attempt.
        failed = state.phase == Phase.DONE and cond.goal_point_m is not None and not state.unreachable
        if failed and self.config.rotate_on_failure and len(self.config.grasp_orientations_deg or []) > 1:
            state.unreachable = True
        if state.phase == Phase.DONE and state.unreachable:
            count = len(self.config.grasp_orientations_deg or [None])
            index = self._orientation.get(env_idx, 0) + 1
            if index >= count:
                index = 0
                self._attempts[env_idx] = self._attempts.get(env_idx, 0) + 1
            self._orientation[env_idx] = index
            state = self._states[env_idx] = EnvState(phase=Phase.REHOME, home_m=self._home_pos.get(env_idx))
            self._grasp_rot.pop(env_idx, None)
        # Chain finished with the goal unsatisfied: run it again rather than idling out the episode,
        # since the probe must spend its budget before calling a task unreachable.
        if state.phase == Phase.DONE and cond.goal_point_m is not None:
            state = self._states[env_idx] = EnvState()
            self._grasp_rot.pop(env_idx, None)
            self._attempts[env_idx] = self._attempts.get(env_idx, 0) + 1

        if self.config.joint_rehome_drift_rad is not None and observation.state.joints is not None:
            rehome = self._joint_rehome(observation, env_idx, state, cond)
            if rehome is not None:
                self._subtasks.append(f"rehome:{cond.target.name}")
                return rehome

        eef_m = self._pads(observation)
        openness = float(np.asarray(observation.state.gripper_state).reshape(-1)[0])
        limit = None
        if self.config.reach_center_m is not None or env_idx in self._walled_envs:
            rotation = self._grasp_rotation(env_idx, cond)
            limit = lambda point: self._bounded(point, env_idx, rotation)  # noqa: E731
        if self.config.container_jaw_margin_m is not None and cond.relation == "in":
            cond = self._jaws_inside(priv, cond, env_idx)
        cond = self._goal_in_reach(priv, cond)
        target_m, want_open = step(
            state, cond, eef_m, openness > 0.5, self._params, openness=openness, limit=limit
        )

        self._subtasks.append(f"{state.phase.value}:{cond.target.name}")
        delivering = state.phase in (Phase.CARRY, Phase.PLACE, Phase.RELEASE, Phase.RETREAT)
        rotation = self._held_rotation(priv, cond, env_idx, delivering)
        return ManipulatorAction(
            arm_action=EndEffectorPoseAction(
                translation=self._command(target_m, env_idx, rotation),
                rotation=rotation,
            ),
            gripper_action=1.0 if want_open else 0.0,
        )

    def _retargeted(self, priv: PrivilegedState, cond: PrivilegedCondition) -> PrivilegedCondition:
        """``cond`` with its target swapped for an instance not yet on the destination (see retarget_identical)."""
        dest = next((o for o in priv.objects if o.name == cond.destination), None)
        if dest is None or cond.satisfied:
            return cond

        def resting(obj: PrivilegedObject) -> bool:
            xy = np.asarray(obj.center_m, dtype=np.float64)[:2]
            inside = bool(
                np.all((xy >= np.asarray(dest.aabb_min_m)[:2]) & (xy <= np.asarray(dest.aabb_max_m)[:2]))
            )
            return inside and float(obj.center_m[2]) > float(dest.center_m[2])

        if not resting(cond.target):
            return cond
        model = cond.target.name.split("#")[0]
        others = [o for o in priv.objects if o.name != cond.target.name and o.name.split("#")[0] == model]
        floor = self.config.fallen_z_m if self.config.fallen_z_m is not None else -np.inf
        spare = next((o for o in others if not resting(o) and float(o.center_m[2]) > floor), None)
        if spare is None:
            return cond
        return dataclasses.replace(cond, target=spare, grasped=spare.grasped)

    def _outside_shifted(self, priv: PrivilegedState, cond: PrivilegedCondition) -> PrivilegedCondition:
        """``cond`` with its target moved to its end outside a container (see grasp_outside_container_m)."""
        target = cond.target
        center = np.asarray(target.center_m, dtype=np.float64)
        for box in priv.objects:
            if box.name == target.name or rim_direction(box, self._params) is None:
                continue
            lo, hi = np.asarray(box.aabb_min_m)[:2], np.asarray(box.aabb_max_m)[:2]
            if not bool(np.all((center[:2] >= lo) & (center[:2] <= hi))):
                continue
            half = target.half_extent_m
            axis = 0 if half[0] >= half[1] else 1
            reach = max(float(half[axis]) - self.config.grasp_outside_container_m, 0.0)
            box_center, box_half = 0.5 * (lo[axis] + hi[axis]), 0.5 * (hi[axis] - lo[axis])
            # How far each end sticks out past the container's wall along that axis.
            ends = [center[axis] + side * float(half[axis]) for side in (1.0, -1.0)]
            outside = [abs(end - box_center) - box_half for end in ends]
            side = 1.0 if outside[0] >= outside[1] else -1.0
            # A short object barely out past the wall is better held at its centre: gripped near its
            # tip it slips (the drumstick in RobotArena's pot).
            if max(outside) < self.config.grasp_outside_container_m:
                return cond
            shift = np.zeros(3)
            shift[axis] = side * reach
            moved = dataclasses.replace(
                target,
                center_m=target.center_m + shift,
                aabb_min_m=target.aabb_min_m + shift,
                aabb_max_m=target.aabb_max_m + shift,
            )
            return dataclasses.replace(cond, target=moved)
        return cond

    def _clear_shifted(
        self, priv: PrivilegedState, cond: PrivilegedCondition, env_idx: int
    ) -> PrivilegedCondition:
        """``cond`` with its grasp point slid along the target's long axis until neither finger lands on a neighbour."""
        target = cond.target
        if target.long_axis_yaw_rad is None or env_idx not in self._home_rot:
            return cond
        center = np.asarray(target.center_m, dtype=np.float64)
        axis = np.array([math.cos(target.long_axis_yaw_rad), math.sin(target.long_axis_yaw_rad)])
        jaw = np.array([-axis[1], axis[0]])
        half_length = 0.5 * float(np.max(target.aabb_max_m[:2] - target.aabb_min_m[:2]))
        others = [
            (np.asarray(o.aabb_min_m)[:2], np.asarray(o.aabb_max_m)[:2])
            for o in priv.objects
            if o.name != target.name and float(o.top_z_m) > float(center[2])
        ]

        def clear(point: np.ndarray) -> bool:
            for side in (1.0, -1.0):
                finger = point + side * self.config.grasp_clear_half_span_m * jaw
                if any(
                    bool(np.all((finger >= lo - _CLEAR_MARGIN_M) & (finger <= hi + _CLEAR_MARGIN_M)))
                    for lo, hi in others
                ):
                    return False
            return True

        if clear(center[:2]):
            return cond
        reach = max(half_length - _CLEAR_END_M, 0.0)
        for distance in np.arange(0.01, reach + 1e-9, 0.01):
            for sign in (1.0, -1.0):
                shift = sign * distance * axis
                if clear(center[:2] + shift):
                    moved = np.array([shift[0], shift[1], 0.0], dtype=np.float32)
                    return dataclasses.replace(
                        cond,
                        target=dataclasses.replace(
                            target,
                            center_m=target.center_m + moved,
                            aabb_min_m=target.aabb_min_m + moved,
                            aabb_max_m=target.aabb_max_m + moved,
                        ),
                    )
        return cond

    def _body_shifted(self, env_idx: int, cond: PrivilegedCondition) -> PrivilegedCondition:
        """``cond`` with its target moved to this attempt's grasp point (see grasp_body_offset)."""
        side = (1.0, -1.0, 0.0)[self._attempts.get(env_idx, 0) % 3]
        target = cond.target
        half = target.half_extent_m
        axis = 0 if half[0] < half[1] else 1
        shift = np.zeros(3)
        shift[axis] = side * max(float(half[axis]) - float(half[2]), 0.0)
        moved = dataclasses.replace(
            target,
            center_m=target.center_m + shift,
            aabb_min_m=target.aabb_min_m + shift,
            aabb_max_m=target.aabb_max_m + shift,
        )
        return dataclasses.replace(cond, target=moved)

    def _wall_side(self, cond: PrivilegedCondition, beyond_m: Optional[float] = None) -> bool:
        """Whether ``cond`` picks or places past ``beyond_m`` (default wall_turn_y_m; see wall_max_y_m)."""
        goal_y = -1.0 if cond.goal_point_m is None else float(cond.goal_point_m[1])
        limit = self.config.wall_turn_y_m if beyond_m is None else beyond_m
        return max(float(cond.target.center_m[1]), goal_y) > limit

    def _walled(self, cond: PrivilegedCondition) -> PrivilegedCondition:
        """``cond`` with its grasp and goal points pulled back from the wall (see wall_max_y_m)."""
        limit = self.config.wall_max_y_m
        target = cond.target
        shift = np.array([0.0, min(0.0, limit - float(target.center_m[1])), 0.0])
        if shift[1] < 0.0:
            target = dataclasses.replace(
                target,
                center_m=target.center_m + shift,
                aabb_min_m=target.aabb_min_m + shift,
                aabb_max_m=target.aabb_max_m + shift,
            )
        goal = cond.goal_point_m
        if goal is not None and float(goal[1]) > limit:
            goal = np.asarray(goal, dtype=np.float64).copy()
            goal[1] = limit
        return dataclasses.replace(cond, target=target, goal_point_m=goal)

    def _pads(self, observation: RobotObservation) -> np.ndarray:
        """Where the finger pads are, which is what every phase target is expressed against."""
        pads = observation.state.eef_pose[:3, 3].astype(np.float64)
        pads[2] -= self.config.tool_offset_m
        return pads

    def _command(self, target_m: np.ndarray, env_idx: int, rotation: np.ndarray) -> np.ndarray:
        """Back out of pad space into the end-effector frame the arm is commanded in, inside reach."""
        command = self._bounded(target_m, env_idx, rotation)
        command[2] += self.config.tool_offset_m
        return command.astype(np.float32)

    def _bounded(self, point_m: np.ndarray, env_idx: int, rotation: np.ndarray) -> np.ndarray:
        """Pad-space ``point_m`` pulled inside the arm's reach and clear of the env's wall, if any."""
        point = np.asarray(point_m, dtype=np.float64).copy()
        if self.config.reach_center_m is not None:
            tool = np.array([0.0, 0.0, self.config.tool_offset_m])
            point = self._within_reach(point + tool, env_idx, rotation) - tool
        if env_idx in self._walled_envs:
            # Linear between the two measured heights, table level and hover.
            rise = min(max((float(point[2]) - _WALL_LOW_Z_M) / (_WALL_HOVER_Z_M - _WALL_LOW_Z_M), 0.0), 1.0)
            low, high = self.config.wall_max_y_m, self.config.wall_hover_max_y_m
            point[1] = min(float(point[1]), low + rise * (high - low))
        return point

    def _within_reach(self, point: np.ndarray, env_idx: int, rotation: np.ndarray) -> np.ndarray:
        """``point`` moved so the wrist behind it lies inside the reach sphere (see the config)."""
        home = self._home_rot[env_idx].astype(np.float64)
        # The gripper axis that points straight down at the home orientation, in the command's rotation.
        approach = np.asarray(rotation, dtype=np.float64) @ (home.T @ np.array([0.0, 0.0, -1.0]))
        offset = approach * self.config.wrist_offset_m
        center = np.asarray(self.config.reach_center_m, dtype=np.float64)
        radius = self.config.reach_radius_m
        wrist = point - offset - center
        if float(np.linalg.norm(wrist)) <= radius:
            return point
        planar = float(np.linalg.norm(wrist[:2]))
        vertical = float(wrist[2])
        if self.config.reach_keep_height and abs(vertical) < radius:
            # The place descent gets the reach back lower down.
            wrist[:2] *= 0.999 * math.sqrt(radius**2 - vertical**2) / planar
        elif planar >= radius:
            wrist[:2] *= 0.999 * radius / planar
            wrist[2] = 0.0
        else:
            wrist[2] = math.copysign(math.sqrt(radius**2 - planar**2), vertical)
        return center + wrist + offset

    def _grasp_rotation(self, env_idx: int, cond: PrivilegedCondition) -> np.ndarray:
        """The arm's own orientation, yawed so the jaws close across the object's short axis.

        A jaw closing along an object's length cannot span it and sweeps it aside instead. The AABB
        names the long axis only when the object lies along one, so each retry adds a further
        quarter turn to try the alternative grip.
        """
        cached = self._grasp_rot.get(env_idx)
        if cached is not None:
            return cached
        home = self._home_rot[env_idx]
        target = cond.target
        rim = rim_direction(target, self._params)
        if rim is not None:
            # Jaws lie along base Y at the home orientation; turn them to close across the wall.
            delta = _jaw_yaw(math.atan2(float(rim[1]), float(rim[0])) - math.pi / 2)
            cos, sin = math.cos(delta), math.sin(delta)
            yaw = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            roll = _rotation_about(
                np.cross([0.0, 0.0, 1.0], [float(rim[0]), float(rim[1]), 0.0]),
                -math.radians(self.config.rim_roll_deg),
            )
            self._grasp_rot[env_idx] = (self._orientation_offset(env_idx) @ roll @ yaw @ home).astype(
                np.float32
            )
            return self._grasp_rot[env_idx]
        delta = 0.0
        near_wall = env_idx in self._walled_envs and self._wall_side(cond, self.config.wall_level_max_y_m)
        extent = (target.aabb_max_m - target.aabb_min_m)[:2]
        # Near the wall, jaws open across y reach the wall: an object lying along y is gripped with
        # the jaws turned along x, one lying along x with them across y but rolled (see wall_roll_deg).
        turn = near_wall and float(extent[1]) > float(extent[0])
        roll = near_wall and not turn
        if turn:
            delta += math.pi / 2
        if not (near_wall or self.config.align_grasp_yaw or self.config.grasp_orientations_deg):
            return home
        if self.config.align_grasp_yaw:
            extent = (target.aabb_max_m - target.aabb_min_m)[:2]
            # Jaws lie along base Y at the home orientation, so an object longer in Y needs a quarter turn.
            retries = self._attempts.get(env_idx, 0)
            if self.config.grasp_body_offset:
                retries //= 3
            if target.long_axis_yaw_rad is not None:
                # The server says which way the object points: square the jaws (along base Y at home)
                # to it. The AABB below only names an axis-aligned object's long side, so a diagonal
                # spoon gets a finger landed on its handle.
                delta += float(target.long_axis_yaw_rad) + (retries % 2) * math.pi / 2
            else:
                quarters = (1 if extent[1] > extent[0] else 0) + retries
                delta += (quarters % 4) * math.pi / 2
        delta = _jaw_yaw(delta)
        cos, sin = math.cos(delta), math.sin(delta)
        yaw = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        if near_wall:
            # The measured wall poses alone: stacked on the retry offsets they leave the IK's reach.
            lean = _rotation_about(
                np.array([1.0, 0.0, 0.0]), math.radians(self.config.wall_roll_deg) if roll else 0.0
            )
            self._grasp_rot[env_idx] = (lean @ yaw @ home).astype(np.float32)
        else:
            self._grasp_rot[env_idx] = (self._orientation_offset(env_idx) @ yaw @ home).astype(np.float32)
        return self._grasp_rot[env_idx]

    def _jaws_inside(
        self, priv: PrivilegedState, cond: PrivilegedCondition, env_idx: int
    ) -> PrivilegedCondition:
        """``cond`` with its goal pulled in from the destination's walls along the jaw axis (see container_jaw_margin_m)."""
        dest = next((o for o in priv.objects if o.name == cond.destination), None)
        if dest is None or cond.goal_point_m is None:
            return cond
        rotation = self._held_rotation(priv, cond, env_idx, delivering=True)
        # Jaws lie along base Y at the home orientation.
        jaw = np.abs((rotation @ self._home_rot[env_idx].T) @ np.array([0.0, 1.0, 0.0]))[:2]
        lo = np.asarray(dest.aabb_min_m, dtype=np.float64)[:2] + self.config.container_jaw_margin_m * jaw
        hi = np.asarray(dest.aabb_max_m, dtype=np.float64)[:2] - self.config.container_jaw_margin_m * jaw
        center = np.asarray(dest.center_m, dtype=np.float64)[:2]
        goal = np.asarray(cond.goal_point_m, dtype=np.float64).copy()
        goal[:2] = np.where(lo <= hi, np.clip(goal[:2], lo, hi), center)
        return dataclasses.replace(cond, goal_point_m=goal.astype(np.float32))

    def _joint_rehome(
        self, observation: RobotObservation, env_idx: int, state: EnvState, cond: PrivilegedCondition
    ) -> Optional[RobotAction]:
        """The home joint command while a joint-space rehome runs (see joint_rehome_drift_rad), else None."""
        joints = np.asarray(observation.state.joints, dtype=np.float64).reshape(-1)[:7]
        home = self._home_joints.setdefault(env_idx, joints.copy())
        drift = np.abs(joints - home)
        starting = state.phase == Phase.APPROACH and state.phase_steps == 0 and not cond.grasped
        drifted = drift[[0, 2, 4]].max() > self.config.joint_rehome_drift_rad
        if env_idx not in self._joint_rehoming and starting and drifted:
            self._joint_rehoming[env_idx] = 0
        if env_idx not in self._joint_rehoming:
            return None
        self._joint_rehoming[env_idx] += 1
        if drift.max() < _JOINT_HOME_TOL_RAD or self._joint_rehoming[env_idx] > _JOINT_HOME_MAX_STEPS:
            del self._joint_rehoming[env_idx]
            return None
        return ManipulatorAction(
            arm_action=JointAnglesAction(angles=home.astype(np.float32)), gripper_action=1.0
        )

    def _goal_in_reach(self, priv: PrivilegedState, cond: PrivilegedCondition) -> PrivilegedCondition:
        """``cond`` with its goal pulled toward the base to within max_reach_radius_m, inside its destination."""
        dest = next((o for o in priv.objects if o.name == cond.destination), None)
        if self.config.max_reach_radius_m is None or dest is None or cond.goal_point_m is None:
            return cond
        goal = np.asarray(cond.goal_point_m, dtype=np.float64).copy()
        distance = float(np.hypot(*goal[:2]))
        if distance <= self.config.max_reach_radius_m:
            return cond
        goal[:2] *= self.config.max_reach_radius_m / distance
        # The object still has to land on the destination: keep its centre inside the footprint.
        half = np.asarray(cond.target.half_extent_m, dtype=np.float64)[:2]
        lo = np.asarray(dest.aabb_min_m, dtype=np.float64)[:2] + half
        hi = np.asarray(dest.aabb_max_m, dtype=np.float64)[:2] - half
        center = np.asarray(dest.center_m, dtype=np.float64)[:2]
        goal[:2] = np.where(lo <= hi, np.clip(goal[:2], lo, hi), center)
        return dataclasses.replace(cond, goal_point_m=goal.astype(np.float32))

    def _held_rotation(
        self, priv: PrivilegedState, cond: PrivilegedCondition, env_idx: int, delivering: bool
    ) -> np.ndarray:
        """The grasp rotation, turned while delivering into a container (see long_object_along_container)."""
        rotation = self._grasp_rotation(env_idx, cond)
        if cond.relation != "in" or not delivering:
            return rotation
        dest = next((o for o in priv.objects if o.name == cond.destination), None)
        # An object longer than the container is wide goes in lengthwise (jaws across the container);
        # anything else may take the jaws along the container's longer side.
        long_object = dest is not None and max((cond.target.aabb_max_m - cond.target.aabb_min_m)[:2]) > min(
            (dest.aabb_max_m - dest.aabb_min_m)[:2]
        )
        if self.config.long_object_along_container and long_object:
            return self._along_container(priv, cond, env_idx, rotation, jaw_along_long=False)
        if self.config.container_jaw_along_long_side:
            return self._along_container(priv, cond, env_idx, rotation, jaw_along_long=True)
        return rotation

    def _along_container(
        self,
        priv: PrivilegedState,
        cond: PrivilegedCondition,
        env_idx: int,
        rotation: np.ndarray,
        jaw_along_long: bool,
    ) -> np.ndarray:
        """``rotation`` turned about vertical so the jaw axis lies along the destination's longer side, or its shorter one."""
        dest = next((o for o in priv.objects if o.name == cond.destination), None)
        if dest is None:
            return rotation
        extent = (dest.aabb_max_m - dest.aabb_min_m)[:2]
        # Jaws lie along base Y at the home orientation.
        jaw = (rotation @ self._home_rot[env_idx].T) @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
        want = 0.0 if (extent[0] >= extent[1]) == jaw_along_long else math.pi / 2
        delta = _jaw_yaw(want - math.atan2(float(jaw[1]), float(jaw[0])))
        cos, sin = math.cos(delta), math.sin(delta)
        yaw = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        return (yaw @ rotation).astype(np.float32)

    def _orientation_offset(self, env_idx: int) -> np.ndarray:
        """Base-frame rotation of the env's current entry in ``grasp_orientations_deg``."""
        options = self.config.grasp_orientations_deg
        if not options:
            return np.eye(3, dtype=np.float32)
        yaw_deg, pitch_deg = options[self._orientation.get(env_idx, 0) % len(options)]
        cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        return (rz @ ry).astype(np.float32)

    def _select_condition(
        self, priv: PrivilegedState, env_idx: int
    ) -> Optional[Tuple[int, PrivilegedCondition]]:
        """The first unsatisfied condition in declared order, sticking with one already under way.

        A delivered object is jostled by the next arrival and its goal flag flickers off, so
        re-reading "first unsatisfied" every step would yank the policy back onto finished goals.
        Returns ``(index, condition)``, or ``None`` when every condition is satisfied.
        """
        count = len(priv.conditions)
        active = self._active_cond.get(env_idx)
        state = self._states.get(env_idx)
        delivering = (
            active is not None
            and active < count
            and state is not None
            and state.phase in (Phase.CARRY, Phase.PLACE, Phase.RELEASE, Phase.RETREAT)
        )
        if delivering and self.config.finish_credited_delivery:
            destination = priv.conditions[active].destination
            occupied = any(
                c.satisfied and c.destination == destination
                for i, c in enumerate(priv.conditions)
                if i != active
            )
            # The task's own relation says whether the destination is a container: size does not
            # (a scaled-up cloth stands almost as tall as the pot).
            container = priv.conditions[active].relation == "in"
            # Lowering onto what already sits there knocks it off, and one lowered into a pot lodges
            # against its wall where the next leg cannot get it out: the drop is the lesser harm.
            if not occupied and not container:
                return active, priv.conditions[active]
        if delivering and self.config.climb_out_after_delivery:
            if state.phase in (Phase.CARRY, Phase.PLACE) and priv.conditions[active].satisfied:
                state.phase, state.phase_steps = Phase.RETREAT, 0
            if state.phase in (Phase.RELEASE, Phase.RETREAT):
                return active, priv.conditions[active]

        def open_goal(index: int) -> bool:
            cond = priv.conditions[index]
            fallen = (
                self.config.fallen_z_m is not None and float(cond.target.center_m[2]) < self.config.fallen_z_m
            )
            radius = self.config.max_reach_radius_m
            far = (
                radius is not None
                and not cond.grasped
                and float(np.hypot(*cond.target.center_m[:2])) > radius
            )
            return not cond.satisfied and not fallen and not far

        if env_idx in self._ordered:
            if active is not None and active < count and not priv.conditions[active].satisfied:
                return active, priv.conditions[active]
            start = 0 if active is None else active + 1
            index = next((i for i in range(start, count) if not priv.conditions[i].satisfied), None)
            if index is None:
                index = next((i for i in range(count) if not priv.conditions[i].satisfied), None)
            return None if index is None else (index, priv.conditions[index])
        if (
            active is not None
            and active < count
            and open_goal(active)
            and self._attempts.get(env_idx, 0) < self.config.max_goal_attempts
        ):
            return active, priv.conditions[active]
        # Resume after the active goal, so giving up moves on instead of re-picking it.
        start = 0 if active is None else active + 1
        for offset in range(count):
            index = (start + offset) % count
            if open_goal(index):
                return index, priv.conditions[index]
        return None

    def _hold(self, observation: RobotObservation, env_idx: int) -> RobotAction:
        rot = self._home_rot.get(env_idx)
        if rot is None:
            rot = observation.state.eef_pose[:3, :3].astype(np.float32).copy()
            self._home_rot[env_idx] = rot
        return ManipulatorAction(
            arm_action=EndEffectorPoseAction(
                translation=observation.state.eef_pose[:3, 3].astype(np.float32),
                rotation=rot,
            ),
            gripper_action=1.0,
        )
