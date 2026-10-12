import dataclasses
import enum
from typing import Callable, Optional, Tuple

import numpy as np

from ladder.components.inference.api.core.privileged_state import PrivilegedCondition


class Phase(enum.Enum):
    REHOME = "rehome"
    APPROACH = "approach"
    DESCEND = "descend"
    CENTER = "center"
    GRASP = "grasp"
    LIFT = "lift"
    CARRY = "carry"
    PLACE = "place"
    RELEASE = "release"
    RETREAT = "retreat"
    DONE = "done"


# Predicates that only pick; ``reach`` never grasps and is handled separately.
_PICK_ONLY_PREDICATES = frozenset({"pickup", "lift", "grasp"})
_REACH_PREDICATES = frozenset({"reach"})

# The descent ends where the jaws close, so it needs a tighter arrival than other phases: closing
# a centimetre high catches an object's top edge and flicks it away.
_DESCENT_Z_TOL_M = 0.004
# Lowering the wrist costs ~2 cm of in-plane accuracy, so the descent waypoint carries the observed
# error forward: ask for target + (target - actual) in xy.
_DESCENT_XY_GAIN = 1.0
_DESCENT_XY_CORRECTION_CAP_M = 0.04
_SHED_DIST_M = 0.06  # a released object's centre this close to the pads is still in the gripper
_SHED_BELOW_M = 0.015
_SHED_PULSE_STEPS = 8
_DESCENT_STALL_EPS_M = 0.0005  # per-step drop below which the arm counts as no longer descending
_DESCENT_STALL_STEPS = 6


@dataclasses.dataclass
class MotionParams:
    """Tunable geometry / timing for the state machine (all distances in metres)."""

    max_translation_per_step_m: float = 0.025  # waypoint step cap (gentle motion)
    # Step cap in free space (approach, lift, carry, retreat); None keeps the gentle cap everywhere.
    # Only the moves that end in contact have to be slow, so a suite whose budget the gentle cap
    # exhausts in transit sets this.
    transit_translation_per_step_m: Optional[float] = None
    approach_height_m: float = 0.10  # hover height above target before descending
    grasp_z_offset_m: float = 0.0  # EEF z relative to target CENTER at grasp (to its TOP with grasp_from_top)
    # An object wider than the open jaws everywhere but near its top (a pomegranate) is gripped
    # this far below its top, where it narrows enough to fit between them. It counts as wide when its
    # narrower horizontal extent exceeds wide_object_min_m and it stands taller than
    # wide_object_min_height_m: height does not change with yaw, while a rotated object's axis-aligned
    # width grows by up to 40%, so a pumpkin at 45 degrees reads as wide as a pomegranate.
    # None disables.
    wide_grasp_depth_m: Optional[float] = None
    wide_object_min_m: float = 0.09
    wide_object_min_height_m: float = 0.0
    # Measure the grasp height from the object's top instead of its centre. The centre of a hollow
    # object (a pot, a bowl) is empty air and the centre of a flat one is at the table, so neither is
    # where the jaws should close; the top is the one height every shape has in common.
    grasp_from_top: bool = False
    # Objects whose narrower side is wider than this cannot be spanned by the open jaws (a pot, a large
    # bowl): grip their wall instead, on the side facing the robot base, ``rim_inset_m`` in from the
    # outer edge, with the jaws closing across the wall. 0 disables the rim grasp.
    rim_grasp_min_span_m: float = 0.0
    rim_inset_m: float = 0.012
    # Take the wall across the jaws' home closing axis (base Y), on the side nearer the robot's
    # centreline, so the wrist keeps its home yaw (the grip a learned policy uses on RobotArena's
    # bowl) instead of the wall facing the base with the wrist turned a quarter.
    rim_along_jaw: bool = False
    # Rim grasps close the jaws this far above the grasp point and keep descending while they close.
    # Opened full width over a thin wall, the inner finger lands on the bowl's floor and presses down
    # rather than across, which no two-finger grasp test accepts.
    close_above_m: float = 0.0
    rim_z_offset_m: Optional[float] = None  # grasp_z_offset_m for rim grasps; None keeps the shared one
    close_descent_step_m: float = 0.002  # descent speed while those jaws close, so they narrow first
    lift_height_m: float = 0.12  # how far to lift after grasping
    carry_height_m: float = 0.12  # hover height above the goal while carrying
    place_clearance_m: float = 0.03  # EEF z above goal_point when releasing
    xy_tol_m: float = 0.012  # in-plane arrival tolerance for phase advance
    z_tol_m: float = 0.015  # vertical arrival tolerance for phase advance
    grasp_close_steps: int = 3  # min steps holding closed before trusting grasp
    max_grasp_steps: int = 20  # give up waiting for the grasp signal after this
    release_steps: int = 3  # steps holding open to confirm release
    # A released object can stay wedged in the open gripper and ride up with it. While retreating
    # with it still at the pads, pulse the jaws closed and open this many times to shake it loose.
    # 0 disables.
    shed_pulses: int = 0
    # Phases advance on arrival or timeout, so a waypoint the PD settles just outside tolerance
    # cannot freeze the episode.
    max_phase_steps: int = 25
    # An approach that times out this far (in-plane) from its hover point does not descend there:
    # the pose counts as out of reach. Descending anyway put a Franka stuck at the edge of its reach
    # onto the plate beside the target, which it then picked up. None descends wherever it stopped.
    approach_timeout_max_xy_m: Optional[float] = None
    # How far below a container's rim (its release_z_m) the release may go. Set down at the floor of
    # RoboLab's 16 cm-deep bin at the edge of the Franka's reach, the wrist ended inside the bin and
    # the arm could not climb back out. None releases at the goal, however deep.
    max_container_insertion_m: Optional[float] = None
    # Steps without progress toward an approach/descent target after which the pose counts as out of
    # reach at the current wrist orientation, so the caller can try another; 0 disables the check.
    reach_stall_steps: int = 0
    # Consecutive steps the grasp signal must hold before lifting, and the consecutive steps without
    # it that abort a lift/carry/place: a finger brushing the object's top edge flickers the signal,
    # and without these the arm flies an empty gripper to the goal. 1 / 0 keep the old behaviour.
    grasp_confirm_steps: int = 1
    grasp_lost_steps: int = 0
    # Jaws that stop part-closed for this many steps hold something even when the simulator's grasp
    # test does not fire (a thin spoon handle), so the lift starts and the object following the
    # gripper confirms it; jaws that shut completely missed, and the chain ends at once rather than
    # waiting out max_grasp_steps. 0 disables both.
    jaw_stall_steps: int = 0
    # A descent that stalls this far above the grasp height did not reach the object at this wrist
    # orientation (the arm's reach, not the object, stopped it), so it counts as out of reach instead
    # of closing on the object's top edge. 0 disables the check.
    descent_high_tol_m: float = 0.0


@dataclasses.dataclass
class EnvState:
    """Per-env mutable execution state for the state machine."""

    phase: Phase = Phase.APPROACH
    phase_steps: int = 0  # steps spent in the current phase
    # EEF at grasp completion; LIFT/RETREAT work from it, not the now-attached object centre.
    grasp_eef_m: Optional[np.ndarray] = None
    # Descent progress: the EEF height last seen, and how many steps it has failed to drop.
    last_z_m: Optional[float] = None
    stalled_steps: int = 0
    # Frozen when the descent begins: the closing fingers nudge the object, and a live target would
    # chase what it is pushing. APPROACH still tracks the live pose.
    descent_target_m: Optional[np.ndarray] = None
    # Reach check (MotionParams.reach_stall_steps): closest the EEF has come to the phase target, and
    # for how many steps it has not got closer. ``unreachable`` ends the chain for another orientation.
    best_dist_m: float = float("inf")
    no_progress_steps: int = 0
    unreachable: bool = False
    # Steps spent shaking a wedged object loose on this retreat (see MotionParams.shed_pulses).
    shed_steps: int = 0
    # Where REHOME returns to (the reset EEF position): the arm's IK solves locally, so a new wrist
    # orientation is tried from the configuration the approach was measured from, not wherever the
    # last attempt stalled.
    home_m: Optional[np.ndarray] = None
    grasp_seen_steps: int = 0
    # Gripper minus object centre (xy) when the grasp formed: the carry and the release put the OBJECT
    # over the goal, so an off-centre grip (a rim grasp holds a pot by its wall) still lands it there.
    hold_offset_m: Optional[np.ndarray] = None
    grasp_missing_steps: int = 0
    # Gripper minus object centre (3D) when the grasp formed; the object keeping it is what "held"
    # means when the simulator's grasp test is silent (see MotionParams.jaw_stall_steps).
    hold_vec_m: Optional[np.ndarray] = None
    last_openness: Optional[float] = None
    jaw_still_steps: int = 0
    # Top of the object when the approach began: an object stuck to the open fingers rises with
    # them, and a hover measured from its live top would climb without end.
    approach_top_m: Optional[float] = None


_HELD_TOL_M = 0.015  # drift of the object from the grip it was taken with, beyond which it is not held
_JAW_SHUT = 0.05  # openness at or below which the jaws closed on nothing
_JAW_STILL_EPS = 0.01


def _held(cond: PrivilegedCondition, state: EnvState, eef_m: np.ndarray) -> bool:
    """The simulator's grasp test, or the object still hanging where it was gripped."""
    if cond.grasped:
        return True
    if state.hold_vec_m is None:
        return False
    drift = (eef_m - np.asarray(cond.target.center_m, dtype=np.float64)) - state.hold_vec_m
    return float(np.linalg.norm(drift)) <= _HELD_TOL_M


def _toward(eef_m: np.ndarray, target_m: np.ndarray, max_step_m: float) -> np.ndarray:
    """Absolute waypoint one clamped step from ``eef_m`` toward ``target_m``."""
    delta = target_m - eef_m
    dist = float(np.linalg.norm(delta))
    if dist <= max_step_m or dist == 0.0:
        return target_m.astype(np.float64).copy()
    return eef_m + delta * (max_step_m / dist)


def _reached(eef_m: np.ndarray, target_m: np.ndarray, xy_tol_m: float, z_tol_m: float) -> bool:
    delta = target_m - eef_m
    return bool(np.linalg.norm(delta[:2]) <= xy_tol_m and abs(delta[2]) <= z_tol_m)


def rim_direction(target, params: MotionParams) -> Optional[np.ndarray]:
    """Unit xy vector from the object's centre to the wall a rim grasp takes, or None when the jaws
    can span the object. The wall facing the robot base is the one the arm reaches first."""
    if params.rim_grasp_min_span_m <= 0.0:
        return None
    span = (
        np.asarray(target.aabb_max_m, dtype=np.float64) - np.asarray(target.aabb_min_m, dtype=np.float64)
    )[:2]
    if float(span.min()) <= params.rim_grasp_min_span_m:
        return None
    if params.rim_along_jaw:
        side = -1.0 if float(target.center_m[1]) >= 0.0 else 1.0
        return np.array([0.0, side])
    toward_base = -np.asarray(target.center_m, dtype=np.float64)[:2]
    norm = float(np.linalg.norm(toward_base))
    return toward_base / norm if norm > 1e-6 else np.array([-1.0, 0.0])


def _grasp_point(cond: PrivilegedCondition, params: MotionParams) -> np.ndarray:
    """EEF target for grasping: target center xy (its wall for a rim grasp), z = reference + offset."""
    point = cond.target.center_m.astype(np.float64).copy()
    rim = rim_direction(cond.target, params)
    if rim is not None:
        half = 0.5 * (np.asarray(cond.target.aabb_max_m) - np.asarray(cond.target.aabb_min_m))[:2]
        reach = abs(rim[0]) * half[0] + abs(rim[1]) * half[1]
        point[:2] += rim * max(reach - params.rim_inset_m, 0.0)
    reference = cond.target.top_z_m if params.grasp_from_top else cond.target.center_m[2]
    offset = params.grasp_z_offset_m
    if rim is not None and params.rim_z_offset_m is not None:
        offset = params.rim_z_offset_m
    point[2] = reference + offset
    extent = (np.asarray(cond.target.aabb_max_m) - np.asarray(cond.target.aabb_min_m))[:2]
    height = float(cond.target.aabb_max_m[2]) - float(cond.target.aabb_min_m[2])
    wide = float(extent.min()) > params.wide_object_min_m and height > params.wide_object_min_height_m
    if params.wide_grasp_depth_m is not None and wide:
        point[2] = float(cond.target.top_z_m) - params.wide_grasp_depth_m
    return point


def _held_over_goal(
    cond, state, height_m: float, clear_rim: bool = True, insertion_m: Optional[float] = None
) -> np.ndarray:
    """``_over_goal`` shifted by how the object hangs from the gripper (see EnvState.hold_offset_m)."""
    point = _over_goal(cond, height_m, clear_rim, insertion_m)
    if state.hold_offset_m is not None:
        point[:2] += state.hold_offset_m
    return point


def _over_goal(
    cond: PrivilegedCondition, height_m: float, clear_rim: bool = True, insertion_m: Optional[float] = None
) -> np.ndarray:
    """EEF waypoint ``height_m`` above the goal, optionally lifted clear of a walled destination.

    A container's goal point is at its FLOOR, so transit measured from it would drag the carried
    object through the rim: ``release_z_m`` raises the approach above it. The RELEASE itself must
    not be raised the same way, though. Opening the jaws above the rim drops every arrival onto
    the pile already in the container, which bounces the earlier ones back out and loses goals
    that were already credited. So transit clears the rim and the release descends to the goal.
    """
    point = cond.goal_point_m.astype(np.float64).reshape(3).copy()
    if clear_rim:
        clear_z = float(cond.release_z_m) if cond.release_z_m is not None else point[2]
        point[2] = max(point[2], clear_z)
    elif insertion_m is not None and cond.release_z_m is not None:
        point[2] = max(point[2], float(cond.release_z_m) - insertion_m)
    point[2] += height_m
    return point


def _corrected(grasp_pt: np.ndarray, eef_m: np.ndarray) -> np.ndarray:
    """``grasp_pt`` with the arm's standing in-plane error fed forward (see _DESCENT_XY_GAIN)."""
    point = grasp_pt.copy()
    error = (grasp_pt - eef_m)[:2]
    point[:2] += np.clip(
        _DESCENT_XY_GAIN * error, -_DESCENT_XY_CORRECTION_CAP_M, _DESCENT_XY_CORRECTION_CAP_M
    )
    return point


def step(  # noqa: C901, PLR0911, PLR0912, PLR0915 -- one flat phase dispatch; splitting it would fragment the machine
    state: EnvState,
    cond: PrivilegedCondition,
    eef_m: np.ndarray,
    gripper_open: bool,
    params: MotionParams,
    openness: Optional[float] = None,
    limit: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, bool]:
    """Advance the state machine one step, mutating ``state``.

    Returns the absolute EEF waypoint to command and whether the gripper should be open. ``limit``
    pulls a free-space target inside the arm's reach, so the phases wait for the point the arm can
    actually get to.
    """
    if limit is None:
        limit = lambda point: point  # noqa: E731
    eef_m = np.asarray(eef_m, dtype=np.float64).reshape(3)
    is_pick_only = cond.predicate in _PICK_ONLY_PREDICATES
    is_reach = cond.predicate in _REACH_PREDICATES
    has_goal = cond.goal_point_m is not None and not is_pick_only and not is_reach
    grasp_pt = _grasp_point(cond, params)

    def advance(phase: Phase) -> None:
        state.phase = phase
        state.phase_steps = 0
        state.best_dist_m = float("inf")
        state.no_progress_steps = 0

    def out_of_reach(target_m: np.ndarray, tol_m: float) -> bool:
        """The arm has stopped closing in on ``target_m`` while still more than ``tol_m`` from it."""
        if params.reach_stall_steps <= 0:
            return False
        dist = float(np.linalg.norm(target_m - eef_m))
        if dist < state.best_dist_m - 0.002:
            state.best_dist_m, state.no_progress_steps = dist, 0
        else:
            state.no_progress_steps += 1
        return dist > tol_m and state.no_progress_steps >= params.reach_stall_steps

    state.phase_steps += 1
    transit_step_m = params.transit_translation_per_step_m or params.max_translation_per_step_m
    # GRASP/RELEASE keep their own dwell-based advance.
    timed_out = state.phase_steps >= params.max_phase_steps

    if state.phase == Phase.REHOME:
        if (
            state.home_m is None
            or _reached(eef_m, state.home_m, params.xy_tol_m, params.z_tol_m)
            or timed_out
        ):
            advance(Phase.APPROACH)
        else:
            return _toward(eef_m, state.home_m, transit_step_m), True

    if state.phase in (Phase.LIFT, Phase.CARRY, Phase.PLACE) and params.grasp_lost_steps > 0:
        state.grasp_missing_steps = 0 if _held(cond, state, eef_m) else state.grasp_missing_steps + 1
        if state.grasp_missing_steps >= params.grasp_lost_steps:
            advance(Phase.DONE)
            return eef_m, True

    if state.phase == Phase.APPROACH:
        if state.approach_top_m is None:
            state.approach_top_m = cond.target.top_z_m
        hover = grasp_pt.copy()
        hover[2] = min(cond.target.top_z_m, state.approach_top_m) + params.approach_height_m
        hover = limit(hover)
        if out_of_reach(hover, 2 * params.xy_tol_m):
            state.unreachable = True
            advance(Phase.DONE)
            return eef_m, True
        stranded = (
            timed_out
            and params.approach_timeout_max_xy_m is not None
            and float(np.linalg.norm((hover - eef_m)[:2])) > params.approach_timeout_max_xy_m
        )
        if stranded:
            state.unreachable = True
            advance(Phase.DONE)
            return eef_m, True
        if _reached(eef_m, hover, params.xy_tol_m, params.z_tol_m) or timed_out:
            state.descent_target_m = grasp_pt.copy()
            advance(Phase.DESCEND)
        return _toward(eef_m, hover, transit_step_m), True

    if state.phase in (Phase.DESCEND, Phase.GRASP) and state.descent_target_m is not None:
        grasp_pt = state.descent_target_m

    if state.phase == Phase.DESCEND:
        # A stall only means "bottomed out" once the wrist is over the object; while the in-plane
        # correction converges the arm trades height for reach and pauses.
        centred = float(np.linalg.norm((grasp_pt - eef_m)[:2])) <= params.xy_tol_m
        dropped = (
            centred
            and state.last_z_m is not None
            and (state.last_z_m - float(eef_m[2])) < _DESCENT_STALL_EPS_M
        )
        state.stalled_steps = state.stalled_steps + 1 if dropped else 0
        state.last_z_m = float(eef_m[2])
        arrived = _reached(eef_m, grasp_pt, params.xy_tol_m, _DESCENT_Z_TOL_M)
        # Stuck high or off to the side: not the object bottoming the descent out, the arm itself.
        if not arrived and not centred and out_of_reach(grasp_pt, 2 * params.xy_tol_m):
            state.unreachable = True
            advance(Phase.DONE)
            return eef_m, True
        stalled = state.stalled_steps >= _DESCENT_STALL_STEPS
        if (
            stalled
            and not is_reach
            and params.descent_high_tol_m > 0.0
            and float(eef_m[2]) - float(grasp_pt[2]) > params.descent_high_tol_m
        ):
            state.unreachable = True
            advance(Phase.DONE)
            return eef_m, True
        closing = (
            params.close_above_m > 0.0
            and rim_direction(cond.target, params) is not None
            and float(eef_m[2]) - float(grasp_pt[2]) < params.close_above_m
        )
        if arrived or stalled or timed_out:
            advance(Phase.DONE if is_reach else (Phase.GRASP if closing else Phase.CENTER))
        speed = params.close_descent_step_m if closing else params.max_translation_per_step_m
        return _toward(eef_m, _corrected(grasp_pt, eef_m), speed), not closing

    if state.phase == Phase.CENTER:
        # Lowering and centring cannot happen at once, so centring gets its own phase.
        # It does converge precisely at a HELD height, so the two are separated: descend, then slide
        # onto the object before the jaws move.
        over = grasp_pt.copy()
        over[2] = float(eef_m[2])
        if float(np.linalg.norm((grasp_pt - eef_m)[:2])) <= params.xy_tol_m or timed_out:
            advance(Phase.GRASP)
        return _toward(eef_m, _corrected(over, eef_m), params.max_translation_per_step_m), True

    if state.phase == Phase.GRASP:
        # Hold the grasp pose, command CLOSE. Advance once the grasp registers
        # (after a brief settle), or give up waiting after max_grasp_steps.
        closed_enough = state.phase_steps >= params.grasp_close_steps
        state.grasp_seen_steps = state.grasp_seen_steps + 1 if cond.grasped else 0
        jaws_stopped = False
        if params.jaw_stall_steps > 0 and openness is not None:
            still = state.last_openness is not None and abs(openness - state.last_openness) < _JAW_STILL_EPS
            state.jaw_still_steps = state.jaw_still_steps + 1 if still else 0
            state.last_openness = openness
            if state.jaw_still_steps >= params.jaw_stall_steps and closed_enough:
                if openness <= _JAW_SHUT:
                    advance(Phase.DONE)
                    return grasp_pt, True
                jaws_stopped = openness < 0.9
        if (state.grasp_seen_steps >= params.grasp_confirm_steps or jaws_stopped) and closed_enough:
            state.grasp_eef_m = eef_m.copy()
            state.hold_offset_m = eef_m[:2] - np.asarray(cond.target.center_m, dtype=np.float64)[:2]
            state.hold_vec_m = eef_m - np.asarray(cond.target.center_m, dtype=np.float64)
            advance(Phase.DONE if (is_pick_only and cond.predicate == "grasp") else Phase.LIFT)
        elif state.phase_steps >= params.max_grasp_steps:
            # Nothing was picked up. Flying the empty gripper through carry and place would spend a
            # third of a goal's budget achieving nothing, so end the chain and let the caller
            # restart it. The retry re-reads where the object actually ended up.
            advance(Phase.DONE)
        return grasp_pt, False

    if state.phase == Phase.LIFT:
        base = state.grasp_eef_m if state.grasp_eef_m is not None else grasp_pt
        lifted = base.copy()
        lifted[2] = base[2] + params.lift_height_m
        lifted = limit(lifted)
        if _reached(eef_m, lifted, params.xy_tol_m, params.z_tol_m) or timed_out:
            advance(Phase.CARRY if has_goal else Phase.DONE)
        return _toward(eef_m, lifted, transit_step_m), False

    if state.phase == Phase.CARRY:
        over_goal = limit(_held_over_goal(cond, state, params.carry_height_m))
        # XY arrival is enough to start the place descent (height is reached in PLACE).
        if float(np.linalg.norm((over_goal - eef_m)[:2])) <= params.xy_tol_m or timed_out:
            advance(Phase.PLACE)
        return _toward(eef_m, over_goal, transit_step_m), False

    if state.phase == Phase.PLACE:
        place = limit(
            _held_over_goal(
                cond,
                state,
                params.place_clearance_m,
                clear_rim=False,
                insertion_m=params.max_container_insertion_m,
            )
        )
        if _reached(eef_m, place, params.xy_tol_m, params.z_tol_m) or timed_out:
            advance(Phase.RELEASE)
        return _toward(eef_m, place, params.max_translation_per_step_m), False

    if state.phase == Phase.RELEASE:
        # Hold position, command OPEN until the object is released (or a few steps).
        # Jaws jammed against a container's floor never read open: leave anyway on the timeout.
        released = (not cond.grasped or gripper_open) and state.phase_steps >= params.release_steps
        if released or timed_out:
            advance(Phase.RETREAT)
        return (
            _held_over_goal(
                cond,
                state,
                params.place_clearance_m,
                clear_rim=False,
                insertion_m=params.max_container_insertion_m,
            ),
            True,
        )

    if state.phase == Phase.RETREAT and state.shed_steps < params.shed_pulses * _SHED_PULSE_STEPS:
        center = np.asarray(cond.target.center_m, dtype=np.float64)
        # Riding with the pads, not resting just under them where it was put down.
        wedged = float(np.linalg.norm(center - eef_m)) < _SHED_DIST_M and center[2] > eef_m[2] - _SHED_BELOW_M
        if wedged and state.phase_steps >= params.release_steps:
            state.shed_steps += 1
            # The first half of each pulse closes the jaws, the second opens them, holding position.
            return eef_m, (state.shed_steps - 1) % _SHED_PULSE_STEPS >= _SHED_PULSE_STEPS // 2

    if state.phase == Phase.RETREAT:
        # Measured against the goal, not against the live EEF: "approach_height above where I am"
        # recedes as fast as the arm climbs, so the phase could only ever end on its timeout.
        up = eef_m.copy()
        up[2] = float(
            _over_goal(cond, params.approach_height_m)[2]
            if has_goal
            else grasp_pt[2] + params.approach_height_m
        )
        up = limit(up)
        if eef_m[2] >= up[2] - params.z_tol_m or timed_out:
            advance(Phase.DONE)
        return _toward(eef_m, up, transit_step_m), True

    # Phase.DONE: hold current position. A completed pickup keeps holding the object (gripper closed)
    # so a following "place" subtask can carry it; everything else opens.
    holding = is_pick_only and state.grasp_eef_m is not None
    return eef_m, not holding
