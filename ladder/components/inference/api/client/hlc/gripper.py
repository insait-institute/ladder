import dataclasses
from typing import Dict, List, Optional

import numpy as np

from ladder.components.inference.api.core.observation import (
    EFFECTOR_NAMES,
    RobotObservation,
    named_effector_states,
)


def gripper_is_open(
    observation: RobotObservation, open_threshold: float, arm: Optional[str] = None
) -> Optional[bool]:
    """True/False when the gripper reads open/closed; None if the observation has no gripper signal.
    With ``arm`` unset the answer is over the whole robot: open only while EVERY arm reads open, so
    a caller asking "is the robot clamping something" sees either arm's grip."""
    readings = [
        float(np.ravel(state.gripper_state)[0]) > open_threshold
        for name, state in named_effector_states(observation.state)
        if state.gripper_state is not None and arm in (None, name)
    ]
    return all(readings) if readings else None


def eef_z(observation: RobotObservation, arm: str = EFFECTOR_NAMES[0]) -> Optional[float]:
    """``arm``'s end-effector height (base-frame z); None if that arm has no eef pose."""
    state = dict(named_effector_states(observation.state)).get(arm)
    pose = None if state is None else state.eef_pose
    return None if pose is None else float(np.asarray(pose).reshape(4, 4)[2, 3])


@dataclasses.dataclass
class _ArmTrack:
    """One arm's grasp state machine."""

    closed: Optional[bool] = None  # current closed latch (None until first signal)
    # Whether any reading has arrived. Distinct from a None latch, which is also what a first
    # reading inside the hysteresis band leaves behind -- absence has to outlive that.
    seen: bool = False
    streak: int = 0  # consecutive CLOSED steps up to and including now
    prev_streak: int = 0  # the streak as of the step before now
    ever_closed: bool = False
    toggled: bool = False  # did open<->closed flip THIS step
    grasp_start_z: Optional[float] = None  # eef z when the current grasp began
    max_lift_m: float = 0.0  # max (z - grasp_start_z) seen during the current grasp


class GripperTracker:
    """Per-arm grasp state machine, one track per (env, arm).

    Every query takes an optional ``arm``. Left unset the answer is over all of the env's arms --
    closed / toggled / ever_closed if ANY arm is, the longest streak of any -- which is what a
    single-arm caller has always got, and what an advancer watching "the gripper" of a two-armed
    robot means unless it says otherwise.
    """

    def __init__(self, open_threshold: float = 0.5, hysteresis: float = 0.0) -> None:
        self.open_threshold = open_threshold
        self.hysteresis = hysteresis  # 0 = plain threshold; >0 holds the latch in the band (anti-chatter)
        self._tracks: Dict[int, Dict[str, _ArmTrack]] = {}

    def reset(self, env_idx: int) -> None:
        self._tracks[env_idx] = {}

    def update(self, observation: RobotObservation, env_idx: int) -> None:
        """Advance each arm's state machine one step from its own gripper (and eef) signal."""
        for arm, state in named_effector_states(observation.state):
            track = self._tracks.setdefault(env_idx, {}).setdefault(arm, _ArmTrack())
            if state.gripper_state is None:
                # no gripper signal: leave latch/streak/lift untouched, no toggle
                track.toggled = False
                continue
            track.seen = True
            openness = float(np.ravel(state.gripper_state)[0])
            prev_closed = track.closed
            if openness <= self.open_threshold - self.hysteresis:
                now_closed: Optional[bool] = True
            elif openness >= self.open_threshold + self.hysteresis:
                now_closed = False
            else:  # inside the hysteresis band: hold the previous latch (anti-chatter)
                now_closed = prev_closed
            track.toggled = prev_closed is not None and now_closed is not None and now_closed != prev_closed
            track.closed = now_closed
            if now_closed:
                track.ever_closed = True
            track.prev_streak = track.streak
            track.streak = track.streak + 1 if now_closed else 0
            # Lift signal: the highest the eef rose since the current grasp began (reset each grasp).
            z = eef_z(observation, arm)
            if track.streak == 1:
                track.grasp_start_z = z
                track.max_lift_m = 0.0
            elif now_closed and z is not None and track.grasp_start_z is not None:
                track.max_lift_m = max(track.max_lift_m, z - track.grasp_start_z)

    def _tracks_for(self, env_idx: int, arm: Optional[str]) -> List[_ArmTrack]:
        tracks = self._tracks.get(env_idx, {})
        if arm is None:
            return list(tracks.values())
        track = tracks.get(arm)
        return [track] if track is not None else []

    def is_closed(self, env_idx: int, arm: Optional[str] = None) -> Optional[bool]:
        latches = [track.closed for track in self._tracks_for(env_idx, arm) if track.closed is not None]
        return any(latches) if latches else None

    def has_signal(self, env_idx: int, arm: Optional[str] = None) -> bool:
        """Whether this env has published any gripper reading, so a caller deriving a completion
        signal from the gripper can tell an absent one from an open one."""
        return any(track.seen for track in self._tracks_for(env_idx, arm))

    def toggled(self, env_idx: int, arm: Optional[str] = None) -> bool:
        return any(track.toggled for track in self._tracks_for(env_idx, arm))

    def closed_streak(self, env_idx: int, arm: Optional[str] = None) -> int:
        return max((track.streak for track in self._tracks_for(env_idx, arm)), default=0)

    def ever_closed(self, env_idx: int, arm: Optional[str] = None) -> bool:
        return any(track.ever_closed for track in self._tracks_for(env_idx, arm))

    def released_after_hold(
        self,
        env_idx: int,
        min_hold_steps: int,
        lift_threshold_m: float = 0.0,
        arm: Optional[str] = None,
    ) -> bool:
        """True on the single step a sustained grasp (>= min_hold_steps closed, and -- if
        lift_threshold_m > 0 -- lifted at least that far) has just released. With ``arm`` unset, any
        arm's release fires it."""
        return any(
            track.closed is False  # only fires on an OPEN step
            and track.prev_streak >= min_hold_steps
            and (lift_threshold_m <= 0.0 or track.max_lift_m >= lift_threshold_m)
            for track in self._tracks_for(env_idx, arm)
        )
