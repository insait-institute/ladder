from typing import Dict, List, Optional

from ladder.components.inference.api.client.hlc.base import Decomposition, SubtaskAdvancer
from ladder.components.inference.api.client.hlc.gripper import GripperTracker
from ladder.components.inference.api.client.hlc.verifiers import Verifier
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.core.databib.config import AutoConfig, AutoConfigurable, Config

_VALID_PHASES = ("reach", "grasp", "place", "release")


def _validate_phase_cycle(phase_cycle: List[str]) -> None:
    if not phase_cycle or any(phase not in _VALID_PHASES for phase in phase_cycle):
        raise ValueError(f"phase_cycle must be a non-empty subset of {_VALID_PHASES}; got {phase_cycle}")


def _phase_advance(
    phase: str,
    *,
    reach_done: bool,
    grasped: Optional[bool],
    placed: Optional[bool],
    budget: int,
    reach_max_steps: int,
    release_steps: int,
) -> bool:
    """Is the current phase complete? reach and release always terminate on a step budget. grasp and
    place terminate ONLY on their signal, so an absent one (``None``, as in `gripper.py`) can never
    complete the phase: the advancer would hold one subtask to the end of the episode and the run
    would score zero without erroring. Raise rather than return that as a result.

    Shared by PhaseAdvancer (gripper signals) and PrivilegedPhaseAdvancer (oracle), so a caller
    cannot forget to check -- an absent signal is unrepresentable as "not yet"."""
    if phase == "reach":
        return reach_done or budget >= reach_max_steps
    if phase == "release":
        return budget >= release_steps
    signal = grasped if phase == "grasp" else placed
    if signal is None:
        raise RuntimeError(
            f"a '{phase}' phase has no completion signal here, so it can never end: the episode "
            f"would command one subtask to the end and score zero"
        )
    return signal


class PhaseAdvancerConfig(Config):
    # Phase structure of the decomposition, POSITIONALLY: subtask i is phase_cycle[i % len(cycle)].
    # Must mirror the decomposer's phase_templates (declared config data, never inferred).
    phase_cycle: List[str] = ["grasp", "place"]
    gripper_open_threshold: float = 0.5
    min_hold_steps: int = 3  # sustained CLOSED steps that confirm a grasp / qualify a release
    lift_threshold_m: float = 0.0  # require the eef to have risen this far before a release advances
    reach_max_steps: int = (
        12  # leave a REACH subtask after this many steps if no grasp starts (no gripper edge)
    )
    release_steps: int = (
        3  # dwell this many steps on a RELEASE subtask before the next object (no gripper edge)
    )
    advance_cooldown_steps: int = (
        6  # after a switch, gripper-edge advances are ignored this many steps (debounce chatter)
    )
    # Plan-end repair: after dwelling repair_stall_steps at plan end, verify each goal's transport
    # subtask and jump back to the reach subtask of the first not-done goal, returning to plan end
    # once its place completes. None disables it (monotonic, idle at the end). 2-phase cycle only.
    repair_verifier: Optional[AutoConfig] = None
    repair_stall_steps: int = 25

    def __post_init__(self):
        super().__post_init__()
        _validate_phase_cycle(self.phase_cycle)
        if self.repair_verifier is not None and len(self.phase_cycle) != 2:
            raise ValueError("plan-end repair assumes the 2-phase grasp/place cycle")


class PhaseAdvancer(SubtaskAdvancer[PhaseAdvancerConfig]):
    """Advance a phase-per-object decomposition (2-phase grasp/place, or 4-phase reach/grasp/place/
    release). Each subtask's phase is its position in ``phase_cycle``; grasp/place switch on the
    gripper edge, reach/release (no gripper edge) fall back to a step budget:
      reach   -> when a grasp begins (gripper closes), else after reach_max_steps;
      grasp   -> when the close is sustained (>= min_hold_steps);
      place   -> when the gripper releases after a held grasp;
      release -> after release_steps.
    Gripper-edge advances are debounced for advance_cooldown_steps. Monotonic; proprioception only."""

    def __init__(self, config: PhaseAdvancerConfig) -> None:
        super().__init__(config)
        self._tracker = GripperTracker(config.gripper_open_threshold)
        self._steps: Dict[int, int] = {}  # steps in the current subtask (drives the reach/release budgets)
        self._cooldown: Dict[int, int] = {}  # gripper-edge advances suppressed while > 0 (chatter debounce)
        self._verifier: Optional[Verifier] = (
            AutoConfigurable(config.repair_verifier) if config.repair_verifier is not None else None
        )
        self._repairing: Dict[int, Optional[int]] = {}  # goal being re-attempted after a plan-end check

    def reset(self, env_idx: int) -> None:
        self._tracker.reset(env_idx)
        self._steps[env_idx] = 0
        self._cooldown[env_idx] = 0
        self._repairing[env_idx] = None

    def _repair_target(self, observation: RobotObservation, env_idx: int, subtasks) -> Optional[int]:
        """Verify each goal's transport subtask; return the reach index of the first not-done goal.
        Every call is logged to ``pending_events`` with the env's privileged flag."""
        achieved = (observation.metadata or {}).get("oracle", {}).get("achieved")
        for goal in range(len(subtasks) // 2):
            done, reply = self._verifier.check(observation, subtasks[2 * goal + 1])
            gt = bool(achieved[goal]) if isinstance(achieved, list) and goal < len(achieved) else None
            self.pending_events.setdefault(env_idx, []).append(
                {"kind": "verify", "subtask": 2 * goal + 1, "verdict": done, "reply": reply[-400:], "gt": gt}
            )
            if done is False:
                return 2 * goal
        return None

    def next_index(
        self,
        observation: RobotObservation,
        env_idx: int,
        current_idx: int,
        decomposition: Optional[Decomposition],
    ) -> int:
        self._tracker.update(observation, env_idx)
        self._steps[env_idx] = self._steps.get(env_idx, 0) + 1
        if self._cooldown.get(env_idx, 0) > 0:
            self._cooldown[env_idx] -= 1
        if decomposition is not None and current_idx >= len(decomposition.subtasks) - 1:
            if (
                self._verifier is not None
                and self._steps.get(env_idx, 0) >= self.config.repair_stall_steps
                and self._tracker.closed_streak(env_idx) == 0  # never retarget mid-carry
            ):
                self._steps[env_idx] = 0  # dwell again before the next round of checks
                target = self._repair_target(observation, env_idx, decomposition.subtasks)
                if target is not None:
                    self._repairing[env_idx] = target // 2
                    self._cooldown[env_idx] = self.config.advance_cooldown_steps
                    return target
            return current_idx
        settled = self._cooldown.get(env_idx, 0) == 0  # gripper-edge advances debounced after a switch
        streak = self._tracker.closed_streak(env_idx)
        gripper = self._tracker.has_signal(env_idx)  # None below: no reading, not "not yet"
        advance = _phase_advance(
            self.config.phase_cycle[current_idx % len(self.config.phase_cycle)],
            reach_done=settled and streak >= 1,  # a grasp has begun
            grasped=(settled and streak >= self.config.min_hold_steps) if gripper else None,
            placed=(
                settled
                and self._tracker.released_after_hold(  # released after a held grasp
                    env_idx, self.config.min_hold_steps, self.config.lift_threshold_m
                )
            )
            if gripper
            else None,
            budget=self._steps[env_idx],
            reach_max_steps=self.config.reach_max_steps,
            release_steps=self.config.release_steps,
        )
        if not advance:
            return current_idx
        self._steps[env_idx] = 0
        self._cooldown[env_idx] = self.config.advance_cooldown_steps
        repairing = self._repairing.get(env_idx)
        if repairing is not None and current_idx == 2 * repairing + 1:
            # re-attempted goal's place phase done: back to plan end, not forward through done goals
            self._repairing[env_idx] = None
            if decomposition is not None:
                return len(decomposition.subtasks) - 1
        return current_idx + 1


class PrivilegedPhaseAdvancerConfig(Config):
    # Phase structure, POSITIONALLY (see PhaseAdvancerConfig); goal i owns indices [i*P, (i+1)*P)
    # with P = len(phase_cycle). Keep aligned with PrivilegedPhaseDecomposer's phase_templates.
    phase_cycle: List[str] = ["grasp", "place"]
    reach_max_steps: int = 12  # reach has no ground-truth signal -> advance on a step budget
    release_steps: int = 3  # release has no ground-truth signal -> advance on a step budget
    achieve_settle_steps: int = 3  # placement must hold this many steps (debounce the achieved flicker)

    def __post_init__(self):
        super().__post_init__()
        _validate_phase_cycle(self.phase_cycle)


class PrivilegedPhaseAdvancer(SubtaskAdvancer[PrivilegedPhaseAdvancerConfig]):
    """The same per-phase decision as PhaseAdvancer (shared via ``_phase_advance``), but grasp/place
    switch on the env's ground-truth per-goal signals (``metadata['oracle']['grasping'/'achieved']``)
    instead of the gripper, isolating switch timing from the base policy. Reach/release have no
    ground-truth signal and run on a step budget. Tests whether perfect timing rescues the LL models.

    No benchmark publishes ``grasping`` today, so wiring this raises rather than deadlocking; an env
    supplies it the same way it supplies ``achieved``, one flag per goal."""

    def __init__(self, config: PrivilegedPhaseAdvancerConfig) -> None:
        super().__init__(config)
        self._steps: Dict[int, int] = {}
        self._settle: Dict[int, int] = {}

    def reset(self, env_idx: int) -> None:
        self._steps[env_idx] = 0
        self._settle[env_idx] = 0

    def next_index(
        self,
        observation: RobotObservation,
        env_idx: int,
        current_idx: int,
        decomposition: Optional[Decomposition],
    ) -> int:
        self._steps[env_idx] = self._steps.get(env_idx, 0) + 1
        if decomposition is not None and current_idx >= len(decomposition.subtasks) - 1:
            return current_idx
        oracle = (observation.metadata or {}).get("oracle") or {}
        grasping: List[bool] = oracle.get("grasping") or []
        achieved: List[bool] = oracle.get("achieved") or []
        goal = current_idx // len(self.config.phase_cycle)
        # None where the env publishes no flag for this goal: absent, not "not yet". No benchmark
        # publishes `grasping` today, which is exactly the case that must not read as False.
        grasped = bool(grasping[goal]) if goal < len(grasping) else None
        holding = bool(achieved[goal]) if goal < len(achieved) else None
        if holding is not None:
            self._settle[env_idx] = self._settle.get(env_idx, 0) + 1 if holding else 0
        advance = _phase_advance(
            self.config.phase_cycle[current_idx % len(self.config.phase_cycle)],
            reach_done=False,  # no ground-truth "reached" signal -> reach_max_steps budget only
            grasped=grasped,
            placed=None if holding is None else self._settle[env_idx] >= self.config.achieve_settle_steps,
            budget=self._steps[env_idx],
            reach_max_steps=self.config.reach_max_steps,
            release_steps=self.config.release_steps,
        )
        if not advance:
            return current_idx
        self._steps[env_idx] = 0
        self._settle[env_idx] = 0
        return current_idx + 1
