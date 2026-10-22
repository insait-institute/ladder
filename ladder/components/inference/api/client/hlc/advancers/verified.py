from typing import Dict, List, Optional

from ladder.components.inference.api.client.hlc.base import Decomposition, SubtaskAdvancer
from ladder.components.inference.api.client.hlc.gripper import GripperTracker
from ladder.components.inference.api.client.hlc.verifiers import Verifier
from ladder.components.inference.api.client.policy import Policy
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.core.common.logger import get_logger
from ladder.core.compute.seeder import SEEDER
from ladder.core.databib.config import AutoConfig, AutoConfigurable, Config

LOGGER = get_logger(__name__)


class VerifiedAdvancerConfig(Config):
    # Attempt segmentation (proprioception): a release-after-held-grasp ends an attempt;
    # attempt_stall_steps with no attempt end is a stall (dithering / unreachable) and forces a check.
    gripper_open_threshold: float = 0.5
    min_hold_steps: int = 3
    lift_threshold_m: float = 0.0
    attempt_stall_steps: int = 45
    # Spread each env's stall clock over [1 - jitter, 1 + jitter] x attempt_stall_steps (0 = exact
    # period), so a lockstep batch does not fire every check in the same second.
    attempt_stall_jitter: float = 0.0
    verify_delay_steps: int = 5  # verify this many steps AFTER a release (arm has cleared the object)
    # Injected task-agnostic yes/no VLM check; a Verifier with model="" disables it (blind round-robin).
    verifier: AutoConfig
    revoke_streak: int = 2  # consecutive 'no' verifications to revoke a prior done (occlusion hysteresis)
    # A clean grasp->lift->release cycle marks the CURRENT subtask provisionally done immediately
    # (monotonic progress even when the VLM undercounts); the VLM later confirms or revokes.
    provisional_done_on_cycle: bool = False
    provisional_lift_m: float = 0.03
    # Same provisional credit but scoped to ordered scenes, where a verifier false-negative would
    # otherwise stall the strict sequence forever.
    ordered_provisional: bool = False
    # Whether a not-done verdict withholds advancement at all. False keeps verifying and keeps
    # logging every verdict, but advances regardless, which is the only way to separate the value of
    # TESTING completion from the value of REFUSING to advance on a negative verdict. The two are
    # otherwise confounded, since only a verifying rule produces the judgement that withholds.
    # `retry_in_place` then has no effect: there is nothing to schedule.
    withhold_on_not_done: bool = True
    # After a not-done verdict: True = retry the CURRENT subtask until it verifies done; False =
    # skip-and-return (next not-done, revisit on wrap-around), which keeps one hard goal from
    # starving the rest but at the cost of leaving a nearly-finished goal for later.
    retry_in_place: bool = True
    # Verification TRIGGER, independent of the backend: default gripper-cycle; set gripper_trigger=
    # False + verify_interval>0 for a fixed-interval requery clock instead.
    gripper_trigger: bool = True
    verify_interval: int = 0  # >0: also fire a verify every N steps on the current goal (requery clock)
    # Auto-pick retry_in_place vs skip-and-return per plan: distinct subtasks -> retry (focus until
    # done), duplicate subtasks (interchangeable objects) -> skip. Overrides retry_in_place when set.
    auto_schedule: bool = False


class _VerifiedState:
    def __init__(self, n: int) -> None:
        self.done = [False] * n
        self.not_done_streak = [0] * n  # consecutive verifications contradicting a done flag (hysteresis)
        self.steps_on_goal = 0
        self.verify_in = -1  # countdown to a scheduled verification (-1 = none)
        self.last_image = None  # frame of the previous verification (verify_delta)
        self.stall_at = 0  # steps on the goal at which this cycle's stall check fires
        self.provisional = [False] * n  # done-marks credited by a gripper cycle, not yet verifier-confirmed


class VerifiedAdvancer(SubtaskAdvancer[VerifiedAdvancerConfig]):
    """Round-robin over per-goal subtasks with observable completion verification. The gripper
    segments the episode into pick-place attempts; at each attempt end the frame is verified and the
    subtask marked done/not-done. The next command is the first not-done subtask after the current
    one (skip-and-return, so one hard goal can't starve the rest); done goals are never re-commanded.
    An ordered plan disables skipping (strictly sequential). Proprioception + camera only."""

    def __init__(self, config: VerifiedAdvancerConfig) -> None:
        super().__init__(config)
        self._tracker = GripperTracker(config.gripper_open_threshold)
        self._verifier: Verifier = AutoConfigurable(config.verifier)
        self._state: Dict[int, _VerifiedState] = {}
        self._prng = SEEDER.new_prng_np()  # jitter draws

    def reset(self, env_idx: int) -> None:
        self._tracker.reset(env_idx)
        self._state.pop(env_idx, None)

    def _verify(
        self,
        observation: RobotObservation,
        subtasks: List[str],
        state: _VerifiedState,
        env_idx: int,
        idx: int,
        revisits: bool,
    ) -> None:
        """Verify subtask ``idx`` and apply the done-flag + hysteresis + provisional bookkeeping.
        Marks done on yes; revokes a prior done only after ``revoke_streak`` consecutive 'no'; None
        (failure or disabled) leaves state as-is. Every call is logged to ``pending_events`` with the
        env's privileged flag, so verifier quality stays benchmarkable against ground truth."""
        if idx >= len(subtasks):
            return
        # A repeated line asks for a second copy only when the plan says the copies are distinct
        # objects. On a revisit chain it is the same object visiting again (a later leg sends the
        # one carrot back to the plate), and counting it asks for two carrots that do not exist.
        required = 1 if revisits else subtasks[: idx + 1].count(subtasks[idx])
        done_now, reply = self._verifier.check(
            observation, subtasks[idx], prev_image=state.last_image, required=required
        )
        achieved = (observation.metadata or {}).get("oracle", {}).get("achieved")
        gt = bool(achieved[idx]) if isinstance(achieved, list) and idx < len(achieved) else None
        self.pending_events.setdefault(env_idx, []).append(
            # `required` travels with the verdict: `count >= required` cannot be read back without it.
            {
                "kind": "verify",
                "subtask": idx,
                "verdict": done_now,
                "reply": reply[-400:],
                "gt": gt,
                "required": required,
            }
        )
        if self._verifier.config.verify_delta:
            # The delta frame is the PRIMARY view only -- carrying every view forward would double the
            # image count on the next call without adding occlusion evidence.
            image = (observation.images or {}).get(self._verifier.config.image_cameras[0])
            if image is not None:
                state.last_image = image
        if done_now is None:
            return
        if done_now:
            state.done[idx] = True
            state.not_done_streak[idx] = 0
            state.provisional[idx] = False
        else:
            if state.done[idx] and self.config.revoke_streak > 0:
                state.not_done_streak[idx] += 1
                if state.not_done_streak[idx] >= self.config.revoke_streak:
                    state.done[idx] = False
                    state.not_done_streak[idx] = 0
            if state.provisional[idx]:  # one contradiction reverts an unconfirmed gripper credit
                state.done[idx] = False
                state.provisional[idx] = False
                state.not_done_streak[idx] = 0
        LOGGER.info(f"VerifiedAdvancer env={env_idx}: subtask {idx} done={done_now} -> {state.done}")

    def _next_not_done(
        self, state: _VerifiedState, current_idx: int, n: int, ordered: bool, retry: bool
    ) -> int:
        if ordered:  # strict sequence: advance only past verified-done PREFIX goals
            idx = 0
            while idx < n and state.done[idx]:
                idx += 1
            return min(idx, n - 1)
        # skip-and-return: first not-done after current, wrap-around. retry_in_place starts the scan AT
        # the current subtask, so a not-done verdict re-commands it immediately instead of deferring it.
        start = 0 if retry else 1
        for offset in range(start, n + 1):
            idx = (current_idx + offset) % n
            if not state.done[idx]:
                return idx
        return current_idx  # everything verified done: hold (env success should end the episode)

    def bind_policy(self, policy: Policy) -> None:
        self._verifier.bind_policy(policy)

    def _stall_at(self, state: _VerifiedState) -> int:
        """Steps on the goal at which the stall clock fires this cycle, drawn once per cycle."""
        if state.stall_at <= 0:
            spread = self.config.attempt_stall_jitter * self.config.attempt_stall_steps
            state.stall_at = max(
                1, round(self.config.attempt_stall_steps + self._prng.uniform(-spread, spread))
            )
        return state.stall_at

    def next_index(
        self,
        observation: RobotObservation,
        env_idx: int,
        current_idx: int,
        decomposition: Optional[Decomposition],
    ) -> int:
        assert decomposition is not None, "VerifiedAdvancer schedules over a fixed plan"
        subtasks = decomposition.subtasks
        state = self._state.get(env_idx)
        if state is None or len(state.done) != len(subtasks):
            state = self._state[env_idx] = _VerifiedState(len(subtasks))
        self._tracker.update(observation, env_idx)
        state.steps_on_goal += 1
        released = self._tracker.released_after_hold(
            env_idx, self.config.min_hold_steps, self.config.lift_threshold_m
        )
        # attempt ended: verify once the arm has cleared the scene
        if self.config.gripper_trigger and released and state.verify_in < 0:
            state.verify_in = self.config.verify_delay_steps
            provisional = self.config.provisional_done_on_cycle or (
                self.config.ordered_provisional and decomposition.ordered
            )
            if provisional and self._tracker.released_after_hold(
                env_idx, self.config.min_hold_steps, self.config.provisional_lift_m
            ):  # clean lifted cycle: credit the commanded goal now; the VLM corrects later
                state.done[current_idx] = True
                state.not_done_streak[current_idx] = 0
                state.provisional[current_idx] = True
        elif (
            self.config.verify_interval > 0
            and state.steps_on_goal >= self.config.verify_interval
            and state.verify_in < 0
        ):
            state.verify_in = 0  # requery clock: fire a verify every verify_interval steps
        elif state.steps_on_goal >= self._stall_at(state) and state.verify_in < 0:
            state.verify_in = 0  # stalled on this goal: force a re-check / move on
            state.stall_at = 0  # a fresh draw for the next cycle
        if state.verify_in > 0:
            state.verify_in -= 1
            return current_idx
        if state.verify_in != 0:
            return current_idx
        state.verify_in = -1
        self._verify(observation, subtasks, state, env_idx, current_idx, decomposition.revisits)
        if self._tracker.is_closed(env_idx):  # holding an object: never retarget mid-carry (the policy
            state.steps_on_goal = 0  # would deliver the held object to the NEW goal's receptacle)
            return current_idx
        # Unordered scheduling depends on object distinctness. Distinct goals reward focus
        # (retry_in_place: finish the current goal before moving on); identical duplicates reward
        # not fixating (skip-and-return: any instance satisfies the shared subtask, so a stuck
        # attempt should defer). With auto_schedule the plan itself decides -- duplicate subtask
        # strings mean interchangeable objects -> skip; all-distinct -> retry.
        state.steps_on_goal = 0
        if not self.config.withhold_on_not_done:
            # Advance on the trigger whatever the verdict said, the way a non-verifying rule does.
            # The verdict is still produced and logged, so verifier accuracy stays measurable and the
            # only difference from the withholding arm is what is done with it.
            return min(current_idx + 1, len(subtasks) - 1)
        retry = self.config.retry_in_place
        if self.config.auto_schedule:
            retry = len(set(subtasks)) == len(subtasks)
        return self._next_not_done(state, current_idx, len(subtasks), decomposition.ordered, retry)
