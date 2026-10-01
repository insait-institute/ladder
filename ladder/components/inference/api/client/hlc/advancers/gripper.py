from typing import Dict, Optional

from ladder.components.inference.api.client.hlc.base import Decomposition, SubtaskAdvancer
from ladder.components.inference.api.client.hlc.gripper import GripperTracker
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.core.databib.config import Config


class GripperAdvancerConfig(Config):
    gripper_open_threshold: float = 0.5  # normalized openness above this counts as OPEN (released)
    min_hold_steps: int = 3  # sustained CLOSED steps before a release counts (rejects a transient pinch)
    lift_threshold_m: float = 0.0  # require the eef to have RISEN this far during the grasp (0 = off)
    advance_cooldown_steps: int = 0  # after advancing, ignore new releases this many steps (0 = off)
    # Advance anyway after this many steps on one subtask with no qualifying release (0 = off).
    # A release gate strict enough to reject failed clenches is also strict enough to strand the
    # arm, and this advancer has no other way out: measured at min_hold_steps 21 + 3 cm lift, 50%
    # of episodes never advanced at all and 8% reached the last subtask, against 100% at the loose
    # gate. VerifiedAdvancer carries the same escape under the same name.
    attempt_stall_steps: int = 0
    cyclic_repair: bool = False  # after the last subtask, wrap to the first and keep re-attempting


class GripperAdvancer(SubtaskAdvancer[GripperAdvancerConfig]):
    """A sustained grasp that LIFTS then releases = one pick-and-place done; advance then.
    ``lift_threshold_m`` rejects in-place pinches; ``advance_cooldown_steps`` rejects re-fires while
    the gripper fidgets after releasing; ``attempt_stall_steps`` advances anyway on a subtask the
    policy never completes, which a gate strict enough to reject failed clenches otherwise strands.
    Proprioception only."""

    def __init__(self, config: GripperAdvancerConfig) -> None:
        super().__init__(config)
        self._tracker = GripperTracker(config.gripper_open_threshold)
        self._cooldown: Dict[int, int] = {}
        self._steps_on_subtask: Dict[int, int] = {}
        self._commanded: Dict[int, int] = {}  # the subtask the step count belongs to

    def reset(self, env_idx: int) -> None:
        self._tracker.reset(env_idx)
        self._cooldown[env_idx] = 0
        self._steps_on_subtask[env_idx] = 0
        self._commanded.pop(env_idx, None)

    def next_index(
        self,
        observation: RobotObservation,
        env_idx: int,
        current_idx: int,
        decomposition: Optional[Decomposition],
    ) -> int:
        if self._cooldown.get(env_idx, 0) > 0:
            self._cooldown[env_idx] -= 1
        if self._commanded.get(env_idx) != current_idx:
            self._commanded[env_idx] = current_idx
            self._steps_on_subtask[env_idx] = 0
        self._steps_on_subtask[env_idx] = self._steps_on_subtask.get(env_idx, 0) + 1
        self._tracker.update(observation, env_idx)
        # Absent is not "not released yet": with no gripper reading a release can never be observed,
        # so unless attempt_stall_steps is set this would command subtask 0 to the end of the
        # episode and score zero without erroring. Raise rather than lean on a knob that defaults off.
        if not self._tracker.has_signal(env_idx):
            raise RuntimeError(
                "GripperAdvancer needs a gripper reading; this env publishes none, so it could "
                "never advance"
            )
        released = self._tracker.released_after_hold(
            env_idx, self.config.min_hold_steps, self.config.lift_threshold_m
        )
        stalled = (
            self.config.attempt_stall_steps > 0
            and self._steps_on_subtask[env_idx] >= self.config.attempt_stall_steps
        )
        if not ((released or stalled) and self._cooldown.get(env_idx, 0) == 0):
            return current_idx
        self._cooldown[env_idx] = self.config.advance_cooldown_steps
        if decomposition is None:  # generative source: every advance generates a fresh subtask
            return current_idx + 1
        if current_idx < len(decomposition.subtasks) - 1:
            return current_idx + 1
        if self.config.cyclic_repair and len(decomposition.subtasks) > 1:
            return 0
        return current_idx
