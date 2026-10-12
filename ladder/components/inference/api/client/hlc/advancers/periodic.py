from typing import Dict, Optional

from ladder.components.inference.api.client.hlc.base import Decomposition, SubtaskAdvancer
from ladder.components.inference.api.client.hlc.gripper import GripperTracker
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.core.databib.config import Config


class PeriodicAdvancerConfig(Config):
    interval: int = 30  # advance every N steps
    # Also advance on a gripper toggle (the natural pick->place boundary), so a generative source
    # re-plans on the fresh post-grasp/post-release frame.
    on_gripper_toggle: bool = False
    # At or below gripper_closed_threshold (normalized openness) the gripper counts as CLOSED; the
    # hysteresis band holds the latch mid-travel (anti-chatter).
    gripper_closed_threshold: float = 0.5
    gripper_toggle_hysteresis: float = 0.1


class PeriodicAdvancer(SubtaskAdvancer[PeriodicAdvancerConfig]):
    """Advance on a fixed step interval, optionally also on a gripper toggle. Under a generative
    source every advance means a fresh generation (the classic re-query clock); under a plan it
    walks the list on a timer, holding at the last subtask."""

    def __init__(self, config: PeriodicAdvancerConfig) -> None:
        super().__init__(config)
        self._tracker = GripperTracker(config.gripper_closed_threshold, config.gripper_toggle_hysteresis)
        self._steps: Dict[int, int] = {}

    def reset(self, env_idx: int) -> None:
        self._tracker.reset(env_idx)
        self._steps[env_idx] = 0

    def next_index(
        self,
        observation: RobotObservation,
        env_idx: int,
        current_idx: int,
        decomposition: Optional[Decomposition],
    ) -> int:
        steps = self._steps.get(env_idx, 0) + 1
        self._steps[env_idx] = steps
        if self.config.on_gripper_toggle:
            self._tracker.update(observation, env_idx)
        advance = steps % self.config.interval == 0 or (
            self.config.on_gripper_toggle and self._tracker.toggled(env_idx)
        )
        if not advance:
            return current_idx
        if decomposition is not None and current_idx >= len(decomposition.subtasks) - 1:
            return current_idx
        return current_idx + 1
