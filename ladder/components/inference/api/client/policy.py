from typing import List, Optional, Protocol

from ladder.components.inference.api.core.action import RobotAction
from ladder.components.inference.api.core.observation import RobotObservation


class Policy(Protocol):
    """Protocol for a batched policy: take N observations + N instructions, return N actions.
    Per-env rollout state is kept internally, keyed by ``env_indices`` (the environment each
    observation is for), which defaults to ``0..N-1``, a full batch in order. Pass ``env_indices``
    explicitly only to advance a SUBSET of envs; otherwise their per-env state would misalign.

    Single-env callers wrap: ``policy.predict_actions([obs], [instr])[0]``.
    """

    def predict_actions(
        self,
        observations: List[RobotObservation],
        instructions: List[str],
        env_indices: Optional[List[int]] = None,
    ) -> List[RobotAction]:
        pass

    def reset(self) -> None:
        pass
