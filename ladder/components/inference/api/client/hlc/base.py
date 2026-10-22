import dataclasses
from abc import abstractmethod
from typing import Dict, List, Optional, Protocol, TypeVar

from ladder.components.inference.api.client.policy import Policy
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.core.databib.config import Config, Configurable
from ladder.core.databib.template import Template


@dataclasses.dataclass
class Decomposition:
    """A subtask plan plus its order semantics. ``ordered`` = goals must be completed in listed
    order; ``revisits`` = a repeated line moves the SAME object again rather than a second copy of
    it. Both are declared by whatever produced the plan, never inferred from the instruction
    wording."""

    subtasks: List[str]
    ordered: bool = False
    revisits: bool = False


class Orchestrator(Protocol):
    def reset(self, task_description: str, env_idx: int = 0, **kwargs) -> None:
        """Called once per env at episode start."""

    def bind_policy(self, policy: Policy) -> None:
        """Offered the low-level policy once, before the first step."""

    def get_commands(self, observations: List[RobotObservation], step: int) -> List[str]:
        """The instruction each env's policy should act on this step."""

    def pop_events(self) -> Dict[int, List[Dict[str, object]]]:
        """Drain per-env events (subtask switches, verdicts) into the step record."""

    def plan_for(self, scene: Optional[str], env_idx: Optional[int] = None) -> List[str]:
        """The decomposition to record for this episode, or empty for a system that plans as it
        goes. ``scene`` keys a plan resolved once per scene; ``env_idx`` is offered as well because
        a system that decomposes from the first frame produces a different plan per episode."""


# Sources: WHERE subtasks come from, either a fixed per-scene plan or on-the-fly generation from a
# model, interchangeable behind this interface.
SourceConfigT = TypeVar("SourceConfigT", bound=Config)


class SubtaskSource(Configurable[SourceConfigT], Template[SourceConfigT]):
    def reset(self, env_idx: int) -> None:
        """Clear per-env state at episode start."""

    def bind_policy(self, policy: Policy) -> None:
        """Offered the LL policy once so a same-server source can reuse its client. Default no-op."""

    def decompose(self, observation: RobotObservation, task: str) -> Optional[Decomposition]:
        """The scene's fixed plan, resolved once and reused (the controller retries while None).
        Generative sources keep this None and produce commands per index instead."""
        del observation, task
        return None

    def commands(
        self,
        observations: List[RobotObservation],
        tasks: List[str],
        env_indices: List[int],
        indices: List[int],
        step: int,
        decomposition: Optional[Decomposition],
    ) -> List[str]:
        """The command for each env at its (new) subtask index, batched over the moved envs. Default
        looks the index up in the plan; generative sources override to produce a fresh subtask."""
        del observations, tasks, env_indices, step
        assert decomposition is not None, f"{type(self).__name__} has no plan; implement commands()"
        last = len(decomposition.subtasks) - 1
        return [decomposition.subtasks[min(index, last)] for index in indices]


# Advancers: WHEN to switch subtasks. Each step they return the subtask index to command now
# (== current means stay). Moving the index emits the next plan entry / a fresh generated subtask,
# so any advancer composes with any source.
AdvancerConfigT = TypeVar("AdvancerConfigT", bound=Config)


class SubtaskAdvancer(Configurable[AdvancerConfigT], Template[AdvancerConfigT]):
    def __init__(self, config: AdvancerConfigT) -> None:
        super().__init__(config)
        # Per-env episode-event channel: advancers append event dicts here (e.g. a verifier
        # verdict) and so does the controller (one per subtask advance); the controller drains it
        # each step into the step metadata, from where the runner records it on the episode.
        self.pending_events: Dict[int, List[Dict[str, object]]] = {}

    def reset(self, env_idx: int) -> None:
        """Clear per-env progress at episode start."""

    def bind_policy(self, policy: Policy) -> None:
        """Offered the LL policy once so a same-server piece can reuse its client. Default no-op."""

    @abstractmethod
    def next_index(
        self,
        observation: RobotObservation,
        env_idx: int,
        current_idx: int,
        decomposition: Optional[Decomposition],
    ) -> int:
        """The subtask index to command now; == current_idx means stay. ``decomposition`` is None
        under a generative source; advancers whose scheduling needs the plan must assert it."""

    def command(self, env_idx: int, subtask: str) -> str:
        """The instruction to send for the selected subtask (advancers may rephrase it)."""
        del env_idx
        return subtask
