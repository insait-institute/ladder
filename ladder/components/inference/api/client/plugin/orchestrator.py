from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from ladder.components.inference.api.client.plugin.system import load, observation_dict
from ladder.components.inference.api.client.policy import Policy
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.core.databib.config import Config, Configurable


class PluginOrchestratorConfig(Config):
    system: str  # path/to/file.py:ClassName
    max_workers: int = 32  # environments whose instruction() runs at the same time


class PluginOrchestrator(Configurable[PluginOrchestratorConfig]):
    """Runs a user's orchestrator, written as one plain object per episode, in the orchestrator seat
    of the low-level policy (``api/client/hlc/base.Orchestrator``).

    The user's class is constructed with no arguments at the start of every episode and asked
    ``instruction(observation) -> str`` every step; ``observation`` is
    ``plugin.system.observation_dict``, whose ``instruction`` is the episode's task. The answer is the
    instruction the low-level policy acts on. All environments are asked at once, each on its own
    thread, so a model call in one does not wait for the others."""

    def __init__(self, config: PluginOrchestratorConfig) -> None:
        super().__init__(config)
        self._cls = load(config.system)
        self._episodes: Dict[int, Any] = {}
        self._tasks: Dict[int, str] = {}
        self._pool = ThreadPoolExecutor(max_workers=config.max_workers)

    def reset(self, task_description: str, env_idx: int = 0, **kwargs) -> None:
        del kwargs
        self._episodes[env_idx] = self._cls()
        self._tasks[env_idx] = task_description

    def bind_policy(self, policy: Policy) -> None:
        del policy

    def get_commands(self, observations: List[RobotObservation], step: int) -> List[str]:
        def ask(env: int) -> str:
            observation = observation_dict(observations[env], self._tasks[env], step)
            return str(self._episodes[env].instruction(observation))

        return list(self._pool.map(ask, range(len(observations))))

    def pop_events(self) -> Dict[int, List[Dict[str, object]]]:
        return {}

    def plan_for(self, scene: Optional[str], env_idx: Optional[int] = None) -> List[str]:
        del scene
        episode = self._episodes.get(env_idx if env_idx is not None else 0)
        return list(getattr(episode, "plan", None) or [])
