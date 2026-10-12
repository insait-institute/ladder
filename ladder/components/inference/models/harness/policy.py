from typing import Dict, List, Optional

from ladder.components.inference.api.client.program import ProgramEpisode, ProgramRunner
from ladder.components.inference.api.core.action import RobotAction
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.components.inference.models.harness.agent import Agent, AgentConfig
from ladder.core.databib.config import AutoConfigurable, Configurable

_AGENT_TIMEOUT_S = 1200.0


class HarnessPolicy(Configurable[AgentConfig]):
    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        vlm = AutoConfigurable(config.vlm)
        vlm.warmup()
        self._agent = Agent(config, vlm)
        self._programs = ProgramRunner(self._run, "harness", _AGENT_TIMEOUT_S)

    def reset(self) -> None:
        self._programs.reset()

    def predict_actions(
        self,
        observations: List[RobotObservation],
        instructions: List[str],
        env_indices: Optional[List[int]] = None,
    ) -> List[RobotAction]:
        return self._programs.actions(observations, instructions, env_indices)

    def pop_events(self) -> Dict[int, List[Dict[str, object]]]:
        return self._programs.pop_events()

    def _run(self, episode: ProgramEpisode) -> None:
        self._agent.run(episode, episode.task, episode.events)
