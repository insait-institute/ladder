from typing import Any, Dict, List, Optional

from ladder.components.inference.api.client.hlc.vlm.base import VLM, VLMConfig
from ladder.components.inference.api.client.policy import Policy


class ServedVLMConfig(VLMConfig):
    pass


class ServedVLM(VLM[ServedVLMConfig]):
    """The VLM hosted by the run's batch inference server, so the workers sharing a GPU share one
    copy of it. Takes no connection settings: it is handed the policy's client via bind_policy."""

    def __init__(self, config: ServedVLMConfig) -> None:
        super().__init__(config)
        self._inference = None

    def bind_policy(self, policy: Policy) -> None:
        self._inference = policy.inference

    def _complete(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        reasoning_effort: Optional[str],
    ) -> str:
        del reasoning_effort
        assert self._inference is not None, "ServedVLM needs bind_policy(); is the policy remote?"
        return self._inference.vlm_complete(messages, max_tokens, temperature)
