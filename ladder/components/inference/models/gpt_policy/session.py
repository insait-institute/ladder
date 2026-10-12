import base64
import dataclasses
import io
import json
from typing import Any, Dict, List, Optional

import gpt_policy.harness.errors
import gpt_policy.harness.models
import gpt_policy.harness.validation
import numpy as np
import PIL.Image

from ladder.components.inference.api.client.hlc.vlm.base import VLM
from ladder.components.inference.models.gpt_policy import protocol
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import Config, Configurable

LOGGER = get_logger(__name__)

_REPAIR = (
    "Your reply was not one valid tool selection: {error}\n"
    "{schema}"
    'Reply with ONLY the JSON object {{"name": "...", "arguments": {{...}}}}. No Markdown, no prose.'
)


@dataclasses.dataclass(frozen=True)
class Frame:
    name: str
    array: np.ndarray

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def mime_type(self) -> str:
        return "image/jpeg"

    @property
    def data(self) -> bytes:
        buffer = io.BytesIO()
        PIL.Image.fromarray(self.array).save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()

    @property
    def rgb_data(self) -> Optional[bytes]:
        return self.array.tobytes()

    def data_url(self) -> str:
        return f"data:{self.mime_type};base64,{base64.b64encode(self.data).decode('ascii')}"


class VlmSessionConfig(Config):
    max_tokens: int = 2048
    temperature: float = 0.0
    reasoning_effort: Optional[str] = None
    history_turns: int = 6
    history_image_turns: int = 1
    repair_attempts: int = 2


class LadderVlmSession(Configurable[VlmSessionConfig]):
    def __init__(self, config: VlmSessionConfig, vlm: VLM) -> None:
        super().__init__(config)
        self._vlm = vlm
        self._context: Optional[gpt_policy.harness.models.AgentContext] = None
        self._turns: List[Dict[str, Any]] = []
        self._schemas: Dict[str, Any] = {}
        self.last_decision_timing: Dict[str, float] = {}

    def start(self, context: gpt_policy.harness.models.AgentContext) -> None:
        self._context = context
        self._turns = []
        self._schemas = {tool["function"]["name"]: tool["function"] for tool in context.tools}

    def close(self) -> None:
        self._turns = []

    def decide(self, turn: gpt_policy.harness.models.AgentTurn) -> gpt_policy.harness.models.AgentDecision:
        assert self._context is not None, "decide() before start()"
        images = [Frame(name, frame.array) for name, frame in (turn.images or {}).items()]
        content: List[Dict[str, Any]] = [{"type": "text", "text": turn.observation}]
        for frame in images:
            content.append({"type": "text", "text": f"camera: {frame.name}"})
            content.append({"type": "image", "image": frame.array})
        attempt_messages = list(self._messages(content))
        last_error = ""
        for attempt in range(self.config.repair_attempts + 1):
            reply = self._vlm.complete(
                messages=attempt_messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                reasoning_effort=self.config.reasoning_effort,
            )
            if not reply.strip():
                raise gpt_policy.harness.errors.AgentOverloadedError(
                    "the VLM returned no content", provider=type(self._vlm).__name__, code="empty"
                )
            try:
                decision = gpt_policy.harness.validation.parse_decision(
                    protocol.normalize_selection(reply, self._schemas), self._context
                )
            except gpt_policy.harness.errors.AgentProtocolError as error:
                last_error = protocol.english_error(str(error))
                LOGGER.warn(f"[gpt-policy] invalid decision (attempt {attempt + 1}): {last_error}")
                attempt_messages = attempt_messages + [
                    {"role": "assistant", "content": reply},
                    {
                        "role": "user",
                        "content": _REPAIR.format(error=last_error, schema=self._named_schema(reply)),
                    },
                ]
                continue
            self._remember(content, reply)
            return decision
        raise gpt_policy.harness.errors.AgentOverloadedError(
            f"the VLM never produced a valid tool selection: {last_error}",
            provider=type(self._vlm).__name__,
            code="protocol",
        )

    def _named_schema(self, reply: str) -> str:
        for name, schema in self._schemas.items():
            if f'"{name}"' in reply:
                return f"The schema of {name} is:\n{json.dumps(schema, ensure_ascii=False)}\n"
        return ""

    def _messages(self, content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        assert self._context is not None
        catalog = json.dumps([tool["function"] for tool in self._context.tools], ensure_ascii=False)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": f"{self._context.instructions}\n\nTool schemas:\n{catalog}"}
        ]
        keep_images = max(0, self.config.history_image_turns)
        history = self._turns[-self.config.history_turns :]
        for index, entry in enumerate(history):
            older = len(history) - index > keep_images
            messages.append(
                {"role": "user", "content": _text_only(entry["content"]) if older else entry["content"]}
            )
            messages.append({"role": "assistant", "content": entry["reply"]})
        messages.append({"role": "user", "content": content})
        return messages

    def _remember(self, content: List[Dict[str, Any]], reply: str) -> None:
        self._turns.append({"content": content, "reply": reply})
        self._turns = self._turns[-self.config.history_turns :]


def _text_only(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        item for item in content if item.get("type") == "text" and not item["text"].startswith("camera: ")
    ]
