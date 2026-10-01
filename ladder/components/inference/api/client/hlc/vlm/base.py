import base64
import binascii
import io
import types
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import numpy as np
import PIL.Image

from ladder.components.inference.api.client.policy import Policy
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import Config, Configurable
from ladder.core.databib.template import Template

LOGGER = get_logger(__name__)


class VLMConfig(Config):
    # A transient backend failure (rate limit, timeout, transport) yields an empty reply and the
    # caller keeps its state. The run aborts after this many consecutive failures, or once more than
    # `max_failure_rate` of at least `min_calls_before_rate_check` calls were lost: a run that got
    # answers 90% of the time must not be scored as if it had been answered.
    max_consecutive_failures: int = 100
    max_failure_rate: float = 0.05
    min_calls_before_rate_check: int = 200


class VLMUnavailableError(RuntimeError):
    """The backend failed more often than its failure budget allows."""


VLMConfigT = TypeVar("VLMConfigT", bound=VLMConfig)


def _from_openai_content(content: Any) -> Any:
    """OpenAI wire format back into ours: ``image_url`` data URIs become HxWx3 uint8 arrays."""
    if not isinstance(content, list):
        return content
    converted = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "image_url":
            converted.append(item)
            continue
        url = (item.get("image_url") or {}).get("url", "")
        if "," not in url:
            raise ValueError(f"image_url is not a data URI, so it cannot be read locally: {url[:40]}")
        try:
            raw = base64.b64decode(url.split(",", 1)[1])
            image = PIL.Image.open(io.BytesIO(raw)).convert("RGB")
        except (binascii.Error, OSError, ValueError) as error:
            raise ValueError(f"could not decode an inline image for a local backend: {error}") from error
        converted.append({"type": "image", "image": np.asarray(image)})
    return converted


class VLM(Configurable[VLMConfigT], Template[VLMConfigT]):
    """Chat-style vision-language backend shared by every VLM role (decomposition, verification,
    generation). Messages are OpenAI-style ``{role, content}`` dicts; ``content`` is a string or a
    list of ``{"type": "text"|"image", ...}`` items, with images as raw HxWx3 uint8 arrays."""

    # Exceptions a backend may raise for a failure worth skipping; anything else is a bug and raises.
    transient_errors: Tuple[Type[BaseException], ...] = ()

    def __init__(self, config: VLMConfigT) -> None:
        super().__init__(config)
        self._calls = 0
        self._failures = 0
        self._consecutive_failures = 0

    def warmup(self) -> None:
        """Initialize shared state on the calling thread (lazy first-touch is not thread-safe)."""

    def bind_policy(self, policy: "Policy") -> None:
        """Offered the LL policy once so a backend served by the same server can reuse its client."""

    def request(self, **kwargs: Any) -> Any:
        """One completion in OpenAI response shape, for a caller that drives the wire format itself.

        Backends that ARE an OpenAI endpoint override this to pass the request through untouched.
        """
        messages = [{**m, "content": _from_openai_content(m.get("content"))} for m in kwargs["messages"]]
        text = self.complete(
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.0),
        )
        message = types.SimpleNamespace(content=text)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=None)

    def complete(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """One completion; the reply text, or \"\" when the backend failed transiently and the failure
        budget allows skipping the call. Raises ``VLMUnavailableError`` once the budget is spent."""
        self._calls += 1
        try:
            text = self._complete(messages, max_tokens, temperature, reasoning_effort)
        except self.transient_errors as error:
            self._failures += 1
            self._consecutive_failures += 1
            rate = self._failures / self._calls
            if self._consecutive_failures > self.config.max_consecutive_failures or (
                self._calls >= self.config.min_calls_before_rate_check and rate > self.config.max_failure_rate
            ):
                raise VLMUnavailableError(
                    f"{type(self).__name__} lost {self._failures} of {self._calls} calls "
                    f"({self._consecutive_failures} in a row); last error: {type(error).__name__}: {error}"
                ) from error
            LOGGER.warn(
                f"{type(self).__name__}: call failed ({type(error).__name__}: {error}); empty reply, "
                f"{self._failures}/{self._calls} lost so far"
            )
            return ""
        self._consecutive_failures = 0
        return text

    @abstractmethod
    def _complete(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        reasoning_effort: Optional[str],
    ) -> str:
        """One completion; returns the reply text. ``reasoning_effort`` is ignored by backends
        without a thinking-budget knob."""
