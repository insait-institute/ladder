import base64
import io
import os
import re
import urllib.parse
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing
import openai
import PIL.Image

from ladder.components.inference.api.client.hlc.cost_track import extract_usage, log_call
from ladder.components.inference.api.client.hlc.ratelimit import EndpointLimiter
from ladder.components.inference.api.client.hlc.vlm.base import VLM, VLMConfig
from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)


def _resolve_headers(headers: Dict[str, str]) -> Dict[str, str]:
    def resolve(match: re.Match) -> str:
        value = os.environ.get(match.group(1))
        if value is None:
            # An empty substitution is never what the config meant, and the failure it causes is
            # remote and opaque (some gateways reject an empty metadata header with a bare 400).
            LOGGER.warn(f"Header references ${{{match.group(1)}}} but it is unset; sending it empty")
            return ""
        return value

    return {key: re.sub(r"\$\{(\w+)\}", resolve, value) for key, value in headers.items()}


def _image_data_url(image: numpy.typing.NDArray[np.uint8]) -> str:
    pil_image = PIL.Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


def _to_openai_content(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    return [
        (
            {"type": "image_url", "image_url": {"url": _image_data_url(item["image"])}}
            if item.get("type") == "image"
            else item
        )
        for item in content
    ]


def _host(base_url: Optional[str]) -> str:
    """The endpoint's hostname, for the cost ledger. ``None`` is OpenAI's default endpoint."""
    return urllib.parse.urlsplit(base_url).hostname or "openai.com" if base_url else "openai.com"


class OpenAIVLMConfig(VLMConfig):
    model: str
    base_url: Optional[str] = None  # None = api.openai.com; any OpenAI-compatible endpoint works
    default_headers: Optional[Dict[str, str]] = None  # values may reference ${ENV_VAR}
    # Named per endpoint: a second provider does not share the first one's key.
    api_key_env: str = "VLM_API_KEY"
    # Some models reject `temperature` outright with a 400. Omitting it costs determinism, so leave
    # it on where it works.
    send_temperature: bool = True
    # SDK-level retries with its own backoff. Raise it for a gateway that throttles.
    max_retries: int = 2
    # Per-request bound. The default suits an eval, where a stalled call must not hold up a batch;
    # a caller that is not latency-bound raises it. A reasoning model planning a ten-goal scene
    # needs well over 30 s, and the timeout is indistinguishable in the result from a refusal.
    timeout: float = 30.0
    # Requests/second this NODE may send to the endpoint (0 = off); adapts down on refusals.
    max_rps: float = 0.0
    rate_limit_burst: float = 0.0
    rate_limit_cooldown_s: float = 2.0
    rate_limit_attempts: int = 5


class OpenAIVLM(VLM[OpenAIVLMConfig]):
    """Any OpenAI-compatible HTTPS endpoint (OpenAI, Gemini via a compat gateway, a self-served
    vLLM, ...). Needs ``VLM_API_KEY``. Every call is cost-logged (see cost_track.py)."""

    transient_errors = (openai.APIError,)

    def __init__(self, config: OpenAIVLMConfig) -> None:
        super().__init__(config)
        self._client: Optional[openai.OpenAI] = None

    def warmup(self) -> None:
        self._get()

    def _limiter_for(self) -> Optional[EndpointLimiter]:
        """Node-wide limiter for this endpoint, or None when max_rps is 0."""
        if self.config.max_rps <= 0:
            return None
        if getattr(self, "_rate_limiter", None) is None:
            self._rate_limiter = EndpointLimiter(
                key=f"{self.config.base_url}|{self.config.api_key_env}",
                max_rps=self.config.max_rps,
                burst=self.config.rate_limit_burst,
                cooldown_s=self.config.rate_limit_cooldown_s,
            )
        return self._rate_limiter

    def _get(self) -> openai.OpenAI:
        if self._client is None:
            api_key = os.environ.get(self.config.api_key_env, "")
            if api_key == "":
                raise ValueError(f"API key not found in environment variable {self.config.api_key_env!r}")
            kwargs: Dict[str, Any] = {
                "api_key": api_key,
                "timeout": self.config.timeout,
                # SDK retries fire inside one limiter slot and cannot see the breaker.
                "max_retries": 0 if self.config.max_rps > 0 else self.config.max_retries,
            }
            if self.config.base_url is not None:
                kwargs["base_url"] = os.path.expandvars(self.config.base_url)
            if self.config.default_headers is not None:
                kwargs["default_headers"] = _resolve_headers(self.config.default_headers)
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def request(self, **kwargs: Any) -> Any:
        """Issue one chat completion through this endpoint's node-wide limiter.

        Separate from `complete` because VoLo drives upstream's own prompts and control flow: it
        takes the raw client and calls it directly, so anything that lives only in `complete` does
        not apply to it. The limit is scoped to the endpoint, not to the caller -- a VoLo cell and
        one of ours on the same node share it -- so the pacing has to sit at the request, where both
        reach it.
        """
        limiter = self._limiter_for()
        if limiter is None:
            return self._get().chat.completions.create(**kwargs)
        # A rate-limited call must PAUSE, not kill the episode. VoLo swallows its own 429s and
        # degrades to a one-element plan, so an arm of ours that dies on the same 429 is scored
        # ABSENT while VoLo is scored degraded-but-alive, and the comparison is unfair.
        attempts = max(1, self.config.rate_limit_attempts)
        for attempt in range(attempts):
            with limiter.hold():
                try:
                    response = self._get().chat.completions.create(**kwargs)
                except openai.RateLimitError as error:
                    limiter.penalise(_retry_after(error))
                    if attempt == attempts - 1:
                        raise
                    LOGGER.warn(
                        f"{_host(self.config.base_url)} rate-limited; " f"attempt {attempt + 1}/{attempts}"
                    )
                except (openai.APIConnectionError, openai.InternalServerError) as error:
                    if attempt == attempts - 1:
                        raise
                    LOGGER.warn(
                        f"{_host(self.config.base_url)} transient {type(error).__name__}; "
                        f"attempt {attempt + 1}/{attempts}"
                    )
                else:
                    limiter.succeed()
                    return response
        raise AssertionError("unreachable: the final attempt re-raises")

    def _complete(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        reasoning_effort: Optional[str],
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [{**m, "content": _to_openai_content(m["content"])} for m in messages],
            "max_tokens": max_tokens,
        }
        if self.config.send_temperature:
            kwargs["temperature"] = temperature
        if reasoning_effort is not None:
            kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
        response = self.request(**kwargs)
        log_call(model=kwargs["model"], provider=_host(self.config.base_url), **extract_usage(response))
        if not response.choices:
            LOGGER.warn(
                f"{_host(self.config.base_url)} returned 200 with no choices "
                f"(model={kwargs['model']}, max_tokens={max_tokens}); treating as empty"
            )
            return ""
        choice = response.choices[0]
        if choice.finish_reason == "length":
            # A truncated reply parses like a wrong one: the JSON never closes. Hidden thinking tokens
            # count against max_tokens and their budget differs per model.
            LOGGER.warn(
                f"{_host(self.config.base_url)} truncated the reply at max_tokens={max_tokens} "
                f"(model={kwargs['model']})"
            )
        return choice.message.content or ""


def _retry_after(error: openai.RateLimitError) -> Optional[float]:
    """Seconds the endpoint asked us to wait, when it says so; None to use our own backoff."""
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    raw = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return max(0.0, float(raw)) if raw is not None else None
    except (TypeError, ValueError):
        return None
