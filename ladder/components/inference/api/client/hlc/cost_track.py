import json
import os
import socket
import sys
import time
from typing import Any, Dict, Optional

_DEFAULT_PATH = os.path.expanduser(
    os.environ.get(
        "COST_TRACK_PATH",
        "~/.cache/long_bench/api_calls.jsonl",
    )
)


def log_call(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    thinking_tokens: int = 0,
    alias: Optional[str] = None,
    path: Optional[str] = None,
    **extra: Any,
) -> None:
    """Append one cost record. Best-effort: never raises."""
    record = {
        "ts": time.time(),
        "model": str(model),
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "thinking_tokens": int(thinking_tokens or 0),
        "alias": alias or os.environ.get("ALIAS"),
        "user": os.environ.get("USER"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    record.update(extra)
    line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
    if len(line.encode("utf-8")) > 3500:
        line = line[:3500] + "...\n"  # keep below PIPE_BUF=4096 for atomic append
    out_path = path or _DEFAULT_PATH
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "a") as f:
            f.write(line)
    except Exception as exc:  # noqa: BLE001 -- cost logging must never break the caller
        sys.stderr.write(f"[cost_track] failed to log to {out_path}: {exc}\n")


def extract_usage(response: Any) -> Dict[str, Any]:
    """Pull (input, output, thinking) token counts from an OpenAI-compat ``ChatCompletion``.

    ``output_tokens`` and ``thinking_tokens`` are disjoint, so a call costs ``input * input_price
    + (output + thinking) * output_price`` on every endpoint. Endpoints disagree on whether
    ``completion_tokens`` contains the reasoning: OpenAI-style ones (NVIDIA, OpenAI) count it
    inside and also report it as ``reasoning_tokens``, Gemini counts it outside and reports it
    only through ``total_tokens``. ``total_tokens - prompt_tokens`` is everything generated
    under both conventions, so it is the reference whenever it is present.

    ``usage_missing`` marks a call whose usage reports nothing generated; its cost cannot be
    read from the record."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "usage_missing": True}

    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or 0
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = (getattr(details, "reasoning_tokens", 0) or 0) if details is not None else 0

    if total_tokens > input_tokens:
        generated = total_tokens - input_tokens
        thinking_tokens = reasoning_tokens or max(generated - completion_tokens, 0)
    else:
        # No usable total: assume the OpenAI convention, where completion includes the reasoning.
        generated = max(completion_tokens, reasoning_tokens)
        thinking_tokens = reasoning_tokens
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(generated - thinking_tokens),
        "thinking_tokens": int(thinking_tokens),
        "usage_missing": generated == 0,
    }
