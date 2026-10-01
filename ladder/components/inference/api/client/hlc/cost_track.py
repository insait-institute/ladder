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
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
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
    """Pull (input, output, thinking) token counts from an OpenAI-compat ``ChatCompletion``,
    returning zeros for any field the response omits."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}

    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    # OpenAI / Gemini-thinking style: usage.completion_tokens_details.reasoning_tokens
    details = getattr(usage, "completion_tokens_details", None)
    thinking_tokens = 0
    if details is not None:
        thinking_tokens = getattr(details, "reasoning_tokens", 0) or 0
    # Fallback: some gateways omit the reasoning_tokens breakdown and report only total_tokens;
    # recover thinking as the remainder so it isn't silently logged as 0.
    if thinking_tokens == 0:
        total_tokens = getattr(usage, "total_tokens", 0) or 0
        remainder = total_tokens - input_tokens - output_tokens
        if remainder > 0:
            thinking_tokens = remainder
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "thinking_tokens": int(thinking_tokens),
    }
