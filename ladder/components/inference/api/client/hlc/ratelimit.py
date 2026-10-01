import contextlib
import dataclasses
import fcntl
import hashlib
import json
import os
import random
import tempfile
import time
from typing import Any, Dict, Iterator, Optional

from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)

_DECREASE = 0.5
_BACKOFF_FACTOR = 4.0
_MAX_COOLDOWN_S = 60.0
_TRIP_AFTER = 3
_DECREASE_WINDOW_S = 2.0
_PROBE_TIMEOUT_S = 60.0
_PROBE_POLL_S = 1.0
_SLOW_WAIT_S = 30.0
_FLOOR_RPS = 0.01


@dataclasses.dataclass(frozen=True)
class LimiterState:
    """What the node currently believes about one endpoint."""

    rate: float
    blocked_for: float
    consecutive_refusals: int


class EndpointLimiter:
    """Node-wide adaptive rate limit and circuit breaker for one HTTP endpoint.

    The limit is scoped to the source IP, so the unit is the node: several eval cells share a box,
    each in its own process, and it is their sum the endpoint sees. State therefore lives in a
    node-local file. The permitted rate starts at ``max_rps``, halves on a refusal and edges back up
    on success; a sustained refusal opens a node-wide breaker that admits one probe while recovering.
    """

    def __init__(
        self,
        key: str,
        max_rps: float,
        burst: float = 0.0,
        min_rps: float = 1.0,
        cooldown_s: float = 5.0,
        decrease_window_s: float = _DECREASE_WINDOW_S,
    ) -> None:
        self._max_rps = max(max_rps, _FLOOR_RPS)
        self._min_rps = max(min(min_rps, self._max_rps), _FLOOR_RPS)
        self._burst = burst if burst > 0 else max(1.0, self._max_rps)
        self._cooldown_s = max(cooldown_s, 0.0)
        self._decrease_window_s = max(decrease_window_s, 0.0)
        self._increase = self._max_rps / 10.0
        digest = hashlib.sha1(key.encode()).hexdigest()[:12]
        self._path = os.path.join(tempfile.gettempdir(), f"vlm_ratelimit_{digest}.json")

    def _fresh(self) -> Dict[str, Any]:
        return {
            "rate": self._max_rps,
            "tokens": self._burst,
            "updated": time.time(),
            "blocked_until": 0.0,
            "level": 0,
            "probing": 0.0,
            "refusals": 0,
            "decreased": 0.0,
        }

    @contextlib.contextmanager
    def _locked(self) -> Iterator[Any]:
        handle = open(self._path, "a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield handle
        finally:
            handle.close()

    def _read(self, handle: Any) -> Dict[str, Any]:
        handle.seek(0)
        try:
            state = json.loads(handle.read() or "{}")
        except ValueError:
            state = {}
        base = self._fresh()
        base.update({k: v for k, v in state.items() if k in base})
        return base

    def _write(self, handle: Any, state: Dict[str, Any]) -> None:
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(state))
        handle.flush()

    def _peek(self) -> Dict[str, Any]:
        try:
            with open(self._path) as handle:
                return {**self._fresh(), **json.loads(handle.read() or "{}")}
        except (OSError, ValueError):
            return self._fresh()

    def state(self) -> LimiterState:
        """Snapshot for logging and health checks. Never blocks, never writes."""
        raw = self._peek()
        return LimiterState(
            rate=float(raw["rate"]),
            blocked_for=max(0.0, float(raw["blocked_until"]) - time.time()),
            consecutive_refusals=int(raw["level"]),
        )

    @property
    def state_path(self) -> str:
        """Node-local file this endpoint's state lives in."""
        return self._path

    def _try_acquire(self) -> float:
        with self._locked() as handle:
            state = self._read(handle)
            now = time.time()
            state["tokens"] = min(
                self._burst, state["tokens"] + max(0.0, now - state["updated"]) * state["rate"]
            )
            state["updated"] = now

            if now < state["blocked_until"]:
                wait = state["blocked_until"] - now
                self._write(handle, state)
                return wait

            if state["level"] > 0:
                probing = state["probing"]
                if probing and now - probing < _PROBE_TIMEOUT_S:
                    self._write(handle, state)
                    return _PROBE_POLL_S
                state["probing"] = now
                self._write(handle, state)
                return 0.0

            if state["tokens"] >= 1.0:
                state["tokens"] -= 1.0
                self._write(handle, state)
                return 0.0

            wait = (1.0 - state["tokens"]) / max(state["rate"], _FLOOR_RPS)
            self._write(handle, state)
            return wait

    @contextlib.contextmanager
    def hold(self) -> Iterator[None]:
        """Block until this node may send, then yield. Report the outcome via succeed/penalise."""
        started = time.time()
        while True:
            wait = self._try_acquire()
            if wait <= 0:
                break
            time.sleep(min(wait, 5.0) * (0.75 + random.random() * 0.5))
        waited = time.time() - started
        if waited > _SLOW_WAIT_S:
            LOGGER.warn(
                f"VLM rate limiter held a call for {waited:.1f}s; permitted rate is now "
                f"{self.state().rate:.2f}/s -- raise max_rps if this is common"
            )
        yield

    def succeed(self) -> None:
        """A completed call: close any open breaker and edge the permitted rate back up."""
        state = self._peek()
        if state["level"] == 0 and state["rate"] >= self._max_rps and not state.get("refusals"):
            return
        with self._locked() as handle:
            state = self._read(handle)
            if state["level"] > 0:
                LOGGER.info(f"VLM endpoint answering again; breaker closed at " f"{state['rate']:.2f}/s")
            state["level"] = 0
            state["probing"] = 0.0
            state["blocked_until"] = 0.0
            state["refusals"] = 0
            state["rate"] = min(self._max_rps, state["rate"] + self._increase)
            self._write(handle, state)

    def penalise(self, retry_after_s: Optional[float] = None) -> None:
        """A refusal: slow the node down, and stop it sending once refusals are sustained."""
        with self._locked() as handle:
            state = self._read(handle)
            now = time.time()
            state["refusals"] = int(state["refusals"]) + 1

            if now - state["decreased"] > self._decrease_window_s:
                state["rate"] = max(self._min_rps, state["rate"] * _DECREASE)
                state["decreased"] = now

            if state["refusals"] < _TRIP_AFTER or now < state["blocked_until"]:
                self._write(handle, state)
                return

            state["level"] = int(state["level"]) + 1
            cooldown = (
                retry_after_s
                if retry_after_s is not None
                else min(self._cooldown_s * _BACKOFF_FACTOR ** (state["level"] - 1), _MAX_COOLDOWN_S)
            )
            state["blocked_until"] = max(state["blocked_until"], now + cooldown)
            state["probing"] = 0.0
            state["tokens"] = 0.0
            state["refusals"] = 0
            self._write(handle, state)
            LOGGER.warn(
                f"VLM endpoint refusing persistently (breaker level {state['level']}); node "
                f"quiet for {cooldown:.0f}s, permitted rate now {state['rate']:.2f}/s"
            )
