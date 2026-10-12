import dataclasses
import queue
import signal
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from ladder.components.inference.api.client.batch_inference.batch_protocol import (
    BATCH_PROTOCOL_VERSION,
    BatchRequestType,
    MetadataPayload,
    ResponseStatus,
)
from ladder.components.inference.api.client.batch_inference.batch_transport import ZmqBatchTransport
from ladder.components.inference.api.client.batch_inference.model_inference import (
    ModelInference,
    RobotInferenceObservation,
)
from ladder.components.inference.api.client.hlc.vlm.base import VLM
from ladder.components.inference.api.core.serialization import deserialize, serialize
from ladder.core.common.logger import get_logger
from ladder.core.compute.seeder import SEEDER
from ladder.core.databib.config import AutoConfig, AutoConfigurable, Config, ConfigParser, Configurable

LOGGER = get_logger(__name__)

# Sample every Nth batch's INFO log to avoid spamming during long evals.
_BATCH_LOG_INTERVAL = 100
_OUTER_POLL_TIMEOUT_MS = 500


@dataclasses.dataclass
class _BatchItem:
    """A wire observation tagged with its request kind. A batch may mix kinds, so
    ``_process_batch`` partitions by kind and runs each group separately."""

    identity: bytes
    observation: RobotInferenceObservation
    kind: BatchRequestType
    max_new_tokens: int = 64  # HIGH_LEVEL_INFER only


class BatchPolicyServerConfig(Config):
    # The ``ModelInference`` to serve: a ladder checkpoint (VLAMInference), an off-the-shelf VLA
    # adapter (models/external/<model>/server.yaml), ...
    policy_config: AutoConfig
    # Hosted alongside the policy so lanes share one copy. Unset for runs with no VLM role.
    vlm_config: Optional[AutoConfig] = None
    port: int = 5560
    max_batch_size: int = 128
    batch_timeout_ms: int = 30
    # The model samples its flow start state from the global RNG, so an unseeded process draws
    # different noise on every launch and a campaign cannot be repeated. Same field and semantics as
    # run_model's: the launcher supplies the value, and leaving it unset keeps a run unseeded.
    seed: Optional[int] = None
    # If >0, exit cleanly after this many ms with no inference request (once at least
    # one batch has been served). Guards against worker-side hangs.
    max_idle_ms: int = 0


class BatchPolicyServer(Configurable[BatchPolicyServerConfig]):
    """Batches policy inference requests across N workers sharing one GPU model copy.

    METADATA and VLM_COMPLETE are answered inline; inference requests are batched
    (``max_batch_size`` / ``batch_timeout_ms``) and run per kind (LOW_LEVEL_INFER /
    HIGH_LEVEL_INFER).
    """

    def __init__(self, config: BatchPolicyServerConfig) -> None:
        super().__init__(config)
        if self.config.seed is not None:
            SEEDER.seed_everything(self.config.seed)
        self._stop_requested: bool = False
        self._total_batches: int = 0
        self._total_requests: int = 0
        self._last_batch_end: float = 0.0

        self._transport = ZmqBatchTransport(port=config.port)

        LOGGER.info(f"Loading policy {config.policy_config.autoclass} ...")
        self._policy: ModelInference = AutoConfigurable(config.policy_config)

        # Warm eagerly so a VLM that cannot fit fails at startup, not mid-campaign.
        self._vlm: Optional[VLM] = None
        if config.vlm_config is not None:
            LOGGER.info("Loading auxiliary VLM ...")
            self._vlm = AutoConfigurable(config.vlm_config)
            self._vlm.warmup()
            LOGGER.info("Auxiliary VLM ready")
        # Generation runs off the batch loop so a slow completion does not stall inference requests.
        # One worker (generate is not thread-safe); replies are queued because the socket is not either.
        self._vlm_pool = ThreadPoolExecutor(max_workers=1)
        self._vlm_replies: "queue.Queue[Tuple[bytes, bytes]]" = queue.Queue()

        # The policy describes itself; the version is a property of the transport, so it is stamped
        # here rather than in every ModelInference implementation
        self._metadata: MetadataPayload = {
            **self._policy.metadata(),
            "protocol_version": BATCH_PROTOCOL_VERSION,
        }

    def run(self) -> None:
        LOGGER.info("Batch inference server running ...")
        last_batch_ts: float = 0.0

        prev_handlers = self._install_signal_handlers()
        try:
            while not self._stop_requested:
                if self.run_once(timeout_ms=_OUTER_POLL_TIMEOUT_MS) > 0:
                    last_batch_ts = time.monotonic()
                elif self._check_idle_timeout(self.config.max_idle_ms, last_batch_ts):
                    break
        finally:
            self._restore_signal_handlers(prev_handlers)
            LOGGER.info(
                f"Batch server stopped after {self._total_batches} batches / "
                f"{self._total_requests} requests"
            )
            self._transport.close()

    def run_once(self, *, timeout_ms: int) -> int:
        """One iteration of ``run()``. Returns the number of inference requests served
        (0 on poll timeout or METADATA-only)."""
        self._drain_vlm_replies()
        if not self._transport.poll(timeout_ms=timeout_ms):
            self._drain_vlm_replies()
            return 0
        batch = self._collect_batch(self.config.max_batch_size, self.config.batch_timeout_ms)
        if not batch:
            return 0
        self._total_batches += 1
        self._total_requests += len(batch)
        self._process_batch(batch)
        return len(batch)

    def _install_signal_handlers(self) -> Optional[Tuple[object, object]]:
        # signal.signal only works on the main thread; off-main threads (test
        # harnesses) own shutdown via _stop_requested directly.
        if threading.current_thread() is not threading.main_thread():
            return None

        def _on_signal(signum: int, _frame: object) -> None:
            LOGGER.info(f"Received signal {signum}, shutting down after current batch...")
            self._stop_requested = True

        return (
            signal.signal(signal.SIGTERM, _on_signal),
            signal.signal(signal.SIGINT, _on_signal),
        )

    @staticmethod
    def _restore_signal_handlers(handlers: Optional[Tuple[object, object]]) -> None:
        if handlers is None:
            return
        prev_sigterm, prev_sigint = handlers
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)

    def _check_idle_timeout(self, max_idle_ms: int, last_batch_ts: float) -> bool:
        if max_idle_ms <= 0 or last_batch_ts <= 0.0:
            return False
        idle_ms = (time.monotonic() - last_batch_ts) * 1000
        if idle_ms <= max_idle_ms:
            return False
        LOGGER.warn(
            f"No inference request for {idle_ms:.0f}ms > max_idle_ms={max_idle_ms}. "
            f"Workers likely hung; exiting cleanly."
        )
        self._stop_requested = True
        return True

    def _collect_batch(self, max_batch_size: int, batch_timeout_ms: int) -> List[_BatchItem]:
        # Outer poll already fired, so at least one request is available. Keep
        # draining until the batch is full or the deadline elapses.
        batch: List[_BatchItem] = list(self._handle_request())
        deadline = time.monotonic() + batch_timeout_ms / 1000.0
        while len(batch) < max_batch_size:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            if not self._transport.poll(timeout_ms=remaining_ms):
                break
            batch.extend(self._handle_request())
        return batch

    def _process_batch(self, batch: List[_BatchItem]) -> None:
        # A collected batch may mix kinds (HL requests are sparse and arrive among LL ones);
        # LL (action) and HL (text) need separate forwards, so run each kind in its own group.
        for kind in (BatchRequestType.LOW_LEVEL_INFER, BatchRequestType.HIGH_LEVEL_INFER):
            items = [item for item in batch if item.kind == kind]
            if items:
                self._run_kind(items)

    def _run_kind(self, items: List[_BatchItem]) -> None:
        """One forward over same-kind ``items``; one reply per identity (a list under the kind's key)."""
        kind = items[0].kind
        batch_start = time.monotonic()
        try:
            observations = [self._policy.prepare_observation(item.observation) for item in items]
            if kind == BatchRequestType.LOW_LEVEL_INFER:
                response_key = "controls"
                values = [control.to_dict() for control in self._policy.low_level_step(observations)]
            else:
                # One controller per run ⇒ max_new_tokens is uniform across items.
                response_key = "subtasks"
                values = self._policy.high_level_step(
                    observations,
                    max_new_tokens=items[0].max_new_tokens,
                )
        except Exception as exc:  # noqa: BLE001 -- any inference failure must reach every waiting client
            LOGGER.error(f"Batch {self._total_batches} ({kind.value}) inference failed: {exc}")
            for identity in {item.identity for item in items}:
                self._send_error(identity, str(exc))
            return

        now = time.monotonic()
        if self._total_batches == 1 or self._total_batches % _BATCH_LOG_INTERVAL == 0:
            duration_ms = (now - batch_start) * 1000
            idle_ms = (batch_start - self._last_batch_end) * 1000 if self._last_batch_end > 0 else 0.0
            LOGGER.info(
                f"Batch {self._total_batches} ({kind.value}): size={len(items)} "
                f"(clients={len({item.identity for item in items})}), latency={duration_ms:.0f}ms, "
                f"avg={duration_ms / len(items):.1f}ms/obs, idle_before={idle_ms:.0f}ms, "
                f"total_served={self._total_requests}"
            )
        self._last_batch_end = now

        grouped: "OrderedDict[bytes, list]" = OrderedDict()
        for item, value in zip(items, values, strict=True):
            grouped.setdefault(item.identity, []).append(value)
        for identity, env_values in grouped.items():
            self._transport.send(
                identity,
                serialize({"status": ResponseStatus.OK.value, response_key: env_values}),
            )

    def _drain_vlm_replies(self) -> None:
        while True:
            try:
                identity, payload = self._vlm_replies.get_nowait()
            except queue.Empty:
                return
            self._transport.send(identity, payload)

    def _complete_vlm(self, identity: bytes, data: Dict) -> None:
        try:
            text = self._vlm.complete(data["messages"], data["max_tokens"], data["temperature"])
            payload = serialize({"status": ResponseStatus.OK.value, "text": text})
        except Exception as exc:  # noqa: BLE001 -- the waiting client must see the real failure
            LOGGER.error(f"VLM completion failed: {exc}")
            payload = serialize({"status": ResponseStatus.ERROR.value, "message": str(exc)})
        self._vlm_replies.put((identity, payload))

    def _handle_request(self) -> List[_BatchItem]:
        # METADATA/unknown are answered inline; INFER / HIGH_LEVEL_INFER are returned for batching.
        identity, raw = self._transport.receive()
        data = deserialize(raw)
        request_type = data.get("type")
        if request_type in (BatchRequestType.LOW_LEVEL_INFER, BatchRequestType.HIGH_LEVEL_INFER):
            kind = BatchRequestType(request_type)
            max_new_tokens = data.get("max_new_tokens", 64)

            def _item(obs_payload: dict) -> _BatchItem:
                return _BatchItem(
                    identity=identity,
                    observation=RobotInferenceObservation.from_dict(obs_payload),
                    kind=kind,
                    max_new_tokens=max_new_tokens,
                )

            # Accept batched ("observations": list) and singleton ("observation": dict) forms.
            observations_payload = data.get("observations")
            if observations_payload is not None:
                return [_item(obs) for obs in observations_payload]
            observation_payload = data.get("observation")
            if observation_payload is not None:
                return [_item(observation_payload)]
            self._send_error(identity, f"{request_type} request missing 'observation(s)' field")
            return []
        if request_type == BatchRequestType.VLM_COMPLETE:
            if self._vlm is None:
                self._send_error(identity, "server has no vlm_config; cannot serve VLM_COMPLETE")
            else:
                self._vlm_pool.submit(self._complete_vlm, identity, data)
            return []
        if request_type == BatchRequestType.METADATA:
            self._transport.send(
                identity,
                serialize({"status": ResponseStatus.OK.value, "metadata": self._metadata}),
            )
            return []
        LOGGER.warn(f"Unknown request type: {request_type}")
        self._send_error(identity, f"unknown type {request_type}")
        return []

    def _send_error(self, identity: bytes, message: str) -> None:
        self._transport.send(
            identity,
            serialize({"status": ResponseStatus.ERROR.value, "message": message}),
        )


def main() -> None:
    parser = ConfigParser(BatchPolicyServerConfig)
    config: BatchPolicyServerConfig = parser.parse_args()
    server = BatchPolicyServer(config)
    server.run()


if __name__ == "__main__":
    main()
