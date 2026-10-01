import dataclasses
import threading
import time
from typing import Any, Dict, List

import numpy as np
import zmq

from ladder.components.inference.api.client.batch_inference.batch_protocol import (
    BATCH_PROTOCOL_VERSION,
    BatchRequestType,
    MetadataPayload,
    ResponseStatus,
)
from ladder.components.inference.api.client.batch_inference.model_inference import (
    RobotInferenceControl,
    RobotInferenceObservation,
)
from ladder.components.inference.api.core.request_reply import RequestReplyClient, RequestReplyClientConfig
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import AutoConfig, AutoConfigurable, Config, Configurable

LOGGER = get_logger(__name__)


@dataclasses.dataclass
class _ImageSize:
    height: int
    width: int


@dataclasses.dataclass
class _ControlIOConfig:
    past_scalars_sequence_length: int
    past_frames_sequence_length: int


@dataclasses.dataclass
class _RemoteProcessorConfig:
    control_io_config: _ControlIOConfig


class _RemoteProcessor:
    """Stand-in for the real VLAM processor that exposes only the surface the eval
    wrappers (``VLAMRobotInference``, ``BatchPolicyServer``) read from a
    processor: ``image_sizes``, ``config.control_io_config``, and ``resize_image``.
    Resize is a pass-through — the server runs the real resize before inference."""

    def __init__(self, image_sizes: Dict[str, _ImageSize], config: _RemoteProcessorConfig) -> None:
        self.image_sizes = image_sizes
        self.config = config

    def resize_image(self, camera_name: str, image: np.ndarray) -> np.ndarray:
        if camera_name not in self.image_sizes:
            raise KeyError(f"Camera {camera_name!r} is not in the model's cameras")
        return image


class RemoteInferenceClientConfig(Config):
    # Transport is abstracted so tests can inject a fake; production sets this to
    # ZmqRequestReplyClient via autoclass with port/recv_timeout/propagate_timeout.
    client_config: AutoConfig[RequestReplyClientConfig]
    metadata_retries: int = 10


class RemoteInferenceClient(Configurable[RemoteInferenceClientConfig]):
    """Drop-in replacement for ``VLAMInference`` backed by a remote server.

    On construction the client connects to the batch inference server, fetches model
    metadata, and exposes the ``processor`` / ``step()`` / ``delta_mode`` surface that
    ``VLAMRobotInference`` uses against an in-process ``VLAMInference``. The
    ``processor`` is structurally compatible with the real VLAM processor for the
    fields the eval wrappers consume (image sizes, control_io_config, resize_image).
    """

    # Wait between metadata-handshake retries while the server is still loading the model.
    _METADATA_RETRY_SLEEP_SEC = 5

    def __init__(self, config: RemoteInferenceClientConfig) -> None:
        super().__init__(config)
        self._client: RequestReplyClient = AutoConfigurable(config.client_config)
        # One REQ socket, strictly lock-step, and callers may reach it from several threads.
        self._lock = threading.Lock()
        try:
            metadata = self._request_metadata()
        except Exception:
            self.close()
            raise

        # A server that sends no version predates versioning, so it also speaks the flat wire format
        # and cannot parse a nested payload.
        server_version = metadata.get("protocol_version")
        if server_version != BATCH_PROTOCOL_VERSION:
            self.close()
            raise RuntimeError(
                f"Batch server speaks protocol version {server_version}, this client speaks "
                f"{BATCH_PROTOCOL_VERSION}. Redeploy both ends."
            )

        self.processor = _RemoteProcessor(
            image_sizes={
                name: _ImageSize(height=int(hw[0]), width=int(hw[1]))
                for name, hw in metadata["image_sizes"].items()
            },
            config=_RemoteProcessorConfig(
                control_io_config=_ControlIOConfig(
                    past_scalars_sequence_length=int(metadata["past_scalars_sequence_length"]),
                    past_frames_sequence_length=int(metadata["past_frames_sequence_length"]),
                ),
            ),
        )
        self.delta_mode: str = metadata["delta_mode"]
        self.num_effectors: int = int(metadata["num_effectors"])
        LOGGER.info(
            f"Connected to batch inference server: image_sizes={self.processor.image_sizes}, "
            f"delta_mode={self.delta_mode}, num_effectors={self.num_effectors}"
        )

    def _request_metadata(self) -> MetadataPayload:
        """Send a METADATA request, retrying if the server is still loading."""
        max_retries = self.config.metadata_retries
        for attempt in range(max_retries):
            try:
                response = self._client.send({"type": BatchRequestType.METADATA.value})
            except zmq.Again:
                LOGGER.info(
                    f"Metadata request timed out (attempt {attempt + 1}/{max_retries}), "
                    f"server may still be loading model. Retrying ..."
                )
                time.sleep(self._METADATA_RETRY_SLEEP_SEC)
                continue
            if response is None or response.get("status") != ResponseStatus.OK.value:
                raise RuntimeError(f"Metadata request failed: {response}")
            return response["metadata"]

        raise RuntimeError(f"Failed to get metadata after {max_retries} attempts")

    def low_level_step(self, obs_list: List[RobotInferenceObservation]) -> List[RobotInferenceControl]:
        """Match ``VLAMInference.low_level_step``. Sends all observations in one LOW_LEVEL_INFER
        request; the server runs one batched forward and returns the controls list.

        Propagates ``zmq.error.Again`` on server-side timeout.
        """
        if not obs_list:
            return []
        observation_payloads = [obs.to_dict() for obs in obs_list]
        with self._lock:
            response = self._client.send(
                {
                    "type": BatchRequestType.LOW_LEVEL_INFER.value,
                    "observations": observation_payloads,
                }
            )
        if response is None or response.get("status") != ResponseStatus.OK.value:
            raise RuntimeError(f"Inference request failed: {response}")
        controls_payload = response.get("controls")
        if controls_payload is None:
            raise RuntimeError(
                f"Inference response missing 'controls' field (got keys: {list(response.keys())})"
            )
        if len(controls_payload) != len(obs_list):
            raise RuntimeError(
                f"Inference response has {len(controls_payload)} controls but request had "
                f"{len(obs_list)} observations"
            )
        return [RobotInferenceControl.from_dict(control) for control in controls_payload]

    def vlm_complete(self, messages: List[Dict[str, Any]], max_tokens: int, temperature: float) -> str:
        """Chat completion on the server's auxiliary VLM."""
        with self._lock:
            response = self._client.send(
                {
                    "type": BatchRequestType.VLM_COMPLETE.value,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
        if response is None or response.get("status") != ResponseStatus.OK.value:
            raise RuntimeError(f"VLM completion request failed: {response}")
        return response["text"]

    def high_level_step(
        self,
        obs_list: List[RobotInferenceObservation],
        max_new_tokens: int = 64,
    ) -> List[str]:
        """Remote ``VLAMInference.high_level_step``: one HIGH_LEVEL_INFER request → subtask strings."""
        if not obs_list:
            return []
        observation_payloads = [obs.to_dict() for obs in obs_list]
        with self._lock:
            response = self._client.send(
                {
                    "type": BatchRequestType.HIGH_LEVEL_INFER.value,
                    "observations": observation_payloads,
                    "max_new_tokens": max_new_tokens,
                }
            )
        if response is None or response.get("status") != ResponseStatus.OK.value:
            raise RuntimeError(f"High-level inference request failed: {response}")
        subtasks = response.get("subtasks")
        if subtasks is None:
            raise RuntimeError(
                f"High-level response missing 'subtasks' field (got keys: {list(response.keys())})"
            )
        if len(subtasks) != len(obs_list):
            raise RuntimeError(
                f"High-level response has {len(subtasks)} subtasks but request had "
                f"{len(obs_list)} observations"
            )
        return list(subtasks)

    def close(self) -> None:
        client = getattr(self, "_client", None)
        if client is not None:
            client.close()

    def __del__(self) -> None:
        self.close()
