"""Wire protocol types for ``BatchPolicyServer`` and its clients."""

import enum
from typing import Dict, List, TypedDict

# Bumped whenever the wire format changes shape. Both ends compare it at the METADATA handshake, so a
# version-skewed deployment fails there rather than as a malformed payload in the middle of an eval.
BATCH_PROTOCOL_VERSION = 1


class BatchRequestType(str, enum.Enum):
    """Kind of request a client sends over the batch-inference ZMQ socket. Distinct from
    the env-server :class:`ladder.components.inference.api.core.requests.RequestType`."""

    METADATA = "METADATA"
    LOW_LEVEL_INFER = "LOW_LEVEL_INFER"  # action-chunk inference (ModelInference.low_level_step)
    HIGH_LEVEL_INFER = "HIGH_LEVEL_INFER"  # subtask text generation (ModelInference.high_level_step)
    VLM_COMPLETE = "VLM_COMPLETE"  # chat completion on the server's auxiliary VLM


class ResponseStatus(str, enum.Enum):
    """Status field on every server response."""

    OK = "ok"
    ERROR = "error"


class PolicyMetadataPayload(TypedDict):
    """The transport stamps its version on top, so a policy never has to know about the wire

    ``image_sizes`` is ``{camera_name: [height, width]}``. The server normalizes
    processor-specific tuple formats to this flat shape before sending. Clients use
    it to know which cameras the model accepts; the actual resize happens server-side
    via the real processor.

    ``num_effectors`` is the embodiment the policy was trained for: 1 for a single arm, 2 bi-manual.
    """

    image_sizes: Dict[str, List[int]]
    past_scalars_sequence_length: int
    past_frames_sequence_length: int
    delta_mode: str
    num_effectors: int


class MetadataPayload(PolicyMetadataPayload):
    """Server metadata sent in response to ``METADATA`` requests"""

    protocol_version: int
