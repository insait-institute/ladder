"""Tests for ``BatchPolicyServer``: a fake ``ModelInference`` behind an in-memory transport."""

import unittest
import unittest.mock as mock
from typing import Dict, List, Tuple

import numpy as np

from ladder.components.inference.api.client.batch_inference import (
    batch_policy_server as batch_policy_server_module,
)
from ladder.components.inference.api.client.batch_inference.batch_policy_server import (
    BatchPolicyServer,
    BatchPolicyServerConfig,
)
from ladder.components.inference.api.client.batch_inference.batch_protocol import (
    BatchRequestType,
    PolicyMetadataPayload,
    ResponseStatus,
)
from ladder.components.inference.api.client.batch_inference.batch_transport import BatchTransport
from ladder.components.inference.api.client.batch_inference.model_inference import (
    ArmsInferenceControl,
    ArmsInferenceObservation,
    HandsInferenceControl,
    HandsInferenceObservation,
    ModelInference,
    RobotInferenceControl,
    RobotInferenceObservation,
)
from ladder.components.inference.api.core.serialization import deserialize, serialize
from ladder.core.databib.config import AutoConfig, Config

_CAMERAS = ("main", "wrist")


class FakePolicyConfig(Config):
    # Non-empty makes every forward raise with this message.
    fail_with: str = ""
    num_effectors: int = 1


class FakePolicy(ModelInference[FakePolicyConfig]):
    """A served policy with no model behind it. ``forwards`` records the size of each batched
    forward, which is how a test tells one batch of N from N batches of one."""

    forwards: List[int] = []

    def metadata(self) -> PolicyMetadataPayload:
        return {
            "image_sizes": {camera: [480, 640] for camera in _CAMERAS},
            "past_scalars_sequence_length": 1,
            "past_frames_sequence_length": 1,
            "delta_mode": "rot_eef_trans_base",
            "num_effectors": self.config.num_effectors,
        }

    def prepare_observation(self, observation: RobotInferenceObservation) -> RobotInferenceObservation:
        return observation

    def low_level_step(self, observations: List[RobotInferenceObservation]) -> List[RobotInferenceControl]:
        FakePolicy.forwards.append(len(observations))
        if self.config.fail_with:
            raise RuntimeError(self.config.fail_with)
        return [
            RobotInferenceControl(
                arms=ArmsInferenceControl(
                    translation=np.zeros((1, observation.num_effectors, 3), dtype=np.float32),
                    rotation=np.tile(np.eye(3, dtype=np.float32), (1, observation.num_effectors, 1, 1)),
                ),
                hands=HandsInferenceControl(
                    gripper_prob=np.zeros((1, observation.num_effectors, 1), dtype=np.float32)
                ),
            )
            for observation in observations
        ]


class FakeBatchTransport(BatchTransport):
    """In-memory transport. Caller pushes serialized requests; server pops + replies."""

    def __init__(self) -> None:
        self._inbox: List[Tuple[bytes, bytes]] = []
        self.replies: Dict[bytes, List[bytes]] = {}

    def push(self, identity: bytes, payload: Dict[str, object]) -> None:
        self._inbox.append((identity, serialize(payload)))

    def reply_for(self, identity: bytes) -> Dict[str, object]:
        [raw] = self.replies[identity]
        return deserialize(raw)

    def poll(self, timeout_ms: int) -> bool:
        del timeout_ms
        return bool(self._inbox)

    def receive(self) -> Tuple[bytes, bytes]:
        return self._inbox.pop(0)

    def send(self, identity: bytes, payload: bytes) -> None:
        self.replies.setdefault(identity, []).append(payload)

    def close(self) -> None:
        pass


def _observation_payload(
    instruction: str = "pick up the red block", num_effectors: int = 1
) -> Dict[str, object]:
    """A wire observation as a client sends it: one history step, ``[H, num_effectors, D]`` scalars."""
    prng = np.random.RandomState(0)
    return RobotInferenceObservation(
        images={camera: prng.randint(0, 256, size=(1, 480, 640, 3)).astype(np.uint8) for camera in _CAMERAS},
        instruction=instruction,
        arms=ArmsInferenceObservation(
            ee_pose_translation=np.zeros((1, num_effectors, 3), dtype=np.float32),
            ee_pose_rotation=np.tile(np.eye(3, dtype=np.float32), (1, num_effectors, 1, 1)),
            joint_positions=np.zeros((1, num_effectors, 7), dtype=np.float32),
        ),
        hands=HandsInferenceObservation(gripper_width=np.zeros((1, num_effectors, 1), dtype=np.float32)),
        timestamps=np.zeros((1,), dtype=np.float64),
        dataset_name="bridge_orig",
    ).to_dict()


class TestBatchPolicyServer(unittest.TestCase):
    def setUp(self) -> None:
        FakePolicy.forwards.clear()
        self.transport = FakeBatchTransport()

    def _build_server(self, fail_with: str = "", num_effectors: int = 1) -> BatchPolicyServer:
        config = BatchPolicyServerConfig(
            policy_config=AutoConfig(
                autoclass=f"{__name__}.FakePolicy",
                fail_with=fail_with,
                num_effectors=num_effectors,
            ),
            max_batch_size=4,
            batch_timeout_ms=50,
        )
        with mock.patch.object(batch_policy_server_module, "ZmqBatchTransport", return_value=self.transport):
            return BatchPolicyServer(config)

    def test_metadata_request_is_answered_inline(self) -> None:
        server = self._build_server()
        self.transport.push(b"client", {"type": BatchRequestType.METADATA})

        server.run_once(timeout_ms=0)

        reply = self.transport.reply_for(b"client")
        self.assertEqual(reply["status"], ResponseStatus.OK.value)
        self.assertEqual(reply["metadata"]["delta_mode"], "rot_eef_trans_base")
        self.assertEqual(FakePolicy.forwards, [])

    def test_infer_requests_from_multiple_clients_are_batched(self) -> None:
        server = self._build_server()
        identities = [b"worker-0", b"worker-1", b"worker-2"]
        for identity in identities:
            self.transport.push(
                identity, {"type": BatchRequestType.LOW_LEVEL_INFER, "observations": [_observation_payload()]}
            )

        server.run_once(timeout_ms=0)

        self.assertEqual(FakePolicy.forwards, [len(identities)])
        for identity in identities:
            reply = self.transport.reply_for(identity)
            self.assertEqual(reply["status"], ResponseStatus.OK.value)
            self.assertEqual(len(reply["controls"]), 1)
            self.assertIn("translation", reply["controls"][0]["arms"])

    def test_bimanual_observation_keeps_its_effector_axis_across_the_wire(self) -> None:
        """``FakePolicy`` sizes its control from the observation it is handed, so a ``[T, 2, ...]``
        reply is proof that the effector axis survived deserialization on the server."""
        server = self._build_server(num_effectors=2)
        self.transport.push(
            b"worker",
            {
                "type": BatchRequestType.LOW_LEVEL_INFER,
                "observations": [_observation_payload(num_effectors=2)],
            },
        )

        server.run_once(timeout_ms=0)

        reply = self.transport.reply_for(b"worker")
        self.assertEqual(reply["status"], ResponseStatus.OK.value)
        [control] = reply["controls"]
        self.assertEqual(np.asarray(control["arms"]["translation"]).shape, (1, 2, 3))
        self.assertEqual(np.asarray(control["arms"]["rotation"]).shape, (1, 2, 3, 3))
        self.assertEqual(np.asarray(control["hands"]["gripper_prob"]).shape, (1, 2, 1))

    def test_unknown_request_type_returns_error(self) -> None:
        server = self._build_server()
        self.transport.push(b"bad", {"type": "GARBAGE"})

        server.run_once(timeout_ms=0)

        reply = self.transport.reply_for(b"bad")
        self.assertEqual(reply["status"], ResponseStatus.ERROR.value)

    def test_inference_failure_returns_error_to_each_client_in_batch(self) -> None:
        server = self._build_server(fail_with="boom")
        for identity in (b"a", b"b"):
            self.transport.push(
                identity, {"type": BatchRequestType.LOW_LEVEL_INFER, "observation": _observation_payload()}
            )

        server.run_once(timeout_ms=0)

        for identity in (b"a", b"b"):
            reply = self.transport.reply_for(identity)
            self.assertEqual(reply["status"], ResponseStatus.ERROR.value)
            self.assertIn("boom", reply["message"])


if __name__ == "__main__":
    unittest.main()
