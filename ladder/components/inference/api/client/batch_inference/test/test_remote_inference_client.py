"""Tests for ``RemoteInferenceClient`` over a fake request/reply transport."""

import unittest
import unittest.mock as mock
from typing import Dict, List, Optional, Tuple

import numpy as np

from ladder.components.inference.api.client.batch_inference import (
    remote_inference_client as remote_inference_client_module,
)
from ladder.components.inference.api.client.batch_inference.batch_protocol import (
    BATCH_PROTOCOL_VERSION,
    BatchRequestType,
    ResponseStatus,
)
from ladder.components.inference.api.client.batch_inference.model_inference import (
    ArmsInferenceControl,
    ArmsInferenceObservation,
    HandsInferenceControl,
    HandsInferenceObservation,
    RobotInferenceControl,
    RobotInferenceObservation,
)
from ladder.components.inference.api.client.batch_inference.remote_inference_client import (
    RemoteInferenceClient,
    RemoteInferenceClientConfig,
)
from ladder.core.databib.config import AutoConfig


class FakeRequestReplyClient:
    def __init__(self) -> None:
        self.sent_requests: List[Dict[str, object]] = []
        self._replies: List[Dict[str, object]] = []

    def queue_reply(self, reply: Dict[str, object]) -> None:
        self._replies.append(reply)

    def send(self, request: Dict[str, object]) -> Optional[Dict[str, object]]:
        self.sent_requests.append(request)
        if not self._replies:
            raise AssertionError(f"No queued reply for request {request['type']!r}")
        return self._replies.pop(0)

    def close(self) -> None:
        pass


def _metadata_reply(num_effectors: int = 1, protocol_version: int = BATCH_PROTOCOL_VERSION) -> Dict:
    return {
        "status": ResponseStatus.OK.value,
        "metadata": {
            "image_sizes": {"main": [480, 640], "wrist": [480, 640]},
            "past_scalars_sequence_length": 1,
            "past_frames_sequence_length": 1,
            "delta_mode": "rot_eef_trans_base",
            "num_effectors": num_effectors,
            "protocol_version": protocol_version,
        },
    }


def _control_data(num_effectors: int = 1) -> Dict[str, object]:
    """One control step on the wire: ``[num_future_control_steps, num_effectors, ...]``"""
    return RobotInferenceControl(
        arms=ArmsInferenceControl(
            translation=np.zeros((1, num_effectors, 3), dtype=np.float32),
            rotation=np.tile(np.eye(3, dtype=np.float32), (1, num_effectors, 1, 1)),
        ),
        hands=HandsInferenceControl(gripper_prob=np.zeros((1, num_effectors, 1), dtype=np.float32)),
    ).to_dict()


def _controls_reply(n: int, num_effectors: int = 1) -> Dict[str, object]:
    return {
        "status": ResponseStatus.OK.value,
        "controls": [_control_data(num_effectors) for _ in range(n)],
    }


def _observation(instruction: str, num_effectors: int = 1) -> RobotInferenceObservation:
    return RobotInferenceObservation(
        images={
            "main": np.zeros((1, 480, 640, 3), dtype=np.uint8),
            "wrist": np.zeros((1, 480, 640, 3), dtype=np.uint8),
        },
        instruction=instruction,
        arms=ArmsInferenceObservation(
            ee_pose_translation=np.zeros((1, num_effectors, 3), dtype=np.float32),
            ee_pose_rotation=np.tile(np.eye(3, dtype=np.float32), (1, num_effectors, 1, 1)),
            joint_positions=np.zeros((1, num_effectors, 7), dtype=np.float32),
        ),
        hands=HandsInferenceObservation(gripper_width=np.zeros((1, num_effectors, 1), dtype=np.float32)),
        timestamps=np.zeros((1,), dtype=np.float64),
        dataset_name="bridge_orig",
    )


def _connect(metadata_reply: Dict[str, object]) -> Tuple[RemoteInferenceClient, FakeRequestReplyClient]:
    """Run the METADATA handshake against a fake transport and hand back both ends."""
    fake_client = FakeRequestReplyClient()
    fake_client.queue_reply(metadata_reply)
    config = RemoteInferenceClientConfig(
        client_config=AutoConfig(
            autoclass="ladder.components.inference.api.core.request_reply.ZmqRequestReplyClient",
            ip="localhost",
            port=0,
        ),
        metadata_retries=1,
    )
    with mock.patch.object(remote_inference_client_module, "AutoConfigurable", return_value=fake_client):
        return RemoteInferenceClient(config), fake_client


class TestRemoteInferenceClient(unittest.TestCase):
    def setUp(self) -> None:
        self.client, self.fake_client = _connect(_metadata_reply())

    def test_metadata_handshake_populates_processor_and_delta_mode(self) -> None:
        image_sizes = self.client.processor.image_sizes
        self.assertEqual(set(image_sizes.keys()), {"main", "wrist"})
        for size in image_sizes.values():
            self.assertEqual((size.height, size.width), (480, 640))
        control_io_config = self.client.processor.config.control_io_config
        self.assertEqual(control_io_config.past_scalars_sequence_length, 1)
        self.assertEqual(control_io_config.past_frames_sequence_length, 1)
        self.assertEqual(self.client.delta_mode, "rot_eef_trans_base")
        self.assertEqual(self.client.num_effectors, 1)
        self.assertEqual(self.fake_client.sent_requests[0]["type"], BatchRequestType.METADATA.value)

    def test_step_round_trips_observations_through_remote(self) -> None:
        self.fake_client.queue_reply(_controls_reply(2))

        controls = self.client.low_level_step([_observation("a"), _observation("b")])

        self.assertEqual(len(controls), 2)
        for control in controls:
            self.assertIsInstance(control, RobotInferenceControl)
            self.assertEqual(control.arms.translation.shape, (1, 1, 3))
            self.assertEqual(control.arms.rotation.shape, (1, 1, 3, 3))
        infer_requests = [
            r
            for r in self.fake_client.sent_requests
            if r.get("type") == BatchRequestType.LOW_LEVEL_INFER.value
        ]
        self.assertEqual(len(infer_requests), 1)
        instructions = [obs["instruction"] for obs in infer_requests[0]["observations"]]
        self.assertEqual(instructions, ["a", "b"])

    def test_step_raises_when_server_returns_error(self) -> None:
        self.fake_client.queue_reply({"status": ResponseStatus.ERROR.value, "message": "boom"})
        with self.assertRaises(RuntimeError):
            self.client.low_level_step([_observation("x")])

    def test_step_keeps_the_effector_axis_and_the_per_side_values(self) -> None:
        client, fake_client = _connect(_metadata_reply(num_effectors=2))
        self.assertEqual(client.num_effectors, 2)
        control = _control_data(num_effectors=2)
        # Distinct per side, so neither a squeeze nor a broadcast of one side can pass
        control["arms"]["translation"] = np.arange(2 * 3, dtype=np.float32).reshape(1, 2, 3)
        fake_client.queue_reply({"status": ResponseStatus.OK.value, "controls": [control]})

        [returned] = client.low_level_step([_observation("a", num_effectors=2)])

        self.assertEqual(returned.num_effectors, 2)
        self.assertEqual(returned.arms.rotation.shape, (1, 2, 3, 3))
        self.assertEqual(returned.hands.gripper_prob.shape, (1, 2, 1))
        self.assertTrue(np.array_equal(returned.arms.translation, control["arms"]["translation"]))

        [request] = [
            r for r in fake_client.sent_requests if r.get("type") == BatchRequestType.LOW_LEVEL_INFER.value
        ]
        [observation_payload] = request["observations"]
        self.assertEqual(np.asarray(observation_payload["arms"]["ee_pose_translation"]).shape, (1, 2, 3))
        self.assertEqual(np.asarray(observation_payload["hands"]["gripper_width"]).shape, (1, 2, 1))

    def test_a_server_on_another_protocol_version_is_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            _connect(_metadata_reply(protocol_version=BATCH_PROTOCOL_VERSION + 1))

    def test_a_server_that_sends_no_protocol_version_is_refused(self) -> None:
        """A pre-versioning server also speaks the flat wire format, so its metadata is not usable."""
        reply = _metadata_reply()
        del reply["metadata"]["protocol_version"]
        with self.assertRaises(RuntimeError):
            _connect(reply)

    def test_resize_image_validates_camera_and_passes_through(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertIs(self.client.processor.resize_image("main", image), image)
        with self.assertRaises(KeyError):
            self.client.processor.resize_image("not_a_camera", image)


if __name__ == "__main__":
    unittest.main()
