import dataclasses
import functools
import os
import pathlib
import unittest
import unittest.mock as mock

import numpy as np
import roma
import scipy.spatial.transform
import torch
import yaml

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
from ladder.components.inference.api.core.action import DualManipulatorAction, ManipulatorAction, RobotAction
from ladder.components.inference.api.core.observation import (
    DualManipulatorState,
    ManipulatorState,
    RobotObservation,
)
from ladder.components.inference.models.vlams.vlam_inference import import_py_module
from ladder.components.inference.models.vlams.vlam_versions import has_effector_axis_processor_api
from ladder.core.compute.checksum import calculate_data_checksum
from ladder.core.compute.seeder import SEEDER
from ladder.core.databib.config import AutoConfig, AutoConfigurable
from ladder.core.databib.dataclasses import Dataclass, dataclass

# Each env keeps its exported bundle in its own package, so the bazel target hands us the path to the
# one it can load, relative to the runfiles root (the working directory under `bazel test`). Resolving
# the checkpoint follows the runfiles symlink back to the source tree, which matters because
# rules_python drops an __init__.py into every runfiles directory - including the `hf_export/` that
# `make_model` scans for exactly one exported model
MOCK_CKPT_PATH = pathlib.Path(os.environ["VLAM_TEST_MOCK_SESSION"], "checkpoints", "mock.ckpt").resolve()
MOCK_SESSION_PATH = MOCK_CKPT_PATH.parent.parent
[MODEL_EXPORT_PATH] = (MOCK_SESSION_PATH / "hf_export").glob("*/src")
[MODELING_PY_PATH] = MODEL_EXPORT_PATH.glob("modeling_*.py")
ENV_VERSION: str = yaml.safe_load((MODEL_EXPORT_PATH / "env_metadata.yaml").read_text())["env_version"]

EXPECTED_ACTION_CHECKSUMS = {
    "v0": {
        "translation": [
            1.2748956680297852,
            0.909843921661377,
            1.5409709215164185,
            1.8646981716156006,
            2.886200189590454,
        ],
        "rotation": [
            4.662650108337402,
            4.121097087860107,
            4.548916816711426,
            4.146217346191406,
            4.775195598602295,
        ],
        "gripper": [0.0, 0.0, 0.0, 1.0, 0.0],
    },
    "v1": {
        "translation": [
            1.2649166584014893,
            0.899864912033081,
            1.530992031097412,
            1.8547192811965942,
            2.876221179962158,
        ],
        "rotation": [
            4.6650824546813965,
            4.144142150878906,
            4.535965919494629,
            4.13247537612915,
            4.772215366363525,
        ],
        "gripper": [0.0, 0.0, 0.0, 1.0, 0.0],
    },
    "v3": {
        "translation": [
            1.2649166584014893,
            0.899864912033081,
            1.530992031097412,
            1.8547192811965942,
            2.876221179962158,
        ],
        "rotation": [
            4.6650824546813965,
            4.144142150878906,
            4.535965919494629,
            4.13247537612915,
            4.772215366363525,
        ],
        "gripper": [0.0, 0.0, 0.0, 1.0, 0.0],
    },
    "v4": {
        "translation": [
            1.2649166584014893,
            0.899864912033081,
            1.530992031097412,
            1.8547192811965942,
            2.876221179962158,
        ],
        "rotation": [
            4.6650824546813965,
            4.144142150878906,
            4.535965919494629,
            4.13247537612915,
            4.772215366363525,
        ],
        "gripper": [0.0, 0.0, 0.0, 1.0, 0.0],
    },
    "v5": {
        "translation": [
            1.2649166584014893,
            0.899864912033081,
            1.530992031097412,
            1.8547192811965942,
            2.876221179962158,
        ],
        "rotation": [
            4.6650824546813965,
            4.144142150878906,
            4.535965919494629,
            4.13247537612915,
            4.772215366363525,
        ],
        "gripper": [0.0, 0.0, 0.0, 1.0, 0.0],
    },
}


# Per-effector action checksums for the bi-manual path, pinned the same way as the single-arm ones
# above. `mock_generate` offsets effector 1 by half of every range, which is why the two sides differ
# and why exactly one of the two gripper probabilities clears the 0.5 threshold
EXPECTED_DUAL_ACTION_CHECKSUMS = {
    "v5": {
        "right_translation": [
            1.2649133205413818,
            1.246534824371338,
            0.42910346388816833,
            1.8369500637054443,
            1.1348881721496582,
        ],
        "right_rotation": [
            4.6650824546813965,
            4.542308807373047,
            4.824953556060791,
            4.674100399017334,
            3.728771686553955,
        ],
        "right_gripper": [0.0, 0.0, 0.0, 1.0, 0.0],
        "left_translation": [
            1.1580514907836914,
            1.444793701171875,
            2.047445774078369,
            0.6427470445632935,
            1.9279930591583252,
        ],
        "left_rotation": [
            4.102081298828125,
            4.4273457527160645,
            4.776448726654053,
            4.448484420776367,
            4.695796489715576,
        ],
        "left_gripper": [1.0, 1.0, 1.0, 0.0, 1.0],
    },
}


@dataclass
class FlatRoboticsOutput(Dataclass):
    translation: torch.Tensor  # [B, num_timesteps, 3]
    rotation: torch.Tensor  # [B, num_timesteps, 3 | 4 | 9]
    gripper: torch.Tensor  # [B, num_timesteps, 1]

    def __getitem__(self, index):
        # Minimal batch-dim slicing so split_dataclass can operate on this fixture.
        # TensorDataclass normally provides this in production code.
        return type(self)(
            translation=self.translation[index],
            rotation=self.rotation[index],
            gripper=self.gripper[index],
        )


@dataclass
class ArmsOutput(Dataclass):
    translation: torch.Tensor  # [B, num_timesteps, num_effectors, 3]
    rotation: torch.Tensor  # [B, num_timesteps, num_effectors, 3 | 4 | 9]

    def __getitem__(self, index):
        return type(self)(translation=self.translation[index], rotation=self.rotation[index])


@dataclass
class HandsOutput(Dataclass):
    gripper_width: torch.Tensor  # [B, num_timesteps, num_effectors, 1]

    def __getitem__(self, index):
        return type(self)(gripper_width=self.gripper_width[index])


@dataclass
class RoboticsOutput(Dataclass):
    """The arms / hands output shape that bundles from EFFECTOR_AXIS_ENV_VERSION onwards consume"""

    arms_output: ArmsOutput
    hands_output: HandsOutput

    def __getitem__(self, index):
        return type(self)(arms_output=self.arms_output[index], hands_output=self.hands_output[index])


class TestVLAMRobotInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Loading the PaliGemma backbone is the expensive step (~75s on CPU); do it once
        # for the whole TestCase and let each test share the same policy instance.
        cls.py_modeling = import_py_module(MODELING_PY_PATH, MODEL_EXPORT_PATH.parent.name)
        policy_config = AutoConfig(
            autoclass="ladder.components.inference.models.vlams.vlam_robot_inference.VLAMRobotInference",
            env_name="droid",
            controls_on_targets=True,
            inference_config={
                "autoclass": "ladder.components.inference.models.vlams.vlam_inference.VLAMInference",
                "ckpt_path": str(MOCK_CKPT_PATH),
                "dtype": "float32",
                "device": "cpu",
            },
            controller_config={
                "autoclass": (
                    "ladder.components.inference.models.vlams.controllers.action_chunking."
                    "ActionChunkingController"
                ),
                "chunk_size": 4,
            },
        )
        with mock.patch.object(cls.py_modeling.MainModel, "load_state_dict", autospec=True):
            cls.policy = AutoConfigurable(policy_config)

    def setUp(self) -> None:
        SEEDER.reset()
        SEEDER.seed_everything(42)
        self.prng = SEEDER.new_prng_np()
        self.policy.reset()

    def test_predict_actions_runs_batched_inference_with_batch_size_1(self):
        """Drives inference through the full policy.predict_actions path with a single-element
        batch, routing through VLAMInference.step at batch_size=1."""
        with mock.patch.object(
            self.policy.inference.model,
            "generate",
            side_effect=functools.partial(mock_generate, num_effectors=1),
        ):
            instruction = "Move the red block on top of the green block."
            actions = []

            for _ in range(5):
                observation = self._make_observation()
                actions.append(self.policy.predict_actions([observation], [instruction])[0])

        actual_action_checksums = {
            "translation": [
                calculate_data_checksum(action.arm_action.translation, hashed=False) for action in actions
            ],
            "rotation": [
                calculate_data_checksum(action.arm_action.rotation, hashed=False) for action in actions
            ],
            "gripper": [float(action.gripper_action) for action in actions],
        }
        for key, expected_values in EXPECTED_ACTION_CHECKSUMS[ENV_VERSION].items():
            with self.subTest(component=key):
                self.assertTrue(
                    np.all(np.isclose(actual_action_checksums[key], expected_values)),
                    f"Checksums for {key} do not match. "
                    f"Actual: {actual_action_checksums[key]}, Expected: {expected_values}",
                )

    def _serving_effectors(self, num_effectors: int):
        """The mock bundle's processor is configured single-arm, so a bi-manual test has to declare what
        the model serves as well as mock what it emits, or the topology assert rejects the observation"""
        return mock.patch.object(
            type(self.policy.inference),
            "num_effectors",
            new_callable=mock.PropertyMock,
            return_value=num_effectors,
        )

    def test_predict_actions_dual_arm_checksums(self):
        """The bi-manual counterpart of the single-arm action checksums: every effector is pinned
        separately, so a change that swaps or shares a side shows up as a moved checksum."""
        if ENV_VERSION not in EXPECTED_DUAL_ACTION_CHECKSUMS:
            self.skipTest(f"No bi-manual model was trained on {ENV_VERSION}")

        with self._serving_effectors(2), mock.patch.object(
            self.policy.inference.model,
            "generate",
            side_effect=functools.partial(mock_generate, num_effectors=2),
        ):
            instruction = "Move the red block on top of the green block."
            actions = []

            for _ in range(5):
                observation = self._make_observation(dual=True)
                actions.append(self.policy.predict_actions([observation], [instruction])[0])

        actual_action_checksums = {}
        for side_name in ("right", "left"):
            sides = [getattr(action, side_name) for action in actions]
            actual_action_checksums[f"{side_name}_translation"] = [
                calculate_data_checksum(side.arm_action.translation, hashed=False) for side in sides
            ]
            actual_action_checksums[f"{side_name}_rotation"] = [
                calculate_data_checksum(side.arm_action.rotation, hashed=False) for side in sides
            ]
            actual_action_checksums[f"{side_name}_gripper"] = [float(side.gripper_action) for side in sides]

        for key, expected_values in EXPECTED_DUAL_ACTION_CHECKSUMS[ENV_VERSION].items():
            with self.subTest(component=key):
                self.assertTrue(
                    np.all(np.isclose(actual_action_checksums[key], expected_values)),
                    f"Checksums for {key} do not match. "
                    f"Actual: {actual_action_checksums[key]}, Expected: {expected_values}",
                )

    def test_predict_actions_emits_a_dual_action_for_a_dual_state(self):
        """A DualManipulatorState in, a DualManipulatorAction out, with each arm driven by its own
        slice of the control chunk rather than both sharing one."""
        if not has_effector_axis_processor_api(ENV_VERSION):
            self.skipTest(f"{ENV_VERSION} bundles have no effector axis")

        with self._serving_effectors(2), mock.patch.object(
            self.policy.inference.model,
            "generate",
            side_effect=functools.partial(mock_generate, num_effectors=2),
        ):
            # Both arms observe the SAME state, so the control chunk is the only thing that can make
            # the two actions differ. With per-arm states they would differ even if one arm's control
            # were applied to both, and the assertion below would prove nothing
            observation = self._make_observation(dual=True, shared_state=True)
            action = self.policy.predict_actions([observation], ["pick up the block"])[0]

        self.assertIsInstance(action, DualManipulatorAction)
        for side in (action.right, action.left):
            self.assertIsInstance(side, ManipulatorAction)
            self.assertEqual(side.arm_action.translation.shape, (3,))
            self.assertEqual(side.arm_action.rotation.shape, (3, 3))

        # mock_generate offsets every effector, so identical arms mean one side's control was applied
        # to both
        self.assertFalse(np.allclose(action.right.arm_action.translation, action.left.arm_action.translation))

    def test_predict_actions_refuses_a_topology_mismatch(self):
        """A single-arm robot against a model that serves two effectors is refused before the forward,
        rather than driving the one arm with effector 0 and silently discarding the other control"""
        with self._serving_effectors(2):
            with self.assertRaises(AssertionError):
                self.policy.predict_actions([self._make_observation()], ["pick up the block"])

    def test_step_matches_single_observation_calls(self):
        """Run VLAMInference.step with multiple observations and verify each output
        matches the one produced by calling step with a single observation."""
        vlam_observations = [
            self._make_vlam_observation("pick up the red block"),
            self._make_vlam_observation("place the green block on the table"),
            self._make_vlam_observation("push"),
        ]

        with mock.patch.object(
            self.policy.inference.model,
            "generate",
            side_effect=functools.partial(mock_generate, num_effectors=1),
        ):
            batched_controls = self.policy.inference.low_level_step(vlam_observations)
            individual_controls = [
                self.policy.inference.low_level_step([obs])[0] for obs in vlam_observations
            ]

        self.assertEqual(len(batched_controls), len(vlam_observations))
        for batched, individual in zip(batched_controls, individual_controls, strict=True):
            self.assertTrue(np.allclose(batched.arms.translation, individual.arms.translation))
            self.assertTrue(np.allclose(batched.arms.rotation, individual.arms.rotation))
            self.assertTrue(np.allclose(batched.hands.gripper_prob, individual.hands.gripper_prob))

    def test_prepare_observation_drops_unknown_cameras_and_resizes_known_ones(self):
        """The server hands every wire observation to the policy before the forward: cameras the
        model does not read are dropped, and the rest go through the real processor's resize."""
        raw_h, raw_w = 480, 640
        observation = dataclasses.replace(
            self._make_vlam_observation("pick up the red block"),
            images={
                "main": np.zeros((1, raw_h, raw_w, 3), dtype=np.uint8),
                "wrist": np.zeros((1, raw_h, raw_w, 3), dtype=np.uint8),
                "extra_unused_cam": np.zeros((1, raw_h, raw_w, 3), dtype=np.uint8),
            },
        )

        prepared = self.policy.inference.prepare_observation(observation)

        known_cameras = set(self.policy.inference.processor.image_sizes)
        self.assertEqual(set(prepared.images.keys()), known_cameras)
        for camera_name in known_cameras:
            shape = prepared.images[camera_name].shape
            self.assertEqual(shape[0], 1)
            self.assertEqual(shape[-1], 3)
            self.assertNotEqual((shape[1], shape[2]), (raw_h, raw_w))

    def _make_observation(self, dual: bool = False, shared_state: bool = False) -> RobotObservation:
        """``shared_state`` gives both arms the same ManipulatorState, so that a difference between the
        two emitted actions can only have come from the per-effector control"""
        return RobotObservation(
            images={
                "main": self.prng.randint(0, 256, size=(480, 640, 3)).astype(np.uint8),
                "wrist": self.prng.randint(0, 256, size=(480, 640, 3)).astype(np.uint8),
            },
            # Drawn here rather than earlier: the single-arm action checksums pin the order in which
            # this method consumes the PRNG, so the state must be built after the images
            state=self._make_dual_state(shared_state) if dual else self._make_state(),
            timestamps=self.prng.uniform(size=(1,)).astype(np.float64),
            metadata={},
        )

    def _make_dual_state(self, shared_state: bool) -> DualManipulatorState:
        right = self._make_state()
        return DualManipulatorState(right=right, left=right if shared_state else self._make_state())

    def _make_state(self) -> ManipulatorState:
        return ManipulatorState(
            eef_pose=make_random_pose(self.prng).astype(np.float32),
            eef_pose_from_target=make_random_pose(self.prng).astype(np.float32),
            gripper_state=self.prng.uniform(size=(1,)).astype(np.float32),
            joints=self.prng.uniform(size=(7,)).astype(np.float32),
        )

    def _make_vlam_observation(self, instruction: str) -> RobotInferenceObservation:
        """A single-arm history window of one frame, shaped [H=1, num_effectors=1, D]"""
        return RobotInferenceObservation(
            images={
                "main": self.prng.randint(0, 256, size=(1, 480, 640, 3)).astype(np.uint8),
                "wrist": self.prng.randint(0, 256, size=(1, 480, 640, 3)).astype(np.uint8),
            },
            instruction=instruction,
            arms=ArmsInferenceObservation(
                ee_pose_translation=self.prng.uniform(size=(1, 1, 3)).astype(np.float32),
                ee_pose_rotation=np.tile(np.eye(3, dtype=np.float32), (1, 1, 1, 1)),
                joint_positions=self.prng.uniform(size=(1, 1, 7)).astype(np.float32),
            ),
            hands=HandsInferenceObservation(
                gripper_width=self.prng.uniform(size=(1, 1, 1)).astype(np.float32)
            ),
            timestamps=self.prng.uniform(size=(1,)).astype(np.float64),
            dataset_name="bridge_orig",
        )


class _FakeRequestReplyClient:
    """Stands in for the ZMQ client the remote inference client builds, replying from a queue."""

    def __init__(self, replies):
        self.sent_requests = []
        self._replies = list(replies)

    def send(self, request):
        self.sent_requests.append(request)
        return self._replies.pop(0)

    def close(self) -> None:
        pass


class TestVLAMRobotInferenceWithRemoteClient(unittest.TestCase):
    """``VLAMRobotInference`` configured (via AutoConfig) to run against a remote batch server."""

    def test_predict_actions_routes_through_remote_inference(self):
        fake_client = _FakeRequestReplyClient(
            [
                {
                    "status": ResponseStatus.OK.value,
                    "metadata": {
                        "image_sizes": {"main": [480, 640], "wrist": [480, 640]},
                        "past_scalars_sequence_length": 1,
                        "past_frames_sequence_length": 1,
                        "delta_mode": "rot_eef_trans_base",
                        "num_effectors": 1,
                        "protocol_version": BATCH_PROTOCOL_VERSION,
                    },
                },
                {
                    "status": ResponseStatus.OK.value,
                    "controls": [
                        # [num_future_control_steps, num_effectors, ...]
                        RobotInferenceControl(
                            arms=ArmsInferenceControl(
                                translation=np.zeros((1, 1, 3), dtype=np.float32),
                                rotation=np.eye(3, dtype=np.float32).reshape(1, 1, 3, 3),
                            ),
                            hands=HandsInferenceControl(gripper_prob=np.zeros((1, 1, 1), dtype=np.float32)),
                        ).to_dict()
                    ],
                },
            ]
        )

        config = AutoConfig(
            autoclass="ladder.components.inference.models.vlams.vlam_robot_inference.VLAMRobotInference",
            env_name="bridge_orig",
            controls_on_targets=True,
            inference_config={
                "autoclass": (
                    "ladder.components.inference.api.client.batch_inference."
                    "remote_inference_client.RemoteInferenceClient"
                ),
                "client_config": {
                    "autoclass": "ladder.components.inference.api.core.request_reply.ZmqRequestReplyClient",
                    "ip": "localhost",
                    "port": 0,
                },
            },
            controller_config={
                "autoclass": (
                    "ladder.components.inference.models.vlams.controllers.action_chunking."
                    "ActionChunkingController"
                ),
                "chunk_size": 4,
            },
        )
        with mock.patch.object(remote_inference_client_module, "AutoConfigurable", return_value=fake_client):
            policy = AutoConfigurable(config)

        observation = RobotObservation(
            images={
                "main": np.zeros((480, 640, 3), dtype=np.uint8),
                "wrist": np.zeros((480, 640, 3), dtype=np.uint8),
            },
            state=ManipulatorState(
                eef_pose=np.eye(4, dtype=np.float32),
                eef_pose_from_target=np.eye(4, dtype=np.float32),
                gripper_state=np.array([0.0], dtype=np.float32),
                joints=np.zeros(7, dtype=np.float32),
            ),
            timestamps=np.array([0.0], dtype=np.float64),
            metadata={},
        )
        action = policy.predict_actions([observation], ["pick up block"])[0]

        self.assertIsInstance(action, RobotAction)
        # delta_mode must delegate to the inference object (no local model config).
        self.assertEqual(policy.delta_mode, "rot_eef_trans_base")
        self.assertEqual(
            [r["type"] for r in fake_client.sent_requests],
            [BatchRequestType.METADATA.value, BatchRequestType.LOW_LEVEL_INFER.value],
        )


def make_random_pose(prng: np.random.Generator) -> np.ndarray:
    translation = prng.uniform(size=[3])  # [3]
    rotation = scipy.spatial.transform.Rotation.random(random_state=prng).as_matrix()  # [3, 3]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = translation
    pose[:3, :3] = rotation

    return pose


def _trim_padding(item):
    """Drop an item's padding - left, right, or both - so its checksum is the same whether it was
    processed alone or padded up to a batch-mate's length. The real model is padding-invariant
    through the attention mask; a tokenizer with a nonzero pad id (qwen) would otherwise leak the
    padding into the checksum, while paligemma's pad id 0 only hid the leak because the checksum is
    a sum."""
    valid = item.attn_mask.bool()
    return dataclasses.replace(
        item,
        input_ids=item.input_ids[valid].reshape(1, -1),
        attn_mask=item.attn_mask[valid].reshape(1, -1),
    )


def mock_generate(model_input, num_effectors: int) -> FlatRoboticsOutput:
    """Batch-aware deterministic generator: produces per-item outputs derived from a checksum of that
    item's slice of the batched input, varied per timestep and per effector. Works for any batch size.
    Every quantity is varied inside its own domain."""
    batch_size = model_input.input_ids.shape[0]
    num_timesteps = 5
    timesteps = torch.arange(num_timesteps, dtype=torch.float32).view(num_timesteps, 1, 1)
    effectors = torch.arange(num_effectors, dtype=torch.float32).view(1, num_effectors, 1)

    checksums = torch.tensor(
        [
            (calculate_data_checksum(_trim_padding(model_input[i : i + 1]), hashed=False) % 10**9) / 10**9
            for i in range(batch_size)
        ],
        dtype=torch.float32,
    ).view(
        batch_size, 1, 1, 1
    )  # broadcast over (num_timesteps, num_effectors, channels)

    translation = (0.01 * (timesteps + 1) + effectors + checksums).repeat(1, 1, 1, 3)
    angles = ((timesteps + 1) + (20.0 / num_effectors) * effectors + 20.0 * checksums).repeat(1, 1, 1, 3)
    rotation = roma.euler_to_rotmat(convention='xyz', angles=angles, degrees=True).flatten(-2)
    gripper = (timesteps / num_timesteps + effectors / num_effectors + checksums) % 1.0

    if not has_effector_axis_processor_api(ENV_VERSION):
        assert num_effectors == 1, f"{ENV_VERSION} bundles have no effector axis"
        return FlatRoboticsOutput(
            translation=translation.squeeze(-2),
            rotation=rotation.squeeze(-2),
            gripper=gripper.squeeze(-2),
        )

    return RoboticsOutput(
        arms_output=ArmsOutput(translation=translation, rotation=rotation),
        hands_output=HandsOutput(gripper_width=gripper),
    )


if __name__ == "__main__":
    unittest.main()
