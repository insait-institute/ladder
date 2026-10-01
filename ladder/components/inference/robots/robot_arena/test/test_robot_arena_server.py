"""
Integration test for RobotArena: start server, connect client, run episodes.

Follows the same pattern as test_simpler_server.py — verifies observation shapes,
deterministic checksums, and runs action steps through the full ZMQ round-trip with Genesis.

Requires GPU (Genesis uses CUDA for physics and rendering).
"""

import os

# Genesis writes to $HOME/.cache on import; Bazel sandboxes don't set HOME.
if "HOME" not in os.environ:
    assert "TEST_TMPDIR" in os.environ, "Neither HOME nor TEST_TMPDIR set"
    os.environ["HOME"] = os.environ["TEST_TMPDIR"]

import socket
import threading
import unittest
from typing import Any, Callable, Dict

import numpy as np
import scipy.spatial.transform

from ladder.components.inference.api.core.action import EndEffectorPoseAction, ManipulatorAction
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.components.inference.robots.robot_arena.robot_arena_server import (
    RobotArenaServer,
    RobotArenaServerConfig,
)
from ladder.components.inference.scripts.example_client import ExampleClient
from ladder.core.compute.checksum import calculate_data_checksum
from ladder.core.compute.seeder import SEEDER

NUM_STEPS = 5

# fmt: off
# Expected checksums captured from a deterministic run (seed=42, scene1, 5 steps).
# To regenerate: clear this dict, run the test, copy the printed values.
EXPECTED_CHECKSUMS: Dict[int, Dict[str, float | int]] = {
    0: {
        'obs.images.main': 15477400791965159919,
        'obs.eef_pose': 5.143066883087158,
        'obs.eef_pose_from_target': 5.143066883087158,
        'obs.gripper_state': 0.5277519822120667,
        'obs.joints': 2.434300422668457,
        'action.translation': 0.01912653391291182,
        'action.rotation': 3.2868855820970713,
        'action.gripper': 0.2982278092806382,
    },
    1: {
        'obs.images.main': 10386353010606461661,
        'obs.eef_pose': 5.743968963623047,
        'obs.eef_pose_from_target': 4.306012153625488,
        'obs.gripper_state': 0.4052318036556244,
        'obs.joints': 4.186976909637451,
        'action.translation': 0.005202053435070068,
        'action.rotation': 3.298296609054808,
        'action.gripper': 0.5056424728082645,
    },
    2: {
        'obs.images.main': 11860010246602230861,
        'obs.eef_pose': 5.75864315032959,
        'obs.eef_pose_from_target': 4.303498268127441,
        'obs.gripper_state': 1.0,
        'obs.joints': 4.51572847366333,
        'action.translation': 0.01232837223831683,
        'action.rotation': 3.1877003395614962,
        'action.gripper': 0.7074211367900608,
    },
    3: {
        'obs.images.main': 13667806475789609121,
        'obs.eef_pose': 5.712717533111572,
        'obs.eef_pose_from_target': 4.200028419494629,
        'obs.gripper_state': 1.0,
        'obs.joints': 4.989813327789307,
        'action.translation': 0.011569261945164007,
        'action.rotation': 3.2282249429603045,
        'action.gripper': 0.1674791832212973,
    },
    4: {
        'obs.images.main': 11956817480893434775,
        'obs.eef_pose': 5.923398017883301,
        'obs.eef_pose_from_target': 4.239794731140137,
        'obs.gripper_state': 0.6331031322479248,
        'obs.joints': 6.039815425872803,
        'action.translation': 0.02226443457393989,
        'action.rotation': 3.2083802911833192,
        'action.gripper': 0.2667126114535301,
    },
    5: {
        'obs.images.main': 4996504570094122543,
        'obs.eef_pose': 5.415618896484375,
        'obs.eef_pose_from_target': 4.230644702911377,
        'obs.gripper_state': 0.40509718656539917,
        'obs.joints': 6.92599630355835,
    },
}
# fmt: on


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_with_catch_system_exit(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except SystemExit:
            pass

    return wrapper


class TestRobotArenaServer(unittest.TestCase):
    """Full server+client integration test for RobotArena.

    Mirrors the structure of TestSimplerBridge in test_simpler_server.py.
    """

    def setUp(self) -> None:
        SEEDER.reset()
        SEEDER.seed_everything(42)
        self.prng = SEEDER.new_prng_np()

    def test_server_client_round_trip(self) -> None:
        try:
            self._verify_k1_round_trip()
        finally:
            # Genesis CUDA teardown crashes with CUDA_ERROR_INVALID_CONTEXT during
            # Python GC when the CUDA context is destroyed before all allocations
            # are freed. Use os._exit to skip Python's destructors and avoid the abort.
            os._exit(0)

    def _verify_k1_round_trip(self) -> None:
        server_port = _free_port()

        server_config = RobotArenaServerConfig.from_yaml(
            "ladder/components/inference/robots/robot_arena/configs/widowx.yaml"
        ).replace(
            **{
                'request_listener_config.req_rep_server_config.port': server_port,
                'test_config.scene_names': ['scene1'],
            }
        )
        server = RobotArenaServer(config=server_config)

        server_thread = threading.Thread(target=_run_with_catch_system_exit(server.run), daemon=True)
        server_thread.start()

        # Genesis scene building (CoACD convex decomposition + CUDA kernel
        # compilation) can take several minutes on first run.
        client = ExampleClient(ip="127.0.0.1", port=server_port, recv_timeout_ms=600_000)

        try:
            client.connect()
            client.reset()

            actual_checksums: Dict[int, Dict[str, float | int]] = {}

            for step in range(NUM_STEPS):
                obs: RobotObservation = client.get_observations()

                self._assert_observation_shapes(obs)

                step_checksums: Dict[str, float | int] = {}
                step_checksums["obs.images.main"] = calculate_data_checksum(obs.images["main"], hashed=True)
                step_checksums["obs.eef_pose"] = calculate_data_checksum(obs.state.eef_pose, hashed=False)
                step_checksums["obs.eef_pose_from_target"] = calculate_data_checksum(
                    obs.state.eef_pose_from_target, hashed=False
                )
                step_checksums["obs.gripper_state"] = calculate_data_checksum(
                    obs.state.gripper_state, hashed=False
                )
                step_checksums["obs.joints"] = calculate_data_checksum(obs.state.joints, hashed=False)

                action = ManipulatorAction(
                    arm_action=EndEffectorPoseAction(
                        translation=self.prng.uniform(low=-0.02, high=0.02, size=(3,)),
                        rotation=scipy.spatial.transform.Rotation.from_euler(
                            "xyz", self.prng.uniform(low=-5, high=5, size=(3,)), degrees=True
                        ).as_matrix(),
                    ),
                    gripper_action=self.prng.uniform(low=0.0, high=1.0, size=None),
                )

                step_checksums["action.translation"] = calculate_data_checksum(
                    action.arm_action.translation, hashed=False
                )
                step_checksums["action.rotation"] = calculate_data_checksum(
                    action.arm_action.rotation, hashed=False
                )
                step_checksums["action.gripper"] = float(action.gripper_action)

                client.execute_action(action)
                actual_checksums[step] = step_checksums

            # Final observation after all steps
            final_obs: RobotObservation = client.get_observations()
            self._assert_observation_shapes(final_obs)

            final_checksums: Dict[str, float | int] = {}
            final_checksums["obs.images.main"] = calculate_data_checksum(
                final_obs.images["main"], hashed=True
            )
            final_checksums["obs.eef_pose"] = calculate_data_checksum(final_obs.state.eef_pose, hashed=False)
            final_checksums["obs.eef_pose_from_target"] = calculate_data_checksum(
                final_obs.state.eef_pose_from_target, hashed=False
            )
            final_checksums["obs.gripper_state"] = calculate_data_checksum(
                final_obs.state.gripper_state, hashed=False
            )
            final_checksums["obs.joints"] = calculate_data_checksum(final_obs.state.joints, hashed=False)
            actual_checksums[NUM_STEPS] = final_checksums

            # Verify gripper state is in canonical range [0, 1]
            self.assertGreaterEqual(float(final_obs.state.gripper_state[0]), 0.0)
            self.assertLessEqual(float(final_obs.state.gripper_state[0]), 1.0)

            # Verify EEF position is reasonable (not NaN or inf)
            eef_pos = final_obs.state.eef_pose[:3, 3]
            self.assertFalse(np.any(np.isnan(eef_pos)))
            self.assertFalse(np.any(np.isinf(eef_pos)))

            # Compare checksums
            if len(EXPECTED_CHECKSUMS) > 0:
                self._assert_checksums_match(actual_checksums)
            else:
                self._print_checksums(actual_checksums)
                self.fail(
                    "EXPECTED_CHECKSUMS is empty. Copy the printed checksums "
                    "above into the EXPECTED_CHECKSUMS dict and re-run."
                )

        finally:
            client.exit()
            server_thread.join(timeout=5.0)
            server.close()

    def _assert_observation_shapes(self, obs: RobotObservation) -> None:
        self.assertIsNotNone(obs)
        self.assertIsNotNone(obs.images)
        self.assertIn("main", obs.images)
        self.assertEqual(len(obs.images["main"].shape), 3)
        self.assertEqual(obs.images["main"].shape[2], 3)
        self.assertEqual(tuple(obs.state.eef_pose.shape), (4, 4))
        self.assertEqual(tuple(obs.state.eef_pose_from_target.shape), (4, 4))
        self.assertEqual(tuple(obs.state.gripper_state.shape), (1,))
        self.assertEqual(tuple(obs.state.joints.shape), (7,))
        self.assertEqual(tuple(obs.timestamps.shape), (1,))
        self.assertIn("env_name", obs.metadata)
        self.assertIn("robot_name", obs.metadata)
        self.assertIn("instruction", obs.metadata)
        self.assertEqual(obs.metadata["robot_name"], "widowx")

    def _assert_checksums_match(self, actual: Dict[int, Dict[str, float | int]]) -> None:
        for step_idx, expected_step in EXPECTED_CHECKSUMS.items():
            self.assertIn(step_idx, actual, f"Missing step {step_idx} in actual checksums")
            actual_step = actual[step_idx]
            for key, expected_val in expected_step.items():
                actual_val = actual_step[key]
                if isinstance(expected_val, int):
                    self.assertEqual(
                        actual_val,
                        expected_val,
                        f"Step {step_idx}, {key}: expected {expected_val}, got {actual_val}",
                    )
                else:
                    self.assertTrue(
                        np.isclose(actual_val, expected_val, rtol=1e-5),
                        f"Step {step_idx}, {key}: expected {expected_val}, got {actual_val}",
                    )

    @staticmethod
    def _print_checksums(checksums: Dict[int, Dict[str, float | int]]) -> None:
        print("\n" + "=" * 80)
        print("Copy this into EXPECTED_CHECKSUMS:")
        print("=" * 80)
        print("EXPECTED_CHECKSUMS: Dict[int, Dict[str, float | int]] = {")
        for step_idx, step_data in sorted(checksums.items()):
            print(f"    {step_idx}: {{")
            for key, value in step_data.items():
                print(f"        {key!r}: {value!r},")
            print("    },")
        print("}")


if __name__ == "__main__":
    unittest.main()
