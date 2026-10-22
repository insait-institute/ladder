import os
import socket
import sys
import threading
import unittest
import unittest.mock as mock
from typing import List, Optional

import numpy as np
import yaml

from ladder.components.inference.api.client.simpler_episode_runner import assign_block_instructions
from ladder.components.inference.api.core.action import EndEffectorPoseAction, ManipulatorAction, RobotAction
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.components.inference.scripts.mock_server import MockServer, MockServerConfig
from ladder.core.common.path_utils import create_tempdir
from ladder.core.compute.seeder import SEEDER
from ladder.core.databib.config import Config, Configurable


class MockPolicyConfig(Config):
    translation_step: float = 0.01
    gripper_action: float = 0.5


class MockPolicy(Configurable[MockPolicyConfig]):
    def __init__(self, config: MockPolicyConfig):
        super().__init__(config)

    def predict_actions(
        self,
        observations: List[RobotObservation],
        instructions: List[str],
        env_indices: Optional[List[int]] = None,
    ) -> List[RobotAction]:
        del instructions, env_indices
        actions: List[RobotAction] = []
        for observation in observations:
            current_pose = EndEffectorPoseAction.from_homogeneous_matrix(observation.state.eef_pose)
            arm_action = current_pose.apply_delta(
                delta_rotation=np.eye(3, dtype=np.float32),
                delta_translation=np.array([self.config.translation_step, 0.0, 0.0], dtype=np.float32),
                delta_mode="rot_eef_trans_base",
            )
            actions.append(
                ManipulatorAction(arm_action=arm_action, gripper_action=float(self.config.gripper_action))
            )
        return actions

    def reset(self) -> None:
        pass


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_with_catch_system_exit(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SystemExit:
            return None

    return wrapper


class TestSimplerEpisodeRunner(unittest.TestCase):
    def setUp(self) -> None:
        SEEDER.reset()
        SEEDER.seed_everything(42)
        self.output_path = create_tempdir(self)

    def test_simpler_episode_runner(self):
        server_port = _free_port()
        server = MockServer(
            config=MockServerConfig(
                port=server_port,
                num_episodes=1,
                episode_steps=10,
                instruction="pick up the red block",
                seed=None,
            )
        )
        server_thread = threading.Thread(
            target=_run_with_catch_system_exit(server.run),
            daemon=True,
        )
        server_thread.start()

        run_model_argv = [
            "run_model.py",
            "--config",
            "ladder/components/inference/api/client/test/configs/test_simpler_episode_runner.yaml",
            "--runner_config.env_config.port",
            str(server_port),
            "--runner_config.recorder_config.output_path",
            os.path.join(self.output_path, "episodes"),
            "--seed",
            "null",
        ]

        try:
            sys.modules["torch"] = mock.MagicMock()

            from ladder.components.inference.api.client import run_model

            with mock.patch.object(sys, "argv", run_model_argv):
                run_model.main()

        finally:
            server_thread.join(timeout=2.0)

        results_path = os.path.join(self.output_path, "results.yaml")
        episode_log_path = os.path.join(self.output_path, "episode_log.yaml")
        self.assertTrue(os.path.exists(results_path))
        self.assertTrue(os.path.exists(episode_log_path))

        with open(results_path, "r") as file:
            saved_stats = yaml.safe_load(file)
        cell = saved_stats["widowx"]["test_env"]
        self.assertEqual(cell["success"], "1/1")
        self.assertEqual(cell["success_terminal"], "1/1")
        # This env publishes no per-goal oracle block, so the composition metrics must be absent
        # rather than reported as zero: an inapplicable metric never dilutes an average.
        self.assertNotIn("goal_progress", cell)

        with open(episode_log_path, "r") as file:
            saved_episode_log = yaml.load(file, Loader=yaml.Loader)
        self.assertEqual(len(saved_episode_log), 1)
        self.assertEqual(str(saved_episode_log[0]["instruction"]), "pick up the red block")
        self.assertEqual(saved_episode_log[0]["instruction_history"], ["pick up the red block"])
        self.assertEqual(saved_episode_log[0]["num_steps"], 10)
        self.assertTrue(saved_episode_log[0]["success"])


class TestAssignBlockInstructions(unittest.TestCase):
    """Unit tests for the K>=2 instruction assignment helper used by the unified eval."""

    def test_block_assign_divisible(self):
        overrides = ["paraphrase 1", "paraphrase 2"]
        instructions = assign_block_instructions(
            n_envs=4,
            default_instruction="default",
            overrides=overrides,
            default_block_size=0,
        )
        self.assertEqual(instructions, ["paraphrase 1", "paraphrase 1", "paraphrase 2", "paraphrase 2"])

    def test_round_robin_when_not_divisible(self):
        overrides = ["a", "b", "c"]
        instructions = assign_block_instructions(
            n_envs=4,
            default_instruction="default",
            overrides=overrides,
            default_block_size=0,
        )
        self.assertEqual(instructions, ["a", "b", "c", "a"])

    def test_default_block_with_overrides(self):
        # 16 default envs + 4 override envs (2 paraphrases × 2 reps).
        overrides = ["p1", "p2"]
        instructions = assign_block_instructions(
            n_envs=20,
            default_instruction="default",
            overrides=overrides,
            default_block_size=16,
        )
        self.assertEqual(instructions[:16], ["default"] * 16)
        self.assertEqual(instructions[16:], ["p1", "p1", "p2", "p2"])

    def test_default_block_no_overrides(self):
        # Default-only scene (no override entry): every env runs the default,
        # no breakdown to extract. Happens for scenes without paraphrase coverage.
        instructions = assign_block_instructions(
            n_envs=137,
            default_instruction="default",
            overrides=None,
            default_block_size=16,
        )
        self.assertEqual(instructions, ["default"] * 137)

    def test_unified_canonical_layout(self):
        # The canonical unified-eval layout: 16 default envs + 121 perturbation envs
        # round-robin'd over 27 paraphrases (121 % 27 != 0).
        overrides = [f"p{i}" for i in range(27)]
        instructions = assign_block_instructions(
            n_envs=137,
            default_instruction="default",
            overrides=overrides,
            default_block_size=16,
        )
        self.assertEqual(instructions[:16], ["default"] * 16)
        # Tail follows round-robin: env i (in tail) → overrides[i % 27].
        for tail_i in range(121):
            self.assertEqual(instructions[16 + tail_i], f"p{tail_i % 27}")

    def test_invalid_default_block_size(self):
        with self.assertRaises(ValueError):
            assign_block_instructions(
                n_envs=4,
                default_instruction="default",
                overrides=None,
                default_block_size=-1,
            )
        with self.assertRaises(ValueError):
            assign_block_instructions(
                n_envs=4,
                default_instruction="default",
                overrides=None,
                default_block_size=5,
            )


if __name__ == "__main__":
    unittest.main()
