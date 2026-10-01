import os

# Genesis writes to $HOME/.cache on import; Bazel sandboxes don't set HOME.
if "HOME" not in os.environ:
    assert "TEST_TMPDIR" in os.environ, "Neither HOME nor TEST_TMPDIR set"
    os.environ["HOME"] = os.environ["TEST_TMPDIR"]

import socket
import unittest
from typing import Any

from ladder.components.inference.robots.robot_arena.robot_arena_server import (
    RobotArenaServer,
    RobotArenaServerConfig,
)
from ladder.components.inference.robots.robot_arena.scene_corrections import (
    UPSTREAM_DATA_CORRECTIONS,
    validate_object_orientations,
)
from ladder.components.inference.robots.robot_arena.scene_loader import list_available_scenes, load_scene


class TestSceneLoader(unittest.TestCase):
    """Test scene loading from the pre-generated RobotArena data."""

    def _get_data_root(self) -> str:
        return "ladder/components/inference/robots/robot_arena/robot_arena_data/examples/data"

    def test_list_available_scenes(self) -> None:
        scenes = list_available_scenes(self._get_data_root())
        self.assertGreater(len(scenes), 0, "No scenes found")
        self.assertIn("scene1", scenes)

    def test_load_scene_camera_calibration(self) -> None:
        scene_data = load_scene(self._get_data_root(), "scene1")
        self.assertEqual(scene_data.camera_extrinsics.shape, (4, 4))
        self.assertEqual(scene_data.camera_intrinsics.shape, (3, 3))

    def test_load_scene_task_instruction(self) -> None:
        scene_data = load_scene(self._get_data_root(), "scene1")
        self.assertIsInstance(scene_data.task_instruction, str)
        self.assertGreater(len(scene_data.task_instruction), 0)

    def test_load_scene_objects(self) -> None:
        scene_data = load_scene(self._get_data_root(), "scene1")
        self.assertGreater(len(scene_data.objects), 0)

        for obj in scene_data.objects:
            self.assertIsInstance(obj.name, str)
            self.assertGreater(len(obj.name), 0)
            self.assertEqual(len(obj.position), 3)
            self.assertEqual(len(obj.rotation_quat), 4)
            self.assertIsInstance(obj.scale, float)
            self.assertGreater(obj.scale, 0)
            # Mesh files should exist
            self.assertTrue(
                os.path.exists(obj.mesh_path),
                f"Mesh not found: {obj.mesh_path}",
            )

    def test_load_scene_object_positions_override(self) -> None:
        scene_data = load_scene(self._get_data_root(), "scene1")
        num_objects = len(scene_data.objects)

        # Override positions
        new_positions = [[float(i), float(i), 0.0] for i in range(num_objects)]
        scene_data_overridden = load_scene(self._get_data_root(), "scene1", object_positions=new_positions)
        for i, obj in enumerate(scene_data_overridden.objects):
            self.assertAlmostEqual(obj.position[0], float(i))
            self.assertAlmostEqual(obj.position[1], float(i))

    def test_upstream_corrections_applied(self) -> None:
        """Verify that known upstream data corrections are applied at load time."""
        scene_data = load_scene(self._get_data_root(), "default1")
        cloth = scene_data.objects[0]
        self.assertEqual(cloth.name, "cloth_destination")
        # After correction, the cloth quaternion should be identity [1, 0, 0, 0] (wxyz),
        # meaning no rotation — the towel lies flat.
        for i, expected in enumerate([1.0, 0.0, 0.0, 0.0]):
            self.assertAlmostEqual(cloth.rotation_quat[i], expected, places=6)

    def test_upstream_corrections_reference_valid_scenes(self) -> None:
        """Corrections that reference scenes in the test dataset must point to real scenes.

        The full dataset has 70+ scenes but the bazel test data may only include a
        subset. We only validate corrections for scenes present in the test data —
        corrections for scenes outside the test data are not errors, they simply
        target scenes in the full dataset.
        """
        available_scenes = set(list_available_scenes(self._get_data_root()))
        for (scene_name, _obj_idx), correction in UPSTREAM_DATA_CORRECTIONS.items():
            if scene_name not in available_scenes:
                continue
            # Scene exists in test data — verify the object index is in range
            scene_data = load_scene(self._get_data_root(), scene_name)
            self.assertLess(
                _obj_idx,
                len(scene_data.objects),
                f"Correction for '{scene_name}' object {_obj_idx} is out of range "
                f"(scene has {len(scene_data.objects)} objects): {correction.reason}",
            )


class TestSceneValidation(unittest.TestCase):
    """Validate that all loaded scenes pass orientation sanity checks."""

    def _get_data_root(self) -> str:
        return "ladder/components/inference/robots/robot_arena/robot_arena_data/examples/data"

    def test_all_scenes_pass_orientation_validation(self) -> None:
        """No flat objects (cloth, towel) should be tilted beyond the threshold.

        This test loads every available scene and runs validate_object_orientations.
        If a scene fails, either the upstream data needs a new entry in
        UPSTREAM_DATA_CORRECTIONS, or the correction itself is wrong.
        """
        all_warnings = []
        for scene_name in list_available_scenes(self._get_data_root()):
            scene_data = load_scene(self._get_data_root(), scene_name)
            warnings = validate_object_orientations(
                scene_name,
                [obj.name for obj in scene_data.objects],
                [obj.rotation_quat for obj in scene_data.objects],
            )
            all_warnings.extend(warnings)

        if all_warnings:
            message_lines = ["Scene orientation validation failures:"]
            for warning in all_warnings:
                message_lines.append(f"  - {warning.scene_name}[{warning.object_index}]: {warning.message}")
            self.fail("\n".join(message_lines))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class TestRobotArenaServerConfig(unittest.TestCase):
    """Test server config loading and episode enumeration."""

    def _make_server(self, yaml_path: str, **overrides: Any) -> RobotArenaServer:
        """Create a server with a free port to avoid address conflicts."""
        overrides["request_listener_config.req_rep_server_config.port"] = _free_port()
        config = RobotArenaServerConfig.from_yaml(yaml_path).replace(**overrides)
        return RobotArenaServer(config=config)

    def test_load_config_from_yaml(self) -> None:
        config = RobotArenaServerConfig.from_yaml(
            "ladder/components/inference/robots/robot_arena/configs/widowx.yaml"
        )
        self.assertEqual(config.env_config.robot_name, "widowx")
        self.assertEqual(config.test_config.test_type, "default")

    def test_episode_config_generation_default(self) -> None:
        server = self._make_server(
            "ladder/components/inference/robots/robot_arena/configs/widowx.yaml",
            **{"test_config.scene_names": ["scene1", "scene2"]},
        )
        # Default test: 1 episode per scene
        self.assertEqual(len(server._episode_configs), 2)  # noqa: SLF001

    def test_episode_config_generation_permutation(self) -> None:
        server = self._make_server(
            "ladder/components/inference/robots/robot_arena/configs/widowx.yaml",
            **{
                "test_config.test_type": "permutation",
                "test_config.scene_names": ["scene1"],
            },
        )
        # Permutation test: 1 default + num_permutations per scene
        expected = 1 + server.config.test_config.num_permutations
        self.assertEqual(len(server._episode_configs), expected)  # noqa: SLF001

    def test_episode_config_generation_camera(self) -> None:
        server = self._make_server(
            "ladder/components/inference/robots/robot_arena/configs/widowx.yaml",
            **{
                "test_config.test_type": "camera",
                "test_config.scene_names": ["scene1"],
            },
        )
        # Camera test: 1 default + len(camera_offsets) per scene (uses server defaults)
        expected = 1 + 6  # 6 default camera offsets (+/- 5cm per axis)
        self.assertEqual(len(server._episode_configs), expected)  # noqa: SLF001

    def test_episode_config_scene_name_set(self) -> None:
        """Verify each generated episode config has the scene_name set."""
        server = self._make_server(
            "ladder/components/inference/robots/robot_arena/configs/widowx.yaml",
            **{"test_config.scene_names": ["scene1"]},
        )
        for ep_config in server._episode_configs:  # noqa: SLF001
            self.assertEqual(ep_config.scene_name, "scene1")
            self.assertGreater(len(ep_config.scene_name), 0)


if __name__ == "__main__":
    unittest.main()
