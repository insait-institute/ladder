"""ZMQ server for RobotArena evaluation, K=1 or K>=2 batched."""

import atexit
import dataclasses
import os
import signal
import traceback
from enum import Enum
from types import FrameType
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.spatial.transform

from ladder.components.inference.api.core.action import RobotAction
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.components.inference.api.core.requests import RequestType
from ladder.components.inference.api.server.request_listener import RequestListener, RequestListenerConfig
from ladder.components.inference.robots.robot_arena.composition.registry import COMPOSITION_TASKS
from ladder.components.inference.robots.robot_arena.robot_arena_env import RobotArenaEnv, RobotArenaEnvConfig
from ladder.components.inference.robots.robot_arena.scene_builder import set_genesis_seed
from ladder.components.inference.robots.robot_arena.scene_loader import (
    ObjectConfig,
    list_available_backgrounds,
    list_available_scenes,
    load_scene,
)
from ladder.components.inference.robots.robot_arena.transforms import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz
from ladder.core.common.logger.logger import get_logger
from ladder.core.compute.seeder import SEEDER
from ladder.core.databib.config import AutoConfig, Config, ConfigParser, Configurable

LOGGER = get_logger(__name__)


class TestType(str, Enum):
    DEFAULT = "default"
    BACKGROUND = "background"
    CAMERA = "camera"
    PERMUTATION = "permutation"
    POSE = "pose"
    OBJECT_VARIATION = "object_variation"


class RobotArenaTestConfig(Config):
    """Defines which test variation to run and how to enumerate episodes."""

    test_type: TestType = TestType.DEFAULT

    # Scenes to evaluate; empty auto-discovers every scene in data_root. An entry naming a
    # COMPOSITION_TASKS task runs that multi-goal task on its underlying scene instead.
    scene_names: List[str] = []

    # Number of background variations (for background test type).
    # 0 means use all available backgrounds.
    num_background_variations: int = 0

    # Camera perturbation offsets in meters (for camera test type).
    # Each entry is [dx, dy, dz] added to the base camera position.
    camera_offsets: List[Tuple[float, float, float]] = []

    # Number of random permutations per scene (for permutation test type).
    num_permutations: int = 5

    # Pose variation grid size per scene (for pose test type).
    num_pose_variations: int = 5

    # Random seed for reproducible permutations/pose variations.
    seed: int = 42

    # Deterministic object XY offset grid for object_variation test type.
    # Format: (min_offset_m, max_offset_m, num_samples)
    # Each (dx, dy) pair is applied uniformly to all movable objects.
    object_x_offset_range: Tuple[float, float, int] = (-0.03, 0.03, 5)
    object_y_offset_range: Tuple[float, float, int] = (-0.03, 0.03, 5)

    # 1: single-env. >=2: one batch per scene with K envs stepped together.
    batch_size_per_scene: int = 1

    # Explicit (dx, dy) offsets used at K>=2 instead of the
    # ``object_x_offset_range`` × ``object_y_offset_range`` grid.
    # Length must equal ``batch_size_per_scene``.
    object_variation_offsets_m: List[List[float]] = []


class RobotArenaServerConfig(Config):
    env_config: RobotArenaEnvConfig
    test_config: RobotArenaTestConfig = RobotArenaTestConfig()
    request_listener_config: RequestListenerConfig = RequestListenerConfig(
        req_rep_server_config=AutoConfig(
            autoclass='ladder.components.inference.api.core.request_reply.ZmqRequestReplyServer',
            port=5556,
        ),
    )

    @property
    def n_envs_per_batch(self) -> int:
        k = int(self.test_config.batch_size_per_scene)
        if k < 1:
            raise ValueError(f"batch_size_per_scene must be >= 1, got {k}")
        return k

    def make_episode_configs(self, rng: np.random.Generator) -> List[RobotArenaEnvConfig]:
        """Generate all episode (K=1) or batch (K>=2) configurations.

        Configs are sorted by scene_name so consecutive episodes with the same
        scene can reuse the built Genesis scene (avoiding expensive rebuilds).
        """
        scene_names = list(self.test_config.scene_names)
        if len(scene_names) == 0:
            # Composition tasks are opt-in: auto-discovery enumerates scenes only.
            scene_names = list_available_scenes(self.env_config.data_root)
            LOGGER.info(f"Auto-discovered {len(scene_names)} scenes: {scene_names}")

        composition_names = [name for name in scene_names if name in COMPOSITION_TASKS]
        scene_names = [name for name in scene_names if name not in COMPOSITION_TASKS]

        if len(scene_names) == 0 and len(composition_names) == 0:
            raise ValueError(f"No scenes found in {self.env_config.data_root}")

        if self.n_envs_per_batch > 1:
            if self.test_config.test_type != TestType.OBJECT_VARIATION:
                raise ValueError(
                    "batch_size_per_scene > 1 is only supported with "
                    f"test_type=object_variation (got {self.test_config.test_type})."
                )
            offsets = self._resolve_batched_offsets()
            configs = self._make_batched_object_variation_configs(scene_names, offsets)
            configs += self._make_composition_configs(composition_names, offsets)
            configs.sort(key=lambda c: c.scene_name)
            return configs

        factory = {
            TestType.DEFAULT: self._make_default_configs,
            TestType.BACKGROUND: self._make_background_configs,
            TestType.CAMERA: self._make_camera_configs,
            TestType.PERMUTATION: lambda names: self._make_permutation_configs(names, rng),
            TestType.POSE: lambda names: self._make_pose_configs(names, rng),
            TestType.OBJECT_VARIATION: self._make_object_variation_configs,
        }
        if self.test_config.test_type not in factory:
            raise NotImplementedError(f"Test type '{self.test_config.test_type}' not implemented")

        configs = factory[self.test_config.test_type](scene_names)
        configs += self._make_composition_configs(composition_names, None)
        configs.sort(key=lambda c: c.scene_name)
        return configs

    def _make_composition_configs(
        self, task_names: List[str], offsets: Optional[List[Tuple[float, float]]]
    ) -> List[RobotArenaEnvConfig]:
        """One episode (K=1) or batch (K>=2) per composition task on its underlying scene."""
        return [
            self.env_config.replace(
                scene_name=COMPOSITION_TASKS[name].scene,
                composition_task=name,
                max_episode_steps=COMPOSITION_TASKS[name].max_episode_steps,
                per_env_object_offsets=offsets,
            )
            for name in task_names
        ]

    def _resolve_batched_offsets(self) -> List[Tuple[float, float]]:
        explicit = self.test_config.object_variation_offsets_m
        k = self.n_envs_per_batch
        if len(explicit) > 0:
            if len(explicit) != k:
                raise ValueError(
                    f"object_variation_offsets_m has {len(explicit)} entries but "
                    f"batch_size_per_scene={k}; they must match."
                )
            return [(float(o[0]), float(o[1])) for o in explicit]

        x_min, x_max, x_n = self.test_config.object_x_offset_range
        y_min, y_max, y_n = self.test_config.object_y_offset_range
        xs = np.linspace(x_min, x_max, int(x_n))
        ys = np.linspace(y_min, y_max, int(y_n))
        offsets = [(float(x), float(y)) for x in xs for y in ys]
        if len(offsets) != k:
            raise ValueError(
                f"grid {int(x_n)}x{int(y_n)}={len(offsets)} offsets does not match "
                f"batch_size_per_scene={k}; either adjust the grid or provide "
                "object_variation_offsets_m explicitly."
            )
        return offsets

    def _make_batched_object_variation_configs(
        self, scene_names: List[str], offsets: List[Tuple[float, float]]
    ) -> List[RobotArenaEnvConfig]:
        configs: List[RobotArenaEnvConfig] = []
        for scene in scene_names:
            configs.append(
                self.env_config.replace(
                    scene_name=scene,
                    per_env_object_offsets=offsets,
                )
            )
        return configs

    def _make_default_configs(self, scene_names: List[str]) -> List[RobotArenaEnvConfig]:
        """One episode per scene with default settings."""
        return [self.env_config.replace(scene_name=scene) for scene in scene_names]

    def _make_background_configs(self, scene_names: List[str]) -> List[RobotArenaEnvConfig]:
        """One episode per (scene, background) combination."""
        backgrounds = list_available_backgrounds(self.env_config.data_root)
        if self.test_config.num_background_variations > 0:
            backgrounds = backgrounds[: self.test_config.num_background_variations]

        if len(backgrounds) == 0:
            LOGGER.warn("No background images found, falling back to default configs")
            return self._make_default_configs(scene_names)

        configs = []
        for scene in scene_names:
            for bg_path in backgrounds:
                configs.append(self.env_config.replace(scene_name=scene, background_path=bg_path))
        return configs

    def _make_camera_configs(self, scene_names: List[str]) -> List[RobotArenaEnvConfig]:
        """One episode per (scene, camera_offset) combination."""
        offsets = self.test_config.camera_offsets
        if len(offsets) == 0:
            offsets = [
                (0.05, 0.0, 0.0),
                (-0.05, 0.0, 0.0),
                (0.0, 0.05, 0.0),
                (0.0, -0.05, 0.0),
                (0.0, 0.0, 0.05),
                (0.0, 0.0, -0.05),
            ]

        configs = []
        for scene in scene_names:
            configs.append(self.env_config.replace(scene_name=scene))
            for offset in offsets:
                configs.append(self.env_config.replace(scene_name=scene, camera_offset=list(offset)))
        return configs

    def _make_permutation_configs(
        self, scene_names: List[str], rng: np.random.Generator
    ) -> List[RobotArenaEnvConfig]:
        """Shuffle object XY positions within each scene."""
        configs = []
        for scene in scene_names:
            scene_data = load_scene(self.env_config.data_root, scene)
            original_positions = [obj.position for obj in scene_data.objects]
            configs.append(self.env_config.replace(scene_name=scene))
            for _ in range(self.test_config.num_permutations):
                permuted = self._shuffle_object_xy(original_positions, rng)
                configs.append(self.env_config.replace(scene_name=scene, object_positions=permuted))
        return configs

    def _make_pose_configs(
        self, scene_names: List[str], rng: np.random.Generator
    ) -> List[RobotArenaEnvConfig]:
        """Vary object positions and rotations within each scene."""
        configs = []
        for scene in scene_names:
            scene_data = load_scene(self.env_config.data_root, scene)
            configs.append(self.env_config.replace(scene_name=scene))
            for _ in range(self.test_config.num_pose_variations):
                positions, rotations = self._vary_object_poses(scene_data.objects, rng)
                configs.append(
                    self.env_config.replace(
                        scene_name=scene,
                        object_positions=positions,
                        object_rotations=rotations,
                    )
                )
        return configs

    def _make_object_variation_configs(self, scene_names: List[str]) -> List[RobotArenaEnvConfig]:
        """Deterministic XY offset grid applied uniformly to all movable objects.

        Produces num_x * num_y episodes per scene (default 5x5 = 25), matching
        the SimplerEnv convention for init-condition sweeps.
        """
        x_min, x_max, x_n = self.test_config.object_x_offset_range
        y_min, y_max, y_n = self.test_config.object_y_offset_range
        x_offsets = np.linspace(x_min, x_max, int(x_n))
        y_offsets = np.linspace(y_min, y_max, int(y_n))

        configs: List[RobotArenaEnvConfig] = []
        for scene in scene_names:
            scene_data = load_scene(self.env_config.data_root, scene)
            original_positions = [list(obj.position) for obj in scene_data.objects]
            for dx in x_offsets:
                for dy in y_offsets:
                    shifted = [(p[0] + dx, p[1] + dy, p[2]) for p in original_positions]
                    configs.append(self.env_config.replace(scene_name=scene, object_positions=shifted))
        return configs

    @staticmethod
    def _shuffle_object_xy(
        original_positions: List[Tuple[float, float, float]], rng: np.random.Generator
    ) -> List[Tuple[float, float, float]]:
        """Shuffle X and Y coordinates among objects while preserving Z heights."""
        positions = [list(p) for p in original_positions]
        xy_coords = [(p[0], p[1]) for p in positions]
        rng.shuffle(xy_coords)
        return [(xy[0], xy[1], p[2]) for xy, p in zip(xy_coords, positions, strict=True)]

    @staticmethod
    def _vary_object_poses(
        objects: List[ObjectConfig], rng: np.random.Generator
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float, float]]]:
        """Generate random position and rotation variations for objects."""
        positions = []
        rotations = []
        for obj in objects:
            pos = list(obj.position)
            pos[0] += rng.uniform(-0.05, 0.05)
            pos[1] += rng.uniform(-0.05, 0.05)
            positions.append(pos)

            angle_rad = rng.uniform(0, 2 * np.pi)
            rot = scipy.spatial.transform.Rotation.from_euler('z', angle_rad)
            orig_wxyz = np.array(obj.rotation_quat)
            orig_xyzw = quat_wxyz_to_xyzw(orig_wxyz)
            original_rot = scipy.spatial.transform.Rotation.from_quat(orig_xyzw)
            combined_xyzw = (rot * original_rot).as_quat()
            rotations.append(quat_xyzw_to_wxyz(combined_xyzw).tolist())

        return positions, rotations


class RobotArenaServer(Configurable[RobotArenaServerConfig]):
    """ZMQ server for RobotArena evaluation. K=1 single-env or K>=2 batched per scene.

    At K>=2 the server pre-builds the first scene during ``run()`` so Genesis
    build time overlaps with client-side policy warmup.
    """

    def __init__(self, config: RobotArenaServerConfig):
        super().__init__(config)
        # Genesis has its own RNG independent of torch's, so seed it explicitly.
        SEEDER.seed_everything(self.config.test_config.seed)
        set_genesis_seed(self.config.test_config.seed)
        self._request_listener = RequestListener(self.config.request_listener_config)
        self._rng = SEEDER.new_prng_np()
        self._env: Optional[RobotArenaEnv] = None
        self._observations: List[RobotObservation] = []
        self._episode_configs = self.config.make_episode_configs(self._rng)
        self._episode_index: int = 0
        self._prebuilt_batch_index: Optional[int] = None
        self._n_envs_per_batch: int = self.config.n_envs_per_batch

        unit = "batch" if self._n_envs_per_batch > 1 else "episode"
        LOGGER.info(
            f"Generated {len(self._episode_configs)} {unit} configs "
            f"(n_envs_per_batch={self._n_envs_per_batch})"
        )

        atexit.register(self.close)
        signal.signal(signal.SIGTERM, self._on_sigterm)

        self._request_listener.register_handler(RequestType.CONNECT, lambda _: {"status": "ok"})
        self._request_listener.register_handler(RequestType.RESET, self._reset)
        self._request_listener.register_handler(RequestType.GET_OBSERVATIONS, self._get_observations)
        self._request_listener.register_handler(RequestType.EXEC_ACTION, self._execute_action)
        self._request_listener.register_handler(RequestType.EXIT, lambda _: {"status": "ok"})

    def _on_sigterm(self, signum: int, frame: Optional[FrameType]) -> None:
        del signum, frame
        self.close()
        os._exit(0)

    def _reset(self, _: Dict[str, Any]) -> Dict[str, Any]:
        if self._episode_index >= len(self._episode_configs):
            self._observations = []
            return {"status": "finished"}

        env_config = self._episode_configs[self._episode_index]
        next_index = self._episode_index
        self._episode_index += 1
        unit = "batch" if self._n_envs_per_batch > 1 else "episode"
        LOGGER.info(
            f"Starting {unit} {self._episode_index}/{len(self._episode_configs)} "
            f"scene={env_config.scene_name} n_envs={env_config.n_envs}"
        )

        if self._prebuilt_batch_index == next_index:
            self._prebuilt_batch_index = None
        elif self._env is not None and self._env.can_reuse_for(env_config):
            self._env.reconfigure(env_config)
            self._observations = self._env.reset()
        else:
            if self._env is not None:
                self._env.close()
            self._env = RobotArenaEnv(config=env_config)
            self._observations = self._env.reset()

        return {"status": "ok", "n_envs": env_config.n_envs}

    def _get_observations(self, _: Dict[str, Any]) -> Dict[str, Any]:
        if not self._observations:
            return {"status": "error", "message": "No observation available. Reset the environment first."}

        if self._n_envs_per_batch == 1:
            return {"status": "ok", "data": dataclasses.asdict(self._observations[0])}

        return {"status": "ok", "data": [dataclasses.asdict(obs) for obs in self._observations]}

    def _execute_action(self, action_request: Dict[str, Any]) -> Dict[str, Any]:
        if self._env is None or not self._observations:
            return {"status": "error", "message": "Environment not initialized"}

        actions = self._parse_action_request(action_request)
        if len(actions) != self._env.n_envs:
            return {
                "status": "error",
                "message": f"expected {self._env.n_envs} actions, got {len(actions)}",
            }

        self._observations = self._env.step(actions)
        return {"success": True}

    def _parse_action_request(self, payload: Dict[str, Any]) -> List[RobotAction]:
        if isinstance(payload, dict) and "actions" in payload:
            action_dicts = payload["actions"]
            if not isinstance(action_dicts, list):
                raise ValueError("EXEC_ACTION 'actions' must be a list")
            return [RobotAction.from_dict(d) for d in action_dicts]
        return [RobotAction.from_dict(payload)]

    def run(self) -> None:
        self._prebuild_first_scene()
        self._request_listener.start()

    def _prebuild_first_scene(self) -> None:
        if self._n_envs_per_batch == 1 or not self._episode_configs:
            return
        first_config = self._episode_configs[0]
        LOGGER.info(
            f"Pre-building first scene (scene={first_config.scene_name} n_envs={first_config.n_envs})"
        )
        self._env = RobotArenaEnv(config=first_config)
        self._observations = self._env.reset()
        self._prebuilt_batch_index = 0

    def __del__(self):
        self.close()

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


if __name__ == "__main__":
    parser = ConfigParser(RobotArenaServerConfig)
    config: RobotArenaServerConfig = parser.parse_args()

    server = RobotArenaServer(config)

    try:
        server.run()
    except KeyboardInterrupt:
        LOGGER.info("Ctrl-C pressed, shutting down.")
    except Exception:  # noqa: BLE001
        LOGGER.error(traceback.format_exc())
