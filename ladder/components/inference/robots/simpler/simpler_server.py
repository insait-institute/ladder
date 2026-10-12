import dataclasses
from typing import Any, Dict, List, Tuple

from ladder.components.inference.api.core.action import RobotAction
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.components.inference.api.core.requests import RequestType
from ladder.components.inference.api.server.request_listener import RequestListener, RequestListenerConfig
from ladder.components.inference.robots.simpler.simpler_env import SimplerEnv, SimplerEnvConfig
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import Config, ConfigParser, Configurable

LOGGER = get_logger(__name__)


class SimplerInitRangeConfig(Config):
    """
    Packs ranges for varying robot and object initialization poses across episodes.
    """

    robot_init_x_range: Tuple[float, float, int] = (0.35, 0.35, 1)  # [x_min, x_max, num]
    robot_init_y_range: Tuple[float, float, int] = (0.20, 0.20, 1)  # [y_min, y_max, num]
    robot_init_quat_center: Tuple[float, float, float, float] = (1, 0, 0, 0)  # [x, y, z, w]
    robot_init_roll_range: Tuple[float, float, int] = (0, 0, 1)  # [roll_min, roll_max, roll_num]
    robot_init_pitch_range: Tuple[float, float, int] = (0, 0, 1)  # [pitch_min, pitch_max, pitch_num]
    robot_init_yaw_range: Tuple[float, float, int] = (0, 0, 1)  # [yaw_min, yaw_max, yaw_num]

    # Whether to vary the xy position of a single object, or to vary predetermined episodes
    object_variation_mode: str = "xy"  # ["xy", "episode"]

    object_episode_range: Tuple[int, int] = (0, 60)  # [start, end]
    object_init_x_range: Tuple[float, float, int] = (-0.35, -0.12, 5)  # [xmin, xmax, num]
    object_init_y_range: Tuple[float, float, int] = (-0.02, 0.42, 5)  # [ymin, ymax, num]

    def __post_init__(self):
        super().__post_init__()

        assert len(self.robot_init_x_range) == 3, f"{self.robot_init_x_range}"
        assert len(self.robot_init_y_range) == 3, f"{self.robot_init_y_range}"
        assert len(self.robot_init_roll_range) == 3, f"{self.robot_init_roll_range}"
        assert len(self.robot_init_pitch_range) == 3, f"{self.robot_init_pitch_range}"
        assert len(self.robot_init_yaw_range) == 3, f"{self.robot_init_yaw_range}"
        assert len(self.robot_init_quat_center) == 4, f"{self.robot_init_quat_center}"

        assert self.object_variation_mode in ["xy", "episode"], f"{self.object_variation_mode}"


class SimplerServerConfig(Config):
    env_config: SimplerEnvConfig
    init_range_config: SimplerInitRangeConfig
    request_listener_config: RequestListenerConfig


class SimplerServer(Configurable[SimplerServerConfig]):
    """
    Class that abstracts the logic for running inference in simpler.

    ManiSkill3 holds ``n_envs`` parallel envs on one GPU, so each RESET advances to the next batch
    of ``n_envs`` episode_ids (from ``object_episode_range``) and returns "finished" once exhausted.
    """

    def __init__(self, config: SimplerServerConfig):
        super().__init__(config)
        self._request_listener = RequestListener(self.config.request_listener_config)

        assert (
            config.init_range_config.object_variation_mode == "episode"
        ), "ManiSkill3 bridge envs index the layout by episode_id; set object_variation_mode=episode"
        start, end = config.init_range_config.object_episode_range
        base_ids = list(range(start, end))
        n_envs = config.env_config.n_envs
        assert base_ids, "init_range_config.object_episode_range is empty"
        layouts = config.env_config.episode_layouts
        assert layouts is None or end <= layouts, (
            f"object_episode_range end={end} exceeds env_config.episode_layouts={layouts}; "
            f"episode_ids index into the generated layouts"
        )
        assert len(base_ids) % n_envs == 0 or n_envs % len(base_ids) == 0, (
            f"n_envs={n_envs} must divide or be a multiple of the {len(base_ids)}-episode object range; "
            f"otherwise the final batch is undersized and crashes the batched env"
        )
        if n_envs > len(base_ids) and n_envs % len(base_ids) == 0:
            # More parallel envs than object positions (language-perturbation eval): tile the
            # position layout so every instruction block (assigned by the runner's
            # instruction_override / default_block_size) spans the same object positions, making
            # each perturbation's score directly comparable to the default's. One batch of n_envs.
            self._batches: List[List[int]] = [base_ids * (n_envs // len(base_ids))]
        else:
            self._batches = [base_ids[i : i + n_envs] for i in range(0, len(base_ids), n_envs)]
        self._batch_index = 0

        self._env = SimplerEnv(config=config.env_config)
        self._observations: List[RobotObservation] = []

        self._request_listener.register_handler(RequestType.CONNECT, lambda _: {"status": "ok"})
        self._request_listener.register_handler(RequestType.RESET, self._reset)
        self._request_listener.register_handler(RequestType.GET_OBSERVATIONS, self._get_observations)
        self._request_listener.register_handler(RequestType.EXEC_ACTION, self._execute_action)
        self._request_listener.register_handler(RequestType.EXIT, lambda _: {"status": "ok"})

    def _reset(self, _: Dict[str, Any]) -> Dict[str, Any]:
        """Reset the environment with the next batch of episodes."""
        if self._batch_index >= len(self._batches):
            LOGGER.info("Episodes finished, sending finished status")
            self._observations = []
            return {"status": "finished"}

        episode_ids = self._batches[self._batch_index]
        self._batch_index += 1
        LOGGER.info(f"Starting batch {self._batch_index}/{len(self._batches)} (episodes {episode_ids})")
        self._observations = self._env.reset(episode_ids)
        return {"status": "ok", "n_envs": self._env.n_envs}

    def _get_observations(self, _: Dict[str, Any]) -> Dict[str, Any]:
        """Return the current observations (a single dict at n_envs==1, else a list)."""
        if not self._observations:
            return {"status": "error", "message": "No observation available. Reset the environment first."}
        obs_dicts = [dataclasses.asdict(obs) for obs in self._observations]
        data = obs_dicts[0] if self._env.n_envs == 1 else obs_dicts
        return {"status": "ok", "data": data}

    def _execute_action(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one action per env (RobotEnv sends a single dict at n_envs==1, else {'actions': [...]})."""
        raw_actions = request["actions"] if "actions" in request else [request]
        actions = [RobotAction.from_dict(action) for action in raw_actions]
        self._observations = self._env.step(actions)
        return {"success": True}

    def run(self):
        """Start the server and handle client requests."""
        self._request_listener.start()

    def __del__(self):
        self.close()

    def close(self):
        if (env := getattr(self, "_env", None)) is not None:
            env.close()


if __name__ == "__main__":
    parser = ConfigParser(SimplerServerConfig)
    config: SimplerServerConfig = parser.parse_args()

    simpler_server = SimplerServer(config)

    try:
        simpler_server.run()
    except KeyboardInterrupt:
        LOGGER.info("Ctrl-C pressed, shutting down.")
