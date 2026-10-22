from typing import Dict, List

from ladder.components.inference.api.client.policy import Policy
from ladder.components.inference.api.client.recorder_wrapper import RecorderWrapper, RecorderWrapperConfig
from ladder.components.inference.api.client.robot_env import RobotEnv, RobotEnvConfig
from ladder.components.inference.api.core.action import RobotAction
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import Config, Configurable

LOGGER = get_logger(__name__)


class IsaacLabEpisodeRunnerConfig(Config):
    env_config: RobotEnvConfig
    recorder_config: RecorderWrapperConfig
    num_episodes: int = 1


class IsaacLabEpisodeRunner(Configurable[IsaacLabEpisodeRunnerConfig]):
    def __init__(self, config: IsaacLabEpisodeRunnerConfig, policy: Policy):
        super().__init__(config)
        self.policy = policy
        self.env = self.wrap_env(RobotEnv(config=self.config.env_config))
        self.stats: Dict[str, Dict[str, List[bool]]] = {}
        self.episode_count = 0

    def wrap_env(self, env: RobotEnv) -> RobotEnv:
        return RecorderWrapper(config=self.config.recorder_config, env=env)

    def run_episode(self):
        self.episode_count += 1
        LOGGER.info(f"Starting episode {self.episode_count} ...")

        self.policy.reset()

        observation: RobotObservation = self.env.reset(instruction="")

        env_name = observation.metadata["env_name"]
        self.stats[env_name] = []

        success, truncated = False, False

        while not truncated and not success:
            action: RobotAction = self.policy.predict_actions(
                [observation], [observation.metadata["instruction"]]
            )[0]
            observation: RobotObservation = self.env.step(action)

            truncated = observation.metadata["truncated"]
            success = observation.metadata["done"]

        self.stats[env_name].append(success)

        self.env.end_episode()

        LOGGER.info(f"Completed episode {self.episode_count}")

    def run(self):
        try:
            while self.episode_count < self.config.num_episodes:
                self.run_episode()
        except StopIteration:
            LOGGER.info("Episodes completed")

        # Save results
        self.save_stats()

    def save_stats(self):
        pass

    def __del__(self):
        self.close()

    def close(self):
        self.env.unwrapped.close()
