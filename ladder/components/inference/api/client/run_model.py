from typing import Optional

from ladder.core.compute.seeder import SEEDER
from ladder.core.databib.config import AutoConfig, AutoConfigurable, Config, ConfigParser


class RunnerConfig(Config):
    policy_config: AutoConfig
    runner_config: AutoConfig
    seed: Optional[int] = None


def main():
    parser = ConfigParser(RunnerConfig)
    config: RunnerConfig = parser.parse_args()

    if config.seed is not None:
        SEEDER.seed_everything(config.seed)

    policy = AutoConfigurable(config.policy_config)
    episode_runner = AutoConfigurable(config.runner_config, policy=policy)
    episode_runner.run()


if __name__ == "__main__":
    main()
