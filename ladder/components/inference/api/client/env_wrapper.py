from typing import Any, Dict, List, TypeVar

from ladder.components.inference.api.client.robot_env import RobotEnv
from ladder.components.inference.api.core.action import RobotAction
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.core.databib.config import Config, Configurable


class EnvWrapperConfig(Config):
    pass


EnvWrapperConfigT = TypeVar('EnvWrapperConfigT', bound=EnvWrapperConfig)


class EnvWrapper(Configurable[EnvWrapperConfigT]):
    """
    Base class for a wrapper around a RobotEnv. Overload to execute additional functionality with every
    env operation, for example, logging, recording videos, action clipping, etc.

    Example:
    ```python

    env = RobotEnv(...)
    env = VideoWrapper(env)

    for i in range(num_steps):
        obs = env.reset()
        action = ...
        obs = env.step(action)
    ```
    """

    # def __init__(self, config: EnvWrapperConfigT, env: RobotEnv | "EnvWrapper"):
    def __init__(self, config: EnvWrapperConfigT, env):
        super().__init__(config)
        self.env = env

    def __del__(self):
        self.close()

    def close(self) -> None:
        pass

    def reset(self, **kwargs) -> RobotObservation:
        return self.env.reset(**kwargs)

    def step(self, action: RobotAction, metadata: Dict[str, Any] | None = None) -> RobotObservation:
        return self.env.step(action=action, metadata=metadata)

    def reset_all(self, **kwargs) -> List[RobotObservation]:
        return self.env.reset_all(**kwargs)

    def step_all(
        self, actions: List[RobotAction], metadata: Dict[str, Any] | None = None
    ) -> List[RobotObservation]:
        return self.env.step_all(actions=actions, metadata=metadata)

    def end_episode(self) -> None:
        self.env.end_episode()

    @property
    def unwrapped(self) -> RobotEnv:
        """Returns the unwrapped RobotEnv."""
        if isinstance(self.env, EnvWrapper):
            return self.env.unwrapped
        return self.env

    def __repr__(self):
        return f'{self.__class__.__name__}({self.env})'
