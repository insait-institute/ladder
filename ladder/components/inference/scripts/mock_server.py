import dataclasses
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from ladder.components.inference.api.core.observation import ManipulatorState, RobotObservation
from ladder.components.inference.api.core.request_reply import (
    ZmqRequestReplyServer,
    ZmqRequestReplyServerConfig,
)
from ladder.components.inference.api.core.requests import RequestType, parse_request_type
from ladder.core.compute.seeder import SEEDER
from ladder.core.databib.config import Config, ConfigParser, Configurable

RequestHandler = Callable[[Dict[str, Any] | None], Dict[str, Any]]


class MockServerConfig(Config):
    port: int = 5556
    env_name: str = "test_env"
    robot_name: str = "widowx"
    instruction: str = "default server instruction"
    image_height: int = 480
    image_width: int = 640
    episode_steps: int = 10
    num_episodes: int = 1
    seed: Optional[int] = None


class MockServer(Configurable[MockServerConfig]):
    def __init__(self, config: MockServerConfig):
        super().__init__(config)
        self._request_reply_server = ZmqRequestReplyServer(ZmqRequestReplyServerConfig(port=self.config.port))
        self._request_handlers: Dict[RequestType, RequestHandler] = {}
        self._episode_step: int = -1
        self._episodes_started: int = 0

        if self.config.seed is not None:
            SEEDER.seed_everything(self.config.seed)
        self.prng = SEEDER.new_prng_np()

        self.register_handler(RequestType.CONNECT, lambda _: {"status": "ok", "data": None})
        self.register_handler(RequestType.RESET, self._reset)
        self.register_handler(RequestType.GET_OBSERVATIONS, self._get_observations)
        self.register_handler(RequestType.EXEC_ACTION, self._execute_action)
        self.register_handler(RequestType.EXIT, lambda _: {"status": "ok", "data": None})

    def register_handler(self, request_type: RequestType, handler: RequestHandler) -> None:
        if request_type in self._request_handlers:
            raise ValueError(f"Handler already registered for {request_type}.")
        self._request_handlers[request_type] = handler

    def _make_observation(self, done: bool, truncated: bool) -> RobotObservation:
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = self.prng.uniform(low=-0.1, high=0.1, size=(3,)).astype(np.float32)
        main_image = self.prng.randint(
            low=0,
            high=256,
            size=(self.config.image_height, self.config.image_width, 3),
            dtype=np.uint8,
        )
        wrist_image = self.prng.randint(
            low=0,
            high=256,
            size=(self.config.image_height, self.config.image_width, 3),
            dtype=np.uint8,
        )
        return RobotObservation(
            images={"main": main_image, "wrist": wrist_image},
            state=ManipulatorState(
                eef_pose=pose,
                eef_pose_from_target=pose.copy(),
                gripper_state=self.prng.uniform(low=0.0, high=1.0, size=(1,)).astype(np.float32),
                joints=self.prng.uniform(low=-1.0, high=1.0, size=(7,)).astype(np.float32),
            ),
            timestamps=np.array([time.time()], dtype=np.float64),
            metadata={
                "env_name": self.config.env_name,
                "robot_name": self.config.robot_name,
                "instruction": self.config.instruction,
                "done": done,
                "truncated": truncated,
            },
        )

    def _reset(self, _: Dict[str, Any] | None) -> Dict[str, Any]:
        if self._episodes_started >= self.config.num_episodes:
            return {"status": "finished", "data": None}

        self._episodes_started += 1
        self._episode_step = 0
        return {"status": "ok", "data": None}

    def _get_observations(self, _: Dict[str, Any] | None) -> Dict[str, Any]:
        if self._episode_step < 0:
            return {"status": "error", "message": "You need to reset before getting observations"}

        done = self._episode_step >= self.config.episode_steps
        observation = self._make_observation(done=done, truncated=done)
        return {"status": "ok", "data": dataclasses.asdict(observation)}

    def _execute_action(self, _: Dict[str, Any] | None) -> Dict[str, Any]:
        if self._episode_step < 0:
            return {"status": "error", "message": "You need to reset before executing actions"}

        self._episode_step += 1
        return {"status": "ok", "success": True}

    def _on_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        request_type = parse_request_type(request)
        handler = self._request_handlers.get(request_type)
        if handler is None:
            raise ValueError(f"No handler registered for request type: {request_type}")
        return handler(request.get("data"))

    def run(self) -> None:
        self._request_reply_server.start(self._on_request)


if __name__ == "__main__":
    parser = ConfigParser(MockServerConfig)
    config: MockServerConfig = parser.parse_args()
    mock_server = MockServer(config=config)
    mock_server.run()
