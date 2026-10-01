import dataclasses
import time
from typing import Any, Dict, List

from ladder.components.inference.api.core.action import RobotAction
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.components.inference.api.core.request_reply import (
    ZmqRequestReplyClient,
    ZmqRequestReplyClientConfig,
)
from ladder.components.inference.api.core.requests import RequestType
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import Config, Configurable

LOGGER = get_logger(__name__)


class RobotEnvConfig(Config):
    ip: str = 'localhost'
    port: int = 5556
    raise_on_server_error: bool = False  # If True, raise an error if the server fails to execute the action
    min_period: float = 0.0
    recv_timeout_ms: int = 15000  # ZMQ receive timeout in milliseconds


class RobotEnv(Configurable[RobotEnvConfig]):
    """Client for any RobotServer/SimplerServer/RobotArenaServer variant.

    ``n_envs`` is determined by the server on the first RESET; at K=1 all
    existing single-env callers work unchanged.
    """

    def __init__(self, config: RobotEnvConfig):
        super().__init__(config)
        self._client = ZmqRequestReplyClient(
            ZmqRequestReplyClientConfig(
                ip=config.ip, port=config.port, recv_timeout_ms=config.recv_timeout_ms
            )
        )
        self._n_envs: int = 1
        self._obs_timestamp: float = 0.0
        self._closed = False

    @property
    def n_envs(self) -> int:
        return self._n_envs

    def reset(self, **kwargs) -> RobotObservation:
        observations = self.reset_all(**kwargs)
        self._require_single_env("reset")
        return observations[0]

    def step(self, action: RobotAction, metadata: Dict[str, Any] | None = None) -> RobotObservation:
        self._require_single_env("step")
        observations = self.step_all([action], metadata=metadata)
        return observations[0]

    def reset_all(self, **kwargs) -> List[RobotObservation]:
        del kwargs
        reply = self._client.send({"type": RequestType.RESET, "data": None})

        if reply is None:
            raise RuntimeError("Robot server failed to reset.")
        if reply["status"] == "finished":
            self.close()
            raise StopIteration("Episodes finished")
        if reply["status"] != "ok":
            raise RuntimeError(f"Robot server reset returned {reply.get('status')!r}")

        # Backward compat: legacy servers omit n_envs → treat as single-env.
        self._n_envs = int(reply.get("n_envs", 1))
        return self._get_observations()

    def step_all(
        self,
        actions: List[RobotAction],
        metadata: Dict[str, Any] | None = None,
    ) -> List[RobotObservation]:
        del metadata
        if len(actions) != self._n_envs:
            raise ValueError(f"step_all expects {self._n_envs} actions, got {len(actions)}")

        reply = self._execute_actions(actions)
        if self.config.raise_on_server_error and (reply is None or not reply.get("success")):
            raise RuntimeError("Robot server failed to execute action.")

        self._wait_for_period()
        return self._get_observations()

    def _require_single_env(self, call: str) -> None:
        if self._n_envs > 1:
            raise RuntimeError(
                f"{call}() is only valid for n_envs=1 (server advertised "
                f"n_envs={self._n_envs}); use {call}_all() for batched mode."
            )

    def _execute_actions(self, actions: List[RobotAction]) -> Dict[str, Any] | None:
        # K=1 keeps the legacy single-action payload (MockServer/SimplerServer/RobotServer
        # don't parse the batched form).
        if self._n_envs == 1:
            payload: Dict[str, Any] = dataclasses.asdict(actions[0])
        else:
            payload = {"actions": [dataclasses.asdict(action) for action in actions]}
        return self._client.send({"type": RequestType.EXEC_ACTION, "data": payload})

    def _get_observations(self) -> List[RobotObservation]:
        reply = self._client.send({"type": RequestType.GET_OBSERVATIONS, "data": None})
        if reply is None or reply["status"] != "ok":
            raise RuntimeError("Failed to fetch observation from robot server.")

        data = reply["data"]
        if self._n_envs == 1:
            self._obs_timestamp = float(data["timestamps"][0])
            return [RobotObservation.from_dict(data)]

        if not isinstance(data, list) or len(data) != self._n_envs:
            raise RuntimeError(
                f"Expected list of {self._n_envs} observations, got {type(data).__name__} "
                f"(len={len(data) if isinstance(data, list) else 'n/a'})"
            )
        return [RobotObservation.from_dict(obs_dict) for obs_dict in data]

    def _wait_for_period(self) -> None:
        # NOTE: This is approximate solution and doesn't account for time taken to
        # communicate with the robot server. Additionally, there's nothing that can
        # be done if model inference is slower than the control frequency.
        if self.config.min_period > 0.0 and self._n_envs == 1:
            prev_timestamp = self._obs_timestamp
            next_timestamp = prev_timestamp + self.config.min_period
            current_timestamp = time.time()
            diff = next_timestamp - current_timestamp
            if diff > 0:
                time.sleep(diff)

    def end_episode(self) -> None:
        pass

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._client.send({"type": RequestType.EXIT, "data": None})
        finally:
            self._closed = True
