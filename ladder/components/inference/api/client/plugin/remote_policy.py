from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from ladder.components.inference.api.client.plugin.system import observation_dict
from ladder.components.inference.api.core.action import EndEffectorPoseAction, ManipulatorAction
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.components.inference.api.core.request_reply import (
    ZmqRequestReplyClient,
    ZmqRequestReplyClientConfig,
)
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import Config, Configurable

LOGGER = get_logger(__name__)


class RemotePolicyConfig(Config):
    address: str = "localhost:8000"  # host:port of the policy server
    timeout_s: float = 600.0  # one reply; the first call may load a model


class RemotePolicy(Configurable[RemotePolicyConfig]):
    """A policy that runs in its own process, in any Python environment, on any machine: every step
    the environments that need an action send their observations to a ZMQ server, which answers
    with one action, or a chunk of actions, per environment. ``examples/policy_server.py`` is a
    complete server.

    Request: ``{"observations": [...]}``, one per environment, as ``plugin.system.observation_dict``
    writes them plus ``env`` (the environment index, for per-environment state) and ``first`` (true
    on the first step of an episode). Reply: ``{"actions": [...]}`` in the same order, each
    ``{"pose": [4, 4] or [k, 4, 4], "gripper": float or [k]}``, the absolute end-effector pose(s) in
    the robot base frame and the gripper command(s) (1 open, 0 closed). A chunk of k actions is
    played one per step before the environment asks again.
    """

    def __init__(self, config: RemotePolicyConfig) -> None:
        super().__init__(config)
        host, _, port = config.address.rpartition(":")
        self._client = ZmqRequestReplyClient(
            ZmqRequestReplyClientConfig(
                ip=host or "localhost",
                port=int(port),
                recv_timeout_ms=int(config.timeout_s * 1000),
                propagate_timeout=True,
            )
        )
        self._queues: Dict[int, Deque[ManipulatorAction]] = {}
        self._steps: Dict[int, int] = {}

    def reset(self) -> None:
        self._queues = {}
        self._steps = {}

    def predict_actions(
        self,
        observations: List[RobotObservation],
        instructions: List[str],
        env_indices: Optional[List[int]] = None,
    ) -> List[ManipulatorAction]:
        indices = list(env_indices) if env_indices is not None else list(range(len(observations)))
        asking = [i for i, env in enumerate(indices) if not self._queues.get(env)]
        if asking:
            request = []
            for i in asking:
                env = indices[i]
                obs = observation_dict(observations[i], instructions[i], self._steps.get(env, 0))
                request.append({**obs, "env": env, "first": env not in self._steps})
            reply = self._client.send({"observations": request})
            if reply is None or len(reply.get("actions", [])) != len(asking):
                raise RuntimeError(
                    f"the policy server at {self.config.address} sent no valid reply: {reply!r}"
                )
            for i, action in zip(asking, reply["actions"], strict=True):
                self._queues[indices[i]] = deque(_actions(action))
        actions = []
        for env in indices:
            self._steps[env] = self._steps.get(env, 0) + 1
            actions.append(self._queues[env].popleft())
        return actions


def _actions(reply: Dict[str, Any]) -> List[ManipulatorAction]:
    poses = np.asarray(reply["pose"], dtype=np.float32).reshape(-1, 4, 4)
    grippers = np.broadcast_to(np.asarray(reply["gripper"], dtype=np.float32).reshape(-1), (len(poses),))
    return [
        ManipulatorAction(
            arm_action=EndEffectorPoseAction(translation=pose[:3, 3].copy(), rotation=pose[:3, :3].copy()),
            gripper_action=float(gripper),
        )
        for pose, gripper in zip(poses, grippers, strict=True)
    ]
