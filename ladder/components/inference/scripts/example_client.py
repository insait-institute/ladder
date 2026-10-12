import argparse
import dataclasses
import time
from typing import Any, Dict, Optional

import numpy as np

from ladder.components.inference.api.core.action import EndEffectorPoseAction, ManipulatorAction, RobotAction
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.components.inference.api.core.request_reply import (
    ZmqRequestReplyClient,
    ZmqRequestReplyClientConfig,
)
from ladder.components.inference.api.core.requests import RequestType
from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)


class ExampleClient:
    def __init__(self, ip: str, port: int, recv_timeout_ms: int = 15000):
        self._client = ZmqRequestReplyClient(
            ZmqRequestReplyClientConfig(ip=ip, port=port, recv_timeout_ms=recv_timeout_ms)
        )

    def connect(self, tries: int = 5, delay: float = 1.0):
        for _ in range(tries):
            reply = self._client.send({"type": RequestType.CONNECT, "data": None})
            if reply is not None and reply["status"] == "ok":
                return
            time.sleep(delay)
        raise RuntimeError("Failed to connect to robot")

    def reset(self):
        reply = self._client.send({"type": RequestType.RESET, "data": None})
        if reply is None or reply["status"] != "ok":
            raise RuntimeError("Failed to reset robot")

    def get_observations(self) -> Optional[RobotObservation]:
        reply = self._client.send({"type": RequestType.GET_OBSERVATIONS, "data": None})
        if reply is None or reply["status"] != "ok":
            return None
        data = reply["data"]

        return RobotObservation.from_dict(data)

    def execute_action(self, action: RobotAction) -> Optional[Dict[str, Any]]:
        reply = self._client.send(
            {
                "type": RequestType.EXEC_ACTION,
                "data": dataclasses.asdict(action),
            }
        )
        return reply

    def exit(self):
        """Sends an EXIT request to the server"""
        self._client.send({"type": RequestType.EXIT, "data": None})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Example client script.")

    parser.add_argument("--ip", type=str, default="localhost", help="The server IP address.")

    parser.add_argument("--port", type=int, default=5556, help="The port where the server should listening.")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    client = ExampleClient(args.ip, args.port)

    try:
        client.reset()
        # move 5 times to the left in a loop
        for _ in range(5):
            LOGGER.info("Requesting new observations")
            obs = client.get_observations()
            if obs is None:
                LOGGER.error("Failed to get observations. Stopping.")
                break
            LOGGER.info(f"Received observation:")
            LOGGER.info(f"\tjoints: {obs.state.joints}")
            LOGGER.info(f"\teef pose: {obs.state.eef_pose}")
            LOGGER.info(f"\tgripper state: {obs.state.gripper_state}")
            if obs.images is not None:
                LOGGER.info(f"\timages:\n")
                for k, v in obs.images.items():
                    LOGGER.info(f"\t\t{k}: {v.shape}")
            LOGGER.info(f"Simulating inference with sleep")
            time.sleep(0.3)
            LOGGER.info("Executing new action")

            next_action = ManipulatorAction(
                arm_action=EndEffectorPoseAction.from_homogeneous_matrix(obs.state.eef_pose).apply_delta(
                    delta_rotation=np.eye(3, dtype=np.float32),
                    delta_translation=np.array([0, 0.01, 0.0], dtype=np.float32),
                    delta_mode="rot_eef_trans_base",
                ),
                gripper_action=1.0,
            )

            reply = client.execute_action(next_action)
            if reply is None or not reply["success"]:
                LOGGER.error("Failed to execute action. Stopping.")
                break
            LOGGER.info("Action executed")
        client.reset()
    except KeyboardInterrupt:
        LOGGER.info("Ctrl-C pressed, stopping client.")
