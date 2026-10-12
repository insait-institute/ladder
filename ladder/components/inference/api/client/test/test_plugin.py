import socket
import textwrap
import threading
from typing import Dict, List

import numpy as np
import zmq

from ladder.components.inference.api.client.plugin.agent import ArmConfig, PluginAgent, PluginAgentConfig
from ladder.components.inference.api.client.plugin.orchestrator import (
    PluginOrchestrator,
    PluginOrchestratorConfig,
)
from ladder.components.inference.api.client.plugin.remote_policy import RemotePolicy, RemotePolicyConfig
from ladder.components.inference.api.core.observation import ManipulatorState, RobotObservation
from ladder.components.inference.api.core.serialization import deserialize, serialize


def _obs(xyz=(0.3, 0.0, 0.2), gripper=1.0, metadata=None) -> RobotObservation:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.diag([1.0, -1.0, -1.0])  # pointing down
    pose[:3, 3] = xyz
    return RobotObservation(
        images={"main": np.zeros((4, 6, 3), dtype=np.uint8)},
        state=ManipulatorState(eef_pose=pose, gripper_state=np.array([gripper], dtype=np.float32)),
        timestamps=None,
        metadata=metadata or {"robot_name": "widowx", "oracle": {"achieved": [True]}},
    )


def _system(tmp_path, source: str, name: str) -> str:
    path = tmp_path / "my_system.py"
    path.write_text(textwrap.dedent(source))
    return f"{path}:{name}"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def test_remote_policy_sends_what_a_robot_sees_and_plays_a_chunk_before_asking_again():
    port = _free_port()
    requests: List[Dict] = []

    def server():
        rep = zmq.Context().socket(zmq.REP)
        rep.bind(f"tcp://*:{port}")
        for _ in range(2):
            request = deserialize(rep.recv())
            requests.append(request)
            chunk = []
            for obs in request["observations"]:
                poses = np.stack([obs["eef_pose"]] * 2)
                poses[1, 2, 3] -= 0.01
                chunk.append({"pose": poses, "gripper": [1.0, 0.0]})
            rep.send(serialize({"actions": chunk}))
        rep.close()

    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    policy = RemotePolicy(RemotePolicyConfig(address=f"localhost:{port}", timeout_s=10))
    policy.reset()
    observations = [_obs(), _obs(xyz=(0.4, 0.1, 0.2))]
    first = policy.predict_actions(observations, ["task a", "task b"])
    second = policy.predict_actions(observations, ["task a", "task b"])
    policy.predict_actions(observations, ["task a", "task b"])
    thread.join(timeout=10)

    assert len(requests) == 2  # the second step played the chunk's second action
    sent = requests[0]["observations"]
    assert [o["env"] for o in sent] == [0, 1] and all(o["first"] for o in sent)
    assert not any(o["first"] for o in requests[1]["observations"])
    assert sent[1]["instruction"] == "task b" and sent[1]["images"]["main"].shape == (4, 6, 3)
    assert "oracle" not in str(sent)  # no privileged state crosses the wire
    np.testing.assert_allclose(first[1].arm_action.translation, [0.4, 0.1, 0.2], atol=1e-6)
    np.testing.assert_allclose(second[1].arm_action.translation, [0.4, 0.1, 0.19], atol=1e-6)
    assert (first[0].gripper_action, second[0].gripper_action) == (1.0, 0.0)


def test_orchestrator_keeps_one_object_per_episode_and_records_its_plan(tmp_path):
    system = _system(
        tmp_path,
        """
        class CountingOrchestrator:
            def __init__(self):
                self.plan = ["first", "second"]
                self.calls = 0

            def instruction(self, obs):
                self.calls += 1
                return obs["instruction"] + f" {self.calls}"
        """,
        "CountingOrchestrator",
    )
    orchestrator = PluginOrchestrator(PluginOrchestratorConfig(system=system))
    orchestrator.reset("pick", env_idx=0)
    orchestrator.reset("place", env_idx=1)
    assert orchestrator.get_commands([_obs(), _obs()], step=0) == ["pick 1", "place 1"]
    assert orchestrator.get_commands([_obs(), _obs()], step=1) == ["pick 2", "place 2"]
    orchestrator.reset("pick", env_idx=0)  # a new episode starts from a new object
    assert orchestrator.get_commands([_obs(), _obs()], step=0) == ["pick 1", "place 3"]
    assert orchestrator.plan_for(None, env_idx=1) == ["first", "second"]


def test_agent_program_moves_the_fingertips_and_closes_the_gripper(tmp_path):
    system = _system(
        tmp_path,
        """
        class Reach:
            def run(self, arm):
                arm.move_to([0.35, 0.05, 0.1])
                arm.close()
        """,
        "Reach",
    )
    offset = 0.05
    agent = PluginAgent(
        PluginAgentConfig(
            system=system,
            arm=ArmConfig(
                tool_offset_m=offset,
                max_step_m=0.02,
                settle_steps=5,
                gripper_steps=5,
                workspace_min_m=[-1.0, -1.0, 0.0],
                workspace_max_m=[1.0, 1.0, 1.0],
            ),
            timeout_s=10,
        )
    )
    agent.reset()
    observation = _obs()
    for _ in range(40):  # a simulator that reaches every commanded pose at once
        action = agent.predict_actions([observation], ["reach"])[0]
        observation = _obs(xyz=action.arm_action.translation, gripper=action.gripper_action)
    flange = observation.state.eef_pose[:3, 3]
    np.testing.assert_allclose(flange - [0.0, 0.0, offset], [0.35, 0.05, 0.1], atol=1e-5)
    assert observation.state.gripper_state[0] == 0.0
