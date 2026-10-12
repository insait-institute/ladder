"""A LADDER policy server: your policy, in its own Python environment, answering the simulator.

    pip install pyzmq msgpack msgpack-numpy numpy
    python examples/policy_server.py --port 8000
    scripts/eval.sh --suite simplerenv_long --policy-server localhost:8000 --name my_policy

Every control step, the environments that need an action send one request with their observations,
and get one action (or a chunk of actions) each back. Replace MyPolicy.act with your model; nothing
else needs to change. This file imports nothing from LADDER, so it runs wherever your model does.
"""

import argparse

import msgpack
import msgpack_numpy
import zmq

msgpack_numpy.patch()  # numpy arrays travel as arrays


class MyPolicy:
    def __init__(self) -> None:
        self.model = None  # load your model here

    def act(self, obs: dict) -> dict:
        """One environment's observation in, its next action out.

        obs["instruction"]  the task, e.g. "put the carrot and the spoon on the plate"
        obs["images"]       {camera: [H, W, 3] uint8}: "main" on the WidowX suites, "scene" and "wrist" on RoboLab
        obs["eef_pose"]     [4, 4] end-effector pose in the robot base frame (metres)
        obs["gripper"]      1.0 open ... 0.0 closed
        obs["joints"]       [n] joint positions (radians)
        obs["robot"]        "widowx" or "franka"
        obs["env"]          which environment this is (episodes run in parallel); keep state per env
        obs["first"]        True on the first step of an episode: reset that env's state
        obs["step"]         control steps since the episode started

        Return the absolute end-effector pose to go to and the gripper command (1 open, 0 closed). A
        chunk of k actions is {"pose": [k, 4, 4], "gripper": [k]}, played one per step.
        """
        pose = obs["eef_pose"].copy()  # this policy holds still; a model would move the arm, e.g.
        # pose[:3, 3] += delta_xyz      (a relative action becomes absolute by adding it to the pose)
        return {"pose": pose, "gripper": obs["gripper"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    policy = MyPolicy()
    socket = zmq.Context().socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")
    print(f"policy server listening on port {args.port}")
    while True:
        request = msgpack.unpackb(socket.recv())
        # one call per environment; a model that runs batched can take the whole list at once
        actions = [policy.act(obs) for obs in request["observations"]]
        socket.send(msgpack.packb({"actions": actions}))


if __name__ == "__main__":
    main()
