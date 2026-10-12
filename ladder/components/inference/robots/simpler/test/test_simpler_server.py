import os
import socket
import threading
import unittest

# SAPIEN writes a physx cache under $HOME/.sapien; the Bazel test sandbox's HOME isn't writable.
os.environ["HOME"] = os.environ["TEST_TMPDIR"]

import numpy as np
import scipy.spatial.transform

from ladder.components.inference.api.core.action import EndEffectorPoseAction, ManipulatorAction
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.components.inference.robots.simpler.simpler_server import SimplerServer, SimplerServerConfig
from ladder.components.inference.scripts.example_client import ExampleClient
from ladder.core.compute.checksum import calculate_data_checksum
from ladder.core.compute.seeder import SEEDER


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run_with_catch_system_exit(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SystemExit:
            pass

    return wrapper


class TestSimplerBridge(unittest.TestCase):
    def setUp(self) -> None:
        SEEDER.reset()
        SEEDER.seed_everything(42)
        self.prng = SEEDER.new_prng_np()

    def test_simpler_server(self):
        server_port = _free_port()

        server_config = SimplerServerConfig.from_yaml(
            "ladder/components/inference/robots/simpler/configs/widowx_put_carrot_on_plate.yaml"
        ).replace(
            **{
                "env_config.n_envs": 1,
                "request_listener_config.req_rep_server_config.port": server_port,
            }
        )
        server = SimplerServer(config=server_config)

        server_thread = threading.Thread(target=run_with_catch_system_exit(server.run), daemon=True)
        server_thread.start()

        client = ExampleClient(ip="127.0.0.1", port=server_port)

        try:
            client.connect()
            client.reset()

            actual_checksums = {
                "action.translation": [],
                "action.rotation": [],
                "action.gripper": [],
                "obs.images.main": [],
                "obs.eef_pose": [],
                "obs.eef_pose_from_target": [],
                "obs.gripper": [],
                "obs.joints": [],
                "obs.metadata": [],
            }

            for _ in range(10):
                obs: RobotObservation = client.get_observations()
                self.assertIsNotNone(obs)
                self.assertEqual(tuple(obs.images["main"].shape), (480, 640, 3))
                self.assertEqual(tuple(obs.state.eef_pose.shape), (4, 4))
                self.assertEqual(tuple(obs.state.eef_pose_from_target.shape), (4, 4))
                self.assertEqual(tuple(obs.state.gripper_state.shape), (1,))
                self.assertEqual(tuple(obs.state.joints.shape), (7,))
                self.assertEqual(tuple(obs.timestamps.shape), (1,))

                action = ManipulatorAction(
                    arm_action=EndEffectorPoseAction(
                        translation=self.prng.uniform(low=-0.04, high=0.04, size=(3,)),
                        rotation=scipy.spatial.transform.Rotation.from_euler(
                            "xyz", self.prng.uniform(low=-10, high=10, size=(3,)), degrees=True
                        ).as_matrix(),
                    ),
                    gripper_action=self.prng.uniform(low=0.0, high=1.0, size=None),
                )
                client.execute_action(action)

                actual_checksums["obs.images.main"].append(
                    calculate_data_checksum(obs.images["main"], hashed=True)
                )
                actual_checksums["obs.eef_pose"].append(
                    calculate_data_checksum(obs.state.eef_pose, hashed=False)
                )
                actual_checksums["obs.eef_pose_from_target"].append(
                    calculate_data_checksum(obs.state.eef_pose_from_target, hashed=False)
                )
                actual_checksums["obs.gripper"].append(
                    calculate_data_checksum(obs.state.gripper_state, hashed=False)
                )
                actual_checksums["obs.joints"].append(calculate_data_checksum(obs.state.joints, hashed=False))
                actual_checksums["obs.metadata"].append(calculate_data_checksum(obs.metadata, hashed=True))
                actual_checksums["action.translation"].append(
                    calculate_data_checksum(action.arm_action.translation, hashed=False)
                )
                actual_checksums["action.rotation"].append(
                    calculate_data_checksum(action.arm_action.rotation, hashed=False)
                )
                actual_checksums["action.gripper"].append(float(action.gripper_action))

            # ManiSkill3 (sapien3) golden values, captured on an A6000 under the torch 2.12.1+cu130
            # simpler env and verified bit-identical across repeated runs (deterministic at n_envs=1
            # with seeded actions).
            expected_checksums = {
                "action.translation": [
                    0.03825306782582364,
                    0.010404106870140137,
                    0.02465674447663366,
                    0.023138523890328014,
                    0.04452886914787978,
                    0.02835620112961792,
                    0.036635287324962594,
                    0.01860372785783774,
                    0.07302826928524955,
                    0.07661200231465476,
                ],
                "action.rotation": [
                    3.5405729975256603,
                    3.57750259294571,
                    3.3607786809569937,
                    3.4460663212742437,
                    3.4094843043810226,
                    3.4823059886710044,
                    3.3333380258702103,
                    3.6061749905417795,
                    3.57110717849808,
                    3.572731850178184,
                ],
                "action.gripper": [
                    0.2982278092806382,
                    0.5056424728082645,
                    0.7074211367900608,
                    0.1674791832212973,
                    0.2667126114535301,
                    0.7400512567832603,
                    0.8074304189703639,
                    0.40140861248018245,
                    0.9426121364625666,
                    0.27051064470247443,
                ],
                "obs.images.main": [
                    17684424679886460840,
                    15222597730126664895,
                    298179027175260522,
                    11361513756118446283,
                    379408445644577588,
                    11315163377292648421,
                    1398749360715686577,
                    17857324056025953885,
                    5952031299452038068,
                    16550416030900962526,
                ],
                "obs.eef_pose": [
                    4.589108467102051,
                    6.116303443908691,
                    6.0634074211120605,
                    5.8827595710754395,
                    5.794885635375977,
                    5.873481750488281,
                    5.995680809020996,
                    6.004488945007324,
                    5.986588478088379,
                    5.953713893890381,
                ],
                "obs.eef_pose_from_target": [
                    4.589108467102051,
                    4.578826427459717,
                    4.587906837463379,
                    4.385435104370117,
                    4.469204902648926,
                    4.454013347625732,
                    4.510662078857422,
                    4.369973182678223,
                    4.624778747558594,
                    4.644135475158691,
                ],
                "obs.gripper": [
                    1.0,
                    0.34678030014038086,
                    0.8276044130325317,
                    0.0,
                    0.001980457454919815,
                    1.6552141460124403e-05,
                    0.6616578102111816,
                    0.9047512412071228,
                    0.30438199639320374,
                    0.7754132747650146,
                ],
                "obs.joints": [
                    1.6890974044799805,
                    6.500315189361572,
                    8.31800365447998,
                    7.491808891296387,
                    6.7446136474609375,
                    6.696233749389648,
                    6.78200101852417,
                    6.803215980529785,
                    6.761592388153076,
                    6.600477695465088,
                ],
                "obs.metadata": [17033207187794648444] * 10,
            }

            float_keys = {
                "action.translation",
                "action.rotation",
                "action.gripper",
                "obs.eef_pose",
                "obs.eef_pose_from_target",
                "obs.gripper",
                "obs.joints",
            }
            for key, expected in expected_checksums.items():
                with self.subTest(component=key):
                    if key in float_keys:
                        self.assertTrue(np.all(np.isclose(actual_checksums[key], expected)))
                    else:
                        self.assertEqual(actual_checksums[key], expected)

        finally:
            client.exit()
            server_thread.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
