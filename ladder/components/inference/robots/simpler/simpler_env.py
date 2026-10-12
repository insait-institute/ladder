import time
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.spatial.transform
import torch

from ladder.components.inference.api.core.action import EndEffectorPoseAction, RobotAction
from ladder.components.inference.api.core.observation import (
    METADATA_CAMERAS_KEY,
    ManipulatorState,
    RobotObservation,
    encode_camera,
)
from ladder.components.inference.api.core.privileged_state import METADATA_PRIVILEGED_KEY
from ladder.components.inference.robots.simpler.env_builder import build_simpler_env
from ladder.components.inference.robots.simpler.privileged import privileged_state
from ladder.components.inference.robots.simpler.proprio import (
    gripper_control_canonical_to_env,
    rotation_control_canonical_to_env,
    widowx_eef_pose_to_canonical,
)
from ladder.core.databib.config import Config, Configurable

CAMERA_NAME = "3rd_view_camera"
# WidowX gripper finger joint range, from ManiSkill3's stock bridge gripper controller
# (mani_skill/envs/tasks/digital_twins/bridge_dataset_eval/base_env.py: PDJointPosMimicControllerConfig
# lower=0.015, upper=0.037, nominal, excluding its +0.001 grasp clearance). ManiSkill3 reports the raw
# joint position, so normalise it to the [0, 1] openness the policy expects (ManiSkill2 served this
# pre-normalised in obs['agent']['eef_pos'][7]).
GRIPPER_LOWER_M = 0.015
GRIPPER_UPPER_M = 0.037


class SimplerEnvConfig(Config):
    env_name: str
    robot_name: str
    scene_name: str
    rgb_overlay_path: str  # ManiSkill3 derives the overlay from env_name; kept to document the eval

    sim_freq: int
    control_mode: str  # Determines the space and reference frames for the actions sent to the environment
    control_freq: int

    max_episode_steps: int = 80
    raytracing: bool = False
    n_envs: int = 8  # parallel ManiSkill3 envs held on one GPU
    # Composition envs only: how many object-position layouts each scene generates (episode_id indexes
    # into them). None = the env's own default; must cover init_range_config.object_episode_range.
    episode_layouts: Optional[int] = None
    # Publish ground-truth geometry for the scripted policy; off so normal runs carry no payload.
    expose_privileged: bool = False
    publish_depth: bool = False
    # Off renders and publishes no images, for a policy that reads none (the scripted one).
    # Rendering never touches physics, so the episodes are the same.
    render_images: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert "delta" not in self.control_mode, "Delta control mode not supported! Use base"


def _to_np(x: Any) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


class SimplerEnv(Configurable[SimplerEnvConfig]):
    """ManiSkill3 widowx bridge env holding ``n_envs`` parallel envs on one GPU.

    ``reset``/``step`` return length-``n_envs`` lists of RobotObservation, so the K>=2 episode
    runner drives it the same way it drives RobotArena.
    """

    def __init__(self, config: SimplerEnvConfig):
        super().__init__(config)
        self._env = self._create_env()
        self._agent = self._env.unwrapped.agent
        self._env_id = self._agent.uid  # e.g. "widowx250s_bridgedataset_flat_table"
        self._arm = self._agent.controller.controllers["arm"]
        self._ee_link = self._arm.ee_link
        self._device = self._env.unwrapped.device
        self._start_timestamp = round(time.time())
        self._obs: Dict[str, Any] = {}
        self._step = 0
        self._truncated = np.zeros(config.n_envs, dtype=bool)
        self._done = np.zeros(config.n_envs, dtype=bool)
        # Full per-step info from evaluate() (e.g. composition per-goal stats).
        self._info: Dict[str, Any] = {}

    @property
    def n_envs(self) -> int:
        return self.config.n_envs

    def _create_env(self):
        return build_simpler_env(
            env_name=self.config.env_name,
            n_envs=self.config.n_envs,
            control_mode=self.config.control_mode,
            sim_freq=self.config.sim_freq,
            control_freq=self.config.control_freq,
            max_episode_steps=self.config.max_episode_steps,
            episode_layouts=self.config.episode_layouts,
            publish_depth=self.config.publish_depth,
            render_images=self.config.render_images,
        )

    def reset(self, episode_ids: List[int]) -> List[RobotObservation]:
        options = {"episode_id": torch.tensor(episode_ids, dtype=torch.long, device=self._device)}
        self._obs, _ = self._env.reset(seed=episode_ids, options=options)
        self._step = 0
        self._truncated[:] = False
        self._done[:] = False
        self._info = {}
        return self._make_observations()

    def step(self, actions: List[RobotAction]) -> List[RobotObservation]:
        env_actions = torch.tensor(
            np.stack([self._to_env_action(action) for action in actions]), device=self._device
        )
        self._obs, _reward, _terminated, truncated, info = self._env.step(env_actions)
        self._step += 1
        self._truncated = _to_np(truncated).astype(bool).reshape(-1)
        self._done = _to_np(info["success"]).astype(bool).reshape(-1)
        self._info = info
        return self._make_observations()

    def _to_env_action(self, action: RobotAction) -> np.ndarray:
        if not isinstance(action.arm_action, EndEffectorPoseAction):
            raise NotImplementedError(f"Action type {type(action.arm_action)} not yet supported")
        rotmat = rotation_control_canonical_to_env(self._env_id, action.arm_action.rotation)
        # ManiSkill3's absolute PDEEPoseController reads the rotation as Euler XYZ; the ManiSkill2
        # fork read it as axis-angle (the two coincide only near pure single-axis rotations).
        euler = scipy.spatial.transform.Rotation.from_matrix(rotmat).as_euler("XYZ")  # [3]
        gripper = gripper_control_canonical_to_env(self._env_id, action.gripper_action)  # sign
        return np.concatenate([action.arm_action.translation, euler, np.atleast_1d(gripper)]).astype(
            np.float32
        )

    def _camera_payloads(self) -> Optional[List[Dict[str, Any]]]:
        """Per-env calibrated depth frames, or None when depth was not requested.

        ``extrinsic_cv`` is already the OpenCV world-to-camera transform, so only the inverse and
        the move into the robot's base frame remain.
        """
        if not self.config.publish_depth:
            return None
        sensor = self._obs["sensor_data"][CAMERA_NAME]
        params = self._obs["sensor_param"][CAMERA_NAME]
        depth = _to_np(sensor["depth"])  # [n_envs, H, W, 1]
        # ManiSkill stores depth as integer millimetres to halve the buffer; floats are metres.
        depth_m = depth.astype(np.float32) / 1000.0 if depth.dtype.kind in "iu" else depth.astype(np.float32)
        if depth_m.ndim == 4 and depth_m.shape[-1] == 1:
            depth_m = depth_m[..., 0]
        intrinsics = _to_np(params["intrinsic_cv"])
        extrinsics = _to_np(params["extrinsic_cv"])
        base_from_world = _to_np(self._agent.robot.pose.inv().to_transformation_matrix())
        payloads: List[Dict[str, Any]] = []
        for i in range(self.n_envs):
            cam_from_world = np.eye(4)
            cam_from_world[:3, :4] = np.asarray(extrinsics[i] if extrinsics.ndim == 3 else extrinsics)
            base = np.asarray(base_from_world[i] if base_from_world.ndim == 3 else base_from_world)
            payloads.append(
                encode_camera(
                    depth_m=depth_m[i],
                    intrinsics=intrinsics[i] if intrinsics.ndim == 3 else intrinsics,
                    base_from_camera=base @ np.linalg.inv(cam_from_world),
                )
            )
        return payloads

    def _make_observations(self) -> List[RobotObservation]:
        rgb = (
            _to_np(self._obs["sensor_data"][CAMERA_NAME]["rgb"]) if self.config.render_images else None
        )  # [n_envs, H, W, 3]
        cameras = self._camera_payloads()
        ee_at_base = self._agent.robot.pose.inv() * self._ee_link.pose
        ee_p, ee_q = _to_np(ee_at_base.p), _to_np(ee_at_base.q)  # base-frame EE pose (q is wxyz)
        target = _to_np(self._arm._target_pose.raw_pose)  # noqa: SLF001  # [n_envs, 7] base-frame target pose
        qpos = _to_np(self._agent.robot.get_qpos())  # [n_envs, n_joints]
        instructions = self._env.unwrapped.get_language_instruction()
        timestamps = self._start_timestamp + np.array([self._step / self.config.control_freq])

        # Surface composition per-goal eval signals into per-env metadata when the env's evaluate()
        # produced them. Gated on key presence, so non-composition envs (and the golden
        # test_simpler_server metadata checksum) are unaffected.
        privileged = privileged_state(self._env.unwrapped) if self.config.expose_privileged else None

        subgoals = self._info.get("num_subgoals_complete")
        subgoals_np = _to_np(subgoals).reshape(-1) if subgoals is not None else None
        goal_keys = sorted(k for k in self._info if k.startswith("success_") and "_on_" in k)
        goal_arrays = {k: _to_np(self._info[k]).reshape(-1) for k in goal_keys}

        observations = []
        for i in range(self.n_envs):
            openness = np.clip((qpos[i, 6] - GRIPPER_LOWER_M) / (GRIPPER_UPPER_M - GRIPPER_LOWER_M), 0.0, 1.0)
            metadata: Dict[str, Any] = {
                "instruction": str(instructions[i]),
                "env_name": self.config.env_name,
                "robot_name": self.config.robot_name,
                "truncated": bool(self._truncated[i]),
                "done": bool(self._done[i]),
            }
            if subgoals_np is not None:
                metadata["num_subgoals_complete"] = int(subgoals_np[i])
            for k, arr in goal_arrays.items():
                metadata[k] = bool(arr[i])
            if privileged is not None:
                metadata[METADATA_PRIVILEGED_KEY] = privileged[i]
            if cameras is not None:
                metadata[METADATA_CAMERAS_KEY] = {"main": cameras[i]}
            # Oracle contract: an env that publishes its ordered subtasks + goal keys gets metadata["oracle"]
            # = {subtasks, per-goal achieved}, which a benchmark-agnostic OracleController consumes.
            oracle_subtasks = getattr(self._env.unwrapped, "oracle_subtasks", None)
            if oracle_subtasks is not None:
                metadata["oracle"] = {
                    "subtasks": oracle_subtasks,
                    # An env whose goals are scored as a sequence publishes it; PrivilegedDecomposer
                    # passes it to the advancer, so the oracle arm walks the chain in order.
                    "ordered": bool(getattr(self._env.unwrapped, "oracle_ordered", False)),
                    # A repeated subtask moves the SAME object again, so a plan consumer must not
                    # read it as a second copy. Plan structure, not world state.
                    "revisits": bool(getattr(self._env.unwrapped, "oracle_revisits", False)),
                    # Per LEG. An env that publishes subtasks publishes this vector alongside them.
                    "achieved": [bool(v) for v in self._env.unwrapped.oracle_achieved[i]],
                }
            observations.append(
                RobotObservation(
                    # widowx bridge has no wrist cam; the model uses main only
                    images={"main": rgb[i].copy()} if rgb is not None else None,
                    state=ManipulatorState(
                        eef_pose=widowx_eef_pose_to_canonical(np.concatenate([ee_p[i], ee_q[i]])),
                        eef_pose_from_target=widowx_eef_pose_to_canonical(target[i]),
                        gripper_state=np.array([openness], dtype=np.float32),
                        joints=qpos[i, :7].astype(np.float32),
                    ),
                    timestamps=timestamps,
                    metadata=metadata,
                )
            )
        return observations

    def close(self):
        self._env.close()
