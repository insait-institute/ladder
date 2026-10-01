"""Genesis-based RobotArena env. K=1 is bit-identical to the legacy path;
K>=2 replicates the scene and steps all envs in lockstep with per-env (dx,
dy) reset offsets, per-env gripper commands, and per-env weld-constraint
grasping. ``step``/``reset`` always return length-K lists."""

import time
from typing import List, Optional, Tuple, Union

import genesis
import numpy as np
import scipy.spatial.transform
import torch

from ladder.components.inference.api.core.action import EndEffectorPoseAction, RobotAction
from ladder.components.inference.api.core.observation import RobotObservation, named_effector_states
from ladder.components.inference.api.core.privileged_state import METADATA_PRIVILEGED_KEY
from ladder.components.inference.robots.robot_arena.composition.registry import COMPOSITION_TASKS
from ladder.components.inference.robots.robot_arena.composition.tasks import compose_scene_data
from ladder.components.inference.robots.robot_arena.gripper_controller import GripperController
from ladder.components.inference.robots.robot_arena.observation import ObservationBuilder
from ladder.components.inference.robots.robot_arena.scene_builder import (
    SceneContext,
    build_genesis_scene,
    resolve_object_overlaps,
)
from ladder.components.inference.robots.robot_arena.scene_loader import (
    ROBOT_CONFIGS,
    SETTLE_STEPS,
    SceneData,
    compute_camera_to_world,
    load_scene,
)
from ladder.components.inference.robots.robot_arena.success_checker import (
    BatchedSuccessChecker,
    MultiGoalSuccessChecker,
)
from ladder.components.inference.robots.robot_arena.transforms import (
    RobotName,
    canonical_pose_to_genesis_target,
    gripper_control_canonical_to_genesis,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
)
from ladder.core.common.logger.logger import get_logger
from ladder.core.databib.config import Config, Configurable

LOGGER = get_logger(__name__)


class RobotArenaEnvConfig(Config):
    """Setting ``per_env_object_offsets`` to a length-K list activates K>=2 mode."""

    data_root: str
    scene_name: str
    robot_name: RobotName
    max_episode_steps: int
    control_freq: int
    sim_freq: int = 200
    # Rigid-solver substeps per physics step. >1 shrinks the rigid integration
    # timestep to (1/sim_freq)/sim_substeps without changing the control cadence
    # (substeps_per_action stays sim_freq//control_freq). Genesis raises
    # "Invalid constraint forces causing 'nan'. Please decrease Rigid simulation
    # timestep." when an over-constrained loop (weld + finger contact on a
    # grasped object) diverges; >1 here is the fix.
    sim_substeps: int = 1
    # Rigid constraint time constant (s). Larger = softer constraints, which
    # bounds the force a deep settle interpenetration or an over-constrained
    # grasp weld generates. None keeps the Genesis default (0.01).
    rigid_constraint_timeconst: Optional[float] = None
    # Publish ground-truth geometry for the scripted policy; off so normal runs carry no payload.
    expose_privileged: bool = False
    publish_depth: bool = False
    # Off publishes no images, for a policy that reads none (the scripted one). Rendering never
    # touches physics, so the episodes are the same.
    render_images: bool = True
    camera_width: int
    camera_height: int

    background_path: Optional[str] = None
    camera_offset: Optional[Tuple[float, float, float]] = None
    object_positions: Optional[List[Tuple[float, float, float]]] = None
    object_rotations: Optional[List[Tuple[float, float, float, float]]] = None

    # K>=2 only; len() becomes n_envs.
    per_env_object_offsets: Optional[List[Tuple[float, float]]] = None

    # Name of a long-horizon task in composition.COMPOSITION_TASKS. When set, the env scores a
    # multi-goal conjunction over the scene's objects and reports the composed instruction.
    composition_task: Optional[str] = None

    @property
    def n_envs(self) -> int:
        return len(self.per_env_object_offsets) if self.per_env_object_offsets else 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.per_env_object_offsets:
            return
        # Length-1 lists silently took the K=1 path and dropped the offset.
        # Force the user to be explicit (drop the field entirely for K=1).
        if len(self.per_env_object_offsets) < 2:
            raise ValueError(
                f"per_env_object_offsets must have length >= 2 (got {len(self.per_env_object_offsets)}); "
                "drop the field entirely for n_envs=1"
            )
        if self.object_positions is not None or self.object_rotations is not None:
            raise ValueError(
                "per_env_object_offsets is incompatible with object_positions / object_rotations"
            )


class RobotArenaEnv(Configurable[RobotArenaEnvConfig]):
    """Genesis RobotArena env. ``step``/``reset`` always return length-``n_envs`` lists."""

    @property
    def substeps_per_action(self) -> int:
        return self.config.sim_freq // self.config.control_freq

    @property
    def n_envs(self) -> int:
        return self._n_envs

    def __init__(self, config: RobotArenaEnvConfig):
        super().__init__(config)
        if config.sim_freq % config.control_freq != 0:
            raise ValueError(
                f"sim_freq ({config.sim_freq}) must be divisible by control_freq ({config.control_freq})"
            )

        self._n_envs = config.n_envs
        self._scene_data: SceneData = load_scene(
            data_root=config.data_root,
            scene_name=config.scene_name,
            background_path=config.background_path,
            object_positions=config.object_positions,
            object_rotations=config.object_rotations,
        )

        if config.composition_task is not None:
            # Composed instruction (obs builder + logs read it) + any imported objects /
            # receptacle scale overrides the task defines.
            self._scene_data = compose_scene_data(
                self._scene_data, COMPOSITION_TASKS[config.composition_task], config.data_root
            )

        self._ctx: SceneContext = build_genesis_scene(
            self._scene_data,
            data_root=config.data_root,
            sim_freq=config.sim_freq,
            sim_substeps=config.sim_substeps,
            rigid_constraint_timeconst=config.rigid_constraint_timeconst,
            robot_name=config.robot_name,
            camera_width=config.camera_width,
            camera_height=config.camera_height,
            camera_offset=config.camera_offset,
            n_envs=self._n_envs,
        )

        # Envs step in lockstep, so a scalar suffices.
        self._step_count = 0
        self._start_timestamp = round(time.time())
        # Always (K, 4, 4); reused as eef_pose_from_target on the next obs build.
        self._last_target_pose_canonical: Optional[np.ndarray] = None

        robot_cfg = ROBOT_CONFIGS[self.config.robot_name]
        self._gripper = GripperController(
            robot=self._ctx.robot,
            robot_name=self.config.robot_name,
            robot_config=robot_cfg,
            scene=self._ctx.scene,
            object_entities=self._ctx.object_entities,
            n_envs=self._n_envs,
        )

        self._obs_builder = ObservationBuilder(
            cameras=self._ctx.cameras,
            robot=self._ctx.robot,
            robot_name=self.config.robot_name,
            world_to_base=self._ctx.world_to_base,
            gripper=self._gripper,
            scene_data=self._scene_data,
            # Composition cells report the task name so their results/logs don't
            # collide with the underlying scene's canonical cell.
            config_scene_name=self.config.composition_task or self.config.scene_name,
            control_freq=self.config.control_freq,
            n_envs=self._n_envs,
            publish_depth=self.config.publish_depth,
            render_images=self.config.render_images,
            camera_fov=self._ctx.camera_fov,
            camera_pose=self._ctx.camera_pose,
        )

        # One batched success checker evaluates all K envs in a single GPU pass. For a composition
        # task it is a MultiGoalSuccessChecker (one BatchedSuccessChecker per goal).
        self._success_checker: Optional[Union[BatchedSuccessChecker, MultiGoalSuccessChecker]] = None

    def reset(self) -> List[RobotObservation]:
        self._step_count = 0
        self._start_timestamp = round(time.time())
        self._last_target_pose_canonical = None
        self._gripper.reset()

        self._ctx.scene.reset()

        robot_config = ROBOT_CONFIGS[self.config.robot_name]
        if robot_config.initial_qpos is not None:
            initial_qpos = np.array(robot_config.initial_qpos, dtype=np.float32)
            self._ctx.robot.set_qpos(initial_qpos)
            # Hold the arm at initial_qpos through the settle below. reset() only
            # sets the qpos, not a position target, so without an active controller
            # the arm drifts off the tucked home pose during SETTLE_STEPS (toward
            # the default ~zero target) into an out-of-distribution, scene-occluding
            # start pose. step() re-commands the target on the first action, so this
            # governs only the settle window.
            if self._n_envs == 1:
                self._ctx.robot.control_dofs_position(initial_qpos)
            else:
                self._ctx.robot.control_dofs_position(
                    torch.as_tensor(np.tile(initial_qpos, (self._n_envs, 1)), device=genesis.device)
                )

        for cam in self._ctx.cameras:
            cam.set_pose(self._ctx.camera_pose)

        if self._n_envs == 1:
            self._apply_object_overrides(self.config.object_positions, self.config.object_rotations)
            resolve_object_overlaps(self._ctx.object_entities)
        else:
            self._apply_per_env_offsets()

        for _ in range(SETTLE_STEPS):
            self._ctx.scene.step()

        self._build_success_checkers()

        observations = self._obs_builder.build(
            step_count=0,
            start_timestamp=self._start_timestamp,
            last_target_pose_canonical=None,
        )
        success_infos = self._success_checker.check_all()
        for env_i, obs in enumerate(observations):
            obs.metadata.update(success_infos[env_i])
        self._inject_privileged(observations, success_infos)

        if self._n_envs == 1:
            first = observations[0]
            arms = ", ".join(
                f"{arm}: eef_pos={np.array2string(state.eef_pose[:3, 3], precision=4)} "
                f"gripper={float(np.ravel(state.gripper_state)[0]):.3f}"
                for arm, state in named_effector_states(first.state)
            )
            LOGGER.info(
                f"[reset] scene={self.config.scene_name}, "
                f"instruction='{self._scene_data.task_instruction}', {arms}"
            )
        else:
            LOGGER.info(
                f"[batched reset] scene={self.config.scene_name} n_envs={self._n_envs} "
                f"instruction='{self._scene_data.task_instruction}'"
            )
        return observations

    def step(self, actions: List[RobotAction]) -> List[RobotObservation]:
        if len(actions) != self._n_envs:
            raise ValueError(f"step expects {self._n_envs} actions, got {len(actions)}")
        for action in actions:
            if not isinstance(action.arm_action, EndEffectorPoseAction):
                raise NotImplementedError(f"Action type {type(action.arm_action)} not supported")

        (
            target_pos_world_batched,
            target_quat_world_batched,
            target_canonical_batched,
            gripper_cmds,
        ) = self._batch_actions(actions)

        self._last_target_pose_canonical = target_canonical_batched

        if self._n_envs == 1:
            self._apply_action_single(
                target_pos_world_batched[0], target_quat_world_batched[0], float(gripper_cmds[0])
            )
        else:
            self._apply_actions_batched(
                target_pos_world_batched,
                target_quat_world_batched,
                gripper_cmds > 0.5,
            )

        self._step_count += 1
        if self._n_envs == 1 and (self._step_count < 3 or self._step_count % 10 == 0):
            LOGGER.info(
                f"[step {self._step_count}] action: "
                f"t_canonical={np.array2string(actions[0].arm_action.translation, precision=4)}, "
                f"t_world={np.array2string(target_pos_world_batched[0], precision=4)}, "
                f"gripper={'OPEN' if gripper_cmds[0] > 0.5 else 'CLOSE'}"
            )

        observations = self._obs_builder.build(
            step_count=self._step_count,
            start_timestamp=self._start_timestamp,
            last_target_pose_canonical=self._last_target_pose_canonical,
        )
        truncated = self._step_count >= self.config.max_episode_steps
        success_infos = self._success_checker.check_all()
        for env_i, obs in enumerate(observations):
            obs.metadata.update(success_infos[env_i])
            obs.metadata["truncated"] = truncated
        self._inject_privileged(observations, success_infos)
        return observations

    def _inject_privileged(self, observations, success_infos) -> None:
        """Attach the scripted policy's ground-truth payload, if this run asked for it."""
        if not self.config.expose_privileged:
            return
        # A composition task publishes per-goal credit; an ordinary scene task has the one goal its
        # own done flag scores.
        achieved = [
            list(info["oracle"]["achieved"]) if isinstance(info.get("oracle"), dict) else [bool(info["done"])]
            for info in success_infos
        ]
        payloads = self._success_checker.privileged_state(
            self._ctx.world_to_base, achieved, self._gripper.welded_object_indices()
        )
        for env_i, obs in enumerate(observations):
            obs.metadata[METADATA_PRIVILEGED_KEY] = payloads[env_i]

    def can_reuse_for(self, new_config: "RobotArenaEnvConfig") -> bool:
        return (
            self.config.scene_name == new_config.scene_name
            and self.config.robot_name == new_config.robot_name
            and self.config.n_envs == new_config.n_envs
            # The composed instruction, conjunction checker, and reported env_name are fixed at
            # construction — reconfigure() cannot swap them, so a composition change must rebuild.
            and self.config.composition_task == new_config.composition_task
        )

    def reconfigure(self, new_config: "RobotArenaEnvConfig") -> None:
        """Apply lightweight config changes without rebuilding the Genesis scene."""
        # Validate before mutating any state — a half-applied reconfigure
        # leaves _config and _ctx out of sync. (new_config is field-validated at
        # construction via __post_init__; the n_envs guard is reconfigure-specific.)
        old_config = self.config
        if new_config.n_envs != old_config.n_envs:
            raise ValueError(
                f"n_envs cannot change via reconfigure ({old_config.n_envs} -> {new_config.n_envs}); "
                "rebuild the env instead."
            )

        self._config = new_config

        self._ctx.camera_pose = compute_camera_to_world(
            self._scene_data.camera_extrinsics,
            camera_offset=new_config.camera_offset,
        )
        for cam in self._ctx.cameras:
            cam.set_pose(self._ctx.camera_pose)

        if new_config.background_path != old_config.background_path:
            self._scene_data.background_path = new_config.background_path
            self._obs_builder.clear_background_cache()

        if self._n_envs == 1 and (
            new_config.object_positions is not None or new_config.object_rotations is not None
        ):
            self._apply_object_overrides(new_config.object_positions, new_config.object_rotations)
            resolve_object_overlaps(self._ctx.object_entities)

    def end_episode(self) -> None:
        self._gripper.release()

    def close(self) -> None:
        if self._ctx.scene is not None:
            self._ctx.scene.destroy()
            self._ctx.scene = None

    def _apply_object_overrides(
        self,
        object_positions: Optional[List[Tuple[float, float, float]]],
        object_rotations: Optional[List[Tuple[float, float, float, float]]],
    ) -> None:
        """K=1 only. Caller runs ``resolve_object_overlaps`` afterward."""
        if object_positions is None and object_rotations is None:
            return
        for i, (entity, _obj_cfg) in enumerate(self._ctx.object_entities):
            if object_positions is not None and i < len(object_positions):
                entity.set_pos(object_positions[i])
            if object_rotations is not None and i < len(object_rotations):
                entity.set_quat(object_rotations[i])

    # K>=2 skips resolve_object_overlaps; cap kept small enough that small XY
    # shifts can't introduce new interpenetration.
    _MAX_PER_ENV_OFFSET_M = 0.05

    def _apply_per_env_offsets(self) -> None:
        envs_idx_t = torch.arange(self._n_envs, device=genesis.device)
        offsets = np.asarray(self.config.per_env_object_offsets, dtype=np.float32)
        max_abs_offset = float(np.abs(offsets).max()) if offsets.size else 0.0
        if max_abs_offset > self._MAX_PER_ENV_OFFSET_M:
            raise ValueError(
                f"per_env_object_offsets max |offset| = {max_abs_offset:.3f} m exceeds "
                f"safety cap of {self._MAX_PER_ENV_OFFSET_M:.3f} m"
            )

        for entity, obj_cfg in self._ctx.object_entities:
            if obj_cfg.fixed:
                continue
            base_xyz = np.asarray(obj_cfg.position, dtype=np.float32)
            per_env_xyz = np.tile(base_xyz[None, :], (self._n_envs, 1))
            per_env_xyz[:, 0] += offsets[:, 0]
            per_env_xyz[:, 1] += offsets[:, 1]
            entity.set_pos(torch.as_tensor(per_env_xyz, device=genesis.device), envs_idx=envs_idx_t)
            base_quat = np.asarray(obj_cfg.rotation_quat, dtype=np.float32)
            per_env_quat = np.tile(base_quat[None, :], (self._n_envs, 1))
            entity.set_quat(torch.as_tensor(per_env_quat, device=genesis.device), envs_idx=envs_idx_t)

    def _build_success_checkers(self) -> None:
        robot_config = ROBOT_CONFIGS[self.config.robot_name]
        ee_link = self._ctx.robot.get_link(name=robot_config.ee_link)
        common = dict(
            scene_name=self.config.scene_name,
            object_entities=self._ctx.object_entities,
            n_envs=self._n_envs,
            camera_extrinsics=self._scene_data.camera_extrinsics,
            robot_ee_link=ee_link,
        )
        if self.config.composition_task is not None:
            task = COMPOSITION_TASKS[self.config.composition_task]
            self._success_checker = MultiGoalSuccessChecker(
                task.goals, ordered=task.ordered, milestones=task.milestones, **common
            )
        else:
            self._success_checker = BatchedSuccessChecker(
                task_instruction=self._scene_data.task_instruction, **common
            )

    def _batch_actions(
        self, actions: List[RobotAction]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos_list, quat_list, canonical_list, grip_list = [], [], [], []
        for action in actions:
            target_pos_base, target_quat_base = canonical_pose_to_genesis_target(
                self.config.robot_name,
                action.arm_action.translation,
                action.arm_action.rotation,
            )
            target_base_mat = np.eye(4, dtype=np.float64)
            target_base_mat[:3, :3] = scipy.spatial.transform.Rotation.from_quat(
                quat_wxyz_to_xyzw(target_quat_base)
            ).as_matrix()
            target_base_mat[:3, 3] = target_pos_base
            target_world_mat = self._ctx.base_to_world @ target_base_mat
            target_pos_world = target_world_mat[:3, 3].astype(np.float32)
            target_quat_xyzw = scipy.spatial.transform.Rotation.from_matrix(
                target_world_mat[:3, :3]
            ).as_quat()
            target_quat_world = quat_xyzw_to_wxyz(target_quat_xyzw).astype(np.float32)

            target_canonical = np.eye(4, dtype=np.float32)
            target_canonical[:3, :3] = action.arm_action.rotation
            target_canonical[:3, 3] = action.arm_action.translation

            pos_list.append(target_pos_world)
            quat_list.append(target_quat_world)
            canonical_list.append(target_canonical)
            grip_list.append(
                gripper_control_canonical_to_genesis(self.config.robot_name, action.gripper_action)
            )

        return (
            np.stack(pos_list, axis=0),
            np.stack(quat_list, axis=0),
            np.stack(canonical_list, axis=0),
            np.asarray(grip_list, dtype=np.float32),
        )

    def _apply_action_single(
        self, target_pos: np.ndarray, target_quat: np.ndarray, gripper_cmd: float
    ) -> None:
        """K=1 path — bit-identical to the legacy implementation."""
        robot_config = ROBOT_CONFIGS[self.config.robot_name]
        ee_link = self._ctx.robot.get_link(name=robot_config.ee_link)
        des_q = self._ctx.robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat)

        gripper_dof_indices = robot_config.gripper_dof_indices
        is_opening = gripper_cmd > 0.5
        if gripper_dof_indices is not None:
            assert robot_config.gripper_open is not None
            assert robot_config.gripper_closed is not None
            gripper_value = robot_config.gripper_open if is_opening else robot_config.gripper_closed
            for idx in gripper_dof_indices:
                des_q[idx] = gripper_value

        self._ctx.robot.control_dofs_position(des_q)
        if is_opening:
            self._gripper.open(self.substeps_per_action)
        else:
            self._gripper.close(self.substeps_per_action)

    def _apply_actions_batched(
        self,
        target_pos_world_batched: np.ndarray,
        target_quat_world_batched: np.ndarray,
        gripper_open_per_env: np.ndarray,
    ) -> None:
        robot_config = ROBOT_CONFIGS[self.config.robot_name]
        ee_link = self._ctx.robot.get_link(name=robot_config.ee_link)

        pos_t = torch.as_tensor(target_pos_world_batched, device=genesis.device)
        quat_t = torch.as_tensor(target_quat_world_batched, device=genesis.device)
        des_q = self._ctx.robot.inverse_kinematics(link=ee_link, pos=pos_t, quat=quat_t)

        gripper_dof_indices = robot_config.gripper_dof_indices
        if gripper_dof_indices is not None:
            assert robot_config.gripper_open is not None
            assert robot_config.gripper_closed is not None
            gripper_value_per_env = np.where(
                gripper_open_per_env, robot_config.gripper_open, robot_config.gripper_closed
            ).astype(np.float32)
            gripper_value_t = torch.as_tensor(gripper_value_per_env, device=genesis.device)
            for idx in gripper_dof_indices:
                des_q[:, idx] = gripper_value_t

        self._ctx.robot.control_dofs_position(des_q)
        self._gripper.step_per_env(gripper_open_per_env, self.substeps_per_action)
