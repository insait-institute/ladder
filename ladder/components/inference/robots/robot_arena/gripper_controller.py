"""Force-based gripper control with weld-constraint grasping for Genesis.

Supports ``n_envs=1`` and ``n_envs>=2`` (per-env forces, per-env grasp
detection, per-env weld constraints). Welding is the same physics in both
cases; only the bookkeeping differs.
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple

import genesis
import numpy as np
import torch

from ladder.components.inference.robots.robot_arena.grasping import (
    apply_safety_limits,
    find_finger_link_indices,
    is_grasping_two_finger,
)
from ladder.components.inference.robots.robot_arena.scene_loader import (
    GRIPPER_CLOSE_FORCE,
    GRIPPER_EFFORT_LIMIT,
    GRIPPER_OPEN_FORCE,
    GRIPPER_SAFETY_KD,
    GRIPPER_SAFETY_KP,
    GRIPPER_SOFT_POSITION_LOWER,
    GRIPPER_VELOCITY_LIMIT,
    ObjectConfig,
    RobotConfig,
)
from ladder.components.inference.robots.robot_arena.transforms import RobotName, gripper_genesis_to_canonical
from ladder.core.common.logger.logger import get_logger

LOGGER = get_logger(__name__)

# Divide-by-zero guard for gripper canonicalization (meters).
_GRIPPER_SPAN_EPSILON_M = 1e-6

# Snug hold for welded-carry fingers: at weld formation the finger positions
# (pressed against the object by the closing force) are captured and held there
# minus this over-close for the whole carry. Zero-force fingers drift open
# around the held object (a visible air gap); the over-close also eats the
# ~1-3mm/side visual-vs-collision inflation of the convexified assets. The PD
# gripper gains bound the contact force to ~0.2N, so no rigid-solver NaN risk.
_WELD_FINGER_OVERCLOSE_M = 0.002


class GripperController:
    """Force-based gripper control with weld-constraint grasping. K=1 enters via
    ``open()``/``close()``; K>=2 enters via
    ``step_per_env()`` with per-env open/close commands and per-env welding."""

    def __init__(
        self,
        robot: "genesis.Entity",
        robot_name: RobotName,
        robot_config: RobotConfig,
        scene: "genesis.Scene",
        object_entities: List[Tuple["genesis.Entity", ObjectConfig]],
        n_envs: int = 1,
    ):
        self._robot = robot
        self._robot_name = robot_name
        self._robot_config = robot_config
        self._scene = scene
        self._object_entities = object_entities
        self._n_envs = n_envs

        # env_idx → weld_link_obj. Membership IS the welded state.
        self._welds: Dict[int, np.ndarray] = {}
        # Snug-hold finger targets, one row per env; rows are valid for the envs
        # in _held_envs (captured at weld formation, commanded every substep).
        # _held_envs_idx caches the sorted indices so the substep loop is a
        # single tensor-indexed control call; rebuilt only on weld/release.
        n_gripper_dofs = len(robot_config.gripper_dof_indices or ())
        self._hold_targets: Optional[torch.Tensor] = None
        if n_gripper_dofs:
            self._hold_targets = torch.zeros((n_envs, n_gripper_dofs), device=genesis.device)
        self._held_envs: Set[int] = set()
        self._held_envs_idx: Optional[torch.Tensor] = None

        if robot_config.gripper_dof_indices is not None:
            self._left_finger_idx, self._right_finger_idx = find_finger_link_indices(self._robot)
            ee_link = self._robot.get_link(name=robot_config.ee_link)
            self._ee_link_idx = np.array([ee_link.idx], dtype=genesis.np_int)
        else:
            self._left_finger_idx = None
            self._right_finger_idx = None
            self._ee_link_idx = None

    @property
    def finger_link_indices(self) -> Tuple[Optional[int], Optional[int]]:
        """(left, right) finger link indices. Both None when the robot has no gripper."""
        return self._left_finger_idx, self._right_finger_idx

    def welded_object_indices(self) -> np.ndarray:
        """Per env, the index into ``object_entities`` of the welded object, or -1.

        The weld is the reliable "is it held" signal: once it forms the fingers stop pushing, so
        contact can read as released while the object is in fact carried.
        """
        held = np.full(self._n_envs, -1, dtype=np.int64)
        by_link = {}
        for index, (entity, _) in enumerate(self._object_entities):
            raw = getattr(entity, "underlying_entity", entity)
            by_link[int(raw.links[0].idx)] = index
        for env_idx, link_obj in self._welds.items():
            held[env_idx] = by_link.get(int(np.asarray(link_obj).reshape(-1)[0]), -1)
        return held

    @property
    def welded_mask(self) -> np.ndarray:
        mask = np.zeros(self._n_envs, dtype=bool)
        for env_idx in self._welds:
            mask[env_idx] = True
        return mask

    def reset(self) -> None:
        # Start-of-episode: scene.reset() will wipe physics state right after,
        # so swallow Genesis-side delete failures rather than aborting reset.
        self._release_all_welds(swallow_errors=True)
        self._welds.clear()
        self._held_envs.clear()
        self._held_envs_idx = None

    def open(self, substeps: int) -> None:
        gripper_dof_indices = self._robot_config.gripper_dof_indices

        # Mid-episode: a failed weld release would leave the object pinned to
        # the EE while the fingers open. Propagate the error so the caller can
        # abort the episode rather than producing silently-corrupted physics.
        self._release_all_welds(swallow_errors=False)

        if gripper_dof_indices is not None:
            self._robot.control_dofs_force(
                np.array([GRIPPER_OPEN_FORCE] * len(gripper_dof_indices), dtype=np.float32),
                dofs_idx_local=np.arange(gripper_dof_indices[0], gripper_dof_indices[-1] + 1),
            )

        for _ in range(substeps):
            self._scene.step()

    def close(self, substeps: int) -> None:
        gripper_dof_indices = self._robot_config.gripper_dof_indices
        gripper_open = self._robot_config.gripper_open

        gripper_dof_local = (
            torch.tensor(gripper_dof_indices, device=genesis.device)
            if gripper_dof_indices is not None
            else None
        )

        for _ in range(substeps):
            if gripper_dof_indices is not None:
                close_force = self._compute_close_force(gripper_dof_indices, gripper_open)
                self._robot.control_dofs_force(close_force, dofs_idx_local=gripper_dof_local)
                # Position control issued after the force command wins for the
                # gripper DOFs: a snug-held weld keeps the fingers clamped on
                # the object instead of drifting under the close force.
                if self._held_envs:
                    self._robot.control_dofs_position(self._hold_targets[0], dofs_idx_local=gripper_dof_local)

            self._scene.step()

            if self._n_envs == 1 and 0 not in self._welds and self._left_finger_idx is not None:
                self._maybe_weld_single_env()
                if 0 in self._welds:
                    self._hold_welded_fingers([0])

    def step_per_env(self, gripper_open_per_env: np.ndarray, substeps: int) -> None:
        """K>=2: per-env open/close over substeps. Mirrors the K=1 ``open()`` /
        ``close()`` welding semantics: welds release on opening (mid-episode
        failures propagate); welds form on closing once a per-env grasp is
        detected."""
        assert self._n_envs > 1, "step_per_env is K>=2 only; use open()/close() at K=1"

        opening_envs = [i for i in range(self._n_envs) if bool(gripper_open_per_env[i])]
        closing_envs = [i for i in range(self._n_envs) if not bool(gripper_open_per_env[i])]

        if self._welds:
            envs_to_release = [i for i in opening_envs if i in self._welds]
            if envs_to_release:
                self._release_welds_for_envs(envs_to_release, swallow_errors=False)

        gripper_dof_indices = self._robot_config.gripper_dof_indices
        if gripper_dof_indices is None:
            for _ in range(substeps):
                self._scene.step()
            return

        gripper_open = self._robot_config.gripper_open
        gripper_dof_local = torch.tensor(gripper_dof_indices, device=genesis.device)
        # Mask shape (K, 1) so we can broadcast over the gripper-DOF width.
        open_mask = torch.as_tensor(gripper_open_per_env, device=genesis.device, dtype=torch.bool).reshape(
            -1, 1
        )
        width = len(gripper_dof_indices)
        open_force = torch.full(
            (self._n_envs, width), GRIPPER_OPEN_FORCE, device=genesis.device, dtype=genesis.tc_float
        )

        for _ in range(substeps):
            close_force = self._compute_close_force(gripper_dof_indices, gripper_open)
            command = torch.where(open_mask, open_force, close_force)
            # Once welded, the weld rigidly holds the object to the EE; welded
            # envs get zero closing force (the snug position-hold below takes
            # over) so the -10N close force can't feed an over-constrained
            # finger-contact + weld NaN loop.
            if self._welds:
                welded_mask = torch.zeros((self._n_envs, 1), device=genesis.device, dtype=torch.bool)
                for i_env in self._welds:
                    welded_mask[i_env] = True
                command = torch.where(welded_mask, torch.zeros_like(command), command)
            self._robot.control_dofs_force(command, dofs_idx_local=gripper_dof_local)
            # Position control issued after the force command wins for those
            # envs' gripper DOFs: snug-held welds keep the fingers clamped on
            # the object instead of drifting open at zero force.
            if self._held_envs_idx is not None:
                self._robot.control_dofs_position(
                    self._hold_targets[self._held_envs_idx],
                    dofs_idx_local=gripper_dof_local,
                    envs_idx=self._held_envs_idx,
                )
            self._scene.step()

            if self._left_finger_idx is not None:
                unwelded_closing = [i for i in closing_envs if i not in self._welds]
                if unwelded_closing:
                    newly_welded = self._maybe_weld_per_env(unwelded_closing)
                    if newly_welded:
                        self._hold_welded_fingers(newly_welded)

    def release(self) -> None:
        # End-of-episode: swallow delete failures so cleanup completes.
        if self._scene is not None:
            self._release_all_welds(swallow_errors=True)

    def set_held(
        self,
        target_entity: "genesis.Entity",
        envs_idx_subset: Optional[List[int]] = None,
    ) -> None:
        """Programmatically establish a held state for ``target_entity``.

        Fires the Genesis weld constraint and updates ``_welds`` bookkeeping,
        reaching the same end-state that ``close()`` reaches when contact
        detection succeeds. Unlike ``close()`` (which fires the weld only after
        ``is_grasping_two_finger`` returns true), ``set_held`` is a
        state-assertion: the caller guarantees the target is at the correct
        pose relative to the EE link, this method does not verify it.

        ``envs_idx_subset`` (K>=2 only): when None, weld every env; when set,
        only weld the listed env indices.

        No-op when welding is disabled or no scene exists.
        """
        if self._scene is None:
            return
        if self._ee_link_idx is None:
            raise ValueError("set_held() called on a robot with no gripper / no EE link configured")
        rigid = self._scene.sim.rigid_solver
        link_obj = np.array([target_entity.links[0].idx], dtype=genesis.np_int)

        if self._n_envs == 1:
            if envs_idx_subset is not None and 0 not in envs_idx_subset:
                return
            rigid.add_weld_constraint(link_obj, self._ee_link_idx)
            self._welds[0] = link_obj
            return

        target_env_indices = list(range(self._n_envs)) if envs_idx_subset is None else envs_idx_subset
        envs_idx_t = torch.tensor(target_env_indices, device=genesis.device, dtype=torch.int32)
        rigid.add_weld_constraint(link_obj, self._ee_link_idx, envs_idx=envs_idx_t)
        for i in target_env_indices:
            self._welds[i] = link_obj

    def get_normalized_state(self, joints: np.ndarray) -> np.ndarray:
        """Canonical [0=closed, 1=open] from ``(K, n_dofs)`` joints → ``(K, 1)``.

        For envs whose gripper is currently welded to an object (either via
        a programmatic ``set_held`` or canonical grasp-detect-and-weld), reports
        the gripper as fully closed (0.0) regardless of the actual finger
        position. Rationale: with a Genesis weld constraint holding the object
        to the EE, the gripper fingers physically sit at ~object_width/2 (the
        natural grip width), which maps to a canonical value of ~0.4 here,
        which the policy interprets as "mid-grasp / not gripped" and tries to
        re-grasp by opening, which releases the weld and drops the object.
        Reporting "closed" while welded matches the training distribution (a
        real robot's gripper is fully closed when carrying a small object) and
        keeps a held object from being dropped on the first action.
        """
        gripper_dof_indices = self._robot_config.gripper_dof_indices

        if gripper_dof_indices is not None:
            raw = joints[:, gripper_dof_indices[0]]
            gripper_open = self._robot_config.gripper_open
            gripper_closed = self._robot_config.gripper_closed
            span = gripper_open - gripper_closed
            canonical = (
                (raw - gripper_closed) / span if abs(span) > _GRIPPER_SPAN_EPSILON_M else np.ones_like(raw)
            )
            canonical = np.asarray(canonical, dtype=np.float32).copy()
            for env_idx in self._welds:
                if env_idx < canonical.shape[0]:
                    canonical[env_idx] = 0.0
            return np.clip(canonical, 0.0, 1.0).reshape(-1, 1).astype(np.float32)

        raw_last = joints[:, -1] if joints.shape[1] > 0 else np.zeros(joints.shape[0])
        canonical = np.array(
            [gripper_genesis_to_canonical(self._robot_name, float(v)) for v in raw_last],
            dtype=np.float32,
        )
        for env_idx in self._welds:
            if env_idx < canonical.shape[0]:
                canonical[env_idx] = 0.0
        return canonical.reshape(-1, 1)

    def _compute_close_force(self, gripper_dof_indices: Tuple[int, ...], gripper_open: float) -> torch.Tensor:
        lo = gripper_dof_indices[0]
        hi = gripper_dof_indices[-1] + 1
        # get_dofs_position() returns (n_dofs,) at K=1 and (n_envs, n_dofs) at K>=2.
        # `[..., lo:hi]` matches both; full_like inherits the right shape/dtype/device.
        q_measured = self._robot.get_dofs_position()[..., lo:hi]
        v_measured = self._robot.get_dofs_velocity()[..., lo:hi]
        command = torch.full_like(q_measured, GRIPPER_CLOSE_FORCE)

        return apply_safety_limits(
            command=command,
            q_measured=q_measured,
            v_measured=v_measured,
            motors_soft_position_lower=GRIPPER_SOFT_POSITION_LOWER,
            motors_soft_position_upper=gripper_open,
            motors_effort_limit=GRIPPER_EFFORT_LIMIT,
            motors_velocity_limit=GRIPPER_VELOCITY_LIMIT,
            kp=GRIPPER_SAFETY_KP,
            kd=GRIPPER_SAFETY_KD,
        )

    def _maybe_weld_single_env(self) -> None:
        """K=1 grasp-detect-and-weld. Walks ``object_entities``; welds the first
        object the gripper is grasping. Kept separate from
        ``_maybe_weld_per_env`` so the K=1 numerics stay unchanged."""
        rigid = self._scene.sim.rigid_solver
        for entity, _ in self._object_entities:
            if entity is self._robot:
                continue
            if (
                is_grasping_two_finger(
                    self._robot,
                    entity,
                    self._left_finger_idx,
                    self._right_finger_idx,
                )
                .cpu()
                .numpy()[0]
            ):
                link_obj = np.array([entity.links[0].idx], dtype=genesis.np_int)
                rigid.add_weld_constraint(link_obj, self._ee_link_idx)
                self._welds[0] = link_obj
                break

    def _maybe_weld_per_env(self, env_indices: Sequence[int]) -> List[int]:
        """K>=2 grasp-detect-and-weld for the given (currently-unwelded) envs.
        Returns the env indices that were newly welded this call (so the caller
        can capture their snug-hold targets; see ``_hold_welded_fingers``).
        For each object, runs the per-env grasp check once. Genesis's
        ``add_weld_constraint`` asserts no existing weld with the same link
        pair (globally, not per-env), so when this object's pair is already
        registered for some other envs we delete-then-re-add for the union;
        the kernel state is per-env, so the brief gap between delete and
        re-add doesn't span a physics step."""
        rigid = self._scene.sim.rigid_solver
        remaining = list(env_indices)
        newly_welded: List[int] = []
        for entity, _ in self._object_entities:
            if entity is self._robot or not remaining:
                continue
            per_env_grasp = (
                is_grasping_two_finger(
                    self._robot,
                    entity,
                    self._left_finger_idx,
                    self._right_finger_idx,
                    n_envs=self._n_envs,
                )
                .cpu()
                .numpy()
            )
            link_obj_idx = int(entity.links[0].idx)
            link_obj = np.array([link_obj_idx], dtype=genesis.np_int)
            new_grasping = [i for i in remaining if per_env_grasp[i]]
            if not new_grasping:
                continue
            LOGGER.info(f"[weld] envs={new_grasping} grasp object link={link_obj_idx}")
            existing = sorted(i for i, link in self._welds.items() if int(link[0]) == link_obj_idx)
            if existing:
                existing_idx_t = torch.tensor(existing, device=genesis.device)
                rigid.delete_weld_constraint(link_obj, self._ee_link_idx, envs_idx=existing_idx_t)
            union = sorted(set(existing) | set(new_grasping))
            union_idx_t = torch.tensor(union, device=genesis.device)
            rigid.add_weld_constraint(link_obj, self._ee_link_idx, envs_idx=union_idx_t)
            for env_idx in union:
                self._welds[env_idx] = link_obj
            newly_welded.extend(new_grasping)
            remaining = [i for i in remaining if i not in new_grasping]
        return newly_welded

    def _hold_welded_fingers(self, env_indices: Sequence[int]) -> None:
        """Capture the snug-hold finger targets for freshly-welded envs.

        The fingers are pressed against the object by the closing force at weld
        time; holding them at that width minus ``_WELD_FINGER_OVERCLOSE_M``
        (re-commanded every substep) keeps them visually clamped on the object
        for the whole carry. ``get_normalized_state`` reports welded envs as
        fully closed, so the policy is unaffected.
        """
        gripper_dof_indices = self._robot_config.gripper_dof_indices
        if gripper_dof_indices is None or len(env_indices) == 0:
            return
        gripper_dof_local = torch.tensor(gripper_dof_indices, device=genesis.device)
        if self._n_envs == 1:
            cur = self._robot.get_dofs_position(dofs_idx_local=gripper_dof_local)
            self._hold_targets[0] = torch.clamp(cur - _WELD_FINGER_OVERCLOSE_M, min=0.0)
        else:
            envs_idx_t = torch.tensor(list(env_indices), device=genesis.device, dtype=torch.long)
            cur = self._robot.get_dofs_position(dofs_idx_local=gripper_dof_local, envs_idx=envs_idx_t)
            self._hold_targets[envs_idx_t] = torch.clamp(cur - _WELD_FINGER_OVERCLOSE_M, min=0.0)
        self._held_envs.update(env_indices)
        self._refresh_held_cache()

    def _refresh_held_cache(self) -> None:
        self._held_envs_idx = (
            torch.tensor(sorted(self._held_envs), device=genesis.device, dtype=torch.long)
            if self._held_envs
            else None
        )

    def _release_all_welds(self, swallow_errors: bool) -> None:
        self._release_welds_for_envs(list(self._welds.keys()), swallow_errors)

    def _release_welds_for_envs(self, env_indices: Sequence[int], swallow_errors: bool) -> None:
        """Delete every active weld in ``env_indices``. ``swallow_errors=True``
        is for end-of-life paths (``reset`` / ``release``) where a stale
        Genesis handle is expected; mid-episode callers (open / step_per_env
        on opening envs) pass ``False`` so failures surface instead of
        silently leaving the object pinned."""
        if self._scene is None:
            return
        rigid = self._scene.sim.rigid_solver
        for env_idx in list(env_indices):
            if env_idx not in self._welds:
                continue
            link_obj = self._welds[env_idx]
            envs_idx_t = torch.tensor([env_idx], device=genesis.device) if self._n_envs > 1 else None
            try:
                rigid.delete_weld_constraint(link_obj, self._ee_link_idx, envs_idx=envs_idx_t)
            except Exception as exc:  # noqa: BLE001
                if not swallow_errors:
                    raise
                LOGGER.debug(f"delete_weld_constraint(env={env_idx}) failed: {exc}")
            del self._welds[env_idx]
            self._held_envs.discard(env_idx)
        self._refresh_held_cache()
