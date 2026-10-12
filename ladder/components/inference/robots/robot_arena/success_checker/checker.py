import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ladder.components.inference.api.core.privileged_geometry import build_payloads, to_base
from ladder.components.inference.robots.robot_arena.scene_loader import ObjectConfig
from ladder.components.inference.robots.robot_arena.success_checker.matching import match_objects
from ladder.components.inference.robots.robot_arena.success_checker.parsing import parse_instruction
from ladder.components.inference.robots.robot_arena.success_checker.predicates import (
    _compute_camera_axes_xy,
    is_between,
    is_in_table_region,
    is_inside,
    is_near,
    is_on_top_of,
    is_picked_up,
    is_relative_to,
)
from ladder.components.inference.robots.robot_arena.success_checker.types import (
    CheckResult,
    ResolvedTask,
    SimEntity,
    SuccessThresholds,
    TaskSpec,
    TaskType,
)

# Fixed WidowX-bridge tabletop extent in world XY (min_x, max_x, min_y, max_y),
# shared by every scene's region tasks.
FIXED_TABLE_BOUNDS_M: Tuple[float, float, float, float] = (-0.10, 0.63, -0.13, 0.24)

# Task type -> the motion the scripted policy has to run. The types absent from this map are
# scored on where the object ENDS UP, so they place; these end at the grasp or the reach.
_SCRIPTED_PREDICATE = {
    TaskType.PICKUP: "pickup",
    TaskType.GRASP: "grasp",
    TaskType.REACH: "reach",
}

# Named table region -> (right_axis_sign, down_axis_sign), matching is_in_table_region. An axis the
# region does not constrain gets 0, which aims at the middle of that axis.
_TABLE_REGION_SIGNS = {
    "upper_left": (-1.0, -1.0),
    "upper": (0.0, -1.0),
    "upper_right": (1.0, -1.0),
    "left": (-1.0, 0.0),
    "center": (0.0, 0.0),
    "right": (1.0, 0.0),
    "lower_left": (-1.0, 1.0),
    "lower": (0.0, 1.0),
    "lower_right": (1.0, 1.0),
}

# Camera-frame direction -> (right_axis_sign, down_axis_sign), matching is_relative_to.
_RELATIVE_DIRECTION_SIGNS = {
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "upper": (0.0, -1.0),
    "back": (0.0, -1.0),
    "lower": (0.0, 1.0),
    "front": (0.0, 1.0),
}

# Per-scene region/burner goal points (world XY + radius_m); scenes not listed fall
# back to is_in_table_region.
REGION_TARGETS_XY_R: dict = {
    "scene2": (0.343, 0.025, 0.09),
    "scene6": (0.372, -0.048, 0.09),
    "scene8": (0.337, -0.124, 0.16),
    "scene10": (0.291, -0.048, 0.09),
    "scene21": (0.252, 0.078, 0.082),
    "scene33": (0.276, -0.063, 0.09),
    "scene34": (0.248, 0.026, 0.09),
    "scene37": (0.277, -0.015, 0.05),
    "scene38": (0.269, 0.182, 0.09),
    "scene45": (0.399, -0.170, 0.075),
    "scene46": (0.393, 0.043, 0.077),
    "scene55": (0.303, 0.213, 0.132),
    "scene57": (0.338, -0.037, 0.12),
    "scene76": (0.414, -0.152, 0.115),
    "scene82": (0.204, 0.017, 0.16),
    "scene7": (0.453, 0.219, 0.09),
    "scene12": (0.354, 0.014, 0.09),
    "scene14": (0.328, 0.114, 0.09),
    "scene18": (0.357, 0.182, 0.05),
    "scene24": (0.357, 0.208, 0.14),
    "scene28": (0.329, 0.104, 0.09),
    "scene30": (0.368, 0.153, 0.087),
    "scene36": (0.360, 0.219, 0.119),
    "scene80": (0.452, -0.054, 0.09),
    "scene84": (0.432, 0.367, 0.16),
    "scene85": (0.472, 0.134, 0.116),
}


def _to_kxn(value, n_envs: int, device: torch.device, cols: int = 3) -> torch.Tensor:
    t = value if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value))
    t = t.to(device=device, dtype=torch.float32)
    if t.ndim == 1:
        t = t.reshape(1, cols).expand(n_envs, cols)
    return t.reshape(n_envs, cols)


def _split_aabb(aabb):
    """Split an entity AABB into (min, max), accepting a (min, max) tuple or a (..., 2, 3) tensor."""
    if isinstance(aabb, (tuple, list)):
        return aabb[0], aabb[1]
    arr = aabb if isinstance(aabb, torch.Tensor) else torch.as_tensor(np.asarray(aabb))
    return arr[..., 0, :], arr[..., 1, :]


def privileged_payloads(
    snapshot_of: "BatchedSuccessChecker",
    world_to_base: np.ndarray,
    goals: Sequence[Dict[str, Any]],
    n_envs: int,
    held_indices=None,
) -> List[Dict[str, Any]]:
    """Assemble privileged payloads from one scene snapshot and a list of goals over it.

    The snapshot is the whole scene rather than any one goal's target, so the policy can be
    grounded onto any object; a multi-goal task takes the same snapshot and passes several goals.
    """
    names, centers, mins, maxs, grasped = snapshot_of.privileged_snapshot(held_indices)
    base_yaw = float(np.arctan2(world_to_base[1, 0], world_to_base[0, 0]))
    return build_payloads(
        names=names,
        centers=to_base(world_to_base, centers),
        mins=to_base(world_to_base, mins),
        maxs=to_base(world_to_base, maxs),
        grasped=grasped,
        goals=goals,
        n_envs=n_envs,
        long_axis_yaws=snapshot_of.privileged_long_axes() + base_yaw,
    )


class BatchedSuccessChecker:
    """Evaluates binary task success for a batch of envs in one GPU pass.

    Resolves a single scene task and evaluates it over the K-env batch (or a
    subset via ``env_indices``); ``check_all`` returns one CheckResult per scored
    env. Grasp is computed over the full batch and sliced to the subset.
    """

    def __init__(
        self,
        task_instruction: str,
        scene_name: str,
        object_entities: List[Tuple[SimEntity, ObjectConfig]],
        n_envs: int,
        env_indices: Optional[Sequence[int]] = None,
        camera_extrinsics: Optional[np.ndarray] = None,
        thresholds: Optional[SuccessThresholds] = None,
        robot_ee_link: Optional[SimEntity] = None,
        task_spec_override: Optional[TaskSpec] = None,
        robot_entity: Optional[object] = None,
        left_finger_idx: Optional[int] = None,
        right_finger_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self._thresholds = thresholds or SuccessThresholds()
        self._entities = object_entities
        self._scene_name = scene_name
        self._robot_ee_link = robot_ee_link
        self._robot_entity = robot_entity
        self._left_finger_idx = left_finger_idx
        self._right_finger_idx = right_finger_idx
        self._n_envs = n_envs
        self._status = "ok"
        self._device = device or self._infer_device(object_entities)

        idx = list(range(n_envs)) if env_indices is None else list(env_indices)
        self._idx = torch.as_tensor(idx, dtype=torch.long, device=self._device)
        self._g = len(idx)

        self._image_right_xy, self._image_down_xy = _compute_camera_axes_xy(camera_extrinsics)

        self._task_spec = task_spec_override or parse_instruction(task_instruction, scene_name)
        if self._task_spec is None:
            raise ValueError(f"Failed to parse task instruction for scene={scene_name}: '{task_instruction}'")
        if self._task_spec.task_type == TaskType.GRASP and (
            robot_entity is None or left_finger_idx is None or right_finger_idx is None
        ):
            raise ValueError(f"GRASP task requires robot_entity + finger indices (scene={scene_name})")

        object_configs = [cfg for _, cfg in object_entities]
        resolved = match_objects(self._task_spec, object_configs)
        if resolved is None:
            raise ValueError(f"Failed to match objects for scene={scene_name}: '{task_instruction}'")
        self._resolved: ResolvedTask = resolved

        self._grasp_latched = torch.zeros(self._g, dtype=torch.bool, device=self._device)
        self._pickup_latched = torch.zeros(self._g, dtype=torch.bool, device=self._device)

        self._initial_pos = self._snapshot_positions()
        self._init_region_satisfied = self._compute_init_region_satisfied()
        self._grasp_cache: dict = {}

    @staticmethod
    def _infer_device(object_entities) -> torch.device:
        for entity, _ in object_entities:
            pos = entity.get_pos()
            if isinstance(pos, torch.Tensor):
                return pos.device
        return torch.device("cpu")

    def _snapshot_positions(self) -> torch.Tensor:
        return torch.stack(
            [_to_kxn(e.get_pos(), self._n_envs, self._device)[self._idx] for e, _ in self._entities], dim=0
        )

    def _snapshot_aabb(self) -> Tuple[torch.Tensor, torch.Tensor]:
        mins, maxs = [], []
        for e, _ in self._entities:
            lo, hi = _split_aabb(e.get_AABB())
            mins.append(_to_kxn(lo, self._n_envs, self._device)[self._idx])
            maxs.append(_to_kxn(hi, self._n_envs, self._device)[self._idx])
        return torch.stack(mins, dim=0), torch.stack(maxs, dim=0)

    @property
    def resolved(self) -> ResolvedTask:
        """The task with its object references resolved to entity indices."""
        return self._resolved

    def privileged_goal(self, world_to_base: np.ndarray, satisfied: Sequence[bool]) -> Dict[str, Any]:
        """This task as one privileged goal: what to move, where to, and whether it is done yet.

        ``predicate`` names the motion, so a pickup or a reach ends before the place phases, and
        ``point`` carries the destination for the task types scored against a PLACE rather than
        another object. Together those cover every scene task type except PUSH, which is scored on
        displacement with no goal pose to aim at.
        """
        resolved = self._resolved
        task_type = resolved.task_spec.task_type
        # The destination is named even when a point overrides where the object goes: several goals
        # sharing one destination are laid out on a grid rather than piled, and dropping the index
        # would silently opt a goal out of that.
        goal = {
            "source": resolved.target_index,
            "destination": resolved.destination_index,
            "relation": resolved.task_spec.spatial_relation or "on",
            "satisfied": list(satisfied),
            "predicate": _SCRIPTED_PREDICATE.get(task_type, "pick_and_place"),
        }
        if task_type in (TaskType.PICK_AND_PLACE, TaskType.STACK):
            return goal
        point = self._placement_point_world()
        if point is not None:
            goal["point"] = to_base(world_to_base, point)
        return goal

    def _placement_point_world(self) -> Optional[np.ndarray]:
        """Where the target object must END UP, ``[g, 3]`` in world, for a place-scored task.

        Branch order mirrors ``check_all``'s, so the policy drives to the point the score tests --
        a scene carrying an explicit region target uses it even when the instruction reads as a
        relative placement, which is the precedence ``check_all`` applies.
        """
        resolved = self._resolved
        task_type = resolved.task_spec.task_type
        positions = self._snapshot_positions()
        # Placements rest on the table, so the height is where this object already sits.
        rest_z = self._initial_pos[resolved.target_index][:, 2:3]

        if task_type == TaskType.BETWEEN and resolved.between_anchor_index is not None:
            midpoint = 0.5 * (
                positions[resolved.destination_index][:, :2] + positions[resolved.between_anchor_index][:, :2]
            )
            return self._to_world_points(midpoint, rest_z)
        if self._scene_name in REGION_TARGETS_XY_R:
            px, py, _ = REGION_TARGETS_XY_R[self._scene_name]
            xy = torch.tensor([px, py], dtype=rest_z.dtype, device=self._device).expand(self._g, 2)
            return self._to_world_points(xy, rest_z)

        # A region task names a patch of table rather than a direction from an object; is_in_table_region
        # sizes those patches from the table's own half-diagonal, so the aim point is derived the same
        # way instead of from the object clearance a relative placement needs.
        region = _TABLE_REGION_SIGNS.get(resolved.task_spec.table_region or "")
        direction = resolved.task_spec.spatial_relation
        signs = region if region is not None else _RELATIVE_DIRECTION_SIGNS.get(direction or "")
        if signs is None:
            return None
        if region is not None:
            min_x, max_x, min_y, max_y = FIXED_TABLE_BOUNDS_M
            table_radius = float(np.linalg.norm(np.array([max_x - min_x, max_y - min_y])) / 2.0)
            # 2x the predicate's own threshold, so the aim sits inside the region rather than on its edge.
            reach = 2.0 * (table_radius / 2.0) * 0.3
        else:
            # Aim past the scoring margin by the target's own half width, so the object clears the
            # reference instead of landing on the boundary the predicate tests.
            _, half = self._snapshot_aabb()
            reach = self._thresholds.relative_margin_m + float(
                (half[resolved.target_index] - positions[resolved.target_index])[:, :2].abs().max()
            )
        right = torch.as_tensor(
            self._image_right_xy if self._image_right_xy is not None else (1.0, 0.0),
            dtype=rest_z.dtype,
            device=self._device,
        )
        down = torch.as_tensor(
            self._image_down_xy if self._image_down_xy is not None else (0.0, -1.0),
            dtype=rest_z.dtype,
            device=self._device,
        )
        offset = reach * (signs[0] * right + signs[1] * down)
        if region is None and resolved.destination_index is not None:
            anchor = positions[resolved.destination_index][:, :2]
        else:
            min_x, max_x, min_y, max_y = FIXED_TABLE_BOUNDS_M
            anchor = torch.tensor(
                [(min_x + max_x) / 2.0, (min_y + max_y) / 2.0], dtype=rest_z.dtype, device=self._device
            ).expand(self._g, 2)
        return self._to_world_points(anchor + offset, rest_z)

    def _to_world_points(self, xy: torch.Tensor, z: torch.Tensor) -> np.ndarray:
        return torch.cat([xy, z], dim=1).detach().cpu().numpy()

    def privileged_state(
        self, world_to_base: np.ndarray, achieved: List[List[bool]], held_indices=None
    ) -> List[Dict[str, Any]]:
        """Ground-truth payload for the scripted policy: this task's single goal, base frame."""
        goals = [self.privileged_goal(world_to_base, [row[0] for row in achieved])]
        return privileged_payloads(self, world_to_base, goals, len(achieved), held_indices)

    def privileged_snapshot(self, held_indices=None):
        """Every scene object's name, centre, AABB and grasp bit, as numpy ``[n_objects, K, ...]``.

        The scripted policy needs the whole scene, not just this checker's target, so it can be
        grounded onto any object; the per-goal conditions are assembled by the caller.
        """
        self._grasp_cache = {}
        centers = self._snapshot_positions()
        mins, maxs = self._snapshot_aabb()
        grasped = torch.stack([self._grasped(i) for i in range(len(self._entities))], dim=0)
        # A welded object is held even though contact has stopped firing; see welded_object_indices.
        if held_indices is not None:
            for env_i, obj_i in enumerate(np.asarray(held_indices).reshape(-1)):
                if 0 <= int(obj_i) < grasped.shape[0]:
                    grasped[int(obj_i), env_i] = True
        names = [cfg.name for _, cfg in self._entities]
        to_numpy = lambda t: t.detach().cpu().numpy()  # noqa: E731
        return names, to_numpy(centers), to_numpy(mins), to_numpy(maxs), to_numpy(grasped)

    def privileged_long_axes(self) -> np.ndarray:
        """World yaw of every object's longest horizontal extent, ``[n_objects, K]``.

        The principal axis of the collision vertices' XY spread, so it holds for an object at any yaw.
        """
        yaws = []
        for entity, _ in self._entities:
            try:
                verts = getattr(entity, "underlying_entity", entity).get_verts()
            except Exception:  # noqa: BLE001 -- an entity without collision vertices publishes no yaw
                yaws.append(torch.full((len(self._idx),), float("nan"), device=self._device))
                continue
            xy = verts.reshape(self._n_envs, -1, 3)[self._idx][..., :2]
            xy = xy - xy.mean(dim=1, keepdim=True)
            cxx, cyy, cxy = (
                (xy[..., 0] ** 2).mean(1),
                (xy[..., 1] ** 2).mean(1),
                (xy[..., 0] * xy[..., 1]).mean(1),
            )
            yaws.append((0.5 * torch.atan2(2.0 * cxy, cxx - cyy)).to(self._device))
        return torch.stack(yaws, dim=0).detach().cpu().numpy()

    def _grasped(self, obj_idx: int) -> torch.Tensor:
        if self._robot_entity is None or self._left_finger_idx is None or self._right_finger_idx is None:
            return torch.zeros(self._g, dtype=torch.bool, device=self._device)
        if obj_idx in self._grasp_cache:
            return self._grasp_cache[obj_idx]
        from ladder.components.inference.robots.robot_arena.grasping import is_grasping_two_finger

        entity = self._entities[obj_idx][0]
        raw = getattr(entity, "underlying_entity", entity)
        per_env = is_grasping_two_finger(
            robot_entity=self._robot_entity,
            obj_entity=raw,
            left_finger_idx=self._left_finger_idx,
            right_finger_idx=self._right_finger_idx,
            n_envs=self._n_envs,
        )
        out = per_env.to(self._device)[self._idx]
        self._grasp_cache[obj_idx] = out
        return out

    def _compute_init_region_satisfied(self) -> torch.Tensor:
        out = torch.zeros(self._g, dtype=torch.bool, device=self._device)
        tt = self._resolved.task_spec.task_type
        if tt not in (TaskType.MOVE_TO_REGION, TaskType.PUSH, TaskType.MOVE_RELATIVE):
            return out
        thr0 = dataclasses.replace(self._thresholds, displacement_min_m=0.0)
        rt = self._resolved.task_spec
        tgt = self._initial_pos[self._resolved.target_index]
        init = self._initial_pos[self._resolved.target_index]
        if tt == TaskType.MOVE_RELATIVE:
            if rt.spatial_relation and self._resolved.destination_index is not None:
                ref = self._initial_pos[self._resolved.destination_index]
                out = is_relative_to(
                    tgt, ref, rt.spatial_relation, init, thr0, self._image_right_xy, self._image_down_xy
                )
            elif rt.spatial_relation and rt.table_region:
                out = is_in_table_region(
                    tgt,
                    rt.table_region,
                    FIXED_TABLE_BOUNDS_M,
                    init,
                    thr0,
                    self._image_right_xy,
                    self._image_down_xy,
                )
        elif rt.table_region:
            out = is_in_table_region(
                tgt,
                rt.table_region,
                FIXED_TABLE_BOUNDS_M,
                init,
                thr0,
                self._image_right_xy,
                self._image_down_xy,
            )
        return out

    def _region_target_satisfied(self, tpos: torch.Tensor, init: torch.Tensor) -> torch.Tensor:
        px, py, r = REGION_TARGETS_XY_R[self._scene_name]
        dist = torch.sqrt((tpos[:, 0] - px) ** 2 + (tpos[:, 1] - py) ** 2)
        disp = torch.linalg.vector_norm(tpos[:, :2] - init[:, :2], dim=1)
        return (dist < r) & (disp >= self._thresholds.displacement_min_m)

    def check_all(self) -> List[CheckResult]:  # noqa: C901, PLR0912, PLR0915
        """Evaluate success for every scored env; returns one CheckResult per env."""
        self._grasp_cache = {}
        obj_pos = self._snapshot_positions()
        obj_min, obj_max = self._snapshot_aabb()

        resolved = self._resolved
        tt = resolved.task_spec.task_type
        thr = self._thresholds
        ti = resolved.target_index
        tpos, tmin, tmax = obj_pos[ti], obj_min[ti], obj_max[ti]
        init = self._initial_pos[ti]
        success = torch.zeros(self._g, dtype=torch.bool, device=self._device)

        if tt == TaskType.PICK_AND_PLACE:
            if resolved.destination_index is not None:
                di = resolved.destination_index
                relation = resolved.task_spec.spatial_relation or "on"
                if relation == "in":
                    success = is_inside(tpos, obj_min[di], obj_max[di], thr)
                else:
                    success = is_on_top_of(tpos, tmin, tmax, obj_min[di], obj_max[di], thr)

        elif tt == TaskType.STACK:
            if resolved.destination_index is not None:
                di = resolved.destination_index
                success = is_on_top_of(tpos, tmin, tmax, obj_min[di], obj_max[di], thr)

        elif tt == TaskType.PICKUP:
            self._pickup_latched = self._pickup_latched | (tpos[:, 2] > (init[:, 2] + thr.pickup_lift_m))
            success = self._pickup_latched

        elif tt == TaskType.GRASP:
            self._grasp_latched = self._grasp_latched | self._grasped(ti)
            success = self._grasp_latched

        elif tt == TaskType.MOVE_RELATIVE:
            direction = resolved.task_spec.spatial_relation
            if self._scene_name in REGION_TARGETS_XY_R:
                success = self._region_target_satisfied(tpos, init)
            elif direction and resolved.destination_index is not None:
                ref = obj_pos[resolved.destination_index]
                success = is_relative_to(
                    tpos, ref, direction, init, thr, self._image_right_xy, self._image_down_xy
                )
            elif direction and resolved.task_spec.table_region:
                success = is_in_table_region(
                    tpos,
                    resolved.task_spec.table_region,
                    FIXED_TABLE_BOUNDS_M,
                    init,
                    thr,
                    self._image_right_xy,
                    self._image_down_xy,
                )

        elif tt in (TaskType.MOVE_TO_REGION, TaskType.PUSH):
            region = resolved.task_spec.table_region
            if self._scene_name in REGION_TARGETS_XY_R:
                success = self._region_target_satisfied(tpos, init)
            elif region:
                success = is_in_table_region(
                    tpos,
                    region,
                    FIXED_TABLE_BOUNDS_M,
                    init,
                    thr,
                    self._image_right_xy,
                    self._image_down_xy,
                )

        elif tt == TaskType.BETWEEN:
            if resolved.destination_index is not None and resolved.between_anchor_index is not None:
                success = is_between(
                    tpos, obj_pos[resolved.destination_index], obj_pos[resolved.between_anchor_index], thr
                )

        elif tt == TaskType.REACH:
            if resolved.destination_index is not None:
                success = is_near(tpos, obj_pos[resolved.destination_index], thr)
            elif self._robot_ee_link is not None:
                ee = _to_kxn(self._robot_ee_link.get_pos(), self._n_envs, self._device)[self._idx]
                dist = torch.linalg.vector_norm(ee[:, :2] - tpos[:, :2], dim=1)
                success = dist < thr.reach_distance_m
            else:
                success = is_picked_up(tpos, init[:, 2], thr)

        # Recalibration gates (mirror the per-env SuccessChecker.check()).
        if (
            tt in (TaskType.MOVE_TO_REGION, TaskType.PUSH, TaskType.MOVE_RELATIVE)
            and self._scene_name not in REGION_TARGETS_XY_R
        ):
            success = success & ~self._init_region_satisfied
        if tt in (TaskType.PICK_AND_PLACE, TaskType.STACK):
            success = success & ~self._grasped(ti)
        if tt in (TaskType.MOVE_TO_REGION, TaskType.PUSH, TaskType.MOVE_RELATIVE):
            airborne = tpos[:, 2] > (init[:, 2] + thr.pickup_height_m)
            success = success & ~(airborne | self._grasped(ti))

        done_cpu = success.cpu().tolist()
        tt_val = tt.value
        return [
            {"done": bool(done_cpu[i]), "success_status": self._status, "task_type": tt_val}
            for i in range(self._g)
        ]
