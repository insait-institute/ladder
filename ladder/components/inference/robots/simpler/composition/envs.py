import collections
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sapien.physx as physx
import torch
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.base_env import (
    BRIDGE_DATASET_ASSET_PATH,
    BaseBridgeEnv,
)
from mani_skill.utils import common
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from ladder.components.inference.api.core.instructions import ON, Placement, subtask
from ladder.components.inference.api.core.privileged_geometry import build_payloads
from ladder.components.inference.robots.simpler.composition.placement import (
    FOOTPRINT_SLACK_M,
    build_layouts,
    strip_instance_id,
)
from ladder.components.inference.robots.simpler.composition.specs import DISPLAY, SPECS, Spec, scene_layout

_DISTRACTOR_MAX_MOVE_M = 0.05  # a distractor displaced more than this counts as a grounding failure
_Z_OFFSET_M = 0.02  # how far above a target's top surface a source may rest and still count as on it
_N_LAYOUTS = 24  # object-position layouts per scene, matching standard SimplerEnv's variation count


class CompositionBridgeEnv(BaseBridgeEnv):
    """Multi-goal bridge task: success is every goal of :attr:`SPEC` met with every distractor left in
    place. Importing this module registers one subclass per :data:`SPECS` entry."""

    scene_setting = "flat_table"
    SPEC: Spec
    # The digital-twin base env allows only "rgb+segmentation" for its greenscreen overlay, which
    # depth does not affect, and "none" renders nothing to overlay. Widened here rather than
    # patched in ManiSkill.
    SUPPORTED_OBS_MODES = ["rgb+segmentation", "rgb+depth+segmentation", "none"]

    def __init__(self, n_layouts: int = _N_LAYOUTS, **kwargs) -> None:
        self.layout = scene_layout(self.SPEC)  # episode_id indexes the n_layouts generated here
        goals = self.layout.goals
        self.objects_excluded_from_greenscreening = list(self.layout.obj_names)
        # Oracle contract: the env publishes the decomposition and the per-goal success keys under
        # metadata["oracle"]; an external controller decides which subtask to command.
        self.oracle_subtasks = [
            subtask(Placement(DISPLAY[strip_instance_id(g.source)], ON, DISPLAY[g.target])) for g in goals
        ]
        # evaluate() below requires the listed order for an ordered spec, so the plan the oracle
        # publishes has to say so too -- otherwise the oracle arm alone schedules the chain freely.
        self.oracle_ordered = self.SPEC.ordered
        self.oracle_revisits = self.SPEC.revisits
        # A revisit spec repeats an (object, destination) pair, so the plain name collides and one
        # leg's result silently overwrites another's; only the repeats are suffixed, so every key
        # that was already unique keeps its name.
        seen: "collections.Counter[str]" = collections.Counter()
        self.oracle_goal_keys = []
        for goal in goals:
            key = f"success_{goal.source}_on_{goal.target}"
            seen[key] += 1
            self.oracle_goal_keys.append(key if seen[key] == 1 else f"{key}_{seen[key]}")
        # Goal destinations and a transfer task's start receptacle: the objects that are placed ONTO
        # rather than picked up. Built kinematic -- see _build_actor_helper.
        self._receptacles = {g.target for g in goals} | (
            {self.SPEC.start_on} if self.SPEC.start_on else set()
        )
        # Which of _on_target's three "is it on there?" tests each target scores under: several goals
        # landing on one target relaxes it the most, a rigid single-goal target gets the strict one.
        self._piled = {t for t, n in collections.Counter(g.target for g in goals).items() if n > 1}
        self._contact = {g.target for g in goals if g.completely_on} - self._piled
        # A (model, target) pair asks for a COUNT of that model on that target -- see evaluate().
        self._demand = collections.Counter((strip_instance_id(g.source), g.target) for g in goals)
        self._instances = {
            model: sorted({g.source for g in goals if strip_instance_id(g.source) == model})
            for model, _ in self._demand
        }
        xyz_configs, quat_configs = build_layouts(
            self.layout.obj_names, self.layout.bbox, goals, n_layouts, on_recep=self.SPEC.start_on
        )
        super().__init__(
            obj_names=list(self.layout.obj_names),
            xyz_configs=xyz_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def _load_scene(self, options: Dict[str, Any]) -> None:
        # "carrot#1" has no model_db entry of its own; alias it so density and bbox lookups resolve.
        for name in self.obj_names:
            self.model_db.setdefault(name, self.model_db[strip_instance_id(name)])
        super()._load_scene(options)
        # BaseBridgeEnv only sizes obj_names[:2], and an enlarged receptacle needs its success footprint
        # scaled with its mesh. Every bridge model lists scales [1.0], so this reproduces what the parent
        # stored for those first two.
        for name in self.obj_names:
            bbox = self.model_db[name]["bbox"]
            size = np.array(bbox["max"]) - np.array(bbox["min"])
            self.episode_model_bbox_sizes[name] = common.to_tensor(
                size * np.array(self._axis_scale(name, 1.0)), device=self.device
            )
        # On a piled target the source may rest on other sources rather than on the receptacle, so the
        # height it is allowed is the height of the objects that can be underneath it. Sized at the whole
        # pile, it rejects only a height no stack could reach -- an object still held. Anything tighter
        # would reject a real pile, which is the one thing a success test must not do.
        self._pile_z = {
            target: _Z_OFFSET_M
            + sum(
                float(self.episode_model_bbox_sizes[goal.source][2])
                for goal in self.layout.goals
                if goal.target == target
            )
            for target in self._piled
        }
        # Success state, cleared per env in _initialize_episode.
        self._first_done = torch.full(
            (self.num_envs, len(self.layout.goals)), -1, dtype=torch.int64, device=self.device
        )
        self._milestone = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        # Published for the privileged consumers; evaluate() replaces it every step.
        self.oracle_achieved = torch.zeros(
            (self.num_envs, len(self.layout.goals)), dtype=torch.bool, device=self.device
        )

    def _axis_scale(self, model_id: str, scale: float) -> List[float]:
        """Per-axis mesh scale. A receptacle grows in the PLANE only.

        Scaling the third axis too raises the rim and lowers the underside, while the spawn height is
        a per-model constant tuned for the native size. A 2.4x plate would then start with its base
        under the table, and the solver would eject it into the movers around it, so some episodes
        would begin partly solved. Growing x and y alone leaves
        thickness, spawn height and table clearance exactly as the native case that was tuned.
        """
        base = strip_instance_id(model_id)
        plane = scale * self.SPEC.target_scales.get(base, 1.0)
        return [plane, plane, scale]

    def _build_actor_helper(
        self,
        model_id: str,
        scale: float = 1,
        kinematic: bool = False,
        initial_pose: Optional[Pose] = None,
    ) -> Actor:
        # Enlarge a shared receptacle to its load so objects rest on it flat instead of being pushed off
        # its edge. Visual and collision mesh both scale; _load_scene matches the success footprint.
        # Movers always keep their native scale.
        base = strip_instance_id(model_id)
        axis = self._axis_scale(model_id, scale)
        # A receptacle is scenery: nothing in these tasks asks for it to be moved, and leaving it
        # dynamic costs twice over. `BaseBridgeEnv._initialize_episode` grants the scene seconds of
        # free simulation to settle, enough for an enlarged plate to slide onto nearby movers. It also
        # stops the policy shunting its own destination around mid-episode. The parent builds every
        # object dynamic and pins only the sink.
        if base in self._receptacles:
            kinematic = True
        # The parent takes one scalar, so anything scaled unevenly is built here.
        if base == model_id and axis[0] == axis[2]:
            return super()._build_actor_helper(
                model_id, scale=scale, kinematic=kinematic, initial_pose=initial_pose
            )
        # A duplicate instance has no asset directory, so build the BASE mesh under the instance name.
        model_dir = BRIDGE_DATASET_ASSET_PATH / "custom" / "models" / base
        builder = self.scene.create_actor_builder()
        builder.add_multiple_convex_collisions_from_file(
            filename=str(model_dir / "collision.obj"),
            scale=axis,
            material=physx.PhysxMaterial(
                static_friction=self.obj_static_friction,
                dynamic_friction=self.obj_dynamic_friction,
                restitution=0.0,
            ),
            density=self.model_db[base].get("density", 1000),
        )
        builder.add_visual_from_file(filename=str(model_dir / "textured.dae"), scale=axis)
        if initial_pose is not None:
            builder.initial_pose = initial_pose
        return builder.build_kinematic(name=model_id) if kinematic else builder.build(name=model_id)

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict[str, Any]) -> None:
        super()._initialize_episode(env_idx, options)
        # World-frame bbox at settle time, as BaseBridgeEnv derives it for its own source/target pair:
        # env 0's orientation, which every env shares here.
        self._bbox_world = {
            name: rotation_conversions.quaternion_to_matrix(self.objs[name].pose.q)[0]
            @ self.episode_model_bbox_sizes[name].float()
            for name in self.obj_names
        }
        self._first_done[env_idx] = -1
        self._milestone[env_idx] = 0
        self.oracle_achieved[env_idx] = False

    def _on_target(self, source: str, target: str) -> torch.Tensor:
        """Whether ``source`` is resting on ``target``, reproducing ``BaseBridgeEnv._evaluate``'s geometry
        under one of three tests. A rigid target with a single goal (``self._contact``) gets the strict
        upstream one: footprint overlap, source resting within :data:`_Z_OFFSET_M` of the target's top, and
        a live contact force between the two. The towel drops that force term, being soft -- upstream's own
        cloth task does the same. Neither survives a target several goals land on (``self._piled``), where
        object two rests ON object one: it clears the single-object height and touches the pile rather than
        the receptacle. That case allows the height of the whole pile (``self._pile_z``) and asks for the
        object to have settled -- and, having no contact force to confirm it, measures the footprint as the
        rectangle it is. Upstream's half-diagonal reaches 14.8 cm on a 1.9x plate, well past the plate
        itself -- which the strict tests hide but a piled target would score."""
        src, tgt = self.objs[source], self.objs[target]
        src_half, tgt_half = self._bbox_world[source] / 2, self._bbox_world[target] / 2
        offset = src.pose.p - tgt.pose.p
        above = offset[:, 2] > 0
        clearance = offset[:, 2] - tgt_half[2] - src_half[2]
        if target in self._piled:
            inside = (offset[:, :2].abs() <= tgt_half[:2].abs() + FOOTPRINT_SLACK_M).all(dim=1)
            settled = torch.linalg.norm(src.linear_velocity, dim=1) < 0.03
            return inside & above & (clearance <= self._pile_z[target]) & settled
        over = torch.linalg.norm(offset[:, :2], dim=1) <= torch.linalg.norm(tgt_half[:2]) + FOOTPRINT_SLACK_M
        on = over & above & (clearance <= _Z_OFFSET_M)
        if target in self._contact:
            forces = self.scene.get_pairwise_contact_forces(src, tgt)
            on = on & (torch.linalg.norm(forces, dim=1) > 0.05)
        return on

    def _resting(self) -> Dict[Tuple[str, str], torch.Tensor]:
        return {
            (source, target): self._on_target(source, target)
            for model, target in self._demand
            for source in self._instances[model]
        }

    def _credit(self, resting: Dict[Tuple[str, str], torch.Tensor]) -> torch.Tensor:
        """Per-goal credit as the score counts it, ``[num_envs, n_goals]``.

        A revisit spec latches milestones: leg k+1 moves the same object off leg k's destination, so a
        passed leg has to stay credited or the advancer walks back onto it. Every other spec scores a
        (model, target) pair by COUNT -- two carrots are physically identical, so the instruction
        cannot bind one to one goal -- and credit goes to the first ``count`` goals of each group.
        """
        goals = self.layout.goals
        if self.SPEC.revisits:
            return torch.stack([self._milestone > i for i in range(len(goals))], dim=1)
        credit = torch.zeros(self.num_envs, len(goals), dtype=torch.bool, device=self.device)
        rank: Dict[Tuple[str, str], int] = collections.Counter()
        for i, goal in enumerate(goals):
            model = strip_instance_id(goal.source)
            landed = sum(resting[(source, goal.target)] for source in self._instances[model])
            credit[:, i] = landed > rank[(model, goal.target)]
            rank[(model, goal.target)] += 1
        return credit

    def evaluate(self) -> Dict[str, torch.Tensor]:
        goals = self.layout.goals
        resting = self._resting()
        on = torch.stack([resting[(g.source, g.target)] for g in goals], dim=1)  # [num_envs, n_goals]
        stats: Dict[str, torch.Tensor] = {key: on[:, i] for i, key in enumerate(self.oracle_goal_keys)}
        step = self.elapsed_steps.to(torch.int64)[:, None]
        self._first_done = torch.where(on & (self._first_done < 0), step, self._first_done)

        if self.SPEC.revisits:
            # Moving the same object twice has no end-state conjunction -- a later leg undoes an earlier
            # goal pose -- so it scores as latching milestones: leg k counts once legs 0..k-1 are done
            # and leg k is CURRENTLY satisfied.
            leg = self._milestone.clamp(max=len(goals) - 1)
            landed = on.gather(1, leg[:, None])[:, 0] & (self._milestone < len(goals))
            self._milestone = self._milestone + landed
            stats["milestone"] = self._milestone.to(torch.int32)

        # The score and the oracle read ONE credit vector, so they cannot form separate opinions
        # about what is done.
        self.oracle_achieved = credit = self._credit(resting)
        matched, progress = credit.all(dim=1), credit.sum(dim=1)
        if self.SPEC.ordered and not self.SPEC.revisits:
            # Distinct movers, so also require first completion in the listed order. A revisit spec
            # gets its order from the milestone latch instead.
            for i in range(len(goals) - 1):
                matched = matched & (self._first_done[:, i] <= self._first_done[:, i + 1])

        for name in self.SPEC.distractors:  # unscored, but has to be left where it settled
            drift = self.objs[name].pose.p[:, :2] - self.episode_obj_xyzs_after_settle[name][:, :2]
            still = torch.linalg.norm(drift, dim=1) <= _DISTRACTOR_MAX_MOVE_M
            stats[f"distractor_{name}_stationary"] = still
            matched = matched & still
        stats["num_subgoals_complete"] = progress.to(torch.int32)
        stats["success"] = matched
        return stats

    def privileged_state(self) -> List[Dict[str, Any]]:
        """Ground-truth geometry for the scripted policy, one condition per goal, per env.

        Poses are in the robot BASE frame, which for translations is the canonical frame the policy
        commands in (``proprio.widowx_eef_pose_to_canonical`` rotates orientation only). The bboxes
        are the settle-time world extents ``_on_target`` scores with, so the goal points the shared
        builder derives and the success test agree by construction.
        """
        robot_inv = self.agent.robot.pose.inv()
        names = list(self.obj_names)
        centers = np.stack([common.to_numpy((robot_inv * self.objs[n].pose).p) for n in names])
        half = np.stack([np.abs(common.to_numpy(self._bbox_world[n])) / 2.0 for n in names])
        half = np.broadcast_to(half[:, None, :], centers.shape)
        grasped = np.stack([common.to_numpy(self.agent.is_grasping(self.objs[n])).reshape(-1) for n in names])
        index = {name: i for i, name in enumerate(names)}
        credit = common.to_numpy(self.oracle_achieved).T
        return build_payloads(
            names=names,
            centers=centers,
            mins=centers - half,
            maxs=centers + half,
            grasped=grasped,
            goals=[
                {
                    "source": index[goal.source],
                    "destination": index[goal.target],
                    "relation": "on",
                    "satisfied": credit[i],
                }
                for i, goal in enumerate(self.layout.goals)
            ],
            n_envs=self.num_envs,
        )

    def get_language_instruction(self, **kwargs) -> List[str]:
        del kwargs
        return [self.SPEC.instruction] * self.num_envs


for _spec in SPECS:
    register_env(
        f"{_spec.name}-v1", max_episode_steps=_spec.max_steps, asset_download_ids=["bridge_v2_real2sim"]
    )(type(_spec.name, (CompositionBridgeEnv,), {"SPEC": _spec}))


if __name__ == "__main__":  # the task list: "<env-v0>|<max_episode_steps>" per registered task
    for _spec in SPECS:
        print(f"{_spec.name}-v0|{_spec.max_steps}")
