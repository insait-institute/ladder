import functools
import math
import os
import pathlib
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional

import isaaclab.envs.mdp as mdp
import robolab.constants
import torch
from isaaclab.managers import EventTermCfg as EventTerm, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from robolab.core.events.reset_pose import reset_pose_uniform
from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import (
    object_in_container,
    object_on_top,
    objects_placed_in_container_in_order,
    pick_and_place,
    pick_and_place_on_surface,
)
from robolab.core.task.task import Task
from robolab.core.world.world_state import get_world

from ladder.components.inference.robots.robolab.composition.catalog import CATALOG, ON, RECEPTACLE_TYPES
from ladder.components.inference.robots.robolab.composition.specs import (
    SPECS,
    CompositionSpec,
    advance_legs,
    chain_complete,
    task_class_name,
)

# Authored scene USDs go to a fixed scratch dir, off the read-only bazel-runfiles tree.
_SCENE_DIR = pathlib.Path(tempfile.gettempdir()) / "robolab_composition_scenes"

# A bare tabletop rig (base_empty.usda: table + franka mount + ground + physics) with each graspable
# and receptacle added as a direct child of `world` via a USD payload. import_scene scrapes exactly
# those direct children as single rigid bodies, so contact sensors resolve one body per name --
# unlike a runtime spawn of the raw catalog USD, whose nested bodies break the per-object filters.
_RIG_TEMPLATE = """#usda 1.0
(
    defaultPrim = "world"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "world"
{{
    def Material "PhysicsMaterial" (
        prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]
    )
    {{
        float physics:dynamicFriction = 2
        float physics:staticFriction = 2
        uniform token physxMaterial:frictionCombineMode = "max"
    }}

    def "table" (
        prepend payload = @{fixtures}/table_oak.usd@
    )
    {{
        double3 xformOp:translate = (0.5471367990922231, 0, -0.35)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}

    def "franka_table" (
        prepend payload = @{fixtures}/franka_table.usd@
    )
    {{
        quatd xformOp:orient = (6.123233995736766e-17, 0, 0, 1)
        double3 xformOp:translate = (-0.087, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    }}

    def Xform "GroundPlane"
    {{
        token visibility = "invisible"
        double3 xformOp:translate = (0, 0, -0.697)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Plane "CollisionPlane" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            uniform token axis = "Z"
            uniform token purpose = "guide"
        }}
    }}
{objects}}}
"""

_OBJECT_TEMPLATE = """    def "{name}" (
        prepend payload = @{usd}@
    )
    {{
        quatf xformOp:orient = (1, 0, 0, 0)
        double3 xformOp:translate = ({x}, {y}, {z})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    }}
"""


def placed(env, obj, receptacle, kind, env_id):
    """Per-object placement predicate: on-top for ON receptacles, containment otherwise."""
    if kind == ON:
        return object_on_top(
            env,
            object=[obj],
            reference_object=receptacle,
            logical="all",
            require_gripper_detached=True,
            env_id=env_id,
        )
    return object_in_container(
        env,
        object=[obj],
        container=receptacle,
        logical="all",
        require_gripper_detached=True,
        env_id=env_id,
    )


def legs_success(env, objects, legs, dest, dest_kind, env_id=None):
    """Every object must have completed ``legs`` IN ORDER (latched, cleared on reset) and currently
    rest in ``dest``. Ordering is what makes the intermediate hops real: without it an object put
    straight into the last receptacle, or one that visits the stops backwards, would pass. A revisit
    names its final receptacle among the legs too, so only the order tells the two visits apart."""
    world = get_world(env)
    objects, legs = list(objects), list(legs)
    state_key = f"legs::{'->'.join(name for name, _ in legs)}->{dest}::{','.join(objects)}"

    def _make_state():
        latched = {
            o: [torch.zeros(env.num_envs, dtype=torch.bool, device=env.device) for _ in legs] for o in objects
        }

        def reset(env_ids):
            for flags in latched.values():
                for flag in flags:
                    flag[env_ids] = False

        return {"latched": latched, "__reset__": reset}

    latched = world.get_or_init_predicate_state(state_key, _make_state)["latched"]
    result = None
    for obj in objects:
        holding = [placed(env, obj, name, kind, None) for name, kind in legs]
        latched[obj] = advance_legs(latched[obj], holding)
        ok = chain_complete(latched[obj], placed(env, obj, dest, dest_kind, None))
        result = ok if result is None else result & ok
    return result if env_id is None else result[env_id]


def _latched(predicate, key: str):
    """Make a placement predicate stay satisfied once achieved, on episode-scoped predicate state.

    A relay leg is a milestone, not an end state: lifting the object to start leg 2 makes the leg-1
    check false again, so an advancer reading the raw predicate walks back onto a leg it finished and
    the task becomes unwinnable. Terminal goals are deliberately NOT wrapped -- an object knocked out
    of its destination must un-achieve so the advancer can repair it. The wrapper carries the inner
    predicate's ``__name__`` and bound nouns, which the oracle renders each subtask's command from."""
    base = getattr(predicate, "func", predicate)

    def latched(env, env_id=None, **_bound_nouns_for_rendering):
        world = get_world(env)

        def _make_state():
            flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

            def reset(env_ids):
                flag[env_ids] = False

            return {"flag": flag, "__reset__": reset}

        flag = world.get_or_init_predicate_state(key, _make_state)["flag"]
        flag |= predicate(env, env_id=None)
        return flag if env_id is None else flag[env_id]

    latched.__name__ = getattr(base, "__name__", "latched")
    return functools.partial(latched, **(getattr(predicate, "keywords", None) or {}))


def _latch_final_condition(subtask, key: str):
    """Latch each object's score-carrying condition -- the one the oracle publishes as ``achieved``."""
    for obj, conditions in subtask.conditions.items():
        predicate, score = conditions[-1]
        conditions[-1] = (_latched(predicate, f"{key}::{obj}"), score)
    return subtask


def _author_usda(spec: CompositionSpec) -> str:
    """Write the task's scene USD (base rig + one payload prim per object/receptacle) and return it."""
    asset_dir = robolab.constants.ASSET_DIR
    prims = []
    for obj in spec.objects:
        t = CATALOG[obj.type_id]
        z = t.rest_z
        if obj.start_on:  # spawn on the source receptacle's rim, dropped slightly so it settles
            z = 2 * RECEPTACLE_TYPES[obj.start_on].rest_z + t.rest_z + 0.02
        prims.append(
            _OBJECT_TEMPLATE.format(name=obj.prim, usd=os.path.join(asset_dir, t.usd), x=obj.x, y=obj.y, z=z)
        )
    for recep in spec.receptacles:
        rt = RECEPTACLE_TYPES[recep.type_id]
        prims.append(
            _OBJECT_TEMPLATE.format(
                name=recep.prim, usd=os.path.join(asset_dir, rt.usd), x=recep.x, y=recep.y, z=rt.rest_z
            )
        )
    usda = _RIG_TEMPLATE.format(fixtures=os.path.join(asset_dir, "fixtures"), objects="".join(prims))
    _SCENE_DIR.mkdir(parents=True, exist_ok=True)
    path = _SCENE_DIR / f"{spec.name}.usda"
    # Publish ATOMICALLY: every process rewrites these on import while running eval processes read
    # them at env-creation time. A reader landing inside a truncating write loads an empty stage --
    # no table, no robot, no objects -- which still burns its step budget and scores as a failure.
    # That cost 10% of the recorded episodes on a node where several jobs shared one /tmp.
    tmp = _SCENE_DIR / f".{spec.name}.{os.getpid()}.usda.tmp"
    tmp.write_text(usda)
    os.replace(tmp, path)
    return str(path)


def _events_cls(spec: CompositionSpec):
    """Per-episode reset randomization: every graspable jittered by +/-``spec.jitter`` in x/y with a
    full random yaw, collision-checked. Receptacles are omitted, so ``reset_to_default_otherwise``
    pins them and the clearances the object positions were solved against stay valid."""
    j = spec.jitter
    event = EventTerm(
        func=reset_pose_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-j, j), "y": (-j, j), "z": (0.0, 0.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
            "asset_cfg": spec.free_object_list,
            "reset_to_default_otherwise": True,
            "use_collision_check": True,
        },
    )
    return configclass(type("CompositionEvents", (), {"randomize_init_pose": event}))


def _grouped_goals(spec: CompositionSpec) -> "OrderedDict[str, List[str]]":
    """receptacle_id -> [object_id, ...], in goal order (also the oracle plan order)."""
    groups: "OrderedDict[str, List[str]]" = OrderedDict()
    for goal in spec.goals:
        groups.setdefault(goal.receptacle, []).append(goal.obj_id)
    return groups


def composition_success(
    env,
    groups: Dict[str, List[str]],
    kinds: Dict[str, str],
    forbidden: Optional[Dict[str, List[str]]] = None,
    env_id=None,
):
    """Whole-task success: a conjunction over receptacle groups, each an ``all`` placement check.
    ``forbidden`` (distractor tasks) additionally requires no named decoy ended up in a receptacle."""
    result = None
    for recep_id, objs in groups.items():
        if kinds[recep_id] == ON:
            group_ok = object_on_top(
                env,
                object=objs,
                reference_object=recep_id,
                logical="all",
                require_gripper_detached=True,
                env_id=env_id,
            )
        else:
            group_ok = object_in_container(
                env,
                object=objs,
                container=recep_id,
                logical="all",
                require_gripper_detached=True,
                env_id=env_id,
            )
        result = group_ok if result is None else result & group_ok
    for recep_id, objs in (forbidden or {}).items():
        intruded = object_in_container(env, object=objs, container=recep_id, logical="any", env_id=env_id)
        result = result & (~intruded)
    return result


def _terminations_cls(spec: CompositionSpec):
    groups = _grouped_goals(spec)
    if spec.via_legs:
        rm = spec.receptacle_map
        dest = spec.receptacles_used[0]  # single by construction
        success = DoneTerm(
            func=legs_success,
            params={
                "objects": [g.obj_id for g in spec.goals],
                "legs": [(name, rm[name].kind) for name in spec.via_legs],
                "dest": dest,
                "dest_kind": rm[dest].kind,
            },
        )
    elif spec.ordered:  # single-receptacle by construction; enforce first-placement order
        success = DoneTerm(
            func=objects_placed_in_container_in_order,
            params={
                "objects": [g.obj_id for g in spec.goals],
                "container": spec.goals[0].receptacle,
                "require_gripper_detached": True,
            },
        )
    else:
        kinds = {recep: spec.receptacle_map[recep].kind for recep in groups}
        params: Dict[str, object] = {"groups": dict(groups), "kinds": kinds}
        distractors = [o.prim for o in spec.distractor_objects]
        if distractors:  # single-receptacle by construction; decoys must stay out of it
            params["forbidden"] = {spec.receptacles_used[0]: distractors}
        success = DoneTerm(func=composition_success, params=params)
    return configclass(
        type(
            "CompositionTerminations",
            (),
            {
                "time_out": DoneTerm(func=mdp.time_out, time_out=True),
                "success": success,
            },
        )
    )


def _place_subtask(objs, recep_id, kind, score):
    if kind == ON:
        return pick_and_place_on_surface(object=objs, surface=recep_id, logical="all", score=score)
    return pick_and_place(object=objs, container=recep_id, logical="all", score=score)


def _subtasks(spec: CompositionSpec):
    """The oracle plan: one placement subtask PER GOAL OBJECT, in plan order. A single ``all`` group
    over every object gives the advancer no per-object achieved flag, and the policy then stalls after
    the first object once N >= 3. Relay emits the two-leg sequence (via, then dest) per object."""
    rm = spec.receptacle_map
    if spec.via_legs:
        score = 1.0 / ((1 + len(spec.via_legs)) * len(spec.goals))
        subtasks = []
        for i, g in enumerate(spec.goals):
            for leg, name in enumerate(spec.via_legs):
                # Every leg but the last is LATCHED: starting the next one necessarily undoes it.
                subtasks.append(
                    _latch_final_condition(
                        _place_subtask([g.obj_id], name, rm[name].kind, score),
                        f"{spec.name}::via{i}_{leg}",
                    )
                )
            subtasks.append(_place_subtask([g.obj_id], g.receptacle, rm[g.receptacle].kind, score))
        return subtasks
    score = 1.0 / len(spec.goals)
    return [_place_subtask([g.obj_id], g.receptacle, rm[g.receptacle].kind, score) for g in spec.goals]


def make_task(spec_name: str):
    """Build the RoboLab ``Task`` subclass for a composition spec. Called at task-file import time,
    inside the env_server sim process, where the RoboLab asset tree is available."""
    spec = SPECS[spec_name]
    name = task_class_name(spec_name)
    return type(
        name,
        (Task,),
        {
            "scene": import_scene(_author_usda(spec), tuple(spec.contact_object_list)),
            "events": _events_cls(spec),
            "instruction": spec.instruction,
            "terminations": _terminations_cls(spec),
            "subtasks": _subtasks(spec),
            "contact_object_list": spec.contact_object_list,
            "episode_length_s": spec.episode_length_s,
            "attributes": ["composition", spec.family, f"n{spec.n}"],
            "task_name": name,
        },
    )
