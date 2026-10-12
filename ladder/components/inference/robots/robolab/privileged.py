from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from robolab.core.world.world_state import get_world

from ladder.components.inference.api.core.privileged_geometry import build_payloads, to_base
from ladder.components.inference.robots.robolab.oracle import OracleGoal

_ROBOT = "robot"
_BASE_LINK = "panda_link0"
_GRIPPER = "gripper"
_IGNORED = {"table"}  # scenery registered as a rigid object, as upstream's EventTracker skips it

Leg = Tuple[str, str, str]


_LONG_ASPECT = 1.2  # a footprint this much longer than wide has a long axis to grip across


def _numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Rotation from an (x, y, z, w) quaternion -- the order this articulation state reports.

    Verified against the sim: the arm's base sits square on the env origin, reporting [0, 0, 0, 1],
    and the end effector resolves to the same x and y through forward kinematics as it does in
    env-local coordinates. Reading that quaternion as (w, x, y, z) instead invents a 180 degree yaw,
    which puts the whole scene behind the robot and sends the policy driving out of the workspace.
    """
    x, y, z, w = (float(v) for v in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _base_transforms(world, n_envs: int) -> np.ndarray:
    """Env-local -> robot base, one 4x4 per env.

    Object geometry comes out of WorldState relative to each env's scene origin, while the policy
    reads an eef pose forward kinematics produced in the arm's own base frame. Publishing the two in
    different frames fails silently: every number looks plausible and every reach misses by the mount
    offset.
    """
    poses = _numpy(world.get_articulation_link_pose(_ROBOT, _BASE_LINK))
    origins = _numpy(world.env.scene.env_origins)

    transforms = np.zeros((n_envs, 4, 4))
    for env_id in range(n_envs):
        base_to_env = np.eye(4)
        base_to_env[:3, :3] = _quat_to_matrix(poses[env_id, 3:7])
        base_to_env[:3, 3] = poses[env_id, :3] - origins[env_id]
        transforms[env_id] = np.linalg.inv(base_to_env)
    return transforms


def _corners(world, name: str) -> np.ndarray:
    """Per-env corners of one body's oriented bounding box, ``[n_envs, 8, 3]`` in env-local coordinates."""
    corners, _ = world.get_bbox(name)
    return _numpy(corners)


def _long_axis_yaw(corners: np.ndarray) -> float:
    """Yaw of an oriented box's longer horizontal side, or NaN when its footprint is round.

    The axis-aligned extents cannot tell which way a lemon at 45 degrees lies, and a grip closed on
    its tapered ends squeezes it out of the jaws.
    """
    xy = corners[:, :2] - corners[:, :2].mean(axis=0)
    _, spread, axes = np.linalg.svd(xy, full_matrices=False)
    if spread[0] < _LONG_ASPECT * spread[1]:
        return float("nan")
    return float(np.arctan2(axes[0][1], axes[0][0]))


def privileged_state(
    env, goals: Sequence[OracleGoal], achieved: Sequence[Sequence[bool]]
) -> List[Dict[str, Any]]:
    """Ground-truth geometry for the scripted completability policy, one payload per env.

    Driven by the goals the task is SCORED on, so it covers every RoboLab task rather than the
    composition suite alone, and emitted in the schema RobotArena and SimplerEnv emit, so one
    ScriptedPolicy drives all three. A goal whose predicate names no placement is skipped: there is
    no destination to drive to.
    """
    world = get_world(env)
    n_envs = int(env.num_envs)

    # A subtask's condition chain is the STAGES of one placement (the upstream builder emits
    # object_grabbed before the placement predicate), not a sequence of destinations. A relay's legs
    # arrive as separate goals already, each latched by the factory, so the geometry to drive to is
    # the last condition that actually names a placement.
    active = [
        next((geom for _, geom in reversed(list(goal.legs)) if geom is not None), None) for goal in goals
    ]

    # Every movable body, so a grasped or knocked distractor shows in the trace.
    named = {name for goal in goals for _, leg in goal.legs if leg for name in leg[:2]}
    names = sorted(named | {name for name in world.objects if name not in _IGNORED})
    if not names:
        return [{"conditions": [], "objects": []} for _ in range(n_envs)]
    index_of = {name: i for i, name in enumerate(names)}

    transforms = _base_transforms(world, n_envs)
    mins, maxs = (np.zeros((len(names), n_envs, 3)) for _ in range(2))
    yaws = np.full((len(names), n_envs), np.nan)
    for i, name in enumerate(names):
        corners = _corners(world, name)
        for env_id in range(n_envs):
            # All eight corners go through the change of frame, so the extremes are those of the
            # base-frame box rather than of two env-local corners that a rotation can swap.
            base = to_base(transforms[env_id], corners[env_id])
            mins[i, env_id], maxs[i, env_id] = base.min(axis=0), base.max(axis=0)
            yaws[i, env_id] = _long_axis_yaw(base)
    centers = 0.5 * (mins + maxs)

    grasped = np.zeros((len(names), n_envs), dtype=bool)
    for i, name in enumerate(names):
        grasped[i] = _numpy(world.in_contact(name, _GRIPPER))

    payload_goals: List[Dict[str, Any]] = []
    for goal_index, leg in enumerate(active):
        if leg is None:
            continue
        source, destination, relation = leg
        payload_goals.append(
            {
                "source": index_of[source],
                "destination": index_of[destination],
                "relation": relation,
                "satisfied": [bool(achieved[env_id][goal_index]) for env_id in range(n_envs)],
            }
        )

    payloads = build_payloads(
        names=names,
        centers=centers,
        mins=mins,
        maxs=maxs,
        grasped=grasped,
        goals=payload_goals,
        n_envs=n_envs,
        long_axis_yaws=yaws,
    )
    return payloads
