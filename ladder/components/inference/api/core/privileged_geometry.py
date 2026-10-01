from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ladder.components.inference.api.core.privileged_state import (
    encode_condition,
    encode_object,
    encode_state,
)

_NEIGHBOUR_GAP_M = 0.005  # so two objects sharing a destination do not touch
_IN_CLEARANCE_M = 0.005  # drop height above a container's floor


def to_base(world_to_base: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map world-frame points ``[..., 3]`` into the robot base frame the policy commands in."""
    flat = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.concatenate([flat, np.ones((len(flat), 1))], axis=1)
    return (homogeneous @ np.asarray(world_to_base, dtype=np.float64).T)[:, :3].reshape(points.shape)


def spread_offsets(
    half_extents: Sequence[np.ndarray], dest_half: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Where to lay N objects on one destination, as XY offsets plus the layer each one sits in.

    Piling is a real failure mode: every arrival dropped on the same point knocks the last one off.
    The constraint is each object's FOOTPRINT, not its centre -- a grid that only keeps centres
    inside the destination still aims every object so its own body overhangs the rim, which for a
    container means setting it down on the wall. Measured on RoboLab's bins, that overhang credited
    a delivery and then lost it: one object per bin succeeded 8/8, three or more never did.

    When a grid sized for the largest object cannot hold them all, each object's own footprint is
    laid in rows along the destination's longer side instead, and what still does not fit piles on
    the floor. Stacked layers aimed every arrival past the first higher above the destination, and a
    fruit dropped from 0.2-0.7 m above RoboLab's bin bounced out of it (6 objects: 0/8).
    """
    count = len(half_extents)
    offsets = np.zeros((count, 2))
    layers = np.zeros(count, dtype=int)
    if count <= 1:
        return offsets, layers

    source_half = np.array([max(h[0] for h in half_extents), max(h[1] for h in half_extents)])
    step = 2.0 * source_half + _NEIGHBOUR_GAP_M
    # The centre of an object at the extreme cell must sit this far in for its edge to clear the rim.
    usable = np.maximum(np.asarray(dest_half[:2], dtype=np.float64) - _NEIGHBOUR_GAP_M - source_half, 0.0)
    cells = np.maximum(np.floor(2.0 * usable / step).astype(int) + 1, 1)

    per_layer = int(cells[0] * cells[1])
    if count > per_layer:
        return _row_offsets(half_extents, np.asarray(dest_half[:2], dtype=np.float64)), layers
    for index in range(count):
        cell, layers[index] = index % per_layer, index // per_layer
        grid_xy = np.array([cell % cells[0], cell // cells[0]], dtype=np.float64)
        offsets[index] = (grid_xy - (cells - 1) / 2.0) * step
    return offsets, layers


def _row_offsets(half_extents: Sequence[np.ndarray], dest_half: np.ndarray) -> np.ndarray:
    """XY offsets laying each object's own footprint in rows along ``dest_half``'s longer axis."""
    along = 0 if dest_half[0] >= dest_half[1] else 1
    across = 1 - along
    length = 2.0 * float(dest_half[along]) - _NEIGHBOUR_GAP_M
    rows: List[List[int]] = [[]]
    used = 0.0
    for index, half in enumerate(half_extents):
        width = 2.0 * float(half[along]) + _NEIGHBOUR_GAP_M
        if rows[-1] and used + width > length:
            rows.append([])
            used = 0.0
        rows[-1].append(index)
        used += width
    depths = [max(2.0 * float(half_extents[i][across]) for i in row) + _NEIGHBOUR_GAP_M for row in rows]
    offsets = np.zeros((len(half_extents), 2))
    row_start = -0.5 * sum(depths)
    for row, depth in zip(rows, depths, strict=True):
        widths = [2.0 * float(half_extents[i][along]) + _NEIGHBOUR_GAP_M for i in row]
        start = -0.5 * sum(widths)
        for index, width in zip(row, widths, strict=True):
            offsets[index, along] = start + 0.5 * width
            offsets[index, across] = row_start + 0.5 * depth
            start += width
        row_start += depth
    # Rows past the destination's edge are pulled back in: those objects pile on the others.
    limit = np.maximum(dest_half[None, :] - np.asarray(half_extents)[:, :2] - _NEIGHBOUR_GAP_M, 0.0)
    return np.clip(offsets, -limit, limit)


def goal_point(
    relation: str,
    dest_center: np.ndarray,
    dest_min: np.ndarray,
    dest_max: np.ndarray,
    source_half: np.ndarray,
    offset_xy: np.ndarray,
    layer: int = 0,
) -> Tuple[np.ndarray, float]:
    """Where a source object's CENTRE must end up, and the EEF height to release it at.

    ``on`` rests it on the destination's top face; ``in`` drops it just above the floor of the
    container, which is what ``is_inside`` scores. Those differ for a walled destination: the object
    must END at the floor, but it must be RELEASED above the rim -- carrying it down inside fouls the
    gripper on the wall and on anything already delivered. Releasing above the rim lets it fall the
    last centimetres, which is how the container gets loaded.
    """
    # "on" rests the object on the top face; "in" aims just above the container's AABB floor,
    # which ``is_inside`` credits.
    rim_z = float(dest_max[2]) + float(source_half[2])
    z = rim_z
    if relation == "in":
        z = float(dest_min[2]) + float(source_half[2]) + _IN_CLEARANCE_M
    # A stacked layer rests on the layer below it.
    z += layer * 2.0 * float(source_half[2])
    point = np.array([dest_center[0] + offset_xy[0], dest_center[1] + offset_xy[1], z], dtype=np.float32)
    return point, max(rim_z, z)


def build_payloads(
    *,
    names: Sequence[str],
    centers: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    grasped: np.ndarray,
    goals: Sequence[Dict[str, Any]],
    n_envs: int,
    long_axis_yaws: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """One privileged payload per env. Arrays are ``[n_objects, n_envs, 3]`` in the base frame.

    Each entry of ``goals`` is ``{"source": idx, "destination": idx | None, "relation": str,
    "satisfied": [n_envs] bool}``, optionally with ``"predicate"`` (default ``"pick_and_place"``;
    ``pickup``/``grasp``/``reach`` end the motion early) and ``"point"`` -- an explicit goal
    position, ``[3]`` or ``[n_envs, 3]``, for a goal whose destination is a place rather than an
    object, which is how a region or a relative-placement task states where the object must end up.
    """
    halves = 0.5 * (maxs - mins)
    per_destination: Dict[Optional[int], List[int]] = {}
    for index, goal in enumerate(goals):
        per_destination.setdefault(goal["destination"], []).append(index)

    last_destination = {goal["source"]: goal["destination"] for goal in goals}

    payloads = []
    for env in range(n_envs):
        offsets = {}
        for destination, indices in per_destination.items():
            if destination is None:
                continue
            # One slot per object, not per goal: a relay or revisit brings the same object back to
            # the same destination, and a slot of its own per visit aims it at the rim. Only objects
            # left here need slots side by side; when every visitor moves on again (a relay's pot),
            # they are there one at a time, so each takes the centre.
            sources = list(dict.fromkeys(goals[i]["source"] for i in indices))
            if not any(last_destination[source] == destination for source in sources):
                offsets.update({i: (np.zeros(2), 0) for i in indices})
                continue
            spread, layers = spread_offsets(
                [halves[source, env] for source in sources], halves[destination, env]
            )
            slots = dict(zip(sources, zip(spread, layers, strict=True), strict=True))
            offsets.update({i: slots[goals[i]["source"]] for i in indices})

        conditions = []
        for index, goal in enumerate(goals):
            source = goal["source"]
            target = encode_object(
                names[source],
                centers[source, env],
                mins[source, env],
                maxs[source, env],
                display_name=names[source],
                grasped=bool(grasped[source, env]),
                long_axis_yaw_rad=(None if long_axis_yaws is None else float(long_axis_yaws[source, env])),
            )
            destination = goal["destination"]
            point, release_z = None, None
            explicit = goal.get("point")
            if explicit is not None:
                explicit = np.asarray(explicit, dtype=np.float64)
                point = explicit[env] if explicit.ndim == 2 else explicit
                release_z = float(point[2])
            elif destination is not None:
                point, release_z = goal_point(
                    goal["relation"],
                    centers[destination, env],
                    mins[destination, env],
                    maxs[destination, env],
                    halves[source, env],
                    offsets[index][0],
                    layer=int(offsets[index][1]),
                )
            conditions.append(
                encode_condition(
                    goal.get("predicate", "pick_and_place"),
                    target,
                    relation=goal["relation"],
                    goal_point_m=point,
                    release_z_m=release_z,
                    destination=(names[destination] if destination is not None else None),
                    grasped=bool(grasped[source, env]),
                    satisfied=bool(goal["satisfied"][env]),
                )
            )
        objects = [
            encode_object(
                name,
                centers[i, env],
                mins[i, env],
                maxs[i, env],
                display_name=name,
                grasped=bool(grasped[i, env]),
                long_axis_yaw_rad=(None if long_axis_yaws is None else float(long_axis_yaws[i, env])),
            )
            for i, name in enumerate(names)
        ]
        payloads.append(encode_state(conditions, objects))
    return payloads
