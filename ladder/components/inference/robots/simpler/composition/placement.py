import dataclasses
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from transforms3d.euler import euler2quat

CARROT = "bridge_carrot_generated_modified"
PLATE = "bridge_plate_objaverse_larger"
SPOON = "bridge_spoon_generated_modified"
TOWEL = "table_cloth_generated_shorter"
EGGPLANT = "eggplant"  # a sink-scene object in SimplerEnv; reused here on the flat table

INSTANCE_SEP = "#"  # a source model used twice becomes "<model>#0", "<model>#1"

# Drop heights and orientations, reused verbatim from the single bridge tasks (the eggplant mirrors the
# carrot, same ~3.8 cm thickness). Every object lies flat with its long axis on a world axis.
_FLAT = (1.0, 0.0, 0.0, 0.0)
_OBJ_Z: Dict[str, float] = {
    CARROT: 0.887529,
    PLATE: 0.869532,
    SPOON: 0.88,
    TOWEL: 0.875,
    EGGPLANT: 0.887529,
}
_OBJ_QUAT: Dict[str, Tuple[float, float, float, float]] = {
    CARROT: tuple(float(x) for x in euler2quat(0, 0, np.pi)),
    SPOON: _FLAT,
    PLATE: _FLAT,
    TOWEL: _FLAT,
    EGGPLANT: _FLAT,
}
# Models that do not keep the yaw they are spawned with, so their footprint has to be bounded under any
# orientation. The curved eggplant turns as it settles and outgrows its box, while the carrot and the
# spoon hold their pose. Bounding every object this way would be too costly: the circumscribed square of
# a 14 cm spoon leaves dense scenes such as move_plate_to_towel_5 with too few valid layouts.
_TUMBLES = frozenset({EGGPLANT})
# From info_bridge_custom_v0.json. Used only to lay out a scene; success geometry reads the live model_db.
BBOX_XY: Dict[str, Tuple[float, float]] = {
    CARROT: (0.115, 0.032),
    PLATE: (0.11, 0.11),
    SPOON: (0.14, 0.035),
    TOWEL: (0.10, 0.10),
    EGGPLANT: (0.078, 0.037),
}

_REACH_X_MIN, _REACH_X_MAX = -0.30, -0.05  # the reachable widowx workspace
_REACH_Y_MAX = 0.16
_OVERLAP_MARGIN_M = 0.01
# Slack on "the source is over the target", from BaseBridgeEnv._evaluate. Shared with composition.envs so
# the region the packer keeps movers out of is the same one the success test reads.
FOOTPRINT_SLACK_M = 0.003
_NEAR_M = 0.08  # how far outside its target's edge a source may spawn
# A transfer mover starts this far above its start receptacle's TOP FACE, so it settles onto it rather
# than intersecting it. Measuring the drop from the receptacle, not the table, keeps the fall short, so
# a curved mover does not bounce off the plate and roll toward its destination.
_RECEP_TOP_M = 0.0025
_ON_RECEP_CLEAR_M = _RECEP_TOP_M + 0.0035


@dataclasses.dataclass
class Goal:
    source: str
    target: str
    completely_on: bool = True  # require contact, not just footprint overlap


def strip_instance_id(name: str) -> str:
    """'carrot#1' -> 'carrot'. The inverse of ``specs._with_instance_ids``, which gives a source used more
    than once its own instance name: every per-model table (:data:`BBOX_XY`, :data:`_OBJ_QUAT`, the env's
    ``model_db``, ``Spec.target_scales``) is keyed by the model, so an instance has to be stripped first."""
    return name.split(INSTANCE_SEP, 1)[0]


def _rot_half(quat: Tuple[float, float, float, float], size: Tuple[float, float]) -> Tuple[float, float]:
    """Axis-aligned half-extents of a ``size`` footprint under the z-rotation ``quat`` (wxyz)."""
    angle = 2.0 * float(np.arctan2(quat[3], quat[0]))
    cos, sin = abs(np.cos(angle)), abs(np.sin(angle))
    hx, hy = 0.5 * size[0], 0.5 * size[1]
    return hx * cos + hy * sin, hx * sin + hy * cos


def _yaw_free_half(size: Tuple[float, float]) -> float:
    """Half-diagonal of a ``size`` footprint: the bound that holds under ANY yaw.

    Not every mover keeps the orientation it is spawned with (a curved body turns as it settles), so a
    destination kept clear only of the spawn box is not kept clear of where the mover comes to rest.
    """
    return 0.5 * float(np.hypot(size[0], size[1]))


def _clear_of_targets(
    name: str,
    x: float,
    y: float,
    receps: Dict[str, Tuple[float, float]],
    goal_targets: Set[str],
    halves: Dict[str, Tuple[float, float]],
    bbox: Dict[str, Tuple[float, float]],
) -> bool:
    """No part of a mover may come to rest over a goal destination.

    Testing only the mover's centre (what the success predicate reads) can leave its body hanging over
    the destination's edge, so a goal could start half done or even be credited before the policy acts.
    """
    rx, ry = (_yaw_free_half(bbox[name]),) * 2 if strip_instance_id(name) in _TUMBLES else halves[name]
    return not any(
        abs(x - receps[t][0]) < rx + halves[t][0] + FOOTPRINT_SLACK_M
        and abs(y - receps[t][1]) < ry + halves[t][1] + FOOTPRINT_SLACK_M
        for t in goal_targets
        if t in receps
    )


def build_layouts(
    obj_names: List[str],
    bbox: Dict[str, Tuple[float, float]],
    goals: List[Goal],
    n_layouts: int,
    on_recep: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(xyz [P, n, 3], quat [1, n, 4])`` in ``obj_names`` order; ``BaseBridgeEnv`` pairs episode ``e``
    with ``(xyz[e // Q], quat[e % Q])``. Receptacles take the back row, then the movers are drawn around
    them. Seeded, so a scene's object positions are a fixed property of the code. Orientation never varies:
    the packing is too tight for a rotated spoon to fit beside its neighbours."""
    goal_targets = {goal.target for goal in goals}
    src_to_tgt = {goal.source: goal.target for goal in goals}
    halves = {name: _rot_half(_OBJ_QUAT[strip_instance_id(name)], bbox[name]) for name in obj_names}

    def by_area(names: List[str]) -> List[str]:
        return sorted(names, key=lambda name: halves[name][0] * halves[name][1], reverse=True)

    tgt_names = by_area([name for name in obj_names if name in goal_targets or name == on_recep])
    movers = by_area([name for name in obj_names if name not in goal_targets and name != on_recep])
    rng = np.random.RandomState(1234)
    # Transfer scenes hold the receptacles further forward, so the movers resting on the start receptacle
    # stay inside the reachable band.
    back_x_min = _REACH_X_MIN + 0.02 if on_recep else _REACH_X_MIN

    def fits(name: str, x: float, y: float, placed: Dict[str, Tuple[float, float]]) -> bool:
        if not (_REACH_X_MIN <= x <= _REACH_X_MAX and abs(y) <= _REACH_Y_MAX):
            return False
        hx, hy = halves[name]
        return all(
            abs(x - px) >= hx + halves[other][0] + _OVERLAP_MARGIN_M
            or abs(y - py) >= hy + halves[other][1] + _OVERLAP_MARGIN_M
            for other, (px, py) in placed.items()
        )

    def draw_mover(name: str, attempt: int, receps: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
        """Inside the transfer start receptacle; else a ring just outside this source's own goal target;
        else anywhere, once that ring has proved too crowded, as it does in the dense scenes."""
        if on_recep:
            px, py = receps[on_recep]
            phx, phy = halves[on_recep]
            hx, hy = halves[name]
            return (
                px + (float(rng.uniform(-(phx - hx), phx - hx)) if phx > hx else 0.0),
                py + (float(rng.uniform(-(phy - hy), phy - hy)) if phy > hy else 0.0),
            )
        target = src_to_tgt.get(name)
        if target in receps and attempt < 400:
            tx, ty = receps[target]
            rmin = halves[name][0] + halves[target][0] + _OVERLAP_MARGIN_M
            radius = float(rng.uniform(rmin, rmin + _NEAR_M))
            angle = float(rng.uniform(0, 2 * np.pi))
            return tx + radius * float(np.cos(angle)), ty + radius * float(np.sin(angle))
        return (
            float(rng.uniform(_REACH_X_MIN, _REACH_X_MAX)),
            float(rng.uniform(-_REACH_Y_MAX, _REACH_Y_MAX)),
        )

    layouts: List[List[List[float]]] = []
    tries = 0
    while len(layouts) < n_layouts and tries < n_layouts * 120:
        tries += 1
        recep_pos: Dict[str, Tuple[float, float]] = {}
        for name in tgt_names:
            for _ in range(300):
                x = float(rng.uniform(back_x_min, back_x_min + 0.10))
                y = float(rng.uniform(-_REACH_Y_MAX, _REACH_Y_MAX))
                if fits(name, x, y, recep_pos):
                    recep_pos[name] = (x, y)
                    break
            else:
                break  # abandon the layout as soon as one object will not fit; retrying wastes draws
        if len(recep_pos) != len(tgt_names):
            continue

        # A transfer mover overlaps the receptacle it starts on, so check it against the movers only. That
        # drops the separation from the goal targets that _clear_of_targets then has to re-impose.
        mover_pos: Dict[str, Tuple[float, float]] = {} if on_recep else dict(recep_pos)
        for name in movers:
            for attempt in range(400 if on_recep else 600):
                x, y = draw_mover(name, attempt, recep_pos)
                if fits(name, x, y, mover_pos) and _clear_of_targets(
                    name, x, y, recep_pos, goal_targets, halves, bbox
                ):
                    mover_pos[name] = (x, y)
                    break
            else:
                break
        pos = {**recep_pos, **mover_pos}
        if len(pos) != len(obj_names):
            continue
        layouts.append(
            [
                [
                    pos[name][0],
                    pos[name][1],
                    _OBJ_Z[strip_instance_id(name)]
                    + (_ON_RECEP_CLEAR_M if on_recep and name in movers else 0.0),
                ]
                for name in obj_names
            ]
        )
    if not layouts:
        raise AssertionError(f"cannot pack {obj_names}" + (f" on {on_recep}" if on_recep else ""))
    quat = np.array([[_OBJ_QUAT[strip_instance_id(name)] for name in obj_names]], dtype=np.float32)
    return torch.tensor(np.array(layouts, dtype=np.float32)), torch.tensor(quat)
