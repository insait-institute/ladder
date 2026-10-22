from typing import Optional, Tuple

import numpy as np

from ladder.components.inference.api.core.privileged_state import PrivilegedCondition, PrivilegedState

# Connectives separating the manipulated object from its destination, longest / most-specific first
# so "on top of" beats "on" and "into" beats "in".
_CONNECTIVES: Tuple[str, ...] = (" on top of ", " onto ", " into ", " in ", " on ", " to ")
_IN_CONNECTIVES: Tuple[str, ...] = (" into ", " in ")
# Leading verbs stripped so only the object noun phrase remains.
_LEADING_VERBS: Tuple[str, ...] = (
    "pick up ",
    "put down ",
    "put ",
    "place ",
    "stack ",
    "move ",
    "set ",
    "grasp ",
    "lift ",
    "reach ",
)


def ground_subtask(subtask: str, state: PrivilegedState) -> Optional[PrivilegedCondition]:
    """Resolve a free-text subtask to a :class:`PrivilegedCondition`, or ``None`` if unparseable.

    "put/place/move/stack X on/in Y" -> pick_and_place(X -> resting-on Y); a verb with no
    destination ("pick up X", "grasp X") -> pickup(X). Object phrases are matched to scene objects
    by display name (see :meth:`PrivilegedState.find_object`). ``goal_point`` is the resting height
    of X's center on Y, the same geometry the structured path uses.
    """
    text = subtask.strip().rstrip(".").lower()
    if not text:
        return None

    relation: Optional[str] = None
    src_phrase, dst_phrase = text, ""
    for conn in _CONNECTIVES:
        idx = text.find(conn)
        if idx != -1:
            src_phrase = text[:idx]
            dst_phrase = text[idx + len(conn) :]
            relation = "in" if conn in _IN_CONNECTIVES else "on"
            break

    src_phrase = src_phrase.strip()
    for verb in _LEADING_VERBS:
        if src_phrase.startswith(verb):
            src_phrase = src_phrase[len(verb) :]
            break

    source = state.find_object(src_phrase)
    if source is None:
        return None

    if not dst_phrase.strip():
        return PrivilegedCondition(
            predicate="pickup",
            relation=None,
            target=source,
            goal_point_m=None,
            grasped=source.grasped,
            satisfied=False,
        )

    dest = state.find_object(dst_phrase)
    if dest is None:
        return None
    goal_point = _placement_goal(source, dest, state)
    return PrivilegedCondition(
        predicate="pick_and_place",
        relation=relation,
        target=source,
        goal_point_m=goal_point,
        grasped=source.grasped,
        satisfied=False,
    )


def _placement_goal(source, dest, state: PrivilegedState) -> np.ndarray:
    """Where ``source``'s center should go to rest on ``dest``.

    If other objects are already resting on ``dest`` (from earlier subtasks), offset ``source`` to a
    free slot along its narrow axis so several objects on one target sit side-by-side instead of
    piling at the center (the multi-object-on-one-target case). Mirrors the server's structured spread,
    but driven incrementally by what is already placed (one subtask is grounded at a time).
    """
    dc, dh = dest.center_m.astype(np.float64), dest.half_extent_m
    sh = source.half_extent_m
    goal = dc.copy()
    goal[2] = dc[2] + dh[2] + sh[2]

    radius = float(np.linalg.norm(dh[:2]))
    on_dest = [
        o
        for o in state.objects
        if o.name not in (source.name, dest.name)
        and float(np.linalg.norm((o.center_m - dc)[:2])) <= radius + 0.01
        and o.center_m[2] > dc[2]
    ]
    n_on = len(on_dest)
    if n_on > 0:
        axis = 1 if sh[1] <= sh[0] else 0  # spread along source's NARROW axis
        spacing = 2.0 * sh[axis] + 0.005
        slot = ((n_on + 1) // 2) * spacing * (1.0 if n_on % 2 == 1 else -1.0)
        cap = max(dh[axis] - 0.005, 0.0)
        goal[axis] += float(np.clip(slot, -cap, cap))
    return goal
