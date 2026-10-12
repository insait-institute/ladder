import dataclasses
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)

# Goal-terminating predicate name -> command template, filled from the predicate's bound
# noun kwargs. Only the placement/stack predicates that CARRY a goal's score need an entry;
# any other terminating predicate falls back to the whole-task instruction (logged once).
# Extend by adding one line when a new goal predicate is used.
_TEMPLATES: Dict[str, str] = {
    "object_on_top": "put the {object} on the {reference_object}.",
    "object_in_container": "put the {object} in the {container}.",
    "object_inside": "put the {object} in the {container}.",
    "object_on_bottom": "put the {object} in the {reference_object}.",
}


# Placement predicate -> (moved-object kwarg, destination kwarg, relation). These are the
# predicates that say WHERE something has to end up, which is what a privileged policy drives to.
_PLACEMENTS: Dict[str, Tuple[str, str, str]] = {
    "object_on_top": ("object", "reference_object", "on"),
    "object_in_container": ("object", "container", "in"),
    "object_inside": ("object", "container", "in"),
    "object_on_bottom": ("object", "reference_object", "in"),
}


@dataclasses.dataclass(frozen=True)
class OracleGoal:
    """One score-carrying goal: its command string, the check that credits it, and its legs.

    ``legs`` is the ordered chain the goal is scored through -- a relay's intermediate stops, then
    the final placement. Each leg pairs its own predicate with the geometry it names, so a
    privileged policy can drive the chain and a text planner can read the command, both off the
    same enumeration.
    """

    command: str
    check: Callable
    legs: Sequence[Tuple[Callable, Optional[Tuple[str, str, str]]]]


def goal_geometry(predicate: Callable) -> Optional[Tuple[str, str, str]]:
    """(moved object, destination, relation) for a placement predicate, else None.

    Read off the predicate's own bound kwargs -- the same source ``_render`` draws its nouns from --
    so the geometry a policy drives to and the condition the task scores agree by construction.
    """
    func = getattr(predicate, "func", predicate)
    kwargs = getattr(predicate, "keywords", None) or {}
    name = getattr(func, "__name__", "")
    if name == "stacked":
        objects = kwargs.get("objects") or []
        return (objects[0], objects[1], "on") if len(objects) >= 2 else None
    entry = _PLACEMENTS.get(name)
    if entry is None:
        return None
    obj_key, dest_key, relation = entry
    if isinstance(kwargs.get(obj_key), str) and isinstance(kwargs.get(dest_key), str):
        return (kwargs[obj_key], kwargs[dest_key], relation)
    return None


def _noun(asset_id: str, instruction: str) -> str:
    """Asset id -> a readable noun, grounded in the task instruction's own words. The per-token
    instance-index suffix is stripped first ("lime01" -> "lime", "orange_01" -> "orange") so that
    duplicated objects ground to their base noun. Keep the id tokens that appear in the instruction
    ("bin_a06" + "...into the bin" -> "bin"; "plate_large" + "...on the plate" -> "plate"); when the
    instruction never names the object (an abstract task), fall back to the stripped id
    ("spam_can" -> "spam can")."""
    tokens = [re.sub(r"\d+$", "", token) for token in asset_id.lower().split("_")]
    tokens = [token for token in tokens if token]
    words = set(re.findall(r"[a-z]+", instruction.lower()))
    grounded = [token for token in tokens if token in words]
    if grounded:
        return " ".join(grounded)
    return " ".join(tokens) or asset_id


def _render(predicate: Callable, instruction: str) -> str:
    """One command string for a goal-terminating predicate, from its name + bound nouns; each
    noun is grounded in the task ``instruction`` (which is also the no-template fallback)."""
    func = getattr(predicate, "func", predicate)  # unwrap functools.partial
    kwargs = getattr(predicate, "keywords", None) or {}
    name = getattr(func, "__name__", "")
    if name == "stacked":
        objects = [_noun(obj, instruction) for obj in kwargs.get("objects", [])]
        return "stack the " + " on the ".join(objects) + "." if objects else instruction
    template = _TEMPLATES.get(name)
    if template is None:
        LOGGER.warn(f"robolab oracle: no command template for predicate {name!r}; using task instruction")
        return instruction
    nouns = {key: _noun(value, instruction) for key, value in kwargs.items() if isinstance(value, str)}
    try:
        return template.format(**nouns)
    except KeyError:
        LOGGER.warn(f"robolab oracle: template for {name!r} missing a noun; using task instruction")
        return instruction


def build_oracle(subtasks, task_instruction: str) -> List[OracleGoal]:
    """Synthesize the privileged oracle plan from the env's OWN subtask spec: one goal per
    score-carrying condition, in goal order. The command is rendered from the goal's terminating
    predicate (verb + bound object/target nouns) and the check IS that predicate, so a plan and the
    credit it is scored against stay aligned by construction. This is the RoboLab-side analogue of
    SimplerEnv's ``oracle_subtasks`` / ``oracle_goal_keys`` (derived from the env's goals, not a
    hand-authored plan); the server just forwards the result.

    The goal enumeration mirrors what RoboLab scores against: a ``logical="all"`` multi-object group
    yields one goal per object (its final, score-carrying condition); ``"any"`` collapses to one
    group-level goal; ``"choose"`` yields K goals (goal i-1 = "at least i of the group placed").
    Returns ``[]`` when the task declares no subtask spec, so the caller publishes the single
    whole-task goal instead."""
    goals: List[OracleGoal] = []
    for subtask in subtasks or []:
        chains = list(subtask.conditions.values())
        finals = [conditions[-1][0] for conditions in chains]
        if subtask.logical == "all" and len(finals) > 1:
            for chain in chains:
                goals.append(_goal(chain[-1][0], task_instruction, chain))
        elif subtask.logical == "any":
            check = lambda env, env_id=None, fs=finals: any(f(env, env_id=env_id) for f in fs)  # noqa: E731
            goals.append(_goal(finals[0], task_instruction, chains[0], check=check))
        elif subtask.logical == "choose":
            for need in range(1, (subtask.K or 1) + 1):
                index = min(need - 1, len(finals) - 1)
                check = lambda env, env_id=None, fs=finals, need=need: (  # noqa: E731
                    sum(bool(f(env, env_id=env_id)) for f in fs) >= need
                )
                goals.append(_goal(finals[index], task_instruction, chains[index], check=check))
        else:
            goals.append(_goal(finals[0], task_instruction, chains[0]))
    return goals


def _goal(predicate: Callable, instruction: str, chain, check: Optional[Callable] = None) -> OracleGoal:
    """One goal from its terminating predicate and the ordered condition chain reaching it."""
    return OracleGoal(
        command=_render(predicate, instruction),
        check=check if check is not None else predicate,
        legs=[(leg[0], goal_geometry(leg[0])) for leg in chain],
    )
