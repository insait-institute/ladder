import dataclasses
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

IN = "in"
ON = "on"

_NUMBER = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


@dataclasses.dataclass(frozen=True)
class Placement:
    """One leg: which object, and where it has to end up."""

    noun: str
    preposition: str  # ON or IN, per the receptacle
    receptacle: str


def _join(phrases: Sequence[str]) -> str:
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def _quantify(noun: str, count: int, definite: bool = True) -> str:
    if not definite:  # the same type is wanted in two places, so "the spoon" would name one twice
        return f"{_NUMBER[count]} {noun}" + ("" if count == 1 else "s")
    if count == 1:
        return f"the {noun}"
    if count == 2:
        return f"both {noun}s"
    return f"the {_NUMBER.get(count, str(count))} {noun}s"


def _clauses(placements: Sequence[Placement]) -> List[Tuple[str, str, str]]:
    """(objects, preposition, receptacle) per destination, in first-mention order."""
    grouped: "OrderedDict[Tuple[str, str], OrderedDict[str, int]]" = OrderedDict()
    for p in placements:
        key = (p.receptacle, p.preposition)
        grouped.setdefault(key, OrderedDict())
        grouped[key][p.noun] = grouped[key].get(p.noun, 0) + 1
    spread = {
        noun
        for noun in {n for counts in grouped.values() for n in counts}
        if sum(noun in counts for counts in grouped.values()) > 1
    }
    return [
        (_join([_quantify(n, c, n not in spread) for n, c in counts.items()]), preposition, receptacle)
        for (receptacle, preposition), counts in grouped.items()
    ]


def compose(
    placements: Sequence[Placement],
    *,
    sequential: bool = False,
    source: Optional[str] = None,
    excluded: Sequence[str] = (),
) -> str:
    """A composition task's instruction, in the style all three benchmarks share.

    Objects are named except where the task hands the system objects to leave alone: resolving
    "everything except the red block" against the scene is what a scope task tests, while a bare
    "everything" reads the same at two objects as at ten. ``sequential`` keeps one clause per leg,
    for a task scored on the order or on the route taken.
    """
    if excluded:
        counts: "OrderedDict[str, int]" = OrderedDict()
        for noun in excluded:
            counts[noun] = counts.get(noun, 0) + 1
        what = "everything except " + _join([_quantify(n, c) for n, c in counts.items()])
        _, preposition, receptacle = _clauses(placements)[0]
        body = f"put {what} {preposition} the {receptacle}"
    elif source is not None:
        what, _, receptacle = _clauses(placements)[0]
        body = f"move {what} from the {source} to the {receptacle}"
    elif sequential:
        steps = [f"put the {p.noun} {p.preposition} the {p.receptacle}" for p in placements]
        body = steps[0] if len(steps) == 1 else "first " + ", then ".join(steps)
    else:
        body = _join([f"put {what} {prep} the {where}" for what, prep, where in _clauses(placements)])
    # A one-leg task's instruction IS its only subtask, so the two match rather than differ by a stop.
    return f"{body}." if len(placements) == 1 else body


def subtask(placement: Placement) -> str:
    """One commanded goal, the unit an orchestrator hands the policy."""
    return compose([placement])
