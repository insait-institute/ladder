import collections
import re
from typing import Counter, Dict, Iterable, List, Optional, Tuple

from ladder.components.inference.api.client.episode import Episode
from ladder.components.inference.api.client.metrics.registry import Fraction, metric

# ---------------------------------------------------------------------------------------------
# READ THIS BEFORE TRUSTING ANYTHING IN THIS MODULE.
#
# Alone among the metric modules, these ask about LANGUAGE, and they answer by pattern-matching
# English. Everything below is a HEURISTIC CALIBRATED ON THE THREE SUITES WE HAVE TODAY
# (SimplerEnv-Comp, RobotArena-Comp, RoboLab-Comp), not a general definition of a well-formed
# subtask. The patterns restate, by hand, the subtask phrasings the action experts are trained on,
# so they are a SECOND copy of something defined elsewhere and the two can drift apart silently: a
# phrasing added to training is NOT picked up here, and every plan using it reads as out-of-register.
#
# It assumes, and breaks without warning outside of, all of:
#   * English, imperative, one object to one destination
#   * single-arm pick-and-place; no other skill family is expressible
#   * our object vocabulary -- `_MODIFIERS` drops "toy" and "block" ONLY because RobotArena names
#     objects "sushi toy" and "blue block" while plans say "sushi" and "blue"
#   * our embodiments -- `_ROBOT_PARTS` enumerates what a WidowX/Franka planner calls the hardware
#
# So a low `plan_register` is evidence a planner drifted; a high one is only evidence it matched
# THESE patterns. Treat the numbers as comparable across planners on these benchmarks and not as
# an absolute measure of subtask quality. Revisit whenever a suite, phrasing or embodiment is added,
# and prefer deriving the accepted phrasings from wherever training defines them over extending the
# patterns here by hand.
# ---------------------------------------------------------------------------------------------
# The union of phrasings the action experts are trained on: a whole pick-and-place, a stack, a move
# off a raised source, and the two halves of the two-phase register RoboLab's plans use. A UNION
# rather than one register, because the question is whether a subtask is a form the policy has seen
# AT ALL, not which register a checkpoint prefers -- "pick up the banana" is in-distribution for a
# two-phase plan and drift for a single-phase one. What it excludes is what actually goes wrong when
# a planner is left unconstrained: motion primitives, coordinates, several goals per line, prose.
_FORMS = re.compile(
    r"(put|place|set) the (?P<a>.+) (on|in|into|onto) the .+"
    r"|stack the (?P<b>.+) on( top of)? the .+"
    r"|move the (?P<c>.+?) (from the .+ )?to the .+"
    r"|(pick up|grasp) the (?P<d>[\w ]+)"
)
# One step commands ONE object, so a conjunction means several goals were packed onto a line, and a
# robot part in the object slot means a motion primitive wearing the shape of a placement ("move the
# gripper to the carrot"). Both match the forms above and neither is a subtask the policy can follow.
_CONJUNCTION = re.compile(r"\b(and|then)\b|[;,]")
_ROBOT_PARTS = frozenset({"gripper", "arm", "hand", "end effector", "wrist"})


def normalize(text: str) -> str:
    """Subtask/instruction equality across suites, tolerating quoting and a trailing stop."""
    return text.lower().strip().strip("\"'").rstrip(".").strip()


def in_register(step: str) -> bool:
    """Whether one subtask is phrased in a form the action experts are trained on. Public because
    the subtask-state benchmark scores single generated subtasks; the scope note above applies in full."""
    text = step.lower().strip().rstrip(".")
    match = _FORMS.fullmatch(text)
    if match is None or _CONJUNCTION.search(text):
        return False
    subject = next((group for group in match.groupdict().values() if group), "")
    return subject.strip() not in _ROBOT_PARTS


# A goal as (object, destination) head nouns, which is what lets a plan be compared against the
# env's reference across wordings: "put the metal pot on the yellow cloth" and "put the pot on the
# cloth" name one goal, and the prompt REQUIRES the attribute when several receptacles are visible,
# so keeping modifiers would score the controller's own instructions as errors.
#
# Head-noun matching is lossy in BOTH directions and the caveats above apply here too. A genuine
# synonym scores as WRONG ("chicken leg" against the env's "chicken drumstick", a real disagreement
# in the RobotArena cache). That is not fixable by matching harder: it needs the env to publish goal
# identity rather than a sentence, which is also what would let these metrics stop parsing English
# altogether.
_GOAL = re.compile(
    r"(?:put|place|set|drop|deposit|stack)\s+(?P<what>.+?)\s+(?:on|in|into|onto)\s+(?:top of\s+)?(?P<where>.+)"
    r"|move\s+(?P<what2>.+?)\s+(?:from\s+.+?\s+)?to\s+(?P<where2>.+)"
)
_MODIFIERS = frozenset({"the", "a", "an", "toy", "block"})


def _head(phrase: str) -> str:
    words = [word for word in re.sub(r"[^a-z ]", "", phrase).split() if word not in _MODIFIERS]
    return words[-1] if words else ""


def goal_of(step: str) -> Optional[Tuple[str, str]]:
    """One subtask's (object, destination) head-noun pair, or None for anything that is not a
    single placement. Public for the same reason as :func:`in_register`."""
    match = _GOAL.fullmatch(step.lower().strip().strip("\"'").rstrip("."))
    if match is None:
        return None
    named = match.groupdict()
    what, where = named["what"] or named["what2"], named["where"] or named["where2"]
    return (_head(what), _head(where)) if what and where else None


def _goals(steps: Iterable[str]) -> Counter[Tuple[str, str]]:
    return collections.Counter(goal for step in steps if (goal := goal_of(step)))


def _commanded(episode: Episode) -> List[str]:
    """Distinct subtasks the controller handed the policy, in order.

    Not the plan: a stepwise system never forms one, and a planned system may command only a prefix
    of its plan. This is what the policy was actually asked to do, which is the right denominator
    for a question about phrasing.
    """
    seen, commanded = set(), []
    for step in episode.instruction_history:
        text = step.strip() if step else ""
        if text and text.lower() not in seen:
            seen.add(text.lower())
            commanded.append(text)
    return commanded


@metric
def plan_register(episode: Episode) -> Optional[Fraction]:
    """Commanded subtasks phrased in a form the action expert was trained on.

    The policy only follows what resembles its training instructions, so a controller can be
    perfectly right about the task and still unusable. Defined over what was COMMANDED rather than
    over a plan, so one number covers a cached plan, a VLM plan and a stepwise generator alike.
    A flat run commands only the compound task and is skipped: it is the task, not a subtask.

    Read the scope note at the top of this module before quoting this: a LOW value is evidence a
    planner drifted, a HIGH one only says it matched the patterns our current suites use.
    """
    commanded = _commanded(episode)
    if not commanded or (len(commanded) == 1 and commanded[0] == episode.instruction.strip()):
        return None
    return float(sum(in_register(step) for step in commanded)), float(len(commanded))


@metric
def plan_goals_missed(episode: Episode) -> Optional[Fraction]:
    """Reference goals the plan never names.

    A goal the plan omits cannot be reached however well the policy executes, so this is a ceiling
    on the episode that no amount of control quality lifts. Scored against the PLAN, not against
    what was commanded, because an episode that runs out of budget commands only a prefix and would
    otherwise read as under-planned.

    Goals are matched on head nouns and counted with multiplicity; see the scope note above
    ``_GOAL`` for what head-noun matching still misses.
    """
    reference = _goals(episode.subtasks)
    if not episode.plan or not reference:
        return None
    return float(sum((reference - _goals(episode.plan)).values())), float(sum(reference.values()))


@metric
def plan_goals_invented(episode: Episode) -> Optional[Fraction]:
    """Plan goals the environment's reference does not contain: an object or destination the task
    never asked for, which spends budget and can undo a goal already reached."""
    reference = _goals(episode.subtasks)
    if not episode.plan or not reference:
        return None
    return float(sum((_goals(episode.plan) - reference).values())), float(sum(reference.values()))


@metric
def plan_exact(episode: Episode) -> Optional[Fraction]:
    """Episodes whose plan names exactly the reference goals, neither missing nor inventing one."""
    reference = _goals(episode.subtasks)
    if not episode.plan or not reference:
        return None
    return float(_goals(episode.plan) == reference), 1.0


@metric
def plan_attempted(episode: Episode) -> Optional[Fraction]:
    """Plan steps the controller commanded before the episode ended: how much of the plan was
    actually tried, which separates a bad plan from one that was never reached."""
    if not episode.plan:
        return None
    commanded = {step.lower() for step in _commanded(episode)}
    return float(sum(step.strip().lower() in commanded for step in episode.plan)), float(len(episode.plan))


def score_plan(plan: Iterable[str], reference: Iterable[str]) -> Dict[str, Fraction]:
    """The same plan questions the metrics above ask, against a KNOWN reference rather than an episode.

    Exists so a decomposition benchmark -- a planner given one frame and a compound instruction, with
    the suite's own subtask list as ground truth -- scores through this module instead of restating
    the phrasings a fourth time. ``goals_missed`` and ``goals_invented`` mirror the metrics of the
    same name; ``register`` mirrors ``plan_register``; ``length`` reports the plan's length against
    the reference's, which is where a granularity error shows up (a pick-and-place split into a grasp
    and a place inflates it, a truncated plan deflates it).
    """
    steps, want = list(plan), list(reference)
    have, expected = _goals(steps), _goals(want)
    return {
        "exact": (float(have == expected), 1.0),
        "goals_missed": (float(sum((expected - have).values())), float(sum(expected.values()))),
        "goals_invented": (float(sum((have - expected).values())), float(sum(expected.values()))),
        "register": (float(sum(in_register(step) for step in steps)), float(len(steps))),
        "length": (float(len(steps)), float(len(want))),
    }
