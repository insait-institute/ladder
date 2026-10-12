from typing import List, Optional, Tuple

from ladder.components.inference.api.client.episode import Episode
from ladder.components.inference.api.client.metrics.registry import Fraction, metric

# How well the controller DECIDED, judged against the env's per-goal state: whether the verifier's
# verdicts were true, and whether each advance came at the right moment. Only a hierarchical run has
# the events these read, so a flat episode reports nothing here.


@metric
def verifier_precision(episode: Episode) -> Optional[Fraction]:
    """Of the subtasks the verifier declared done, the share that really were. Ground truth is the
    env's own per-goal state, recorded on each verdict."""
    calls = [e for e in episode.of_kind("verify") if e.get("gt") is not None and e.get("verdict")]
    return (float(sum(bool(e["gt"]) for e in calls)), float(len(calls))) if calls else None


@metric
def verifier_recall(episode: Episode) -> Optional[Fraction]:
    """Of the subtasks that really were done, the share the verifier caught."""
    calls = [e for e in episode.of_kind("verify") if e.get("gt")]
    return (float(sum(bool(e["verdict"]) for e in calls)), float(len(calls))) if calls else None


@metric
def verify_failed(episode: Episode) -> Optional[Fraction]:
    """Verifier calls that returned no verdict, over all calls."""
    calls = episode.of_kind("verify")
    return (float(sum(e.get("verdict") is None for e in calls)), float(len(calls))) if calls else None


def _switches(episode: Episode) -> List[Tuple[int, int]]:
    """(goal left, step) per advance. Judged against ``goal_trace`` rather than the instantaneous
    flag on the event: a placement predicate flickers, so a switch landing in a trough would score
    as abandoning a goal that was plainly already complete.

    Only subtasks that correspond to a goal can be judged: a GENERATIVE source invents its own plan,
    so its indices can run past the env's goal count, and those switches have no ground truth to be
    right or wrong about."""
    goals = len(episode.goals)
    return [
        (int(event["from"]), int(event["step"]))
        for event in episode.of_kind("advance")
        if event.get("from") is not None and event.get("step") is not None and int(event["from"]) < goals
    ]


@metric
def premature_switch(episode: Episode) -> Optional[Fraction]:
    """Advances off a goal not yet achieved when the switch happened (the controller's JUDGEMENT).

    Deliberately not softened by a tolerance window around the switch time: premature switches
    whose goal completes later typically do so long after, so a window small enough to mean "nearly
    right" changes nothing, and one wide enough to matter is really crediting a later revisit. Harm
    is :func:`abandoned_goal`."""
    switches = _switches(episode)
    if not switches or not episode.goal_trace:
        return None
    early = sum(
        1 for goal, step in switches if (first := episode.first_achieved(goal)) is None or first > step
    )
    return float(early), float(len(switches))


@metric
def abandoned_goal(episode: Episode) -> Optional[Fraction]:
    """Advances off a goal the episode never completed at all: the premature switches that actually
    cost something. It can rank systems differently from :func:`premature_switch`, since a system
    may switch early often yet return later and strand little work."""
    switches = _switches(episode)
    if not switches or not episode.goal_trace:
        return None
    return float(sum(1 for goal, _ in switches if episode.first_achieved(goal) is None)), float(len(switches))


@metric
def switch_lag(episode: Episode) -> Optional[Fraction]:
    """Mean steps between a goal becoming achieved and the controller moving off it: budget spent
    commanding work already finished. Counts only advances off a goal satisfied AT THE MOMENT of the
    switch, measured from the most recent time it became satisfied.

    Measuring from the FIRST achievement instead conflates this with regression: a goal completed
    at step 18, lost at 27, regained at 259 and advanced off at 262 would score 244 rather than 3."""
    lags: List[int] = []
    for goal, step in _switches(episode):
        achieved_at = episode.achieved_since(goal, step)
        if achieved_at is not None:
            lags.append(step - achieved_at)
    return (float(sum(lags)), float(len(lags))) if lags else None


@metric
def instruction_churn(episode: Episode) -> Optional[Fraction]:
    """Distinct instructions issued, per goal. 1.0 is one instruction per goal; a flat run scores
    below it and a system that re-decides mid-goal scores above."""
    goals = episode.goals
    if not goals or not episode.instruction_history:
        return None
    return float(len(set(episode.instruction_history))), float(len(goals))
