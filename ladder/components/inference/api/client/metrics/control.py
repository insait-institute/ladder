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
    flag on the event: a placement predicate flickers — one measured RobotArena episode satisfied and
    lost goal 0 five times — so a switch landing in a trough would score as abandoning a goal that
    was plainly already complete.

    Only subtasks that correspond to a goal can be judged: a GENERATIVE source invents its own plan,
    so its indices run past the env's goal count (a measured stepwise run reached subtask 4 on a
    2-goal scene) and those switches have no ground truth to be right or wrong about. Scoring them
    counted 10 of 24 stepwise advances as abandoning goals that never existed."""
    goals = len(episode.goals)
    return [
        (int(event["from"]), int(event["step"]))
        for event in episode.of_kind("advance")
        if event.get("from") is not None and event.get("step") is not None and int(event["from"]) < goals
    ]


@metric
def premature_switch(episode: Episode) -> Optional[Fraction]:
    """Advances off a goal not yet achieved when the switch happened — the controller's JUDGEMENT.

    Deliberately not softened by a tolerance window around the switch time. Of 13 measured premature
    switches whose goal did complete later, none completed within 10 steps and the median gap was 38
    on a 200-step episode: a window small enough to mean "nearly right" changes nothing, and one
    wide enough to matter is really crediting a later revisit. Harm is :func:`abandoned_goal`."""
    switches = _switches(episode)
    if not switches or not episode.goal_trace:
        return None
    early = sum(
        1 for goal, step in switches if (first := episode.first_achieved(goal)) is None or first > step
    )
    return float(early), float(len(switches))


@metric
def abandoned_goal(episode: Episode) -> Optional[Fraction]:
    """Advances off a goal the episode never completed at all — the premature switches that actually
    cost something. It ranks systems differently from :func:`premature_switch`: a blind requery clock
    leaves 100% of goals unachieved at the instant it switches but only 38% unfinished by the end,
    while a stepwise planner switches early more often (64%) yet strands far less work (21%)."""
    switches = _switches(episode)
    if not switches or not episode.goal_trace:
        return None
    return float(sum(1 for goal, _ in switches if episode.first_achieved(goal) is None)), float(len(switches))


@metric
def switch_lag(episode: Episode) -> Optional[Fraction]:
    """Mean steps between a goal becoming achieved and the controller moving off it: budget spent
    commanding work already finished. Counts only advances off a goal satisfied AT THE MOMENT of the
    switch, measured from the most recent time it became satisfied.

    Measuring from the FIRST achievement instead conflates this with regression: an oracle episode
    that completed goal 0 at step 18, lost it at 27, regained it at 259 and advanced at 262 scored
    244 rather than 3. That single artifact put the privileged advancer at 15.2 steps against
    plan_gripper's 5.1 — implying the oracle reacts 3x slower than a gripper heuristic. Corrected,
    it reads 4.8 against 4.9: both sit on the same action-chunk polling floor, which is what an
    advancer that switches the instant its goal registers must do."""
    lags: List[int] = []
    for goal, step in _switches(episode):
        achieved_at = episode.achieved_since(goal, step)
        if achieved_at is not None:
            lags.append(step - achieved_at)
    return (float(sum(lags)), float(len(lags))) if lags else None


@metric
def instruction_churn(episode: Episode) -> Optional[Fraction]:
    """Distinct instructions issued, per goal. 1.0 is one instruction per goal; a flat run scores
    below it and a system that re-decides mid-goal scores above. On RoboLab comp_bin_2 under VoLo,
    subgoal issued 2 and scored 6/8 where next_goal issued 16 for the same goals and scored 4/8."""
    goals = episode.goals
    if not goals or not episode.instruction_history:
        return None
    return float(len(set(episode.instruction_history))), float(len(goals))
