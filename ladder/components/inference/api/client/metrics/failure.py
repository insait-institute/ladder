from typing import Optional

from ladder.components.inference.api.client.episode import Episode
from ladder.components.inference.api.client.metrics.registry import Fraction, metric

# WHY an episode failed, from the env's per-goal oracle and the arm's own proprioception. Defined
# over FAILED episodes only, so the shares read as a distribution over what went wrong. Attributing
# a failure to the WRONG OBJECT is absent: no env of ours publishes which object is held.

# Steps without effector motion or a grip change that count as stalled. Absolute rather than a
# fraction of the episode, so it means the same duration on a 60-step and a 1350-step benchmark;
# normally-behaving episodes stay well below it. STALL_SHARE covers the short episode an absolute
# cut cannot fire on (an arm parked for all 60 steps stalls for 59).
STALL_STEPS = 60
STALL_SHARE = 0.9


@metric
def failure_stuck(episode: Episode) -> Optional[Fraction]:
    """Of failed episodes, those where every arm parked: no arm moved for :data:`STALL_STEPS`, or
    for :data:`STALL_SHARE` of a shorter episode."""
    if episode.success or not episode.motion_trace:
        return None
    return float(episode.longest_stall >= min(STALL_STEPS, STALL_SHARE * episode.num_steps)), 1.0


@metric
def failure_no_grasp(episode: Episode) -> Optional[Fraction]:
    """Of failed episodes, those where no arm's gripper ever closed: nothing downstream of the grasp
    is implicated."""
    if episode.success or not episode.motion_trace:
        return None
    return float(not episode.grasps), 1.0


@metric
def failure_dropped(episode: Episode) -> Optional[Fraction]:
    """Of failed episodes, those where an arm was still holding something at the end: out of budget
    mid-carry rather than placing badly."""
    if episode.success or not episode.grasps:
        return None
    return float(episode.holding_at_end), 1.0


@metric
def grasp_churn(episode: Episode) -> Optional[Fraction]:
    """Grasp cycles per goal. ~1 is one pick per goal; far above it the gripper is oscillating,
    which no yes/no metric can see (both read as "grasped, then failed to place")."""
    goals = episode.goals
    if not goals or not episode.motion_trace:
        return None
    return float(len(episode.grasps)), float(len(goals))


# A loss must outlast this to count. Most measured losses last a single step or a few: that is the
# success predicate flickering, not a placement being knocked off.
MIN_LOSS_STEPS = 5


@metric
def recovery_rate(episode: Episode) -> Optional[Fraction]:
    """Goals completed, lost for longer than :data:`MIN_LOSS_STEPS`, and completed AGAIN, over the
    goals so lost: repair a final-state metric cannot see. Undefined when nothing regressed."""
    goals = episode.goals
    if not goals:
        return None
    lost = recovered = 0
    for goal in range(len(goals)):
        held = [(step, goal < len(achieved) and achieved[goal]) for step, achieved in episode.goal_trace]
        for index in range(1, len(held)):
            if not held[index - 1][1] or held[index][1]:
                continue
            regained = next((step for step, on in held[index:] if on), None)
            out = (episode.num_steps if regained is None else regained) - held[index][0]
            if out <= MIN_LOSS_STEPS:
                continue
            lost += 1
            recovered += regained is not None
            break
    return (float(recovered), float(lost)) if lost else None
