from typing import Optional

from ladder.components.inference.api.client.episode import Episode
from ladder.components.inference.api.client.metrics.registry import Fraction, metric

# What the episode achieved, read off the env's own success and per-goal state. Nothing here parses
# language or inspects the controller, so these apply to any benchmark that publishes the oracle
# block, and to a flat run as readily as a hierarchical one.


@metric
def success(episode: Episode) -> Optional[Fraction]:
    """The goal state was reached at some step."""
    return float(episode.success), 1.0


@metric
def success_terminal(episode: Episode) -> Optional[Fraction]:
    """The goal state held at the last step. Stricter than :func:`success` on composition tasks,
    where a placement can be knocked off after it counted."""
    return float(episode.success_terminal), 1.0


@metric
def goal_progress(episode: Episode) -> Optional[Fraction]:
    """Goals satisfied at the end, over the task's goal count. Normalised per episode so tasks with
    different goal counts contribute equally to an average."""
    goals = episode.goals
    return (float(sum(goals)), float(len(goals))) if goals else None


@metric
def goal_progress_best(episode: Episode) -> Optional[Fraction]:
    """Most goals held at once at any step: progress ignoring later damage."""
    goals = episode.goals
    return (float(episode.goals_best), float(len(goals))) if goals else None


@metric
def goal_regression(episode: Episode) -> Optional[Fraction]:
    """Goals completed and then undone. Work the policy did and destroyed."""
    goals = episode.goals
    return (float(episode.goals_best - sum(goals)), float(len(goals))) if goals else None


@metric
def any_goal(episode: Episode) -> Optional[Fraction]:
    """Episodes that completed at least one goal: separates "did nothing" from "stalled late"."""
    if not episode.goal_trace:
        return None
    return float(any(any(achieved) for _, achieved in episode.goal_trace)), 1.0


@metric
def order_violation(episode: Episode) -> Optional[Fraction]:
    """Of the episodes that completed every goal, the share the env still failed: an ordered task
    done in the wrong sequence. Measured on robot_arena_scene76_2goal_ordered, where goal 1 landed at
    step 18 and goal 0 at step 200: goal_progress reads 2/2 and success 0, with nothing in between to
    say why. Only defined once all goals are in, so unordered scenes report nothing."""
    goals = episode.goals
    if not goals or not all(goals):
        return None
    return float(not episode.success_terminal), 1.0
