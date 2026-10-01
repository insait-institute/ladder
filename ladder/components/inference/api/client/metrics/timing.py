from typing import Optional

from ladder.components.inference.api.client import scene_events
from ladder.components.inference.api.client.episode import Episode
from ladder.components.inference.api.client.metrics.registry import Fraction, metric

# Where the budget went. Steps are absolute, so compare within one benchmark.


@metric
def success_steps(episode: Episode) -> Optional[Fraction]:
    """Mean steps a successful episode took."""
    return (float(episode.num_steps), 1.0) if episode.success else None


@metric
def first_grasp_steps(episode: Episode) -> Optional[Fraction]:
    """Mean steps until the first object was held."""
    holds = scene_events.holds(episode) if episode.held_trace else []
    return (float(min(start for _, start, _ in holds)), 1.0) if holds else None


@metric
def steps_per_goal(episode: Episode) -> Optional[Fraction]:
    """Steps until the last newly achieved goal, per goal ever achieved."""
    firsts = [
        first for goal in range(len(episode.goals)) if (first := episode.first_achieved(goal)) is not None
    ]
    return (float(max(firsts)), float(len(firsts))) if firsts else None


@metric
def idle_tail(episode: Episode) -> Optional[Fraction]:
    """Of a failed episode, the share after the last hold change, object move or goal flip."""
    if episode.success or not episode.object_trace or not episode.num_steps:
        return None
    return float(episode.num_steps - scene_events.last_activity(episode)), float(episode.num_steps)


@metric
def dead_time(episode: Episode) -> Optional[Fraction]:
    """Share of the episode spent commanding a subtask whose goals were already satisfied."""
    if not episode.num_steps or not episode.goal_trace:
        return None
    dead = scene_events.dead_steps(episode)
    return None if dead is None else (float(dead), float(episode.num_steps))
