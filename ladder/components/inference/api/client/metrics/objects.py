from typing import List, Optional

from ladder.components.inference.api.client import scene_events
from ladder.components.inference.api.client.episode import Episode
from ladder.components.inference.api.client.metrics.registry import Fraction, Metric, metric

# What became of the objects, from the env's scene trace. Undefined for an episode without one.


def _completed(episode: Episode) -> List[scene_events.Grasp]:
    return (
        [grasp for grasp in scene_events.grasps(episode) if grasp.end is not None]
        if episode.held_trace
        else []
    )


def _outcome_share(episode: Episode, outcome: str) -> Optional[Fraction]:
    completed = _completed(episode)
    if not completed:
        return None
    return float(sum(grasp.outcome == outcome for grasp in completed)), float(len(completed))


@metric
def grasp_placed(episode: Episode) -> Optional[Fraction]:
    """Completed grasps whose release credited a goal."""
    return _outcome_share(episode, scene_events.PLACED)


@metric
def grasp_misplaced(episode: Episode) -> Optional[Fraction]:
    """Completed grasps set down on a destination without credit."""
    return _outcome_share(episode, scene_events.MISPLACED)


@metric
def grasp_dropped(episode: Episode) -> Optional[Fraction]:
    """Completed grasps lifted and released anywhere but a destination."""
    return _outcome_share(episode, scene_events.DROPPED)


@metric
def grasp_slipped(episode: Episode) -> Optional[Fraction]:
    """Completed grasps that left the gripper before the object was lifted, over either arm."""
    return _outcome_share(episode, scene_events.SLIPPED)


@metric
def grasp_off_target(episode: Episode) -> Optional[Fraction]:
    """Grasps of an object the current command did not name, over all grasps."""
    all_grasps = scene_events.grasps(episode) if episode.held_trace else []
    if not all_grasps:
        return None
    return float(sum(grasp.intent != scene_events.ON_TARGET for grasp in all_grasps)), float(len(all_grasps))


def _intent(intent: str) -> Metric:
    def share(episode: Episode) -> Optional[Fraction]:
        all_grasps = scene_events.grasps(episode) if episode.held_trace else []
        if not all_grasps:
            return None
        return float(sum(grasp.intent == intent for grasp in all_grasps)), float(len(all_grasps))

    share.__name__ = f"grasp_{intent}"
    share.__doc__ = f"Grasps whose object was {intent!r} relative to the command, over all grasps."
    return metric(share)


for _intent_name in scene_events.OFF_TARGET:
    _intent(_intent_name)


@metric
def misplaced_wrong_destination(episode: Episode) -> Optional[Fraction]:
    """Misplaced grasps of a goal object set down on another goal's destination, over misplaced grasps."""
    own = {
        name: {episode.goal_objects[goal][1] for goal in goals}
        for name, goals in scene_events.sources(episode).items()
    }
    misplaced = [
        grasp
        for grasp in _completed(episode)
        if grasp.outcome == scene_events.MISPLACED and grasp.object in own
    ]
    if not misplaced:
        return None
    return float(sum(grasp.resting_on not in own[grasp.object] for grasp in misplaced)), float(len(misplaced))


@metric
def grasp_missed(episode: Episode) -> Optional[Fraction]:
    """Grasp attempts within reach of an object that held nothing, over all attempts."""
    if not episode.held_trace or not episode.motion_trace:
        return None
    missed, attempts = scene_events.missed_grasps(episode)
    return (float(missed), float(attempts)) if attempts else None


@metric
def goal_knocked(episode: Episode) -> Optional[Fraction]:
    """Goals ever achieved that were undone by a knock while nothing held their object."""
    if not episode.object_trace or not episode.goal_trace:
        return None
    ever = sum(
        any(goal < len(achieved) and achieved[goal] for _, achieved in episode.goal_trace)
        for goal in range(len(episode.goals))
    )
    if not ever:
        return None
    knocked = {
        event.regressed_goal
        for event in scene_events.displacements(episode)
        if event.regressed_goal is not None
    }
    return float(len(knocked)), float(ever)


def _loss(mechanism: str) -> Metric:
    def share(episode: Episode) -> Optional[Fraction]:
        losses = scene_events.goal_losses(episode) if episode.object_trace else []
        if not losses:
            return None
        found = sum(scene_events.loss_mechanism(episode, goal, step) == mechanism for goal, step in losses)
        return float(found), float(len(losses))

    share.__name__ = f"lost_{mechanism}"
    share.__doc__ = f"Lasting goal losses whose mechanism is {mechanism!r}, over all losses."
    return metric(share)


for _loss_name in scene_events.LOSSES:
    _loss(_loss_name)


@metric
def object_fell(episode: Episode) -> Optional[Fraction]:
    """Episodes in which some object ended well below its rest pose."""
    if not episode.object_trace:
        return None
    return float(bool(scene_events.fallen(episode))), 1.0


def _controller_share(episode: Episode, numerator: str, denominator: str) -> Optional[Fraction]:
    if not episode.object_trace or not episode.events:
        return None
    counts = scene_events.controller_events(episode)
    return (float(counts[numerator]), float(counts[denominator])) if counts[denominator] else None


@metric
def advance_carrying(episode: Episode) -> Optional[Fraction]:
    """Subtask advances made while an arm still held an object, over all advances."""
    return _controller_share(episode, "advances_carrying", "advances")


@metric
def advance_to_done(episode: Episode) -> Optional[Fraction]:
    """Advances onto a subtask whose goals were already satisfied, over all advances."""
    return _controller_share(episode, "advances_to_done", "advances")


@metric
def verdict_off_destination(episode: Episode) -> Optional[Fraction]:
    """Verifier done-verdicts given while the object was not resting on its destination, over done-verdicts."""
    return _controller_share(episode, "verdicts_done_off_destination", "verdicts_done")


def _cause(cause: str) -> Metric:
    def share(episode: Episode) -> Optional[Fraction]:
        found = scene_events.failure_cause(episode)
        return None if found is None else (float(found == cause), 1.0)

    share.__name__ = f"cause_{cause}"
    share.__doc__ = f"Failed episodes whose cause is {cause!r}."
    return metric(share)


for _cause_name in scene_events.CAUSES:
    _cause(_cause_name)
