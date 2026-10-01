import dataclasses
from typing import Dict, List, Optional, Sequence, Set, Tuple, TypeVar

from ladder.components.inference.api.client.episode import MIN_GRIPPER_TRAVEL, Episode

# Object-level events derived from an Episode's scene trace, one definition for all three simulators.
# Distances are in metres against each object's rest pose (its pose once the scene settled after reset).
LIFT_M = 0.03
DISPLACED_M = 0.03
FELL_M = 0.05
FOOTPRINT_MARGIN_M = 0.02
REACH_M = 0.08
# Vertical slack for a grasp attempt: the logged effector point is the flange on the Franka.
REACH_Z_M = {"franka": 0.25}
REACH_Z_DEFAULT_M = 0.06
OPEN_FRACTION = 0.85
SETTLE_STEPS = 10
REST_GAP_STEPS = 15
GRASP_GAP_STEPS = 8  # contact-based grasp flags flicker mid-carry
MIN_GOAL_HOLD_STEPS = 5  # RoboLab's on-surface predicate flickers while an object lies still

T = TypeVar("T")

PLACED, MISPLACED, DROPPED, SLIPPED, HELD = "placed", "misplaced", "dropped", "slipped", "held"
OUTCOMES = (PLACED, MISPLACED, DROPPED, SLIPPED, HELD)
ON_TARGET, OTHER_GOAL, DONE_GOAL, DESTINATION, DISTRACTOR = (
    "on_target",
    "other_goal",
    "done_goal",
    "destination",
    "distractor",
)
OFF_TARGET = (OTHER_GOAL, DONE_GOAL, DESTINATION, DISTRACTOR)
ORDER, KNOCKED, UNDONE = "order", "knocked", "undone"
REGRASP, DRIFT = "regrasp", "drift"
LOSSES = (REGRASP, KNOCKED, DRIFT)
NO_GRASP, WRONG_OBJECT, NOT_COMMANDED, GRASP_FAILED = (
    "no_grasp",
    "wrong_object",
    "not_commanded",
    "grasp_failed",
)
SLIP, DROP, MISPLACE, CARRYING = "slip", "drop", "misplace", "carrying"
CAUSES = (
    ORDER,
    KNOCKED,
    UNDONE,
    NO_GRASP,
    WRONG_OBJECT,
    NOT_COMMANDED,
    GRASP_FAILED,
    SLIP,
    DROP,
    MISPLACE,
    CARRYING,
)


@dataclasses.dataclass(frozen=True)
class Grasp:
    object: str
    start: int
    end: Optional[int]  # None while still held at the end
    outcome: str  # PLACED | MISPLACED | DROPPED | SLIPPED | HELD
    resting_on: Optional[str] = None
    intent: str = ON_TARGET  # how the object relates to the command at ``start``


@dataclasses.dataclass(frozen=True)
class Displacement:
    """An object that moved while nothing held it."""

    object: str
    start: int  # last step at rest
    moved_at: int
    end: int
    distance_m: float
    holding: List[str]
    regressed_goal: Optional[int]
    fell: bool


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5)


def _at(rows: Sequence[Tuple[int, T]], step: int) -> Optional[T]:
    value = None
    for at, current in rows:
        if at > step:
            break
        value = current
    return value


def position_at(episode: Episode, name: str, step: int) -> Optional[List[float]]:
    return _at(episode.object_trace.get(name, []), step)


def rest_position(episode: Episode, name: str) -> Optional[List[float]]:
    rows = episode.object_trace.get(name)
    if not rows:
        return None
    first_held = next((step for step, held in episode.held_trace if name in held), None)
    limit = SETTLE_STEPS if first_held is None else min(SETTLE_STEPS, first_held - 1)
    return list(_at(rows, limit) or rows[0][1])


def held_at(episode: Episode, step: int) -> List[str]:
    return list(_at(episode.held_trace, step) or [])


def achieved_at(episode: Episode, step: int) -> List[bool]:
    return list(_at(episode.goal_trace, step) or [False] * len(episode.goals))


def sources(episode: Episode) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for goal, (source, *_) in enumerate(episode.goal_objects):
        if source:
            out.setdefault(source, []).append(goal)
    return out


def destinations(episode: Episode) -> Set[str]:
    return {dest for _, dest, _ in episode.goal_objects if dest}


def resting_on(episode: Episode, name: str, step: int) -> Optional[str]:
    position = position_at(episode, name, step)
    if position is None:
        return None
    for dest in sorted(destinations(episode)):
        dest_position, half = position_at(episode, dest, step), episode.object_extents.get(dest)
        if dest == name or dest_position is None or half is None:
            continue
        inside_xy = all(
            abs(position[axis] - dest_position[axis]) <= half[axis] + FOOTPRINT_MARGIN_M for axis in (0, 1)
        )
        if inside_xy and position[2] >= dest_position[2] - half[2] - FOOTPRINT_MARGIN_M:
            return dest
    return None


def _credited_near(episode: Episode, goal: int, step: int) -> bool:
    after = achieved_at(episode, step + SETTLE_STEPS)
    if goal < len(after) and after[goal]:
        return True
    return any(
        step - SETTLE_STEPS <= at <= step + SETTLE_STEPS and goal < len(achieved) and achieved[goal]
        for at, achieved in episode.goal_trace
    )


def _plan_steps(episode: Episode) -> List[str]:
    if episode.plan:
        return list(episode.plan)
    plans = episode.of_kind("plan")
    return [str(subtask) for subtask in plans[-1].get("subtasks", [])] if plans else []


def goals_of_step(episode: Episode, index: int) -> Optional[Set[int]]:
    """Reference goals plan step ``index`` names, by head-noun match; None when unresolvable."""
    from ladder.components.inference.api.client.metrics.plan import goal_of  # avoids a package cycle

    plan = _plan_steps(episode)
    if not plan or not episode.subtasks or not episode.goal_objects or index >= len(plan):
        return None
    wanted = goal_of(plan[index])
    if wanted is None:
        return None
    matched = {
        goal
        for goal, reference in enumerate(episode.subtasks)
        if goal < len(episode.goal_objects) and goal_of(reference) == wanted and episode.goal_objects[goal][0]
    }
    return matched or None


def commanded_objects_at(episode: Episode, step: int) -> Optional[Set[str]]:
    """Objects the controller's command at ``step`` names; None for a flat run or an unresolvable step."""
    index = 0
    for event in sorted(episode.of_kind("advance"), key=lambda advance: int(advance.get("step", 0))):
        if int(event.get("step", 0)) <= step and event.get("subtask") is not None:
            index = int(event["subtask"])
    goals = goals_of_step(episode, index)
    return {episode.goal_objects[goal][0] for goal in goals} if goals else None


def _intent(episode: Episode, name: str, step: int) -> str:
    goals = sources(episode).get(name, [])
    if not goals:
        return DESTINATION if name in destinations(episode) else DISTRACTOR
    achieved = achieved_at(episode, step)
    pending = [goal for goal in goals if goal >= len(achieved) or not achieved[goal]]
    commanded = commanded_objects_at(episode, step)
    if commanded is None:
        return ON_TARGET if pending else DONE_GOAL
    if name in commanded:
        return ON_TARGET
    return OTHER_GOAL if pending else DONE_GOAL


def _lifted_between(episode: Episode, name: str, start: int, end: int) -> bool:
    rest = rest_position(episode, name)
    if rest is None:
        return False
    return any(
        start <= at <= end and center[2] >= rest[2] + LIFT_M
        for at, center in episode.object_trace.get(name, [])
    )


def _close(episode: Episode, name: str, start: int, end: int) -> Grasp:
    intent = _intent(episode, name, start)
    for goal in sources(episode).get(name, []):
        if _credited_near(episode, goal, end):
            return Grasp(name, start, end, PLACED, episode.goal_objects[goal][1], intent)
    on = resting_on(episode, name, end + SETTLE_STEPS)
    if on is not None and name not in held_at(episode, end + SETTLE_STEPS):
        return Grasp(name, start, end, MISPLACED, on, intent)
    return Grasp(
        name, start, end, DROPPED if _lifted_between(episode, name, start, end) else SLIPPED, None, intent
    )


def _stayed_open(rows: Sequence[Tuple[int, List[float]]], start: int, until: int) -> Optional[bool]:
    """Whether one arm's gripper stayed open over ``[start, until]``, against that arm's own travel,
    or None when the arm says nothing."""
    openness = [gripper for _, (*_, gripper) in rows]
    if not openness or max(openness) - min(openness) < MIN_GRIPPER_TRAVEL:
        return None
    open_above = min(openness) + OPEN_FRACTION * (max(openness) - min(openness))
    readings = [row[3] for at, row in rows if start <= at <= until]
    before = _at(rows, start)
    if before is not None:
        readings.append(before[3])
    return all(reading >= open_above for reading in readings) if readings else None


def _is_brush(episode: Episode, name: str, start: int, end: Optional[int]) -> bool:
    """A contact that never lifted the object while some arm's gripper stayed open through it, or
    while the robot held something else."""
    until = episode.num_steps if end is None else end
    if _lifted_between(episode, name, start, until):
        return False
    if len(held_at(episode, start)) > 1:
        return True
    open_per_arm = [_stayed_open(episode.motion_trace[arm], start, until) for arm in episode.arms]
    return any(stayed_open is True for stayed_open in open_per_arm) and not any(
        stayed_open is False for stayed_open in open_per_arm
    )


def holds(episode: Episode) -> List[Tuple[str, int, Optional[int]]]:
    """(object, start, end) per hold, brushes dropped and holds closer than GRASP_GAP_STEPS merged."""
    open_since: Dict[str, int] = {}
    raw: List[Tuple[str, int, Optional[int]]] = []
    for step, held in episode.held_trace:
        for name in [held_name for held_name in open_since if held_name not in held]:
            raw.append((name, open_since.pop(name), step))
        for name in held:
            open_since.setdefault(name, step)
    raw.extend((name, start, None) for name, start in open_since.items())
    merged: List[Tuple[str, int, Optional[int]]] = []
    for name, start, end in sorted(raw, key=lambda hold: (hold[1], hold[0])):
        if _is_brush(episode, name, start, end):
            continue
        previous = next((hold for hold in reversed(merged) if hold[0] == name), None)
        if previous and previous[2] is not None and start - previous[2] <= GRASP_GAP_STEPS:
            merged[merged.index(previous)] = (name, previous[1], end)
        else:
            merged.append((name, start, end))
    return merged


def grasps(episode: Episode) -> List[Grasp]:
    out = [
        _close(episode, name, start, end)
        if end is not None
        else Grasp(name, start, None, HELD, None, _intent(episode, name, start))
        for name, start, end in holds(episode)
    ]
    return sorted(out, key=lambda grasp: (grasp.start, grasp.object))


def missed_grasps(episode: Episode) -> Tuple[int, int]:
    """(attempts that held nothing, attempts): gripper closures within reach of some object, summed
    over the arms. Reach is measured from the arm that closed."""
    if not episode.held_trace or not episode.object_trace:
        return 0, 0
    missed = attempts = 0
    reach_z = REACH_Z_M.get(episode.robot_name, REACH_Z_DEFAULT_M)
    for arm in episode.arms:
        rows = episode.motion_trace[arm]
        for closed_at, released_at in episode.grasp_cycles(arm):
            eef = _at(rows, closed_at)
            if eef is None:
                continue
            near = any(
                (position := position_at(episode, name, closed_at)) is not None
                and _distance(eef[:2], position[:2]) <= REACH_M
                and abs(eef[2] - position[2]) <= reach_z
                for name in episode.object_trace
            )
            if not near:
                continue
            attempts += 1
            until = released_at if released_at is not None else episode.num_steps
            still_held = any(closed_at <= at <= until and held for at, held in episode.held_trace)
            if not still_held and not held_at(episode, closed_at):
                missed += 1
    return missed, attempts


def displacements(episode: Episode) -> List[Displacement]:
    """Rest-to-rest movements of at least DISPLACED_M while nothing held the object, settling excluded."""
    releases = {(grasp.object, grasp.end) for grasp in grasps(episode) if grasp.end is not None}
    out: List[Displacement] = []
    for name, rows in episode.object_trace.items():
        rest = rest_position(episode, name) or rows[0][1]
        segment: List[Tuple[int, List[float]]] = []
        for previous, current in zip(rows, rows[1:], strict=False):
            free = name not in held_at(episode, current[0]) and name not in held_at(episode, previous[0])
            if free and segment and current[0] - segment[-1][0] <= REST_GAP_STEPS:
                segment.append(current)
                continue
            if segment:
                out.extend(_displacement(episode, name, segment, rest, releases))
            segment = [previous, current] if free else []
        if segment:
            out.extend(_displacement(episode, name, segment, rest, releases))
    return sorted(out, key=lambda event: (event.start, event.object))


def _displacement(
    episode: Episode,
    name: str,
    segment: List[Tuple[int, List[float]]],
    rest: Sequence[float],
    releases: Set[Tuple[str, int]],
) -> List[Displacement]:
    start, moved_at, end = segment[0][0], segment[1][0], segment[-1][0]
    distance = _distance(segment[0][1], segment[-1][1])
    fell = segment[-1][1][2] < rest[2] - FELL_M
    settling = moved_at <= SETTLE_STEPS or any(
        obj == name and 0 <= moved_at - released <= SETTLE_STEPS for obj, released in releases
    )
    if distance < DISPLACED_M or (settling and not fell):
        return []
    before = [
        any(flags) for flags in zip(achieved_at(episode, start), achieved_at(episode, moved_at), strict=False)
    ]
    after = achieved_at(episode, end + SETTLE_STEPS)
    regressed = next(
        (
            goal
            for goal in sources(episode).get(name, [])
            if goal < len(before) and before[goal] and (goal >= len(after) or not after[goal])
        ),
        None,
    )
    return [
        Displacement(
            name, start, moved_at, end, round(distance, 4), held_at(episode, moved_at), regressed, fell
        )
    ]


def fallen(episode: Episode) -> List[str]:
    out = []
    for name, rows in episode.object_trace.items():
        rest = rest_position(episode, name)
        if rest is not None and rows[-1][1][2] < rest[2] - FELL_M:
            out.append(name)
    return sorted(out)


def failure_cause(episode: Episode) -> Optional[str]:
    """One of CAUSES for a failed composition episode with a scene trace, else None."""
    if episode.success or not episode.goal_objects or not episode.goal_trace or not episode.object_trace:
        return None
    n_goals = len(episode.goal_objects)
    ever = [
        any(goal < len(achieved) and achieved[goal] for _, achieved in episode.goal_trace)
        for goal in range(n_goals)
    ]
    all_grasps = grasps(episode)
    if all(ever):
        if episode.goals_best == n_goals:
            return ORDER
        return (
            KNOCKED if any(event.regressed_goal is not None for event in displacements(episode)) else UNDONE
        )
    missing = {episode.goal_objects[goal][0] for goal in range(n_goals) if not ever[goal]}
    if not all_grasps:
        return NO_GRASP
    if not any(grasp.object in missing for grasp in all_grasps):
        if all(grasp.intent in (DISTRACTOR, DESTINATION, DONE_GOAL) for grasp in all_grasps):
            return WRONG_OBJECT
        if commanded_objects_at(episode, 0) is None:
            return GRASP_FAILED
        steps = {0} | {
            int(event["step"]) for event in episode.of_kind("advance") if event.get("step") is not None
        }
        commanded = {name for step in steps for name in (commanded_objects_at(episode, step) or set())}
        return GRASP_FAILED if commanded & missing else NOT_COMMANDED
    last = [grasp for grasp in all_grasps if grasp.object in missing][-1]
    return {HELD: CARRYING, DROPPED: DROP, MISPLACED: MISPLACE, SLIPPED: SLIP}.get(last.outcome, SLIP)


def goal_flips(episode: Episode, goal: int) -> List[Tuple[int, bool]]:
    """(step, achieved) per lasting state change, dated from the first step the new state appeared."""
    raw = [(step, goal < len(achieved) and achieved[goal]) for step, achieved in episode.goal_trace]
    out: List[Tuple[int, bool]] = []
    confirmed, candidate = False, None
    for index, (step, flag) in enumerate(raw):
        until = raw[index + 1][0] if index + 1 < len(raw) else episode.num_steps
        long_enough = until - step >= MIN_GOAL_HOLD_STEPS or index + 1 == len(raw)
        if flag == confirmed:
            if long_enough:
                candidate = None
            continue
        if candidate is None:
            candidate = step
        if long_enough:
            out.append((candidate, flag))
            confirmed, candidate = flag, None
    return out


def goal_losses(episode: Episode) -> List[Tuple[int, int]]:
    """(goal, step) per lasting loss of an achieved goal."""
    return sorted(
        (goal, step)
        for goal in range(len(episode.goals))
        for step, achieved in goal_flips(episode, goal)
        if not achieved
    )


def loss_mechanism(episode: Episode, goal: int, step: int) -> str:
    """REGRASP when an arm took the object back, KNOCKED when it moved untouched, else DRIFT."""
    name = episode.goal_objects[goal][0] if goal < len(episode.goal_objects) else None
    if name is None:
        return DRIFT
    window = range(step - GRASP_GAP_STEPS, step + SETTLE_STEPS + 1)
    if name in held_at(episode, window.start) or any(
        name in held for at, held in episode.held_trace if at in window
    ):
        return REGRASP
    if any(
        event.object == name and event.moved_at - MIN_GOAL_HOLD_STEPS <= step <= event.end + SETTLE_STEPS
        for event in displacements(episode)
    ):
        return KNOCKED
    return DRIFT


def last_activity(episode: Episode) -> int:
    """Last step at which the hold changed, an object moved or a goal flipped."""
    steps = [step for step, _ in episode.held_trace]
    steps += [step for rows in episode.object_trace.values() for step, _ in rows]
    steps += [step for goal in range(len(episode.goals)) for step, _ in goal_flips(episode, goal)]
    return max(steps, default=0)


def dead_steps(episode: Episode) -> Optional[int]:
    """Steps spent commanding a subtask whose goals were already satisfied; None for a flat run."""
    if commanded_objects_at(episode, 0) is None:
        return None
    segments = [(0, 0)] + sorted(
        (int(event["step"]), int(event["subtask"]))
        for event in episode.of_kind("advance")
        if event.get("step") is not None and event.get("subtask") is not None
    )
    dead = 0
    for index, (start, subtask) in enumerate(segments):
        stop = segments[index + 1][0] if index + 1 < len(segments) else episode.num_steps
        goals = goals_of_step(episode, subtask)
        if not goals:
            continue
        marks = [(start, achieved_at(episode, start))] + [
            (at, achieved) for at, achieved in episode.goal_trace if start < at < stop
        ]
        for mark, (at, achieved) in enumerate(marks):
            until = marks[mark + 1][0] if mark + 1 < len(marks) else stop
            if all(goal < len(achieved) and achieved[goal] for goal in goals):
                dead += until - at
    return dead


def controller_events(episode: Episode) -> Dict[str, int]:
    """Advances (total, mid-carry, onto finished goals) and done-verdicts (total, object off its destination)."""
    keys = (
        "advances",
        "advances_carrying",
        "advances_to_done",
        "verdicts_done",
        "verdicts_done_off_destination",
    )
    counts = {key: 0 for key in keys}
    if not episode.object_trace:
        return counts
    for event in episode.of_kind("advance"):
        step = int(event.get("step", 0))
        counts["advances"] += 1
        counts["advances_carrying"] += bool(held_at(episode, step))
        goals = goals_of_step(episode, int(event["subtask"])) if event.get("subtask") is not None else None
        achieved = achieved_at(episode, step)
        if goals and all(goal < len(achieved) and achieved[goal] for goal in goals):
            counts["advances_to_done"] += 1
    for event in episode.of_kind("verify"):
        if not event.get("verdict") or event.get("subtask") is None:
            continue
        counts["verdicts_done"] += 1
        step, goals = int(event.get("step", 0)), goals_of_step(episode, int(event["subtask"]))
        if not goals:
            continue
        on_destination = any(
            resting_on(episode, episode.goal_objects[goal][0], step) == episode.goal_objects[goal][1]
            for goal in goals
            if episode.goal_objects[goal][1]
        )
        counts["verdicts_done_off_destination"] += not on_destination
    return counts


def timeline(episode: Episode) -> List[Tuple[int, str]]:
    """Controller, goal and object events of one episode in step order."""
    lines: List[Tuple[int, str]] = []
    for event in episode.events:
        step, kind = int(event.get("step", 0)), event.get("kind")
        if kind == "plan":
            lines.append((step, "plan " + " | ".join(str(subtask) for subtask in event.get("subtasks", []))))
        elif kind == "advance":
            lines.append(
                (
                    step,
                    f"advance {event.get('from')} -> {event.get('subtask')} (goal held: {event.get('gt')})",
                )
            )
        elif kind == "verify":
            lines.append(
                (
                    step,
                    f"verify subtask {event.get('subtask')}: {event.get('verdict')} (truth {event.get('gt')})",
                )
            )
    for goal in range(len(episode.goals)):
        what = episode.goal_objects[goal][0] if goal < len(episode.goal_objects) else "?"
        for step, flag in goal_flips(episode, goal):
            lines.append((step, f"goal {goal} {'achieved' if flag else 'LOST'} ({what})"))
    for grasp in grasps(episode):
        tag = "" if grasp.intent == ON_TARGET else f" [{grasp.intent}]"
        lines.append((grasp.start, f"grasp {grasp.object}{tag}"))
        if grasp.end is not None:
            where = f" on {grasp.resting_on}" if grasp.resting_on else ""
            lines.append((grasp.end, f"release {grasp.object}: {grasp.outcome}{where}"))
    for event in displacements(episode):
        cause = f" while holding {', '.join(event.holding)}" if event.holding else ""
        regressed = f", LOST goal {event.regressed_goal}" if event.regressed_goal is not None else ""
        fell = ", FELL" if event.fell else ""
        lines.append(
            (event.moved_at, f"knocked {event.object} {event.distance_m:.2f} m{cause}{regressed}{fell}")
        )
    return sorted(lines, key=lambda line: line[0])
