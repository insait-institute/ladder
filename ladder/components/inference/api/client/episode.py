import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ladder.components.inference.api.core.observation import EFFECTOR_NAMES, RobotState, named_effector_states
from ladder.components.inference.api.core.privileged_state import METADATA_PRIVILEGED_KEY

# A grasp is read against the openness range the episode itself spans, not a fixed cut: RoboLab
# bottoms out at 0.407 and never reports a closed gripper, so a fixed 0.5 finds no grasp there at
# all. Below MIN_GRIPPER_TRAVEL of spread the gripper never actuated and there is nothing to find.
GRIPPER_CLOSED_FRACTION = 0.5
MIN_GRIPPER_TRAVEL = 0.2
# Effector travel / openness change worth a motion_trace row.
MOTION_EPSILON_M = 0.005
GRIPPER_EPSILON = 0.05
# Object travel worth an object_trace row; a resting body jitters at the millimetre level.
OBJECT_EPSILON_M = 0.01


def record_motion(trace: Dict[str, List[Tuple[int, List[float]]]], step: int, state: RobotState) -> None:
    """Append a change point per arm, keyed by arm name, or nothing for an arm that has not moved
    since its own last one. Each arm is traced independently, so an arm parked while the other works
    writes nothing and its stall stays visible as a gap in its own rows."""
    for arm, effector in named_effector_states(state):
        if effector.eef_pose is None or effector.gripper_state is None:
            continue
        row = [*(round(float(x), 4) for x in effector.eef_pose[:3, 3])]
        row.append(round(float(np.ravel(effector.gripper_state)[0]), 4))
        rows = trace.setdefault(arm, [])
        if rows:
            previous = rows[-1][1]
            moved = max(abs(now - was) for now, was in zip(row[:3], previous[:3], strict=False))
            if moved < MOTION_EPSILON_M and abs(row[3] - previous[3]) < GRIPPER_EPSILON:
                continue
        rows.append((step, row))


@dataclasses.dataclass
class SceneTrace:
    """Ground-truth object state over one episode, from the ``privileged`` payload, as change points."""

    goal_objects: List[List[Optional[str]]] = dataclasses.field(
        default_factory=list
    )  # [object, destination, relation]
    object_extents: Dict[str, List[float]] = dataclasses.field(
        default_factory=dict
    )  # half extents per object
    object_trace: Dict[str, List[Tuple[int, List[float]]]] = dataclasses.field(
        default_factory=dict
    )  # (step, centre)
    held_trace: List[Tuple[int, List[str]]] = dataclasses.field(
        default_factory=list
    )  # (step, grasped objects)

    def record(self, step: int, metadata: Optional[Dict[str, Any]]) -> None:
        payload = (metadata or {}).get(METADATA_PRIVILEGED_KEY)
        if not payload:
            return
        if not self.goal_objects:
            self.goal_objects = [
                [str(condition["target"]["name"]), condition.get("destination"), condition.get("relation")]
                for condition in payload.get("conditions", [])
            ]
        held: List[str] = []
        for entry in payload.get("objects", []):
            name = str(entry["name"])
            center = [round(float(x), 4) for x in np.ravel(entry["center_m"])]
            if name not in self.object_extents:
                half = 0.5 * (np.ravel(entry["aabb_max_m"]) - np.ravel(entry["aabb_min_m"]))
                self.object_extents[name] = [round(float(x), 4) for x in half]
            rows = self.object_trace.setdefault(name, [])
            if (
                not rows
                or max(abs(now - was) for now, was in zip(center, rows[-1][1], strict=True))
                >= OBJECT_EPSILON_M
            ):
                rows.append((step, center))
            if entry.get("grasped"):
                held.append(name)
        held.sort()
        if not self.held_trace or self.held_trace[-1][1] != held:
            self.held_trace.append((step, held))


@dataclasses.dataclass
class Episode:
    """One evaluation episode: the unit every metric is computed from, and the schema of one
    ``episode_log.yaml`` entry.

    The composition fields are populated for any env that publishes the per-goal ``oracle`` block:
    RobotArena's ``ConjunctionSuccessChecker``, SimplerEnv's ``oracle_goal_keys``, RoboLab's
    task-spec checks. That block is the one per-goal signal all three benchmarks already agree on,
    so a metric written against it works on all three without a per-benchmark branch.
    """

    env_name: str
    robot_name: str
    instruction: str
    batch: int
    num_steps: int = 0
    success: bool = False  # the goal state was reached at some step
    success_terminal: bool = False  # the last recorded step was in the goal state
    success_per_step: List[int] = dataclasses.field(default_factory=list)
    instruction_history: List[str] = dataclasses.field(default_factory=list)
    batch_env_index: Optional[int] = None  # None when the batch held a single env

    # Composition. ``goal_trace`` holds CHANGE POINTS, [(step, [achieved, ...]), ...]: the vector is
    # near-constant, so a 1350-step RoboLab episode stores a handful of rows rather than one per
    # step, and nothing is lost: the value between two change points is the earlier one.
    # ``subtasks`` is the ENVIRONMENT's reference decomposition, published by the oracle block and
    # therefore identical whatever system ran, so a flat episode carries it too. ``plan`` is what the
    # controller decided to do, so the two together are what makes a plan scorable: the reference
    # says which goals exist, the plan says which the controller set out to reach. Empty for a flat
    # run and for a stepwise system, neither of which forms a plan up front.
    subtasks: List[str] = dataclasses.field(default_factory=list)
    plan: List[str] = dataclasses.field(default_factory=list)
    goal_trace: List[Tuple[int, List[bool]]] = dataclasses.field(default_factory=list)

    # Hierarchical control: verifier verdicts and subtask advances, each stamped with its step.
    events: List[Dict[str, Any]] = dataclasses.field(default_factory=list)

    # Proprioception per arm, as change points like ``goal_trace``:
    # {"right": [(step, [x, y, z, gripper]), ...], "left": [...]}. An arm that is not moving writes
    # NOTHING, so a stall is a gap between its rows rather than a per-step dump, and a single-arm
    # episode traces one "right" arm.
    motion_trace: Dict[str, List[Tuple[int, List[float]]]] = dataclasses.field(default_factory=dict)

    # Simulator-native trajectory metrics, when the env server publishes them (RoboLab).
    trajectory: Dict[str, float] = dataclasses.field(default_factory=dict)

    # The :class:`SceneTrace` fields; empty when the env published no privileged payload.
    goal_objects: List[List[Optional[str]]] = dataclasses.field(default_factory=list)
    object_extents: Dict[str, List[float]] = dataclasses.field(default_factory=dict)
    object_trace: Dict[str, List[Tuple[int, List[float]]]] = dataclasses.field(default_factory=dict)
    held_trace: List[Tuple[int, List[str]]] = dataclasses.field(default_factory=list)

    @property
    def goals(self) -> List[bool]:
        """Per-goal achievement at the last observed step; empty for a non-composition env."""
        return list(self.goal_trace[-1][1]) if self.goal_trace else []

    @property
    def goals_best(self) -> int:
        """Most goals held simultaneously at any step. Above :attr:`goals` when a completed goal was
        later undone, e.g. a placement knocked off by the next one."""
        return max((sum(achieved) for _, achieved in self.goal_trace), default=0)

    def first_achieved(self, goal: int) -> Optional[int]:
        """Step at which ``goal`` was first achieved, or None if it never was."""
        for step, achieved in self.goal_trace:
            if goal < len(achieved) and achieved[goal]:
                return step
        return None

    def achieved_since(self, goal: int, step: int) -> Optional[int]:
        """Step at which ``goal`` most recently BECAME achieved and still held at ``step``; None if it
        was not achieved at ``step`` at all.

        Distinct from :meth:`first_achieved`, and the difference is not academic: a goal can be
        completed, knocked off, and completed again. Measuring from the FIRST achievement would count
        the time spent re-achieving it as budget wasted on finished work.
        """
        since = None
        for at, achieved in self.goal_trace:
            if at > step:
                break
            held = goal < len(achieved) and achieved[goal]
            since = at if (held and since is None) else (since if held else None)
        return since

    def of_kind(self, kind: str) -> List[Dict[str, Any]]:
        """The controller events of one kind (``verify``, ``advance``)."""
        return [event for event in self.events if event.get("kind") == kind]

    @property
    def arms(self) -> List[str]:
        """The arms this episode traced, in effector order."""
        return [arm for arm in EFFECTOR_NAMES if arm in self.motion_trace]

    def grasp_cycles(self, arm: str) -> List[Tuple[int, Optional[int]]]:
        """(closed_at, released_at) per grasp ``arm`` completed; ``released_at`` is None for one
        still held at the end. The openness range is that arm's own: two arms on the same robot can
        travel differently, and one that never actuated must not borrow the other's range."""
        rows = self.motion_trace.get(arm, [])
        openness = [gripper for _, (*_, gripper) in rows]
        if not openness or max(openness) - min(openness) < MIN_GRIPPER_TRAVEL:
            return []
        closed_below = min(openness) + GRIPPER_CLOSED_FRACTION * (max(openness) - min(openness))
        cycles: List[Tuple[int, Optional[int]]] = []
        closed_at: Optional[int] = None
        for step, (*_, gripper) in rows:
            if gripper < closed_below and closed_at is None:
                closed_at = step
            elif gripper >= closed_below and closed_at is not None:
                cycles.append((closed_at, step))
                closed_at = None
        if closed_at is not None:
            cycles.append((closed_at, None))
        return cycles

    @property
    def grasps(self) -> List[Tuple[int, Optional[int]]]:
        """Every arm's grasp cycles together, ordered by the step the grasp closed. What a metric
        counting grasps over the whole robot reads."""
        return sorted(
            (cycle for arm in self.arms for cycle in self.grasp_cycles(arm)), key=lambda cycle: cycle[0]
        )

    @property
    def holding_at_end(self) -> bool:
        """Whether any arm was still holding when the episode ran out."""
        return any(released is None for _, released in self.grasps)

    @property
    def longest_stall(self) -> int:
        """Most steps the robot spent neither moving nor changing grip: the largest gap between
        change points, over ALL arms together, including the tail. One arm working while the other
        rests is not a stall."""
        steps = sorted({step for rows in self.motion_trace.values() for step, _ in rows})
        if not steps:
            return 0
        gaps = [later - earlier for earlier, later in zip(steps, steps[1:], strict=False)]
        return max(gaps + [self.num_steps - steps[-1]], default=0)

    def to_log_entry(self) -> Dict[str, Any]:
        """The ``episode_log.yaml`` row. Empty blocks are dropped, so a non-composition episode logs
        exactly what it always did; ``False`` and ``0`` are kept, only empty containers go."""
        entry = dataclasses.asdict(self)
        entry["goal_trace"] = [[step, list(achieved)] for step, achieved in self.goal_trace]
        entry["motion_trace"] = {
            arm: [[step, list(values)] for step, values in rows] for arm, rows in self.motion_trace.items()
        }
        entry["object_trace"] = {
            name: [[step, list(center)] for step, center in rows] for name, rows in self.object_trace.items()
        }
        entry["held_trace"] = [[step, list(names)] for step, names in self.held_trace]
        return {key: value for key, value in entry.items() if value is not None and value not in ([], {})}

    @classmethod
    def from_log_entry(cls, entry: Dict[str, Any]) -> "Episode":
        """Rebuild from a logged row, tolerating rows written before a field existed."""
        names = {field.name for field in dataclasses.fields(cls)}
        known = {key: value for key, value in entry.items() if key in names}
        known.setdefault("batch", entry.get("episode", 0))  # pre-batching logs numbered episodes
        known.setdefault("trajectory", entry.get("metrics", {}))  # RoboLab logged these as "metrics"
        episode = cls(**known)
        episode.goal_trace = [(int(step), [bool(x) for x in a]) for step, a in episode.goal_trace]
        trace = episode.motion_trace
        if isinstance(trace, list):  # logs written before the trace was split per arm carried one arm
            trace = {EFFECTOR_NAMES[0]: trace}
        episode.motion_trace = {
            arm: [(int(step), [float(x) for x in values]) for step, values in rows]
            for arm, rows in trace.items()
        }
        episode.object_trace = {
            str(name): [(int(step), [float(x) for x in c]) for step, c in rows]
            for name, rows in episode.object_trace.items()
        }
        episode.held_trace = [(int(step), [str(n) for n in names]) for step, names in episode.held_trace]
        return episode
