import dataclasses
from typing import Dict, List, Optional, Tuple

from ladder.components.inference.api.core.instructions import IN, ON, Placement, compose
from ladder.components.inference.robots.robot_arena.scene_loader import SceneData, load_scene
from ladder.components.inference.robots.robot_arena.success_checker import TaskSpec, TaskType, match_instances

_PP = TaskType.PICK_AND_PLACE

# Long-horizon step budget: 100 per goal, against 60 for a canonical single-goal RA scene.
_STEPS_PER_GOAL = 100

# How far above a start receptacle a transfer task's movers spawn, so the settle rests them on it.
_START_CLEARANCE_M = 0.03


@dataclasses.dataclass(frozen=True)
class ImportedObject:
    """A movable object borrowed from another scene's asset set."""

    donor_scene: str
    donor_object: str  # exact ObjectConfig.name in the donor scene
    position: Tuple[float, float, float]  # spawn in the host scene (settle drops it onto the table)
    scale_multiplier: float = 1.0
    # Rename for a second copy: two objects sharing a name are one object to the matcher.
    alias: Optional[str] = None


def _grouped_by_destination(legs: List["Leg"]) -> List[Tuple[str, str, List[str]]]:
    """Consecutive legs sharing a destination become one clause: "the spoon and the donut on the cloth"."""
    grouped: List[Tuple[str, str, List[str]]] = []
    for leg in legs:
        if grouped and grouped[-1][0] == leg.destination and grouped[-1][1] == leg.relation:
            grouped[-1][2].append(leg.source)
        else:
            grouped.append((leg.destination, leg.relation, [leg.source]))
    return grouped


@dataclasses.dataclass(frozen=True)
class Leg:
    """One (mover, destination) placement. A task is a sequence of these."""

    source: str
    destination: str
    # "in" for a WALLED receptacle (pan, pot): under "on" only a rim-level rest counts.
    relation: str = "on"


@dataclasses.dataclass(frozen=True)
class CompositionTask:
    """A multi-goal task over one scene, as a sequence of :class:`Leg` placements.

    :attr:`goals` and :attr:`instruction` are both derived from the legs, so the scoring and the
    wording cannot drift apart. Movers onto ONE destination are written shorter as
    :attr:`sources` plus :attr:`destination`. Declaring both forms is an error, since the goal
    order would then depend on which one won.
    """

    scene: str
    _: dataclasses.KW_ONLY  # the rest are keyword-only, so a registry row names what it sets
    sources: List[str] = dataclasses.field(default_factory=list)  # movers onto one shared destination
    destination: Optional[str] = None
    legs: List[Leg] = dataclasses.field(default_factory=list)  # a destination per mover
    relation: str = "on"  # "in" for a WALLED receptacle; under "on" only a rim-level rest counts
    extra_objects: List[ImportedObject] = dataclasses.field(default_factory=list)
    # Native scene objects to leave OUT, by exact scene name. The mirror of extra_objects, and what
    # makes an "everything" wording well posed: that phrasing covers every movable object in the
    # scene, so a task worded that way may not hold a movable the goals do not name.
    remove_objects: List[str] = dataclasses.field(default_factory=list)
    # Object name -> mesh scale multiplier. Receptacle enlargement only; movers keep native scale.
    scale_overrides: Dict[str, float] = dataclasses.field(default_factory=dict)
    # Object name -> replacement spawn position, keyed by exact scene name.
    position_overrides: Dict[str, Tuple[float, float, float]] = dataclasses.field(default_factory=dict)
    # Object name -> fixed flag. Pin a light destination, or placements drag it and shed each other.
    fixed_overrides: Dict[str, bool] = dataclasses.field(default_factory=dict)
    # The goals must FIRST be satisfied in list order; the final conjunction must hold either way.
    ordered: bool = False
    # Score the legs as ordered milestones. Required when a task moves one object more than once,
    # since its earlier legs do not survive to the end.
    milestones: bool = False
    # The movers start resting on this receptacle, which is scenery rather than a goal.
    start_on: Optional[str] = None
    # Movable objects to leave alone; the only wording that does not name its objects.
    excluded: List[str] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if bool(self.sources) == bool(self.legs):
            raise ValueError(
                f"{self.scene}: declare either sources with a destination, or legs, not both or neither"
            )
        if self.sources and self.destination is None:
            raise ValueError(f"{self.scene}: sources without a destination")

    @property
    def _legs(self) -> List[Leg]:
        if self.legs:
            return list(self.legs)
        return [Leg(source, self.destination, self.relation) for source in self.sources]

    @property
    def goals(self) -> List[TaskSpec]:
        return [
            TaskSpec(_PP, leg.source, leg.destination, spatial_relation=leg.relation) for leg in self._legs
        ]

    @property
    def instruction(self) -> str:
        """The task's command, in the wording shared with SimplerEnv and RoboLab."""
        placements = [
            Placement(leg.source, ON if leg.relation == "on" else IN, leg.destination) for leg in self._legs
        ]
        return compose(
            placements,
            sequential=self.ordered or self.milestones,
            source=self.start_on,
            excluded=self.excluded,
        )

    @property
    def max_episode_steps(self) -> int:
        return _STEPS_PER_GOAL * len(self._legs)


def compose_scene_data(scene_data: SceneData, task: CompositionTask, data_root: str) -> SceneData:
    """Return ``scene_data`` with the task's instruction, imported objects, and per-object overrides."""
    objects = list(scene_data.objects)
    if task.remove_objects:
        unknown = [name for name in task.remove_objects if not any(o.name == name for o in objects)]
        if unknown:
            raise ValueError(f"remove_objects names unknown object(s) {unknown} in {task.scene!r}")
        objects = [o for o in objects if o.name not in task.remove_objects]
    for imp in task.extra_objects:
        donor = load_scene(data_root, imp.donor_scene)
        cfg = next((o for o in donor.objects if o.name == imp.donor_object), None)
        if cfg is None:
            raise ValueError(f"{imp.donor_object!r} not found in donor scene {imp.donor_scene!r}")
        name = imp.alias or cfg.name
        if any(o.name == name for o in objects):
            raise ValueError(
                f"import {name!r} collides with a host object; give the copy an alias if the task "
                "wants two of them"
            )
        objects.append(
            dataclasses.replace(
                cfg, name=name, position=list(imp.position), scale=cfg.scale * imp.scale_multiplier
            )
        )

    index = {cfg.name: i for i, cfg in enumerate(objects)}

    def locate(kind: str, name: str) -> int:
        if name not in index:
            raise ValueError(f"{kind} names unknown object {name!r}")
        return index[name]

    for name, multiplier in task.scale_overrides.items():
        i = locate("scale_overrides", name)
        objects[i] = dataclasses.replace(objects[i], scale=objects[i].scale * multiplier)
    for name, position in task.position_overrides.items():
        i = locate("position_overrides", name)
        objects[i] = dataclasses.replace(objects[i], position=list(position))
    for name, fixed in task.fixed_overrides.items():
        i = locate("fixed_overrides", name)
        objects[i] = dataclasses.replace(objects[i], fixed=fixed)
    if task.start_on is not None:
        # Only the height is derived; the certified XY is what keeps the movers clear of each other.
        # Resolved by fuzzy name like every other source, where the overrides above key on exact names.
        start = objects[match_instances(task.start_on, objects, 1)[0]]
        for source in dict.fromkeys(goal.target_ref for goal in task.goals):
            i = match_instances(source, objects, 1)[0]
            x, y, _ = objects[i].position
            objects[i] = dataclasses.replace(
                objects[i], position=[x, y, start.position[2] + _START_CLEARANCE_M]
            )
    return dataclasses.replace(scene_data, objects=objects, task_instruction=task.instruction)
