import collections
import dataclasses
from typing import Dict, List, Optional, Tuple

from ladder.components.inference.api.core.instructions import ON, Placement, compose
from ladder.components.inference.robots.simpler.composition.placement import (
    BBOX_XY,
    CARROT,
    EGGPLANT,
    INSTANCE_SEP,
    PLATE,
    SPOON,
    TOWEL,
    Goal,
    strip_instance_id,
)

STEPS_PER_GOAL = 100  # measured p90 of one successful pick-place in a chain is 73-98 steps
# The benchmark's length axis, shared by every family that sweeps all of it. It starts at 2 because
# the single-destination family declares its N=1 rung explicitly, at native receptacle size rather
# than this ladder's scaling; 10 movers is the most the packer fits in the workspace. A family that
# covers less of the axis says so where it stops.
# The rungs every length-swept family is built at. Every one of them earns its place: measured over
# the shipped campaigns, each separates the advancement rules by 10-21pp of goal progress, so
# dropping a rung to save eval time would cost a comparison rather than a duplicate.
GOAL_LADDER = range(2, 11)
# The towel is always "the blue towel": a bare destination word is weakly grounded, and naming the colour
# measurably improved delivery to it. Every instruction below and every cached plan says it the same way.
DISPLAY = {CARROT: "carrot", SPOON: "spoon", EGGPLANT: "eggplant", PLATE: "plate", TOWEL: "blue towel"}


def _and(items: List[str]) -> str:
    """'A' / 'A and B' / 'A, B and C'."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" and {items[-1]}"


@dataclasses.dataclass
class Spec:
    """One benchmark task, as declared. ``composition.envs`` turns each into a registered env class."""

    name: str  # env id stem: configs use "<name>-v0", the ManiSkill registry "<name>-v1"
    _: dataclasses.KW_ONLY  # the rest are keyword-only, so a SPECS row names what it sets
    goals: List[Tuple[str, str]]  # (source, target) pairs
    # Receptacles only, and never past the 2.4x the workspace admits: rescaling a mover would confound
    # sequence length with grasp difficulty.
    target_scales: Dict[str, float] = dataclasses.field(default_factory=dict)
    instruction: str = ""  # generated from the goals when left empty
    distractors: List[str] = dataclasses.field(default_factory=list)  # present, unscored, must not move
    ordered: bool = False  # goals must complete in the listed order
    revisits: bool = False  # a repeated source moves the SAME object again, rather than a duplicate
    start_on: Optional[str] = None  # movers start resting on this receptacle, an unscored fixture

    def __post_init__(self) -> None:
        if not self.instruction:
            placements = [
                Placement(DISPLAY[strip_instance_id(source)], ON, DISPLAY[target])
                for source, target in self.goals
            ]
            self.instruction = compose(
                placements,
                sequential=self.ordered or self.revisits,
                source=DISPLAY[self.start_on] if self.start_on else None,
                # A receptacle listed here is a second destination in frame, not a decoy to avoid.
                excluded=[DISPLAY[d] for d in self.distractors if d not in (PLATE, TOWEL)],
            )

    @property
    def max_steps(self) -> int:
        return STEPS_PER_GOAL * len(self.goals)


@dataclasses.dataclass(frozen=True)
class SceneLayout:
    """A resolved :class:`Spec`: what is in the scene and where it may go. No simulator involved, so this
    and the packer it feeds are testable without ManiSkill."""

    goals: List[Goal]
    obj_names: List[str]
    bbox: Dict[str, Tuple[float, float]]  # receptacles already enlarged


def _with_instance_ids(goals: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """A source used more than once becomes its own instance: two carrot goals -> carrot#0, carrot#1."""
    total = collections.Counter(source for source, _ in goals)
    used: Dict[str, int] = collections.Counter()
    resolved = []
    for source, target in goals:
        if total[source] == 1:
            resolved.append((source, target))
            continue
        resolved.append((f"{source}{INSTANCE_SEP}{used[source]}", target))
        used[source] += 1
    return resolved


def scene_layout(spec: Spec) -> SceneLayout:
    pairs = spec.goals if spec.revisits else _with_instance_ids(spec.goals)
    goals = [Goal(source, target, target != TOWEL) for source, target in pairs]  # the towel is soft
    bbox = dict(BBOX_XY)
    for name, scale in spec.target_scales.items():
        bbox[name] = (BBOX_XY[name][0] * scale, BBOX_XY[name][1] * scale)
    for source, _ in pairs:
        bbox.setdefault(source, BBOX_XY[strip_instance_id(source)])
    fixtures = spec.distractors + ([spec.start_on] if spec.start_on else [])  # scene objects, not goals
    objects = [obj for goal in goals for obj in (goal.source, goal.target)] + fixtures
    return SceneLayout(goals=goals, obj_names=list(dict.fromkeys(objects)), bbox=bbox)


# Only these 3 movers exist, so a longer chain duplicates them. The cubes are left out because the yellow
# one is low-contrast against the cream plate, not because stacking is hard.
_MOVER_CYCLE = [CARROT, SPOON, EGGPLANT]


def _movers(n: int) -> List[str]:
    return [_MOVER_CYCLE[i % len(_MOVER_CYCLE)] for i in range(n)]


def _on(target: str, n: int) -> List[Tuple[str, str]]:
    return [(mover, target) for mover in _movers(n)]


def _split(n_plate: int, n_towel: int) -> List[Tuple[str, str]]:
    """Two destinations off ONE mover cycle, so the objects stay distinct until the total exceeds 3."""
    movers = _movers(n_plate + n_towel)
    return [(m, PLATE) for m in movers[:n_plate]] + [(m, TOWEL) for m in movers[n_plate:]]


def _excl(n: int) -> List[Tuple[str, str]]:
    """Exclusion goals cycle the two non-eggplant movers, since the eggplant is the distractor."""
    return [([CARROT, SPOON][i % 2], PLATE) for i in range(n)]


def _spoons(n_plate: int, n_towel: int) -> List[Tuple[str, str]]:
    return [(SPOON, PLATE)] * n_plate + [(SPOON, TOWEL)] * n_towel


def _relay_legs(n: int) -> List[Tuple[str, str]]:
    """One mover at a time completes its out-and-back, table -> plate -> towel."""
    return [(mover, target) for mover in _movers(n) for target in (PLATE, TOWEL)][:n]


def _shuttle_legs(n: int) -> List[Tuple[str, str]]:
    """All 3 movers to the plate, then all 3 to the towel, then back."""
    rounds = (n + 2) // 3
    return [(m, PLATE if r % 2 == 0 else TOWEL) for r in range(rounds) for m in _MOVER_CYCLE][:n]


def _ladder(n: int) -> float:
    """Receptacle growth for the single-destination ladder: sqrt, not linear, since a shared receptacle
    credits piled objects too, so the surface grows with footprint rather than volume."""
    return round(min(2.4, max(1.2, 0.78 * n**0.5)), 1)


def _revisit(name: str, legs: List[Tuple[str, str]], plate: float, towel: float) -> Spec:
    return Spec(name, goals=legs, target_scales={PLATE: plate, TOWEL: towel}, ordered=True, revisits=True)


# One canonical scene per (task type, sequence length N), sweeping task TYPES across sequence length
# rather than enumerating object combos.
SPECS: List[Spec] = [
    # N=1: the two canonical SimplerEnv widowx pick-places, at native receptacle size. The towel rung
    # takes the spoon rather than the mover cycle's carrot, since that is the published pairing.
    Spec("all1_plate", goals=[(CARROT, PLATE)]),
    Spec("all1_towel", goals=[(SPOON, TOWEL)]),
    # single destination: pure sequence, one receptacle
    *[Spec(f"all{n}_plate", goals=_on(PLATE, n), target_scales={PLATE: _ladder(n)}) for n in GOAL_LADDER],
    *[Spec(f"all{n}_towel", goals=_on(TOWEL, n), target_scales={TOWEL: _ladder(n)}) for n in GOAL_LADDER],
    # exclusion: the eggplant is present and must be left in place
    Spec(
        "except_eggplant_2",
        goals=[(CARROT, PLATE), (SPOON, PLATE)],
        target_scales={PLATE: 1.3},
        distractors=[EGGPLANT],
    ),
    Spec(
        "except_eggplant_3",
        goals=[(CARROT, PLATE), (SPOON, PLATE), (CARROT, PLATE)],
        target_scales={PLATE: 1.7},
        distractors=[EGGPLANT],
    ),
    *[
        Spec(
            f"except_eggplant_{n}",
            goals=_excl(n),
            target_scales={PLATE: _ladder(n)},
            distractors=[EGGPLANT],
        )
        for n in (4, 5)
    ],
    # transfer: movers start resting on one receptacle and must reach the other
    *[
        Spec(
            f"move_plate_to_towel_{n}",
            goals=_on(TOWEL, n),
            target_scales={PLATE: s, TOWEL: s},
            start_on=PLATE,
        )
        for n, s in ((2, 1.3), (3, 1.7), (4, 2.0))
    ],
    Spec(
        "move_plate_to_towel_5",
        goals=_on(TOWEL, 5),
        target_scales={PLATE: 2.0, TOWEL: 2.0},
        start_on=PLATE,
    ),
    Spec(
        "move_towel_to_plate_3",
        goals=_on(PLATE, 3),
        target_scales={TOWEL: 1.7, PLATE: 1.7},
        start_on=TOWEL,
    ),
    # ordered: distinct movers, so the required order is well defined. Only 3 movers exist, and a
    # repeated one would make the stated order unfollowable ("first the carrot, then the carrot"),
    # so this family stops at 3 rather than for want of a rung.
    Spec("ordered_plate_2", goals=_on(PLATE, 2), target_scales={PLATE: 1.3}, ordered=True),
    Spec("ordered_plate_3", goals=_on(PLATE, 3), target_scales={PLATE: 1.7}, ordered=True),
    # split: routing on top of sequence length
    Spec("split2", goals=_split(1, 1)),
    Spec("split3", goals=_split(2, 1), target_scales={PLATE: 1.3}),
    Spec("split4", goals=_split(2, 2), target_scales={PLATE: 1.3, TOWEL: 1.3}),
    Spec("split5", goals=_split(3, 2), target_scales={PLATE: 1.5, TOWEL: 1.3}),
    Spec("split6", goals=_split(3, 3), target_scales={PLATE: 1.7, TOWEL: 1.7}),
    Spec("split7", goals=_split(4, 3), target_scales={PLATE: 1.7, TOWEL: 1.5}),
    # revisit: movers stop packing past ~7 and mover scale is fixed, so N=6-10 revisits instead of adding
    # objects. Such a chain scores as sequential milestones (see composition.envs.evaluate). "e" eases the
    # towel, the relay accumulator; "sp" also shrinks the plate for a lower rim, easing spoon extraction.
    # A shuttle rotates all 3 movers to one receptacle before turning round, so nothing is revisited
    # until the third round opens at leg 7: shuttle4-6 would be relays under another name. 7 is the
    # family's floor, which is why revisit has no rung in the 2-5 window and relay does.
    *[_revisit(f"shuttle{n}", _shuttle_legs(n), 1.7, 1.7) for n in (7, 8, 9, 10)],
    # ONE receptacle scaling for the whole relay ladder. The three that shipped at rungs 6-10 are the
    # same goals with the towel at 1.5, at 2.0 ("e"), and at 2.0 with a 1.2 plate ("sp"), so two of
    # them are probes rather than tasks.
    #
    # The eased towel is kept on SOLVABILITY, not on scoring highest: under the privileged rule,
    # which is perfect switch timing and therefore the ceiling any orchestrator could reach, the 1.5
    # towel scores 1 success in 960 episodes and exactly zero at four of the five rungs. A scene no
    # system can finish with perfect decisions measures manipulation, not composition, which is the
    # audit every scene here is supposed to pass. The towel is the relay accumulator, holding every
    # mover by the late legs, so at 1.5 it obstructs them physically.
    #
    # Choosing on rate alone would have been choosing on the outcome: the scaling interacts with the
    # system, worth +4.1pp success to the visual rule and +0.1pp to the direct instruction, so the
    # eased towel widens the orchestrator effect on this family. It does so by lifting a floor that
    # suppressed every arm, not by favouring one.
    #
    # The "e" suffix is kept rather than renamed to relay{n}: composition.csv already carries
    # relay6-relay10 rows under the 1.5 towel, and reusing those names would pool two different
    # scenes under one unit.
    *[_revisit(f"relay{n}e", _relay_legs(n), 1.5, 2.0) for n in GOAL_LADDER],
    # identical movers: 3 spoons, so no wording can bind a particular spoon to a particular goal
    Spec("spoons2", goals=_spoons(1, 1)),
    Spec(
        "spoons3",
        goals=[(SPOON, PLATE), (SPOON, TOWEL), (SPOON, PLATE)],
        target_scales={PLATE: 1.3},
    ),
    Spec(
        "spoons4",
        goals=_spoons(2, 2),
        target_scales={PLATE: 1.3, TOWEL: 1.3},
    ),
    Spec(
        "spoons5",
        goals=_spoons(3, 2),
        target_scales={PLATE: 1.5, TOWEL: 1.3},
    ),
]
