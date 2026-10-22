import dataclasses
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from ladder.components.inference.api.core.instructions import Placement, compose, subtask
from ladder.components.inference.robots.robolab.composition.catalog import (
    CATALOG,
    FRUIT_LADDER,
    RECEPTACLE_TYPES,
    verb,
)
from ladder.components.inference.robots.robolab.composition.entities import (
    Goal,
    PlacedObject,
    PlacedReceptacle,
    Receptacle,
)
from ladder.components.inference.robots.robolab.composition.layout import (
    JITTER_LADDER,
    REACH_MAX,
    RECEP_ANCHOR_1,
    RECEP_ANCHOR_2A,
    RECEP_ANCHOR_2B,
    ROBOLAB_BUDGET_S,
    SOURCE_ANCHOR,
    dist,
    fit_on_receptacle,
    fit_scatter,
    pull_into_reach,
)

# The primary N-scaling axis. It starts at 2 because the N=1 rung is SINGLES below, and stops at 10,
# the longest rung the tabletop scatters within reach.
#
# DENSE, deliberately, and it is the CONSTRAINED families that sweep it: route and ordered_bin run
# every rung from 2 to 10. That is what makes the length curve readable, because a rung's score then
# varies with length alone rather than with which family happens to reach that far.
GOAL_LADDER = range(2, 11)
# The single-destination control family is sparse where the others are dense: short, middle and long.
# It is the family the direct instruction already handles, so it is the length axis's control rather
# than a discriminator, and intermediate rungs would add episodes without adding a question.
BIN_LADDER = (2, 6, 10)
SINGLES: Tuple[Tuple[str, str], ...] = (("comp_bin_1", "bin"), ("comp_plate_1", "plate"))

# ------------------------------ scene instances ------------------------------


def task_class_name(spec_name: str) -> str:
    """comp_bin_6 -> CompBin6Task (the RoboLab Task subclass name / env task_name)."""
    return "".join(part.capitalize() for part in spec_name.split("_")) + "Task"


def task_target(spec_name: str) -> str:
    """The RoboLab task identifier for a spec: the shared tasks module, addressing one class in it.
    A bare path would resolve to whichever Task that module defines first."""
    tasks_module = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.py")
    return f"{tasks_module}::{task_class_name(spec_name)}"


@dataclasses.dataclass(frozen=True)
class CompositionSpec:
    """One long-horizon composition task, fully described physically: the exact graspable
    instances on the table (goals + distractors), the receptacle instances, and an optional
    ordering constraint. This is the declarative unit the factory turns into a programmatic
    RoboLab ``Task`` (one RigidObjectCfg per instance on a bare tabletop) and the
    oracle/decomposer read as the plan. The visible object count is exactly ``len(objects)``."""

    name: str
    family: str  # single_dest | split | distractor | ordered | transfer | relay | revisit | identical
    objects: Tuple[PlacedObject, ...]
    receptacles: Tuple[PlacedReceptacle, ...]
    ordered: bool = False
    # The receptacles each object passes through BEFORE its final one, in order. A relay names one
    # ("table -> plate -> bin"); a revisit names two, the second of which is the final receptacle
    # again ("table -> bin -> plate -> bin"). Empty for every family that places each object once.
    via_legs: Tuple[str, ...] = ()
    jitter: float = JITTER_LADDER[0]  # per-episode uniform x/y jitter (m); DERIVED, see fit_scatter

    def __post_init__(self) -> None:
        # The reset is ONE event with ONE pose_range for every asset it randomizes, so a spec cannot
        # mix objects scattered on the table with objects confined to a source receptacle -- those
        # need different jitters to stay on their respective surfaces. Every family satisfies this
        # (transfer puts all of its graspables on the source); this keeps it true.
        if len({o.start_on is not None for o in self.objects}) > 1:
            raise ValueError(
                f"{self.name}: mixes table-scattered and source-resting graspables, which cannot "
                f"share one reset jitter"
            )

    @property
    def source(self) -> Optional[str]:
        """Transfer tasks: the receptacle the objects start resting on (named in the instruction)."""
        return next((o.start_on for o in self.objects if o.start_on), None)

    @property
    def goal_objects(self) -> List[PlacedObject]:
        return [o for o in self.objects if o.receptacle is not None]

    @property
    def distractor_objects(self) -> List[PlacedObject]:
        return [o for o in self.objects if o.receptacle is None]

    @property
    def n(self) -> int:
        return len(self.goal_objects)

    @property
    def n_ops(self) -> int:
        """Pick-and-place operations, the unit the budget scales on. Each intermediate leg is one
        more move of every object: a relay moves each twice, a revisit three times."""
        return (1 + len(self.via_legs)) * self.n

    @property
    def episode_length_s(self) -> int:
        """From RoboLab's own benchmark tasks. A larger budget does not convert into more completed
        episodes: horizon length is the binding constraint, not the clock."""
        return ROBOLAB_BUDGET_S[min(self.n_ops, max(ROBOLAB_BUDGET_S))]

    @property
    def receptacle_map(self) -> Dict[str, Receptacle]:
        return {
            r.prim: Receptacle(r.prim, RECEPTACLE_TYPES[r.type_id].kind, RECEPTACLE_TYPES[r.type_id].noun)
            for r in self.receptacles
        }

    @property
    def goals(self) -> List[Goal]:
        return [Goal(o.prim, o.receptacle, CATALOG[o.type_id].noun) for o in self.goal_objects]

    @property
    def receptacles_used(self) -> List[str]:
        seen: List[str] = []
        for g in self.goals:
            if g.receptacle not in seen:
                seen.append(g.receptacle)
        return seen

    @property
    def contact_object_list(self) -> List[str]:
        """Every entity tracked for pose/contact: graspables then receptacles. No ``table`` -- the
        tabletop is part of the base USD, not a tracked rigid body."""
        return [o.prim for o in self.objects] + [r.prim for r in self.receptacles]

    @property
    def free_object_list(self) -> List[str]:
        """The graspables the reset randomizes: all of them. The spec's jitter is sized so no draw
        can leave the intended surface. Receptacles are never listed, so they stay pinned."""
        return [o.prim for o in self.objects]

    @property
    def instruction(self) -> str:
        """The task's command, in the wording shared with SimplerEnv and RobotArena."""
        rm = self.receptacle_map
        stops = [*self.via_legs, self.goals[0].receptacle] if self.via_legs else None
        if stops is not None:
            placements = [
                Placement(g.noun, verb(rm[stop].kind), rm[stop].noun) for g in self.goals for stop in stops
            ]
        else:
            placements = [
                Placement(g.noun, verb(rm[g.receptacle].kind), rm[g.receptacle].noun) for g in self.goals
            ]
        return compose(
            placements,
            sequential=self.ordered or bool(self.via_legs),
            source=rm[self.source].noun if self.source else None,
            excluded=[CATALOG[o.type_id].noun for o in self.distractor_objects],
        )


# ------------------------------ instruction composition ------------------------------


def _instances(
    type_ids: List[str],
    positions: List[Tuple[float, float]],
    receptacle_of: Dict[int, Optional[str]],
    start_on: Optional[str] = None,
) -> Tuple[PlacedObject, ...]:
    """Build unique-primed object instances. A type used once keeps its bare name ("orange"); a
    repeated type is suffixed ("orange_1", "orange_2"). Prim names are internal ids only; the
    instruction noun always comes from the catalog, so suffixes never leak into language."""
    counts: Dict[str, int] = {}
    for t in type_ids:
        counts[t] = counts.get(t, 0) + 1
    seen: Dict[str, int] = {}
    objs: List[PlacedObject] = []
    for i, t in enumerate(type_ids):
        if counts[t] == 1:
            prim = t
        else:
            seen[t] = seen.get(t, 0) + 1
            prim = f"{t}_{seen[t]}"
        objs.append(
            PlacedObject(
                prim,
                type_id=t,
                x=positions[i][0],
                y=positions[i][1],
                receptacle=receptacle_of[i],
                start_on=start_on,
            )
        )
    return tuple(objs)


def single_dest(
    name: str,
    n: int,
    receptacle: str = "bin",
    pool: Tuple[str, ...] = FRUIT_LADDER,
    family: str = "single_dest",
    distractor_types: Tuple[str, ...] = (),
    ordered: bool = False,
    reach_max: Optional[float] = None,
) -> CompositionSpec:
    """Put the first ``n`` pool objects into one receptacle; optional decoy distractors that must
    stay put. The core N-goal ladder and its distractor variants."""
    rt = RECEPTACLE_TYPES[receptacle]
    recep = PlacedReceptacle(receptacle, type_id=receptacle, x=RECEP_ANCHOR_1[0], y=RECEP_ANCHOR_1[1])
    reserved = [(recep.x, recep.y, rt.footprint[0], rt.footprint[1])]
    type_ids = list(pool[:n]) + list(distractor_types)
    jitter, positions = fit_scatter([CATALOG[t].radius for t in type_ids], reserved, reach_max)
    if ordered:
        # An ordered rung only tests ordering if the required order FIGHTS the policy's own
        # preference, and policies tend to go nearest-first. Requiring the farthest object first
        # makes that prior maximally unhelpful, so a rung measures instruction-following rather than
        # which layout fit_scatter happened to produce. Pairs move together, so every object keeps
        # the slot fit_scatter sized for its radius.
        goals = sorted(zip(type_ids[:n], positions[:n], strict=True), key=lambda tp: -dist(tp[1], (0.0, 0.0)))
        type_ids = [t for t, _ in goals] + list(distractor_types)
        positions = [p for _, p in goals] + list(positions[n:])
    receptacle_of = {i: (receptacle if i < n else None) for i in range(len(type_ids))}
    return CompositionSpec(
        name=name,
        family=family,
        objects=_instances(type_ids, positions, receptacle_of),
        receptacles=(recep,),
        ordered=ordered,
        jitter=jitter,
    )


def split(
    name: str,
    groups: List[Tuple[Tuple[str, ...], str]],
    family: str = "split",
    reach_max: Optional[float] = None,
    anchors: Optional[List[Tuple[float, float]]] = None,
    reserve_as: Optional[Dict[str, str]] = None,
) -> CompositionSpec:
    """Objects partitioned across two receptacles. ``groups`` = [(object_types, receptacle_id), ...].
    Receptacles sit at the two back anchors; every object is a goal (no distractors).

    ``groups`` fixes BOTH the plan order and, by position, which anchor each receptacle takes.
    ``anchors`` decouples them, so a variant can reorder the plan while every receptacle and object
    stays exactly where the base rung puts it.

    ``reserve_as`` maps a receptacle id to the type whose footprint the scatter should reserve for
    it. Object placement is a function of the reserved boxes, so two rungs differing only in
    receptacle otherwise get different object positions, and the resulting change in transport
    distance would dominate the receptacle effect. Reserving the bowl AS a plate makes the twins
    identical in layout, leaving the receptacle as the only difference. Name the LARGER type: a smaller reservation would let an object spawn overlapping
    the bigger rim. Only the differing slot is remapped, so the shared bin stays where it is."""
    anchors = anchors or [RECEP_ANCHOR_2A, RECEP_ANCHOR_2B]
    receps = tuple(
        PlacedReceptacle(rid, type_id=rid, x=anchors[k][0], y=anchors[k][1])
        for k, (_, rid) in enumerate(groups)
    )
    reserve_as = reserve_as or {}
    reserved = [(r.x, r.y, *RECEPTACLE_TYPES[reserve_as.get(r.type_id, r.type_id)].footprint) for r in receps]
    type_ids: List[str] = []
    recep_ids: List[str] = []
    for types, rid in groups:
        type_ids += list(types)
        recep_ids += [rid] * len(types)
    jitter, positions = fit_scatter([CATALOG[t].radius for t in type_ids], reserved, reach_max)
    receptacle_of = {i: recep_ids[i] for i in range(len(type_ids))}
    return CompositionSpec(
        name=name,
        family=family,
        objects=_instances(type_ids, positions, receptacle_of),
        receptacles=receps,
        jitter=jitter,
    )


def transfer(
    name: str,
    object_types: Tuple[str, ...],
    source: str = "plate",
    dest: str = "bin",
    source_anchor: Tuple[float, float] = RECEP_ANCHOR_2A,
    reach_max: Optional[float] = None,
) -> CompositionSpec:
    """Objects start resting ON/IN the ``source`` receptacle and must be moved to ``dest``. Success
    only requires them in ``dest`` (the source is a scene fixture, not a goal); the "from the source"
    is expressed by the initial state + instruction (mirrors SimplerEnv's transfer family). The
    objects are laid out on a ring within the source and re-drawn every episode, with a jitter sized
    so no draw can leave the source's top surface."""
    dst = PlacedReceptacle(dest, type_id=dest, x=RECEP_ANCHOR_2B[0], y=RECEP_ANCHOR_2B[1])
    radii = [CATALOG[t].radius for t in object_types]
    jitter, offsets = fit_on_receptacle(radii, RECEPTACLE_TYPES[source].footprint)
    anchor = pull_into_reach(source_anchor, offsets, jitter, reach_max)
    src = PlacedReceptacle(source, type_id=source, x=anchor[0], y=anchor[1])
    positions = [(src.x + dx, src.y + dy) for dx, dy in offsets]
    objects = _instances(
        list(object_types), positions, {i: dest for i in range(len(object_types))}, start_on=source
    )
    return CompositionSpec(
        name=name, family="transfer", objects=objects, receptacles=(src, dst), jitter=jitter
    )


def relay(
    name: str,
    object_types: Tuple[str, ...],
    via: str = "plate",
    dest: str = "bin",
    via_anchor: Tuple[float, float] = RECEP_ANCHOR_2A,
    reach_max: Optional[float] = None,
) -> CompositionSpec:
    """Objects start on the table and must be relayed table -> ``via`` -> ``dest``. Success is a
    latched milestone (each object must pass through ``via`` before its placement in ``dest`` counts),
    so a direct table->dest delivery does not satisfy it (mirrors SimplerEnv's relay family)."""
    via_r = PlacedReceptacle(via, type_id=via, x=via_anchor[0], y=via_anchor[1])
    dst_r = PlacedReceptacle(dest, type_id=dest, x=RECEP_ANCHOR_2B[0], y=RECEP_ANCHOR_2B[1])
    reserved = [
        (via_r.x, via_r.y, *RECEPTACLE_TYPES[via].footprint),
        (dst_r.x, dst_r.y, *RECEPTACLE_TYPES[dest].footprint),
    ]
    jitter, positions = fit_scatter([CATALOG[t].radius for t in object_types], reserved, reach_max)
    objects = _instances(list(object_types), positions, {i: dest for i in range(len(object_types))})
    return CompositionSpec(
        name=name,
        family="relay",
        objects=objects,
        receptacles=(via_r, dst_r),
        ordered=True,
        via_legs=(via,),
        jitter=jitter,
    )


def revisit(
    name: str,
    object_types: Tuple[str, ...],
    home: str = "bin",
    away: str = "plate",
    away_anchor: Tuple[float, float] = RECEP_ANCHOR_2A,
    reach_max: Optional[float] = None,
) -> CompositionSpec:
    """Objects go table -> ``home`` -> ``away`` -> ``home``: the last leg returns each object to a
    receptacle it has already left, which no end state can witness, since the first visit is undone
    by the second leg. Only the ORDER separates the two visits to ``home``, so the legs latch in
    sequence (see :func:`advance_legs`)."""
    away_r = PlacedReceptacle(away, type_id=away, x=away_anchor[0], y=away_anchor[1])
    home_r = PlacedReceptacle(home, type_id=home, x=RECEP_ANCHOR_2B[0], y=RECEP_ANCHOR_2B[1])
    reserved = [
        (away_r.x, away_r.y, *RECEPTACLE_TYPES[away].footprint),
        (home_r.x, home_r.y, *RECEPTACLE_TYPES[home].footprint),
    ]
    jitter, positions = fit_scatter([CATALOG[t].radius for t in object_types], reserved, reach_max)
    objects = _instances(list(object_types), positions, {i: home for i in range(len(object_types))})
    return CompositionSpec(
        name=name,
        family="revisit",
        objects=objects,
        receptacles=(away_r, home_r),
        ordered=True,
        via_legs=(home, away),
        jitter=jitter,
    )


def identical(
    name: str, obj_type: str, n: int, receptacle: str = "bin", reach_max: Optional[float] = None
) -> CompositionSpec:
    """``n`` instances of the SAME object into one receptacle -- a counting task ("put the 3 lemons in
    the bin"). Tracks multi-instance / matched-count success, as in the SimplerEnv/RobotArena
    extended benchmarks."""
    return single_dest(
        name, n, receptacle=receptacle, pool=(obj_type,) * n, family="identical", reach_max=reach_max
    )


# --------------------------------- benchmark matrix ---------------------------------


def _build_specs() -> "OrderedDict[str, CompositionSpec]":
    specs: List[CompositionSpec] = []

    for single_name, single_recep in SINGLES:
        specs.append(single_dest(single_name, 1, receptacle=single_recep, reach_max=REACH_MAX))

    # Family A - single-destination ladder into a bin, the primary N-scaling axis. Three rungs,
    # short/middle/long, rather than every N: this is the family the direct instruction already
    # handles, so it is the length axis's control, not a discriminator, and intermediate rungs add
    # episodes without adding a question.
    for n in BIN_LADDER:
        specs.append(single_dest(f"comp_bin_{n}", n, reach_max=REACH_MAX))

    # Family B - distractors present: put the fruit in the bin while non-fruit blocks sit on the
    # table and must be left behind (goals are pure fruit so "all the fruit" is unambiguous).
    #
    # Deliberately only TWO rungs. Exclusion reads as a composition constraint but mostly tests
    # low-level grounding (a policy told "put the lemon in the bin" may take the block anyway), and
    # ground-truth subtasks barely help, so more rungs would add floor rather than discrimination.
    specs.append(
        single_dest(
            "comp_bin_2_distractor",
            2,
            family="distractor",
            distractor_types=("red_block",),
            reach_max=REACH_MAX,
        )
    )
    specs.append(
        single_dest(
            "comp_bin_4_distractor",
            4,
            family="distractor",
            distractor_types=("red_block", "blue_block"),
            reach_max=REACH_MAX,
        )
    )

    # Family D - ordered sequence into a bin (distinct objects so the required order is meaningful).
    # The sharpest discriminator the suite has: a flat policy can place every object and still fail
    # by delivering them in an arbitrary order. It spans N rather than sitting at one rung because a
    # policy that ignores
    # the order has an exactly known success rate, 1/N! -- 1/2, 1/6, 1/24, 1/120 -- so the rungs form
    # a null curve that a real ordering capability must beat, and that a decomposing system should
    # hold flat against. N=2 is a weak rung on its own (a coin flip) and is kept as that curve's
    # anchor, not as a test.
    # Swept over the full GOAL_LADDER so the long end of the length axis is not held by one family.
    for n in GOAL_LADDER:
        specs.append(
            single_dest(f"comp_ordered_bin_{n}", n, family="ordered", ordered=True, reach_max=REACH_MAX)
        )

    # Family E - transfer: objects start on the plate, move them all to the bin.
    for types in (("orange", "lemon"), ("orange", "lemon", "lime")):
        specs.append(
            transfer(
                f"comp_transfer_{len(types)}",
                types,
                source_anchor=SOURCE_ANCHOR,
                reach_max=REACH_MAX,
            )
        )

    # Family F - relay: table -> plate -> bin, milestone-enforced. The milestone needs an object that
    # RESTS on the plate, which rules out the orange (a sphere), the avocado and the onion (both
    # taller than they are wide). It does NOT rule out the banana: at radius/rest_z 4.45 it is the
    # FLATTEST thing in the catalog, ahead of lemon (1.46) and lime (1.27).
    #
    # A third object does not have to SHARE the plate (the banana's 17.8 cm long axis does not fit
    # beside others on a 16.6 cm plate): advance_legs chains leg k on every earlier leg latching, so
    # an object may be relayed on its own and the plate is only ever asked to hold one. Co-residency
    # is permitted (an early placement that still holds when the leg before it latches counts), not
    # required. So comp_relay_3 is solvable object-by-object, but NOT by the collective reading
    # "first put everything on the plate".
    # NOTE ON THE NAMES: this family is parameterised by OBJECT count while goal counts are
    # placements, so comp_relay_N is a 2N-goal task (comp_revisit_N a 3N-goal one). The names are
    # kept for stability; goal counts are placements everywhere else.
    for objs in (
        ("lemon", "lime"),
        ("lemon", "lime", "banana"),
        ("lemon", "lime", "banana", "orange"),
        ("lemon", "lime", "banana", "orange", "avocado"),
    ):
        specs.append(relay(f"comp_relay_{len(objs)}", objs, via_anchor=SOURCE_ANCHOR, reach_max=REACH_MAX))
    # revisit, as in the SimplerEnv composition benchmark. One object keeps the chain
    # short enough that the budget still covers three moves of it.
    for objs in (("lemon",), ("lemon", "lime")):
        specs.append(
            revisit(f"comp_revisit_{len(objs)}", objs, away_anchor=SOURCE_ANCHOR, reach_max=REACH_MAX)
        )

    # Family H - routing with the plate side held to objects that rest flat by construction. Family
    # C-b covers the plate/bin split; these vary the goal count on a nested ladder
    # (2 subset 4 subset 6) with the receptacle question already settled.
    for n, (to_plate, to_bin) in (
        (2, (("lemon",), ("orange",))),
        (3, (("lemon", "lime"), ("orange",))),
        (4, (("lemon", "lime"), ("orange", "avocado"))),
        (5, (("lemon", "lime", "banana"), ("orange", "avocado"))),
        (6, (("lemon", "lime", "banana"), ("orange", "avocado", "onion"))),
    ):
        specs.append(
            split(
                f"comp_route_{n}", [(to_plate, "plate"), (to_bin, "bin")], family="route", reach_max=REACH_MAX
            )
        )

    # The long end of the routing ladder, so the long rungs are not held by comp_bin_N alone. These
    # take FRUIT_LADDER[:n] (the SAME objects comp_bin_n uses at that rung) and split them across two receptacles, so route_n against
    # bin_n is one instruction naming two destinations against one naming a single destination, with
    # the object set held fixed. The plate side is the three objects that rest flat by construction
    # (banana 4.45, lemon 1.46, lime 1.27 on radius/rest_z); everything else goes to the walled bin,
    # where shape does not matter. Nests with comp_route_6, which is already FRUIT_LADDER[:6].
    _PLATE_SAFE = ("banana", "lemon", "lime")
    for n in range(7, 11):
        pool = FRUIT_LADDER[:n]
        to_plate = tuple(o for o in pool if o in _PLATE_SAFE)
        to_bin = tuple(o for o in pool if o not in _PLATE_SAFE)
        specs.append(
            split(
                f"comp_route_{n}",
                [(to_plate, "plate"), (to_bin, "bin")],
                family="route",
                reach_max=REACH_MAX,
            )
        )

    # comp_route_2 with the PLAN ORDER reversed and nothing else. Explicit anchors undo the swap
    # `groups` would otherwise cause, so both receptacles and both objects stay exactly where
    # comp_route_2 puts them. Isolates whether a completed goal is lost because it sits exposed for
    # the whole of the next subtask, or because the policy re-engages an object it has placed.
    specs.append(
        split(
            "comp_route_2_binfirst",
            [(("orange",), "bin"), (("lemon",), "plate")],
            family="route",
            reach_max=REACH_MAX,
            anchors=[RECEP_ANCHOR_2B, RECEP_ANCHOR_2A],
        )
    )

    # A third rung for revisit, a family the direct instruction handles poorly.
    specs.append(
        revisit("comp_revisit_3", ("lemon", "lime", "banana"), away_anchor=SOURCE_ANCHOR, reach_max=REACH_MAX)
    )

    # Family C-b - the SPLIT family on a plate. A bin paired with a BOWL does not measure routing:
    # both receptacles are dark and deep, the scene reads as one container, and policies confuse
    # them. A plate is visually distinct from the bin, so these rungs test routing itself.
    #
    # Rungs 5, 7 and 8 exist to give the LENGTH axis a third routing family: without them 5 and 7
    # carried only comp_ordered_bin_N and comp_route_N, so a step in the ladder there was a step in
    # two families rather than a property of the horizon. The plate group is capped at the four
    # objects that rest stably on a flat unwalled surface (avocado, banana, lemon, lime); everything
    # rounder rolls off and would score the asset rather than the routing. That cap is also why the
    # family stops at eight: nine would need a fifth plate object, and the only candidates left are
    # the pomegranate, which policies rarely place, and orange2, whose subtask renders identically
    # to the first orange's. comp_route_N carries the
    # ladder at nine and ten.
    for n, (to_plate, to_bin) in (
        (2, (("avocado",), ("orange",))),
        (3, (("avocado", "banana"), ("orange",))),
        (4, (("avocado", "banana"), ("orange", "lime"))),
        (5, (("avocado", "banana"), ("orange", "onion", "lychee"))),
        (6, (("avocado", "banana", "lemon"), ("orange", "lime", "onion"))),
        (7, (("avocado", "banana", "lemon"), ("orange", "lime", "onion", "lychee"))),
        (8, (("avocado", "banana", "lemon", "lime"), ("orange", "onion", "lychee", "pumpkin"))),
    ):
        specs.append(
            split(
                f"comp_split_plate_{n}",
                [(to_plate, "plate"), (to_bin, "bin")],
                family="split_plate",
                reach_max=REACH_MAX,
            )
        )

    # Family G - identical instances (counting): N copies of one object into the bin.
    # TWO RUNGS ONLY, and deliberately. Duplicates is a family the direct instruction already
    # solves about as well as the privileged oracle, so rungs past four would add length to the suite without adding anything an orchestrator could win. A family the
    # flat policy handles reliably serves as a type control, not as a ladder.
    for n, obj in ((2, "orange"), (4, "lime")):
        specs.append(identical(f"comp_identical_{n}", obj, n, reach_max=REACH_MAX))

    return OrderedDict((s.name, s) for s in specs)


SPECS: "OrderedDict[str, CompositionSpec]" = _build_specs()


def advance_legs(latched, holding):
    """Advance a per-object leg chain by one observation, returning the updated latch flags.

    ``latched[k]`` says leg k was completed at some earlier step, ``holding[k]`` that its placement
    holds right now. A leg latches only once every leg before it has, which is what separates a
    relay from two unrelated placements: an object dropped straight into the last receptacle never
    passed through the intermediate one, and one that visits them in the wrong order did not either.
    Latches never clear, since finishing leg k+1 necessarily undoes leg k.

    Operates on anything supporting ``|`` and ``&`` elementwise, so the same rule serves the batched
    boolean tensors the simulator produces and the plain booleans the tests use.
    """
    updated, prior = [], None
    for was, now in zip(latched, holding, strict=True):
        step = (was | now) if prior is None else (was | (now & prior))
        updated.append(step)
        prior = step
    return updated


def chain_complete(latched, holding_final):
    """A leg chain is done when every intermediate leg has latched AND the final placement holds
    NOW: an object knocked back out of its last receptacle has not completed the task."""
    done = holding_final
    for was in latched:
        done = done & was
    return done


def _plan_name(colour: str, noun: str) -> str:
    return f"{colour} {noun}".strip()


def decomp_plan(spec: CompositionSpec) -> Dict[str, object]:
    """The cached plan for a spec, one command per goal in order, naming objects by colour and type.

    The colour separates look-alikes: told a bare noun, a policy can confuse e.g. the lemon and the
    orange. The task's own instruction stays on plain nouns.
    """
    receptacle_type = {r.prim: RECEPTACLE_TYPES[r.type_id] for r in spec.receptacles}

    def command(obj: PlacedObject, receptacle_prim: str) -> str:
        source, target = CATALOG[obj.type_id], receptacle_type[receptacle_prim]
        return subtask(
            Placement(
                _plan_name(source.colour, source.noun),
                verb(target.kind),
                _plan_name(target.colour, target.noun),
            )
        )

    if spec.via_legs:
        legs = [command(o, stop) for o in spec.goal_objects for stop in (*spec.via_legs, o.receptacle)]
        return {"subtasks": legs, "ordered": True, "revisits": True}
    return {
        "subtasks": [command(o, o.receptacle) for o in spec.goal_objects],
        "ordered": bool(spec.ordered),
        "revisits": False,
    }


def decomp_cache_entries() -> Dict[str, object]:
    """{task_class_name: plan} for every composition spec, to merge into decomp_cache.json."""
    return {task_class_name(name): decomp_plan(spec) for name, spec in SPECS.items()}


if __name__ == "__main__":
    for spec in SPECS.values():
        instr = spec.instruction
        objs = ", ".join(f"{o.prim}@({o.x:.2f},{o.y:.2f})" for o in spec.objects)
        print(
            f"{spec.name:22s} n={spec.n:2d} fam={spec.family:11s} "
            f"objs={len(spec.objects):2d} len={spec.episode_length_s:4d}s"
        )
        print(f"    instruction: {instr}")
        print(f"    layout:      {objs}")
