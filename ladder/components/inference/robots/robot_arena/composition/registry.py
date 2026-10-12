import dataclasses
from typing import Dict, Optional, Sequence, Tuple

from ladder.components.inference.robots.robot_arena.composition.tasks import (
    CompositionTask,
    ImportedObject,
    Leg,
)

_BANANA = ImportedObject("scene70", "yellow banana_target", (0.30, -0.115, 0.06))
_CHEESE = ImportedObject("scene75", "cheese block_target", (0.40, -0.115, 0.06))
_MUSHROOM = ImportedObject("scene76", "mushroom toy", (0.34, -0.03, 0.06))
_CORN_COB = ImportedObject("scene36", "toy corn cob", (0.24, -0.17, 0.06))

_SCENE45_MOVERS: Dict[str, Tuple[float, float, float]] = {
    "sushi toy": (0.315, -0.175, 0.02),
    "blue bowl_target": (0.24, -0.035, 0.02),
}
_SCENE45_LAYOUT: Dict[str, Tuple[float, float, float]] = {
    **_SCENE45_MOVERS,
    "small can": (0.45, -0.17, 0.02),
}
_SCENE76_LAYOUT: Dict[str, Tuple[float, float, float]] = {
    "metal pot_target": (0.335, -0.025, 0.02),
    "mushroom toy": (0.23, -0.03, 0.02),
}
_SCENE54_LAYOUT: Dict[str, Tuple[float, float, float]] = {
    "spoon_target": (0.275, 0.02, 0.02),
    "toy chicken drumstick": (0.28, 0.165, 0.02),
    "pot": (0.405, -0.025, 0.02),
}
# Restrictions of the layouts above, so a task that drops an object stands the rest where its
# siblings stand them.
_SCENE45_BOWL = {name: _SCENE45_MOVERS[name] for name in ("blue bowl_target",)}
_SCENE76_POT = {name: _SCENE76_LAYOUT[name] for name in ("metal pot_target",)}
_SCENE54_SPOON = {name: _SCENE54_LAYOUT[name] for name in ("spoon_target",)}
_SCENE54_MOVERS = {name: _SCENE54_LAYOUT[name] for name in ("spoon_target", "toy chicken drumstick")}
_SCENE54_SPOON_POT = {name: _SCENE54_LAYOUT[name] for name in ("spoon_target", "pot")}


_SCENE45_LADDER = ("blue bowl", "sushi toy", "banana", "cheese", "mushroom", "corn cob")
_SCENE45_IMPORTS = {"banana": _BANANA, "cheese": _CHEESE, "mushroom": _MUSHROOM, "corn cob": _CORN_COB}


def _scene45_rung(n: int, *, excluded: Sequence[str], drop_can: bool) -> CompositionTask:
    """Rung ``n`` of an invariantly worded scene45 ladder."""
    return CompositionTask(
        "scene45",
        sources=list(_SCENE45_LADDER[:n]),
        destination="purple cloth",
        extra_objects=[_SCENE45_IMPORTS[source] for source in _SCENE45_LADDER[2:n]],
        scale_overrides={"purple cloth": 1.3 if n <= 3 else 1.4},
        position_overrides=_SCENE45_MOVERS if drop_can else _SCENE45_LAYOUT,
        fixed_overrides={"purple cloth": True},
        remove_objects=["small can"] if drop_can else [],
        excluded=list(excluded),
    )


_SCENE54_CHAIN_MOVERS = ("spoon", "chicken drumstick", "corn cob")
_SCENE54_CHAIN_LAYOUT_3: Dict[str, Tuple[float, float, float]] = {
    "spoon_target": (0.255, 0.115, 0.02),
    "toy chicken drumstick": (0.255, -0.030, 0.02),
    "pot": (0.405, -0.025, 0.02),
}
_SCENE54_CORN_COB = dataclasses.replace(_CORN_COB, position=(0.265, -0.125, 0.06))


def _scene54_chain(
    movers: Tuple[str, ...],
    route: Tuple[Tuple[str, str], ...],
    *,
    cloth_scale: float = 1.0,
    layout: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> CompositionTask:
    """A relay or revisit over ``movers``, each completing ``route`` before the next starts."""
    return CompositionTask(
        "scene54",
        legs=[Leg(mover, dest, relation) for mover in movers for dest, relation in route],
        milestones=True,
        extra_objects=[_SCENE54_CORN_COB] if "corn cob" in movers else [],
        scale_overrides={"cloth_destination": cloth_scale} if cloth_scale != 1.0 else {},
        position_overrides=layout or _SCENE54_LAYOUT,
        fixed_overrides={"pot": True},
    )


_PINGPONG_STOPS = (("cloth", "on"), ("pot", "in"))


def _scene54_pingpong(legs: int) -> CompositionTask:
    """A ``legs``-long out-and-back over the spoon, alternating the cloth and the pot."""
    stops = [_PINGPONG_STOPS[i % 2] for i in range(legs)]
    return CompositionTask(
        "scene54",
        legs=[Leg("spoon", destination, relation) for destination, relation in stops],
        milestones=True,
        position_overrides=_SCENE54_SPOON_POT,
        fixed_overrides={"pot": True},
        remove_objects=["toy chicken drumstick"],
    )


_RELAY_ROUTE = (("cloth", "on"), ("pot", "in"))
_REVISIT_ROUTE = (("cloth", "on"), ("pot", "in"), ("cloth", "on"))


_SCENE54_ROUTE_LADDER = ("spoon", "chicken drumstick", "corn cob", "cheese")
_SCENE54_ROUTE_IMPORTS = {"corn cob": _CORN_COB, "cheese": _CHEESE}
_SCENE54_ROUTE_NATIVE = {"spoon": "spoon_target", "chicken drumstick": "toy chicken drumstick"}
_SCENE54_ROUTE_SLOTS = (
    (0.255, 0.115),
    (0.255, -0.030),
    (0.250, -0.125),
    (0.315, -0.120),
)
_ROUTE_VIA_POT = (("pot", "in"), ("cloth", "on"))


def _scene54_route(n: int, direct: int = 0) -> CompositionTask:
    """Route the first ``n`` movers through the pot onto the cloth; the last ``direct`` go straight."""
    movers = _SCENE54_ROUTE_LADDER[:n]
    routed, straight = movers[: n - direct], movers[n - direct :]
    return CompositionTask(
        "scene54",
        legs=(
            [Leg(mover, dest, relation) for mover in routed for dest, relation in _ROUTE_VIA_POT]
            + [Leg(mover, "cloth") for mover in straight]
        ),
        milestones=True,
        extra_objects=[
            dataclasses.replace(
                _SCENE54_ROUTE_IMPORTS[mover],
                position=(*_SCENE54_ROUTE_SLOTS[i], 0.06),
            )
            for i, mover in enumerate(movers)
            if mover in _SCENE54_ROUTE_IMPORTS
        ],
        scale_overrides={"cloth_destination": 1.3 if n <= 3 else 1.4},
        position_overrides={
            **{
                name: (*_SCENE54_ROUTE_SLOTS[i], 0.02)
                for i, (mover, name) in enumerate(_SCENE54_ROUTE_NATIVE.items())
                if mover in movers
            },
            "pot": (0.405, -0.025, 0.02),
        },
        fixed_overrides={"pot": True},
        remove_objects=[name for mover, name in _SCENE54_ROUTE_NATIVE.items() if mover not in movers],
    )


# Only the except_* families leave an object unnamed by the instruction, and they name it in
# `excluded`. Every other task lists the natives it does not use in `remove_objects`.
COMPOSITION_TASKS: Dict[str, CompositionTask] = {
    "scene36_1goal": CompositionTask(
        "scene36",
        sources=["fork"],
        destination="pan",
        relation="in",
        position_overrides={
            "toy pan": (0.295, 0.06, 0.02),
            "toy fork_target": (0.445, 0.055, 0.02),
        },
        remove_objects=["toy corn cob"],
    ),
    "scene45_1goal": CompositionTask(
        "scene45",
        sources=["blue bowl"],
        destination="purple cloth",
        position_overrides=_SCENE45_BOWL,
        remove_objects=["sushi toy", "small can"],
    ),
    "scene54_1goal": CompositionTask(
        "scene54",
        sources=["spoon"],
        destination="cloth",
        scale_overrides={"cloth_destination": 1.15},
        position_overrides=_SCENE54_SPOON,
        remove_objects=["pot", "toy chicken drumstick"],
    ),
    "scene76_1goal": CompositionTask(
        "scene76",
        sources=["metal pot"],
        destination="yellow cloth",
        position_overrides=_SCENE76_POT,
        remove_objects=["mushroom toy"],
    ),
    "scene36_2goal": CompositionTask(
        "scene36",
        sources=["fork", "corn cob"],
        destination="pan",
        relation="in",
        position_overrides={
            "toy pan": (0.295, 0.06, 0.02),
            "toy fork_target": (0.445, 0.055, 0.02),
            "toy corn cob": (0.385, 0.125, 0.02),
        },
    ),
    "scene45_2goal": CompositionTask(
        "scene45",
        sources=["blue bowl", "sushi toy"],
        destination="purple cloth",
        position_overrides={"sushi toy": (0.41, 0.06, 0.02)},
        remove_objects=["small can"],
    ),
    "scene54_2goal": CompositionTask(
        "scene54",
        sources=["spoon", "chicken drumstick"],
        destination="cloth",
        scale_overrides={"cloth_destination": 1.15},
        position_overrides=_SCENE54_MOVERS,
        remove_objects=["pot"],
    ),
    "scene76_2goal": CompositionTask(
        "scene76",
        sources=["metal pot", "mushroom toy"],
        destination="yellow cloth",
        position_overrides=_SCENE76_LAYOUT,
    ),
    "scene45_3goal": CompositionTask(
        "scene45",
        sources=["blue bowl", "sushi toy", "banana"],
        destination="purple cloth",
        extra_objects=[_BANANA],
        scale_overrides={"purple cloth": 1.3},
        position_overrides=_SCENE45_MOVERS,
        fixed_overrides={"purple cloth": True},
        remove_objects=["small can"],
    ),
    "scene50_3goal": CompositionTask(
        "scene50",
        sources=["stuffed animal", "blue block", "cheese"],
        destination="pot",
        relation="in",
        extra_objects=[dataclasses.replace(_CHEESE, position=(0.47, -0.10, 0.06))],
        scale_overrides={"metal pot_destination": 1.6},
        position_overrides={
            "metal pot_destination": (0.34, 0.145, 0.02),
            "stuffed animal_target": (0.245, -0.115, 0.02),
            "blue block": (0.34, -0.05, 0.02),
        },
        remove_objects=["metal lid", "toy hammer"],
    ),
    "scene54_3goal": CompositionTask(
        "scene54",
        sources=["spoon", "chicken drumstick", "pot"],
        destination="cloth",
        scale_overrides={"cloth_destination": 1.5},
        position_overrides={
            "spoon_target": (0.275, 0.11, 0.02),
            "toy chicken drumstick": (0.28, -0.02, 0.02),
            "pot": (0.39, -0.035, 0.02),
        },
    ),
    "scene45_4goal": CompositionTask(
        "scene45",
        sources=["blue bowl", "sushi toy", "cheese", "banana"],
        destination="purple cloth",
        extra_objects=[_BANANA, _CHEESE],
        scale_overrides={"purple cloth": 1.4},
        position_overrides=_SCENE45_MOVERS,
        fixed_overrides={"purple cloth": True},
        remove_objects=["small can"],
    ),
    "scene45_5goal": CompositionTask(
        "scene45",
        sources=["blue bowl", "sushi toy", "cheese", "banana", "mushroom"],
        destination="purple cloth",
        extra_objects=[_BANANA, _CHEESE, _MUSHROOM],
        scale_overrides={"purple cloth": 1.4},
        position_overrides=_SCENE45_MOVERS,
        fixed_overrides={"purple cloth": True},
        remove_objects=["small can"],
    ),
    "scene45_6goal": CompositionTask(
        "scene45",
        sources=["blue bowl", "sushi toy", "cheese", "banana", "mushroom", "corn cob"],
        destination="purple cloth",
        extra_objects=[_BANANA, _CHEESE, _MUSHROOM, _CORN_COB],
        scale_overrides={"purple cloth": 1.4},
        position_overrides=_SCENE45_MOVERS,
        fixed_overrides={"purple cloth": True},
        remove_objects=["small can"],
    ),
    "scene76_2goal_ordered": CompositionTask(
        "scene76",
        sources=["metal pot", "mushroom toy"],
        destination="yellow cloth",
        ordered=True,
        position_overrides=_SCENE76_LAYOUT,
    ),
    "scene54_2goal_ordered": CompositionTask(
        "scene54",
        sources=["chicken drumstick", "spoon"],
        destination="cloth",
        scale_overrides={"cloth_destination": 1.15},
        position_overrides=_SCENE54_MOVERS,
        ordered=True,
        remove_objects=["pot"],
    ),
    "scene54_except": CompositionTask(
        "scene54",
        sources=["spoon", "chicken drumstick"],
        destination="cloth",
        scale_overrides={"cloth_destination": 1.15},
        excluded=["pot"],
        position_overrides=_SCENE54_LAYOUT,
    ),
    "scene54_except_3": CompositionTask(
        "scene54",
        sources=["spoon", "chicken drumstick", "corn cob"],
        destination="cloth",
        extra_objects=[_SCENE54_CORN_COB],
        scale_overrides={"cloth_destination": 1.3},
        excluded=["pot"],
        position_overrides=_SCENE54_CHAIN_LAYOUT_3,
    ),
    "scene54_split_2": CompositionTask(
        "scene54",
        legs=[Leg("spoon", "cloth"), Leg("chicken drumstick", "pot", relation="in")],
        position_overrides=_SCENE54_LAYOUT,
        fixed_overrides={"pot": True},
    ),
    "scene54_transfer_2": CompositionTask(
        "scene54",
        sources=["spoon", "chicken drumstick"],
        destination="pot",
        relation="in",
        start_on="cloth",
        position_overrides=_SCENE54_LAYOUT,
        fixed_overrides={"pot": True},
    ),
    "scene54_relay_2": CompositionTask(
        "scene54",
        legs=[Leg("spoon", "cloth"), Leg("spoon", "pot", relation="in")],
        milestones=True,
        position_overrides=_SCENE54_SPOON_POT,
        fixed_overrides={"pot": True},
        remove_objects=["toy chicken drumstick"],
    ),
    "scene54_revisit_3": _scene54_pingpong(3),
    "scene54_revisit_7": _scene54_pingpong(7),
    "scene54_revisit_8": _scene54_pingpong(8),
    "scene54_revisit_10": _scene54_pingpong(10),
    "scene54_route_4": _scene54_route(2),
    "scene54_route_6": _scene54_route(3),
    "scene54_route_8": _scene54_route(4),
    "scene54_route_7": _scene54_route(4, direct=1),
    "scene54_route_2": _scene54_route(1),
    "scene54_split_3": CompositionTask(
        "scene54",
        legs=[
            Leg("spoon", "cloth"),
            Leg("corn cob", "cloth"),
            Leg("chicken drumstick", "pot", relation="in"),
        ],
        extra_objects=[_SCENE54_CORN_COB],
        scale_overrides={"cloth_destination": 1.3},
        position_overrides=_SCENE54_CHAIN_LAYOUT_3,
        fixed_overrides={"pot": True},
    ),
    "scene54_3goal_ordered": CompositionTask(
        "scene54",
        sources=["corn cob", "spoon", "chicken drumstick"],
        destination="cloth",
        extra_objects=[_SCENE54_CORN_COB],
        scale_overrides={"cloth_destination": 1.5},
        ordered=True,
        position_overrides={
            "spoon_target": (0.275, 0.11, 0.02),
            "toy chicken drumstick": (0.28, -0.02, 0.02),
        },
        remove_objects=["pot"],
    ),
    "scene54_duplicates_3": CompositionTask(
        "scene54",
        legs=[Leg("corn cob", "cloth")] * 3,
        extra_objects=[
            dataclasses.replace(_CORN_COB, position=(0.300, -0.075, 0.06)),
            dataclasses.replace(_CORN_COB, position=(0.300, -0.145, 0.06), alias="toy corn cob copy"),
            dataclasses.replace(_CORN_COB, position=(0.240, -0.150, 0.06), alias="toy corn cob copy 2"),
        ],
        scale_overrides={"cloth_destination": 1.5},
        remove_objects=["spoon_target", "toy chicken drumstick", "pot"],
    ),
    "scene54_relay_4": _scene54_chain(_SCENE54_CHAIN_MOVERS[:2], _RELAY_ROUTE),
    "scene54_revisit_6": _scene54_chain(_SCENE54_CHAIN_MOVERS[:2], _REVISIT_ROUTE, cloth_scale=1.15),
    "scene54_revisit_9": _scene54_chain(
        _SCENE54_CHAIN_MOVERS,
        _REVISIT_ROUTE,
        cloth_scale=1.3,
        layout=_SCENE54_CHAIN_LAYOUT_3,
    ),
    "scene54_duplicates_2": CompositionTask(
        "scene54",
        legs=[Leg("corn cob", "cloth"), Leg("corn cob", "cloth")],
        extra_objects=[
            dataclasses.replace(_CORN_COB, position=(0.27, -0.095, 0.06)),
            dataclasses.replace(_CORN_COB, position=(0.33, -0.100, 0.06), alias="toy corn cob copy"),
        ],
        remove_objects=["spoon_target", "toy chicken drumstick", "pot"],
    ),
    **{f"scene45_all_{n}": _scene45_rung(n, excluded=(), drop_can=True) for n in range(2, 7)},
    **{f"scene45_except_{n}": _scene45_rung(n, excluded=("small can",), drop_can=False) for n in range(2, 7)},
}
