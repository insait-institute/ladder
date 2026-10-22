import dataclasses
from typing import Dict, Tuple

# How an object satisfies its goal on a receptacle.
IN = "in"  # object_in_container: bins, bowls, boxes
ON = "on"  # object_on_top: plates and other flat surfaces


def verb(kind: str) -> str:
    """The preposition a receptacle of this kind takes."""
    return ON if kind == ON else IN


@dataclasses.dataclass(frozen=True)
class ObjectType:
    """A graspable catalog object. ``usd`` is relative to the RoboLab ASSET_DIR (the factory
    joins it); ``rest_z`` is the object origin's resting height above the table top (~half the
    bbox height) so it spawns sitting on the surface; ``radius`` is the footprint bounding radius
    (half the larger XY bbox dim, from the sim) used to keep it clear of receptacles and edges after
    jitter. ``category`` drives collective phrasing ("all the fruit") and goal-vs-distractor choice."""

    usd: str
    noun: str
    rest_z: float
    radius: float
    category: str  # fruit | vegetable | block
    # Visual attribute a plan names the object by ("the yellow lemon"); empty when the noun alone
    # is what a viewer would say (an orange, a red block). The reference plans render it.
    colour: str = ""


@dataclasses.dataclass(frozen=True)
class ReceptacleType:
    usd: str
    kind: str  # IN | ON
    noun: str
    rest_z: float
    footprint: Tuple[float, float]  # (half_x, half_y) top-surface extent (m), from the sim bbox
    colour: str = ""


# Graspable objects. The DROID VLA is grasp-limited to produce-like objects, so the pool is
# fruit/veg; blocks are decoys only. ``radius`` / receptacle ``footprint`` are measured from the sim
# bbox (usd_utils.get_dimensions), so the placement clearances match the real geometry.
# Every asset here is verified single-rigid-body AND confirmed to render (objaverse/apple_01 is
# excluded -- it spawns invisible and its nested collision shapes break the per-object contact filter).
CATALOG: Dict[str, ObjectType] = {
    "banana": ObjectType("objects/ycb/banana.usd", "banana", 0.020, 0.089, "fruit", colour="yellow"),
    "orange": ObjectType("objects/fruits_veggies/orange1.usd", "orange", 0.042, 0.036, "fruit"),
    "lemon": ObjectType("objects/fruits_veggies/lemon1.usd", "lemon", 0.026, 0.038, "fruit", colour="yellow"),
    "lime": ObjectType("objects/fruits_veggies/lime.usd", "lime", 0.030, 0.038, "fruit", colour="green"),
    "avocado": ObjectType("objects/fruits_veggies/avocado.usd", "avocado", 0.046, 0.031, "fruit"),
    "pomegranate": ObjectType(
        "objects/fruits_veggies/pomegranate.usd", "pomegranate", 0.058, 0.058, "fruit", colour="red"
    ),
    "lychee": ObjectType("objects/fruits_veggies/lychee.usd", "lychee", 0.025, 0.021, "fruit"),
    "orange2": ObjectType("objects/fruits_veggies/orange2.usd", "orange", 0.037, 0.036, "fruit"),
    "onion": ObjectType(
        "objects/fruits_veggies/red_onion.usd", "onion", 0.045, 0.030, "vegetable", colour="red"
    ),
    "pumpkin": ObjectType(
        "objects/vomp/pumpkinsmall/pumpkinsmall.usd", "pumpkin", 0.038, 0.038, "vegetable", colour="orange"
    ),
    "red_block": ObjectType("objects/basic/red_block.usd", "red block", 0.024, 0.030, "block"),
    "blue_block": ObjectType("objects/basic/blue_block.usd", "blue block", 0.024, 0.030, "block"),
}

RECEPTACLE_TYPES: Dict[str, ReceptacleType] = {
    # A roomy bin (30x20cm) and a wide 30cm bowl -- both hold a long banana and several fruits.
    # The flat plate is the transfer/relay source (objects resting ON it stay easiest to grasp out).
    "bin": ReceptacleType(
        "objects/vomp/bin_a05/bin_a05.usd", IN, "bin", 0.080, (0.149, 0.100), colour="grey"
    ),
    "bowl": ReceptacleType(
        "objects/vomp/blackandbrassbowl_large/blackandbrassbowl_large.usd",
        IN,
        "bowl",
        0.075,
        (0.151, 0.151),
        colour="black",
    ),
    "plate": ReceptacleType(
        "objects/vomp/plate_large/plate_large.usd", ON, "plate", 0.013, (0.166, 0.166), colour="white"
    ),
}

# Ladder pool, best-grasp-first. The N-goal single-destination tasks take the first N. Nine distinct
# produce types plus a second orange for the tenth rung (so N=10 has a "both oranges" duplicate, like
# RoboLab's own multi-banana tasks).
FRUIT_LADDER: Tuple[str, ...] = (
    "banana",
    "orange",
    "lemon",
    "lime",
    "avocado",
    "onion",
    "lychee",
    "pumpkin",
    "orange2",
    # Last, because it is the one object that policies consistently fail to place. Earlier in the
    # order it would cap every longer rung by an asset rather than by the horizon the suite exists
    # to measure. The rest of the order is deliberately not ranked by any one model's per-object
    # success rates.
    "pomegranate",
)

# ------------------------------ workspace layout ------------------------------
# Franka base at world origin, +x forward, table top at z=0. Graspables scatter over a reachable band;
# receptacles are PINNED at the deep back (the reset randomizes only the free table objects). Authored
# positions are the per-episode means; the reset jitters each free object in x/y + yaw (see factory).
# _TABLE is the measured table_oak top extent (from the scene USD); its front ~0.18 m is unreachable
# behind the mount. Every placement is checked so that, at the WORST jitter+footprint, an object never
# leaves the table and never enters a receptacle's rectangle -- see _scatter.
