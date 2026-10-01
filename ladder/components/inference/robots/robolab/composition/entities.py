import dataclasses
from typing import Optional


@dataclasses.dataclass(frozen=True)
class PlacedObject:
    """A graspable instance: a unique prim name, its catalog type, its authored (x, y), and the
    receptacle prim it must end up on/in (``None`` for a distractor that must be left behind).
    ``start_on`` names a receptacle the object is spawned resting on/in at episode start (transfer
    tasks); ``None`` means it starts on the bare table."""

    prim: str
    _: dataclasses.KW_ONLY  # the rest are keyword-only, so a construction site names what it sets
    type_id: str
    x: float
    y: float
    receptacle: Optional[str] = None
    start_on: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class PlacedReceptacle:
    prim: str
    _: dataclasses.KW_ONLY
    type_id: str
    x: float
    y: float


@dataclasses.dataclass(frozen=True)
class Goal:
    obj_id: str
    receptacle: str
    noun: str


@dataclasses.dataclass(frozen=True)
class Receptacle:
    obj_id: str
    kind: str
    noun: str
