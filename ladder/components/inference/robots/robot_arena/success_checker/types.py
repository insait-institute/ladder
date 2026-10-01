"""Data structures for the binary success checker."""

import dataclasses
import enum
from typing import Optional, Protocol, Tuple, TypedDict, runtime_checkable

import numpy as np


@runtime_checkable
class SimEntity(Protocol):
    """Structural type for Genesis rigid entities used by the success checker.

    Any object implementing get_pos() and get_AABB() satisfies this protocol,
    including real Genesis RigidEntity instances and test mocks.

    Note: Genesis returns torch.Tensor on GPU backend, numpy.ndarray on CPU.
    The type hints here specify numpy as the canonical type; callers should use
    the helper functions in predicates.py (_get_entity_pos, _get_entity_aabb)
    which handle torch->numpy conversion.
    """

    def get_pos(self) -> np.ndarray:
        ...

    def get_AABB(self) -> Tuple[np.ndarray, np.ndarray]:  # noqa: N802
        ...


class TaskType(enum.Enum):
    PICK_AND_PLACE = "pick_and_place"
    STACK = "stack"
    MOVE_RELATIVE = "move_relative"
    MOVE_TO_REGION = "move_to_region"
    PICKUP = "pickup"
    BETWEEN = "between"
    PUSH = "push"
    REACH = "reach"
    GRASP = "grasp"


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    """Structured representation of a manipulation task parsed from a text instruction.

    Fields:
        task_type: The type of manipulation task.
        target_ref: Text reference to the object being manipulated.
        destination_ref: Text reference to the destination/reference object.
            For PICK_AND_PLACE/STACK: the object to place on/in.
            For BETWEEN: the first of two reference objects.
            For MOVE_RELATIVE: the reference object for spatial relation.
        between_anchor_ref: Text reference to the second anchor object (BETWEEN tasks only).
            E.g., "put X between Y and Z" → destination_ref=Y, between_anchor_ref=Z.
        spatial_relation: Relation like "on", "in", "left", "right", etc.
        table_region: Target region like "upper_left", "center", etc.
    """

    task_type: TaskType
    target_ref: str
    destination_ref: Optional[str] = None
    between_anchor_ref: Optional[str] = None
    spatial_relation: Optional[str] = None
    table_region: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class SuccessThresholds:
    """Numeric thresholds for geometric success predicates (all in meters)."""

    # "on top" band: target center must be at/above the destination top (not below
    # it), with a tight xy margin so a neighbouring object isn't counted as placed.
    on_top_height_min_m: float = -0.005
    # Upper bound keeps flat + tall-resting placements while rejecting
    # standing-on-end / hovering-held (center well above the destination top).
    on_top_height_max_m: float = 0.06
    on_top_xy_margin_m: float = 0.015
    # Fraction of the target's XY footprint that must overlap the destination's
    # for "on top" / "stack".
    on_top_overlap_min_frac: float = 0.47
    inside_z_tolerance_m: float = 0.03
    pickup_height_m: float = 0.05
    # PICKUP success: a grasped object lifted this far off its rest counts as
    # picked up. Lower than pickup_height_m because a loosely-held tall object is
    # often lifted only 3-4 cm.
    pickup_lift_m: float = 0.03
    relative_margin_m: float = 0.02
    displacement_min_m: float = 0.03
    between_tolerance_m: float = 0.05
    reach_distance_m: float = 0.08
    region_padding_m: float = 0.05
    # "left/right of X" placements are accepted within this distance of the reference.
    relative_max_distance_m: float = 0.30
    region_max_distance_m: float = 0.15


class CheckResult(TypedDict):
    """Result of a single success evaluation step."""

    done: bool
    success_status: str
    task_type: str


@dataclasses.dataclass(frozen=True)
class ResolvedTask:
    """A TaskSpec with object references resolved to entity indices.

    Fields:
        task_spec: The original parsed task specification.
        target_index: Index of the object being manipulated.
        destination_index: Index of the destination/reference object.
            Corresponds to TaskSpec.destination_ref.
        between_anchor_index: Index of the second anchor object (BETWEEN only).
            Corresponds to TaskSpec.between_anchor_ref.
    """

    task_spec: TaskSpec
    target_index: int
    destination_index: Optional[int] = None
    between_anchor_index: Optional[int] = None
