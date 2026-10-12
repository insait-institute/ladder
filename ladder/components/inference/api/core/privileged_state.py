import dataclasses
from typing import Any, Dict, List, Optional

import numpy as np

# Key under which the privileged payload lives in ``observation.metadata``.
METADATA_PRIVILEGED_KEY = "privileged"


@dataclasses.dataclass(frozen=True)
class PrivilegedObject:
    """A scene object's ground-truth geometry in the canonical base frame."""

    name: str
    center_m: np.ndarray  # [3] object center (canonical frame)
    aabb_min_m: np.ndarray  # [3] axis-aligned bounding-box min corner
    aabb_max_m: np.ndarray  # [3] axis-aligned bounding-box max corner
    # Name as the instructions refer to it, for grounding a free-text subtask.
    display_name: str = ""
    grasped: bool = False  # privileged: is THIS object currently grasped
    # Direction of the object's longest horizontal extent (radians about base Z, modulo pi), where the
    # server publishes it. The AABB cannot say it for an object lying diagonally.
    long_axis_yaw_rad: Optional[float] = None

    @property
    def top_z_m(self) -> float:
        """Z of the object's top surface (for placing another object on it)."""
        return float(self.aabb_max_m[2])

    @property
    def half_extent_m(self) -> np.ndarray:
        return 0.5 * (self.aabb_max_m - self.aabb_min_m)

    @property
    def height_m(self) -> float:
        return float(self.aabb_max_m[2] - self.aabb_min_m[2])


@dataclasses.dataclass(frozen=True)
class PrivilegedCondition:
    """One success condition with its target/goal resolved to ground truth.

    A canonical (single-condition) task has exactly one of these; a long-horizon
    task has one per sub-goal, evaluated in order against ``satisfied``.
    """

    predicate: str  # e.g. "pick_and_place", "stack", "reach", "pickup"
    relation: Optional[str]  # "on" | "in" | "left" | ... | None
    target: PrivilegedObject
    # Where the target's centre must end up, base frame. None for reach/grasp/pickup.
    goal_point_m: Optional[np.ndarray]
    grasped: bool  # privileged: is the target currently grasped
    satisfied: bool  # privileged: does this condition currently hold
    # Lowest release height clearing the destination's walls; above the rim for a walled one,
    # so the object falls in rather than the gripper reaching down inside.
    release_z_m: Optional[float] = None
    # The scene object the target must end up on or in, by name; None for a goal stated as a point.
    destination: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class PrivilegedState:
    conditions: List[PrivilegedCondition]
    # Every scene object, for grounding a subtask in ``subtask`` mode. Empty when structured.
    objects: List[PrivilegedObject] = dataclasses.field(default_factory=list)

    def first_unsatisfied(self) -> Optional[PrivilegedCondition]:
        """The first condition not yet satisfied (drives sequential execution)."""
        for condition in self.conditions:
            if not condition.satisfied:
                return condition
        return None

    def find_object(self, phrase: str) -> Optional[PrivilegedObject]:
        """Resolve a noun phrase from a subtask to a scene object by display name.

        Prefers an exact display-name match, then the longest display name contained in the phrase
        (so "green block" beats "block"). Returns ``None`` if nothing matches.
        """
        text = phrase.strip().lower()
        exact = [o for o in self.objects if o.display_name and o.display_name.lower() == text]
        if exact:
            return exact[0]
        contained = [o for o in self.objects if o.display_name and o.display_name.lower() in text]
        if contained:
            return max(contained, key=lambda o: len(o.display_name))
        return None


def _arr(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _object_from_dict(data: Dict[str, Any]) -> PrivilegedObject:
    return PrivilegedObject(
        name=str(data["name"]),
        center_m=_arr(data["center_m"]),
        aabb_min_m=_arr(data["aabb_min_m"]),
        aabb_max_m=_arr(data["aabb_max_m"]),
        display_name=str(data.get("display_name", "")),
        grasped=bool(data.get("grasped", False)),
        long_axis_yaw_rad=(
            float(data["long_axis_yaw_rad"]) if data.get("long_axis_yaw_rad") is not None else None
        ),
    )


def from_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[PrivilegedState]:
    """Parse the privileged payload from observation metadata, or ``None`` if absent.

    ``None`` lets the policy fail loudly when privileged state was expected but the
    server did not emit it (a misconfiguration), rather than silently no-op.
    """
    if not metadata:
        return None
    payload = metadata.get(METADATA_PRIVILEGED_KEY)
    if not payload:
        return None
    conditions: List[PrivilegedCondition] = []
    for raw in payload["conditions"]:
        goal = raw.get("goal_point_m")
        conditions.append(
            PrivilegedCondition(
                predicate=str(raw["predicate"]),
                relation=(str(raw["relation"]) if raw.get("relation") is not None else None),
                target=_object_from_dict(raw["target"]),
                goal_point_m=(_arr(goal) if goal is not None else None),
                release_z_m=(float(raw["release_z_m"]) if raw.get("release_z_m") is not None else None),
                destination=(str(raw["destination"]) if raw.get("destination") else None),
                grasped=bool(raw["grasped"]),
                satisfied=bool(raw["satisfied"]),
            )
        )
    objects = [_object_from_dict(o) for o in payload.get("objects", [])]
    return PrivilegedState(conditions=conditions, objects=objects)


def encode_object(
    name: str,
    center_m: np.ndarray,
    aabb_min_m: np.ndarray,
    aabb_max_m: np.ndarray,
    *,
    display_name: str = "",
    grasped: bool = False,
    long_axis_yaw_rad: Optional[float] = None,
) -> Dict[str, Any]:
    """Build the msgpack-safe dict for one object (poses in canonical frame)."""
    encoded = {
        "name": str(name),
        "center_m": np.asarray(center_m, dtype=np.float32).reshape(3),
        "aabb_min_m": np.asarray(aabb_min_m, dtype=np.float32).reshape(3),
        "aabb_max_m": np.asarray(aabb_max_m, dtype=np.float32).reshape(3),
        "display_name": str(display_name),
        "grasped": bool(grasped),
    }
    if long_axis_yaw_rad is not None and np.isfinite(long_axis_yaw_rad):
        encoded["long_axis_yaw_rad"] = float(long_axis_yaw_rad)
    return encoded


def encode_condition(
    predicate: str,
    target: Dict[str, Any],
    *,
    relation: Optional[str] = None,
    goal_point_m: Optional[np.ndarray] = None,
    release_z_m: Optional[float] = None,
    destination: Optional[str] = None,
    grasped: bool = False,
    satisfied: bool = False,
) -> Dict[str, Any]:
    """Build the msgpack-safe dict for one success condition."""
    return {
        "predicate": str(predicate),
        "relation": (str(relation) if relation is not None else None),
        "target": target,
        "goal_point_m": (
            None if goal_point_m is None else np.asarray(goal_point_m, dtype=np.float32).reshape(3)
        ),
        "release_z_m": (None if release_z_m is None else float(release_z_m)),
        "destination": (str(destination) if destination is not None else None),
        "grasped": bool(grasped),
        "satisfied": bool(satisfied),
    }


def encode_state(
    conditions: List[Dict[str, Any]], objects: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Build the full privileged payload for ``metadata[METADATA_PRIVILEGED_KEY]``."""
    return {"conditions": list(conditions), "objects": list(objects or [])}
