"""Object name matching: resolve text references to scene entity indices."""

import re
from typing import List, Optional

from ladder.components.inference.robots.robot_arena.scene_loader import ObjectConfig
from ladder.components.inference.robots.robot_arena.success_checker.types import (
    ResolvedTask,
    TaskSpec,
    TaskType,
)
from ladder.core.common.logger.logger import get_logger

LOGGER = get_logger(__name__)


def _normalize_name(name: str) -> str:
    """Lowercase and strip common prefixes/suffixes for matching."""
    name = name.lower().strip()
    # Remove leading articles.
    name = re.sub(r"^(a|an|the)\s+", "", name)
    return name


def _strip_role_suffix(name: str) -> str:
    """Remove _target / _destination suffixes."""
    return re.sub(r"_(target|destination)\s*$", "", name, flags=re.IGNORECASE).strip()


def _tokenize(text: str) -> set[str]:
    """Split text into a set of lowercase tokens."""
    return set(re.split(r"[\s_\-]+", text.lower()))


def _match_single_object(  # noqa: C901, PLR0912
    reference: str,
    objects: List[ObjectConfig],
    role: str,
    exclude_indices: Optional[set[int]] = None,
    allow_role_fallback: bool = True,
) -> Optional[int]:
    """Match a text reference to the best-matching object index.

    Uses a 4-tier matching strategy:
    1. Exact match (after normalization)
    2. Suffix-stripped match (_target/_destination removed)
    3. Substring containment
    4. Token overlap (Jaccard similarity)

    Args:
        reference: Text reference from the instruction (e.g. "spoon").
        objects: List of ObjectConfig from the scene.
        role: Either "target" or "destination" — used for disambiguation.
        exclude_indices: Object indices to skip (already matched to another role).

    Returns:
        Index into ``objects`` or None if no match found.
    """
    exclude = exclude_indices or set()
    ref_normalized = _normalize_name(reference)
    ref_tokens = _tokenize(ref_normalized)

    # Tier 1: Exact match (normalized).
    for i, obj in enumerate(objects):
        if i in exclude:
            continue
        if _normalize_name(obj.name) == ref_normalized:
            return i

    # Tier 2: Suffix-stripped exact match.
    suffix_matches: List[int] = []
    for i, obj in enumerate(objects):
        if i in exclude:
            continue
        stripped = _strip_role_suffix(_normalize_name(obj.name))
        if stripped == ref_normalized:
            suffix_matches.append(i)

    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        preferred_suffix = "_target" if role == "target" else "_destination"
        for i in suffix_matches:
            if preferred_suffix in objects[i].name.lower():
                return i
        return suffix_matches[0]

    # Tier 3: Substring containment (reference is substring of object name or vice versa).
    substring_matches: List[int] = []
    for i, obj in enumerate(objects):
        if i in exclude:
            continue
        obj_normalized = _normalize_name(obj.name)
        obj_stripped = _strip_role_suffix(obj_normalized)
        if ref_normalized in obj_stripped or obj_stripped in ref_normalized:
            substring_matches.append(i)

    if len(substring_matches) == 1:
        return substring_matches[0]
    if len(substring_matches) > 1:
        # Disambiguate: prefer objects with the matching role suffix.
        preferred_suffix = "_target" if role == "target" else "_destination"
        for i in substring_matches:
            if preferred_suffix in objects[i].name.lower():
                return i
        # Fall back to first match.
        return substring_matches[0]

    # Tier 4: Token overlap — fraction of reference tokens found in object name.
    # This is asymmetric (not Jaccard) so short references like "yellow knife"
    # can match long object names like "toy knife with yellow handle and white blade".
    best_score = 0.0
    best_index: Optional[int] = None
    min_coverage_threshold = 0.5
    for i, obj in enumerate(objects):
        if i in exclude:
            continue
        obj_tokens = _tokenize(_strip_role_suffix(_normalize_name(obj.name)))
        if not ref_tokens:
            continue
        coverage = len(ref_tokens & obj_tokens) / len(ref_tokens)
        if coverage > best_score:
            best_score = coverage
            best_index = i

    if best_score >= min_coverage_threshold and best_index is not None:
        return best_index

    # Tier 5: role-suffix fallback. Several source instructions are garbled (mention
    # an object that isn't in the scene), so name matching fails entirely. When the
    # scene explicitly tags exactly one non-excluded object with the matching role
    # (_target / _destination), trust that tag rather than the wording.
    if allow_role_fallback:
        preferred_suffix = "_target" if role == "target" else "_destination"
        role_matches = [
            i for i, obj in enumerate(objects) if i not in exclude and preferred_suffix in obj.name.lower()
        ]
        if len(role_matches) == 1:
            return role_matches[0]

    return None


def match_instances(reference: str, objects: List[ObjectConfig], count: int) -> List[int]:
    """Indices of ``count`` DISTINCT objects that ``reference`` names, best match first.

    A duplicate-object task ("put two spoons on the plate") names one noun for several
    interchangeable objects. Resolving it once and reusing that index would let a single spoon
    satisfy every spoon goal, so each instance is matched with the previous ones excluded.
    Raises when the scene holds fewer instances than the task asks for.
    """
    matched: List[int] = []
    for _ in range(count):
        index = _match_single_object(
            reference, objects, role="target", exclude_indices=set(matched), allow_role_fallback=False
        )
        if index is None:
            raise ValueError(
                f"scene holds {len(matched)} object(s) matching {reference!r}, task needs {count}"
            )
        matched.append(index)
    return matched


def match_objects(
    task_spec: TaskSpec,
    objects: List[ObjectConfig],
) -> Optional[ResolvedTask]:
    """Resolve TaskSpec object references against scene objects.

    Returns None if the target object cannot be matched.
    """
    target_index = _match_single_object(task_spec.target_ref, objects, role="target")

    if target_index is None:
        LOGGER.warn(
            f"Could not match target '{task_spec.target_ref}' to any scene object. "
            f"Available: {[o.name for o in objects]}"
        )
        return None

    # If target matched a fixed object, log a warning (unusual but not fatal).
    if objects[target_index].fixed:
        LOGGER.warn(f"Target '{task_spec.target_ref}' matched fixed object '{objects[target_index].name}'")

    exclude = {target_index}

    # Match destination.
    destination_index: Optional[int] = None
    if task_spec.destination_ref:
        destination_index = _match_single_object(
            task_spec.destination_ref, objects, role="destination", exclude_indices=exclude
        )
        if destination_index is None:
            LOGGER.warn(
                f"Could not match destination '{task_spec.destination_ref}' to any scene object. "
                f"Available: {[o.name for i, o in enumerate(objects) if i not in exclude]}"
            )

    # Match second anchor (for BETWEEN tasks).
    between_anchor_index: Optional[int] = None
    if task_spec.between_anchor_ref:
        exclude_2 = exclude | ({destination_index} if destination_index is not None else set())
        between_anchor_index = _match_single_object(
            task_spec.between_anchor_ref, objects, role="destination", exclude_indices=exclude_2
        )

    # Fail matching if required references couldn't be resolved.
    # For these task types, a missing destination means the task cannot be evaluated.
    task_type = task_spec.task_type
    if task_type in (TaskType.PICK_AND_PLACE, TaskType.STACK):
        if task_spec.destination_ref and destination_index is None:
            LOGGER.warn(
                f"Task type {task_type.value} requires a destination, but match failed for "
                f"'{task_spec.destination_ref}'"
            )
            return None

    if task_type == TaskType.BETWEEN:
        if destination_index is None or between_anchor_index is None:
            LOGGER.warn(
                f"Task type BETWEEN requires two anchor objects, but matching failed. "
                f"destination_index={destination_index}, between_anchor_index={between_anchor_index}"
            )
            return None

    return ResolvedTask(
        task_spec=task_spec,
        target_index=target_index,
        destination_index=destination_index,
        between_anchor_index=between_anchor_index,
    )
