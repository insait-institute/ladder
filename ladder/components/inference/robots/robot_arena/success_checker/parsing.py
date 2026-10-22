"""Instruction parsing: text instruction -> structured TaskSpec."""

import re
from typing import Dict, List, Optional, Tuple

from ladder.components.inference.robots.robot_arena.success_checker.types import TaskSpec, TaskType
from ladder.core.common.logger.logger import get_logger

LOGGER = get_logger(__name__)


# Map free-text region keywords to a normalized region name.
_REGION_KEYWORDS: Dict[str, str] = {
    "upper left": "upper_left",
    "top left": "upper_left",
    "upper right": "upper_right",
    "top right": "upper_right",
    "lower left": "lower_left",
    "bottom left": "lower_left",
    "lower right": "lower_right",
    "bottom right": "lower_right",
    "left": "left",
    "right": "right",
    "upper": "upper",
    "top": "upper",
    "lower": "lower",
    "bottom": "lower",
    "center": "center",
    "centre": "center",
    "back": "upper",
    "front": "lower",
}


def _classify_table_region(text: str) -> Optional[str]:
    """Try to extract a table region from free text. Returns None if no match."""
    text_lower = text.lower().strip()
    # Check multi-word regions first (longer matches take priority).
    for keyword in sorted(_REGION_KEYWORDS, key=len, reverse=True):
        if keyword in text_lower:
            return _REGION_KEYWORDS[keyword]
    return None


# When the regex parser cannot handle a scene's instruction (ambiguous phrasing,
# typos, multi-step tasks), add an explicit entry here. Keyed by scene name.
TASK_SPEC_OVERRIDES: Dict[str, TaskSpec] = {
    # ---- default scenes ----
    "default1": TaskSpec(TaskType.PICK_AND_PLACE, "spoon", "cloth", spatial_relation="on"),
    # ---- standard scenes: object-name mismatches or unparseable instructions ----
    # Objects verified against the bundled data (bridge/<scene>/masks/result.json).
    "scene1": TaskSpec(TaskType.PICK_AND_PLACE, "metal bowl", "blue cloth", spatial_relation="on"),
    "scene2": TaskSpec(TaskType.MOVE_TO_REGION, "blue cloth", table_region="upper_right"),
    "scene3": TaskSpec(TaskType.PICK_AND_PLACE, "sauce_pan", "cloth", spatial_relation="on"),
    "scene4": TaskSpec(TaskType.PICK_AND_PLACE, "metal bowl", "folded towel", spatial_relation="on"),
    "scene5": TaskSpec(TaskType.PICK_AND_PLACE, "metal knife", "metal pot", spatial_relation="in"),
    "scene6": TaskSpec(TaskType.MOVE_TO_REGION, "Canned peaches", table_region="upper_right"),
    "scene7": TaskSpec(TaskType.MOVE_RELATIVE, "spatula", "metal pot", spatial_relation="left"),
    "scene8": TaskSpec(TaskType.MOVE_TO_REGION, "brown plastic pot", table_region="right"),
    "scene9": TaskSpec(TaskType.PICK_AND_PLACE, "toy knife", "orange plastic bowl", spatial_relation="in"),
    "scene10": TaskSpec(TaskType.MOVE_TO_REGION, "silver pot", table_region="lower_right"),
    "scene11": TaskSpec(TaskType.PICK_AND_PLACE, "white toy spoon", "orange toy pan", spatial_relation="on"),
    "scene12": TaskSpec(TaskType.MOVE_RELATIVE, "toy corn cob", "metal pot lid", spatial_relation="right"),
    "scene14": TaskSpec(TaskType.MOVE_RELATIVE, "white cloth", "tin can", spatial_relation="left"),
    # lang.txt: "put the blue block on top of the red block". result.json names are
    # garbled here; physical_properties has the correct blue block_target / red block_destination.
    "scene15": TaskSpec(TaskType.STACK, "blue block", "red block"),
    "scene16": TaskSpec(TaskType.STACK, "wooden cube block", "wooden rectangular block"),
    "scene18": TaskSpec(TaskType.MOVE_RELATIVE, "toy potato", "toy carrot", spatial_relation="left"),
    # Source-data swap: the mesh NAMED "blue spiky ball_target" is physically the
    # cloth, and the mesh named "orange folded cloth" is physically the blue object
    # (the actual target). So point target_ref at "orange folded cloth" to grab the
    # real blue object.
    "scene20": TaskSpec(TaskType.PICK_AND_PLACE, "orange folded cloth", "metal pot", spatial_relation="in"),
    "scene21": TaskSpec(TaskType.MOVE_TO_REGION, "salt shaker", table_region="lower"),
    "scene23": TaskSpec(
        TaskType.BETWEEN, "toy ice cream tub", "metal strainer", between_anchor_ref="pineapple"
    ),
    "scene24": TaskSpec(
        TaskType.MOVE_RELATIVE, "orange plastic bowl", "yellow toy banana", spatial_relation="left"
    ),
    "scene25": TaskSpec(TaskType.MOVE_TO_REGION, "cube", table_region="left"),
    "scene27": TaskSpec(TaskType.PICK_AND_PLACE, "toy orange half", "green cloth", spatial_relation="on"),
    "scene29": TaskSpec(TaskType.STACK, "wooden block", "wooden block"),
    "scene30": TaskSpec(TaskType.MOVE_RELATIVE, "pink spoon", "orange cloth", spatial_relation="left"),
    "scene31": TaskSpec(TaskType.MOVE_TO_REGION, "plastic cup", table_region="right"),
    "scene32": TaskSpec(TaskType.PICK_AND_PLACE, "small metal pot", "purple cloth", spatial_relation="on"),
    "scene33": TaskSpec(TaskType.MOVE_TO_REGION, "purple towel", table_region="lower_right"),
    "scene34": TaskSpec(TaskType.MOVE_TO_REGION, "toy potato", table_region="lower"),
    "scene35": TaskSpec(TaskType.PICK_AND_PLACE, "toy carrot", "cup", spatial_relation="in"),
    "scene36": TaskSpec(TaskType.MOVE_RELATIVE, "toy fork", "toy pan", spatial_relation="left"),
    # "put the yellow knife on the right burner": the right burner is a stove
    # location (no movable object for it), NOT the metal pot lid. Paraphrases:
    # "empty burner to the right". Treat as region-arrival on the right side.
    "scene37": TaskSpec(TaskType.MOVE_TO_REGION, "yellow toy knife", table_region="right"),
    "scene39": TaskSpec(TaskType.PICK_AND_PLACE, "toy sushi", "metal pot", spatial_relation="in"),
    "scene40": TaskSpec(TaskType.PICK_AND_PLACE, "red toy tomato", "metal pot", spatial_relation="in"),
    "scene41": TaskSpec(TaskType.PICK_AND_PLACE, "metal pot", "green cloth", spatial_relation="on"),
    "scene42": TaskSpec(TaskType.STACK, "yellow block", "green block"),
    "scene43": TaskSpec(TaskType.PICK_AND_PLACE, "toy knife", "green cloth", spatial_relation="on"),
    "scene44": TaskSpec(TaskType.PICKUP, "red capped bottle"),
    "scene45": TaskSpec(TaskType.MOVE_TO_REGION, "blue bowl", table_region="upper_right"),
    "scene46": TaskSpec(TaskType.MOVE_TO_REGION, "orange salmon sushi toy", table_region="center"),
    "scene47": TaskSpec(TaskType.REACH, "Toy frying pan"),
    "scene48": TaskSpec(TaskType.PICK_AND_PLACE, "yellow brush", "purple towel", spatial_relation="on"),
    "scene49": TaskSpec(
        TaskType.PICK_AND_PLACE, "yellow toy cheese wedge", "toy metal pot", spatial_relation="in"
    ),
    "scene50": TaskSpec(TaskType.PICK_AND_PLACE, "stuffed animal", "metal pot", spatial_relation="in"),
    "scene53": TaskSpec(TaskType.PUSH, "milk carton", table_region="upper"),
    "scene54": TaskSpec(TaskType.PICK_AND_PLACE, "spoon", "cloth", spatial_relation="on"),
    "scene55": TaskSpec(TaskType.MOVE_TO_REGION, "green cloth", table_region="lower_left"),
    "scene57": TaskSpec(TaskType.MOVE_TO_REGION, "metal pot", table_region="upper_right"),
    "scene72": TaskSpec(TaskType.PICK_AND_PLACE, "radish half", "plate", spatial_relation="on"),
    "scene77": TaskSpec(TaskType.PICK_AND_PLACE, "toy pot", "folded blue towel", spatial_relation="on"),
    "scene79": TaskSpec(
        TaskType.BETWEEN, "toy knife", "small metal pot", between_anchor_ref="yellow toy pepper"
    ),
    "scene80": TaskSpec(TaskType.MOVE_RELATIVE, "patterned container", "metal pot", spatial_relation="right"),
    "scene81": TaskSpec(TaskType.PICK_AND_PLACE, "Can of food", "Small pot", spatial_relation="in"),
    "scene82": TaskSpec(
        TaskType.MOVE_TO_REGION, "toy food can with red and yellow label", table_region="lower"
    ),
    "scene84": TaskSpec(TaskType.MOVE_RELATIVE, "blue cloth", "metal bowl", spatial_relation="left"),
    # "place the fork NEAR the yellow cloth": proximity only, no direction. REACH
    # with a destination evaluates is_near(target, dest), which is pure proximity.
    "scene85": TaskSpec(TaskType.REACH, "fork", "folded cloth"),
    "scene90": TaskSpec(TaskType.PICK_AND_PLACE, "toy pot lid", "toy pot", spatial_relation="on"),
    "scene93": TaskSpec(TaskType.MOVE_TO_REGION, "rectangular block", table_region="center"),
    "scene94": TaskSpec(TaskType.PICKUP, "toy milk carton"),
    "scene97": TaskSpec(
        TaskType.PICK_AND_PLACE, "toy knife", "green toy cutting board", spatial_relation="on"
    ),
    "scene98": TaskSpec(TaskType.PICK_AND_PLACE, "blue bowl", "blue plate", spatial_relation="on"),
}

# Priority-ordered regex patterns. First match wins.
_INSTRUCTION_PATTERNS: List[Tuple[re.Pattern[str], TaskType, List[str]]] = [
    # STACK: "stack X on Y" or "put X on top of Y"
    (
        re.compile(r"stack\s+(?:the\s+)?(.+?)\s+on(?:\s+top\s+of)?\s+(?:the\s+)?(.+)", re.IGNORECASE),
        TaskType.STACK,
        ["target_ref", "destination_ref"],
    ),
    (
        re.compile(r"(?:put|place|move)\s+(?:the\s+)?(.+?)\s+on\s+top\s+of\s+(?:the\s+)?(.+)", re.IGNORECASE),
        TaskType.STACK,
        ["target_ref", "destination_ref"],
    ),
    # BETWEEN: "place X between Y and Z"
    (
        re.compile(
            r"(?:put|place|take|move)\s+(?:the\s+)?(.+?)\s+(?:and\s+place\s+)?between\s+(?:the\s+)?(?:two\s+)?(.+?)\s+and\s+(?:the\s+)?(.+)",
            re.IGNORECASE,
        ),
        TaskType.BETWEEN,
        ["target_ref", "destination_ref", "between_anchor_ref"],
    ),
    # PICKUP: "pick up X", "pickup X", "lift X"
    (
        re.compile(r"(?:pick\s*up|pickup|lift)\s+(?:the\s+)?(.+)", re.IGNORECASE),
        TaskType.PICKUP,
        ["target_ref"],
    ),
    # PUSH: "push X to ..."
    (
        re.compile(r"push\s+(?:the\s+)?(.+?)\s+to\s+(?:the\s+)?(.+)", re.IGNORECASE),
        TaskType.PUSH,
        ["target_ref", "_region_text"],
    ),
    # REACH: "reaching X", "reach X"
    (
        re.compile(r"(?:end\s+effector\s+)?reach(?:ing)?\s+(?:the\s+)?(.+)", re.IGNORECASE),
        TaskType.REACH,
        ["target_ref"],
    ),
    # MOVE RELATIVE: "move X to the left/right/... of Y"
    (
        re.compile(
            r"(?:move|moved|put|place|take)\s+(?:the\s+)?(.+?)\s+(?:to|in|at|into)\s+(?:the\s+)?"
            r"(left|right|upper|lower|top|bottom|front|back|above|below)"
            r"(?:\s+(.+))?",
            re.IGNORECASE,
        ),
        TaskType.MOVE_RELATIVE,
        ["target_ref", "_direction", "_ref_or_region"],
    ),
    # PICK AND PLACE: "put/place/take X on/in/into/onto/inside/near Y"
    (
        re.compile(
            r"(?:put|place|take|move|relocate)\s+(?:the\s+)?(.+?)\s+"
            r"(?:and\s+(?:put|place)\s+(?:it\s+)?)?"
            r"(?:on(?:to)?|in(?:to|side)?|inside(?:\s+of)?|near|beside)\s+(?:the\s+)?(.+)",
            re.IGNORECASE,
        ),
        TaskType.PICK_AND_PLACE,
        ["target_ref", "destination_ref"],
    ),
    # MOVE TO REGION (fallback for "move X to ..."): "move X to corner/edge of table"
    (
        re.compile(
            r"(?:move|put|place|take|moved)\s+(?:the\s+)?(.+?)\s+(?:to|in|at|from\s+\w+\s+to)\s+(?:the\s+)?(.+)",
            re.IGNORECASE,
        ),
        TaskType.MOVE_TO_REGION,
        ["target_ref", "_region_text"],
    ),
]


def parse_instruction(instruction: str, scene_name: str) -> Optional[TaskSpec]:  # noqa: PLR0912
    """Parse a text instruction into a structured TaskSpec.

    Returns None if the instruction cannot be parsed.
    """
    # Check per-scene overrides first.
    if scene_name in TASK_SPEC_OVERRIDES:
        return TASK_SPEC_OVERRIDES[scene_name]

    instruction = instruction.strip()
    if not instruction:
        return None

    for pattern, task_type, group_names in _INSTRUCTION_PATTERNS:
        match = pattern.match(instruction)
        if match is None:
            continue

        groups = [g.strip() if g else None for g in match.groups()]
        fields: Dict[str, Optional[str]] = {}

        for i, name in enumerate(group_names):
            value = groups[i] if i < len(groups) else None
            if value:
                # Strip trailing punctuation and common noise.
                value = re.sub(r"[.\s]+$", "", value)

            if name == "_region_text":
                region = _classify_table_region(value or "")
                if region is not None:
                    fields["table_region"] = region
                else:
                    # Couldn't classify as a region; treat the text as a destination object.
                    fields["destination_ref"] = value
                    # Reclassify as MOVE_RELATIVE if there's no region.
                    if task_type == TaskType.MOVE_TO_REGION:
                        task_type = TaskType.MOVE_RELATIVE
            elif name == "_direction":
                direction = value.lower() if value else None
                # Normalize direction names.
                direction_map = {"top": "upper", "bottom": "lower", "above": "upper", "below": "lower"}
                fields["spatial_relation"] = direction_map.get(direction, direction)
            elif name == "_ref_or_region":
                if value:
                    # Strip common preposition noise ("side of the", "corner of the", "of the").
                    cleaned = re.sub(
                        r"^(?:side|corner|edge|part)\s+(?:of\s+)?(?:the\s+)?",
                        "",
                        value,
                        flags=re.IGNORECASE,
                    ).strip()
                    cleaned = re.sub(r"^(?:of|on)\s+(?:the\s+)?", "", cleaned, flags=re.IGNORECASE).strip()
                    # Check if this looks like a table region reference.
                    if "table" in cleaned.lower():
                        fields["table_region"] = fields.get("spatial_relation", "center")
                        task_type = TaskType.MOVE_TO_REGION
                    else:
                        region = _classify_table_region(cleaned)
                        if region is not None:
                            fields["table_region"] = region
                            task_type = TaskType.MOVE_TO_REGION
                        else:
                            fields["destination_ref"] = cleaned
            else:
                fields[name] = value

        # Determine spatial relation for PICK_AND_PLACE from preposition.
        if task_type == TaskType.PICK_AND_PLACE and "spatial_relation" not in fields:
            lower = instruction.lower()
            if " in " in lower or "into" in lower or "inside" in lower:
                fields["spatial_relation"] = "in"
            else:
                fields["spatial_relation"] = "on"

        return TaskSpec(task_type=task_type, **fields)

    LOGGER.warn(f"Could not parse instruction: '{instruction}' (scene={scene_name})")
    return None
