import dataclasses
import json
import re
from typing import List, Optional, Sequence, Tuple

_VAGUE_WORDS = frozenset({"the", "a", "an", "of", "on", "in", "and", "object", "item", "one"})


@dataclasses.dataclass(frozen=True)
class Box:
    corners: Tuple[float, float, float, float]
    label: str = ""


def parse_boxes(reply: str) -> List[Box]:
    text = re.sub(r"```(?:json)?|```", "", reply or "").strip()
    match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    entries = payload if isinstance(payload, list) else [payload]
    boxes = []
    for entry in entries:
        if isinstance(entry, dict):
            # The prompt asks for "box_2d", but Gemini also answers "box" or "box_": the four
            # coordinates are the box whatever their key.
            candidates = [entry.get("box_2d")] + [v for k, v in entry.items() if k != "label"]
            raw = next((v for v in candidates if _is_corners(v)), None)
            label = str(entry.get("label", ""))
        else:
            raw, label = entry, ""
        if _is_corners(raw):
            boxes.append(Box(corners=tuple(float(v) for v in raw), label=label))
    return boxes


def _is_corners(value: object) -> bool:
    return isinstance(value, Sequence) and len(value) == 4 and all(isinstance(v, (int, float)) for v in value)


def _words(text: str) -> frozenset:
    return frozenset(re.findall(r"[a-z]+", text.lower())) - _VAGUE_WORDS


def best_match(boxes: List[Box], name: str, strict: bool = False) -> Optional[Box]:
    # A lone box is taken on trust. `strict` is for when one reply answers several names, where
    # that would hand one name another's box.
    wanted = _words(name)
    best = max(boxes, key=lambda box: len(wanted & _words(box.label)))
    if (strict or len(boxes) > 1) and not (wanted & _words(best.label)):
        return None
    return best
