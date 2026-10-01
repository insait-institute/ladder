import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import PIL.Image

from ladder.components.inference.api.client.hlc.vlm.base import VLM
from ladder.components.inference.api.client.policy import Policy
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import AutoConfig, AutoConfigurable, Config, Configurable

LOGGER = get_logger(__name__)


# The one completion question, asked of every subtask everywhere. Phrased as a COUNT rather than a
# yes/no because a plan carries one line per goal: two carrots give two identical sentences, and a
# yes/no is true for both the moment one carrot is placed. It names no object -- the model reads the
# kind out of the subtask text -- so it needs no per-task variant.
# The "held directly above the goal counts" clause is benchmark-measured: on 135 sim-labeled frames
# it cut missed completions from 20/34 to 3/34 (the gap is the hover case) at zero false positives.
_VERIFY_PROMPT = (
    "You are checking the CURRENT state of a robot tabletop scene from its camera views. The subtask "
    "below moves ONE object to a goal location, and the scene may hold SEVERAL interchangeable "
    "objects of that same kind. Count how many DISTINCT objects of the kind the subtask names are in "
    "the subtask's END STATE RIGHT NOW -- at the goal location the subtask asks for, regardless of "
    "how they got there. An object counts as placed when any part of it rests on the goal location, "
    "INCLUDING when partially hidden by the robot arm or other objects. An object the gripper is "
    "still holding does NOT count, however close to the goal location it is held: the release is not "
    "yet a placement and it is the release that most often fails. An object away from the goal "
    "location, or lifted well above it, is not counted. Count each "
    "physical object at most once, however many views show it. First write one short line listing "
    'what you counted, then output STRICT JSON: {"count": <integer>}.'
)


# A bin is seen from outside, so an object dropped in is never visible: count it by what has left
# the rest of the scene. RoboLab frames vs the simulator's labels: recall 0.535 -> 0.646, precision unchanged.
_CONTAINER_SUFFIX = (
    " When a goal location is a container whose interior you cannot see into, do not look for the "
    "object inside it. Instead account for every object of that kind in the scene: if an object of "
    "that kind is no longer resting anywhere else in the scene, and the gripper is not holding it, "
    "it is in the container and counts as placed. Objects still resting elsewhere do not count."
)


_DELTA_SUFFIX = (
    " Two frames are provided: BEFORE (an earlier check) and NOW. Judge the CURRENT state from the "
    "NOW frame; use BEFORE only to resolve occlusion -- an object resting in place BEFORE that has "
    "not visibly been picked up or moved away is STILL in place NOW."
)

# Appended when more than one camera is sent, so the model knows the views show ONE scene at ONE
# instant rather than a sequence. Without this a second view reads as a second timestep.
_MULTIVIEW_SUFFIX = (
    " Several camera views of the SAME scene at the SAME instant are provided, each labelled. They do "
    "not show different moments. A view may not contain the goal location at all -- judge from "
    "whichever view shows it, and treat evidence from a close-up view as authoritative over a distant "
    "one where the goal location is small or occluded."
)


def _shrink(image: np.ndarray, max_side: Optional[int]) -> np.ndarray:
    """Downscale so the longest side is ``max_side``; pass through when already smaller or unset.

    Done HERE rather than in a backend because every backend pays for the frame -- a hosted model in
    input tokens (133,719 gateway calls measured 2314 each, 309M in a day, nearly all of it the frame)
    and a served one in prefill -- and because doing it here is what makes the trade measurable on a
    LOCAL verifier, for free, instead of only against a rate-limited API.
    """
    if max_side is None or max(image.shape[:2]) <= max_side:
        return image
    scale = max_side / max(image.shape[:2])
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return np.asarray(PIL.Image.fromarray(image).resize(size, PIL.Image.LANCZOS))


def _is_count(value: Any) -> bool:
    """A non-negative integer. ``bool`` is an ``int`` subclass, so ``true`` is a malformed count
    rather than a count of one; a reply carrying no count must not coerce to "nothing placed"."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class VerifierConfig(Config):
    # A VLM answers a task-agnostic yes/no on the subtask text. vlm=None disables the check (returns
    # None) so the caller round-robins blindly.
    vlm: Optional[AutoConfig] = None  # any VLM backend (OpenAIVLM, LocalVLM, ...)
    max_tokens: int = 2048  # flash spends hidden thinking tokens; 1024 truncated ~20-40% of replies
    temperature: float = 0.0
    # Which camera views the check sees, in order. More than one is worth it wherever the goal
    # location is not legible from a single view: on RoboLab the over-shoulder "main" camera shows
    # the bin as a ~15 px box at the frame edge, routinely occluded by the arm, and the verifier
    # scored 4% recall there (362 of 377 completed subtasks missed) against 56-68% on the benchmarks
    # whose tabletop view shows the goal clearly. The wrist view looks into the bin. Cameras absent
    # from the observation are skipped, so a list is safe across embodiments.
    image_cameras: List[str] = ["main"]
    # Also send the previous verification's frame to disambiguate occlusion ("it was there and
    # nothing removed it"), cutting the false-negatives that drive re-command/undo loops.
    verify_delta: bool = False
    container_absence: bool = True  # append _CONTAINER_SUFFIX; neutral where the goal is in plain view
    verify_prompt: str = _VERIFY_PROMPT  # override to change what "done" means per task
    # Longest side of every frame sent to the check. None sends the render untouched. It applies to
    # all backends, so the accuracy cost is measurable against a LOCAL verifier without spending API
    # quota -- compare advance precision on the episodes' gt flag at two values.
    max_image_side: Optional[int] = None
    # Passed through to the backend when set. A reasoning model spends hidden thinking tokens on every
    # call and they are billed: measured over 133,719 gateway verifier calls, 292 thinking tokens each,
    # 39M in a day, against 16 output tokens of actual answer. The question here is a COUNT of objects
    # visible in one frame, so the deliberation buys little -- but it is a per-config knob rather than
    # a new default, because the answer quality is what the whole advancer rests on and it has to be
    # re-measured against `gt` before being trusted. None leaves the backend at its own default.
    reasoning_effort: Optional[str] = None


class Verifier(Configurable[VerifierConfig]):
    """Per-subtask completion check: one yes/no VLM call on the subtask text. Returns True/False, or
    None when the reply carries no verdict; a backend that cannot answer raises, so a run never
    scores a system whose verifier was silently dead. Stateless (API call + JSON parse only, no
    advancer state), so one instance serves every env of a fan-out; ``prev_image`` is the earlier
    frame for delta checks."""

    def __init__(self, config: VerifierConfig) -> None:
        super().__init__(config)
        self._vlm: Optional[VLM] = AutoConfigurable(config.vlm) if config.vlm is not None else None

    def bind_policy(self, policy: Policy) -> None:
        if self._vlm is not None:
            self._vlm.bind_policy(policy)

    def check(
        self, observation: RobotObservation, subtask: str, prev_image=None, required: int = 1
    ) -> Tuple[Optional[bool], str]:
        """(verdict, raw reply). The reply is returned, not stashed on self, so it stays correct
        under the per-env fan-out.

        ``required`` is how many instances must be in the end state for this subtask to count as
        done: 1 for a goal named once, n for the n-th of n identical lines. One question serves
        both -- the model reports how many it sees, and only the threshold differs.
        """
        assert required >= 1, f"required must be at least 1, got {required}"
        images = observation.images or {}
        views = [(name, images[name]) for name in self.config.image_cameras if images.get(name) is not None]
        if self._vlm is None or not views:
            return None, ""
        prompt = self.config.verify_prompt
        if self.config.container_absence:
            prompt += _CONTAINER_SUFFIX
        content: List[Dict[str, object]] = [{"type": "text", "text": f"Subtask: {subtask}"}]
        if self.config.verify_delta and prev_image is not None:
            prompt += _DELTA_SUFFIX
            content += [
                {"type": "text", "text": "BEFORE:"},
                {"type": "image", "image": _shrink(prev_image, self.config.max_image_side)},
                {"type": "text", "text": "NOW:"},
            ]
        if len(views) > 1:
            prompt += _MULTIVIEW_SUFFIX
            for name, image in views:
                content += [
                    {"type": "text", "text": f"{name} view:"},
                    {"type": "image", "image": _shrink(image, self.config.max_image_side)},
                ]
        else:
            content.append({"type": "image", "image": _shrink(views[0][1], self.config.max_image_side)})
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": content}]
        # Compared here rather than in the prompt, so the threshold stays out of what is asked.
        placed, text = self._ask(messages, key="count", is_usable=_is_count)
        return (None if placed is None else placed >= required), text

    def _ask(
        self, messages: List[Dict[str, object]], key: str, is_usable: Callable[[Any], bool]
    ) -> Tuple[Optional[Any], str]:
        """(the value under ``key``, raw reply), or (None, reply) when the model produced no usable one.

        An unusable reply is tolerated: the caller keeps its prior state. A backend that cannot
        answer at all is not -- it never recovers, so swallowing it would score a run that checked
        nothing.
        """
        text = ""
        for _ in range(2):  # one retry: thinking-token truncation often leaves no JSON in the reply
            text = self._vlm.complete(
                messages,
                self.config.max_tokens,
                self.config.temperature,
                self.config.reasoning_effort,
            )
            match = re.search(r"\{.*\}", text, re.DOTALL)
            try:
                value = json.loads(match.group(0)).get(key) if match else None
            except json.JSONDecodeError:
                value = None
            if is_usable(value):
                return value, text
            LOGGER.warn(f"Verifier: no usable {key!r} in reply; retrying")
        return None, text
