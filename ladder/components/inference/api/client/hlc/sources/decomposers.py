import json
import os
import re
from typing import Dict, List, Optional

from ladder.components.inference.api.client.hlc.base import Decomposition, SubtaskSource
from ladder.components.inference.api.client.hlc.vlm.base import VLM
from ladder.components.inference.api.client.policy import Policy
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.core.common.logger import get_logger
from ladder.core.common.platform import REPO_ROOT
from ladder.core.databib.config import AutoConfig, AutoConfigurable, Config

LOGGER = get_logger(__name__)


_DECOMP_PROMPT = """You decompose a tabletop robot task into single-object pick-and-place subtasks. \
You see ONE camera frame and the overall task. Output STRICT JSON and nothing else:
{{"subtasks": ["<subtask>", ...], "ordered": true|false}}
Each subtask moves exactly ONE object to ONE destination and is EXACTLY of the form "put the <object> \
in the <container>." when the destination is a container (bin, bowl, mug, pot, crate, basket), or \
"put the <object> on the <receptacle>." when it is a flat surface (plate, shelf, towel, table). Use no \
other form, whatever the object currently rests on: this is the wording every benchmark's own plan \
uses, and a subtask phrased any other way matches no goal. {object_naming} \
Name each receptacle by a distinguishing visual attribute plus its type, read from the image ("the \
grey bin", "the white plate"); when several receptacles are visible the attribute is what tells them \
apart. Visually identical objects get the same subtask text, repeated once per instance. Look at the \
image to decide WHICH objects the task refers to when it is abstract ("put everything except the \
eggplant in the bin", "pack the canned foods"); include every object the task requires and nothing \
else -- an object the task excludes gets no subtask. When an object must pass through intermediate \
stops ("first put it on the plate, then move it to the bin"), emit one subtask per stop for that \
object, in order. Set "ordered": true when the task prescribes a sequence (first/then/before/after, or \
intermediate stops that must precede a final placement) and false when the subtasks may be done in \
any order."""

# How objects are named in the plan. Colour + type names the object the way it is seen, for the policy
# and the completion check alike: under a bare noun pi0.5 grasps a look-alike (the lemon for the orange)
# in a third of its grasps.
_OBJECT_NAMING = {
    False: 'Name each object by its plain type noun as it appears ("the lemon", "the banana").',
    True: (
        'Name each object by a distinguishing visual attribute plus its type, read from the image '
        '("the yellow lemon", "the green lime", "the orange"); the attribute is what separates '
        "look-alike objects of different kinds, so give it wherever another object could be mistaken "
        "for this one."
    ),
}


def _json_plan(text: str) -> Optional[Decomposition]:
    """The plan in a JSON reply, or None when the reply carries no ``subtasks`` list."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    raw = data.get("subtasks") if isinstance(data, dict) else None
    if not isinstance(raw, list) or not all(isinstance(s, str) for s in raw):
        return None
    subs = [re.sub(r"^[-*\d.)\s]+", "", s).strip() for s in raw]
    subs = [s for s in subs if s]
    return (
        Decomposition(
            subs, ordered=bool(data.get("ordered", False)), revisits=bool(data.get("revisits", False))
        )
        if subs
        else None
    )


class DecomposerConfig(Config):
    # Cache-first, model-fallback: cache_path serves a pre-computed per-scene JSON (deterministic);
    # a miss (or no cache_path) queries the vlm backend, or returns None when vlm=None. Cache is
    # keyed by metadata["env_name"]; each value is a [subtask, ...] list or {"subtasks", "ordered"}.
    # The reserved "_meta" key records how the file was minted (model, date) rather than a scene.
    cache_path: str = ""
    vlm: Optional[AutoConfig] = None  # backend queried on a cache miss (None = cache-only)
    max_tokens: int = 1024
    temperature: float = 0.0
    image_camera: str = "main"
    system_prompt: str = ""  # "" = _DECOMP_PROMPT; a custom prompt is used verbatim
    object_attributes: bool = False  # colour + type object names in the plan (_OBJECT_NAMING)
    user_prompt_template: str = "Task: {task}"  # the user turn next to the reset frame


class Decomposer(SubtaskSource[DecomposerConfig]):
    def __init__(self, config: DecomposerConfig) -> None:
        super().__init__(config)
        self._cache: Dict[str, object] = {}
        if config.cache_path:
            # A repo-relative cache_path resolves against the repo root, because under bazel the cwd
            # is the runfiles tree; join leaves an absolute path alone.
            path = os.path.join(REPO_ROOT, config.cache_path)
            with open(path) as cache_file:
                self._cache = json.load(cache_file)
            meta = self._cache.pop("_meta", "unrecorded")  # provenance, not a scene
            LOGGER.info(
                f"Decomposer: loaded {len(self._cache)} cached scene decompositions from {path}, "
                f"minted by {meta}"
            )
        self._vlm: Optional[VLM] = AutoConfigurable(config.vlm) if config.vlm is not None else None

    def bind_policy(self, policy: Policy) -> None:
        if self._vlm is not None:
            self._vlm.bind_policy(policy)

    def decompose(self, observation: RobotObservation, task: str) -> Optional[Decomposition]:
        scene = (observation.metadata or {}).get("env_name")
        if scene is not None:
            scene = re.sub(r"-v[01]$", "", scene)  # metadata id may carry a gym version suffix
        cached = self._cache.get(scene) if scene is not None else None
        if cached:
            if isinstance(cached, dict):
                return Decomposition(
                    list(cached["subtasks"]),
                    ordered=bool(cached.get("ordered", False)),
                    revisits=bool(cached.get("revisits", False)),
                )
            return Decomposition(list(cached))
        if self._vlm is None:
            # Returning None here used to surface much later as an assertion inside
            # SubtaskSource.commands(), which said nothing about which env was uncached. A cache-only
            # Decomposer that cannot serve the env can never recover, so fail here with the cause.
            raise KeyError(
                f"Decomposer: env {scene!r} has no entry in {self.config.cache_path!r} and no vlm "
                f"backend is configured, so no plan can be produced. Either regenerate the cache for "
                f"this env or give the source a vlm to decompose live."
            )
        image = (observation.images or {}).get(self.config.image_camera)
        if image is None:
            return None
        prompt = self.config.system_prompt or _DECOMP_PROMPT.format(
            object_naming=_OBJECT_NAMING[self.config.object_attributes]
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.config.user_prompt_template.format(task=task)},
                    {"type": "image", "image": image},
                ],
            },
        ]
        text = self._vlm.complete(messages, self.config.max_tokens, self.config.temperature)
        # A JSON reply carries the order flag; a plain list of lines is read as unordered.
        decomposition = _json_plan(text)
        if decomposition is None:
            subs = [re.sub(r"^[-*\d.)\s]+", "", line).strip() for line in text.strip().split("\n")]
            subs = [s for s in subs if s]
            decomposition = Decomposition(subs or [task])
        LOGGER.info(
            f"Decomposer: task={task!r} -> {decomposition.subtasks}"
            f"{' (ordered)' if decomposition.ordered else ''}"
        )
        return decomposition


class PrivilegedDecomposerConfig(Config):
    pass


class PrivilegedDecomposer(SubtaskSource[PrivilegedDecomposerConfig]):
    """Read the ground-truth subtasks the env publishes in ``metadata['oracle']['subtasks']``."""

    def decompose(self, observation: RobotObservation, task: str) -> Optional[Decomposition]:
        del task
        oracle = (observation.metadata or {}).get("oracle") or {}
        subtasks = list(oracle.get("subtasks") or [])
        # An env that scores its goals as a sequence says so here, and the advancer needs to hear it:
        # without the flag the oracle arm would schedule a dependent chain out of order while the
        # planner arms (whose cached plans carry it) do not, making the arms differ in more than the
        # plan's source. Envs that publish no flag are unordered, as before.
        ordered = bool(oracle.get("ordered", False))
        revisits = bool(oracle.get("revisits", False))
        return Decomposition(subtasks, ordered=ordered, revisits=revisits) if subtasks else None


class PrivilegedPhaseDecomposerConfig(Config):
    # Each env goal ("put the X on the Y") is expanded into these phase templates in order; {obj}/
    # {dest} are filled from the goal.
    phase_templates: Optional[List[str]] = None


class PrivilegedPhaseDecomposer(SubtaskSource[PrivilegedPhaseDecomposerConfig]):
    """Expand the env's privileged ordered goals into per-goal phases (grasp/place, or reach/grasp/
    place/release). Goal i's phases occupy indices [i*P, (i+1)*P), so PrivilegedPhaseAdvancer reads
    its achieved/grasping signal at ``current_idx // P``."""

    def decompose(self, observation: RobotObservation, task: str) -> Optional[Decomposition]:
        del task
        goals = ((observation.metadata or {}).get("oracle") or {}).get("subtasks") or []
        templates = self.config.phase_templates or ["grasp the {obj}", "place the {obj} on the {dest}"]
        if not goals:
            return None
        subs: List[str] = []
        for goal in goals:
            match = re.match(
                r"(?:put|place|stack) the (.+?) (?:on|in|onto) the (.+)", goal.strip().rstrip(".")
            )
            obj, dest = (match.group(1), match.group(2)) if match else (goal, "")
            subs.extend(template.format(obj=obj, dest=dest) for template in templates)
        return Decomposition(subs)
