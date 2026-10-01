import argparse
import io
import json
import os
from typing import Any, Dict, List

import huggingface_hub
import numpy as np
import pandas
from PIL import Image

from ladder.components.inference.api.client.hlc.sources.decomposers import Decomposer, DecomposerConfig
from ladder.components.inference.api.client.metrics.plan import score_plan
from ladder.components.inference.api.client.metrics.tools.backends import BACKENDS, vlm_config
from ladder.components.inference.api.client.metrics.tools.scenes import Frame, load_scenes
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import AutoConfigurable

LOGGER = get_logger(__name__)


def build_decomposer(model: str, backend: str, max_tokens: int, object_attributes: bool) -> Decomposer:
    """The shipped Decomposer, not a reimplementation: it owns the prompt and the reply parsing, and
    leaving ``cache_path`` unset is what makes it decompose from the frame rather than serve a stored
    plan -- so this measures the path an orchestrator actually takes at reset.

    The system prompt is deliberately NOT overridable. Tuning it against this benchmark's own
    failures gained 4.5 exact plans of 92 on average and exactly 0.0 once the same instructions were
    restated without the dataset's object nouns, so a prompt knob here only invites reporting a
    leaked answer key as an improvement.
    """
    return AutoConfigurable(
        DecomposerConfig(
            autoclass="ladder.components.inference.api.client.hlc.sources.decomposers.Decomposer",
            cache_path="",
            vlm=vlm_config(model, backend),
            max_tokens=max_tokens,
            temperature=0.0,
            object_attributes=object_attributes,
        )
    )


def load_hub_scenes(repo_id: str) -> List[Dict[str, Any]]:
    """The same benchmark from the Hub, so a planner can be scored with no simulator installed.

    Image columns are HF Image features -- a struct carrying the PNG bytes -- so the frames arrive
    inline and there is nothing to render.
    """
    parquets = sorted(
        name
        for name in huggingface_hub.list_repo_files(repo_id, repo_type="dataset")
        if name.endswith(".parquet")
    )
    # The repo also carries the subtask-state samples as a second config (under subtask_states/); the scenes
    # live under decomposition/ (data/ in repos published before the config was named). Without the
    # filter every sample row would be read as a scene; the fallbacks keep older repos loadable.
    for prefix in ("decomposition/", "data/"):
        scenes_only = [name for name in parquets if name.startswith(prefix)]
        if scenes_only:
            parquets = scenes_only
            break
    if not parquets:
        raise ValueError(f"{repo_id}: no parquet files")
    scenes: List[Dict[str, Any]] = []
    for name in parquets:
        local = huggingface_hub.hf_hub_download(repo_id, name, repo_type="dataset")
        for row in pandas.read_parquet(local).to_dict("records"):
            frames = {
                column.removeprefix("image_"): row[column]["bytes"]
                for column in ("image_main", "image_wrist")
                if isinstance(row.get(column), dict) and row[column].get("bytes")
            }
            scenes.append(
                {
                    **{key: value for key, value in row.items() if not key.startswith("image_")},
                    "subtasks": list(row["subtasks"]),
                    "n_goals": int(row["n_goals"]),
                    "frames": frames,
                }
            )
    LOGGER.info(f"{len(scenes)} scene(s) from {repo_id}")
    return scenes


def publish(scenes: List[Dict[str, Any]], repo_id: str, private: bool) -> None:
    """Push the exported benchmark to the Hub, so it can be scored with `--dataset` and no simulator.

    The schema must stay in step with what the exporters write -- `family` and `distractors` were
    once absent here while the published dataset carried them, so a republish silently dropped two
    columns.
    """
    import datasets  # only the authors' push needs it; it is not in the scoring env

    features = datasets.Features(
        {
            "id": datasets.Value("string"),
            "benchmark": datasets.Value("string"),
            "scene": datasets.Value("string"),
            "instruction": datasets.Value("string"),
            "subtasks": datasets.Sequence(datasets.Value("string")),
            "n_goals": datasets.Value("int32"),
            "ordered": datasets.Value("bool"),
            "revisits": datasets.Value("bool"),
            "family": datasets.Value("string"),
            "distractors": datasets.Sequence(datasets.Value("string")),
            "objects": datasets.Sequence(datasets.Value("string")),
            "image_main": datasets.Image(),
            "image_wrist": datasets.Image(),  # RoboLab only; the other two suites publish one view
        }
    )
    rows = []
    for scene in scenes:
        frames = {camera: Image.open(path).convert("RGB") for camera, path in scene["frames"].items()}
        rows.append(
            {
                **{key: scene[key] for key in features if not key.startswith("image_") and key != "revisits"},
                "revisits": bool(scene.get("revisits", False)),
                "image_main": frames["main"],
                "image_wrist": frames.get("wrist"),
            }
        )
    datasets.Dataset.from_list(rows, features=features).push_to_hub(
        repo_id, config_name="decomposition", private=private
    )
    LOGGER.info(f"{len(rows)} scene(s) -> https://huggingface.co/datasets/{repo_id}")


def _image(frame: Frame) -> np.ndarray:
    handle = io.BytesIO(frame) if isinstance(frame, bytes) else frame
    return np.asarray(Image.open(handle).convert("RGB"), dtype=np.uint8)


# The frame each suite's env_server publishes to a controller. RoboLab letterboxes to 224x224 while
# its exporter writes the native 1280x720 render, so scoring the render shows the planner 25x the
# pixels a live arm gets; the other two publish their camera frame untouched.
_DEPLOYED_SIZE = {"robolab_comp": (224, 224), "robotarena_comp": None, "simplerenv_comp": None}


def _letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """openpi's ``resize_with_pad``: fit on the longer side, then centre on black."""
    source = Image.fromarray(image)
    ratio = max(source.width / width, source.height / height)
    resized = source.resize((int(source.width / ratio), int(source.height / ratio)), Image.BILINEAR)
    canvas = Image.new("RGB", (width, height), 0)
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def _as_published(image: np.ndarray, benchmark: str) -> np.ndarray:
    if benchmark not in _DEPLOYED_SIZE:
        raise KeyError(f"{benchmark}: add its published frame size to _DEPLOYED_SIZE")
    size = _DEPLOYED_SIZE[benchmark]
    return image if size is None else _letterbox(image, *size)


def _no_plan(row: Dict[str, Any]) -> bool:
    """The instruction handed straight back having scored nothing, which is what an empty reply
    becomes: ``complete`` returns "" on a transient failure and the Decomposer falls back to the task.
    Six scenes are single-goal with the subtask equal to the instruction, so the score matters."""
    return bool(row["error"]) or (row["predicted"] == [row["instruction"]] and row["exact"][0] == 0.0)


def observation(scene: Dict[str, Any]) -> RobotObservation:
    images = {
        camera: _as_published(_image(frame), scene["benchmark"]) for camera, frame in scene["frames"].items()
    }
    # state and timestamps are required fields the decomposer never reads: it sees one frame and the
    # compound instruction. env_name would key a plan cache, which this decomposer has none of.
    return RobotObservation(images=images, state=None, timestamps=None, metadata={"env_name": scene["scene"]})


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a planner's zero-shot decompositions.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frames-dir", help="holds <suite>/scenes.json next to the frames")
    source.add_argument("--dataset", help="Hub dataset id instead; needs no simulator and no frames")
    parser.add_argument("--out", help="results json; not needed with --publish")
    parser.add_argument("--publish", metavar="REPO_ID", help="push the export to the Hub and exit")
    parser.add_argument("--private", action="store_true", help="with --publish")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--backend", choices=BACKENDS, default="local")
    # 8192, because hidden thinking tokens count against this and a truncated reply scores as a plan
    # that missed every goal: over 803 gateway calls, 1024 truncated 22.5% and 8192 none (p99 7940).
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--object-attributes",
        action="store_true",
        help="name objects by colour plus type, the way the RoboLab live arm is deployed",
    )
    parser.add_argument("--limit", type=int, help="first N scenes only, for a smoke run")
    args = parser.parse_args()

    scenes = load_scenes(args.frames_dir) if args.frames_dir else load_hub_scenes(args.dataset)
    if args.publish:
        publish(scenes, args.publish, args.private)
        return
    if not args.out:
        parser.error("--out is required unless --publish is given")
    scenes = scenes[: args.limit]
    decomposer = build_decomposer(args.model, args.backend, args.max_tokens, args.object_attributes)
    LOGGER.info(f"scoring {args.model} on {len(scenes)} scene(s)")

    results, streak = [], 0
    for index, scene in enumerate(scenes, start=1):
        try:
            plan = decomposer.decompose(observation(scene), scene["instruction"])
            predicted, error = list(plan.subtasks) if plan else [], ""
            ordered = bool(plan.ordered) if plan else False
        except Exception as exc:  # noqa: BLE001  -- a refusal or a malformed reply is a result, not a crash
            predicted, error, ordered = [], repr(exc)[:200], False
        results.append(
            {
                **{k: scene[k] for k in ("id", "benchmark", "n_goals", "instruction")},
                "predicted": predicted,
                "reference": scene["subtasks"],
                "error": error,
                **score_plan(predicted, scene["subtasks"]),
                # The prompt requires this flag and VerifiedAdvancer sequences on it.
                "ordered": (float(ordered == bool(scene["ordered"])), 1.0),
            }
        )
        # A backend answering nothing must not read as a planner that plans badly. The VLM's own
        # failure budget cannot catch it: its rate check needs 200 calls and this is 143 scenes.
        streak = streak + 1 if _no_plan(results[-1]) else 0
        if streak >= 10:
            raise SystemExit(f"{scene['id']}: 10 scenes in a row with no usable plan")
        LOGGER.info(f"[{index}/{len(scenes)}] {scene['id']}")

    # sum/sum, as every metric in this package is reported, so a ten-goal scene weighs ten times a
    # one-goal scene rather than equally.
    summary: Dict[str, Any] = {
        "model": args.model,
        "object_attributes": args.object_attributes,
        "scenes": len(results),
        "errors": sum(1 for row in results if row["error"]),
        "no_plan": sum(1 for row in results if _no_plan(row)),
    }
    for key in ("exact", "goals_missed", "goals_invented", "register", "length", "ordered"):
        got, want = (sum(row[key][side] for row in results) for side in (0, 1))
        summary[key] = round(got / want, 4) if want else 0.0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump({"summary": summary, "results": results}, handle, indent=1)
    LOGGER.info(f"{json.dumps(summary, indent=1)}\nwrote {args.out}")


if __name__ == "__main__":
    main()
