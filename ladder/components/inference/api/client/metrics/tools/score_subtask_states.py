import argparse
import collections
import glob
import io
import json
import os
from typing import Any, Counter, Dict, List, Optional, Tuple

import huggingface_hub
import numpy as np
import pandas
from PIL import Image

from ladder.components.inference.api.client.hlc.sources.vlm_source import VLMSource, VLMSourceConfig
from ladder.components.inference.api.client.hlc.verifiers import Verifier, VerifierConfig
from ladder.components.inference.api.client.metrics.plan import goal_of, in_register, normalize
from ladder.components.inference.api.client.metrics.tools.backends import BACKENDS, vlm_config
from ladder.components.inference.api.client.metrics.tools.scenes import load_scenes
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.core.common.logger import get_logger
from ladder.core.common.platform import REPO_ROOT
from ladder.core.databib.config import AutoConfigurable, load_config

LOGGER = get_logger(__name__)

# The deployed on-demand subtask generator: its system prompt and reply handling are what a
# stepwise eval runs, so next_subtask is scored through the exact component, not a restatement.
GENERATOR_CONFIG = "ladder/components/inference/api/client/hlc/configs/subtask_generator.yaml"

TASKS = ("verification", "state_tracking", "next_subtask")


def load_samples(samples_dir: str) -> List[Dict[str, Any]]:
    """Every suite's subtask_states.json under ``samples_dir``, frames resolved to absolute paths: the same
    relocatable-manifest contract the decomposition benchmark's frames directory uses."""
    samples = []
    for manifest in sorted(glob.glob(os.path.join(samples_dir, "*", "subtask_states.json"))):
        with open(manifest) as handle:
            rows = json.load(handle)
        for row in rows:
            row["frames"] = {
                camera: os.path.join(os.path.dirname(manifest), name)
                for camera, name in row["frames"].items()
            }
            for path in row["frames"].values():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{row['id']}: {path}")
        samples += rows
    if not samples:
        raise FileNotFoundError(f"no <suite>/subtask_states.json under {samples_dir}")
    return samples


def _hub_parquets(repo_id: str, config: str) -> List[str]:
    names = [
        name
        for name in huggingface_hub.list_repo_files(repo_id, repo_type="dataset")
        if name.endswith(".parquet") and name.startswith(f"{config}/")
    ]
    if not names:
        raise ValueError(f"{repo_id}: no parquet files under {config}/")
    return [huggingface_hub.hf_hub_download(repo_id, name, repo_type="dataset") for name in sorted(names)]


def load_hub_samples(repo_id: str) -> List[Dict[str, Any]]:
    """The samples config of the published benchmark, so nothing local is needed to score."""
    samples = []
    for local in _hub_parquets(repo_id, "subtask_states"):
        for row in pandas.read_parquet(local).to_dict("records"):
            samples.append(
                {
                    **{k: row[k] for k in ("id", "benchmark", "scene", "instruction", "arm", "source")},
                    **{k: int(row[k]) for k in ("n_goals", "step", "num_steps", "query_idx")},
                    **{k: bool(row[k]) for k in ("ordered", "label")},
                    "subtasks": [str(s) for s in row["subtasks"]],
                    "achieved": [bool(a) for a in row["achieved"]],
                    "tags": [str(t) for t in row["tags"]],
                    **({"objects": [str(o) for o in row["objects"]]} if "objects" in row else {}),
                    "frames": {
                        column.removeprefix("image_"): row[column]["bytes"]
                        for column in ("image_main", "image_wrist")
                        if isinstance(row.get(column), dict) and row[column].get("bytes")
                    },
                }
            )
    LOGGER.info(f"{len(samples)} sample(s) from {repo_id}")
    return samples


def load_hub_reference(repo_id: str) -> Dict[str, Dict[str, Any]]:
    """The decomposition config of the same repo, keyed by scene: the published copy of the
    reference the samples were cut against."""
    reference: Dict[str, Dict[str, Any]] = {}
    for local in _hub_parquets(repo_id, "decomposition"):
        columns = ["scene", "instruction", "subtasks", "ordered", "revisits", "objects"]
        frame = pandas.read_parquet(local, columns=columns)
        for row in frame.to_dict("records"):
            reference[row["scene"]] = {
                "instruction": row["instruction"],
                "subtasks": [str(s) for s in row["subtasks"]],
                "ordered": bool(row["ordered"]),
                "revisits": bool(row["revisits"]),
                "objects": [str(o) for o in row["objects"]],
            }
    return reference


def check_reference(
    samples: List[Dict[str, Any]], reference: Dict[str, Dict[str, Any]], allow_drift: bool
) -> None:
    """Fail loudly when a sample's task no longer matches the current reference decomposition.

    The reference is regenerated from the task registries by each suite's exporter, so this is the
    sync contract: a task edit changes the reference, every sample of that task turns stale here, and
    nothing scores against the old task by accident. ``allow_drift`` downgrades to a warning for
    deliberate archaeology."""
    drifted = []
    for sample in samples:
        ref = reference.get(sample["scene"])
        if ref is None:
            drifted.append(sample["id"])
            continue
        sample["revisits"] = ref["revisits"]
        # Scene CONTENTS, not wording. A task that drops an object leaves every wording field below
        # byte-identical, so a sample cut before the change still passes them while its frame shows a
        # scene the task no longer builds.
        if "objects" in sample and sorted(sample["objects"]) != sorted(ref["objects"]):
            drifted.append(sample["id"])
            continue
        mine = [goal_of(s) or ("__raw__", normalize(s)) for s in sample["subtasks"]]
        theirs = [goal_of(s) or ("__raw__", normalize(s)) for s in ref["subtasks"]]
        if not sample["ordered"]:
            # A sample keeps its episode's goal order (the goal vector indexes it); on an unordered
            # task the reference lists the same goals in another order, so compare the multiset.
            mine, theirs = sorted(mine), sorted(theirs)
        if (
            normalize(sample["instruction"]) != normalize(ref["instruction"])
            or mine != theirs
            or bool(sample["ordered"]) != bool(ref["ordered"])
        ):
            drifted.append(sample["id"])
    if not drifted:
        return
    message = f"{len(drifted)} sample(s) no longer match the reference decomposition: {drifted[:10]}"
    if not allow_drift:
        raise RuntimeError(message + " — re-export the scenes and re-harvest, or pass --allow-drift")
    LOGGER.warn(message)


def _required(sample: Dict[str, Any], copies: int) -> int:
    """VerifiedAdvancer's instance threshold, so the benchmark asks what the deployment asks: the
    n-th of n identical lines needs n objects placed, except on a revisit chain, where the repeated
    line sends the same object back and n copies of it do not exist.

    ``check_reference`` fills the flag from the reference, which is where plan structure lives; a
    ``--samples-dir`` run given no reference skips that check entirely and falls back to the count."""
    return 1 if sample.get("revisits") else copies


def observation(sample: Dict[str, Any], cameras: List[str]) -> RobotObservation:
    images = {}
    for camera in cameras:
        frame = sample["frames"].get(camera)
        if frame is None:
            continue
        handle = io.BytesIO(frame) if isinstance(frame, bytes) else frame
        images[camera] = np.asarray(Image.open(handle).convert("RGB"), dtype=np.uint8)
    return RobotObservation(
        images=images, state=None, timestamps=None, metadata={"env_name": sample["scene"]}
    )


def distinct_queries(sample: Dict[str, Any]) -> List[Tuple[str, bool, int]]:
    """(subtask text, done, copies) per DISTINCT subtask of the sample's plan. A text that appears several
    times counts done only when every copy is achieved: a yes/no on the text cannot single out one
    instance, and under 'put everything ...' the subtask is finished exactly when no more of that
    object needs moving. The copy count comes back with it because that is the threshold matching
    the label: asking for 1 answers done as soon as one of two carrots is placed."""
    grouped: Dict[str, Tuple[str, List[bool]]] = {}
    for subtask, done in zip(sample["subtasks"], sample["achieved"], strict=False):
        text, flags = grouped.setdefault(normalize(subtask), (subtask, []))
        flags.append(done)
    return [(text, all(flags), len(flags)) for text, flags in grouped.values()]


def classify_next(command: str, subtasks: List[str], achieved: List[bool], ordered: bool) -> str:
    """'valid' when the generated subtask names a goal still to do (the NEXT one, on an ordered
    task), 'redundant' when it re-commands a goal already achieved (the bookkeeping error an
    orchestrator pays budget for) and 'invalid' for anything else."""
    pair = goal_of(command)
    if pair is None:
        return "invalid"
    remaining: Counter[Tuple[str, str]] = collections.Counter(
        p for s, done in zip(subtasks, achieved, strict=False) if not done and (p := goal_of(s))
    )
    if ordered:
        first = next((s for s, done in zip(subtasks, achieved, strict=False) if not done), None)
        valid = first is not None and pair == goal_of(first)
    else:
        valid = pair in remaining
    if valid:
        return "valid"
    done_pairs = {p for s, done in zip(subtasks, achieved, strict=False) if done and (p := goal_of(s))}
    return "redundant" if pair in done_pairs else "invalid"


def _breakdown(rows: List[Dict[str, Any]], key: str, correct: str = "correct") -> Dict[str, str]:
    totals: Dict[str, List[int]] = {}
    for row in rows:
        for group in row[key] if isinstance(row[key], list) else [row[key]]:
            entry = totals.setdefault(group, [0, 0])
            entry[0] += int(bool(row[correct]))
            entry[1] += 1
    return {group: f"{got}/{n}" for group, (got, n) in sorted(totals.items())}


class FailureStreak:
    """Aborts once ``limit`` calls in a row have failed.

    An unusable reply is recorded as a result rather than raised, which is right for one bad frame
    and wrong for a backend refusing everything: the run would finish and report a rate built out of
    errors. ``limit`` is not 1 so a single bad sample still passes.
    """

    def __init__(self, limit: int = 10) -> None:
        self.limit = limit
        self.streak = 0

    def record(self, error: str) -> None:
        self.streak = self.streak + 1 if error else 0
        if self.streak >= self.limit:
            raise SystemExit(f"aborting after {self.limit} consecutive backend failures: {error}")


def run_verification(
    samples: List[Dict[str, Any]], verifier: Verifier, cameras: List[str]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    results = []
    streak = FailureStreak()
    for index, sample in enumerate(samples, start=1):
        subtask = sample["subtasks"][sample["query_idx"]]
        copies = sample["subtasks"][: sample["query_idx"] + 1].count(subtask)
        try:
            verdict, reply = verifier.check(
                observation(sample, cameras), subtask, required=_required(sample, copies)
            )
            error = ""
        except Exception as exc:  # noqa: BLE001 -- an unusable backend reply is a result, not a crash
            verdict, reply, error = None, "", repr(exc)[:200]
        results.append(
            {
                **{k: sample[k] for k in ("id", "benchmark", "tags", "label")},
                "subtask": subtask,
                "verdict": verdict,
                "correct": verdict is not None and verdict == sample["label"],
                "reply": reply[-400:],
                "error": error,
            }
        )
        streak.record(error)
        LOGGER.info(f"[{index}/{len(samples)}] {sample['id']} label={sample['label']} verdict={verdict}")
    answered = [row for row in results if row["verdict"] is not None]
    tp = sum(1 for r in answered if r["label"] and r["verdict"])
    fp = sum(1 for r in answered if not r["label"] and r["verdict"])
    tn = sum(1 for r in answered if not r["label"] and not r["verdict"])
    fn = sum(1 for r in answered if r["label"] and not r["verdict"])
    summary = {
        "samples": len(results),
        "unanswered": len(results) - len(answered),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round((tp + tn) / len(answered), 4) if answered else 0.0,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "by_benchmark": _breakdown(results, "benchmark"),
        "by_tag": _breakdown(results, "tags"),
    }
    return results, summary


def run_state_tracking(
    samples: List[Dict[str, Any]], verifier: Verifier, cameras: List[str]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    results = []
    streak = FailureStreak()
    for index, sample in enumerate(samples, start=1):
        obs = observation(sample, cameras)
        goals, correct, unanswered, error = [], 0, 0, ""
        for subtask, done, copies in distinct_queries(sample):
            try:
                verdict, _ = verifier.check(obs, subtask, required=_required(sample, copies))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warn(f"{sample['id']}: {exc!r}")
                verdict, error = None, repr(exc)[:200]
            goals.append({"subtask": subtask, "done": done, "verdict": verdict})
            correct += int(verdict is not None and verdict == done)
            unanswered += int(verdict is None)
        results.append(
            {
                **{k: sample[k] for k in ("id", "benchmark", "tags")},
                "goals": goals,
                "correct": correct == len(goals) and not unanswered,
                "n_correct": correct,
                "n_goals": len(goals),
                "unanswered": unanswered,
            }
        )
        # A sample counts as a failure only when EVERY query in it failed; a partly answered one
        # means the backend is up.
        streak.record(error if unanswered == len(goals) else "")
        LOGGER.info(f"[{index}/{len(samples)}] {sample['id']} {correct}/{len(goals)}")
    n_goals = sum(row["n_goals"] for row in results)
    summary = {
        "samples": len(results),
        "goal_queries": n_goals,
        "unanswered": sum(row["unanswered"] for row in results),
        "goal_accuracy": round(sum(row["n_correct"] for row in results) / n_goals, 4) if n_goals else 0.0,
        "exact_frames": sum(row["correct"] for row in results),
        "by_benchmark": _breakdown(results, "benchmark"),
        "by_tag": _breakdown(results, "tags"),
    }
    return results, summary


def run_next_subtask(
    samples: List[Dict[str, Any]], source: VLMSource, cameras: List[str]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    results = []
    streak = FailureStreak()
    scored = [sample for sample in samples if not all(sample["achieved"])]
    if len(scored) < len(samples):
        # With every goal done the deployed prompt says "repeat the last subtask", which needs the
        # history this benchmark deliberately has none of, so an all-done frame has no right answer.
        LOGGER.info(f"skipping {len(samples) - len(scored)} all-done sample(s)")
    for index, sample in enumerate(scored, start=1):
        source.reset(0)  # each sample stands alone: no history of earlier commands
        try:
            # The sample's own step, which is never 0, so the initial-command template shortcut
            # cannot swallow the call: a mid-episode sample must be a real generation.
            command = source.commands(
                [observation(sample, cameras)], [sample["instruction"]], [0], [0], sample["step"], None
            )[0]
            error = ""
        except Exception as exc:  # noqa: BLE001
            command, error = "", repr(exc)[:200]
        verdict = (
            classify_next(command, sample["subtasks"], sample["achieved"], sample["ordered"])
            if command
            else "invalid"
        )
        results.append(
            {
                **{k: sample[k] for k in ("id", "benchmark", "instruction", "achieved", "tags")},
                "command": command,
                "verdict": verdict,
                "correct": verdict == "valid",
                "register": in_register(command) if command else False,
                "error": error,
            }
        )
        streak.record(error)
        LOGGER.info(f"[{index}/{len(scored)}] {sample['id']} -> {command!r} ({verdict})")
    summary = {
        "samples": len(results),
        "errors": sum(1 for row in results if row["error"]),
        "valid": sum(1 for row in results if row["verdict"] == "valid"),
        "redundant": sum(1 for row in results if row["verdict"] == "redundant"),
        "invalid": sum(1 for row in results if row["verdict"] == "invalid"),
        "register": sum(1 for row in results if row["register"]),
        "by_benchmark": _breakdown(results, "benchmark"),
        "by_tag": _breakdown(results, "tags"),
    }
    return results, summary


def build_source(model: str, backend: str, max_tokens: Optional[int]) -> VLMSource:
    config = load_config(os.path.join(REPO_ROOT, GENERATOR_CONFIG))
    config["vlm"] = vlm_config(model, backend)
    if max_tokens is not None:
        config["max_tokens"] = max_tokens
    return AutoConfigurable(VLMSourceConfig(**config))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a VLM on the orchestrator samples.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--samples-dir", help="holds <suite>/subtask_states.json next to the frames")
    source.add_argument("--dataset", help="Hub dataset id instead; reads the 'subtask_states' config")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--backend", choices=BACKENDS, default="local")
    parser.add_argument("--max-tokens", type=int, help="default: the component's deployed value")
    parser.add_argument(
        "--image-cameras",
        nargs="+",
        default=["main", "wrist"],
        help="views sent per check, where the sample has them (deployed axis, e.g. RoboLab needs wrist)",
    )
    parser.add_argument(
        "--frames-dir",
        help="decomposition export to sync-check --samples-dir against (hub mode checks itself)",
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        help="downscale every frame's longest side before the check, to price accuracy against cost",
    )
    parser.add_argument("--reasoning-effort", help="thinking budget passed to the backend when set")
    parser.add_argument("--allow-drift", action="store_true", help="warn instead of fail on stale samples")
    parser.add_argument("--limit", type=int, help="first N samples only, for a smoke run")
    args = parser.parse_args()

    if args.samples_dir:
        samples = load_samples(args.samples_dir)
        if args.frames_dir:
            reference = {scene["scene"]: scene for scene in load_scenes(args.frames_dir)}
            check_reference(samples, reference, args.allow_drift)
    else:
        samples = load_hub_samples(args.dataset)
        check_reference(samples, load_hub_reference(args.dataset), args.allow_drift)
    samples = samples[: args.limit]
    LOGGER.info(f"scoring {args.model} on {len(samples)} sample(s), task={args.task}")

    if args.task == "next_subtask":
        generator = build_source(args.model, args.backend, args.max_tokens)
        results, summary = run_next_subtask(samples, generator, args.image_cameras)
    else:
        verifier = Verifier(
            VerifierConfig(
                vlm=vlm_config(args.model, args.backend),
                image_cameras=args.image_cameras,
                max_image_side=args.max_image_side,
                reasoning_effort=args.reasoning_effort,
                **({"max_tokens": args.max_tokens} if args.max_tokens is not None else {}),
            )
        )
        run = run_verification if args.task == "verification" else run_state_tracking
        results, summary = run(samples, verifier, args.image_cameras)

    summary = {"model": args.model, "task": args.task, **summary}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump({"summary": summary, "results": results}, handle, indent=1)
    LOGGER.info(f"{json.dumps(summary, indent=1)}\nwrote {args.out}")


if __name__ == "__main__":
    main()
