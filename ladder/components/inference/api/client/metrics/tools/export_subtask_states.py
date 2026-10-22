import argparse
import dataclasses
import datetime
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import yaml

from ladder.components.inference.api.client.episode import Episode
from ladder.components.inference.api.client.metrics.plan import goal_of, normalize
from ladder.components.inference.api.client.metrics.tools.scenes import load_scenes
from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)

# A sample is cut a fixed distance from the goal-flip that motivates it, and only where the goal
# vector holds still around it: the simulator's predicate flips several steps after a release and
# can flicker while an object rolls, so a frame near a transition carries a noisy label.
SETTLE_STEPS = 8
# In-progress negatives are cut this many settle-windows BEFORE a goal completes, clear of the
# hover phase right before the flip, where the predicate (which counts a held object at the goal
# surface as placed) may disagree with the frame by a step or two.
NEGATIVE_LEAD = 3
# The recording is trusted only when it has (nearly) one frame per step; a bigger gap means the
# video and the log disagree about time and every label cut from it would be misaligned.
FRAME_COUNT_TOLERANCE = 5

_ENV_VERSION = re.compile(r"-v\d+$")
_ARM = re.compile(r"^(?P<arm>.+)_c\d+(\.a\d+)?$")


def scene_of(env_name: str) -> str:
    """The exporter's scene name for a logged env name: RobotArena prefixes the suite
    ('robot_arena_scene36_2goal'), SimplerEnv suffixes a gym version ('all2_plate-v0')."""
    name = _ENV_VERSION.sub("", env_name.strip())
    return name.removeprefix("robot_arena_")


def achieved_at(goal_trace: Sequence[Tuple[int, List[bool]]], step: int) -> Optional[List[bool]]:
    """The goal vector at ``step``: the last change point at or before it, None before the first."""
    value = None
    for at, achieved in goal_trace:
        if at > step:
            break
        value = achieved
    return list(value) if value is not None else None


def stable_at(goal_trace: Sequence[Tuple[int, List[bool]]], step: int, margin: int) -> bool:
    """No change point within ``margin`` steps of ``step``: the label holds on the whole window,
    so an off-by-a-few frame index cannot flip it."""
    return not any(step - margin < at <= step + margin for at, _ in goal_trace[1:])


def _held_at(episode: Episode, step: int) -> bool:
    """Whether any arm was holding at ``step``."""
    return any(
        closed <= step and (released is None or step < released) for closed, released in episode.grasps
    )


@dataclasses.dataclass
class Candidate:
    step: int
    achieved: List[bool]
    query_idx: int
    label: bool
    tag: str  # fresh | in_progress | cold | regressed | terminal
    held: bool
    episode: str  # episode recording dir name


def _duplicate_groups(subtasks: Sequence[str]) -> List[List[int]]:
    groups: Dict[str, List[int]] = {}
    for idx, subtask in enumerate(subtasks):
        groups.setdefault(normalize(subtask), []).append(idx)
    return [groups[normalize(subtask)] for subtask in subtasks]


def episode_candidates(episode: Episode, episode_dir: str) -> List[Candidate]:
    """Every sample this episode could contribute, one per motivating event.

    A completion yields a positive just after its flip ('fresh'); a completion also yields an
    in-progress negative well before it, where the goal is demonstrably still undone; a goal that
    was completed and then knocked undone yields a 'regressed' negative just after the loss; the
    final settled frame yields a 'terminal' sample on whichever goal describes it best. A query on a
    subtask that appears several times in the plan is kept only while all of its copies agree, since
    a yes/no on the TEXT cannot single out one instance.
    """
    trace = episode.goal_trace
    if not trace or not episode.subtasks:
        return []
    groups = _duplicate_groups(episode.subtasks)
    last_step = episode.num_steps - 1

    def candidate(step: int, query_idx: int, tag: str) -> Optional[Candidate]:
        if not 0 <= step <= last_step or not stable_at(trace, step, SETTLE_STEPS):
            return None
        achieved = achieved_at(trace, step)
        if achieved is None or query_idx >= len(achieved):
            return None
        group_values = {achieved[i] for i in groups[query_idx]}
        if len(group_values) > 1:
            return None
        return Candidate(
            step=step,
            achieved=achieved,
            query_idx=query_idx,
            label=achieved[query_idx],
            tag=tag,
            held=_held_at(episode, step),
            episode=episode_dir,
        )

    candidates = []
    previous = trace[0][1]
    for at, achieved in trace[1:]:
        for goal, (was, now) in enumerate(zip(previous, achieved, strict=False)):
            if was == now:
                continue
            if now:
                candidates.append(candidate(at + SETTLE_STEPS, goal, "fresh"))
                candidates.append(candidate(at - NEGATIVE_LEAD * SETTLE_STEPS, goal, "in_progress"))
            else:
                candidates.append(candidate(at + SETTLE_STEPS, goal, "regressed"))
        previous = achieved

    # A cold negative from before anything happened, on a goal the episode later completes if there
    # is one (so the frame's undone state is unambiguous), else on goal 0.
    first_completed = next(
        (goal for goal in range(len(episode.subtasks)) if episode.first_achieved(goal) is not None), 0
    )
    candidates.append(candidate(trace[0][0] + SETTLE_STEPS, first_completed, "cold"))

    final = achieved_at(trace, last_step) or []
    terminal_goal = (
        next((g for g, done in enumerate(final) if done), None)
        if all(final)
        else next((g for g, done in enumerate(final) if not done), None)
    )
    if terminal_goal is not None:
        candidates.append(candidate(last_step - SETTLE_STEPS, terminal_goal, "terminal"))

    # Drop the label the tag promised but the trace took back (e.g. an in-progress cut that lands
    # after an earlier completion of the same goal): a mislabeled tag poisons the per-tag breakdown.
    expected = {"fresh": True, "in_progress": False, "cold": False, "regressed": False}
    return [c for c in candidates if c is not None and (c.tag not in expected or c.label == expected[c.tag])]


# Positive slots prefer the hover case (held while counting as placed: the measured hard case for
# a verifier), then a fresh completion over a terminal frame; negatives prefer a goal the policy is
# demonstrably mid-way through over one it never touched.
_POSITIVE_RANK = {"fresh": 1, "terminal": 2}
_NEGATIVE_RANK = {"in_progress": 0, "regressed": 1, "cold": 2, "terminal": 3}


def select(candidates: List[Candidate], per_task: int) -> List[Candidate]:
    """At most ``per_task`` samples for one task: a positive, a negative, then a 'regressed' frame
    when the task produced one, then whatever balances the two classes. Deterministic (sorted, no
    RNG) and spread over distinct episodes and goals before repeating either."""

    positives = sorted(
        (c for c in candidates if c.label),
        key=lambda c: (-int(c.held), _POSITIVE_RANK.get(c.tag, 9), c.episode, c.step),
    )
    negatives = sorted(
        (c for c in candidates if not c.label),
        key=lambda c: (_NEGATIVE_RANK.get(c.tag, 9), -int(c.held), c.episode, c.step),
    )

    chosen: List[Candidate] = []

    def take(pool: List[Candidate]) -> bool:
        used_eps = {c.episode for c in chosen}
        used_goals = {c.query_idx for c in chosen}
        keys = {(c.episode, c.step, c.query_idx) for c in chosen}
        for prefer in (
            lambda c: c.episode not in used_eps and c.query_idx not in used_goals,
            lambda c: c.episode not in used_eps,
            lambda c: True,
        ):
            for c in pool:
                if (c.episode, c.step, c.query_idx) not in keys and prefer(c):
                    chosen.append(c)
                    return True
        return False

    take(positives) or take(negatives)
    if len(chosen) < per_task:
        take(negatives) or take(positives)
    regressed = [c for c in negatives if c.tag == "regressed"]
    while len(chosen) < per_task:
        n_pos = sum(c.label for c in chosen)
        pools = [regressed] if regressed and not any(c.tag == "regressed" for c in chosen) else []
        pools += [positives, negatives] if n_pos <= len(chosen) - n_pos else [negatives, positives]
        if not any(take(pool) for pool in pools):
            break
    return sorted(chosen, key=lambda c: (c.episode, c.step))


def load_reference(frames_dir: str) -> Dict[str, Dict[str, Any]]:
    """The decomposition benchmark's manifests, keyed by scene name. That export is regenerated from
    the task registries whenever a task changes, which is what makes it the sync anchor here: a
    sample is only ever cut against the reference the suites currently define."""
    reference = {scene["scene"]: scene for scene in load_scenes(frames_dir)}
    if not reference:
        raise FileNotFoundError(f"no <suite>/scenes.json under {frames_dir}")
    return reference


def _goal_keys(steps: Sequence[str]) -> List[Tuple[str, str]]:
    return [goal_of(step) or ("__raw__", normalize(step)) for step in steps]


def matches_reference(episode: Episode, reference: Dict[str, Any]) -> bool:
    """The stale-recording guard: an episode is usable only while the env's own published
    decomposition still equals the current reference, subtask for subtask. A recording from before
    a task edit fails this and is skipped rather than silently mislabeling the new task.

    Only ``subtasks`` is compared: the logged instruction is whatever the policy was last commanded,
    which on an orchestrated run is a subtask rather than the compound task, while the logged
    subtasks are the ENV's reference and identical whatever system ran. An unordered task is
    compared as a MULTISET: the env and the exporter list the same goals in different orders, and
    only on an ordered task is that order part of the task. Sample rows keep the EPISODE's order,
    which is what the goal vector indexes."""
    logged = _goal_keys(episode.subtasks)
    expected = _goal_keys(reference["subtasks"])
    if reference.get("ordered"):
        return logged == expected
    return sorted(logged) == sorted(expected)


def _recording_dirs(unit_dir: str) -> List[str]:
    for name in ("episodes", "rec"):
        root = os.path.join(unit_dir, name)
        if os.path.isdir(root):
            return sorted(
                os.path.join(root, entry)
                for entry in os.listdir(root)
                if os.path.isdir(os.path.join(root, entry))
            )
    return []


def _recording_of(episode: Episode, index: int, recordings: List[str]) -> Optional[str]:
    """The recording dir of log entry ``index``, verified against its episode_meta.yaml:
    ``episode_index`` is 1-based log order, and a num_steps/success disagreement means the
    recordings and the log are not from the same run."""
    for directory in recordings:
        meta_path = os.path.join(directory, "episode_meta.yaml")
        if not os.path.exists(meta_path):
            continue
        meta = yaml.safe_load(open(meta_path))
        if meta.get("episode_index") != index + 1:
            continue
        if meta.get("num_steps") == episode.num_steps and meta.get("success") == episode.success:
            return directory
        LOGGER.warn(f"{directory}: episode_meta disagrees with the log entry; skipping")
        return None
    return None


def _extract_frames(recording: str, steps: List[int], num_steps: int) -> Optional[Dict[str, Dict[int, Any]]]:
    """{camera: {step: BGR frame}} for every <camera>.mp4 in the recording, or None when the main
    video's frame count does not match the log's step count (a misaligned video mislabels every
    frame cut from it, so the whole recording is refused)."""
    frames: Dict[str, Dict[int, Any]] = {}
    for video in sorted(glob.glob(os.path.join(recording, "*.mp4"))):
        camera = os.path.splitext(os.path.basename(video))[0]
        capture = cv2.VideoCapture(video)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if abs(count - num_steps) > FRAME_COUNT_TOLERANCE:
            LOGGER.warn(f"{video}: {count} frames for {num_steps} steps; skipping recording")
            capture.release()
            return None
        for step in steps:
            capture.set(cv2.CAP_PROP_POS_FRAMES, min(step, count - 1))
            ok, image = capture.read()
            if ok:
                frames.setdefault(camera, {})[step] = image
        capture.release()
    return frames if frames.get("main") else None


def _arm_of(path: str) -> str:
    for part in reversed(path.split(os.sep)):
        match = _ARM.match(part)
        if match:
            return match.group("arm")
    return ""


def _write_scene(
    scene: str,
    chosen: List[Candidate],
    ref: Dict[str, Any],
    sources: Dict[Tuple[str, str], Tuple[str, str, Episode]],
    out_dir: str,
) -> List[Dict[str, Any]]:
    """One task's selected samples as manifest rows, with their PNG frames written next to them."""
    suite_dir = os.path.join(out_dir, ref["benchmark"])
    os.makedirs(suite_dir, exist_ok=True)
    rows = []
    for sample_idx, c in enumerate(chosen):
        source, arm, episode = sources[(scene, c.episode)]
        frames = _extract_frames(c.episode, [c.step], episode.num_steps)
        if frames is None:
            continue
        names = {}
        for camera, by_step in frames.items():
            name = f"{scene}__s{sample_idx}__{camera}.png"
            cv2.imwrite(os.path.join(suite_dir, name), by_step[c.step])
            names[camera] = name
        rows.append(
            {
                "id": f"{ref['benchmark']}/{scene}/s{sample_idx}",
                "benchmark": ref["benchmark"],
                "scene": scene,
                "instruction": ref["instruction"],
                # The EPISODE's subtask order, not the reference's: ``achieved`` indexes it, and
                # on an unordered task the two list the same goals in different orders.
                "subtasks": list(episode.subtasks),
                "n_goals": ref["n_goals"],
                # The scene contents this frame was cut under, so a later contents change is visible.
                "objects": list(ref.get("objects", [])),
                "ordered": bool(ref.get("ordered", False)),
                "step": c.step,
                "num_steps": episode.num_steps,
                "achieved": list(c.achieved),
                "query_idx": c.query_idx,
                "label": c.label,
                "tags": [c.tag] + (["held"] if c.held else []),
                "arm": arm,
                "source": source,
                "frames": names,
            }
        )
    return rows


def harvest(
    episode_roots: List[str],
    reference: Dict[str, Dict[str, Any]],
    out_dir: str,
    per_task: int,
    benchmarks: Optional[List[str]] = None,
    since: Optional[float] = None,
) -> None:
    """Walk every episode_log.yaml under ``episode_roots``, pool candidates per task, select
    ``per_task`` of them, and write one <suite>/subtask_states.json + PNG frames per suite under
    ``out_dir``: the same relocatable manifest contract the decomposition export uses.

    ``since`` drops logs written before that epoch, PER FILE: the reference guard catches a task
    whose plan changed, but a geometry-only rebuild leaves the plan intact, so recordings from
    before one can only be excluded by when they were written (a campaign can straddle it)."""
    skipped = {
        "no_composition": 0,
        "stale_reference": 0,
        "unknown_scene": 0,
        "no_recording": 0,
        "too_old": 0,
        "lane_aggregate": 0,
    }
    by_scene: Dict[str, List[Candidate]] = {}
    sources: Dict[Tuple[str, str], Tuple[str, str, Episode]] = {}  # (scene, episode dir) -> origin

    for root in episode_roots:
        for log_path in sorted(glob.glob(os.path.join(root, "**", "episode_log.yaml"), recursive=True)):
            if any(re.fullmatch(r"lane\d+", part) for part in log_path.split(os.sep)):
                skipped["lane_aggregate"] += 1
                continue
            if since is not None and os.path.getmtime(log_path) < since:
                skipped["too_old"] += 1
                continue
            unit_dir = os.path.dirname(log_path)
            recordings = _recording_dirs(unit_dir)
            entries = yaml.safe_load(open(log_path)) or []
            for index, entry in enumerate(entries):
                episode = Episode.from_log_entry(entry)
                if not episode.goal_trace or not episode.subtasks:
                    skipped["no_composition"] += 1
                    continue
                scene = scene_of(episode.env_name)
                ref = reference.get(scene)
                if ref is None:
                    skipped["unknown_scene"] += 1
                    continue
                if benchmarks and ref["benchmark"] not in benchmarks:
                    continue
                if not matches_reference(episode, ref):
                    skipped["stale_reference"] += 1
                    continue
                recording = _recording_of(episode, index, recordings)
                if recording is None:
                    skipped["no_recording"] += 1
                    continue
                for c in episode_candidates(episode, recording):
                    by_scene.setdefault(scene, []).append(c)
                sources[(scene, recording)] = (os.path.relpath(unit_dir, root), _arm_of(log_path), episode)

    manifests: Dict[str, List[Dict[str, Any]]] = {}
    for scene in sorted(by_scene):
        ref = reference[scene]
        rows = _write_scene(scene, select(by_scene[scene], per_task), ref, sources, out_dir)
        manifests.setdefault(ref["benchmark"], []).extend(rows)

    for benchmark, rows in manifests.items():
        with open(os.path.join(out_dir, benchmark, "subtask_states.json"), "w") as handle:
            json.dump(rows, handle, indent=1)

    written = sum(len(rows) for rows in manifests.values())
    tasks = {row["scene"] for rows in manifests.values() for row in rows}
    positives = sum(row["label"] for rows in manifests.values() for row in rows)
    LOGGER.info(f"{written} sample(s) across {len(tasks)} task(s) ({positives} positive); skipped {skipped}")
    relevant = {s for s, ref in reference.items() if not benchmarks or ref["benchmark"] in benchmarks}
    missing = sorted(relevant - tasks)
    if missing:
        LOGGER.warn(
            f"{len(missing)} reference task(s) without samples: {missing[:20]}{'...' if len(missing) > 20 else ''}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cut orchestrator samples (frame + per-goal ground truth) from recorded episodes."
    )
    parser.add_argument("--episodes", nargs="+", required=True, help="campaign/session roots to walk")
    parser.add_argument(
        "--frames-dir",
        required=True,
        help="decomposition export holding <suite>/scenes.json (the sync anchor)",
    )
    parser.add_argument("--out-dir", required=True)
    # 10, which is what the published benchmark is built with. Much smaller values leave confidence
    # intervals too wide to order the top models and skew the class balance.
    parser.add_argument("--per-task", type=int, default=10)
    parser.add_argument("--benchmarks", nargs="*", help="restrict to these benchmark names")
    parser.add_argument(
        "--since", help="ISO timestamp (UTC); episode logs written before it are skipped, per file"
    )
    args = parser.parse_args()

    since = (
        datetime.datetime.fromisoformat(args.since).replace(tzinfo=datetime.timezone.utc).timestamp()
        if args.since
        else None
    )
    reference = load_reference(args.frames_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    harvest(args.episodes, reference, args.out_dir, args.per_task, args.benchmarks, since)


if __name__ == "__main__":
    main()
