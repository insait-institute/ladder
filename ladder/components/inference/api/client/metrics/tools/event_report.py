import argparse
import collections
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from ladder.components.inference.api.client import scene_events
from ladder.components.inference.api.client.episode import Episode
from ladder.components.inference.api.client.metrics import aggregate

# Per-arm failure causes, grasp fates, per-object grasp reliability, controller decisions and
# per-episode timelines from the object-level trace in recorded episode logs.
#   bazel run //ladder/components/inference/api/client:event_report -- <campaign dir> [...]
#       [--arms a,b] [--timeline <unit>] [--json <out.json>]

_ARM = re.compile(r"^(?P<arm>.+)_c\d+(\.a\d+)?$")


def _load_log(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return yaml.load(handle, Loader=yaml.UnsafeLoader) or []  # noqa: S506  # our own log


def find_logs(roots: Iterable[str]) -> List[Tuple[str, str, str]]:
    """(arm, unit, path) per cell log, skipping each lane's private copy of the cell."""
    found = []
    for root in roots:
        for dirpath, _, files in os.walk(root):
            if (
                "episode_log.yaml" not in files
                or os.sep + "lane" in dirpath
                or os.path.basename(dirpath) == "all"
            ):
                continue
            arm = next(
                (match.group("arm") for part in dirpath.split(os.sep) if (match := _ARM.match(part))), "?"
            )
            found.append((arm, os.path.basename(dirpath), os.path.join(dirpath, "episode_log.yaml")))
    return sorted(found)


def episodes_by_arm(
    logs: List[Tuple[str, str, str]], arms: Optional[List[str]]
) -> Dict[str, List[Tuple[str, Episode]]]:
    out: Dict[str, List[Tuple[str, Episode]]] = collections.defaultdict(list)
    for arm, unit, path in logs:
        if arms and arm not in arms:
            continue
        for entry in _load_log(path):
            out[arm].append((unit, Episode.from_log_entry(entry)))
    return out


def grasp_reliability(episodes: List[Episode]) -> Dict[str, Dict[str, int]]:
    """Per object: grasps begun and how each ended."""
    table: Dict[str, Dict[str, int]] = collections.defaultdict(collections.Counter)
    for episode in episodes:
        for grasp in scene_events.grasps(episode):
            table[grasp.object]["grasps"] += 1
            table[grasp.object][grasp.outcome] += 1
    return {name: dict(counts) for name, counts in sorted(table.items())}


def knock_sources(episodes: List[Episode]) -> Dict[str, int]:
    """What the robot held when a finished goal was knocked apart."""
    counts: collections.Counter = collections.Counter()
    for episode in episodes:
        for event in scene_events.displacements(episode):
            if event.regressed_goal is not None:
                counts["holding " + ", ".join(event.holding) if event.holding else "holding nothing"] += 1
    return dict(counts.most_common())


def arm_summary(episodes: List[Episode]) -> Dict[str, Any]:
    traced = [episode for episode in episodes if episode.object_trace]
    failed = [episode for episode in traced if not episode.success]
    causes = collections.Counter(scene_events.failure_cause(episode) for episode in failed)
    return {
        "episodes": len(episodes),
        "with_scene_trace": len(traced),
        "failed": len(failed),
        "failure_causes": {cause: causes[cause] for cause in scene_events.CAUSES if causes[cause]},
        "metrics": aggregate(traced),
        "knock_sources": knock_sources(traced),
        "grasp_reliability": grasp_reliability(traced),
    }


def _percent(fraction: str) -> str:
    numerator, denominator = (float(part) for part in fraction.split("/"))
    if not denominator or numerator > denominator:  # a count or a mean, not a rate
        return fraction
    return f"{fraction} = {100 * numerator / denominator:.1f}%"


def print_report(summaries: Dict[str, Dict[str, Any]]) -> None:
    arms = list(summaries)
    width = max([len(arm) for arm in arms] + [22]) + 2
    print("arm".ljust(28) + "".join(arm.rjust(width) for arm in arms))
    for key in ("episodes", "with_scene_trace", "failed"):
        print(key.ljust(28) + "".join(str(summaries[arm][key]).rjust(width) for arm in arms))
    print("-- failure cause (episodes)")
    for cause in scene_events.CAUSES:
        row = [summaries[arm]["failure_causes"].get(cause, 0) for arm in arms]
        if any(row):
            print(("  " + cause).ljust(28) + "".join(str(count).rjust(width) for count in row))
    print("-- metrics")
    names = sorted({name for arm in arms for name in summaries[arm]["metrics"]})
    for name in names:
        cells = [
            _percent(summaries[arm]["metrics"][name]) if name in summaries[arm]["metrics"] else ""
            for arm in arms
        ]
        print(("  " + name).ljust(28) + "".join(cell.rjust(width) for cell in cells))
    for arm in arms:
        print(f"-- {arm}: what was held when a finished goal was knocked apart")
        for source, count in summaries[arm]["knock_sources"].items():
            print(f"  {count:4d}  {source}")
        print(f"-- {arm}: grasp reliability per object (grasps -> {' / '.join(scene_events.OUTCOMES)})")
        for name, counts in summaries[arm]["grasp_reliability"].items():
            fate = " / ".join(str(counts.get(outcome, 0)) for outcome in scene_events.OUTCOMES)
            print(f"  {name:28s} {counts.get('grasps', 0):4d} -> {fate}")


def print_timelines(by_arm: Dict[str, List[Tuple[str, Episode]]], unit: str) -> None:
    for arm, entries in by_arm.items():
        for index, (_unit, episode) in enumerate(entry for entry in entries if entry[0] == unit):
            cause = scene_events.failure_cause(episode)
            head = f"=== {arm} / {unit} / episode {index}: {'SUCCESS' if episode.success else 'FAIL'}"
            print(head + (f" ({cause})" if cause else "") + f", {episode.num_steps} steps")
            print(f"    goals: {episode.goal_objects}")
            for step, line in scene_events.timeline(episode):
                print(f"    {step:5d}  {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Object-level failure report over composition episode logs.")
    parser.add_argument(
        "roots", nargs="+", help="campaign directories (or any directory holding episode_log.yaml files)"
    )
    parser.add_argument("--arms", default=None, help="comma-separated arms to include (default: all)")
    parser.add_argument(
        "--timeline", default=None, help="print every episode of this unit as an event timeline"
    )
    parser.add_argument("--json", default=None, help="also write the per-arm summaries here")
    args = parser.parse_args()

    by_arm = episodes_by_arm(find_logs(args.roots), args.arms.split(",") if args.arms else None)
    if not by_arm:
        raise SystemExit("no episode_log.yaml under " + ", ".join(args.roots))
    summaries = {
        arm: arm_summary([episode for _, episode in entries]) for arm, entries in sorted(by_arm.items())
    }
    print_report(summaries)
    if args.timeline:
        print_timelines(by_arm, args.timeline)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(summaries, handle, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
