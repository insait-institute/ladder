import argparse
import os
import shutil
from typing import Dict, List

import yaml

from ladder.components.inference.api.client.episode import Episode
from ladder.components.inference.api.client.metrics import aggregate

# Recompute every results.yaml under a root from the episode_log.yaml beside it with the current
# metric registry; the previous file is kept once as results.yaml.orig.
#   bazel run //ladder/components/inference/api/client:rescore -- <campaign dir> [...] [--dry-run]


def rescore(cell_dir: str, dry_run: bool = False) -> Dict[str, int]:
    with open(os.path.join(cell_dir, "episode_log.yaml"), encoding="utf-8", errors="replace") as handle:
        entries = yaml.load(handle, Loader=yaml.UnsafeLoader) or []  # noqa: S506  # our own log
    by_cell: Dict[str, Dict[str, List[Episode]]] = {}
    for entry in entries:
        episode = Episode.from_log_entry(entry)
        by_cell.setdefault(episode.robot_name, {}).setdefault(episode.env_name, []).append(episode)
    stats = {
        robot: {env: aggregate(episodes) for env, episodes in envs.items()} for robot, envs in by_cell.items()
    }
    results_path = os.path.join(cell_dir, "results.yaml")
    if not dry_run:
        backup = results_path + ".orig"
        if os.path.exists(results_path) and not os.path.exists(backup):
            shutil.copy2(results_path, backup)
        with open(results_path, "w") as handle:
            yaml.dump(stats, handle, sort_keys=True)
    return {
        f"{robot}/{env}": len(episodes) for robot, envs in by_cell.items() for env, episodes in envs.items()
    }


def cell_dirs(roots: List[str]) -> List[str]:
    """Directories holding an episode_log.yaml, skipping each lane's private copy of the cell."""
    found = []
    for root in roots:
        for dirpath, _, files in os.walk(root):
            if (
                "episode_log.yaml" in files
                and os.sep + "lane" not in dirpath
                and os.path.basename(dirpath) != "all"
            ):
                found.append(dirpath)
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute results.yaml from episode_log.yaml with the current metrics."
    )
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dirs = cell_dirs(args.roots)
    if not dirs:
        raise SystemExit("no episode_log.yaml under " + ", ".join(args.roots))
    for cell in dirs:
        counts = rescore(cell, dry_run=args.dry_run)
        verb = "would rescore" if args.dry_run else "rescored"
        print(f"{verb} {cell}: " + ", ".join(f"{cell_name} x{count}" for cell_name, count in counts.items()))


if __name__ == "__main__":
    main()
