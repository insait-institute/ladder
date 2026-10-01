import argparse
import os
import pathlib
from typing import List

import huggingface_hub

from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)

SUITES = {"simplerenv_long": "simpler", "robotarena_long": "robot_arena", "robolab_long": "robolab"}
LEARNED = {
    "low_level_policy",
    "periodic_trigger",
    "proprioceptive_trigger",
    "proprioceptive_verifier",
    "volo",
    "privileged_verifier",
}


def fetch_policy(repo: str, out: pathlib.Path) -> None:
    huggingface_hub.snapshot_download(repo_id=repo, local_dir=out)
    LOGGER.info(f"{repo} -> {out}")


def fetch_episodes(repo: str, out: pathlib.Path, suites: List[str], methods: List[str], videos: bool) -> None:
    patterns = ["README.md", "index.parquet"]
    patterns += [f"{suite}/{method}/*" for suite in suites or SUITES for method in methods or ["*"]]
    if not suites:
        patterns += [f"real_world/*/{method}/*" for method in methods or ["*"]] + ["real_world/*.csv"]
    huggingface_hub.snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=out,
        allow_patterns=patterns,
        ignore_patterns=None if videos else ["*.mp4"],
        max_workers=32,
    )
    results = [str(link_view(out))] + ([str(out / "real_world")] if (out / "real_world").is_dir() else [])
    LOGGER.info(f"done: bazel run //ladder/components/dashboard:serve -- --results {' '.join(results)}")


def link_view(out: pathlib.Path) -> pathlib.Path:
    view = out / "dashboard"
    for method_dir in sorted(p for suite in SUITES for p in (out / suite).glob("*") if p.is_dir()):
        suite, method = method_dir.parent.name, method_dir.name
        if method == "scripted":
            continue
        policy = ("pi05" if suite == "robolab_long" else "bridge") if method in LEARNED else "none"
        link = view / SUITES[suite] / policy / f"{method}_c1" / suite
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(os.path.relpath(method_dir, link.parent))
    return view


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a low-level policy or the evaluation episodes.")
    sub = parser.add_subparsers(dest="what", required=True)
    policy = sub.add_parser("policy", help="a policy session directory, e.g. the Bridge policy")
    policy.add_argument("--repo", required=True, help="Hugging Face model id")
    policy.add_argument("--out", type=pathlib.Path, required=True)
    episodes = sub.add_parser("episodes", help="the evaluation episodes, for the dashboard")
    episodes.add_argument("--repo", required=True, help="Hugging Face dataset id")
    episodes.add_argument("--out", type=pathlib.Path, required=True)
    episodes.add_argument("--suite", nargs="+", default=[], choices=sorted(SUITES), help="default: all")
    episodes.add_argument("--method", nargs="+", default=[], help="default: all")
    episodes.add_argument("--no-videos", dest="videos", action="store_false", help="the logs only")
    args = parser.parse_args()
    out = args.out.expanduser()
    if args.what == "policy":
        fetch_policy(args.repo, out)
    else:
        fetch_episodes(args.repo, out, args.suite, args.method, args.videos)


if __name__ == "__main__":
    main()
