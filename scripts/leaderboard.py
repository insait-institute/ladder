#!/usr/bin/env python3
"""Leaderboard submissions: package your results, check a submission, build the leaderboard's data.

  scripts/leaderboard.py package --results out --name my_agent   # writes leaderboard/submissions/my_agent/
  scripts/leaderboard.py check leaderboard/submissions/my_agent   # what CI runs on a pull request
  scripts/leaderboard.py build                                    # docs/static/data/submissions.json

A submission is a directory with submission.yaml (who, what, links) and episodes.jsonl (one line per
episode, read from the evaluation's own episode logs). Every number on the leaderboard is recomputed
from episodes.jsonl, so nothing is typed in by hand. leaderboard/README.md has the whole process.
Needs Python 3.9+ and PyYAML (pip install pyyaml).
"""

import argparse
import datetime
import glob
import json
import math
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = yaml.safe_load(open(os.path.join(ROOT, "leaderboard", "tasks.yaml")))
SUITES = {"simplerenv_long": 24, "robotarena_long": 8, "robolab_long": 8}  # episodes per task
# the suite's directory name in scripts/eval.sh's output
SUITE_DIRS = {"simpler": "simplerenv_long", "robot_arena": "robotarena_long", "robolab": "robolab_long"}
KINDS = ["policy", "orchestrator", "agent"]
REQUIRED = [
    "name",
    "kind",
    "authors",
    "contact",
    "results_url",
    "low_level_policy",
    "vlm",
    "inputs",
    "ladder_commit",
    "description",
    "privileged",
    "tuned_on_ladder",
]
Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

TEMPLATE = """\
# Your system on the LADDER leaderboard. Fill in every field; leaderboard/README.md explains them.
name: {name}                     # as shown on the leaderboard
kind: {kind}                     # policy | orchestrator | agent
authors: ""                      # e.g. "Ada Lovelace, Alan Turing"
organization: ""
contact: ""                      # an email address for questions about this entry
paper_url: ""                    # optional
code_url: ""                     # optional: a public implementation lets the maintainers verify it
results_url: ""                  # the full results directory (episode logs and videos), e.g. a Hugging Face dataset
low_level_policy: ""             # e.g. "Bridge / pi0.5" (the suites' own), your policy's name, or "none"
vlm: ""                          # the model(s) your system calls, or "none"
inputs: "RGB, proprioception"    # what the system reads, e.g. "RGB, depth, proprioception"
ladder_commit: {commit}             # the LADDER commit you evaluated with
date: {date}
privileged: false                # true if the system reads simulator state a real robot would not have
tuned_on_ladder: false           # true if any part was trained or tuned on LADDER's tasks or scenes
description: >-
  One paragraph: what the system is and how it decides what to do.
verified: false                  # set by the maintainers once they have re-run the system
"""


def episodes_of(results_dir):
    """One record per episode under an eval output directory (scripts/eval.sh's layout)."""
    rows = []
    for log in sorted(glob.glob(os.path.join(results_dir, "**", "episode_log.yaml"), recursive=True)):
        parts = os.path.relpath(log, results_dir).split(os.sep)
        suite = next((SUITE_DIRS[p] for p in parts if p in SUITE_DIRS), None) or next(
            (p for p in parts if p in SUITES), None
        )
        task = parts[-2]
        if suite is None or task not in TASKS[suite]:
            continue
        for i, e in enumerate(yaml.load(open(log), Loader=Loader) or []):
            rows.append(
                {
                    "suite": suite,
                    "task": task,
                    "episode": i,
                    "success": bool(e["success"]),
                    "subgoals_done": subgoals_done(e),
                    "subgoals": TASKS[suite][task]["subgoals"],
                    "steps": int(e.get("num_steps", 0)),
                }
            )
    return rows


def subgoals_done(episode):
    """Subgoals met at the end of an episode, each counted only if it was met after every earlier
    subgoal of the same object (a Relay's or Revisit's legs are ordered). RoboLab checks each leg on
    its own, so without this it would credit "the lemon in the bin" when the lemon skipped the
    plate; the WidowX suites already order them. A successful episode has met every subgoal."""
    objects, trace = episode.get("goal_objects") or [], episode.get("goal_trace") or []
    if episode["success"]:
        return len(objects)
    first = [
        next((step for step, met in trace if g < len(met) and met[g]), None) for g in range(len(objects))
    ]
    final = trace[-1][1] if trace else []
    done = 0
    for g, (source, *_) in enumerate(objects):
        earlier = [first[leg] for leg in range(g) if objects[leg][0] == source]
        ready = None if None in earlier else max(earlier, default=-1)
        done += bool(
            g < len(final)
            and final[g]
            and ready is not None
            and any(step > ready and g < len(met) and met[g] for step, met in trace)
        )
    return done


def read_episodes(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def problems(submission_dir):
    """Everything wrong with a submission, as sentences; empty when it is valid."""
    out = []
    meta_path = os.path.join(submission_dir, "submission.yaml")
    rows_path = os.path.join(submission_dir, "episodes.jsonl")
    if not os.path.exists(meta_path) or not os.path.exists(rows_path):
        return [f"{submission_dir} needs submission.yaml and episodes.jsonl"]
    meta = yaml.safe_load(open(meta_path)) or {}
    for key in REQUIRED:
        if meta.get(key) in (None, ""):
            out.append(f"submission.yaml: {key} is empty")
    if meta.get("kind") not in KINDS:
        out.append(f"submission.yaml: kind must be one of {KINDS}")
    if "One paragraph" in str(meta.get("description", "")):
        out.append("submission.yaml: description is still the template's")
    rows = read_episodes(rows_path)
    by_task = {}
    for r in rows:
        if r["suite"] not in SUITES or r["task"] not in TASKS[r["suite"]]:
            out.append(f"episodes.jsonl: unknown task {r['suite']}/{r['task']}")
            continue
        if (
            r["subgoals"] != TASKS[r["suite"]][r["task"]]["subgoals"]
            or not 0 <= r["subgoals_done"] <= r["subgoals"]
        ):
            out.append(
                f"episodes.jsonl: {r['suite']}/{r['task']} episode {r['episode']} has wrong subgoal counts"
            )
        if r["success"] and r["subgoals_done"] != r["subgoals"]:
            out.append(
                f"episodes.jsonl: {r['suite']}/{r['task']} episode {r['episode']} succeeded with subgoals left"
            )
        by_task.setdefault((r["suite"], r["task"]), []).append(r)
    suites = sorted({s for s, _ in by_task})
    if not suites:
        out.append("episodes.jsonl: no episodes")
    for suite in suites:
        for task in TASKS[suite]:
            n = len(by_task.get((suite, task), []))
            if n != SUITES[suite]:
                out.append(f"{suite}/{task}: {n} episodes, the suite runs {SUITES[suite]}")
    return out


def wilson(k, n, z=1.96):
    """The 95% Wilson interval of k successes in n, as the larger distance (in percent) from k/n to
    either of its ends: the interval is not symmetric, and near 0 or 100 the far side is the wider."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    spread = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * max(centre + spread - p, p - (centre - spread))


def score(rows):
    """The leaderboard's numbers for one system, in the paper's definitions: success rate per suite
    (episodes pooled) with its Wilson interval, their mean over the three suites, the rate over all
    episodes, subgoal progress, and success by horizon and by kind of task."""

    def rate(sel):
        return (
            {
                "v": round(100 * sum(r["success"] for r in sel) / len(sel), 1),
                "pm": round(wilson(sum(r["success"] for r in sel), len(sel)), 1),
            }
            if sel
            else None
        )

    def progress(sel):
        return round(100 * sum(r["subgoals_done"] for r in sel) / sum(r["subgoals"] for r in sel), 1)

    suites = {s: [r for r in rows if r["suite"] == s] for s in SUITES}
    suites = {s: sel for s, sel in suites.items() if sel}
    out = {
        "suites": {s: rate(sel) for s, sel in suites.items()},
        "progress": {s: progress(sel) for s, sel in suites.items()},
        "episodes": len(rows),
        "tasks": len({(r["suite"], r["task"]) for r in rows}),
        "horizon": {},
        "horizon_progress": {},
        "category": {},
    }
    if len(suites) == len(SUITES):
        rates = {s: out["suites"][s]["v"] / 100 for s in SUITES}
        # each suite weighs the same; the interval of a mean of three independent rates
        se = math.sqrt(sum(p * (1 - p) / len(suites[s]) for s, p in rates.items())) / 3
        out["mean"] = {"v": round(100 * sum(rates.values()) / 3, 1), "pm": round(196 * se, 1)}
        out["pooled"] = rate(rows)
        out["progress"]["mean"] = round(sum(out["progress"][s] for s in SUITES) / 3, 1)
    for s, sel in suites.items():
        for key, group in (
            ("horizon", lambda r: f"{r['subgoals']} subgoal" + ("s" if r["subgoals"] > 1 else "")),
            ("category", lambda r: TASKS[r["suite"]][r["task"]]["category"]),
        ):
            buckets = {}
            for r in sel:
                buckets.setdefault(group(r), []).append(r)
            out[key][s] = {b: rate(v)["v"] for b, v in buckets.items()}
            if key == "horizon":
                out["horizon_progress"][s] = {b: progress(v) for b, v in buckets.items()}
    return out


def package(args):
    rows = []
    for d in args.results:
        found = sorted(glob.glob(os.path.join(d, "*", "*", f"{args.name}_c1"))) if args.name else []
        for system_dir in found or [d]:
            rows += episodes_of(system_dir)
    if not rows:
        sys.exit(f"no episode logs of a LADDER task under {args.results}")
    out = args.out or os.path.join(ROOT, "leaderboard", "submissions", args.name or "my_system")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "episodes.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = os.path.join(out, "submission.yaml")
    if not os.path.exists(meta):
        commit = subprocess.run(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], check=False, capture_output=True, text=True
        ).stdout.strip()
        open(meta, "w").write(
            TEMPLATE.format(
                name=args.name or "My system",
                kind=args.kind,
                commit=commit or '""',
                date=datetime.date.today().isoformat(),
            )
        )
    print(f"wrote {out}: {len(rows)} episodes")
    report(out)


def report(submission_dir):
    s = score(read_episodes(os.path.join(submission_dir, "episodes.jsonl")))
    for suite, v in s["suites"].items():
        print(
            f"  {suite:16s} success {v['v']:5.1f} +- {v['pm']:.1f}   subgoal progress {s['progress'][suite]:5.1f}"
        )
    if "mean" in s:
        print(f"  {'mean of suites':16s} success {s['mean']['v']:5.1f} +- {s['mean']['pm']:.1f}")
    bad = problems(submission_dir)
    for p in bad:
        print("  TODO:", p)
    if not bad:
        print("  ready to submit: open a pull request with this directory (leaderboard/README.md)")
    return bad


def check(args):
    failed = False
    for d in args.submissions:
        print(d)
        failed |= bool(report(d))
    sys.exit(1 if failed else 0)


def build(args):
    """docs/static/data/submissions.json, in the shape the leaderboard page draws (suites se/ra/rl)."""
    short = {"simplerenv_long": "se", "robotarena_long": "ra", "robolab_long": "rl"}
    entries = []
    for d in sorted(glob.glob(os.path.join(ROOT, "leaderboard", "submissions", "*", ""))):
        d = d.rstrip(os.sep)
        if problems(d):
            sys.exit(f"{d} is not a valid submission: run scripts/leaderboard.py check {d}")
        meta = yaml.safe_load(open(os.path.join(d, "submission.yaml")))
        s = score(read_episodes(os.path.join(d, "episodes.jsonl")))
        by_suite = lambda table, total: {short[k]: {**v, "All tasks": total[k]} for k, v in table.items()}  # noqa: E731
        entries.append(
            {
                **{k: meta.get(k) for k in REQUIRED + ["organization", "paper_url", "code_url", "verified"]},
                "id": os.path.basename(d),
                "date": str(meta.get("date", "")),
                "scale": {"tasks": s["tasks"], "episodes": s["episodes"]},
                "suites": {short[k]: v for k, v in s["suites"].items()},
                "mean": s.get("mean"),
                "pooled": s.get("pooled"),
                "progress": {short.get(k, k): v for k, v in s["progress"].items()},
                "horizon": by_suite(s["horizon"], {k: v["v"] for k, v in s["suites"].items()}),
                "horizon_progress": by_suite(s["horizon_progress"], s["progress"]),
                "category": by_suite(s["category"], {k: v["v"] for k, v in s["suites"].items()}),
            }
        )
    out = args.out or os.path.join(ROOT, "docs", "static", "data", "submissions.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(entries, f, indent=1)
    print(f"wrote {out}: {len(entries)} submissions")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("package", help="turn eval output into a submission")
    p.add_argument(
        "--results", nargs="+", default=["out"], help="scripts/eval.sh's output directory (default: out)"
    )
    p.add_argument("--name", help="your system's --name (or method) in that output")
    p.add_argument("--kind", default="agent", choices=KINDS)
    p.add_argument("--out", help="default: leaderboard/submissions/NAME")
    p.set_defaults(run=package)
    p = sub.add_parser("check", help="validate submissions and print their scores")
    p.add_argument("submissions", nargs="+")
    p.set_defaults(run=check)
    p = sub.add_parser("build", help="write the leaderboard's data from every submission")
    p.add_argument("--out")
    p.set_defaults(run=build)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
