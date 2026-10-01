# Eval dashboard

A browser over eval results of the three suites: compare systems on the same tasks, replay an
episode with the orchestrator's subgoals and events synced to the video, and read the per-episode
logs. It reads the output of `scripts/eval.sh` and the baselines' evaluation episodes
([`INSAIT-Institute/LADDER-Evaluation-Episodes`](https://huggingface.co/datasets/INSAIT-Institute/LADDER-Evaluation-Episodes)).

## Running it

From the repository root:

```bash
bazel run //ladder/components/dashboard:serve -- --results <dir> [<dir> ...] --port 8080
```

Then open `http://localhost:8080`. `--results` takes one or more directories, and the flag may
repeat. Open your own runs and the baselines' episodes side by side to compare them.

A run is found by its directory name: give the launcher an output directory (`OUT_ROOT` on
SimplerEnv-Long, `CAMPAIGN_ROOT` on the other two) of the form
`<root>/<suite>/<low-level policy>/<method>_c<n>`, e.g. `out/robolab/pi05/volo_c1`, as
`scripts/eval.sh` does. `<suite>` starts with `simpler`, `robot_arena` or `robolab`.

To browse the evaluation episodes of the baselines, download them first:

```bash
bazel run //ladder/components/dashboard:fetch -- episodes --repo INSAIT-Institute/LADDER-Evaluation-Episodes --out ~/ladder_episodes
bazel run //ladder/components/dashboard:serve -- --results ~/ladder_episodes/episodes
```

## What it reads

| what | where |
|---|---|
| the tallies | `<cell>/<checkpoint>/<task>/results.yaml` |
| the episodes | `episode_log.yaml`, one row per episode |
| the recordings | `episodes/episode_*/` (`main.mp4`, RoboLab-Long adds `wrist.mp4`) |
| logs | the `*.log` files beside them |

Nothing is configured. When one task of a system was run more than once, the fuller tally wins.

## Videos

Browsers cannot decode the mp4v videos the simulators record, so a clip is transcoded to H.264 once
(with `ffmpeg`, if it is installed) and cached under `~/.cache/ladder-dashboard`
(`LADDER_DASHBOARD_CACHE` moves it). The published evaluation episodes are already H.264.
