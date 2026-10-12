# Evaluation

```bash
scripts/eval.sh --suite SUITE --method METHOD [--task TASK ... | --task all] [--episodes N] [--output DIR]
```

- [Suites](#suites)
- [Methods](#methods)
- [Your own system](#your-own-system)
- [Options](#options)
- [Models and API keys](#models-and-api-keys)
- [Offline VLM benchmark](#offline-vlm-benchmark)
- [Launchers and configs](#launchers-and-configs)
- [Licenses of external components](#licenses-of-external-components)

## Suites

| `--suite` | suite | simulator | robot | tasks | episodes per task |
|---|---|---|---|---|---|
| `robolab_long` | RoboLab-Long | Isaac Sim | Franka | 44 | 8 |
| `simplerenv_long` | SimplerEnv-Long | SAPIEN | WidowX | 54 | 24 |
| `robotarena_long` | RobotArena-&infin;-Long | Genesis | WidowX | 47 | 8 |

The tasks of each suite are in [TASKS.md](TASKS.md) and `scripts/tasks/<suite>.txt`.
`--task lite` runs LADDER-Lite, a fixed 10-task subset (`scripts/tasks/<suite>_lite.txt`: 4 RoboLab-Long,
2 SimplerEnv-Long and 4 RobotArena-&infin;-Long tasks, 112 episodes), about 1/18 of the episodes of the full benchmark.
On it, the two best systems of the paper keep their places, and every system's score is within 5% of its
full-benchmark score.

## Methods

| `--method` | method | low-level policy |
|---|---|---|
| `low_level_policy` | Low-level policy | Bridge policy / pi0.5 |
| `periodic_trigger` | Periodic trigger | Bridge policy / pi0.5 |
| `proprioceptive_trigger` | Proprioceptive trigger | Bridge policy / pi0.5 |
| `proprioceptive_verifier` | Proprioceptive verifier | Bridge policy / pi0.5 |
| `volo` | VoLo | Bridge policy / pi0.5 |
| `privileged_verifier` | Privileged verifier | Bridge policy / pi0.5 |
| `cap_x` | CaP-X | none |
| `tiptop` | TiPToP | none |
| `psp_agent` | PSP-Agent | none |
| `gpt_policy` | GPT-Policy | none |
| `scripted` | Scripted policy | none |

The Bridge policy ([`INSAIT-Institute/LADDER-Bridge-Policy`](https://huggingface.co/INSAIT-Institute/LADDER-Bridge-Policy)) runs on the two WidowX
suites and is downloaded on the first run. pi0.5 (DROID, from openpi) runs on RoboLab-Long.
The scripted policy solves the tasks from the simulator's ground-truth object state; its videos are
the ones in [TASKS.md](TASKS.md).

## Your own system

Instead of `--method`, give your own system ([`examples/README.md`](examples/README.md)):

| option | your system | runs on |
|---|---|---|
| `--policy-server HOST:PORT` | a policy served from any Python environment | the plugin entry point |
| `--orchestrator FILE:CLASS` | an orchestrator instructing the suite's low-level policy | the low-level policy's entry point |
| `--agent FILE:CLASS` | an agent driving the arm, with depth | the plugin entry point |

`--name` sets the name its results are filed under (default: the class name). The configs are in
`ladder/components/inference/models/plugin/configs/<suite>/`.

## Options

| option | default |
|---|---|
| `--task` | one two-subgoal task; `all` runs every task of the suite |
| `--episodes` | `robolab_long`: episodes per task (8). `simplerenv_long`: how many of the 24 episodes run at once (24). `robotarena_long` always runs 8 |
| `--output` | `out` |

Results go to `<output>/<suite>/<low-level policy>/<method>_c1/`, one directory per task with its
`results.yaml` and the episode videos. The dashboard reads this layout:

```bash
scripts/dashboard.sh
```

To download the evaluation episodes of every method ([`INSAIT-Institute/LADDER-Evaluation-Episodes`](https://huggingface.co/datasets/INSAIT-Institute/LADDER-Evaluation-Episodes)),
one directory per suite and method:

```bash
# one method on one suite; leave out --suite or --method for all of them, add --no-videos for the logs only
scripts/dashboard.sh fetch --suite robolab_long --method psp_agent
scripts/dashboard.sh out ~/ladder_episodes/dashboard ~/ladder_episodes/dashboard_real
```

## Models and API keys

`low_level_policy`, `privileged_verifier` and `scripted` need no API. The other methods call a
vision-language model through any OpenAI-compatible Chat Completions endpoint:

```bash
export VLM_BASE_URL=<endpoint>   # e.g. https://generativelanguage.googleapis.com/v1beta/openai/
export VLM_API_KEY=<key>
export VLM_MODEL=<model>         # optional: gemini-3.7-flash by default, gpt-6-astra for gpt_policy
```

`cap_x` uses SAM 3, whose weights are gated: accept the license at
[`facebook/sam3`](https://huggingface.co/facebook/sam3), then `export HF_TOKEN=<token>`.

The first `robolab_long` run downloads Isaac Sim's extensions (about 5 GB), which can take 15
minutes or more. The first `tiptop` run compiles its CUDA extensions.

## Offline VLM benchmark

[`INSAIT-Institute/LADDER-Offline-VLM-Benchmark`](https://huggingface.co/datasets/INSAIT-Institute/LADDER-Offline-VLM-Benchmark) scores a
vision-language model on the judgements an orchestrator makes, without running a simulator.

| task | command |
|---|---|
| plan decomposition | `scripts/vlm_benchmark.sh decomposition --model MODEL` |
| subgoal verification, state tracking, subgoal switching | `scripts/vlm_benchmark.sh verification\|state_tracking\|next_subtask --model MODEL` |

`--backend local` (the default) runs the model on your GPU from Hugging Face; `--backend gateway`
calls it on the endpoint in `VLM_BASE_URL` / `VLM_API_KEY`. `--limit N` scores the first N items.
The metrics are in [`api/client/metrics/README.md`](ladder/components/inference/api/client/metrics/README.md).

## Launchers and configs

`scripts/eval.sh` picks the method's config and runs the suite's launcher (paths under
`ladder/components/inference/`):

| suite | launcher | config | tasks |
|---|---|---|---|
| `robolab_long` | `robots/robolab/scripts/vlam_robolab.sh` | `LADDER_CONFIG` | `TASKS` |
| `simplerenv_long` | `models/vlams/scripts/vlam_simpler_bridge.sh` | `SIMPLER_CONFIG` | `ENV_NAMES` |
| `robotarena_long` | `models/vlams/scripts/vlam_robot_arena.sh` | `RA_CLIENT_CONFIG` | `SCENES` |

The configs of the methods that drive a low-level policy are in
`models/vlams/configs/<suite>/composition/` (pi0.5: `models/external/openpi/configs/robolab/composition/`),
and those of the other methods in `models/{capx,tiptop,harness,gpt_policy}/configs/<suite>/composition/`.
`scripts/eval.sh` shows which config and variables each method uses.

## Licenses of external components

The build downloads third-party code and assets, each under its own license, which is included
in its archive. Some of them, such as RoboLab's assets, cuRobo, M2T2 and cuTAMP, allow
non-commercial use only. `robolab_long` installs NVIDIA Isaac Sim and accepts its
[license](https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html)
for you, so read it before the first run.
