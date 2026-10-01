# Evaluation

```bash
scripts/eval.sh --suite SUITE --method METHOD [--task TASK ... | --task all] [--episodes N] [--output DIR]
```

- [Suites](#suites)
- [Methods](#methods)
- [Options](#options)
- [Models and API keys](#models-and-api-keys)
- [Offline VLM benchmark](#offline-vlm-benchmark)
- [Launchers and configs](#launchers-and-configs)
- [Licenses of external components](#licenses-of-external-components)

## Suites

| `--suite` | suite | simulator | robot | tasks | episodes per task |
|---|---|---|---|---|---|
| `simplerenv_long` | SimplerEnv-Long | SAPIEN | WidowX | 54 | 24 |
| `robotarena_long` | RobotArena-&infin;-Long | Genesis | WidowX | 47 | 8 |
| `robolab_long` | RoboLab-Long | Isaac Sim | Franka | 44 | 8 |

The tasks of each suite are in [TASKS.md](TASKS.md) and `scripts/tasks/<suite>.txt`.

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

## Options

| option | default |
|---|---|
| `--task` | one two-subgoal task; `all` runs every task of the suite |
| `--episodes` | 24 on `simplerenv_long`, 8 on `robolab_long` (`robotarena_long` always runs 8) |
| `--output` | `out` |

Results go to `<output>/<suite>/<low-level policy>/<method>_c1/`, one directory per task with its
`results.yaml` and the episode videos. The dashboard reads this layout:

```bash
bazel run //ladder/components/dashboard:serve -- --results $PWD/out
```

To download the evaluation episodes of every method ([`INSAIT-Institute/LADDER-Evaluation-Episodes`](https://huggingface.co/datasets/INSAIT-Institute/LADDER-Evaluation-Episodes)),
one directory per suite and method:

```bash
# one method on one suite; leave out --suite or --method for all of them, add --no-videos for the logs only
bazel run //ladder/components/dashboard:fetch -- episodes --repo INSAIT-Institute/LADDER-Evaluation-Episodes --out ~/ladder_episodes \
    --suite robolab_long --method psp_agent
bazel run //ladder/components/dashboard:serve -- --results $PWD/out ~/ladder_episodes/dashboard
```

## Models and API keys

`low_level_policy`, `privileged_verifier` and `scripted` need no API. The other methods call a
vision-language model through an OpenAI-compatible endpoint:

```bash
export VLM_BASE_URL=<endpoint>   # e.g. https://generativelanguage.googleapis.com/v1beta/openai/
export VLM_API_KEY=<key>
```

The models are Gemini 3.7 Flash (`ladder/components/inference/api/client/hlc/configs/vlm.yaml`) and,
for `gpt_policy`, GPT-6 Astra (`ladder/components/inference/models/gpt_policy/configs/vlm.yaml`).
Change `model:` there if your endpoint names them differently.

`cap_x` uses SAM 3, whose weights are gated: accept the license at
[`facebook/sam3`](https://huggingface.co/facebook/sam3), then `export HF_TOKEN=<token>`.

The first `robolab_long` run downloads Isaac Sim's extensions (about 5 GB), which can take 15
minutes or more. The first `tiptop` run compiles its CUDA extensions.

## Offline VLM benchmark

[`INSAIT-Institute/LADDER-Offline-VLM-Benchmark`](https://huggingface.co/datasets/INSAIT-Institute/LADDER-Offline-VLM-Benchmark) scores a
vision-language model on the judgements an orchestrator makes, without running a simulator.

| task | command |
|---|---|
| plan decomposition | `bazel run //ladder/components/inference/models/vlams:score_decomposition -- --dataset INSAIT-Institute/LADDER-Offline-VLM-Benchmark --out OUT.json --model MODEL` |
| subgoal verification, state tracking, subgoal switching | `bazel run //ladder/components/inference/models/vlams:score_subtask_states -- --dataset INSAIT-Institute/LADDER-Offline-VLM-Benchmark --task verification\|state_tracking\|next_subtask --out OUT.json --model MODEL` |

`--backend local` (the default) runs the model on your GPU from Hugging Face; `--backend gateway`
calls it on the endpoint in `VLM_BASE_URL` / `VLM_API_KEY`. `--limit N` scores the first N items.
The metrics are in [`api/client/metrics/README.md`](ladder/components/inference/api/client/metrics/README.md).

## Launchers and configs

`scripts/eval.sh` picks the method's config and runs the suite's launcher (paths under
`ladder/components/inference/`):

| suite | launcher | config | tasks |
|---|---|---|---|
| `simplerenv_long` | `models/vlams/scripts/vlam_simpler_bridge.sh` | `SIMPLER_CONFIG` | `ENV_NAMES` |
| `robotarena_long` | `models/vlams/scripts/vlam_robot_arena.sh` | `RA_CLIENT_CONFIG` | `SCENES` |
| `robolab_long` | `robots/robolab/scripts/vlam_robolab.sh` | `LADDER_CONFIG` | `TASKS` |

The configs of the methods that drive a low-level policy are in
`models/vlams/configs/<suite>/composition/` (pi0.5: `models/external/openpi/configs/robolab/composition/`),
and those of the other methods in `models/{capx,tiptop,harness,gpt_policy}/configs/<suite>/composition/`.
`scripts/eval.sh` shows which config and variables each method uses.

## Licenses of external components

The build downloads these components, and this repository does not redistribute them:

- ManiSkill3 and SAPIEN, Genesis, RoboLab, openpi: Apache-2.0.
- RobotArena: MIT.
- Isaac Sim and Isaac Lab: NVIDIA's license.
- VoLo, CaP-X, TiPToP and its dependencies (cuTAMP, cuRobo, M2T2, SAM 2): their own licenses.
- GPT-Policy: no license upstream; fetched by pinned commit.

The RobotArena-&infin;-Long and RoboLab-Long asset bundles hold the files of the public
[RobotArena](https://github.com/offjangir/RobotArena) and [RoboLab](https://github.com/NVlabs/RoboLab)
releases that the tasks load, unchanged.
