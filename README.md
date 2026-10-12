# LADDER: A Long-Horizon Robot Manipulation Benchmark

[🌐 Website](https://musical-garbanzo-5w422vy.pages.github.io) · [🤗 Models and datasets](https://huggingface.co/collections/INSAIT-Institute/ladder-6abbbddc46691080f5d954c8)

**LADDER** is a benchmark for long-horizon robot manipulation: 145 tasks of 1 to 10 subgoals in
three simulated suites, each subgoal checked automatically.

<div align="center">
  <img src="docs/static/images/teaser.png" alt="LADDER overview" width="800"/>
</div>

## Installation

Install [Bazelisk](https://github.com/bazelbuild/bazelisk#installation) as `bazel` (below into
`~/.local/bin`, which has to be on your `PATH`), then clone the repository:

```bash
curl -L --create-dirs -o ~/.local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 && chmod +x ~/.local/bin/bazel
git clone https://github.com/insait-institute/ladder.git && cd ladder
```

Everything else (Python environments, simulators, assets, the Bridge policy) is installed by Bazel
on the first run.

## Running an evaluation

```bash
scripts/eval.sh --suite simplerenv_long --method privileged_verifier
```

This runs a method ([all methods](EVALUATION.md#methods)) on a suite ([all suites](EVALUATION.md#suites)),
here on its default two-subgoal task. Results and videos are written to `out/`, and the
[dashboard](#dashboard) shows them.

```bash
# specific tasks (all tasks: TASKS.md)
scripts/eval.sh --suite robolab_long --method low_level_policy --task comp_bin_2 comp_relay_3

# every task of a suite
scripts/eval.sh --suite robotarena_long --method privileged_verifier --task all

# a method that uses a vision-language model: any OpenAI-compatible endpoint
export VLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export VLM_API_KEY=<your key>
scripts/eval.sh --suite simplerenv_long --method psp_agent
```

[EVALUATION.md](EVALUATION.md) has the remaining options.

## Tasks

[TASKS.md](TASKS.md) lists every task with its instruction and a video of it being solved.

<div align="center">
  <img src="docs/static/readme/simpler.gif" width="32%" alt="SimplerEnv-Long">
  <img src="docs/static/readme/robot_arena.gif" width="32%" alt="RobotArena-Long">
  <img src="docs/static/readme/robolab.gif" width="32%" alt="RoboLab-Long">
</div>
<p align="center"><sub>Flawless episodes from the released evaluation: GPT-Policy on tasks of 10, 8 and 7 subgoals. All play at 4&times; speed.</sub></p>

## Dashboard

The dashboard replays episodes next to their subgoals and events, and compares methods task by task.

<div align="center">
  <img src="docs/static/readme/dashboard.gif" alt="The dashboard" width="800"/>
</div>

```bash
bazel run //ladder/components/dashboard:serve -- --results $PWD/out
# open http://localhost:8080
```

The evaluation episodes of every method ([`INSAIT-Institute/LADDER-Evaluation-Episodes`](https://huggingface.co/datasets/INSAIT-Institute/LADDER-Evaluation-Episodes))
open in it too, here those of one method on one suite:

```bash
bazel run //ladder/components/dashboard:fetch -- episodes --repo INSAIT-Institute/LADDER-Evaluation-Episodes --out ~/ladder_episodes \
    --suite robolab_long --method psp_agent
bazel run //ladder/components/dashboard:serve -- --results $PWD/out ~/ladder_episodes/dashboard
```

## Offline VLM benchmark

The judgements an orchestrator asks a vision-language model to make (plan decomposition, subgoal
verification, state tracking, subgoal switching), on 145 scenes and 562 frames, scored without a
simulator ([`INSAIT-Institute/LADDER-Offline-VLM-Benchmark`](https://huggingface.co/datasets/INSAIT-Institute/LADDER-Offline-VLM-Benchmark)):

```bash
bazel run //ladder/components/inference/models/vlams:score_decomposition -- \
    --dataset INSAIT-Institute/LADDER-Offline-VLM-Benchmark --out decomposition.json --model <model>
```

See [EVALUATION.md](EVALUATION.md#offline-vlm-benchmark) for the other tasks.

## Results

Task success rate (%) of the baselines, from the paper.

| method | SimplerEnv-Long | RobotArena-&infin;-Long | RoboLab-Long | all tasks |
|---|---:|---:|---:|---:|
| Low-level policy | 8.7 | 10.9 | 19.6 | 11.0 |
| Periodic trigger | 17.4 | 15.2 | 25.6 | 18.4 |
| Proprioceptive trigger | 21.5 | 16.2 | 28.1 | 21.7 |
| Proprioceptive verifier | 25.2 | **24.2** | 25.6 | 25.0 |
| VoLo | **25.5** | 21.3 | **29.0** | **25.3** |
| Privileged verifier | 29.5 | 25.5 | 34.4 | 29.6 |
| CaP-X | 28.7 | 16.2 | 20.7 | 25.0 |
| TiPToP | 6.7 | 11.2 | 17.3 | 9.4 |
| PSP-Agent | 37.1 | 39.9 | 51.4 | 40.1 |
| GPT-Policy | **60.3** | **66.8** | **62.8** | **62.0** |

<div align="center">
  <img src="docs/static/images/length_ladder.png" alt="Success rate by number of subgoals" width="800"/>
</div>

## Documentation

- [Evaluation](EVALUATION.md): suites, methods, options and the launchers behind `scripts/eval.sh`
- [Tasks](TASKS.md)
- [Dashboard](ladder/components/dashboard/README.md)
- [Metrics](ladder/components/inference/api/client/metrics/README.md)

## Tested setup

Debian 13 with NVIDIA RTX A6000, H200 and RTX PRO 6000 GPUs, and an RTX 3090.

## License

The code and the Bridge policy are released under the [MIT License](LICENSE), and the evaluation
episodes under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The benchmark also uses
third-party assets and code under their own licenses, some of which allow non-commercial use only.
They are listed in [EVALUATION.md](EVALUATION.md#licenses-of-external-components).

## Citation

```bibtex
@article{ladder2026,
  title   = {{LADDER}: A Long-Horizon Robot Manipulation Benchmark},
  author  = {Petkov, Petko and Albanese, Giuliano and Nikolov, Nikolay and Van Gool, Luc and Tremblay, Jonathan and Paudel, Danda Pani},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026},
  url     = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
