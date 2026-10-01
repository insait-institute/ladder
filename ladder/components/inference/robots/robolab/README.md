# RoboLab benchmark integration

Bazel-packages NVIDIA's [RoboLab](https://github.com/NVlabs/RoboLab) manipulation benchmark
(DROID/Franka), from [a fork](https://github.com/psi-robotics/RoboLab) that upgrades it to Isaac Sim
6.0.0.1 and Isaac Lab 3.0.0b2.

- **`env_server.py`** — serves RoboLab envs over
  ladder's RobotServer ZMQ protocol to the standard `run_model` + `SimplerEpisodeRunner` stack,
  so runner-side features (HLC configs, RecorderWrapper, dashboard-format outputs)
  apply to RoboLab exactly as to RobotArena/SimplerEnv. One Isaac app serves the whole
  (task x run) campaign. Model/behavior config:
  `models/vlams/configs/robolab/vlam_robolab.yaml` — it lives with the other per-benchmark `run_model`
  configs (`vlam_simpler.yaml`, `vlam_robot_arena_*.yaml`) because `run_model`'s bazel target
  packages that directory and the config imports `vlam_simpler.yaml`.

The launcher takes any policy, not only our checkpoints. Two variables cover one that holds its
own model (an external model under `models/external/`, or the scripted policy): `RUN_MODEL` names
the `run_model` target to launch, which is the model's own when it brings its own pip environment,
and `INFERENCE_SERVER=0` says there is nothing to serve. Everything else is identical, so a policy
either way is scored by the same checker, recorder and metric layer, and pairs with the HLC systems
below.

RoboLab runs in a py3.12 + isaac env (numpy 2.x) and the model in the vlams torch env, so the
path from env to policy crosses one ZMQ socket into `run_model` (which talks to the shared
`batch_inference_server`), mirroring RobotArena's process layout.

`droid_conversions.py` owns the DROID conventions (gripper inversion both directions,
maintained EE targets with `controls_on_targets`, `resize_with_pad(224)` matching RoboLab's own
client, and a `pytorch_kinematics` Franka FK/IK in `franka_kinematics.py` over ladder's panda
URDF (torch/numpy-2 native). RoboLab publishes `base_link`/`eef_frame` poses offset
from ladder's canonical `panda_link8`, so staying in joint space + the canonical flange frame
keeps the integration frame-convention-free.

## Packaging (see `http-assets.MODULE.bazel`)

- **Code** — `@robolab_repo`: the RoboLab source archive (sha256-pinned, code only). A `constants.py`
  patch makes it resolve assets from `ROBOLAB_ASSET_DIR`.
- **Assets** — `@robolab_assets_archive`: the ~6 GB asset tree (HuggingFace tarball), staged into
  `assets/` by `copy_to_directory`. First build on a fresh machine fetches it once.
- **openpi client** — `@openpi_client_repo` vendors `openpi_client` as source (its `numpy<2` pin
  conflicts with isaacsim's numpy 2.x; the source runs fine on 2.x).

## Running an eval

`scripts/vlam_robolab.sh` (env-var driven, ports auto-allocated). GPU node with RT cores;
Isaac Sim 6 needs driver 595.

```bash
# one of our checkpoints, behind the shared inference server (the default)
SESSION_NAME=sess_... CKPT_NAME=model-checkpoint-ema-050000.pt \
TASKS="BananaOnPlateTask" NUM_ENVS=8 NUM_RUNS=1 \
./ladder/components/inference/robots/robolab/scripts/vlam_robolab.sh
# an external model -- one model per lane, no inference server
RUN_MODEL=models/external/openpi:run_model INFERENCE_SERVER=0 TASKS="comp_bin_2" \
LADDER_CONFIG=ladder/components/inference/models/external/openpi/configs/robolab/composition/openpi_robolab_composition.yaml \
./ladder/components/inference/robots/robolab/scripts/vlam_robolab.sh
```

Manual invocation of the individual pieces (the launcher does all of this for you):

```bash
# env server — serves RoboLab envs over the RobotServer ZMQ protocol
bazel run //ladder/components/inference/robots/robolab:env_server -- \
    --port 5556 --task BananaOnPlateTask --num-envs 8 --num-runs 1 --headless
# dashboard export — map a campaign to episode_results.jsonl
bazel run //ladder/components/inference/robots/robolab:export_robolab_results -- \
    --episode-log <campaign>/all/episode_log.yaml --output-dir <campaign>
```

Runs write `results.yaml`, `episode_log.yaml` and per-env mp4s AND —
via `export_robolab_results.py`, run automatically at campaign end — RoboLab's
`episode_results.jsonl`, so their dashboard reads our campaigns too. Env-side diagnostics (hdf5
trajectories, SPARC/path-length metrics, event streams) are the one thing only RoboLab's own runner
ever wrote.

An external model runs in its own pip environment, which bazel provisions, and holds one copy of
the weights per lane rather than one per card. That costs roughly a lane's worth of VRAM, so run
fewer lanes per card. Its own package documents its caches
(`models/external/openpi/README.md` for pi0.5).

## Hierarchical eval systems

The ladder path drives the shared HLC stack (`api/client/hlc`), the same one RobotArena and
SimplerEnv use, so a system is defined ONCE in `api/client/hlc/configs/` and each benchmark ships
one eval config per system, **named after it**. `LADDER_CONFIG` selects the system; the RoboLab
configs live in `models/vlams/configs/robolab/composition/` and each imports its shared system plus
this benchmark's plan cache and RoboLab's own timing:

| `LADDER_CONFIG` (`…/configs/robolab/composition/`) | Shared system | Plan | Switches on |
|---|---|---|---|
| `vlam_robolab_composition.yaml`                       | — (flat, no HLC)         | — (raw instruction) | — |
| `…_oracle.yaml`                                       | `oracle`                 | from the env        | per-goal `achieved` |
| `…_plan_gripper.yaml`                                 | `plan_gripper`           | cached              | the gripper cycle |
| `…_plan_periodic.yaml`                                | `plan_periodic`          | cached              | a step clock (no feedback) |
| `…_plan_verifier.yaml`                                | `plan_verifier`          | cached              | a VLM check |
| `…_plan_verifier_interval.yaml`                       | `plan_verifier_interval` | cached              | a VLM check on a clock |
| `…_plan_verifier_local.yaml`                          | `plan_verifier_local`    | cached              | a VLM check, in-process Qwen3-VL |
| `…_plan_verifier_local_interval.yaml`                 | `plan_verifier_local`    | cached              | the local check, on a clock |
| `…_plan_verifier_noretry.yaml`                        | `plan_verifier_noretry`  | cached              | a VLM check, no retry |
| `…_plan_verifier_retry.yaml`                          | `plan_verifier_retry`    | cached              | a VLM check, retry in place |
| `…_stepwise_gripper.yaml`                             | `stepwise_gripper`       | none (one at a time) | the gripper cycle |
| `…_stepwise_interval.yaml`                            | `stepwise_interval`      | none (one at a time) | a step clock |
| `…_volo.yaml`                                         | `volo`                   | cached              | VoLo's own tool loop |

`…_plan_verifier_live.yaml` is the one file that is not a system: it is `_plan_verifier` with
`cache_path` omitted, so the VLM decomposes the reset frame — the pass that mints a plan for a newly
added task. `composition/language/` holds the same paraphrase probes the other two benchmarks ship.
This is the same set, file for file, as SimplerEnv's and RobotArena's composition configs, so a
system compares across all three.

```bash
SESSION_NAME=sess_... CKPT_NAME=model-checkpoint-ema-015000.pt \
LADDER_CONFIG=ladder/components/inference/models/vlams/configs/robolab/composition/vlam_robolab_composition_plan_verifier.yaml \
TASKS="comp_bin_2" ./ladder/components/inference/robots/robolab/scripts/vlam_robolab.sh
```

**Plans are a config concern, not the launcher's.** Only the `_plan_*` systems read
`composition/decomp_cache.json` (zero API calls at eval time), via `hlc_config.source.cache_path`.
Flat and `_oracle` need no plans at all and the `_stepwise_*` systems hold none by design. The cache
sits beside the composition package exactly as it does on the other two benchmarks; RoboLab's is keyed
by task class name (`CompBin4Task`).

The oracle needs no cache: `oracle.py` synthesizes the per-goal `metadata['oracle']` signal from the
task's OWN subtask spec — one `(command, check)` pair per score-carrying goal, rendered from the
goal's terminating predicate (verb + object/target nouns), the RoboLab-side analogue of SimplerEnv's
`oracle_subtasks`. Commands and `achieved` are built in one pass so `oracle.achieved[i]` lines up
with commanded subtask `i` by construction.

RoboLab's timescale is the one thing that does not carry over from the other two benchmarks: a Franka
pick-and-place takes ~250 steps against ~50 on SimplerEnv/RA, so every advancer window
(`stuck_patience_steps`, `attempt_stall_steps`, `interval`, `verify_interval`) is set from RoboLab's
own measured median subtask rather than inherited. A window shorter than one placement abandons every
goal mid-reach and reads as a difficulty wall that is not there.

## Throughput

`NUM_ENVS` parallel Isaac envs are stacked into one batched model forward per decode, sharing the
fixed model cost across N episodes — raise it for throughput (`MAX_BATCH_SIZE >= NUM_ENVS`).
Per-step wall is dominated by the upstream Isaac render, not the model.

## Parity contract vs the native runner

Validated against RoboLab's native runner:
clean-start success 86.4% ladder vs 86.7% native; frame-level camera/action-input equivalence
verified; ladder path measured equal-to-30% faster.

Identical by construction (any change here must keep both paths in lockstep):
- Task iteration = `run_evaluation`'s loop: `create_env` per task (same args, seed 0 default),
  `reset_eval_state` between runs, double `env.reset()`, timeline pump before each step.
- Env-side semantics untouched: success predicates, frozen envs, per-task episode caps,
  the ep_len<=2 physics-artifact re-reset.
- Model I/O: same frames (`resize_with_pad(224)`, both cameras), gripper flips + binarize,
  maintained-EE-target + clamped IK (`droid_conversions.py`), same `ActionChunkingController`
  cadence.

Known, deliberate differences:
- Native-only features: `--num-episodes-adaptive` sampling; resume-skip of completed cells
  (the ladder runner appends); hdf5 trajectories + event streams. Trajectory metrics
  (`ee_sparc`, `joint_sparc_mean`, `ee_isj`, `joint_isj`, `ee_path_length`, `joint_rmse_mean`,
  `ee_speed_max/mean`) are native-parity: the env server samples the same articulation state
  the native hdf5 recorder writes and calls robolab's own `compute_episode_metrics` at each
  env's terminal step, so `export_robolab_results.py` fills `metrics`; `events`/`timing` stay
  empty.
- The ladder runner queries the model for frozen envs too (results-neutral; the batched
  server amortizes it — measured net-faster than native).
- Ladder videos are the 224x224 letterboxed model-input frames (native records 720p half-scale);
  each env's recording ends at its own terminal step.
