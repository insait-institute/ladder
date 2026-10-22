# SIMPLER Simulator (ManiSkill3)

SimplerEnv widowx bridge eval on ManiSkill3 / SAPIEN3. Envs are GPU-vectorized: one
`simpler_server` holds `n_envs` parallel envs and `run_model` batches their observations
into a single model forward per step (same client-server split + ZMQ protocol as
RobotArena).

## Usage

See `ladder/components/inference/models/vlams/scripts/vlam_simpler_bridge.sh`. Per
(checkpoint, scene) it starts one `simpler_server` (`configs/<scene>.yaml`, `n_envs`
parallel envs) and one `run_model` client. `N_ENVS` must divide the 24 canonical episodes.

## Long-horizon composition benchmark

`composition/specs.py` defines a `{task type x sequence length}` matrix of multi-goal tasks
(`composition/envs.py` registers them with ManiSkill; `composition/placement.py` packs their object
layouts). They run through the same launcher as the four canonical scenes. A composition
`SIM_CONFIG` makes `ENV_NAMES` scene ids instead of config names:

```bash
SESSION_NAME=sess_… CKPT_NAMES=model-checkpoint-ema-030000.pt \
SIM_CONFIG=ladder/components/inference/robots/simpler/configs/widowx_composition.yaml \
SIMPLER_CONFIG=ladder/components/inference/models/vlams/configs/simpler/composition/vlam_simpler_composition.yaml \
ENV_NAMES="all2_plate split2 move_plate_to_towel_2" \
OUT_ROOT=$HOME/eval/comp VIDEO=True LANES_PER_GPU=1 \
./ladder/components/inference/models/vlams/scripts/vlam_simpler_bridge.sh
```

**Every object in a scene is named by the instruction**, except in the `except_eggplant_*` family,
whose wording ("put everything except the eggplant on the plate") is what makes the unnamed eggplant
part of the task. `envs.evaluate` fails an episode whose distractor drifted more than 5 cm, so a
distractor anywhere else is a hidden penalty on an object nothing mentions. The transfer family's
`start_on` receptacle is not a distractor: the instruction names it as where the movers come from.

The length axis starts at one placement, with `all1_plate` (carrot on plate) and `all1_towel` (spoon
on towel): the canonical bridge pairs at native receptacle size, run here rather than as the upstream
envs, which allow 60 steps to this suite's 100 and publish their own success test.

`SIMPLER_CONFIG` selects the system: `vlam_simpler_composition.yaml` is flat (the model gets only the
composed instruction), and one `_<system>` variant exists per hierarchical system in
`api/client/hlc/configs/`, named after it: `_oracle`, `_plan_gripper`, `_plan_periodic`,
`_plan_verifier`, `_plan_verifier_interval`, `_plan_verifier_local`, `_stepwise_gripper`,
`_stepwise_interval`. `_plan_verifier_live` is not a system of its own: it is `_plan_verifier` with
`cache_path` omitted (see **Plans** below). Each writes to its own `simpler_composition*/` directory.

The three plan-driven arms are meant to be read together: `_plan_periodic` switches on a clock with no
feedback, `_plan_gripper` on an event the robot produces, `_plan_verifier` on a judgement about the
scene. The clock is set to the oracle's median time on a subtask, so it is not a strawman. A gap to
it measures what the feedback is worth. `_plan_verifier_local` is `_plan_verifier` with the VLM
in-process (Qwen3-VL) instead of a hosted API.

**Language probes.** Two independent knobs on the runner, for testing how much instruction following
matters. `runner_config.instruction_override` rewords the TASK instruction, keyed by `<task>-v0` (a
task's registered instruction is only a default); a key matching no registered env is silently ignored,
so check names against `composition_manifest`. It lives in `configs/simpler/composition/language/`:
`vlam_simpler_composition_paraphrase.yaml`.

**Choosing envs.** `ENV_NAMES` is the only control. Pass a subset, or enumerate the whole registry.
`composition_manifest` prints `<env-v0>|<max_episode_steps>` per task, straight from the registry:

```bash
# every env, in registry order (54 of them)
ENV_NAMES=$(bazel run //ladder/components/inference/robots/simpler:composition_manifest 2>/dev/null \
            | cut -d'|' -f1 | sed 's/-v0$//' | tr '\n' ' ')

# longest first, which shortens the tail when running lanes
ENV_NAMES=$(bazel run //ladder/components/inference/robots/simpler:composition_manifest 2>/dev/null \
            | sort -t'|' -k2 -rn | cut -d'|' -f1 | sed 's/-v0$//' | tr '\n' ' ')
```

**Ordering.** Step budgets are 100 per goal, so the registry spans 10x (100 to 1000 steps). A lane
picks up the next env as soon as any lane frees up, so order barely changes the total; longest-first
still shortens the tail when the env count is not a multiple of the lane count.

**Lanes.** `LANES_PER_GPU>1` runs that many envs concurrently against one shared inference server.

**Plans are a config concern, not the launcher's.** Only the `_plan_*` systems read
`composition/decomp_cache.json`, via `hlc_config.source.cache_path` in their YAML. Flat and `_oracle`
need no plans at all (the oracle's come from the env) and the `_stepwise_*` systems hold none by
design: each switch generates the next subtask from the current frame. `_plan_verifier_live` just
omits `cache_path`, so the VLM decomposes the reset frame instead. That is the pass that mints a plan
for a newly added env.

**Repeats.** The inference server seeds its own process, because the model draws its flow start
state from the global RNG (`roma.random_unitquat` takes no generator). Seeded, two identical runs are
byte identical, `num_steps` and `success_per_step` included. In-process runs are reproducible too,
since `run_model` seeds itself at startup. The `_stepwise_*` systems stay the exception:
their subtask source samples (`VLMSource` defaults to `temperature: 0.3` and `subtask_generator.yaml`
does not override it). Once repeats replay exactly, added confidence comes from more episodes, not
from more repeats of the same seed; `VLA_SEED` varies it deliberately.

### Exporting the decomposition-benchmark scenes

The decomposition benchmark asks one question of a planner, offline: shown a scene's frame and its
instruction, does it produce the right subtasks? No simulator runs during scoring, so the scenes have
to be exported first. `export_scenes` renders each scene's reset frame, env 0 at episode 0 (the one
layout a controller ever decomposes from) and writes a `scenes.json` beside the frames: one record
per scene with its instruction, cached plan, goal count and ordering. That file is the entire interface
to the scorer, which globs `<frames-dir>/*/scenes.json`
(`api/client/metrics/tools/score_decomposition.py`); frames are named relative to it so the directory
can be copied to whichever machine scores it.

```bash
bazel run //ladder/components/inference/robots/simpler:export_scenes -- --out-dir $BENCH/frames/simplerenv
```

`robots/robot_arena` and `robots/robolab` carry the same target for their own suites, and the
concatenation of the three `scenes.json` files is the benchmark. See
`api/client/metrics/README.md`.

## Dependencies / assets

`mani_skill==3.0.1` + `sapien==3.0.3` + CUDA torch (see `pyproject.toml`). ManiSkill3
assets and the SAPIEN PhysX library are fetched by Bazel, so no runtime download is needed.
