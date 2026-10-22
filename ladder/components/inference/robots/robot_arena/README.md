# RobotArena-Infinity Integration

Integration of the [RobotArena-Infinity](https://github.com/offjangir/RobotArena) benchmark
into the ladder evaluation pipeline. RobotArena evaluates robot manipulation policies under
systematic scene perturbations (background, camera, object pose, object permutation) using
the Genesis physics simulator and a deterministic geometric success checker.

## Architecture

The integration follows the same workflow as SimplerEnv:

```
SimplerEpisodeRunner  -->  RobotEnv (ZMQ client)  -->  RobotArenaServer (ZMQ server)
                                                            |
                                                       RobotArenaEnv (Genesis)
```

| Module                | Role |
|-----------------------|------|
| `robot_arena_env.py`  | Genesis environment wrapper (reset/step interface) |
| `robot_arena_server.py` | ZMQ server enumerating episodes, compatible with `SimplerEpisodeRunner` |
| `scene_loader.py`     | Loads scene data from disk (meshes, camera calibration, tasks) |
| `transforms.py`       | Coordinate frame conversions (canonical <-> Genesis robot frames) |
| `grasping.py`         | Force-based gripper control and weld-constraint grasping |

## Quick start

```bash
SESSION_NAME=sess_... CKPT_NAMES=model-checkpoint-ema-030000.pt SCENES="default1 scene2" \
./ladder/components/inference/models/vlams/scripts/vlam_robot_arena.sh
```

## Configuration

All test types use a single base config (`configs/widowx.yaml`) with CLI overrides
for test-specific parameters:

```bash
# Default test (no perturbations):
--config configs/widowx.yaml

# Camera perturbation test:
--config configs/widowx.yaml --test_config.test_type camera

# Permutation test with custom seed:
--config configs/widowx.yaml --test_config.test_type permutation --test_config.seed 123

# Pose test with fewer variations:
--config configs/widowx.yaml --test_config.test_type pose --test_config.num_pose_variations 3
```

These are `robot_arena_server` flags. The launcher's default server config is
`configs/widowx_batched_unified.yaml`; `RA_SERVER_CONFIG` selects another.

## Test Types

| Type          | Description                                          | CLI override |
|---------------|------------------------------------------------------|--------------|
| `default`     | Standard scenes, no perturbations                    | (none needed) |
| `background`  | Background texture variations                        | `--test_config.test_type background` |
| `camera`      | Camera position offsets (+/- 5cm per axis)           | `--test_config.test_type camera` |
| `permutation` | Object XY position permutations                     | `--test_config.test_type permutation` |
| `pose`        | Random position (+/- 5cm) and Z-rotation variations | `--test_config.test_type pose` |

## Long-horizon composition benchmark

`composition/registry.py` defines 47 tasks (1 to 10 goals) over these scenes. They run through
the SAME launcher as the single-goal scenes: the server resolves a composition task name passed in
`--test_config.scene_names`, so `SCENES` takes task names instead of scene names.

```bash
SESSION_NAME=sess_... CKPT_NAMES=model-checkpoint-ema-030000.pt \
RA_SERVER_CONFIG=ladder/components/inference/robots/robot_arena/configs/widowx_composition_k8.yaml \
RA_CLIENT_CONFIG=ladder/components/inference/models/vlams/configs/robot_arena/composition/vlam_robot_arena_composition.yaml \
SCENES="scene45_2goal scene45_4goal scene54_3goal" \
RUN_LABEL=robot_arena_composition LANES_PER_GPU=2 \
./ladder/components/inference/models/vlams/scripts/vlam_robot_arena.sh
```

`RA_CLIENT_CONFIG` selects the system: `vlam_robot_arena_composition.yaml` is flat, and one
`_<system>` variant exists per hierarchical system in `api/client/hlc/configs/`, named after it:
`_oracle`, `_plan_gripper`, `_plan_periodic`, `_plan_verifier`, `_plan_verifier_interval`,
`_plan_verifier_local`, `_stepwise_gripper`, `_stepwise_interval`. `_plan_verifier_live` is not a ninth
system: it is `_plan_verifier` with `cache_path` omitted (see **Plans** below). The two `_unified*`
variants run the 137-env layout instead of per-task batches. Each writes to its own directory. The same
system names exist on the SimplerEnv and RoboLab composition benchmarks, so a system compares across
all three.

The three plan-driven arms are meant to be read together: `_plan_periodic` switches on a clock with no
feedback, `_plan_gripper` on an event the robot produces, `_plan_verifier` on a judgement about the
scene, so a gap to `_plan_periodic` is what makes either feedback signal worth its cost. Its clock is
the shared config's 27 steps, which is SimplerEnv's median subtask length. `_plan_verifier_local` is `_plan_verifier` with the VLM
in-process (Qwen3-VL) instead of a hosted API.

`SCENES` is the only env control. Omit it to sweep the single-goal scenes; pass task names to run
composition. The 47 tasks (`COMPOSITION_TASKS` in `composition/registry.py`):

```
scene36_1goal         scene45_1goal         scene54_1goal
scene76_1goal         scene36_2goal         scene45_2goal
scene45_all_2
scene45_except_2      scene54_2goal         scene54_2goal_ordered
scene54_duplicates_2  scene54_except        scene54_relay_2
scene54_route_2       scene54_split_2       scene54_transfer_2
scene76_2goal         scene76_2goal_ordered
scene45_3goal         scene45_all_3         scene45_except_3
scene50_3goal         scene54_3goal         scene54_3goal_ordered
scene54_duplicates_3  scene54_except_3      scene54_revisit_3
scene54_split_3       scene45_4goal         scene45_all_4
scene45_except_4      scene54_relay_4       scene54_route_4
scene45_5goal         scene45_all_5         scene45_except_5
scene45_6goal         scene45_all_6         scene45_except_6
scene54_revisit_6     scene54_route_6       scene54_revisit_7
scene54_route_7       scene54_revisit_8     scene54_route_8
scene54_revisit_9     scene54_revisit_10
```

The `_all` and `_except` rungs are the enumerated ladder re-worded: one instruction string over
rungs 2 to 6, so a fall in success reads as the length rather than as the referring expression.
`scene54_relay_4` / `_revisit_6` / `_revisit_9` add a LEG per rung rather than an object, which is
how the suite gets past six placements in a workspace that holds six movers. `scene54_revisit_7` /
`_8` / `_10` take that to its limit: one spoon alternating between the cloth and the pot, on the
3-goal rung's own scene, so every goal count from 2 to 10 is covered without packing the counter.
`scene54_route_2` to `_8` give that length axis a second shape -- several DISTINCT objects, two legs
each, staged through the pot onto the cloth. Routing onward is what lets it scale: a relay leaves
each mover in the walled pot, which holds two or three and cannot be scaled, while a route asks the
pot for one object at a time and accumulates the arrivals on the flat cloth. Its odd rung sends the
last mover straight to the cloth, so the instruction is not one clause repeated.

**Every object on the table is named by the instruction**, except in the two `_except` families,
whose wording ("put everything except the pot on the cloth") is what makes an unnamed object part of
the task. Each scene ships with its own objects and the tasks use different subsets, so a task lists
the ones it does not use in `remove_objects`; each scene's background plate is an empty table, so a
removed object leaves bare wood.

A lane picks up the next task as soon as any lane frees up, so ordering affects only the tail; step
budgets are 100 per goal (100 to 1000 here).

**Plans** are a config concern, not the launcher's: only the `_plan_*` systems (and
`_unified_plan_verifier`) read `composition/decomp_cache.json`, via
`hlc_config.source.cache_path`. Flat and `_oracle` need no plans at all (the oracle's come from the
env) and the `_stepwise_*` systems hold none by design: each switch generates the next subtask from
the current frame. `_plan_verifier_live` just omits `cache_path`, so the VLM decomposes the reset
frame instead. That is the pass that mints a plan for a newly added task.

**Language probes.** Two independent knobs on the runner, for testing how much instruction following
matters. They are the same pair the SimplerEnv composition benchmark uses.
`runner_config.instruction_override` rewords the TASK instruction, keyed by the env_name the server
reports (`robot_arena_<task>`, not the bare task name); a key matching no task is silently ignored.
It lives in `models/vlams/configs/robot_arena/composition/language/`:
`vlam_robot_arena_composition_paraphrase.yaml`.

### Exporting the decomposition-benchmark scenes

`export_scenes` renders each task's reset frame and writes a `scenes.json` beside the frames holding
each task's instruction, its cached plan, goal count and ordering. Frames are named relative to that
file, so the directory can be copied to whichever machine scores it.

```bash
bazel run //ladder/components/inference/robots/robot_arena:export_scenes -- --out-dir $BENCH/frames/robotarena
```

It drives `RobotArenaEnv` rather than rendering the camera directly, because the observation pipeline
composites the foreground over the scene's real-image background -- a bare render puts the objects on
a dark void. `robots/simpler` and `robots/robolab` carry the same target for their own suites, and the
concatenation of the three `scenes.json` files is the benchmark -- see
`api/client/metrics/README.md`.

## Binary Success Checker

The `success_checker/` module provides deterministic, per-step success evaluation
using object geometry (positions and AABBs). It sets `metadata["done"]` on every
observation, following the same convention as SimplerEnv. This runs server-side
inside `RobotArenaEnv` and is independent of the client-side VLA model. Language
perturbation configs (paraphrase, negation, etc.) do not affect it since the
checker always evaluates against the original scene instruction.

### How it works

At `env.reset()`, the checker:

1. **Parses** the scene's text instruction (e.g. "put spoon on cloth") into a
   structured `TaskSpec` identifying the task type, target object, destination,
   and spatial relation. 46 scenes use explicit per-scene overrides in
   `TASK_SPEC_OVERRIDES` for ambiguous instructions; the remaining scenes are
   handled by priority-ordered regex patterns.
2. **Matches** object references (e.g. "spoon") to actual Genesis entities in the
   scene using 4-tier fuzzy name matching: exact → suffix-stripped (`_target`/
   `_destination` removed) → substring containment → token overlap (≥50% coverage).
3. **Records** initial object positions for displacement-based checks.

At each `env.step()`, the checker calls `get_pos()` / `get_AABB()` on the relevant
entities and evaluates the geometric predicate for the task type:

| Task Type | Example | Success Condition |
|-----------|---------|-------------------|
| `PICK_AND_PLACE` | "put spoon on cloth" | Target center XY within destination AABB footprint, Z near destination top |
| `STACK` | "stack green cube on red block" | Same as PICK_AND_PLACE (on-top check) |
| `PICKUP` | "pick up eggplant" | Target Z > initial Z + 5cm |
| `MOVE_RELATIVE` | "move spatula left of pot" | Target displaced from start, in correct direction relative to reference (camera-frame) |
| `MOVE_TO_REGION` | "move cloth to upper right" | Target displaced from start, in correct region of table (camera-frame) |
| `BETWEEN` | "place X between Y and Z" | Target XY within 5cm of midpoint between the two references |
| `PUSH` | "push carton to back" | Same as MOVE_TO_REGION |
| `REACH` | "reach the pan" | Robot end-effector within 5cm of target, or target lifted as fallback |

Spatial directions ("left", "right", "upper", "lower") are resolved in the
camera's image frame by projecting the extrinsics rotation matrix onto the world
XY plane. This means "left of the pot" is left as seen from the camera, matching
how a human would interpret the instruction from the observation image.

### Adding a new scene

When a new scene is added, `test/test_scene_manifest.py` verifies that
its instruction parses and matches correctly. If parsing fails (unusual phrasing,
object name mismatch), add an entry to `TASK_SPEC_OVERRIDES` in `parsing.py`:

```python
"scene123": TaskSpec(TaskType.PICK_AND_PLACE, "toy knife", "cutting board", spatial_relation="on"),
```

Then verify: `bazel test //ladder/components/inference/robots/robot_arena:robot_arena_tests`

### Thresholds

All distance thresholds are in `SuccessThresholds` (frozen dataclass with defaults)
and can be overridden per-checker instance. Key values: on-top height tolerance
-2cm to +8cm, XY margin 3cm, pickup height 5cm, displacement minimum 3cm,
between tolerance 5cm, reach distance 5cm.

## Results Directory Structure

`CAMPAIGN_ROOT` defaults to `$SESSIONS_HOME/$SESSION_NAME` and `RUN_LABEL` to `robot_arena`.

```
<CAMPAIGN_ROOT>/<RUN_LABEL>/<checkpoint>/
    batch_inference_server.log
    <scene or task>/
        results.yaml          # Per-scene metrics (see api/client/metrics/README.md)
        episode_log.yaml      # Per-episode metadata
        ra_server.log, client.log
        episodes/
            episode_<i>_..._success/   # or _failure
                <camera>.mp4      # one video per camera, e.g. main.mp4
                episode_meta.yaml
```

## Comparison with SimplerEnv

| Aspect               | SimplerEnv                          | RobotArena                                |
|-----------------------|-------------------------------------|-------------------------------------------|
| Simulator             | ManiSkill3 (SAPIEN)                | Genesis (GPU physics)                     |
| Success detection     | Inline (env reports done=True)     | Inline (geometric `success_checker/`)     |
| Scene source          | Predefined Gymnasium envs          | Procedural 3D-reconstructed scenes        |
| Perturbation types    | Robot/object init pose ranges      | Background, camera, permutation, pose     |
| Batched evaluation    | Supported (GPU-vectorized envs)    | Supported (n_envs=K GPU-parallel)         |
| Server                | `simpler_server`                   | `robot_arena_server`                      |
| Client                | `SimplerEpisodeRunner` (shared)    | `SimplerEpisodeRunner` (shared)           |
| Config pattern        | `vlam_simpler.yaml`                | `vlam_robot_arena_batched_envs.yaml` (imports simpler) |
| Benchmark script      | `vlam_simpler_bridge.sh`           | `vlam_robot_arena.sh`                     |

## Data

Scene data (meshes, camera calibration, task descriptions) is downloaded automatically
by the Bazel build. The data is stored under `robot_arena_data/` in the build output.

## Supported Robots

| Robot   | Status         | Notes                                    |
|---------|----------------|------------------------------------------|
| WidowX  | Fully supported | Tuned PD gains, gripper config, validated transforms |
| Franka  | URDF only      | Needs PD gains, `gripper_dof_indices`, `gripper_open`/`gripper_closed`, transform validation |
| UR5     | URDF only      | Needs PD gains, `gripper_dof_indices`, `gripper_open`/`gripper_closed`, transform validation (`transforms.py` uses identity placeholder) |

To add full support for Franka/UR5: fill in the missing fields in `ROBOT_CONFIGS`
(scene_loader.py) and empirically validate the correction rotation in `transforms.py`
by comparing the neutral-pose EEF orientation reported by Genesis with the canonical
(Franka) convention.

## Troubleshooting

**Genesis scene build takes minutes on first run**: This is expected. Genesis uses CoACD
convex decomposition for collision geometry, which is cached after the first build.
`models/vlams/configs/robot_arena/vlam_robot_arena_batched_envs.yaml` sets a 600 s receive timeout to account for this.

**CUDA OOM with large scenes**: Genesis loads all meshes into GPU memory. For scenes
with many objects, reduce `camera_width`/`camera_height` or use a GPU with more VRAM.

**ZMQ connection timeout**: If the client times out before the server finishes building
the Genesis scene, increase `recv_timeout_ms` in `vlam_robot_arena_batched_envs.yaml`.

**"No scenes found" error**: Verify the Bazel build downloaded the data:
`ls bazel-bin/ladder/components/inference/robots/robot_arena/robot_arena_data/examples/data/bridge/`
