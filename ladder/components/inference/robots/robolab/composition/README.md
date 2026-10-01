# RoboLab long-horizon composition benchmark

Systematic long-horizon manipulation tasks for RoboLab, mirroring the extended SimplerEnv / RobotArena
composition benchmarks: the same task-type families (single-destination, split across receptacles,
distractors, ordered sequence) scaled along an N-goal axis (2 → 10), so we can measure how flat VLAs,
a privileged oracle, and the HLC orchestrators degrade as the horizon grows.

## Design

Every composition task is one `CompositionSpec` (`specs.py`): a subset of the object catalog
(`catalog.py`), each object assigned to a receptacle, with an optional ordering constraint. The N-goal
count is `len(goals)`. `factory.make_task(spec_name)` turns a spec into a RoboLab `Task`
subclass, and `tasks.py` defines one such class per spec. RoboLab addresses a single one of them as
`<path to tasks.py>::<ClassName>` (`specs.task_target`), so a spec name is all a caller passes.

## The task set

**44 tasks, and `SPECS` is the whole benchmark.** There is no calibration tier, no easy/hard split
and no held-out set: one list, pinned name by name in `test/test_composition_benchmark.py`, so a task
cannot enter or leave the suite as a side effect of editing a generator.

| family | tasks | goals |
|---|---|---|
| single pick-and-place (bin, plate) | 2 | 1 |
| single-destination into a bin | 3 | 2, 6, 10 |
| distractor (`everything except ...`) | 2 | 2, 4 |
| route (two receptacles, plate + bin) | 10 | 2–10 |
| split to a plate | 7 | 2–8 |
| ordered into a bin | 9 | 2–10 |
| relay (table → plate → bin) | 4 | 4, 6, 8, 10 |
| revisit | 3 | 3, 6, 9 |
| transfer (off the plate into the bin) | 2 | 2, 3 |
| identical instances (counting) | 2 | 2, 4 |

Every task names the objects it asks for. A collective phrasing (`put everything in the bin`) is word
for word identical at two objects and at ten, so a ladder built on it measures how many placements
fit in the step budget rather than whether the instruction was understood; naming them cost the
direct instruction 19.4 points across the bin rungs and left the privileged rule where it was.

`FRUIT_LADDER` is the object order the N-goal tasks draw from, and the pomegranate is deliberately
last: it is the only object BOTH the DROID checkpoint and π0.5 fail on (9/24 and 0/24 against 21–24
for every other catalog object), so at any earlier position it capped every longer rung by the asset
rather than by the horizon. The rest of the order is NOT ranked by our own checkpoint's per-object
rates — doing that scores worse for π0.5 at every rung from two to five than the order already here,
which is what fitting a benchmark to the model under test looks like when measured.

## Where objects start

A graspable's distance from the base dominates everything else in this suite. One lemon into the bin,
16 episodes per position, nothing else varied:

| distance | placed |
|---|---|
| 0.41 m | 21/24 |
| 0.63 m | 21/24 |
| 0.77 m | 9/24 |

`_scatter` maximises the minimum separation between objects and `_OBJ_RECT` admits 0.83 m, so from two
goals up it authored everything after the first at ~0.80 — a 50-point handicap on every goal but one,
in every rung. The pairwise constraint already guaranteed nothing could overlap at worst-case jitter,
so the extra spread bought no safety; it only cost reach. Every spec is now laid out against
`layout.REACH_MAX` (0.62 m) at worst-case jitter — transfer and relay lay their objects out on a
receptacle, so their source is slid toward the base until it complies instead of being positioned by
hand. Worst start distance across the registry: 0.619 m.

The bound is on an object's CENTRE, which makes it exactly invariant to the reset's full-yaw
randomization (yaw rotates an object about its own centre). Position jitter is handled by evaluating
at the far corner of the jitter box, so the bound holds for every episode rather than on average. The
price is per-episode variation at the long rungs: jitter tapers from 0.07 m at one to three goals down
to 0.02 m at eight to ten, since the tighter band holds less.

Scores collected before this change were measured on different geometry and do not describe these
scenes.

`factory._author_usda` writes one scene USD per task into `$TMPDIR/robolab_composition_scenes/`:
a bare tabletop rig (`fixtures/table_oak.usd` + `fixtures/franka_table.usd`, ground plane, physics
material) with every graspable and receptacle attached as a direct child of `world` through a USD
**payload**. Nothing is hand-authored and no asset is re-vendored — the payloads point straight into
RoboLab's own `ASSET_DIR`, so the whole benchmark stays a ladder-side addition with no fork edit.

Composing the scene rather than selecting movers out of a pre-made one is what makes the per-object
scoring possible: `import_scene` scrapes those direct children as single rigid bodies, so a contact
sensor resolves one body per name. A runtime spawn of a raw catalog USD nests its bodies instead,
which breaks the per-object contact filter every goal predicate is built on.

The pools the specs draw from, both in `catalog.py`:

| pool | entries |
|---|---|
| `CATALOG` movers | 10 produce (banana, two oranges, lemon, lime, avocado, pomegranate, lychee, onion, pumpkin) + 2 coloured blocks (red, blue), decoys only |
| `RECEPTACLE_TYPES` | `bin` (in), `bowl` (in), `plate` (on) |

Only known-graspable produce is used as a mover: RoboLab's droid policy is grasp-floored on
blocks/cans, so a benchmark built from those would measure grasp, not orchestration. Every catalog
entry is verified single-rigid-body and confirmed to render (`objaverse/apple_01` is excluded — it
spawns invisible).

## Why the oracle works for free

A spec's `subtasks` is one `pick_and_place[_on_surface]` per receptacle group with `logical="all"`.
`oracle.build_oracle` expands a `logical="all"` group into one goal per object, so the privileged
plan is exactly the per-object N-goal sequence — no per-scene oracle authoring. The `_plan_*`
systems read `composition/decomp_cache.json`, which carries an instruction-echo plan per task;
`_plan_verifier_live` needs no cache (it decomposes the reset frame live).

## Inspecting

`python3 -m ladder.components.inference.robots.robolab.composition.specs`, run from the repo root,
prints the full task manifest (name, N, family, scene, instruction).

## Running an eval

These run through the SAME launcher as RoboLab's own tasks: `TASKS` takes composition task **names**
(env_server resolves a `comp_*` name to its Task class in `tasks.py`), and
`LADDER_CONFIG` selects the hierarchical system — exactly the shape `SCENES`/`ENV_NAMES` +
`RA_CLIENT_CONFIG`/`SIMPLER_CONFIG` have on the RobotArena and SimplerEnv launchers.

```bash
SESSION_NAME=<droid-subset session> CKPT_NAME=model-checkpoint-ema-015000.pt \
LADDER_CONFIG=ladder/components/inference/models/vlams/configs/robolab/composition/vlam_robolab_composition_oracle.yaml \
TASKS="comp_bin_6 comp_split_plate_4" \
NUM_ENVS=8 NUM_RUNS=1 VIDEO=True \
bash ladder/components/inference/robots/robolab/scripts/vlam_robolab.sh
```

`LADDER_CONFIG` selects the system: `vlam_robolab_composition.yaml` is flat (the model gets only the
composed instruction), and one `_<system>` variant exists per hierarchical system in
`api/client/hlc/configs/`, named after it — `_oracle`, `_plan_gripper`, `_plan_periodic`,
`_plan_verifier`, `_plan_verifier_interval`, `_plan_verifier_local`, `_stepwise_gripper`,
`_stepwise_interval`. `_plan_verifier_live` is not a system of its own: it is `_plan_verifier` with
`cache_path` omitted, so the VLM decomposes the reset frame. Each writes to its own
`robolab_composition*/` directory. The full table, including the RoboLab-only variants
(`_plan_verifier_retry`, `_plan_verifier_noretry`), is in the
[package README](../README.md#hierarchical-eval-systems).

**Task selection.** `TASKS` is the only env control — pass a subset, or the whole registry:

```bash
# every composition task, in registry order (44 of them)
TASKS=$(python3 -c 'from ladder.components.inference.robots.robolab.composition.specs \
        import SPECS; print(" ".join(SPECS))')

# longest budget first, which shortens the tail: a lane owns a slice for the life of one Isaac app,
# so a lane dealt the long tasks finishes last
TASKS=$(python3 -c 'from ladder.components.inference.robots.robolab.composition.specs \
        import SPECS; print(" ".join(sorted(SPECS, key=lambda n: -SPECS[n].episode_length_s)))')
```

**Language probes.** Two independent knobs on the runner, the same pair the other two composition
benchmarks use, in `models/vlams/configs/robolab/composition/language/`.
`runner_config.instruction_override` rewords the TASK instruction, keyed by the env_name env_server
reports — the RoboLab Task CLASS name (`CompBin4Task`, not `comp_bin_4`); a key matching no task is
silently ignored, so check names against `task_class_name`. RoboLab additionally has an env-side instruction axis
(`INSTRUCTION_TYPE=collective|enumerated`, from each spec's own `instruction` dict), which is
independent of both.

Per-episode `episode_meta.yaml` records the subtask switches (`instruction_log`), so the dashboard
shows the subtask timeline.
