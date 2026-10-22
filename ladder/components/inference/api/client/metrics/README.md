# Episode metrics and the offline orchestrator benchmarks

This package computes the per-episode metrics written to every cell's `results.yaml`, and holds the
tools for two offline benchmarks that score a VLM planner or verifier without running a policy.

`plan.py` holds the metrics that ask about a controller's PLAN rather than its control:

* whether the plan names the reference goals,
* whether it invents any,
* whether its subtasks are phrased in a form the action expert was trained on.

Read the scope note at the top of that module before quoting any of them. They answer by
pattern-matching English against the three current suites (SimplerEnv, RobotArena and RoboLab).

## The decomposition benchmark

The benchmark asks the plan questions of a planner in isolation. One reset frame and one compound
instruction go in, a subtask list comes out, and the list is scored against the suite's own
reference plan. This isolates decomposition from execution, so planners can be compared without
running a policy at all.

A planner is scored on the frame its env server publishes, not on the exporter's raw render. RoboLab
letterboxes to 224x224 before a controller sees anything, while its exporter writes the native
1280x720 render. Scoring the render would show the planner 25x the pixels a live arm gets, which
flatters weaker models most. The other two suites publish their camera frame untouched.

The benchmark is a view over the composition benchmarks, not a new dataset. Each suite exports its
own scenes from its own simulator, and the benchmark is the concatenation. There is no
object-position randomization: every scene contributes the single layout a controller decomposes
from in an eval (SimplerEnv env 0 / episode 0, RobotArena's native K=1 spawns, RoboLab's seeded
reset).

### What counts as correct

A plan is compared against the scene's `subtasks`, the task's curated decomposition. This is the
same reference the privileged controller is served, and it defines success for the task. The
comparison is on HEAD NOUNS and counts MULTIPLICITY. Two plans agree when they name the same
multiset of (object, destination) pairs. So "put the metal pot on the yellow cloth" and "put the
pot on the cloth" are one goal, while a scene holding two oranges needs two lines. Order is not
scored.

Head-noun matching makes the comparison possible across wordings, but it is lossy in both
directions. A true synonym ("chicken leg" against "chicken drumstick") scores as wrong, so a naming
error count is an upper bound on genuine misidentification. The scope note at the top of `plan.py`
applies in full here.

Lead with `exact` (plan equals reference as a multiset). `goals_invented` is unbounded, and for a
planner that degenerates into repetition it is set by `max_tokens` rather than by the model. Count
degenerate scenes separately instead.

### Rules the benchmark is measured under

The benchmark stands in for real-world inference, so a planner gets ONE frame and ONE compound
instruction and nothing else. Two rules follow:

* **No privileged input.** Naming the scene's objects in the prompt helps substantially, but the
  names can only come from the reference and no robot has them. `score_decomposition` has no flag
  for it.
* **No prompt tuning against these scenes.** Gains from rewriting the system prompt around observed
  failures disappeared once the dataset's object nouns were removed from the rewrite. The gain was
  leaked vocabulary, so the prompt is not overridable and every planner is scored under the
  deployed one. If you do change the prompt, ablate it LEXICALLY. Structural equivalence is not
  enough.

### How the pieces connect

Every suite has an `export_scenes` target that writes PNG frames plus a `scenes.json` into one
directory. That file is the only interface between the suites and everything downstream. It is a
list of records with `id, benchmark, scene, instruction, subtasks, n_goals, ordered, frames`, where
`frames` maps a camera name to a file NAME next to the manifest. `score_decomposition` globs
`<frames-dir>/*/scenes.json`, so adding a suite means adding an exporter and nothing else. The
suite is identified by the `benchmark` field of each record, not by its directory. Because frames
are named relative to the manifest, the directory can be copied between machines as-is.

The exporters cannot share code, since each needs its own simulator and pip environment. What they
share is the record schema.

### Recreating it

Render once per suite, on a GPU node. Each command writes frames plus a `scenes.json` into its own
directory:

```
bazel run //ladder/components/inference/robots/simpler:export_scenes -- \
    --out-dir $BENCH/frames/simplerenv
bazel run //ladder/components/inference/robots/robot_arena:export_scenes -- \
    --out-dir $BENCH/frames/robotarena
bazel run //ladder/components/inference/robots/robolab:export_scenes -- \
    --out-dir $BENCH/frames/robolab --headless \
    --task comp_bin_2 comp_bin_6 ...   # composition specs, by name
```

Gather the three directories under one root, then score:

```
bazel run //ladder/components/inference/models/vlams:score_decomposition -- \
    --frames-dir $BENCH/frames --out $BENCH/qwen3vl_8b.json \
    --model Qwen/Qwen3-VL-8B-Instruct
```

`--backend gateway` scores a served model instead of a local one and needs no GPU. The results file
records every scene's instruction, reference and prediction, so it describes the benchmark it was
computed against without a separate manifest.

**Check `no_plan` before believing a low score.** A transient failure makes `complete` return `""`,
which the `Decomposer` reads as the one-line plan `[task]`. The VLM's own failure budget cannot
catch this, because its rate check needs more calls than a benchmark run makes. A backend that
rejects every call would otherwise score 0.0 exact with `errors: 0`. The run stops after 10 such
scenes in a row.

**Do not lower `--max-tokens`.** A reasoning model's hidden thinking tokens count against it, and a
truncated reply is indistinguishable in the result from a plan that missed every goal. At 1024
tokens a large share of reasoning-model replies are truncated; at the default of 8192 essentially
none are.

To run N planners in parallel, call the built wrapper (`bzl-build/bin/.../score_decomposition`)
directly rather than `bazel run` N times, which serializes on the output-base lock. That wrapper is
NOT stable: any later bazel command in the same worktree rebuilds `bzl-build/bin` and deletes it
mid-run, so run no bazel there while scoring jobs are live.

### Scoring without a simulator

The rendered benchmark is published, so `--dataset` replaces `--frames-dir` and skips the three
renders entirely. No simulator, no assets, and no GPU if the backend is served:

```
bazel run //ladder/components/inference/models/vlams:score_decomposition -- \
    --dataset INSAIT-Institute/LADDER-Offline-VLM-Benchmark \
    --out $BENCH/served.json --backend gateway \
    --model <served-model-name>
```

`--backend gateway` sends the calls to the OpenAI-compatible endpoint in `VLM_BASE_URL`, with the
key in `VLM_API_KEY`, through the same client an eval uses (`hlc/configs/vlm.yaml`). A full-dataset
run bills roughly 0.13 M input and 0.10 M output tokens, most of it thinking.

The dataset has one row per scene, covering SimplerEnv, RobotArena and RoboLab. Each row holds the
record fields above plus `image_main` and `image_wrist` (RoboLab only; the other two suites publish
one view). It is read through `huggingface_hub` and `pandas`, both already in this pip env, and
deliberately NOT through `datasets`, which would enlarge every eval lane's build to serve one
convenience.

### Notes on results

Score only each subtask's DESTINATION, ignoring which object it names, and strong served planners
are right on essentially every scene. The residual error is almost entirely the object noun, which
is why a stronger reasoning model does not automatically lead. Degeneration (repeating one line up
to the token cap) is model-family specific rather than a matter of size alone.

`score_decomposition` drives the shipped `Decomposer` with `cache_path` unset, which makes it
decompose from the frame instead of serving a stored plan. So it measures the path an orchestrator
takes at reset, with the same prompt and reply parsing the eval uses.

## The subtask-state benchmark

Decomposition is one of the judgements an orchestrator delegates to a VLM. The rest are questions
about the CURRENT state of the subtasks mid-episode, so they share one sample type: a frame from a
recorded rollout plus the ground-truth completion state of every subtask at that instant (the
`subtask_states` config of the published dataset). Three tasks are scored over the same samples, so
every VLM-facing decision can be compared offline before paying for it in simulator time:

* **`verification`**: one mid-episode frame and one subtask in, done/not-done out, against the
  simulator's own per-goal predicate. Scored through the shipped `Verifier`, with the same prompt,
  reply parsing and multi-view handling an eval uses.
* **`state_tracking`**: the same frames, asked about EVERY subtask of the plan. This is the
  bookkeeping an orchestrator keeps ("which goals are already done"), measured as per-goal accuracy
  and exact-frame rate.
* **`next_subtask`**: the stepwise configuration in isolation. Frame plus compound instruction in,
  one subtask out, through the shipped `VLMSource` with the deployed generator prompt. A reply
  names a still-undone goal (`valid`), re-commands a finished one (`redundant`, the bookkeeping
  error an orchestrator pays budget for), or neither (`invalid`). An ordered task accepts only the
  first unachieved goal.

What the samples deliberately do NOT measure: the trigger, withholding, revocation and retry/defer
policies. Those are deterministic code paths whose value depends on closed-loop dynamics (a
verifier is queried on states its own decisions induced), so they stay simulator ablations. The
samples rank the MODELS behind a variant; the top of that ranking is what earns a full eval.

### Where a sample comes from

Every recorded eval episode already carries the ground truth. `episode_log.yaml` logs the env's
reference `subtasks` and `goal_trace` (change points of the privileged per-goal vector), and the
recording holds one frame per step per camera. `export_subtask_states` cuts labeled frames from
that, with no simulator, no GPU and no new render path:

```
bazel run //ladder/components/inference/api/client:export_subtask_states -- \
    --episodes <eval or session roots...> \
    --frames-dir $BENCH/frames --out-dir $BENCH/samples --per-task 10
```

Per task it keeps at most `--per-task` samples (a positive, a negative, then a regression frame when
the task produced one, then whatever balances the classes), pooled over every episode it finds, so
the benchmark stays small.

A 1-goal task contributes NEGATIVES only. RoboLab ends the episode the step its goal is achieved,
so `goal_trace` flips on the last frame and no window is stable for `SETTLE_STEPS` after it.

Sampling rules that make a label trustworthy:

* A sample is cut only where the goal vector is STABLE for `SETTLE_STEPS` around it. The predicate
  flips several steps after a release and flickers while an object rolls.
* In-progress negatives are cut well before a completion, clear of the hover phase.
* A recording whose frame count disagrees with the log's step count is refused whole.

Tags record what a frame shows: `fresh`, `in_progress`, `cold`, `regressed`, `terminal`, plus
`held` for the hover case, which is the hard case for a verifier. Accuracy can therefore be read per
situation, not only overall.

Labels inherit the predicate's own semantics (a held object at the goal surface counts as placed;
shared targets check no contact force). That is the right target, since it is what the eval
scores, but quote the numbers as agreement with the ENV, not with a human.

A query on a subtask that appears several times in the plan is kept only while all copies agree,
since a yes/no on the text cannot single out one instance. `state_tracking` asks each distinct
subtask once and counts it done when every copy is.

### Staying in sync with the tasks

The samples are downstream of the same `export_scenes` manifests the decomposition benchmark uses,
and that is the whole sync contract:

1. A task changes, so its suite's `export_scenes` re-render changes `scenes.json`.
2. `export_subtask_states` refuses any episode whose logged reference no longer matches that
   manifest (`stale_reference` in its summary), so old recordings cannot leak old tasks in.
3. `score_subtask_states` re-checks every sample against the reference at load time (in hub mode,
   against the repo's own decomposition config) and FAILS on drift. `--allow-drift` downgrades the
   failure to a warning. A republished decomposition config therefore invalidates stale samples
   everywhere, until they are re-harvested from post-change recordings.

Each record also carries `objects`, the scene's contents, read from the same code that builds the
scene. The wording fields cannot see a contents change: a task can change which objects it places
while leaving every instruction and subtask list byte-identical. A sample records the contents it
was cut under, and `check_reference` fails it when they no longer match.

### Publishing

```
bazel run //ladder/components/inference/models/vlams:publish_orchestrator_benchmark -- \
    --samples $BENCH/samples --repo INSAIT-Institute/LADDER-Offline-VLM-Benchmark
```

This pools one or more harvest out-dirs into the config's single parquet. There is one file per
config at a fixed name, because the scorer reads every parquet under the prefix and a second name
would pool two harvests. `--out` writes the file and skips the upload, which is how a schema change
is checked.

The samples are the `subtask_states` config of the same Hub repo as the scenes.
`score_decomposition` reads only the parquets under `decomposition/`, and `score_subtask_states`
only those under `subtask_states/`, so the two configs cannot contaminate each other. Current
baselines for both configs are on the dataset card.

### Scoring

```
bazel run //ladder/components/inference/models/vlams:score_subtask_states -- \
    --dataset INSAIT-Institute/LADDER-Offline-VLM-Benchmark \
    --task verification --out $BENCH/served_verify.json \
    --backend gateway --model <served-model-name>
```

`--samples-dir` scores a local harvest instead (add `--frames-dir` to sync-check it). Everything the
decomposition section says about backends, `--max-tokens` and the bazel wrapper applies unchanged.
`--image-cameras` is a deployed configuration axis, not a leak. The RoboLab verifier needs the
wrist view (its recall collapses without it), so both views are sent by default wherever a sample
has them. The prompts are the deployed ones and, as everywhere in this benchmark, not overridable.

All three tasks break down `by_benchmark` and `by_tag`. The tags are worth reading: `fresh` and
`regressed` are the frames where the answer has just changed, the rest are the steady state, and
the first group is where the difficulty is.

Both scorers ask with `VerifiedAdvancer`'s instance threshold, `1 if revisits else the copy count`,
so the question matches both the deployment and the label.

**Never quote `state_tracking`'s `goal_accuracy` on its own.** Most of its goal queries are not
done, so answering "not done" to everything already scores high. Pair it with `exact_frames`, and
compare every task against its trivial baseline (always "not done" for `verification`, a uniform
pick from the reference plan for `next_subtask`). `next_subtask` skips all-done frames: the deployed
prompt says "repeat the last subtask" there, which needs history a standalone sample does not have.

## The object-level trace

Every composition episode also logs the simulator's object state, taken from the `privileged`
payload the three envs publish. It is stored as change points, like `motion_trace`:

* `object_trace`: each object's centre when it moved.
* `held_trace`: the grasped objects when they changed.
* `goal_objects`: object, destination and relation per goal.
* `object_extents`.

`scene_events.py` derives events from this trace after the fact, with one definition for all three
simulators:

* grasps with their outcome (placed / misplaced / dropped / slipped / held) and whether the object
  was the commanded one,
* knocks, with what the gripper held and the goal they undid,
* falls and missed grasp attempts,
* one `failure_cause` per failed episode,
* the controller's decisions against object state (advances mid-carry, advances onto finished goals,
  done-verdicts with the object off its destination).

`metrics/objects.py` registers all of it, so every cell's `results.yaml` carries the `cause_*`,
`grasp_*`, `goal_knocked`, `object_fell`, `advance_*` and `verdict_off_destination` metrics next to
`control.py`'s verifier and switching metrics. Off-target grasps are split by what was grabbed
(`grasp_other_goal` / `grasp_done_goal` / `grasp_destination` / `grasp_distractor`), misplacements
by whether the object landed on another goal's destination (`misplaced_wrong_destination`), and
lasting goal losses by mechanism (`lost_regrasp` / `lost_knocked` / `lost_drift`).

`metrics/timing.py` says where the step budget went:

* `success_steps` and `first_grasp_steps`.
* `steps_per_goal`.
* `idle_tail`: the share of a failed episode after the last object event. It is high when the policy
  gave up early, and low when the budget ran out mid-work.
* `dead_time`: the share spent commanding an already-satisfied subtask, including a verifier that
  never confirmed.

The `failure_*` and `grasp_churn` metrics count gripper closures from proprioception. The `grasp_*`
and `cause_*` metrics count object holds, so a closure that held nothing is a `grasp_missed`, not a
grasp. Proprioception is per arm (`motion_trace` is keyed by arm name), so a closure is read against
the openness range of the arm that made it, and reach is measured from that arm. The object trace
stays robot-level: the simulator publishes a `grasped` flag per object, which says that an object
is held but not which arm holds it, so a hold is never attributed to an arm.

```
bazel run //ladder/components/inference/api/client:event_report -- <eval dir> [--timeline <unit>]
bazel run //ladder/components/inference/api/client:rescore -- <eval dir>   # after changing a definition
```
