# Plan metrics and the offline orchestrator benchmarks

`plan.py` holds the metrics that ask about a controller's PLAN rather than its control: whether the
plan names the reference goals, whether it invents any, and whether its subtasks are phrased in a
form the action expert was trained on. Read the scope note at the top of that module before quoting
any of them -- they answer by pattern-matching English against the three suites we have today.

## The benchmark

The same questions, asked of a planner in isolation: one reset frame and one compound instruction in,
a subtask list out, scored against the suite's own cached plan. It isolates decomposition from
execution, so a planner can be compared without running a policy at all.

A planner is scored on the frame its env_server publishes, not on the exporter's raw render.
RoboLab letterboxes to 224x224 before a controller sees anything while its exporter writes the
native 1280x720, so scoring the render would show the planner 25x the pixels a live arm gets. Over
the 42 RoboLab scenes the render is worth +1 exact plan to GPT-5.6-sol and +3 to
Nemotron-Nano-12B-VL, so it flatters weaker models most. The other two suites publish their camera
frame untouched.

It is a view over the composition benchmarks, not a new dataset. Each suite exports its own scenes
from its own simulator, and the benchmark is the concatenation. There is no object-position
randomization: every scene contributes the single layout a controller decomposes from in an eval
(SimplerEnv env 0 / episode 0, RobotArena's native K=1 spawns, RoboLab's seeded reset).

### What counts as correct

A plan is compared against the scene's `subtasks`, which is the task's curated decomposition -- the
same reference the privileged controller is served and which defines success for that task. The
comparison is on HEAD NOUNS and counts MULTIPLICITY: two plans agree when they name the same multiset
of (object, destination) pairs, so "put the metal pot on the yellow cloth" and "put the pot on the
cloth" are one goal, while a scene holding two oranges needs two lines. Order is not scored.

Head-noun matching is what makes the comparison possible across wordings and it is lossy in both
directions -- a true synonym ("chicken leg" against "chicken drumstick") scores as wrong -- so a
naming error count is an upper bound on genuine misidentification. See the scope note at the top of
`plan.py`, which applies in full here.

`exact` (plan == reference as a multiset) is the number to lead with. `goals_invented` is unbounded
and, for a planner that degenerates into repetition, is set by `max_tokens` rather than by the model:
the same Qwen3-VL-2B run scores 17.5 at 1024 tokens and 23.8 at 8192. Count degenerate scenes
separately instead.

### Rules this benchmark is measured under

The benchmark stands in for real-world inference, so a planner gets ONE frame and ONE compound
instruction and nothing else. Two things follow, and both were tried and rejected on evidence:

* **No privileged input.** Naming the scene's objects in the prompt is worth 14-18 scenes of 92, but
  the names can only come from the reference and no robot has them, so it is not a configuration
  this benchmark can report. `score_decomposition` has no flag for it.
* **No prompt tuning against these scenes.** Rewriting the system prompt from the observed failures
  gained 4.5 exact plans of 92 on average and exactly 0.0 once the same instructions were restated
  without the dataset's object nouns (Qwen3-VL-2B/8B/32B, Gemma-3-4B). The gain was leaked
  vocabulary, so the prompt is not overridable and every planner is scored under the deployed one.
  If you do change the prompt, ablate it LEXICALLY -- structural equivalence is not enough.

### How the pieces connect

Every suite has an `export_scenes` target that writes PNG frames plus a `scenes.json`
into one directory. That file is the only interface between the suites and everything downstream: a
list of records with `id, benchmark, scene, instruction, subtasks, n_goals, ordered, frames`, where
`frames` maps a camera name to a file NAME next to the manifest. `score_decomposition` globs
`<frames-dir>/*/scenes.json`, so adding a suite means adding an exporter, nothing else. The suite is
identified by the `benchmark` field in each record, not by the directory it was written to, and
naming frames relative to the manifest is what lets the directory be copied between clusters as-is.

The exporters cannot share code: each needs its own simulator and pip environment, which is also why
each lives on its own suite's branch. What they share is that record schema.

### Recreating it

One render per suite, on a GPU node. Each writes frames plus a `scenes.json` into its own directory:

```
bazel run //ladder/components/inference/robots/simpler:export_scenes -- \
    --out-dir $BENCH/frames/simplerenv
bazel run //ladder/components/inference/robots/robot_arena:export_scenes -- \
    --out-dir $BENCH/frames/robotarena
bazel run //ladder/components/inference/robots/robolab:export_scenes -- \
    --out-dir $BENCH/frames/robolab --headless \
    --task comp_bin_2 comp_bin_6 ...   # composition specs, by name
```

Gather those three directories under one root (they are rendered on whichever cluster has the GPU),
then score:

```
bazel run //ladder/components/inference/models/vlams:score_decomposition -- \
    --frames-dir $BENCH/frames --out $BENCH/qwen3vl_8b.json \
    --model Qwen/Qwen3-VL-8B-Instruct
```

`--backend gateway` scores a served model instead of a local one and needs no GPU. The results file
records every scene's instruction, reference and prediction, so it describes the benchmark it was
computed against without a separate manifest.

**Check `no_plan` before believing a low score.** A transient failure makes `complete` return `""`,
which the `Decomposer` reads as the one-line plan `[task]`, and the VLM's own failure budget cannot
catch it because its rate check needs 200 calls against 143 scenes. One model rejects every call with
`400 "Prompting with images is incompatible with system messages"` and used to score 0.0 exact with
`errors: 0`. The run now also stops after 10 such scenes in a row.

**Do not lower `--max-tokens`.** A reasoning model's hidden thinking tokens count against it and a
truncated reply is indistinguishable in the result from a plan that missed every goal. Over 803
gateway calls the median spends 635 and p99 spends 7940: 1024 truncates 22.5% of calls, 8192 none.
The default is 8192 for that reason, and one planner scored as "missing 43% of goals" at 1024 when
nothing was wrong with it.

Run N planners by calling the built wrapper (`bzl-build/bin/.../score_decomposition`) directly rather
than `bazel run` N times, which serializes on the output-base lock. That wrapper is NOT stable: any
later bazel command in the same worktree rebuilds `bzl-build/bin` and deletes it mid-run, so run no
bazel there while scoring jobs are live.

### Scoring without a simulator

The rendered benchmark is published, so `--dataset` replaces `--frames-dir` and skips the three
renders entirely -- no simulator, no assets, no GPU if the backend is served:

```
bazel run //ladder/components/inference/models/vlams:score_decomposition -- \
    --dataset INSAIT-Institute/LADDER-Offline-VLM-Benchmark \
    --out $BENCH/gemini.json --backend gateway \
    --model google-ai-studio/gemini-3-flash-preview
```

The gateway endpoint and its cost-tracking headers come from `hlc/configs/vlm.yaml`, so a served
planner is scored through the same client an eval calls. It routes an ALLOWLIST, not a provider:
`gemini-3.1-pro-preview`, `gemini-robotics-er-2-preview` and the flash previews work, everything
else returns 500 ("Route terminated") or 404 regardless of whether the model exists.
`~/.config/cloudflare_gateway/credentials.env` supplies `VLM_API_KEY`. Cost is negligible -- a
full-dataset run bills ~0.13 M input and ~0.10 M output tokens, of which the great majority is
thinking -- so the constraint is the allowlist, not budget.

One row per scene (145 as of the 2026-09-12 re-render: 54 SimplerEnv, 47 RobotArena, 44 RoboLab):
the record fields above plus `image_main` and `image_wrist` (RoboLab only;
the other two suites publish one view). Read through `huggingface_hub` and `pandas`, both already in
this pip env -- deliberately NOT through `datasets`, which is in neither the inference-client nor the
vlams env and would enlarge every eval lane's build to serve one convenience.

does need `datasets`, and pushing to a personal namespace is distribution, not pipeline).

### What has been scored (2026-08-05, on the 92-scene revision)

Six planners over the 92 scenes published at the time, deployed prompt, `--max-tokens 8192`,
temperature 0. Exact plans of
92: Qwen3-VL-2B 49, Gemma-3-4B 51, Qwen3-VL-8B 55, Qwen3-VL-32B 57, Gemini 3 Flash 76, Gemini 3.1
`~/decomp_bench/`; `aggregate.py` beside them rolls the runs into one table.

Two results are worth knowing before repeating any of this. Score only each subtask's DESTINATION,
ignoring which object it names, and both served planners are exactly right on every scene -- the
plans are correct and the whole residual is the object noun, which is why a stronger reasoning model
does not lead. And degeneration (repeating one line to the token cap) is a Qwen trait below 32B
rather than a small-model one: Gemma-3-4B has none at a comparable size.

`score_decomposition` drives the shipped `Decomposer` with `cache_path` unset, which is what makes it
decompose from the frame instead of serving a stored plan. So it measures the path an orchestrator
takes at reset, with the prompt and reply parsing the eval uses.

## The subtask-state benchmark

Decomposition is one of the judgements an orchestrator delegates to a VLM. The rest are questions
about the CURRENT state of the subtasks mid-episode, so they share one sample type: a frame from a
recorded rollout plus the ground-truth completion state of every subtask at that instant (the
`subtask_states` config of the published dataset). Three tasks are scored over the same samples,
so every VLM-facing decision can be compared offline before any variant is paid for in simulator
time:

* **`verification`** -- one mid-episode frame and one subtask in, done/not-done out, against the
  simulator's own per-goal predicate. Scored through the shipped `Verifier`: same prompt, same
  reply parsing, same multi-view handling an eval runs.
* **`state_tracking`** -- the same frames, asked about EVERY subtask of the plan: the bookkeeping an
  orchestrator keeps ("which goals are already done") measured as per-goal accuracy and
  exact-frame rate.
* **`next_subtask`** -- the stepwise configuration in isolation: frame plus compound instruction in,
  one subtask out, through the shipped `VLMSource` with the deployed generator prompt. A reply
  names a still-undone goal (`valid`), re-commands a finished one (`redundant` -- the bookkeeping
  error an orchestrator pays budget for), or neither (`invalid`); an ordered task accepts only the
  first unachieved goal.

What the samples deliberately do NOT measure: the trigger, withholding, revocation and retry/defer
policies. Those are deterministic code paths whose value depends on closed-loop dynamics (a
verifier is queried on states its own decisions induced), so they stay simulator ablations. The
samples rank the MODELS behind a variant; the top of that ranking is what earns a full eval.

### Where a sample comes from

Every recorded eval episode already carries the ground truth: `episode_log.yaml` logs the env's
reference `subtasks` and `goal_trace` (change points of the privileged per-goal vector), and the
recording holds one frame per step per camera. `export_subtask_states` cuts labeled frames from that -- no
simulator, no GPU, no new render path:

```
bazel run //ladder/components/inference/api/client:export_subtask_states -- \
    --episodes <campaign or session roots...> \
    --frames-dir $BENCH/frames --out-dir $BENCH/samples --per-task 10
```

Per task it keeps at most `--per-task` samples (a positive, a negative, then a regression frame when
the task produced one, then whatever balances the classes), pooled over every episode it finds, so
the benchmark stays small. The current harvest is 562 samples, 42.5% positive: 176 RoboLab, 198
SimplerEnv and 188 RobotArena, all cut at `--per-task 4` so the three suites carry comparable weight.

A 1-goal task contributes NEGATIVES only. RoboLab ends the episode the step its goal is achieved, so
`goal_trace` flips on the last frame and no window is stable for `SETTLE_STEPS` after it; the two
1-goal rungs therefore give 8 samples, all `held`, which is the case a verifier finds hardest anyway. Sampling rules that make a label
trustworthy: a sample is cut only where the goal vector is STABLE for `SETTLE_STEPS` around it (the
predicate flips several steps after a release and flickers while an object rolls); in-progress
negatives are cut well before a completion, clear of the hover phase; a recording whose frame count
disagrees with the log's step count is refused whole. Tags record what a frame shows -- `fresh`,
`in_progress`, `cold`, `regressed`, `terminal`, plus `held` for the hover case that is the measured
hard case for a verifier -- so accuracy can be read per situation, not only overall.

Labels inherit the predicate's own semantics (a held object at the goal surface counts as placed;
shared targets check no contact force). That is the right target -- it is what the eval scores --
but quote the numbers as agreement with the ENV, not with a human.

A query on a subtask that appears several times in the plan is kept only while all copies agree,
since a yes/no on the text cannot single out one instance; `state_tracking` asks each distinct
subtask once and counts it done when every copy is.

### Staying in sync with the tasks

The samples are downstream of the same `export_scenes` manifests the decomposition benchmark uses,
and that is the whole sync contract:

Each record also carries `objects`, the scene's contents, read from the same code that builds the
scene. This is the half the wording fields cannot see: the 2026-09-12 distractor cleanup changed what
43 of 47 RobotArena scenes PLACE while leaving every instruction and subtask list byte-identical, so
162 samples went stale with nothing to show for it. A sample records the contents it was cut under
and `check_reference` fails it when they no longer match.

1. A task changes -> its suite's `export_scenes` re-render changes `scenes.json`.
2. `export_subtask_states` refuses any episode whose logged reference no longer matches that manifest
   (`stale_reference` in its summary), so old recordings cannot leak old tasks in.
3. `score_subtask_states` re-checks every sample against the reference at load time -- in hub mode against
   the repo's own decomposition config -- and FAILS on drift (`--allow-drift` downgrades it for
   archaeology). A republished decomposition config therefore invalidates stale samples everywhere,
   until they are re-harvested from post-change recordings.

### Publishing

```
bazel run //ladder/components/inference/models/vlams:publish_orchestrator_benchmark -- \
    --samples $BENCH/samples --repo INSAIT-Institute/LADDER-Offline-VLM-Benchmark
```

Pools one or more harvest out-dirs into the config's single parquet. One file per config at a fixed
name, because the scorer reads every parquet under the prefix and a second name would pool two
harvests. `--out` writes the file and skips the upload, which is how a schema change is checked.

### Scoring

```
bazel run //ladder/components/inference/models/vlams:score_subtask_states -- \
    --dataset INSAIT-Institute/LADDER-Offline-VLM-Benchmark \
    --task verification --out $BENCH/gemini_verify.json \
    --backend gateway --model google-ai-studio/gemini-3.6-flash
```

`--samples-dir` scores a local harvest instead (add `--frames-dir` to sync-check it). Everything the
decomposition section says about backends, the gateway allowlist, `--max-tokens` and the bazel
wrapper applies unchanged. `--image-cameras` is a deployed configuration axis, not a leak: the
RoboLab verifier needs the wrist view (4% recall without it), so both views are sent by default
wherever a sample has them. The prompts are the deployed ones and, as everywhere in this benchmark,
not overridable.

All three tasks break down `by_benchmark` and `by_tag`. The tags are worth reading: `fresh` and
`regressed` are the frames where the answer has just changed, the rest are the steady state, and the
first group is where the difficulty is.

Both scorers ask with `VerifiedAdvancer`'s instance threshold, `1 if revisits else the copy count`,
so the question matches the deployment and matches the label.

**Never quote `state_tracking`'s `goal_accuracy` on its own.** Only 18.7% of its goal queries are
done, so answering "not done" to everything scores 81.3%. Pair it with `exact_frames`, whose floor is
43.8%. The other floors are 58.3% for `verification` and, for `next_subtask`, 62.5% valid from a
uniform pick out of the reference plan that no planner is given. `next_subtask` skips all-done frames (the deployed prompt says "repeat the last
subtask" there, which needs history a standalone sample does not have).

Publishing: the samples are the `subtask_states` config of the same Hub repo, pushed with
publisher next to it). `score_decomposition` reads only the parquets under `decomposition/`,
`score_subtask_states` only those under `subtask_states/`, so the two configs cannot contaminate each
other. Current baselines for both configs live on the dataset card.

## The object-level trace

Every composition episode also logs the simulator's object state from the `privileged` payload the
three envs publish, as change points like `motion_trace`: `object_trace` (each object's centre when
it moved), `held_trace` (the grasped objects when they changed), `goal_objects` (object, destination,
relation per goal) and `object_extents`. `scene_events.py` derives the events from it after the fact,
with one definition for all three simulators: grasps with their outcome (placed / misplaced / dropped /
slipped / held) and whether the object was the commanded one, knocks with what the gripper held and
the goal they undid, falls, missed grasp attempts, one `failure_cause` per failed episode and the
controller's decisions against object state (advances mid-carry, advances onto finished goals,
done-verdicts with the object off its destination). `metrics/objects.py` registers all of it, so every
cell's results.yaml carries the `cause_*`, `grasp_*`, `goal_knocked`, `object_fell`, `advance_*` and
`verdict_off_destination` metrics next to `control.py`'s verifier and switching metrics. Off-target
grasps are split by what was grabbed (`grasp_other_goal` / `grasp_done_goal` / `grasp_destination` /
`grasp_distractor`), misplacements by whether the object landed on another goal's destination
(`misplaced_wrong_destination`), and lasting goal losses by mechanism (`lost_regrasp` / `lost_knocked`
/ `lost_drift`). `metrics/timing.py` says where the step budget went: `success_steps`,
`first_grasp_steps`, `steps_per_goal`, `idle_tail` (share of a failed episode after the last object
event, high when the policy gave up early, low when the budget ran out mid-work) and `dead_time` (share
spent commanding an already-satisfied subtask, including a verifier that never confirmed). The
`failure_*` and `grasp_churn` metrics count gripper closures from proprioception; the `grasp_*` and
`cause_*` metrics count object holds, so a closure that held nothing is a `grasp_missed`, not a grasp.
Proprioception is per arm (`motion_trace` is keyed by arm name), so a closure is read against the
openness range of the arm that made it and reach is measured from that arm. The object trace stays
robot-level: the simulator publishes a `grasped` flag per object, which says that an object is held
but not which arm holds it, so a hold is never attributed to an arm.

```
bazel run //ladder/components/inference/api/client:event_report -- <campaign dir> [--timeline <unit>]
bazel run //ladder/components/inference/api/client:rescore -- <campaign dir>   # after changing a definition
```
