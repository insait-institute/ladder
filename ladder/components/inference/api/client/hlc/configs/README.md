# Hierarchical (HLC) system configs

One file per hierarchical system, shared by **every** benchmark. Most are a `SubtaskController` = a
**source** (where subtasks come from) + an **advancer** (when to switch), so those files are just the
pairings, each named `<source>_<advancer>`. Nearly every source pairs with every advancer; the one
combination that does not is called out under the table. `volo.yaml` is the exception to the shape
itself: a second orchestrator with its own decision loop, described in `../README.md`.

| | switch on the gripper cycle | switch on a step clock | switch on a VLM check |
|---|---|---|---|
| **plan from the env** (privileged) | | | `oracle.yaml` |
| **one plan per episode** | `plan_gripper.yaml` | `plan_periodic.yaml` | `plan_verifier.yaml`, `plan_verifier_interval.yaml`, `plan_verifier_local.yaml`, `plan_verifier_retry.yaml`, `plan_verifier_noretry.yaml` |
| **one subtask at a time** | `stepwise_gripper.yaml` | `stepwise_interval.yaml` | none (see below) |

- **`oracle`** is the privileged upper bound: the env publishes the plan and the per-goal `achieved`
  flag, so it needs no plan file and no API.
- **`plan_*`** decompose the task once, then work through that plan.
- **`stepwise_*`** hold no plan at all: each switch generates the next subtask from the current frame.
- **`_interval`** switches on a step clock rather than the gripper: `plan_verifier_interval`
  re-verifies on a clock, `stepwise_interval` re-queries the generator.
- **`plan_periodic`** is the no-feedback control: it walks the plan on a clock set to the oracle's
  median time on a subtask, so its timing is right on average and only the absence of feedback
  separates it from `plan_gripper` and `plan_verifier`. Without it, a gap between those two and flat
  cannot be attributed to the feedback rather than to simply changing the instruction periodically.
- **`plan_verifier_noretry`** and **`plan_verifier_retry`** are `plan_verifier` with only the
  re-attempt decision changed. `_noretry` still verifies and still logs every verdict but advances
  regardless, which is what separates the value of TESTING completion from the value of REFUSING to
  advance on a negative verdict; those are confounded in every other configuration, since only a
  verifying rule produces the judgement that withholds. `_retry` re-commands the same subtask
  instead of deferring it.
- **The empty cell is not an omission.** A stepwise source cannot use `VerifiedAdvancer`:
  `next_index` asserts `decomposition is not None` ("VerifiedAdvancer schedules over a fixed plan"),
  and a stepwise source holds no plan to schedule over. It mints the next subtask on demand, so
  "is the current one done" is already the question that produced it. Writing the pairing anyway
  crashes on the first step.
- **`plan_verifier_local`** is `plan_verifier` with every VLM role answered by the run's own batch
  inference server, for runs that must not depend on a hosted API. Same system, different backend, so
  the two are directly comparable. It needs that server started with
  `api/client/batch_inference/configs/vlm_server.yaml`; the launchers pass it when
  `VLM_SERVER_CONFIG` is set.

These files are **not** systems. They are the pieces more than one system needs, so each is written once:

| | what it is |
|---|---|
| `vlm.yaml` | the VLM endpoint: one model for every role that calls one |
| `vlm_served.yaml` | the role answered by the run's own batch inference server: one copy of the model for every lane sharing it, and no network |
| `vlm_local.yaml` | the same model loaded in-process instead. This is the spec `vlm_server.yaml` hands the server, so it is what "served" serves; as a client backend it costs a copy per lane, which is why `plan_verifier_local` uses `vlm_served` |
| `subtask_generator.yaml` | the source behind both `stepwise_*` systems: generate the next subtask from the current frame, and the prompt that constrains it |

`vlm.yaml` reads the endpoint, key and model from `VLM_BASE_URL`, `VLM_API_KEY` and `VLM_MODEL`, so any
OpenAI-compatible endpoint works. `plan_verifier_skip.yaml` is `plan_verifier` with a not-done subtask deferred
to the round-robin instead of retried.

`subtask_generator.yaml` is why `stepwise_gripper` and `stepwise_interval` differ **only** in their
advancer. Everything about how subtasks are produced lives in that one file. Its prompt is limited to
the single-pick-and-place forms the subtask policy was trained on (the same forms `Decomposer` asks
for), because a motion primitive or a coordinate is out of distribution for it.

## Using one from a benchmark

Name the system; if it reads a plan, point it at yours. That is the whole per-benchmark cost.

```yaml
import: vlam_simpler_remote.yaml
runner_config:
  hlc_config:
    import: ../../../api/client/hlc/configs/plan_verifier.yaml
    source:
      cache_path: ladder/components/inference/robots/simpler/composition/decomp_cache.json
```

The path is relative to the importing config's directory, so from `models/vlams/configs/` it climbs to
`components/inference/`. These live with the classes they wire rather than in `models/vlams/configs/`,
which holds runnable eval configs only. A partial there would break `test_configs_parse`, whose
invariant is that every config in that tree is a `RunnerConfig`.

Anything these files do not set is the code default, so only what makes a system that system is
written down.

**`plan_*` ship `cache_path: ""`, which is deliberately not a usable path.** `Decomposer` is
cache-first with model fallback, so `cache_path` is the benchmark's only knob:

- set it: the tracked plans are used, deterministically;
- leave it: the VLM decomposes the reset frame instead, which is the pass that mints a plan for a
  newly added env.

There is no separate cached-vs-live config, because that is one component either way. A missing plan
falls back to the VLM rather than failing, which is safe here because each benchmark's own tests assert
its plan file covers its registry exactly, so a gap is caught in CI rather than at eval time.

## Not covered

`PhaseAdvancer` / `PrivilegedPhaseAdvancer` (per-goal reach/grasp/place/release phases) have no config
here: the privileged one needs `metadata["oracle"]["grasping"]`, which no benchmark publishes yet.

A `grasp` or `place` phase completes ONLY on its signal. Unlike `reach` and `release`, neither has a
step-budget fallback. Wiring either advancer where the signal it reads is absent therefore raises at
the first step, rather than holding one subtask to the end of the episode and scoring zero.
