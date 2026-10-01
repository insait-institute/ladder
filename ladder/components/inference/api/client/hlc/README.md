# High-Level Controller (HLC)

Hierarchical inference: the HLC emits language commands for a low-level VLA policy from the
current observation, instead of a fixed task instruction for the whole episode. There are TWO systems behind the contract in `base.py`: `SubtaskController` (`controller.py`),
described below, and `VoloOrchestrator` (`volo.py`), a vendored third-party orchestrator.

`SubtaskController` is composed of two independently-swappable, YAML-selected pieces:

- a **source** (`sources/`): WHERE subtasks come from — a fixed per-scene plan (a decomposition)
  or on-the-fly generation from a model. Sources are interchangeable behind `SubtaskSource`
  (`base.py`).
- an **advancer** (`advancers/`): WHEN to switch to the next subtask — a step clock, gripper
  state, VLM verification, or privileged env signals. Each step the advancer returns the subtask
  *index* to command; moving the index makes a plan source emit the next plan entry and a
  generative source produce a fresh subtask, so **every advancer composes with every source**.

Everything keys per-env state by `env_idx` (correct under K>1 batched envs) and reads only
`images` / `gripper_state` / `eef_pose` / `metadata` off the observation — the same code serves
SimplerEnv, RobotArena and a real rig. Control decisions are never inferred from instruction or
subtask wording: order semantics, phase structure and switch triggers are all declared config or
plan data.

## Sources

- `Decomposer` (`sources/decomposers.py`) — cache-first plan (`cache_path`: per-scene JSON of
  `[subtask, ...]` lists, or `{"subtasks": [...], "ordered": true}` for order-constrained scenes),
  falling back to one VLM call on the reset frame when `vlm` is set.
- `PrivilegedDecomposer` — the ground-truth plan the env publishes in
  `metadata["oracle"]["subtasks"]`.
- `PrivilegedPhaseDecomposer` — expands each ground-truth goal into per-phase subtasks via
  `phase_templates` (e.g. grasp/place).
- `VLMSource` (`sources/vlm_source.py`) — on-the-fly generation from any `VLM` backend, with a
  rolling per-env (image, command) history; misses fan out concurrently.
- `VLAMSubtaskSource` (`models/vlams/vlam_subtask_source.py`) — on-the-fly generation from a
  trained VLAM checkpoint's own VLM backbone, batched over the shared inference server (self-HLC
  via `bind_policy`, or a separate checkpoint via `inference_config`). Lives under `models/vlams`
  because that package depends on `api/client`, not the reverse; it is a dedicated source (its
  trained subtask prompt, not generic chat).

## Advancers

- `PeriodicAdvancer` (`advancers/periodic.py`) — every `interval` steps, optionally also the step
  the gripper toggles (`on_gripper_toggle`). Under a generative source this is the classic
  re-query clock; under a plan it walks the list on a timer.
- `GripperAdvancer` (`advancers/gripper.py`) — a sustained grasp that LIFTS (`lift_threshold_m`)
  then releases = one pick-and-place done; `min_hold_steps`, `advance_cooldown_steps` (debounce),
  `cyclic_repair`. Proprioception only.
- `PhaseAdvancer` (`advancers/phase.py`) — advances a phase-per-object plan; each subtask's phase
  is its POSITION in the configured `phase_cycle` (mirroring the decomposer's templates).
  Grasp/place switch on gripper edges, reach/release on step budgets.
- `PrivilegedAdvancer` (`advancers/privileged.py`) — the env's `metadata["oracle"]["achieved"]`,
  debounced and decoupled from the release instant; retargets the first unachieved goal.
- `PrivilegedPhaseAdvancer` (`advancers/phase.py`) — PhaseAdvancer's decision driven by the env's
  ground-truth per-goal `grasping`/`achieved` signals instead of the gripper. No benchmark publishes
  `grasping` yet, so wiring it raises until one does.
- `VerifiedAdvancer` (`advancers/verified.py`) — skip-and-return round-robin with OBSERVABLE
  completion verification: the gripper segments attempts, an injected `Verifier` (`verifiers.py`)
  marks subtasks done, with revoke hysteresis and optional provisional gripper credit. An ordered
  plan (`Decomposition.ordered`) runs strictly sequential. Requires a plan source.

An advancer that switches on an env signal **raises when that signal is absent** rather than reading
absence as a negative — no `achieved` would make every goal look complete to `PrivilegedAdvancer`, no
gripper reading makes a release unobservable to `GripperAdvancer`, and `grasp`/`place` phases have no
step-budget fallback at all. Each would otherwise hold one subtask to the end of the episode and
score zero without erroring. `PeriodicAdvancer` and `VerifiedAdvancer` are exempt: a step clock and
`attempt_stall_steps` respectively force progress whatever the env publishes.

## Live plans (`object_attributes`)

With no cache entry for the scene the `Decomposer` asks the VLM once per (scene, task), or every
episode with `replan_each_episode`, for the plan as `{"subtasks": [...], "ordered": bool}`, so a live
plan can declare a sequence the way the cached plans do; a reply that is not JSON is read one
subtask per line, unordered. `object_attributes` names objects as colour plus type ("the yellow
lemon"): under a bare noun pi0.5
grasps a look-alike in a third of its grasps, and the completion check is asked about the same
string the policy is given. On the 42-unit RoboLab composition suite with pi0.5, the live colour-named
arm scores 26.8 against 19.0 for the cached-plan arm and 23.2 for VoLo, paired per unit over three
draws (24 episodes per unit).

## Not spending calls where they cannot count

An env whose episode has ended (`done` or `truncated` in the metadata) gets no decision and no VLM
call, which in a 12-env lockstep batch is most of the verifier calls near the end. `VerifiedAdvancer.attempt_stall_jitter`
spreads each env's stall clock so a batch does not fire every check in the same second, which is what
tripped the endpoint's burst limit. Every backend (`VLMConfig.max_failure_rate` and its companions)
turns a transient failure into an empty reply, so a rate-limited burst costs a verdict or a plan
rather than the lane, and raises `VLMUnavailableError` once more than the ceiling of calls is lost, so a
run that was mostly unanswered is never scored.

## Several objects of one kind

A plan carries one line per goal, so two carrots give the same sentence twice, and the verifier is
asked the identical question about both — true for each as soon as one carrot is placed.

The completion question is therefore a COUNT, and there is one of it for every subtask everywhere.
The model reports how many instances are in the end state, `VerifiedAdvancer` passes `required` (how
many times that line appears in the plan up to this one), and the verdict is `count >= required`.
The prompt names no object, and `required` counts occurrences of the line rather than parsing it, so
this holds for any object anywhere.

It fixes the judgement, not the control: the policy gets the same undifferentiated sentence and will
re-grasp what it already placed.

## VLM backends (`vlm/`)

Every VLM-consuming role — decomposition (`Decomposer.vlm`), verification (`Verifier.vlm`),
generation (`VLMSource.vlm`) — takes the same `VLM` interface (`vlm/base.py`) and works with any
backend. Messages are OpenAI-style dicts whose image items carry raw numpy arrays; each backend
does its own encoding.

- `OpenAIVLM` (`vlm/openai_api.py`) — any OpenAI-compatible endpoint (OpenAI, Gemini via a compat
  gateway, a self-served vLLM). Needs `VLM_BASE_URL` and `VLM_API_KEY`; every call is cost-logged via `cost_track.py`
  (`COST_TRACK_PATH` overrides the JSONL path, `ALIAS` tags rows).
- `LocalVLM` (`vlm/local_hf.py`) — a HuggingFace-format checkpoint run on-device
  (lazy-loaded; chat-template and raw-prompt models both handled). It lives with the
  torch/transformers deps — this package stays torch-free; backends resolve via `autoclass`
  from wherever their dependencies are.
- `ServedVLM` (`vlm/served.py`) — the model hosted by the run's batch inference server, so the
  workers sharing a GPU share one copy of it. Takes no connection settings: the controller hands
  it the policy's client via `bind_policy`.

## Choosing the granularity

Granularity is declared entirely in config — the plan's structure and the switching signal are
chosen together, never inferred from the subtask wording:

| Subtask granularity | Source | Advancer | Switch signal |
|---|---|---|---|
| one goal = one subtask ("put the X on the Y") | cache / VLM / oracle plan | `GripperAdvancer` | release after a held (lifted) grasp |
| 2-phase ("grasp the X" / "move the X to the Y") | `phase_templates` with 2 entries | `PhaseAdvancer`, `phase_cycle: [grasp, place]` | sustained close / release-after-hold |
| 4-phase (+ reach / release) | `phase_templates` with 4 entries | `PhaseAdvancer`, `phase_cycle: [reach, grasp, place, release]` | close-begin + budgets around the gripper edges |
| one goal = one subtask, VLM-verified | cache (`ordered` per scene) | `VerifiedAdvancer` | attempt end -> verifier verdict |
| on-the-fly (no fixed plan) | `VLMSource` / `VLAMSubtaskSource` | `PeriodicAdvancer` (± `on_gripper_toggle`) or `GripperAdvancer` | clock / gripper events |

`phase_cycle` assigns one of the four completion signals to each POSITION of the repeating cycle,
so a "move to" subtask can complete on a release (`place`) or on a step budget (`reach`) — whatever
the experiment declares. The cycle must mirror the decomposer's templates; goal i owns positions
[i*P, (i+1)*P).

## Usage

```yaml
runner_config:
  hlc_config:
    autoclass: ladder.components.inference.api.client.hlc.controller.SubtaskController
    source:
      autoclass: ladder.components.inference.api.client.hlc.sources.decomposers.Decomposer
      cache_path: /path/to/decomp_cache.json
    advancer:
      autoclass: ladder.components.inference.api.client.hlc.advancers.gripper.GripperAdvancer
      min_hold_steps: 3
      advance_cooldown_steps: 8
```

The privileged oracle is the SAME class with `PrivilegedDecomposer` + `PrivilegedAdvancer`; the
self-HLC is `VLAMSubtaskSource` + `PeriodicAdvancer(on_gripper_toggle: true)`. See
`ladder/components/inference/models/vlams/configs/` for full examples.

## VoLo (`volo.py`)

[VoLo](https://github.com/NVlabs/VoLoAgent) as a second orchestrator, for comparison against
`SubtaskController` on the same scenes, policy and success checker. Upstream is vendored at a pinned
commit (`volo_agent_repo`) and its strategies run unmodified; `volo.py` carries ladder observations
into the dict they read and publishes what they decided onto the same event channel.

Three modes, all VLA-only — each changes the instruction and nothing else:

| `mode` | behaviour |
|---|---|
| `subgoal` | decompose the task upfront, advance when a VLM check says the step is done |
| `next_goal` | no upfront plan; the VLM predicts the next instruction every `check_interval` |
| `passthrough` | never edits the instruction — the null control, and it spends no VLM calls |

Upstream's `tool_chain` mode is not wired: it manipulates via GraspGen off ground-truth simulator
state, so it is a different system rather than a different orchestrator.

`failure_monitor` / `recovery_mode` apply to `subgoal` only; `__post_init__` rejects them elsewhere
rather than ignoring them. `recovery_mode` is restricted to `replan` (or `null` alongside
`failure_monitor: null`) — upstream's other values drive its grasp/place tools, which need a
GraspGen server this integration does not run and which fail as a swallowed warning without one. `check_interval` must fit inside the benchmark's episode budget — an
interval longer than the episode means the strategy decides once at step 0 and never again, which
still plans, still logs and still scores.

One eval config per benchmark — `models/vlams/configs/hlc/vlam_{simpler,robot_arena,robolab}_volo.yaml`
— imports `configs/volo.yaml` and carries that benchmark's camera, interval and `stop_on_success`
(which must track the runner's `terminate_on_success`: RobotArena runs full episodes, the other
two stop at success).

Mode is a run-time override rather than a config per combination — `ConfigParser` takes dotted
overrides, and all three launchers (`vlam_simpler_bridge.sh`, `vlam_robot_arena.sh`,
`robots/robolab/scripts/vlam_robolab.sh`) forward `EXTRA_RUN_ARGS` verbatim:

```
SESSION_NAME=sess_… \
SIMPLER_CONFIG=ladder/components/inference/models/vlams/configs/hlc/vlam_simpler_volo.yaml \
EXTRA_RUN_ARGS="--runner_config.hlc_config.mode=passthrough" \
./ladder/components/inference/models/vlams/scripts/vlam_simpler_bridge.sh
```

Pair a mode override with a distinct `SESSION_NAME`; `output_path` does not encode the mode, so
two modes under one session overwrite each other. `subgoal` and `next_goal` need `VLM_BASE_URL` and `VLM_API_KEY` —
VoLo issues its own requests to the endpoint in `configs/vlm.yaml`; `passthrough` runs without it.

## Episode events

Advancers may record per-env events into their `pending_events` channel (e.g. every verifier
call, with its verdict, raw reply and the simulator's ground-truth flag at call time). The episode
runner drains `SubtaskController.pop_events()` each step into the step metadata, where the
recorder persists them.

## Adding a piece

- A new source: subclass `SubtaskSource[YourConfig]`; implement `decompose(obs, task)` for a fixed
  plan (index lookup is inherited) or `commands(...)` for on-the-fly generation.
- A new advancer: subclass `SubtaskAdvancer[YourConfig]`, implement
  `next_index(obs, env_idx, idx, decomposition)`.
- A new VLM backend: subclass `VLM[YourConfig]`, implement `complete(messages, ...)` — it becomes
  usable for decomposition, verification and generation at once.
- A new benchmark: publish `metadata["oracle"]` per env, with `subtasks` (the ground-truth plan,
  one instruction per goal in the order they should be done) and `achieved` (per-goal done flags,
  current). That pair is the entire contract `PrivilegedDecomposer` + `PrivilegedAdvancer` read, so
  an env emitting it runs the oracle system with no new code. SimplerEnv
  (`robots/simpler/composition/envs.py`), RobotArena (`robots/robot_arena/composition/tasks.py`) and
  RoboLab each derive it from their own success checker — that contract is all the benchmarks share,
  since the simulators themselves have nothing in common.

Reference it via `autoclass` in the YAML — no change to the controller wiring.
