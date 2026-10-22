# Run a VLAM model

This runs a VLAM session directory, such as the released Bridge policy (see the top-level README). Its `hf_export/` folder holds the code for model inference (forward passes), independent of `ladder`, so a model exported by an older version of the code runs in a newer one.

## Inference environments (versioning)

An exported model is tied to the pip package versions (e.g. `transformers`, `torch`, `flash-attn`) it was trained with. To serve models trained before and after a dependency bump, each model records the environment it needs, and inference selects a matching pinned environment automatically.

- **You always run the same target**, `//ladder/components/inference/models/vlams:run_model` (and `:batch_inference_server`). These are version-neutral launchers (`env_resolver/target_launcher.py`): they read the env version stamped into the exported model and `os.execv` the matching per-env implementation.
- **Dispatch**: the launcher is handed every env's impl binary via `data`, plus one `--env-target <version>=$(rlocationpath ...)` arg per env (both set in this package's `BUILD`). It resolves the model's env version and execs the impl registered for it. Each env provides its impls as `envs/<tag>:run_model` / `envs/<tag>:batch_inference_server`.
- **Provenance**: an exported model may carry `hf_export/<nickname>/src/env_metadata.yaml` (the env version and the resolved package versions). A model without it falls back to `DEFAULT_ENV_VERSION`.
- **Environments**: each env lives in its own `envs/<tag>/` folder (pyproject, lock and `BUILD`) and has its own `inference-models-vlams-<tag>` declaration in `pip-inference.MODULE.bazel`. The release ships one env, `v0`. Its declaration also claims the vlams package root in its `paths`, so the shared sources resolve to its pins. `env_resolver/env_registry.py` holds `DEFAULT_ENV_VERSION` and `ENV_PACKAGES` (tag to bazel package, validated by `test_env_registry`).
- **Two libraries, one source tree**: `:vlams_adaptive` is the inference Python rooted at the shared `inference-client` env with no ML pins, so each env's binaries can recompile it under whichever env they target. `:vlams` is `:vlams_adaptive` plus the current env's ML pins.
- **Only the pins are frozen, not the code**: an `envs/<tag>` binary recompiles the live `:vlams_adaptive` sources, so any edit here ships into every environment.

## Run inference in simulation (SIMPLER)

### 1. Manual
```bash
# Bridge environments
SESSION_NAME=sess_<date>_<tag> \
./ladder/components/inference/models/vlams/scripts/vlam_simpler_bridge.sh
```

### 2. Debug mode
```bash
SESSION_NAME=sess_<date>_<tag> \
CKPT_NAMES="model-checkpoint-ema-030000.pt" \
ENV_NAMES="widowx_put_carrot_on_plate" \
./ladder/components/inference/models/vlams/scripts/vlam_simpler_bridge.sh
```

### 3. Debug with `ipdb`

WidowX
```bash
# Server terminal: WidowX
bazel run //ladder/components/inference/robots/simpler:simpler_server -- \
--config ladder/components/inference/robots/simpler/configs/widowx_put_carrot_on_plate.yaml \
--request_listener_config.req_rep_server_config.port 5556

# Client terminal
SESSION_NAME=sess_<date>_<tag> \
CKPT_NAME="model-checkpoint-ema-030000.pt" \
ENV_NAME=widowx_put_carrot_on_plate \
bazel run //ladder/components/inference/models/vlams:run_model -- \
--config ladder/components/inference/models/vlams/configs/vlam_simpler.yaml \
--runner_config.env_config.port 5556 \
--policy_config.env_name bridge \
--policy_config.controller_config.chunk_size 4
```

## Run inference in RobotArena simulation

[`scripts/vlam_robot_arena.sh`](scripts/vlam_robot_arena.sh) evaluates a VLAM
session on the canonical set of RobotArena Genesis scenes. For each
checkpoint it spawns one shared batch inference server (one model copy on
GPU) and runs `LANES_PER_GPU` `(scene, run_model)` lanes against it in
parallel; tune `LANES_PER_GPU` to your GPU's VRAM. See
[Batched inference server](#batched-inference-server) below for the
underlying architecture.

```bash
CKPT_NAMES="model-checkpoint-ema-030000.pt" \
SESSION_NAME=sess_<date>_<tag> \
SCENES="default1 scene2 scene4" \
LANES_PER_GPU=2 \
./ladder/components/inference/models/vlams/scripts/vlam_robot_arena.sh
```

Only `SESSION_NAME` is required; `CKPT_NAMES` defaults to all session
checkpoints and `SCENES` to the canonical scene set. The session must sit under
`SESSIONS_HOME` (default `/home/$USER/experiments/vlams/control`).

## Batched Inference

`VLAMInference.low_level_step(obs_list)` processes N observations in a single
`model.generate()` call instead of running them one at a time. For evaluation
workloads where multiple simulation environments produce observations in
parallel, this lifts GPU utilisation from a few percent to near peak.
`high_level_step(obs_list)` is the text counterpart: one batched
`model.generate_text()` call that returns a subtask string per observation
(requires a checkpoint with `lm_head=True`).

Observations with different instructions produce different `input_ids` lengths;
`low_level_step` pads to the max length in the batch and the model's attention
mask masks the padded positions. Output is split back into one `RobotInferenceControl`
per input observation.

### API

```python
from ladder.components.inference.models.vlams.vlam_inference import (
    VLAMInference,
    VLAMInferenceConfig,
)

inference = VLAMInference(VLAMInferenceConfig(ckpt_path=..., device="cuda", dtype="bfloat16"))

# Single observation
[control] = inference.low_level_step([obs])

# Batched
controls: List[RobotInferenceControl] = inference.low_level_step([obs1, obs2, obs3])

# High-level: one subtask string per observation
subtasks: List[str] = inference.high_level_step([obs1, obs2, obs3])
```

### Batched inference server

`low_level_step` is a single-process primitive: one Python process batches N
observations it already holds. For multi-environment evaluation where each
environment runs in its own process (e.g. a RobotArena worker per scene),
`BatchPolicyServer` provides the same batching across processes:

```
            ┌────────────────────────────┐
            │ BatchPolicyServer   │   model loaded once
            │   batched forwards         │
            └──────▲───▲───▲────▲────────┘
                   │   │   │    │          ZMQ over TCP
             ┌─────┘  ┌┘   └┐   └─────┐
       ┌─────┴──┐ ┌───┴──┐ ┌┴────┐ ┌──┴────┐
       │worker 0│ │worker 1│ │...  │ │worker N│   each worker: VLAMRobotInference
       └────────┘ └──────┘ └──────┘ └───────┘     wrapping a RemoteInferenceClient
```

The server collects requests into a batch up to `max_batch_size` or
`batch_timeout_ms`, runs each request kind (`LOW_LEVEL_INFER` /
`HIGH_LEVEL_INFER`) as its own forward, and returns each client its own
reply. Clients expose the same `low_level_step(obs_list)` /
`high_level_step(obs_list)` interface as local `VLAMInference`, so
`VLAMRobotInference` accepts either backend, selected via
`inference_config.autoclass`.

```python
# Server (one per GPU)
from ladder.components.inference.api.client.batch_inference.batch_policy_server import (
    BatchPolicyServer,
    BatchPolicyServerConfig,
)
from ladder.core.databib.config import AutoConfig
BatchPolicyServer(
    BatchPolicyServerConfig(
        policy_config=AutoConfig(
            autoclass="ladder.components.inference.models.vlams.vlam_inference.VLAMInference",
            ckpt_path=..., dtype="bfloat16", device="cuda",
        ),
        port=5560,
        max_batch_size=128,
        batch_timeout_ms=30,
    )
).run()

# Client (one per worker)
from ladder.core.databib.config import AutoConfig
from ladder.components.inference.api.client.batch_inference.remote_inference_client import (
    RemoteInferenceClient,
    RemoteInferenceClientConfig,
)
client = RemoteInferenceClient(
    RemoteInferenceClientConfig(
        client_config=AutoConfig(
            autoclass="ladder.components.inference.api.core.request_reply.ZmqRequestReplyClient",
            ip="localhost",
            port=5560,
            recv_timeout_ms=60_000,
            propagate_timeout=True,
        ),
    )
)
[control] = client.low_level_step([observation])  # transparently batched on the server
```

**Parameter tuning.** `max_batch_size` is bounded by VRAM; pick it so the
typical real-eval batch at the configured worker count fits comfortably. The
right `batch_timeout_ms` balances "GPU idle on under-filled batches" (too low)
against "queueing latency without throughput gain" (too high). The optimum is
GPU- and model-specific, so run a sweep on the target hardware before
committing.

SIGTERM/SIGINT trigger graceful shutdown after the in-flight batch finishes;
the server reports total batches and requests served before exiting.

### Hang recovery (`max_idle_ms`)

Long-running evals occasionally stall when one worker hangs (for example, a
RobotArena worker stuck in a Genesis physics/render kernel): the remaining
workers complete their partition and exit, leaving the batch server idle while
still holding the GPU. `max_idle_ms` causes the server to exit cleanly after
this many ms with no `INFER` request, once at least one batch has been served.
Default `0` disables the check; launchers that start the server for a
specific eval should pass a nonzero value (e.g. 5 min) to cap the zombie
window.
