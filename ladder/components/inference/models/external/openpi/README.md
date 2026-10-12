# openpi (pi0.5)

Physical Intelligence's pi0.5 DROID checkpoints, run through ladder's episode runners so their
numbers come from the same checker, recorder and metric layer as our own.

`policy.py` is the whole integration:
`OpenpiModel` loads a checkpoint from the vendored openpi source, `OpenpiPolicy` turns its output
into robot actions and holds the chunk each environment is partway through. The model loads in the
runner's process, so there is nothing to start or address. Environments needing a new chunk in the
same call infer together, in one batched forward.

## Which checkpoint

openpi publishes two DROID checkpoints. They are the same model; only the arm dimensions differ,
and `action_space` has to agree with the one you load.

| `model_id` | arm dimensions | for |
| --- | --- | --- |
| `pi05_droid` | joint velocities | a velocity-controlled arm, the real FR3 |
| `pi05_droid_jointpos` | absolute joint targets | a position-controlled arm, the simulators |

`checkpoint_dir` is derived from `model_id` when empty. Set it to a local directory to run without
reaching `gs://`.

A checkpoint trained on another dataset is config, not code: point `exterior_key`, `wrist_key`,
`joint_key`, `gripper_key`, `image_resolution` and `arm_joints` at that dataset's convention.

## Running

RoboLab composition, through the launcher that brings up the simulator per lane. The two
variables are the launcher's generic interface for a policy that is not one of our checkpoints:
run from this package's `run_model`, and serve nothing.

```bash
RUN_MODEL=models/external/openpi:run_model INFERENCE_SERVER=0 TASKS="comp_bin_2" \
LADDER_CONFIG=ladder/components/inference/models/external/openpi/configs/robolab/composition/openpi_robolab_composition.yaml \
bash ladder/components/inference/robots/robolab/scripts/vlam_robolab.sh
```

`configs/robolab/composition/` holds one file per hierarchical system. Each imports the arm the
vlams checkpoints run and overwrites `policy_config`, so both are scored by the same checker,
recorder and metric layer, and the numbers are comparable.

## Caches

Two, and on a multi-node setup both belong on shared storage.

- The checkpoint, about 12 GB, under `OPENPI_DATA_HOME` (default `~/.cache/openpi`). If you copy
  one in by hand, do not preserve timestamps: openpi deletes a checkpoint directory whose mtime
  predates 2025-02-03.
- XLA's compiled kernels, under `model_config.jax_compilation_cache_dir` (default `~/.cache/jax`).
  Each distinct batch shape compiles once; without a hit that costs minutes per process.

## Environment

Python 3.12 with a JAX CUDA build, separate from the 3.10 inference environment. `run_model`'s
`env` attribute carries what the GPU needs, so there is nothing to set at the call site:

- `LD_LIBRARY_PATH` over the venv's `nvidia/*/lib` directories, because jax looks for its CUDA
  libraries beside the plugin wheel and a venv of one repository per wheel does not put them there.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`, so jax grows on demand instead of claiming a share of the
  card up front. A simulator lane shares the card with the model, and a fraction that fits one card
  starves either the model or the simulator on the next one.

An interactive session (`bazel run ...:ipython3`) carries no such attribute, so set
`LD_LIBRARY_PATH` yourself when you need the GPU there.
