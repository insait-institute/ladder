# External models

Pretrained policies published by someone else, run through ladder's own episode runners so their
numbers are produced by the same checker, recorder and metric layer as our checkpoints.

One directory per model, and each one is self-contained:

```
<model>/
  policy.py         the policy an episode runner drives; it loads the checkpoint itself
  configs/          one config per robot or simulator it is run on
  pyproject.toml    the model's pip environment, declared in pip-inference.MODULE.bazel
  requirements.txt.lock
  BUILD             the policy library, and a run_model binary in that environment
```

The model runs **in the runner's process**. A published checkpoint usually brings a dependency set
that cannot share an environment with ours -- openpi needs Python 3.12 and a JAX build, while the
rest of inference is on 3.10 -- so the model gets its own pip environment and the runner is built
into it, rather than the model being put behind a server to keep the two apart. That is one process
per environment group and nothing to start, wait for, or address.

A policy implements `api/client/policy.py`'s `Policy` (`predict_actions` plus `reset`), so a
config names it at `policy_config.autoclass` and the runners are unchanged. openvla, spatialvla
and whatever comes next follow the same shape: one directory here, one policy, one environment.

## Adding a model

1. `pyproject.toml` with the model's dependencies, and an entry in `pip-inference.MODULE.bazel`
   whose `paths` is the model's directory. Generate the lock with the package's `pip_compile`
   target, or its `uv_export` target if the environment needs `override-dependencies`: uv honours
   those only when it resolves a project.
2. `policy.py`: a `Config`/`Configurable` pair that loads the checkpoint in `__init__` and
   translates in `predict_actions` -- ladder's `RobotObservation` into whatever the weights were
   trained on, and the predicted vector back into a `RobotAction`.
3. Make the conventions of the training data explicit as config fields rather than assumptions in
   the code. Camera names, gripper sign and action space all differ between checkpoints of the same
   model, and a config that names one is a config that can be read.
4. One config per runner under `configs/`, and a `run_model` binary in the BUILD so the runner is
   built into the model's environment. Keep the configs here rather than beside another model's:
   a config resolves its `autoclass` when it is parsed, so only this environment can read them.
5. To put the model on a simulated benchmark, its config imports that benchmark's shared config
   chain and overwrites `policy_config`, so the two are scored identically. The launcher needs
   generic variables and no change: `RUN_MODEL` (this package's `run_model` target),
   `INFERENCE_SERVER=0` (nothing to serve, because the model is in the runner) and, when the model
   reads more or larger images than the default, `PUBLISH_EXTERIOR_CAMERAS` and
   `PUBLISH_IMAGE_SIZE`. Set them on the launcher command line.
