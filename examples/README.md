# Evaluate your own system

LADDER evaluates three kinds of systems. Pick the one that matches yours, copy its example, and
replace the part marked for your model.

| your system | it decides | example | you run |
|---|---|---|---|
| a **policy** (e.g. a VLA) | the next arm pose, every step | [`policy_server.py`](policy_server.py) | `--policy-server HOST:PORT` |
| an **orchestrator** | what the suite's low-level policy should do next, in words | [`orchestrator.py`](orchestrator.py) | `--orchestrator FILE:CLASS` |
| an **agent** | everything: it drives the arm itself, with blocking moves | [`agent.py`](agent.py) | `--agent FILE:CLASS` |

Each example is one short file you can read top to bottom.

## Policy: serve it from anywhere

Your model keeps its own Python environment, GPU and machine. It answers LADDER over the network,
so it needs no Bazel and no code from this repository:

```bash
pip install pyzmq msgpack msgpack-numpy numpy           # in your model's environment
python examples/policy_server.py --port 8000            # your policy, waiting for observations

scripts/eval.sh --suite simplerenv_long --policy-server localhost:8000 --name my_policy
```

Every step, LADDER sends the observations of the environments that need an action, and your server
answers with one action per environment. An action is the absolute end-effector pose (a 4x4 matrix
in the robot base frame) and the gripper command (1 open, 0 closed). Return a chunk of `k` actions
(`"pose": [k, 4, 4]`, `"gripper": [k]`) and they are played one per step before the next request.
Episodes run in parallel: `obs["env"]` says which one an observation belongs to, and `obs["first"]`
marks the first step of a new episode.

## Orchestrator: one class, steering the low-level policy

```bash
export VLM_BASE_URL=<OpenAI-compatible endpoint> VLM_API_KEY=<key> VLM_MODEL=<model>
scripts/eval.sh --suite simplerenv_long --orchestrator examples/orchestrator.py:PlanThenSwitch
```

The low-level policy is the one the paper uses on that suite: the
[Bridge policy](https://huggingface.co/INSAIT-Institute/LADDER-Bridge-Policy) on SimplerEnv-Long and
RobotArena-&infin;-Long, and pi0.5 on RoboLab-Long. Your class is created at the start of every
episode, and `instruction(obs)` is called every step. It returns the instruction the policy should
act on now. The example plans once with a vision-language model and switches to the next subgoal
when the gripper releases what it held. Set `plan` on the object and LADDER records it with the
episode.

## Agent: one class, driving the arm

```bash
export VLM_BASE_URL=<OpenAI-compatible endpoint> VLM_API_KEY=<key> VLM_MODEL=<model>
scripts/eval.sh --suite simplerenv_long --agent examples/agent.py:PointAndPlace
```

Your class is created at the start of every episode, and `run(arm)` is called once, on its own
thread. Write it as straight-line code: `arm.move_to(xyz)`, `arm.close()` and `arm.open()` return
once the arm has done it, and the simulator waits while your program thinks. `arm.locate(camera,
x, y)` turns a pixel into a 3D point from the depth image. Agents get depth; policies and
orchestrators do not.

## What every system sees

The same observation, as a plain dictionary of numpy arrays, in all three cases:

| key | what it is |
|---|---|
| `instruction` | the task, e.g. "put the carrot and the spoon on the plate" (a policy gets the orchestrator's instruction when there is one) |
| `images` | `{camera: [H, W, 3] uint8}`: `main` on the WidowX suites, `scene` and `wrist` on RoboLab-Long |
| `eef_pose` | `[4, 4]` end-effector pose in the robot base frame, metres |
| `gripper` | 1.0 open ... 0.0 closed |
| `joints` | `[n]` joint positions, radians |
| `depth` | agents only: `{camera: {depth_m [H, W], intrinsics [3, 3], base_from_camera [4, 4]}}` |
| `robot` | `widowx` (SimplerEnv-Long, RobotArena-&infin;-Long) or `franka` (RoboLab-Long) |
| `step` | control steps since the episode started |

This is all a real robot would have. The simulator's own state (object poses, which subgoals are
done) is not given to plugged-in systems. That is what keeps the comparison fair.

## Then

- `--task all` runs every task of a suite (`TASKS.md` lists them). `--name` files the results
  under `out/<suite>/<low-level policy>/<name>_c1/`. The default name is the class name, or
  `policy_server`.
- The dashboard replays your episodes next to the paper's:
  `bazel run //ladder/components/dashboard:serve -- --results $PWD/out`.
- To put your system on the [leaderboard](https://ladderrobotics.ai/#leaderboard), see
  [`leaderboard/README.md`](../leaderboard/README.md).

## How it is wired

The adapters are in `ladder/components/inference/api/client/plugin/`: `remote_policy.py`,
`orchestrator.py` and `agent.py`. Their configs, one per suite, are in
`ladder/components/inference/models/plugin/configs/`. To change how a system is set up, such as the
arm's speed or another low-level policy, copy that config and pass it to the suite's launcher, as
EVALUATION.md describes.
