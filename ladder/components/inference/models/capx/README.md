# CaP-X as a policy

[CaP-X](https://github.com/capgym/cap-x) (arXiv:2603.22435) for the three composition benchmarks.
This is the paper's Tier M3 agent: high-level primitives, multi-turn interaction,
visual differencing in text. A VLM writes a Python program against the robot API, the program drives the arm, and a second VLM
pass describes the before/after frames and decides to regenerate or finish. No learned policy runs.

Upstream's `control.py` defines the six primitives the program is written against:
`get_object_pose`, `sample_grasp_pose`, `goto_pose`, `open_gripper`, `close_gripper`,
`home_pose`. Upstream's harder tiers drop `get_object_pose` and make the model do
segmentation and IK itself; only the six-primitive tier ships here, because it is
also the only tier the WidowX suites can run (their sims accept end-effector
targets only).

## Layout

| path | what it is |
| --- | --- |
| `policy.py` | `CapxPolicy`: prompts, code extraction, the multi-turn decision, one program thread per env |
| `api/tiered.py` | the six primitives of upstream's `control.py`, and the motion constants per embodiment |
| `perception/camera.py` | depth/intrinsics frames from the simulators, plus the quaternion helpers |
| `perception/scene.py` | the `Scene` seam: camera geometry as a pose |
| `perception/rgbd.py` | ground the name, segment it, unproject through the depth calibration, fit a box |
| `perception/sam3.py` | SAM3, upstream's primary segmenter: the name is the prompt and the masks come back |

## How an object becomes a pose

`get_object_pose` grounds the name with SAM3, segments that detection to an instance mask,
unprojects the masked pixels through the depth calibration, and fits an oriented box. Only the
object's own pixels reach the point cloud. Every config reads the one scene camera.

Nothing under `api/` or `perception/` reads simulator state, so the same primitives drive real
hardware.

## Running it

| config | suite |
| --- | --- |
| `configs/robolab/composition/capx_robolab_composition_tiered.yaml` | RoboLab (Franka) |
| `configs/robot_arena/composition/capx_robot_arena_composition_tiered.yaml` | RobotArena (WidowX) |
| `configs/simpler/composition/capx_simpler_composition_tiered.yaml` | SimplerEnv (WidowX) |

The policy brings its own pip environment, so it has its own runner: `RUN_MODEL=models/capx:run_model`.
No checkpoint is served. RoboLab takes `INFERENCE_SERVER=0`; RobotArena and SimplerEnv take
`NO_VLA=1` and any `SESSION_NAME` (it only labels the output). Depth is off by default, so all three also need `PUBLISH_DEPTH=True`. SAM 3's weights are gated on Hugging Face: accept the license at
[`facebook/sam3`](https://huggingface.co/facebook/sam3) and export `HF_TOKEN` first.

## Deviations from upstream

- **`sample_grasp_pose` is analytic.** Upstream samples 6-DoF grasps with Contact-GraspNet;
  here the grasp is the fitted box's center from straight above. The sampler's contacts sit
  high on small objects at a tilt these arms cannot hold, so it drops them.
- **RoboLab moves in small steps.** Upstream commands each waypoint at once; the Franka here
  holds objects by bare contact and loses them on fast moves, so the arm ramps to the target
  and the workspace is clamped to the arm's reach.
- **The box is closed at the support plane.** Upstream fits the visible points alone, which
  puts the center of a flat object in the middle of its top shell instead of the middle of
  the object.
- **`tool_offset_m` is per embodiment.** Each simulator reports the end effector in its own
  frame; the offset converts the grasp pose into the commanded one.
- Step counts are upstream's 20 Hz values converted to each suite's control rate.
