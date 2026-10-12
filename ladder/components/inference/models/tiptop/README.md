# TiPToP

TiPToP (arXiv:2603.09971) as a low-level policy: perceive the scene once, plan the whole task with
a task-and-motion planner, then execute the plan open loop. There is no VLA and no robot training
data. An episode runner drives it through the same `Policy` interface as every other model here.

## Where this lives, and why

`models/tiptop` is a peer of `models/capx`, not of `models/external/*`.

- `models/external/<name>` is a pretrained policy published by someone else. We load their
  checkpoint and run it. openpi is the example.
- `models/<name>` is a system we implement, using third-party libraries where they do real work.
  CaP-X does this with SAM-3. TiPToP does it with cuTAMP, cuRobo, M2T2 and SAM-2.

TiPToP is a method, not a checkpoint, so it belongs in the second group. The pipeline in this
package is ours; the four vendored libraries are pinned in `http-assets.MODULE.bazel` and used
unmodified.

`api/client/hlc` is for something different again: high-level controllers such as VoLo that sit on
top of a VLA and emit subtask instructions. They need no heavy pip dependencies and they do not
produce actions. TiPToP produces actions, so it is a policy and not an hlc controller.

## Pipeline

One plan request runs, in order:

1. Ground the instruction. One VLM call returns object bounding boxes and the goal as predicates
   (`on(movable, surface)`, `holding(movable)`, `near(movable, reference)`). The VLM is the repo's
   own `hlc/vlm` backend, so it inherits the rate limiting and the failure budget. The shipped
   configs use Gemini 3.7 Flash through `VLM_BASE_URL`.
2. Segment. SAM-2 turns each box into a mask.
3. Lift to 3D. Metric depth plus the camera intrinsics and its pose in the robot base frame give a
   structured point cloud, voxel downsampled for the grasp model.
4. Propose grasps. M2T2 runs on the downsampled cloud; grasps are re-associated to objects by
   contact-point proximity, then optionally filtered by approach tilt, by whether they close on
   material, and by whether that material fits between the jaws.
5. Reconstruct. RANSAC picks the plane that the most detected objects rest on, and each object
   becomes the convex hull of its masked points.
6. Plan. cuTAMP optimizes over particles with cuRobo doing IK and motion generation, and returns a
   sequence of joint trajectories and gripper actions.
7. Execute. `PlanCursor` steps the serialized plan out at the control rate.

The policy reads only what a real robot measures: the calibrated camera's colour and depth, the arm
joints and the instruction. The simulators put ground-truth object poses, the ordered subtask list
and a per-pixel robot segmentation in the same metadata, and none of it is read.

## Files

| file | what it holds |
| --- | --- |
| `config.py` | every knob, in one `Config` |
| `policy.py` | the `Policy` the runner drives: per-env lanes, replanning, episode-log events |
| `planner.py` | one planning stack per process; the request pipeline above |
| `perception.py` | grounding, SAM-2, depth to point cloud, M2T2 |
| `scene.py` | table fitting, object hulls, grasp association and filtering |
| `tamp.py` | cuRobo solvers, the cuTAMP problem, planning, plan serialization |
| `plan.py` | `PlanCursor`, which turns a serialized plan into per-step actions |

## Settings worth knowing

Each of these was tuned against saved scenes and should not be changed casually.

- `table_top_offset_m = -0.01` with `object_z_filter_tolerance_m = -0.01`. Object points are kept
  above the slab top plus the tolerance, as in the reference, but the values are ours. The
  reference's stricter filter removes most of a 2 cm object resting on the surface, leaving too few
  points to keep it. The slab is sunk because the optimizer inflates collision spheres by
  `coll_sphere_radius`, so a flush top reports every resting object as already in collision.
- `table_thickness_m = 0.05`. Depth cannot observe thickness, and a thin slab lets the planner route
  the arm underneath the table.
- `pos_err_tolerance_m = 0.015` on the WidowX. A 6-DOF arm has no spherical wrist and generically
  cannot reach an arbitrary 6-DOF pose, so cuTAMP's default 5 mm IK tolerance rejects every
  particle. The motion generator's convergence threshold is raised to match, or it rejects the
  poses the optimizer accepted.
- `m2t2_apply_bounds = False`. M2T2's demo bounds clip the cloud to the cell it was recorded in,
  which removes the goal object entirely in many RoboLab scenes.
- `grasp_depth_offset_m = 0.0`, the reference's own default. The knob slides the grasp frame further
  down its approach axis, past the 0.1034 m that takes an ACRONYM pose to the point between the
  pads. A positive offset pushes the closing point down to the table.
- `num_particles = 256`, `opt_steps = 500`, `max_planning_time_s = 60`. The reference
  implementation's search budget. A smaller budget is a false economy: cuTAMP then reports "No
  satisfying particles" on the harder scenes.
- `time_dilation_factor = 0.6`. The reference calls this the robot's speed ratio and advises against
  going higher. It changes execution only where waypoints are strided by the trajectory's own
  timestep, which is RoboLab; the WidowX suites resample by distance instead.
- `waypoint_spacing` is set for the WidowX suites and not for RoboLab, which matters when comparing
  their scores. RoboLab strides the joint trajectory by its own timestep, as the reference intends.
  SimplerEnv and RobotArena accept only end-effector pose actions, so there the planned path is
  resampled every 5 cm and re-tracked through IK.
- `vlm_reasoning_effort = "low"`. The reference grounds with thinking switched off. Some endpoints
  reject that setting, and "low" is the closest accepted match; it keeps grounding quality and cuts
  latency. Grounding runs once per planning request, so with `goals_per_plan` set this is paid per
  goal.
- A lane that fails to plan holds for the rest of the episode rather than trying again. Otherwise
  the goals it has not reached would trigger a replan on every control step, each a fresh grounding
  call and a full cuTAMP search, with the whole batch waiting on it.
- `experiment_root` is per process. cuTAMP names each run's directory after the second it started
  in and raises when a file in it already exists, so two lanes starting a plan in the same second
  would collide.
- `goals_per_plan`. `null` plans the whole task as one skeleton, which cuTAMP rarely solves beyond a
  few goals. Setting it to 1 plans one goal per request and replans the rest from a fresh
  observation, which is the re-planning after each pick-and-place the paper suggests.
- The request is seeded from the task, the joint state and the depth. cuTAMP samples grasps with
  `torch.randint` and has no seed of its own, so without this the same scene draws different
  candidates on every run.

## Running it

The benchmark configs under `configs/<bench>/composition/` import the shared vlams chain and
overwrite `policy_config`, so the scoring is identical to every other model on that suite. The
launcher needs `RUN_MODEL=models/tiptop:run_model`, depth publishing on, and no VLA server.
The WidowX launchers also need any `SESSION_NAME` (it only labels the output).

The M2T2 and SAM-2 checkpoints are fetched by bazel (`http_file` in `http-assets.MODULE.bazel`)
and reach the process through the runfiles, so a run needs no network and the bytes are
integrity-checked. The first boot on a machine still compiles cuRobo's fused operators and M2T2's
pointnet2 extension against the staged `cuda_home`. Budget tens of minutes for it; later boots
reuse the extensions.

That compile happens in the package's `__init__.py`, before any module in the package is imported,
because both libraries build their extensions at import time and need the toolchain and the cache
directory configured first. Doing it there is what lets every module import them at the top of the
file.

The extension directory is keyed by the card's compute capability. torch names an extension's
build directory after the extension alone, so one cache shared by two GPU models has each
architecture overwriting the other's object and rebuilding on every alternation.

That build is serialized on a lock file in the extension directory, and any lock torch left behind
is removed while we hold it. Torch guards the build with a file it creates and deletes itself, so a
process killed mid-compile leaves the file there for good and every later process waits on it
forever, with the card allocated and idle. Our lock is an OS lock the kernel drops when the process
dies, so a kill cannot poison the cache.
