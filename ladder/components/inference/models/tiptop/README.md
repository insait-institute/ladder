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

These carry measured reasons and should not be changed casually.

- `table_top_offset_m = -0.01` with `object_z_filter_tolerance_m = -0.01`. Object points are kept
  above the slab top plus the tolerance, which is how the reference measures them, but the values
  are ours and the reference's do not transfer. Its own are a 2 cm sink and a +1 cm tolerance,
  putting the filter 1 cm below the detected surface; replayed over 30 saved scenes that plans
  12/30 against 16/30 for the values here, and the loss is concentrated at one goal, 5/9 against
  8/9. The reference works on real-world objects, and a filter that strict removes most of a 2.3 cm
  object resting on the surface, after which it has too few points to keep at all.
  Sinking the slab matters because objects rest on the surface and the optimizer inflates their
  collision spheres by `coll_sphere_radius`, so a flush top reports every object as already in
  collision. Replaying 22 saved observations: flush plans 13, sunk plans 19.
- `table_thickness_m = 0.05`. Depth cannot observe thickness, and a thin slab lets the planner route
  the arm underneath the table.
- `pos_err_tolerance_m = 0.015` on the WidowX. A 6-DOF arm has no spherical wrist and generically
  cannot reach an arbitrary 6-DOF pose, so cuTAMP's default 5 mm IK tolerance rejects every
  particle. The motion generator's convergence threshold is raised to match, or it rejects the
  poses the optimizer accepted.
- `m2t2_apply_bounds = False`. M2T2's demo bounds clip the cloud to the cell it was recorded in,
  which removed the goal object entirely in about a third of RoboLab scenes.
- `grasp_depth_offset_m = 0.0`, which is the reference's own default. The knob slides the grasp
  frame further down its approach axis, past the 0.1034 m that takes an ACRONYM pose to the point
  between the pads. It was added here when the commanded grasp appeared to sit near the top of the
  object, but that measurement was taken on object clouds that reached a centimetre or two into the
  table, so the object looked taller than it was. Swept over 30 saved scenes against one shared
  perception pass, 1.5 cm costs 8 plans in 30 and puts the closing point at 0.04 of object height,
  i.e. on the table; 0 plans 21/30 and closes at 0.46, the middle of the object.
- `num_particles = 256`, `opt_steps = 500`, `max_planning_time_s = 60`. The reference
  implementation's search budget, as declared by each of its entry points. A smaller budget was
  tried and is a false economy: at 64 particles over 100 steps cuTAMP reports "No satisfying
  particles" on the harder scenes, and the single-goal units fell to 1/24 with the arm never
  grasping in every failure.
- `time_dilation_factor = 0.6`. The reference calls this the robot's speed ratio, ships 0.2, runs
  0.6 on its own hardware and advises against going higher. The default here was 1.0, faster than
  the reference ever runs. It changes execution only where waypoints are strided by the
  trajectory's own timestep, which is RoboLab; the WidowX suites resample by distance instead.
- `waypoint_spacing` is set for the WidowX suites and not for RoboLab, and the difference is worth
  knowing when reading their scores against each other. The reference hands its whole joint
  trajectory to a servo. RoboLab keeps that shape, striding the trajectory by its own timestep. The
  WidowX suites instead resample the path every 5 cm, which is coarse next to a 2.3 cm object, and
  they cannot use joint control at all: both SimplerEnv and RobotArena reject anything but an
  end-effector pose action, so the planned joint trajectory is re-tracked through IK there. TiPToP's
  open-loop plan is therefore executed as the reference intends only on RoboLab.

- `vlm_reasoning_effort = "low"`. The reference grounds with thinking switched off entirely. The
  endpoint we used rejects the equivalent setting here, answering 400 "Thinking level is unsupported" and
  returning an empty reply, so the lowest level it accepts is the closest match. Measured over 25
  saved scenes it costs nothing and saves a third of the latency: 81 boxes and 44 goal atoms in
  17.5 s per call against 77 and 43 in 26.7 s. Grounding runs once per planning request, so with
  `goals_per_plan` set this is paid per goal.

- A lane that fails to plan holds for the rest of the episode rather than trying again. The goals a
  decomposed request has not reached yet are what triggers the next replan, so carrying them past a
  failure made the lane replan on every control step, each one a fresh grounding call and a full
  cuTAMP search, with the whole batch waiting on it.

- `experiment_root` is per process. cuTAMP names each run's directory after the second it started
  in, under one shared root, and raises when a file in it already exists. Lanes are separate
  processes on one node, so any two starting a plan in the same second collided and the second lost
  its plan to "File ... already exists", which the episode then scored as the arm never grasping.

- `goals_per_plan`. `null` plans the whole task as one skeleton. cuTAMP cannot do that much past
  four goals: the no-plan rate on SimplerEnv is 8% at one goal, 40% at two and 100% at five or
  more. Setting it to 1 plans one goal per request and replans the rest from a fresh observation,
  which is the re-planning after each pick-and-place the paper suggests.
- The request is seeded from the task, the joint state and the depth. cuTAMP samples grasps with
  `torch.randint` and has no seed of its own, so without this the same scene draws different
  candidates on every run and arm-vs-arm comparisons fight the sampler.

## Running it

The benchmark configs under `configs/<bench>/composition/` import the shared vlams chain and
overwrite `policy_config`, so the scoring is identical to every other model on that suite. The
launcher needs `RUN_MODEL=models/tiptop:run_model`, depth publishing on, and no VLA server.
The WidowX launchers also need any `SESSION_NAME` (it only labels the output).

The M2T2 and SAM-2 checkpoints are fetched by bazel (`http_file` in `http-assets.MODULE.bazel`)
and reach the process through the runfiles, so a run needs no network and the bytes are
integrity-checked. The first boot on a node still compiles cuRobo's fused operators and M2T2's
pointnet2 extension against the staged `cuda_home`. Budget tens of minutes for it; later boots on
the same node reuse the extensions.

That compile happens in the package's `__init__.py`, before any module in the package is imported,
because both libraries build their extensions at import time and need the toolchain and the cache
directory configured first. Doing it there is what lets every module import them at the top of the
file.

The extension directory is keyed by the card's compute capability. torch names an extension's
build directory after the extension alone, so one cache shared by an A100 and an A6000 has each
architecture overwriting the other's object and rebuilding on every alternation.

That build is serialized on a lock file in the extension directory, and any lock torch left behind
is removed while we hold it. Torch guards the build with a file it creates and deletes itself, so a
process killed mid-compile leaves the file there for good and every later process waits on it
forever, with the card allocated and idle. Our lock is an OS lock the kernel drops when the process
dies, so a kill cannot poison the cache.
