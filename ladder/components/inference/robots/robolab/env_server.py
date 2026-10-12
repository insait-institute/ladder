import argparse
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
# @robolab_repo ships no assets; point its constants.py at the runfiles-staged tree before any import.
os.environ.setdefault(
    "ROBOLAB_ASSET_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets"),
)

import cv2  # must be imported before isaaclab. Do not move
from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve RoboLab envs over ladder's RobotServer protocol.")
    parser.add_argument("--port", type=int, required=True, help="ZMQ request-reply port.")
    parser.add_argument(
        "--expose-privileged",
        action="store_true",
        help="Publish ground-truth composition geometry for the scripted completability policy.",
    )
    parser.add_argument(
        "--publish-depth",
        action="store_true",
        help="Render depth and publish it with the camera calibration, for a camera-based policy.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Render no camera, for a policy that reads no image (the scripted one). Physics is unchanged.",
    )
    parser.add_argument(
        "--exterior-cameras",
        type=int,
        choices=(1, 2),
        default=1,
        help="Exterior views to render and publish; 2 adds the over-shoulder right camera.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(224, 224),
        help="Size of the published camera images; set it to what the policy was trained on.",
    )
    return parser


# AppLauncher must launch the sim app before the sim-dependent (# noqa: E402) imports below.
parser = _build_parser()

from robolab.eval.runner import add_common_eval_args  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb.settings  # noqa: E402
import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.timeline  # noqa: E402
import robolab.constants  # noqa: E402
import torch  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.metrics.compute_metrics import compute_episode_metrics  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402
from robolab.registrations.droid.camera_presets import WRIST_LEFT, WRIST_LEFT_RIGHT  # noqa: E402
from robolab.variations.camera import with_depth  # noqa: E402

from ladder.components.inference.api.core.action import RobotAction  # noqa: E402
from ladder.components.inference.api.core.observation import METADATA_CAMERAS_KEY, encode_camera  # noqa: E402
from ladder.components.inference.api.core.privileged_state import METADATA_PRIVILEGED_KEY  # noqa: E402
from ladder.components.inference.api.core.request_reply import (  # noqa: E402
    ZmqRequestReplyServer,
    ZmqRequestReplyServerConfig,
)
from ladder.components.inference.api.core.requests import RequestType  # noqa: E402
from ladder.components.inference.robots.robolab.composition.specs import (  # noqa: E402
    SPECS,
    decomp_cache_entries,
    task_class_name,
    task_target,
)
from ladder.components.inference.robots.robolab.droid_conversions import DroidJointposConverter  # noqa: E402
from ladder.components.inference.robots.robolab.oracle import build_oracle  # noqa: E402
from ladder.components.inference.robots.robolab.privileged import privileged_state  # noqa: E402
from ladder.core.common.logger import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

# MODE=ladder never uses RoboLab's own per-step subtask recorder: subtask boundaries are logged by
# ladder's RecorderWrapper (run_model side) and the oracle's per-goal achieved flags are computed here
# in _get_observations, not by that recorder. Leaving it enabled also crashes on the droid robot
# (its get_wrong_object_grabbed -> gripper_fully_closed path looks up an articulation named "robot",
# which world_state does not resolve), so keep it off regardless of the shared --enable-subtask flag.
robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False

# Kit mutes USD diagnostics by default, which is why an entire class of renderer failure -- the
# payload geometry being evicted mid-episode -- produced 188 blind episodes without one log line.
carb.settings.get_settings().set("/persistent/app/usd/muteUsdDiagnostics", False)

# A composition task is named by its spec (comp_bin_2); everything else is a RoboLab task name or
# path, passed through. Both register and are then served under the Task class name.
_requested = args_cli.task or []
# One view leaves the camera set at robolab's own default; two matches the layout the cosmos3
# DROID policies were post-trained on. The second costs GPU, so it is opt-in.
_CAMERAS = WRIST_LEFT_RIGHT if args_cli.exterior_cameras == 2 else None
if args_cli.publish_depth:
    # with_depth extends each camera's data_types, leaving the rgb-only registration alone.
    _CAMERAS = [with_depth(camera) for camera in _CAMERAS or WRIST_LEFT]
auto_register_droid_envs(
    task_dirs=None if _requested and all(t in SPECS for t in _requested) else args_cli.task_dirs,
    task=[task_target(t) if t in SPECS else t for t in _requested] or None,
    cameras=_CAMERAS,
)
_TASKS = [task_class_name(t) if t in SPECS else t for t in _requested]
# Task class name -> whether its spec is scored as a sequence, for the oracle metadata below.
_ORDERED_TASKS = {task_class_name(name): spec.ordered for name, spec in SPECS.items()}
# Task class name -> whether a repeated subtask moves the SAME object again (a relay/revisit leg).
_REVISIT_TASKS = {task_class_name(name): bool(spec.via_legs) for name, spec in SPECS.items()}
# Task class name -> the spec's own plan. The oracle publishes THIS wording, so the privileged arm
# and the planner arms (served the same list from decomp_cache.json) command identical text and
# differ only in where the plan came from.
_PLAN_SUBTASKS = {name: entry["subtasks"] for name, entry in decomp_cache_entries().items()}
# Task class name -> its spec, for the payload the scripted completability policy reads.


def _half(frame) -> np.ndarray:
    """The camera's own frame at half size and UNPADDED, to pair with the depth it publishes.

    Only the aspect ratio has to match the depth frame for normalized coordinates to map, and
    halving each side preserves it exactly while quartering what goes over the wire.
    """
    image = frame.detach().cpu().numpy()
    return cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_AREA)


def _transform(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    """A 4x4 pose from a position and a wxyz quaternion, the shape Isaac reports."""
    w, x, y, z = (float(v) for v in np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4))
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return matrix


class RobolabServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        task_envs = get_envs(task=_TASKS) if _TASKS else get_envs()
        self._batches = [(task, run) for task in task_envs for run in range(args.num_runs)]
        self._batch_index = 0
        self._task = None
        self._env = None
        self._env_cfg = None
        self._converter = None
        self._raw_obs = None
        self._step_count = 0
        self._oracle = None  # (oracle command strings, per-goal checks) for the current task
        self._prev_achieved = None  # last logged privileged achieved vector, per batch
        self._metrics_dt = None
        self._ee_body_index = None
        self._traj = None  # per-env post-step trajectory samples for the current batch
        self._metrics = None  # per-env computed metric dicts (see _reset)
        self._timeline = omni.timeline.get_timeline_interface()
        self._kit_app = omni.kit.app.get_app()
        LOGGER.info(
            f"Serving {len(self._batches)} batches: {len(task_envs)} task(s) x {args.num_runs} run(s)"
        )

    def handle(self, message):
        request_type = message.get("type")
        if request_type == RequestType.RESET.value:
            return self._reset()
        if request_type == RequestType.GET_OBSERVATIONS.value:
            return self._get_observations()
        if request_type == RequestType.EXEC_ACTION.value:
            return self._execute_actions(message["data"])
        return {"status": "ok"}  # CONNECT / EXIT

    def _reset(self):
        if self._batch_index >= len(self._batches):
            return {"status": "finished"}
        task, run_index = self._batches[self._batch_index]
        self._batch_index += 1

        if task != self._task:
            if self._env is not None:
                self._env.close()
            self._env, self._env_cfg = create_env(
                task,
                device=self._args.device,
                num_envs=self._args.num_envs,
                instruction_type=self._args.instruction_type,
                policy="ladder_runner",
                renderer=self._args.renderer,
                rendering_mode=self._args.rendering_type,
            )
            self._task = task
            if self._args.no_render:
                # Isaac Lab's own switch: skips the Kit app loop and the RTX camera render, not physics.
                getattr(self._env, "unwrapped", self._env).render_enabled = False
            # Env-owned oracle: the per-goal checks from the task's own subtask spec, paired below
            # with the composition spec's plan (or, for RoboLab's own tasks, commands rendered from
            # the goal predicates).
            self._oracle = build_oracle(getattr(self._env.cfg, "subtasks", None), self._env_cfg.instruction)
            # The achieved vector is per-task, so a lane moving to a task with a different goal
            # count would die on the strict zip below. Start the new task with no history.
            self._prev_achieved = None
            # Trajectory-metric inputs, matching the native hdf5 recorder: dt = sim.dt x decimation,
            # EE body resolved like PostStepEndEffectorPoseRecorder (exact "base_link", then substring).
            unwrapped = getattr(self._env, "unwrapped", self._env)
            self._metrics_dt = float(unwrapped.cfg.sim.dt * unwrapped.cfg.decimation)
            body_names = unwrapped.scene["robot"].data.body_names
            self._ee_body_index = (
                body_names.index("base_link")
                if "base_link" in body_names
                else next((i for i, name in enumerate(body_names) if "base_link" in name), None)
            )
            if self._ee_body_index is None:
                LOGGER.warn(f"no base_link EE body among {body_names}; trajectory metrics disabled")
        else:
            self._env.reset_eval_state()
        obs, _ = self._env.reset()
        obs, _ = self._env.reset()  # native double reset (robolab.eval.episode.run_episode)
        self._raw_obs = obs
        self._step_count = 0
        # Per batch: the next task's oracle has its own number of goals, and an episode starts with
        # nothing achieved, so a vector carried over here would both mis-diff and mis-zip.
        self._prev_achieved = None
        self._converter = DroidJointposConverter(
            num_envs=self._env.num_envs, image_size=tuple(args_cli.image_size)
        )
        self._traj = [
            {
                "ee_position": [],
                "ee_linear_velocity": [],
                "joint_position": [],
                "joint_velocity": [],
                "actions": [],
            }
            for _ in range(self._env.num_envs)
        ]
        # None = episode still running; {} = closed but too short for metrics; else the metric dict.
        self._metrics = [None] * self._env.num_envs
        self._diverged = set()
        self._wedged = set()
        LOGGER.info(
            f"Batch {self._batch_index}/{len(self._batches)} task={task} run={run_index} "
            f"instruction={self._env_cfg.instruction!r}"
        )
        return {"status": "ok", "n_envs": self._env.num_envs}

    _MAIN_CAMERA = "over_shoulder_left_camera"
    _base_logged = False

    def _camera_payloads(self, image_obs):
        """Per-env calibrated depth frames for the main camera, or None when depth is off.

        ``"depth"`` aliases ``distance_to_image_plane``, so the buffer is already metres along the
        optical axis, not ray length. Poses are relative to the articulation root, the frame
        ``eef_pose`` is computed in.
        """
        if not self._args.publish_depth:
            return None
        key = next(
            (k for k in image_obs if self._MAIN_CAMERA in k and "depth" in k.lower()),
            None,
        )
        if key is None:
            raise RuntimeError(
                f"publish_depth is set but no depth observation was found; image_obs has "
                f"{sorted(image_obs)}"
            )
        depth = image_obs[key].detach().cpu().numpy().astype(np.float32)
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        sensor = self._env.unwrapped.scene[self._MAIN_CAMERA]
        intrinsics = sensor.data.intrinsic_matrices.detach().cpu().numpy()
        cam_pos = sensor.data.pos_w.detach().cpu().numpy()
        # quat_w_ros is a Warp quatf: ordered XYZW despite the name.
        cam_quat_xyzw = sensor.data.quat_w_ros.detach().cpu().numpy()
        cam_quat = np.concatenate([cam_quat_xyzw[:, 3:4], cam_quat_xyzw[:, :3]], axis=1)
        # root_quat_w is XYZW too.
        robot = self._env.unwrapped.scene["robot"]
        base_pos = robot.data.root_pos_w.detach().cpu().numpy()
        base_quat_xyzw = robot.data.root_quat_w.detach().cpu().numpy()
        if not self._base_logged:
            self._base_logged = True
            # Logged once: a wrong base frame misses every reach while every number looks fine.
            origin = self._env.unwrapped.scene.env_origins.detach().cpu().numpy()[0]
            LOGGER.info(
                f"camera base frame: root={np.round(base_pos[0] - origin, 4).tolist()} rel env_origin, "
                f"quat_xyzw={np.round(base_quat_xyzw[0], 4).tolist()}, bodies={list(robot.data.body_names)[:6]}"
            )
        base_quat = np.concatenate([base_quat_xyzw[:, 3:4], base_quat_xyzw[:, :3]], axis=1)
        payloads = []
        for env_i in range(len(depth)):
            k = intrinsics[env_i] if intrinsics.ndim == 3 else intrinsics
            world_from_base = _transform(base_pos[env_i], base_quat[env_i])
            world_from_camera = _transform(cam_pos[env_i], cam_quat[env_i])
            payloads.append(
                encode_camera(
                    depth_m=depth[env_i].astype(np.float32),
                    intrinsics=k,
                    base_from_camera=np.linalg.inv(world_from_base) @ world_from_camera,
                )
            )
        return payloads

    def _get_observations(self):
        if self._env is None or self._raw_obs is None:
            return {"status": "error", "message": "Environment not initialized"}
        image_obs = self._raw_obs["image_obs"]
        cameras = self._camera_payloads(image_obs)
        proprio = self._raw_obs["proprio_obs"]
        results = self._env.get_env_results()
        active = set(self._env.active_env_ids)
        # Belt against a never-freezing env; RoboLab's own time_out termination freezes first.
        max_steps_reached = self._step_count >= int(self._env.max_episode_length)
        commands, achieved_all, plan_subtasks = None, None, None
        if self._oracle:
            commands = [goal.command for goal in self._oracle]
            plan_subtasks = _PLAN_SUBTASKS.get(self._task)
            if plan_subtasks is not None and len(plan_subtasks) != len(commands):
                LOGGER.warn(
                    f"{self._task}: spec plan has {len(plan_subtasks)} subtasks but the env oracle has "
                    f"{len(commands)} goals; publishing the env's own commands"
                )
                plan_subtasks = None
            plan_subtasks = plan_subtasks or commands
            try:
                achieved_all = [
                    [bool(goal.check(self._env, env_id=env_i)) for goal in self._oracle]
                    for env_i in range(self._env.num_envs)
                ]
            except Exception as error:
                LOGGER.exception(  # noqa: PLE1205
                    f"Per-goal oracle checks failed for {self._task}; falling back to single-goal", error
                )
                achieved_all = None
        # Log every change of the privileged achieved vector. This is the oracle progress trace the
        # eval protocol reports per-goal progress from, and it is the only way to tell a goal that
        # was never reached from one that was reached and then UN-achieved: a goal flipping
        # True -> False means its predicate is instantaneous rather than a latched milestone, which
        # sends the advancer backwards onto work it already finished.
        if achieved_all is not None:
            prev = self._prev_achieved
            if prev != achieved_all:
                LOGGER.info(f"ORACLE_ACHIEVED {achieved_all}")
                if prev is not None:
                    for env_i, (was, now) in enumerate(zip(prev, achieved_all, strict=True)):
                        lost = [i for i, (a, b) in enumerate(zip(was, now, strict=True)) if a and not b]
                        if lost:
                            LOGGER.warn(f"ORACLE_UNACHIEVED env {env_i}: goals {lost} went True->False")
                self._prev_achieved = [list(a) for a in achieved_all]

        privileged = (
            privileged_state(getattr(self._env, "unwrapped", self._env), self._oracle, achieved_all)
            if self._args.expose_privileged and self._oracle and achieved_all is not None
            else None
        )

        data = []
        for env_i in range(self._env.num_envs):
            success = bool(results[env_i]["success"])
            truncated = max_steps_reached or env_i not in active
            if (success or truncated) and self._metrics[env_i] is None:
                # Episode just closed: native metric set over the trajectory up to this step.
                traj = {key: np.asarray(values) for key, values in self._traj[env_i].items()}
                self._metrics[env_i] = (
                    compute_episode_metrics(traj, dt=self._metrics_dt) if len(traj["ee_position"]) else None
                ) or {}
            metadata = {
                "env_name": self._task,
                "robot_name": "franka",
                "instruction": self._env_cfg.instruction,
                "done": success,
                "truncated": truncated,
                # Privileged ground truth for oracle HLCs (PrivilegedDecomposer/Advancer): the
                # synthesized per-goal commands paired with each goal's task-spec condition
                # (ladder/.../robolab/oracle.py). A task with no subtask spec publishes the
                # single-goal form — the instruction with the env-side success predicate.
                "oracle": {
                    "subtasks": plan_subtasks,
                    "achieved": achieved_all[env_i],
                    # A composition spec scored as a sequence (relay, revisit, ordered) publishes it,
                    # so the oracle arm walks the chain in the order its cached plan already gives
                    # the planner arms. RoboLab's own tasks carry no such spec and stay unordered.
                    "ordered": _ORDERED_TASKS.get(self._task, False),
                    "revisits": _REVISIT_TASKS.get(self._task, False),
                }
                if achieved_all is not None
                else {"subtasks": [self._env_cfg.instruction], "achieved": [success]},
            }
            if self._metrics[env_i]:
                metadata["metrics"] = self._metrics[env_i]
            if privileged is not None:
                metadata[METADATA_PRIVILEGED_KEY] = privileged[env_i]
            if cameras is not None:
                metadata[METADATA_CAMERAS_KEY] = {"scene": cameras[env_i]}
            data.append(
                self._converter.observation_dict(
                    env_i=env_i,
                    main_image=image_obs["over_shoulder_left_camera"][env_i].detach().cpu().numpy(),
                    wrist_image=image_obs["wrist_cam"][env_i].detach().cpu().numpy(),
                    scene_image=(_half(image_obs["over_shoulder_left_camera"][env_i]) if cameras else None),
                    right_image=(
                        None
                        if "over_shoulder_right_camera" not in image_obs
                        else image_obs["over_shoulder_right_camera"][env_i].detach().cpu().numpy()
                    ),
                    joints=proprio["arm_joint_pos"][env_i].detach().cpu().numpy(),
                    gripper_position=proprio["gripper_pos"][env_i].detach().cpu().numpy(),
                    metadata=metadata,
                )
            )
        return {"status": "ok", "data": data[0] if self._env.num_envs == 1 else data}

    def _execute_actions(self, payload):
        if self._env is None or self._raw_obs is None:
            return {"status": "error", "message": "Environment not initialized"}
        if isinstance(payload, dict) and "actions" in payload:
            actions = [RobotAction.from_dict(action_dict) for action_dict in payload["actions"]]
        else:
            actions = [RobotAction.from_dict(payload)]
        if len(actions) != self._env.num_envs:
            return {
                "status": "error",
                "message": f"expected {self._env.num_envs} actions, got {len(actions)}",
            }

        joints = self._raw_obs["proprio_obs"]["arm_joint_pos"].detach().cpu().numpy()
        commands = self._converter.actions_to_jointpos(actions, joints)
        while not self._timeline.is_playing():
            self._kit_app.update()
        obs, _, _, _, _ = self._env.step(torch.as_tensor(commands, device=self._env.device))
        self._raw_obs = obs
        self._step_count += 1
        self._record_trajectory(commands)
        self._log_divergence(obs, commands)
        return {"success": True}

    def _record_trajectory(self, commands: np.ndarray) -> None:
        """One post-step sample per still-running env, from the same articulation state the native
        hdf5 recorder writes (PostStepEndEffectorPoseRecorder: env-local EE position, world EE
        linear velocity) — the inputs of robolab's compute_episode_metrics."""
        if self._ee_body_index is None:
            return
        unwrapped = getattr(self._env, "unwrapped", self._env)
        robot = unwrapped.scene["robot"]
        ee_pos = robot.data.body_pos_w[:, self._ee_body_index] - unwrapped.scene.env_origins
        ee_pos = ee_pos.detach().cpu().numpy()
        ee_vel = robot.data.body_lin_vel_w[:, self._ee_body_index].detach().cpu().numpy()
        joint_pos = robot.data.joint_pos.detach().cpu().numpy()
        joint_vel = robot.data.joint_vel.detach().cpu().numpy()
        for env_i in range(self._env.num_envs):
            if self._metrics[env_i] is not None:
                continue  # episode closed: trajectory stays at its terminal step, like a native demo
            traj = self._traj[env_i]
            traj["ee_position"].append(ee_pos[env_i])
            traj["ee_linear_velocity"].append(ee_vel[env_i])
            traj["joint_position"].append(joint_pos[env_i])
            traj["joint_velocity"].append(joint_vel[env_i])
            traj["actions"].append(commands[env_i])

    def _log_divergence(self, obs, commands):
        """Dump the physics state the first time an env's joints go non-finite.

        The blow-up is silent, so this is the only place the cause is still visible: a joint-velocity
        ramp over several steps means the position drive was driven unstable, a one-step jump means a
        contact impulse, and objects going non-finite alongside the arm means the whole scene
        diverged rather than the articulation alone.
        """
        joints = obs["proprio_obs"]["arm_joint_pos"]
        scene = getattr(self._env, "unwrapped", self._env)
        scene = getattr(scene, "scene", None)
        robot = scene["robot"] if scene is not None else None
        if robot is not None:
            # Gripper wedge detector: the Robotiq joints are physically bounded at ~1.05 rad, so
            # a linkage past 2.0 rad is bent/wrecked even though it is far below the 6.5-rad ARM
            # sanity threshold (that gap let m27 env9 run three visibly-broken episodes without a
            # single flag). Phrased as "diverged joints" so every existing harvest/report
            # exclusion regex picks these episodes up unchanged.
            grip_q = robot.data.joint_pos[:, 7:]
            wedged = torch.isfinite(grip_q).all(dim=-1) & (
                torch.nan_to_num(grip_q).abs().max(dim=-1).values > _GRIP_WEDGE_RAD
            )
            for env_i in wedged.nonzero(as_tuple=False).flatten().tolist():
                if env_i in self._wedged or env_i in self._diverged:
                    continue
                self._wedged.add(env_i)
                LOGGER.warn(
                    f"env {env_i}: diverged joints (gripper linkage wedged, max |q| = "
                    f"{float(grip_q[env_i].abs().max()):.3g} rad); flagging episode"
                )
        finite = torch.isfinite(joints).all(dim=-1)
        for env_i in (~finite).nonzero().flatten().tolist():
            if env_i in self._diverged:
                continue
            self._diverged.add(env_i)
            objs = "n/a"
            if scene is not None:
                try:
                    objs = (
                        ",".join(
                            name
                            for name, obj in scene.rigid_objects.items()
                            if not torch.isfinite(obj.data.root_pos_w[env_i]).all()
                        )
                        or "none"
                    )
                except Exception as error:  # noqa: BLE001  -- diagnostic only; the message carries the cause
                    objs = f"unreadable ({error!r})"
            LOGGER.warn(
                f"DIVERGENCE env {env_i} step {self._step_count} task {self._task}: "
                f"nonfinite_objects=[{objs}] "
                f"commanded={commands[env_i].round(3).tolist()}"
            )


# Gripper joint magnitude that means the linkage is bent/wrecked (its physical limits are
# ~1.05 rad) while staying far below the arm sanity threshold. Detects the wedge class that
# degrades episodes without an arm explosion.
_GRIP_WEDGE_RAD = 2.0


def main() -> None:
    server = ZmqRequestReplyServer(ZmqRequestReplyServerConfig(port=args_cli.port, recv_timeout_ms=120000))
    server.start(RobolabServer(args_cli).handle)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        LOGGER.exception("[env_server] Terminated with error", error)  # noqa: PLE1205
        simulation_app.close()
        sys.exit(1)
