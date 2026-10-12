import argparse
import json
import logging
import os
import sys
import traceback

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
# composition/ is one level under robolab/; the runfiles-staged asset tree is robolab/assets.
os.environ.setdefault(
    "ROBOLAB_ASSET_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"),
)

import cv2  # noqa: F401 -- must import before isaaclab. Do not remove
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Render each RoboLab composition scene's start state from the over-shoulder and wrist "
    "cameras, and write the scene records the decomposition benchmark is assembled from."
)
# NOT --out: add_common_eval_args contributes --output-folder-name, which makes a bare --out an
# ambiguous abbreviation and aborts argument parsing.
parser.add_argument("--out-dir", required=True, help="Output directory for the PNGs and scenes.json.")
parser.add_argument(
    "--settle-updates",
    type=int,
    default=8,
    help="Kit updates after reset to let the tiled cameras render a fresh frame.",
)
parser.add_argument(
    "--scenes",
    nargs="*",
    default=None,
    help="render only these tasks (Task class names); default renders every composition spec",
)

from robolab.eval.runner import add_common_eval_args  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import robolab.constants  # noqa: E402
import torch  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402

from ladder.components.inference.robots.robolab.composition.specs import (  # noqa: E402
    SPECS,
    task_class_name,
    task_target,
)

# The composition tasks live in their own modules, so they are registered by target path the way
# env_server does it. Without this the exporter renders whatever else is registered (RoboLab's
# own tasks), and every one of them fails the spec lookup below -- frames written, no records.
# --task-dirs must NOT gate this: it defaults to DEFAULT_TASK_SUBFOLDERS, i.e. it is never empty.
if not args_cli.task:
    args_cli.task = [task_target(name) for name in SPECS]

# stdlib logging, as this package's other composition entry points use: ladder.core.common.logger
# needs //ladder/core/common:platform, which only env_server's target declares.
LOGGER = logging.getLogger("export_scenes")
# The plans the plan_* eval arms are served, so the benchmark's ground truth is what an eval uses.
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decomp_cache.json")
# The two views env_server publishes to a policy, so a planner scored on these frames sees what it
# would see at reset in an eval.
CAMERAS = {"main": "over_shoulder_left_camera", "wrist": "wrist_cam"}

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
# contact sensors ENABLED: env.step()'s termination check needs them (eval keeps them too)

try:
    auto_register_droid_envs(task_dirs=args_cli.task_dirs, task=args_cli.task)
except Exception as exc:  # noqa: BLE001 -- diagnostic: registration failing silently loses every task
    print(f"[export_scenes] auto_register raised: {type(exc).__name__}: {exc}", flush=True)


def _to_bgr(image) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:  # float renders arrive in [0, 1]
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2BGR)  # drop alpha if present


def main() -> None:
    with open(CACHE) as handle:
        plans = json.load(handle)
    plans.pop("_meta", None)  # provenance header, not a scene
    by_class = {task_class_name(name): spec for name, spec in SPECS.items()}
    os.makedirs(args_cli.out_dir, exist_ok=True)
    # get_envs(task=...) filters REGISTERED NAMES, and the composition tasks are asked for by target
    # path ("tasks.py::CompBin2Task"), which matches no name -- it returns empty rather than raising.
    # auto_register above registered them under their class names, so filter the full registry.
    registered = get_envs()
    tasks = [task for task in registered if task in by_class]
    if args_cli.scenes is not None:
        wanted = set(args_cli.scenes)
        tasks = [task for task in tasks if task in wanted]
    LOGGER.info(f"{len(registered)} env(s) registered, {len(tasks)} of them composition tasks")
    LOGGER.info(f"rendering {len(tasks)} composition task(s): {tasks}")

    scenes, env = [], None
    for task in tasks:
        try:
            if env is not None:
                env.close()
            env, env_cfg = create_env(
                task,
                device=args_cli.device,
                num_envs=1,
                instruction_type=args_cli.instruction_type,
                policy="ladder_runner",
                renderer=args_cli.renderer,
                rendering_mode=args_cli.rendering_type,
            )
            env.reset()
            obs, _ = env.reset()  # native double reset (robolab.eval.episode.run_episode)
            # Step the full render pipeline: the reset-return image can predate the sampled poses.
            hold = torch.zeros_like(env.action_manager.action)
            for _ in range(args_cli.settle_updates):
                obs, _, _, _, _ = env.step(hold)

            # Frame names, not paths: they sit next to scenes.json, so the directory stays relocatable.
            frames = {label: f"{task}__{label}.png" for label in CAMERAS}
            for label, camera in CAMERAS.items():
                cv2.imwrite(
                    os.path.join(args_cli.out_dir, frames[label]),
                    _to_bgr(obs["image_obs"][camera][0].detach().cpu().numpy()),
                )

            spec = by_class[task]
            # A cache value is a plain list or {"subtasks", "ordered"} (see DecomposerConfig);
            # storing the dict whole made len(subtasks) the key count.
            plan = plans[task]
            subtasks = plan["subtasks"] if isinstance(plan, dict) else plan
            scenes.append(
                {
                    "id": f"robolab/{task}",
                    "benchmark": "robolab_comp",
                    "scene": task,
                    "instruction": env_cfg.instruction,
                    "subtasks": subtasks,
                    "n_goals": len(spec.goal_objects),
                    "ordered": spec.ordered,
                    "revisits": bool(spec.via_legs),
                    # Part of the published schema. The registry carries the family directly; a
                    # distractor is an object the scene places with no receptacle to reach.
                    "family": spec.family,
                    "distractors": [o.type_id for o in spec.objects if o.receptacle is None],
                    # Scene contents, so a task that changes what it PLACES invalidates samples cut
                    # against it. The wording fields cannot see such a change.
                    "objects": sorted([o.prim for o in spec.objects] + [r.prim for r in spec.receptacles]),
                    "frames": frames,
                }
            )
            LOGGER.info(f"rendered {task} instruction={env_cfg.instruction!r}")
        except Exception:  # noqa: BLE001  -- one unrenderable scene must not cost the others
            LOGGER.warning(f"{task}: FAILED {traceback.format_exc(limit=2)}")

    if env is not None:
        env.close()
    # Each scene is rendered under its own try, so a total failure still reaches here with an
    # empty list and would publish an empty benchmark under a zero exit code.
    if not scenes:
        raise RuntimeError(f"no scene rendered out of {len(tasks)}; see the warnings above")
    manifest = os.path.join(args_cli.out_dir, "scenes.json")
    if args_cli.scenes is not None and os.path.isfile(manifest):
        merged = {record["id"]: record for record in json.load(open(manifest))}
        merged.update({record["id"]: record for record in scenes})
        scenes = sorted(merged.values(), key=lambda record: record["id"])
    with open(manifest, "w") as handle:
        json.dump(scenes, handle, indent=1)
    LOGGER.info(f"{len(scenes)}/{len(tasks)} scene(s) -> {args_cli.out_dir}/scenes.json")
    simulation_app.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    try:
        main()
    except Exception as e:  # noqa: BLE001  -- top-level handler: close the sim app before exiting
        print(f"[export_scenes] Terminated with error: {e}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
