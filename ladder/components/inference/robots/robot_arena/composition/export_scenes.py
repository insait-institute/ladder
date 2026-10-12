import argparse
import json
import os
from typing import Any, Dict, List

from PIL import Image

from ladder.components.inference.robots.robot_arena.composition.registry import COMPOSITION_TASKS
from ladder.components.inference.robots.robot_arena.composition.tasks import compose_scene_data
from ladder.components.inference.robots.robot_arena.robot_arena_env import RobotArenaEnv, RobotArenaEnvConfig
from ladder.components.inference.robots.robot_arena.scene_loader import load_scene
from ladder.components.inference.robots.robot_arena.transforms import RobotName
from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)
# The plans the plan_* eval arms are served, so the benchmark's ground truth is what an eval uses.
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decomp_cache.json")
# Composition tasks run K=1 on native spawns (RobotArenaServer._make_composition_configs passes no
# per-env offsets), so there is one layout per task and it is the one a controller decomposes from.
DATA_ROOT = "ladder/components/inference/robots/robot_arena/robot_arena_data/examples/data"
CAMERA = (640, 480)


def scene_objects(task) -> List[str]:
    """Every object the scene places, through the same composition the env runs."""
    scene = compose_scene_data(load_scene(DATA_ROOT, task.scene), task, DATA_ROOT)
    return sorted(obj.name for obj in scene.objects)


def reset_frame(task_name: str, path: str) -> None:
    """Save the reset frame a controller decomposes from.

    Drives RobotArenaEnv rather than a bare camera render: the observation pipeline composites the
    foreground over the scene's real-image background, which a raw render leaves out.
    """
    task = COMPOSITION_TASKS[task_name]
    env = RobotArenaEnv(
        RobotArenaEnvConfig(
            data_root=DATA_ROOT,
            scene_name=task.scene,
            robot_name=RobotName.WIDOWX,
            max_episode_steps=task.max_episode_steps,
            control_freq=5,
            sim_freq=200,
            camera_width=CAMERA[0],
            camera_height=CAMERA[1],
            composition_task=task_name,
        )
    )
    try:
        Image.fromarray(env.reset()[0].images["main"]).save(path)
    finally:
        env.close()


# The registry carries no family field, so it is derived from the task's own structure -- the same
# shape dump_sim.py uses for SimplerEnv, where families are read off the name.
def family_of(name: str, task) -> str:
    if task.ordered:
        return "ordered"
    if task.legs:
        if "route" in name:
            return "relay"
        for key in ("revisit", "relay", "duplicates"):
            if key in name:
                return key
        return "split"
    for key, fam in (("except", "exclusion"), ("split", "split"), ("transfer", "transfer")):
        if key in name:
            return fam
    return "transfer" if task.start_on else "single_destination"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RobotArena-Comp decomposition scenes.")
    parser.add_argument("--out-dir", required=True, help="directory for the frames and scenes.json")
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="render only these scenes (registry names); default renders every plan in the cache",
    )
    args = parser.parse_args()

    with open(CACHE) as handle:
        plans: Dict[str, Any] = json.load(handle)
    plans.pop("_meta", None)  # the cache's provenance header, not a scene
    os.makedirs(args.out_dir, exist_ok=True)

    scenes: List[Dict[str, Any]] = []
    items = sorted(plans.items())
    if args.scenes is not None:
        wanted = set(args.scenes)
        items = [(key, plan) for key, plan in items if key.removeprefix("robot_arena_") in wanted]
    for index, (key, plan) in enumerate(items, start=1):
        # A cache value is a plain list OR {"subtasks", "ordered"}, the shape a sequence-scored
        # task is served as.
        subtasks = plan["subtasks"] if isinstance(plan, dict) else plan
        name = key.removeprefix("robot_arena_")  # the cache keys are namespaced across suites
        task = COMPOSITION_TASKS[name]
        frame = f"{name}__main.png"
        try:
            reset_frame(name, os.path.join(args.out_dir, frame))
        except Exception as exc:  # noqa: BLE001  -- one unrenderable scene must not cost the others
            LOGGER.warn(f"[{index}/{len(items)}] {name}: FAILED {exc!r}")
            continue
        scenes.append(
            {
                "id": f"robotarena/{name}",
                "benchmark": "robotarena_comp",
                "scene": name,
                "instruction": task.instruction,
                "subtasks": list(subtasks),
                "n_goals": len(task.goals),
                # The plan is the newer source: a sequence-scored task (relay, revisit) is SERVED as
                # ordered even where task.ordered, the narrower per-task flag, reads False.
                "ordered": bool(plan["ordered"]) if isinstance(plan, dict) else task.ordered,
                "revisits": bool(task.milestones),
                # Part of the published schema; only the exclusion families place one.
                "family": family_of(name, task),
                "distractors": list(task.excluded),
                # Scene contents, so a task that changes what it PLACES invalidates samples cut
                # against it. The wording fields cannot see such a change.
                "objects": scene_objects(task),
                # Frame names, not paths: they sit next to this file, so the directory stays relocatable.
                "frames": {"main": frame},
            }
        )
        LOGGER.info(f"[{index}/{len(items)}] {name}")

    # Each scene is rendered under its own try, so a total failure still reaches here with an
    # empty list and would publish an empty benchmark under a zero exit code.
    if not scenes:
        raise RuntimeError(f"no scene rendered out of {len(items)}; see the warnings above")
    manifest = os.path.join(args.out_dir, "scenes.json")
    if args.scenes is not None and os.path.isfile(manifest):
        merged = {record["id"]: record for record in json.load(open(manifest))}
        merged.update({record["id"]: record for record in scenes})
        scenes = sorted(merged.values(), key=lambda record: record["id"])
    with open(manifest, "w") as handle:
        json.dump(scenes, handle, indent=1)
    LOGGER.info(f"{len(scenes)}/{len(items)} scene(s) -> {args.out_dir}/scenes.json")


if __name__ == "__main__":
    main()
