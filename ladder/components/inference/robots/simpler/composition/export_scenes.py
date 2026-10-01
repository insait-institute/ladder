import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

from ladder.components.inference.robots.simpler.composition.placement import strip_instance_id
from ladder.components.inference.robots.simpler.composition.specs import DISPLAY, SPECS, scene_layout
from ladder.components.inference.robots.simpler.env_builder import build_simpler_env
from ladder.components.inference.robots.simpler.simpler_env import CAMERA_NAME, _to_np
from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)
# The plans the plan_* eval arms are served, so the benchmark's ground truth is what an eval uses.
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decomp_cache.json")
CONTROL_MODE = "arm_pd_ee_target_base_pose_gripper_pd_joint_pos"


def reset_frame(scene: str, path: str) -> None:
    """The frame the orchestrator is shown: env 0's reset observation, from the camera SimplerServer
    fills ``images["main"]`` from.

    The episode_id matters. The eval draws one per env from ``init_range_config.object_episode_range``
    and a controller decomposes from ``observations[0]`` only, so env 0's id -- 0 -- selects the one
    layout that is ever decomposed. Omitting it takes the env default, which is a different layout.
    """
    env = build_simpler_env(
        env_name=f"{scene}-v0",
        n_envs=1,
        control_mode=CONTROL_MODE,
        sim_freq=500,
        control_freq=5,
        max_episode_steps=0,
        episode_layouts=24,
    )
    try:
        observation, _ = env.reset(seed=[0], options={"episode_id": torch.tensor([0])})
        rgb = _to_np(observation["sensor_data"][CAMERA_NAME]["rgb"])[0]
        Image.fromarray(rgb.astype(np.uint8)).save(path)
    finally:
        env.close()


# `Spec` carries no family field -- the registry groups by name prefix. Mirrors dump_sim.py, which
# feeds the protocol site from the same registry; the two must not drift.
_FAMILIES = (
    ("all", "single_destination"),
    ("split", "split"),
    ("move_plate_to_towel", "transfer"),
    ("move_towel_to_plate", "transfer"),
    ("relay", "relay"),
    ("shuttle", "shuttle"),
    ("ordered_", "ordered"),
    ("except_", "exclusion"),
    ("spoons", "color_binding"),
)


def family_of(name: str) -> str:
    for prefix, family in _FAMILIES:
        if name.startswith(prefix):
            return family
    raise KeyError(f"no family for spec {name!r} -- add it to _FAMILIES")


def show(name: str) -> str:
    # The registry's own display table -- reusing it keeps the published distractor names identical
    # to the ones the instructions use ("blue towel", not table_cloth_generated_shorter).
    return DISPLAY.get(strip_instance_id(name), strip_instance_id(name))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SimplerEnv-Comp decomposition scenes.")
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

    by_name = {spec.name: spec for spec in SPECS}
    items = sorted(plans.items())
    if args.scenes is not None:
        wanted = set(args.scenes)
        items = [(name, plan) for name, plan in items if name in wanted]
    scenes: List[Dict[str, Any]] = []
    for index, (name, plan) in enumerate(items, start=1):
        spec = by_name[name]
        frame = f"{name}__main.png"
        try:
            reset_frame(name, os.path.join(args.out_dir, frame))
        except Exception as exc:  # noqa: BLE001  -- one unrenderable scene must not cost the others
            LOGGER.warn(f"[{index}/{len(items)}] {name}: FAILED {exc!r}")
            continue
        subtasks = plan["subtasks"] if isinstance(plan, dict) else plan
        scenes.append(
            {
                "id": f"simplerenv/{name}",
                "benchmark": "simplerenv_comp",
                "scene": name,
                "instruction": spec.instruction,
                "subtasks": list(subtasks),
                "n_goals": len(spec.goals),
                "ordered": bool(spec.ordered),
                "revisits": bool(spec.revisits),
                # Part of the published schema. The registry carries distractors directly; the
                # family is read off the name, the same table dump_sim.py uses for the site.
                "family": family_of(name),
                "distractors": [show(d) for d in spec.distractors],
                # Scene contents, so a task that changes what it PLACES invalidates samples cut
                # against it. The wording fields cannot see such a change.
                "objects": sorted(scene_layout(spec).obj_names),
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
