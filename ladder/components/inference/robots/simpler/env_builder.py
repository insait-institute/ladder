import dataclasses
import os
from typing import Optional

import gymnasium as gym
import sapien.physx

# ManiSkill3 resolves its asset directory from MS_ASSET_DIR when mani_skill is imported, so point it
# at the assets bundled into the runfiles (`:maniskill_assets` in BUILD) before that import. Evals
# and tests then need no runtime asset download. Mirrors the ManiSkill2 MS2_REAL2SIM_ASSET_DIR setup.
if "MS_ASSET_DIR" not in os.environ:
    os.environ["MS_ASSET_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maniskill3")

# Same idea for SAPIEN's PhysX GPU lib, which it downloads from GitHub to ~/.sapien/physx/<version>
# on first enable_gpu(): seed that cache from the vendored copy (`:sapien_physx` in BUILD). Seed via a
# per-process temp + atomic replace so concurrent eval lanes don't race, and so a stale symlink left by
# a rotated runfiles tree is clobbered rather than blocking the seed (and SAPIEN's own download).
_physx = os.path.join(
    os.path.expanduser("~"), ".sapien", "physx", sapien.physx.version(), "libPhysXGpu_64.so"
)
_vendored = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sapien_physx", "libPhysXGpu_64.so")
if os.path.exists(_vendored) and not os.path.exists(_physx):
    os.makedirs(os.path.dirname(_physx), exist_ok=True)
    _physx_tmp = f"{_physx}.{os.getpid()}"
    os.symlink(_vendored, _physx_tmp)
    os.replace(_physx_tmp, _physx)

# Do NOT REMOVE noqa, otherwise autoflake will remove the import (it registers the envs).
import mani_skill.envs  # noqa
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.base_env import (  # noqa: E402
    WidowX250SBridgeDatasetFlatTable,
)
from mani_skill.utils.structs.types import SimConfig  # noqa: E402

from ladder.components.inference.robots.simpler.composition import envs  # noqa: E402,F401
from ladder.core.common.logger import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

# A ManiSkill2 fork added an absolute base-frame EE-pose controller
# (arm_pd_ee_target_base_pose). ManiSkill3's bridge robot registers only the delta variant, so
# register an absolute copy of it (use_delta=False, reusing all stock gains) under the control_mode
# the configs request. Sink inherits this robot, so eggplant is covered too.
ABS_CONTROL_MODE = "arm_pd_ee_target_base_pose_gripper_pd_joint_pos"
_DELTA_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
_stock_controller_configs = WidowX250SBridgeDatasetFlatTable._controller_configs.fget  # noqa: SLF001


def _controller_configs(self):
    configs = _stock_controller_configs(self)
    delta = configs[_DELTA_MODE]
    configs[ABS_CONTROL_MODE] = dict(
        arm=dataclasses.replace(delta["arm"], use_delta=False),
        gripper=delta["gripper"],
    )
    return configs


WidowX250SBridgeDatasetFlatTable._controller_configs = property(_controller_configs)  # noqa: SLF001


def build_simpler_env(
    env_name: str,
    n_envs: int,
    control_mode: str,
    sim_freq: int,
    control_freq: int,
    max_episode_steps: int,
    episode_layouts: Optional[int] = None,
    publish_depth: bool = False,
    render_images: bool = True,
) -> gym.Env:
    env_id = env_name.replace("-v0", "-v1")  # ManiSkill3 ids end in -v1
    # Composition envs generate their object layouts at construction; only they accept n_layouts, so
    # forward it solely when the config sets it (base bridge envs have fixed grids and no such kwarg).
    layout_kwargs = {"n_layouts": episode_layouts} if episode_layouts is not None else {}
    return gym.make(
        env_id,
        num_envs=n_envs,
        obs_mode=("rgb+depth+segmentation" if publish_depth else "rgb+segmentation")
        if render_images
        else "none",
        sim_backend="gpu",
        control_mode=control_mode,
        # <=0 means "use the env's own registered max_episode_steps" (e.g. the per-goal step budget
        # baked into the SPECS) instead of overriding it from the sim config.
        max_episode_steps=max_episode_steps if max_episode_steps > 0 else None,
        sim_config=SimConfig(sim_freq=sim_freq, control_freq=control_freq),
        **layout_kwargs,
    )
