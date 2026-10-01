from typing import Any, Dict, List, Optional

import numpy as np
from mani_skill.utils import common

from ladder.components.inference.api.core.privileged_geometry import build_payloads


def privileged_state(env) -> Optional[List[Dict[str, Any]]]:
    """Privileged payload for whichever kind of scene this env is, or None if it is neither.

    A composition env states its own goals, so it publishes its payload directly. The four canonical
    bridge tasks are upstream ManiSkill classes that cannot carry a method of ours, but they do name
    their source/target pair and keep the settle-time bboxes their success test scores with, so the
    single goal is derived here. Both go through the same shared builder.
    """
    if hasattr(env, "privileged_state"):
        return env.privileged_state()
    if not hasattr(env, "source_obj_name"):
        return None
    robot_inv = env.agent.robot.pose.inv()
    names = [env.source_obj_name, env.target_obj_name]
    bboxes = [env.episode_source_obj_bbox_world, env.episode_target_obj_bbox_world]
    centers = np.stack([common.to_numpy((robot_inv * env.objs[n].pose).p) for n in names])
    # The bbox is already per-env here (upstream rotates it by each env's quaternion at settle).
    half = np.abs(np.stack([common.to_numpy(b) for b in bboxes])) / 2.0
    half = np.broadcast_to(half.reshape(len(names), -1, 3), centers.shape)
    grasped = np.stack([common.to_numpy(env.agent.is_grasping(env.objs[n])).reshape(-1) for n in names])
    satisfied = common.to_numpy(env.evaluate()["success"]).reshape(-1)
    return build_payloads(
        names=names,
        centers=centers,
        mins=centers - half,
        maxs=centers + half,
        grasped=grasped,
        goals=[{"source": 0, "destination": 1, "relation": "on", "satisfied": satisfied}],
        n_envs=env.num_envs,
    )
