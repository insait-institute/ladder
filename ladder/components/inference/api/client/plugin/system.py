import functools
import importlib.util
import os
from typing import Any, Dict, Optional

import numpy as np

from ladder.components.inference.api.core.observation import RobotObservation


@functools.lru_cache(maxsize=None)
def load(spec: str) -> Any:
    """The object ``Name`` defined in ``path/to/file.py``, from ``"path/to/file.py:Name"``.

    The file is loaded by path, not imported by module name, so it needs no package, no BUILD target
    and cannot shadow (or be shadowed by) a module of the same name. ``scripts/eval.sh`` passes an
    absolute path."""
    path, _, name = spec.rpartition(":")
    if not path or not name:
        raise ValueError(f"expected path/to/file.py:Name, got {spec!r}")
    name_in_sys = "ladder_system_" + os.path.splitext(os.path.basename(path))[0]
    module_spec = importlib.util.spec_from_file_location(name_in_sys, os.path.expanduser(path))
    if module_spec is None or module_spec.loader is None:
        raise FileNotFoundError(f"cannot load {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, name)


def observation_dict(observation: RobotObservation, instruction: str, step: int) -> Dict[str, Any]:
    """What a plugged-in system sees of one environment: images, the arm, depth when the run publishes
    it, and the instruction. Only what a real robot would have: the simulator's privileged state in
    ``observation.metadata`` is left out on purpose.

    - ``images``: {camera: [H, W, 3] uint8}
    - ``eef_pose``: [4, 4] end-effector pose in the robot base frame (metres)
    - ``gripper``: 1.0 open ... 0.0 closed
    - ``joints``: [n] joint positions (radians), when the robot publishes them
    - ``depth``: {camera: {depth_m [H, W] (NaN = no reading), intrinsics [3, 3], base_from_camera [4, 4]}}
    - ``instruction``, ``step`` (control steps since the episode started), ``robot`` (widowx or franka)
    """
    state = observation.state
    out: Dict[str, Any] = {
        "instruction": instruction,
        "step": step,
        "robot": str((observation.metadata or {}).get("robot_name", "")),
        "images": dict(observation.images or {}),
        "eef_pose": np.asarray(state.eef_pose, dtype=np.float32),
        "gripper": float(np.asarray(state.gripper_state).reshape(-1)[0]),
    }
    joints: Optional[np.ndarray] = getattr(state, "joints", None)
    if joints is not None:
        out["joints"] = np.asarray(joints, dtype=np.float32)
    if observation.cameras:
        out["depth"] = {
            name: {
                "depth_m": frame.depth_m,
                "intrinsics": frame.intrinsics.astype(np.float32),
                "base_from_camera": frame.base_from_camera.astype(np.float32),
            }
            for name, frame in observation.cameras.items()
        }
    return out
