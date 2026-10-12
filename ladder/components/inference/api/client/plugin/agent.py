from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ladder.components.inference.api.client.plugin.system import load, observation_dict
from ladder.components.inference.api.client.program import ProgramEpisode, ProgramRunner, truncated
from ladder.components.inference.api.core.action import EndEffectorPoseAction, ManipulatorAction
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.core.databib.config import Config, Configurable

_STILL_STEPS = 3
_STILL_M = 0.002
_GRIPPER_STILL = 0.005
# every suite starts the episode with the hand pointing straight down, and Arm keeps it so
_DOWN = np.array([0.0, 0.0, -1.0])


class ArmConfig(Config):
    """How ``Arm`` moves this suite's robot. Taken from GPT-Policy's per-suite host settings."""

    tool_offset_m: float  # from the end-effector frame down to the point between the fingertips
    max_step_m: float  # the most the commanded point moves in one control step
    settle_steps: int  # the most steps spent waiting for the arm to arrive
    gripper_steps: int  # the most steps spent waiting for the fingers to stop
    workspace_min_m: List[float]
    workspace_max_m: List[float]


class PluginAgentConfig(Config):
    system: str  # path/to/file.py:ClassName
    arm: ArmConfig
    timeout_s: float = 1800.0  # an episode whose program asks for no step for this long is abandoned


class Arm:
    """The robot as an agent's program drives it: blocking moves in metres, in the robot base frame.

    ``move_to`` and ``close``/``open`` return once the arm or the fingers have stopped. ``step`` is
    one raw control step, for a program that wants to command poses itself. The fingertip point
    moves; the hand keeps the orientation it had at the start of the episode (pointing down)."""

    def __init__(self, episode: ProgramEpisode, config: ArmConfig) -> None:
        self._episode = episode
        self._config = config
        self._rotation = np.asarray(episode.observation.state.eef_pose, dtype=np.float64)[:3, :3].copy()
        self._gripper = 1.0 if self._openness() > 0.5 else 0.0
        self._target = self.position()

    @property
    def task(self) -> str:
        return self._episode.task

    @property
    def observation(self) -> Dict[str, Any]:
        """``plugin.system.observation_dict`` of the current step."""
        return observation_dict(self._episode.observation, self._episode.task, self._episode.step)

    def position(self) -> np.ndarray:
        """Where the fingertip point is now, [x, y, z]."""
        pose = np.asarray(self._episode.observation.state.eef_pose, dtype=np.float64)
        return pose[:3, 3] + _DOWN * self._config.tool_offset_m

    def move_to(self, xyz: Sequence[float]) -> np.ndarray:
        """Move the fingertip point to ``xyz`` in a straight line; returns where it arrived."""
        target = np.clip(
            np.asarray(xyz, dtype=np.float64), self._config.workspace_min_m, self._config.workspace_max_m
        )
        start = self._target
        count = max(1, int(np.ceil(np.linalg.norm(target - start) / self._config.max_step_m)))
        for i in range(1, count + 1):
            self._command(start + (target - start) * i / count)
        self._wait(
            lambda: float(np.linalg.norm(self.position() - target)), self._config.settle_steps, _STILL_M
        )
        return self.position()

    def close(self) -> bool:
        """Close the fingers; True if they stopped on something rather than shutting completely."""
        self._set_gripper(0.0)
        return self._openness() > 0.05

    def open(self) -> None:
        self._set_gripper(1.0)

    def locate(self, camera: str, x: float, y: float) -> Optional[np.ndarray]:
        """The base-frame point seen at pixel (x, y) of ``camera``'s image, from its depth; None
        when the camera has no depth there."""
        frame = self._episode.observation.cameras.get(camera)
        image = (self._episode.observation.images or {}).get(camera)
        if frame is None or image is None:
            return None
        scale = np.array([frame.shape[1] / image.shape[1], frame.shape[0] / image.shape[0]])
        point = frame.unproject(np.array([[x, y]]) * scale)[0]
        return None if not np.isfinite(point).all() else point

    def step(self, eef_pose: np.ndarray, gripper: float) -> None:
        """One control step: the end-effector pose [4, 4] and the gripper (1 open, 0 closed)."""
        pose = np.asarray(eef_pose, dtype=np.float32)
        self._episode.act(
            ManipulatorAction(
                arm_action=EndEffectorPoseAction(
                    translation=pose[:3, 3].copy(), rotation=pose[:3, :3].copy()
                ),
                gripper_action=float(gripper),
            )
        )

    def _command(self, tip: np.ndarray) -> None:
        self._target = tip
        pose = np.eye(4)
        pose[:3, :3] = self._rotation
        pose[:3, 3] = tip - _DOWN * self._config.tool_offset_m
        self.step(pose, self._gripper)

    def _set_gripper(self, value: float) -> None:
        self._gripper = value
        self._wait(self._openness, self._config.gripper_steps, _GRIPPER_STILL)

    def _wait(self, measure: Callable[[], float], steps: int, still: float) -> None:
        """Hold the last target until ``measure`` stops changing, for at most ``steps`` steps."""
        history = []
        for _ in range(steps):
            self._command(self._target)
            history.append(measure())
            if len(history) > _STILL_STEPS and abs(history[-1] - history[-1 - _STILL_STEPS]) < still:
                return

    def _openness(self) -> float:
        return float(np.asarray(self._episode.observation.state.gripper_state).reshape(-1)[0])


class PluginAgent(Configurable[PluginAgentConfig]):
    """Runs a user's agent: a program that drives the arm directly, one plain object per episode.

    The user's class is constructed with no arguments at the start of every episode and its
    ``run(arm)`` is called once, on its own thread; ``arm`` is an ``Arm``. The program may block
    (move, wait for a model) as long as it likes: the simulator steps only when it acts, and every
    environment's program runs at the same time (``api/client/program.ProgramRunner``). The episode
    ends when the suite's step budget is spent or the task is done, whatever the program is doing."""

    def __init__(self, config: PluginAgentConfig) -> None:
        super().__init__(config)
        self._cls = load(config.system)
        self._programs = ProgramRunner(self._run, "agent", config.timeout_s, over=_over)

    def reset(self) -> None:
        self._programs.reset()

    def predict_actions(
        self,
        observations: List[RobotObservation],
        instructions: List[str],
        env_indices: Optional[List[int]] = None,
    ) -> List[ManipulatorAction]:
        return self._programs.actions(observations, instructions, env_indices)

    def pop_events(self) -> Dict[int, List[Dict[str, object]]]:
        return self._programs.pop_events()

    def _run(self, episode: ProgramEpisode) -> None:
        self._cls().run(Arm(episode, self.config.arm))


def _over(observation: RobotObservation) -> bool:
    return truncated(observation) or bool((observation.metadata or {}).get("done", False))
