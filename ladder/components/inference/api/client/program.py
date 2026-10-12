import dataclasses
import threading
from typing import Callable, Dict, List, Optional, Protocol

import numpy as np

from ladder.components.inference.api.core.action import EndEffectorPoseAction, ManipulatorAction
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)


class Robot(Protocol):
    """One arm as a blocking program drives it: read where it is, command where it goes next."""

    @property
    def observation(self) -> RobotObservation:
        ...

    def act(self, action: ManipulatorAction) -> None:
        ...


class EpisodeOver(BaseException):
    """Raised inside an episode's thread when the runner has moved on, to unwind the program. A
    BaseException so that a program catching Exception around its own work cannot swallow it."""


def truncated(observation: RobotObservation) -> bool:
    return bool((observation.metadata or {}).get("truncated", False))


@dataclasses.dataclass
class ProgramEpisode:
    """One environment's episode as the program sees it: a robot to drive, the task to do, and
    somewhere to record what happened.

    A program calls ``act`` and ``record`` and reads ``observation``, ``task``, ``step`` and
    ``done``. The methods under "the runner's side" belong to ``ProgramRunner``; a program must not
    call them. The rendezvous is a single slot behind one condition variable: both sides take the
    same lock and ``step`` only ever grows, so neither can miss or overwrite the other's signal.
    """

    env_idx: int
    task: str
    observation: RobotObservation
    step: int = 0
    events: List[Dict[str, object]] = dataclasses.field(default_factory=list)
    _action: Optional[ManipulatorAction] = None
    _last: Optional[ManipulatorAction] = None
    _done: bool = False
    _state: threading.Condition = dataclasses.field(default_factory=threading.Condition, repr=False)

    @property
    def done(self) -> bool:
        """True once the runner has retired this episode, whichever side ended it."""
        return self._done

    def act(self, action: ManipulatorAction) -> None:
        """Offer this step's action and return once the environment has stepped."""
        with self._state:
            if self._done:
                raise EpisodeOver()
            self._action = action
            acted_on = self.step
            self._state.notify_all()
            self._state.wait_for(lambda: self._done or self.step > acted_on)
            if self._done:
                raise EpisodeOver()

    def record(self, event: Dict[str, object]) -> None:
        self.events.append(event)

    # --- the runner's side ---

    def deliver(self, observation: RobotObservation, ended: bool = False) -> None:
        """The environment has stepped. ``ended`` retires the episode in the same breath, so a
        program can never wake to the final observation and act on it."""
        with self._state:
            self.observation = observation
            self.step += 1
            self._done = self._done or ended
            self._state.notify_all()

    def await_action(self, timeout_s: float) -> bool:
        with self._state:
            return self._state.wait_for(lambda: self._done or self._action is not None, timeout_s)

    def take(self) -> ManipulatorAction:
        """This step's action: whatever the program left, else the one it gave last, else hold. An
        episode that never acted holds the pose it started at rather than tracking drift."""
        with self._state:
            if self._action is not None:
                self._last, self._action = self._action, None
            elif self._last is None:
                self._last = _hold(self.observation)
            return self._last

    def release(self) -> None:
        with self._state:
            self._done = True
            self._state.notify_all()


class ProgramRunner:
    """The standard for a policy whose behaviour is a PROGRAM rather than a function of the
    observation: straight-line code that reaches, closes, looks and decides, one program per
    environment.

    A batched runner steps every environment together and wants one action back per environment per
    step, while a program holds a call stack across many steps. Those two shapes cannot share a
    stack, so each environment gets a thread that hands back one action per step and blocks until
    the runner has taken it. Threads rather than generators because a program may be written by a
    model at run time and run with ``exec``, and code you did not write cannot yield.

    Use this ONLY for that shape. A policy that computes an action from the observation in front of
    it, learned or scripted, keeps a per-environment state object and steps it synchronously, as
    ``ScriptedPolicy`` and ``VoLo`` do; it needs none of this.

    The runner owns the whole lifecycle. It starts each program, retires an episode the moment its
    environment reports the episode over, abandons one that stops asking for steps, and records a
    program that raises as an episode event so a failure costs one environment rather than the
    cell. A program is therefore plain straight-line code with no bookkeeping of its own.
    """

    def __init__(
        self,
        program: Callable[[ProgramEpisode], None],
        name: str,
        timeout_s: float,
        over: Callable[[RobotObservation], bool] = truncated,
    ) -> None:
        self._program = program
        self._name = name
        self._timeout_s = timeout_s
        self._over = over
        self._episodes: Dict[int, ProgramEpisode] = {}

    def reset(self) -> None:
        for episode in self._episodes.values():
            episode.release()
        self._episodes = {}

    def actions(
        self,
        observations: List[RobotObservation],
        instructions: List[str],
        env_indices: Optional[List[int]] = None,
    ) -> List[ManipulatorAction]:
        indices = list(env_indices) if env_indices is not None else list(range(len(observations)))
        for env_idx in set(self._episodes) - set(indices):
            self._episodes[env_idx].release()
        episodes: List[ProgramEpisode] = []
        for env_idx, observation, instruction in zip(indices, observations, instructions, strict=True):
            episode = self._episodes.get(env_idx)
            if episode is None:
                episode = self._episodes[env_idx] = self._start(env_idx, instruction, observation)
            else:
                episode.deliver(observation, ended=self._over(observation))
            episodes.append(episode)
        for episode in episodes:
            if not episode.await_action(self._timeout_s):
                LOGGER.warn(
                    f"{self._name} env={episode.env_idx}: stalled for {self._timeout_s:.0f}s; abandoned"
                )
                episode.release()
        return [episode.take() for episode in episodes]

    def pop_events(self) -> Dict[int, List[Dict[str, object]]]:
        pending = {idx: list(episode.events) for idx, episode in self._episodes.items() if episode.events}
        for env_idx in pending:
            self._episodes[env_idx].events.clear()
        return pending

    def _start(self, env_idx: int, instruction: str, observation: RobotObservation) -> ProgramEpisode:
        episode = ProgramEpisode(env_idx=env_idx, task=instruction, observation=observation)
        threading.Thread(
            target=self._execute, args=(episode,), name=f"{self._name}-env{env_idx}", daemon=True
        ).start()
        return episode

    def _execute(self, episode: ProgramEpisode) -> None:
        try:
            self._program(episode)
            LOGGER.info(f"{self._name} env={episode.env_idx}: program finished at step {episode.step}")
        except EpisodeOver:
            LOGGER.info(f"{self._name} env={episode.env_idx}: episode ended at step {episode.step}")
        except Exception as error:  # noqa: BLE001 -- one dead program must not stop the others
            episode.record({"type": "error", "step": episode.step, "error": repr(error)})
            LOGGER.exception(f"{self._name} env={episode.env_idx}: program died: {error!r}")
        finally:
            episode.release()


def _hold(observation: RobotObservation) -> ManipulatorAction:
    pose = np.asarray(observation.state.eef_pose, dtype=np.float32)
    gripper = float(np.asarray(observation.state.gripper_state).reshape(-1)[0])
    return ManipulatorAction(
        arm_action=EndEffectorPoseAction(translation=pose[:3, 3].copy(), rotation=pose[:3, :3].copy()),
        gripper_action=1.0 if gripper > 0.5 else 0.0,
    )
