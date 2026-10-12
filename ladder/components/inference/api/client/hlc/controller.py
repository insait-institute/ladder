from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from ladder.components.inference.api.client.hlc.base import Decomposition, SubtaskAdvancer, SubtaskSource
from ladder.components.inference.api.client.policy import Policy
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import AutoConfig, AutoConfigurable, Config, Configurable

LOGGER = get_logger(__name__)


class SubtaskControllerConfig(Config):
    source: AutoConfig  # WHERE subtasks come from (fixed plan or on-the-fly generation)
    advancer: AutoConfig  # WHEN to switch to the next subtask (clock, gripper, verifier, oracle)
    # Decompose again at every episode start rather than reusing the task's plan. A benchmark wants
    # the cache: one plan per scene keeps the comparison between systems about execution, and costs
    # one call. Repeated trials of one task want it off, or every trial replays the first plan and
    # the planner's own variance is invisible.
    replan_each_episode: bool = False


class SubtaskController(Configurable[SubtaskControllerConfig]):
    """The high-level controller: one language command per env, composed from two YAML-selected,
    independently swappable pieces: a :class:`SubtaskSource` (where subtasks come from) and a
    :class:`SubtaskAdvancer` (when to switch). Every source pairs with every advancer, so all
    combinations are configs of this one class. Per-env state is keyed by ``env_idx`` (K>1 safe)."""

    def __init__(self, config: SubtaskControllerConfig) -> None:
        super().__init__(config)
        self._source: SubtaskSource = AutoConfigurable(config.source)
        self._advancer: SubtaskAdvancer = AutoConfigurable(config.advancer)
        self._plans: Dict[
            Tuple[Optional[str], str], Decomposition
        ] = {}  # plan per (env_name, task), shared across envs+episodes
        self._tasks: Dict[int, str] = {}
        self._idx: Dict[int, int] = {}
        self._commands: Dict[int, str] = {}
        # Advancer decisions may block on HTTPS (verifier calls); fan them out so the sim batch does
        # not serialize on the VLM (per-env advancer state is disjoint).
        self._pool = ThreadPoolExecutor(max_workers=12)

    def reset(self, task_description: str, env_idx: int = 0, **kwargs) -> None:
        """Called once per env at episode start; scene plans are kept unless replan_each_episode."""
        del kwargs
        if self.config.replan_each_episode:
            # Only this task's plans: a sibling env mid-episode on another task keeps its own.
            self._plans = {key: plan for key, plan in self._plans.items() if key[1] != task_description}
        self._tasks[env_idx] = task_description
        self._idx[env_idx] = 0
        self._commands.pop(env_idx, None)
        self._advancer.reset(env_idx)
        self._source.reset(env_idx)

    def bind_policy(self, policy: Policy) -> None:
        """Offered the LL policy once so same-server pieces can reuse its client."""
        self._source.bind_policy(policy)
        self._advancer.bind_policy(policy)

    def plan_for(self, scene: Optional[str], env_idx: Optional[int] = None) -> List[str]:
        """This scene's decomposition, or empty when there is none.

        Only a planning system has one: a stepwise source generates each subtask on demand and never
        holds a plan, so it reports nothing here rather than an invented one. Recorded per episode so
        a plan can be scored against the env's goals independently of how far execution got.
        """
        del env_idx  # this system's plan is per scene+task; a per-episode planner keys on the env
        decomposition = self._plans.get((scene, self._task(0)))
        return list(decomposition.subtasks) if decomposition is not None else []

    def pop_events(self) -> Dict[int, List[Dict[str, object]]]:
        """Drain the advancer's per-env event channel into the step metadata (empty when none)."""
        events = dict(self._advancer.pending_events)
        self._advancer.pending_events.clear()
        return events

    def get_commands(self, observations: List[RobotObservation], step: int) -> List[str]:
        """One command per env: the advancer picks each env's subtask index; envs whose index moved
        (or with no command yet) get theirs from the source in one batched call; the rest reuse."""
        if not observations:
            return []
        # Resolve each plan once, keyed by (observed scene, task): a multi-scene session never
        # serves another scene's plan, and a real rig (which publishes no env_name and gets a
        # newly typed task each episode) never serves a previous task's plan.
        scene = (observations[0].metadata or {}).get("env_name")
        plan_key = (scene, self._task(0))
        decomposition = self._plans.get(plan_key)
        if decomposition is None:
            decomposition = self._source.decompose(observations[0], self._task(0))
            if decomposition is not None:
                self._plans[plan_key] = decomposition
        live = [i for i, obs in enumerate(observations) if not self._episode_over(obs)]
        new_indices = [self._idx.get(i, 0) for i in range(len(observations))]
        for env_idx, index in zip(
            live,
            self._pool.map(
                lambda i: self._advancer.next_index(observations[i], i, self._idx.get(i, 0), decomposition),
                live,
            ),
            strict=True,
        ):
            new_indices[env_idx] = index
        miss = [
            env_idx
            for env_idx in live
            if new_indices[env_idx] != self._idx.get(env_idx, 0) or not self._commands.get(env_idx)
        ]
        if miss:
            commands = self._source.commands(
                [observations[i] for i in miss],
                [self._task(i) for i in miss],
                miss,
                [new_indices[i] for i in miss],
                step,
                decomposition,
            )
            for env_idx, command in zip(miss, commands, strict=True):
                if not command.strip(" .") and self._commands.get(env_idx, "").strip(" ."):
                    command = self._commands[env_idx]  # a blank generation never overwrites a real subtask
                if new_indices[env_idx] != self._idx.get(env_idx, 0):
                    previous = self._idx.get(env_idx, 0)
                    # ``gt`` is the env's own state for the goal being LEFT, the same convention the
                    # verify event uses: switching away from an unachieved goal is a premature
                    # advance, and the step stamp the runner adds makes the switch lag measurable.
                    achieved = (observations[env_idx].metadata or {}).get("oracle", {}).get("achieved")
                    self._advancer.pending_events.setdefault(env_idx, []).append(
                        {
                            "kind": "advance",
                            "from": previous,
                            "subtask": new_indices[env_idx],
                            "gt": bool(achieved[previous])
                            if isinstance(achieved, list) and previous < len(achieved)
                            else None,
                        }
                    )
                    LOGGER.info(
                        f"SubtaskController env={env_idx}: advance -> subtask "
                        f"{new_indices[env_idx]}: {command!r}"
                    )
                self._idx[env_idx] = new_indices[env_idx]
                self._commands[env_idx] = command
        return [
            self._advancer.command(env_idx, self._commands.get(env_idx, ""))
            for env_idx in range(len(observations))
        ]

    @staticmethod
    def _episode_over(observation: RobotObservation) -> bool:
        """An env whose episode has ended gets no decision and no VLM call; it keeps its last command.
        The runner steps a batch in lockstep until its last env finishes, so without this a finished
        env keeps spending checks on an episode already scored."""
        metadata = observation.metadata or {}
        return bool(metadata.get("truncated", False)) or bool(metadata.get("done", False))

    def _task(self, env_idx: int) -> str:
        if env_idx not in self._tasks:
            raise RuntimeError(f"reset(env_idx={env_idx}) was not called before get_commands")
        return self._tasks[env_idx]
