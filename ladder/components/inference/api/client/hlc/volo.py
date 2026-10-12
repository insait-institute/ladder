import collections
import dataclasses
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Deque, Dict, List, Optional

import numpy as np
from vlm_orchestrator.strategies.base import SessionState, StrategyContext
from vlm_orchestrator.strategies.next_goal import NextGoalConfig, NextGoalStrategy
from vlm_orchestrator.strategies.passthrough import PassthroughStrategy
from vlm_orchestrator.strategies.subgoal import SubgoalConfig, SubgoalStrategy

from ladder.components.inference.api.client.hlc.cost_track import extract_usage, log_call
from ladder.components.inference.api.client.hlc.vlm.base import VLM
from ladder.components.inference.api.client.policy import Policy
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import AutoConfig, AutoConfigurable, Config, Configurable

LOGGER = get_logger(__name__)
MODES = ("subgoal", "next_goal", "passthrough")
# Upstream's other recovery modes drive its grasp/place tool executor, which needs a GraspGen
# server we deliberately do not run; without one the tool call fails as a swallowed WARNING and
# the episode scores as if recovery had happened.
RECOVERY_MODES = ("replan", None)


@dataclasses.dataclass
class _EnvState:
    task: str
    strategy: Any
    session: SessionState
    command: Optional[str] = None
    plan: List[str] = dataclasses.field(default_factory=list)
    subgoal_idx: int = 0
    finished: bool = False
    events: Deque[Dict[str, object]] = dataclasses.field(default_factory=collections.deque)


class VoloOrchestratorConfig(Config):
    vlm: AutoConfig

    # Upstream SubgoalConfig. 512 truncates a reply mid-JSON on our gateway models.
    check_mode: str = "vlm"  # "vlm" | "timer" | "hybrid"
    subgoal_timeout: int = 9999
    vlm_temperature: float = 0.0
    vlm_max_tokens: int = 8192

    # A benchmark whose episodes are shorter than this gets no check at all; see configs/.
    check_interval: int = 80

    # "subgoal" decomposes upfront, "next_goal" predicts the next instruction every check_interval,
    # "passthrough" is a no-op. Upstream's "tool_chain" is absent: it manipulates off ground-truth
    # simulator state, so it is a different system rather than a different orchestrator.
    mode: str = "subgoal"

    # subgoal only. `signal_*` needs proprioception and `gt_*` needs ground truth; neither is here.
    failure_monitor: Optional[str] = "vlm"
    recovery_mode: Optional[str] = "replan"
    prompt_style: str = "default"

    # Name a second camera only where the env renders one; RobotArena's `wrist` key is zeros.
    primary_camera: str = "main"
    extra_cameras: List[str] = []

    stop_on_success: bool = True  # must track the runner's terminate_on_success
    max_workers: int = 32

    def __post_init__(self):
        super().__post_init__()
        # Upstream's rules (cli.py). Our configs are the only entry point, so its argparse
        # validation never runs and an ignored knob here would read as active.
        assert self.mode in MODES, f"mode must be one of {MODES}, got {self.mode!r}"
        if self.mode != "subgoal":
            defaults = VoloOrchestratorConfig.failure_monitor, VoloOrchestratorConfig.recovery_mode
            assert (self.failure_monitor, self.recovery_mode) == defaults, (
                f"failure_monitor / recovery_mode apply to mode='subgoal' only, but mode="
                f"{self.mode!r} sets ({self.failure_monitor!r}, {self.recovery_mode!r})"
            )
        assert not (
            self.failure_monitor is None and self.recovery_mode is not None
        ), f"recovery_mode={self.recovery_mode!r} needs a failure_monitor"
        assert self.recovery_mode in RECOVERY_MODES, (
            f"recovery_mode must be one of {RECOVERY_MODES}, got {self.recovery_mode!r}: the rest "
            f"drive upstream's grasp/place tools, which this integration excludes"
        )


class VoloOrchestrator(Configurable[VoloOrchestratorConfig]):
    """VoLo (NVlabs/VoLoAgent) driven in-process.

    Upstream ships a websocket proxy, but its decision-making is `strategy.process(obs, state)` --
    a synchronous call returning the observation with the prompt possibly replaced. Calling that
    directly answers the orchestrator contract in `base.py`, so VoLo is scored by the same recorder
    as our own systems.
    """

    def __init__(self, config: VoloOrchestratorConfig) -> None:
        super().__init__(config)
        self._vlm: VLM = AutoConfigurable(config.vlm)
        self._states: Dict[int, _EnvState] = {}
        self._pool = ThreadPoolExecutor(max_workers=config.max_workers)
        # The context names the observation keys, so the dict we write and the strategy that reads
        # it cannot drift. `vlm` is unused: only the optional rewrite strategies call it.
        self._context = StrategyContext(vlm=None, prompt_style=config.prompt_style)
        self._strategy_cls = None

    def reset(self, task_description: str, env_idx: int = 0, **kwargs) -> None:
        del kwargs
        # Per-env strategy and session: upstream keys a session to a connection, and both carry
        # episode state (recycle count, failure retries) a batch must not share.
        self._states[env_idx] = _EnvState(
            task=task_description, strategy=self._build_strategy(), session=SessionState()
        )

    def bind_policy(self, policy: Policy) -> None:
        # The backbone may need the policy: a served VLM is handed its client here.
        self._vlm.bind_policy(policy)
        if self.config.mode != "passthrough":
            self._vlm.warmup()

    def plan_for(self, scene: Optional[str], env_idx: Optional[int] = None) -> List[str]:
        # Keyed by env, not scene: VoLo decomposes from the first frame, so two episodes of one
        # scene can be planned differently.
        del scene
        state = self._states.get(env_idx) if env_idx is not None else None
        return list(state.plan) if state is not None else []

    def pop_events(self) -> Dict[int, List[Dict[str, object]]]:
        events: Dict[int, List[Dict[str, object]]] = {}
        for env_idx, state in self._states.items():
            drained = []
            while state.events:  # popleft is atomic, so a worker may append while we drain
                drained.append(state.events.popleft())
            if drained:
                events[env_idx] = drained
        return events

    def get_commands(self, observations: List[RobotObservation], step: int) -> List[str]:
        # Decides synchronously, as SubtaskController does: the runner advances the simulator once
        # per call, so a decision resolved here costs wall-clock but zero sim steps. Envs fan out
        # across the pool, so a K-batch pays one decision's latency rather than K.
        if not observations:
            return []
        running: List[int] = []
        for env_idx, observation in enumerate(observations):
            state = self._state(env_idx)
            if self._episode_over(observation):
                state.finished = True  # the runner keeps calling frozen lanes until the batch ends
            if not state.finished:
                running.append(env_idx)

        if running:
            list(self._pool.map(lambda env_idx: self._advance(env_idx, observations[env_idx], step), running))
        return [
            self._state(env_idx).command or self._state(env_idx).task for env_idx in range(len(observations))
        ]

    def _advance(self, env_idx: int, observation: RobotObservation, step: int) -> None:
        state = self._state(env_idx)
        state.session.episode_step = step
        obs = self._observation(observation, state.task)
        obs, session = state.strategy.process(obs, state.session)
        state.session = session
        session.infer_count += 1  # upstream's proxy kept this; its failure handler logs against it
        self._record(state, env_idx, obs.get(self._context.prompt_key) or state.task, observation)

    def _observation(self, observation: RobotObservation, prompt: str) -> Dict[str, Any]:
        # The prompt is always the episode TASK, never the subgoal VoLo last chose: upstream reads a
        # prompt it did not choose as a new episode, so feeding its own subgoal back would restart
        # the episode every step.
        obs: Dict[str, Any] = {self._context.prompt_key: prompt}
        images = observation.images or {}
        primary = images.get(self.config.primary_camera)
        if primary is not None:
            obs[self._context.image_key] = np.asarray(primary)
        # strict=False: a benchmark may name fewer cameras than upstream has keys for.
        for key, name in zip(self._context.extra_image_keys, self.config.extra_cameras, strict=False):
            if name in images:
                obs[key] = np.asarray(images[name])
        return obs

    def _record(self, state: _EnvState, env_idx: int, command: str, observation: RobotObservation) -> None:
        # Publishes on the channel the runner records; a system reporting nothing here is recorded
        # as a flat run.
        plan = [str(subgoal) for subgoal in (state.session.subgoals or [])]
        if plan and plan != state.plan:
            state.plan = plan
            state.events.append({"kind": "plan", "subtasks": plan})
            LOGGER.info(f"VoloOrchestrator env={env_idx}: plan {plan}")

        index = int(state.session.current_subgoal_idx)
        if index != state.subgoal_idx:
            # `gt` is the env's state for the goal being LEFT, matching the controller's advance
            # event: leaving an unachieved goal is a premature advance.
            achieved = (observation.metadata or {}).get("oracle", {}).get("achieved")
            previous = state.subgoal_idx
            state.events.append(
                {
                    "kind": "advance",
                    "from": previous,
                    "subtask": index,
                    "gt": bool(achieved[previous])
                    if isinstance(achieved, list) and previous < len(achieved)
                    else None,
                }
            )
            state.subgoal_idx = index
            LOGGER.info(f"VoloOrchestrator env={env_idx}: advance -> subtask {index}: {command!r}")
        elif command != state.command and state.command is not None:
            state.events.append({"kind": "refine", "subtask": index, "command": command})
            LOGGER.info(f"VoloOrchestrator env={env_idx}: refine subtask {index}: {command!r}")
        state.command = command

    def _episode_over(self, observation: RobotObservation) -> bool:
        metadata = observation.metadata or {}
        if bool(metadata.get("truncated", False)):
            return True
        return self.config.stop_on_success and bool(metadata.get("done", False))

    def _state(self, env_idx: int) -> _EnvState:
        state = self._states.get(env_idx)
        if state is None:
            raise RuntimeError(f"reset(env_idx={env_idx}) was not called before get_commands")
        return state

    def _build_strategy(self):
        # Before the client: passthrough never calls a VLM, so it must not require credentials.
        if self.config.mode == "passthrough":
            return PassthroughStrategy(self._context)

        if self._strategy_cls is None:
            # Upstream resolves its own client from --vlm-base-url / VLM_API_KEY; overriding that
            # property is the whole of the change, so prompts and control flow stay theirs. Built
            # once -- reset() runs per episode per env, and each would mint another class.
            client = _metered(self._vlm)
            base = NextGoalStrategy if self.config.mode == "next_goal" else SubgoalStrategy
            self._strategy_cls = type(base.__name__, (base,), {"client": property(lambda _: client)})

        if self.config.mode == "next_goal":
            return self._strategy_cls(
                self._context,
                NextGoalConfig(
                    vlm_model=_model_name(self.config.vlm),
                    vlm_temperature=self.config.vlm_temperature,
                    vlm_max_tokens=self.config.vlm_max_tokens,
                    check_interval=self.config.check_interval,
                ),
            )

        return self._strategy_cls(
            self._context,
            SubgoalConfig(
                check_mode=self.config.check_mode,
                check_interval=self.config.check_interval,
                subgoal_timeout=self.config.subgoal_timeout,
                vlm_model=_model_name(self.config.vlm),
                vlm_temperature=self.config.vlm_temperature,
                vlm_max_tokens=self.config.vlm_max_tokens,
            ),
            failure_monitor=self.config.failure_monitor,
            recovery_mode=self.config.recovery_mode,
        )


def _metered(vlm: VLM):
    # VoLo issues its own requests; routing them through request() keeps pricing and pacing.
    def create(**kwargs):
        try:
            response = vlm.request(**kwargs)
        except Exception as error:
            # Upstream catches Exception and degrades to a one-element plan, scoring as flat.
            raise SystemExit(f"VoLo VLM request failed: {error}") from error
        log_call(model=kwargs["model"], **extract_usage(response))
        return response

    completions = types.SimpleNamespace(create=create)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))


def _model_name(vlm_config: Any) -> str:
    """What to call the backbone on the wire; a served backbone has no model id."""
    return getattr(vlm_config, "model", "served")
