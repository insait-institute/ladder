import dataclasses
import os
from typing import Any, Dict, List, Optional, Tuple

import filelock
import yaml

from ladder.components.inference.api.client.episode import Episode, SceneTrace, record_motion
from ladder.components.inference.api.client.hlc.base import Orchestrator
from ladder.components.inference.api.client.metrics import aggregate, merge
from ladder.components.inference.api.client.policy import Policy
from ladder.components.inference.api.client.recorder_wrapper import RecorderWrapper, RecorderWrapperConfig
from ladder.components.inference.api.client.robot_env import RobotEnv, RobotEnvConfig
from ladder.components.inference.api.core.action import RobotAction
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.core.common.logger import get_logger
from ladder.core.compute.seeder import SEEDER
from ladder.core.databib.config import AutoConfig, AutoConfigurable, Config, Configurable

LOGGER = get_logger(__name__)


def assign_block_instructions(
    n_envs: int,
    default_instruction: str,
    overrides: Optional[List[str]],
    default_block_size: int,
) -> List[str]:
    """K>=2 instruction assignment.

    The first ``default_block_size`` envs receive ``default_instruction``;
    the remaining envs follow the override assignment (block-assign if
    divisible by ``len(overrides)``, otherwise round-robin). When
    ``overrides`` is empty/None the tail also receives ``default_instruction``.
    """
    if n_envs < 2:
        raise ValueError(f"assign_block_instructions requires n_envs>=2, got {n_envs}")
    if default_block_size < 0 or default_block_size > n_envs:
        raise ValueError(f"default_block_size={default_block_size} must be in [0, n_envs={n_envs}]")

    head = [default_instruction] * default_block_size
    tail_n = n_envs - default_block_size
    if tail_n == 0 or not overrides:
        return head + [default_instruction] * tail_n
    num_paraphrases = len(overrides)
    if tail_n % num_paraphrases == 0:
        block = tail_n // num_paraphrases
        tail = [str(overrides[env_i // block]) for env_i in range(tail_n)]
    else:
        tail = [str(overrides[env_i % num_paraphrases]) for env_i in range(tail_n)]
    return head + tail


class SimplerEpisodeRunnerConfig(Config):
    env_config: RobotEnvConfig
    recorder_config: RecorderWrapperConfig
    # Per-env instruction overrides keyed by env_name.
    # n_envs=1: random sample per episode (paraphrase robustness).
    # n_envs>=2: block-assign when n_envs is divisible by len(overrides); else round-robin.
    instruction_override: Dict[str, List[str]] = {}
    # Optional high-level controller; drives the whole K-batch (per-env reset + get_commands).
    hlc_config: Optional[AutoConfig] = None
    # n_envs>=2 only: when > 0, the first ``default_block_size`` envs receive the
    # scene's default instruction; the rest follow ``instruction_override``.
    default_block_size: int = 0
    # End each env's episode the moment it first reaches success (all goals satisfied) instead of running
    # to max_episode_steps; the batch stops once every env has finished (succeeded or truncated). Success
    # is locked at that first-success step (any-step semantics preserved). Set False to run full budget.
    terminate_on_success: bool = True


class SimplerEpisodeRunner(Configurable[SimplerEpisodeRunnerConfig]):
    """Runs batched eval episodes against a sim eval server over ZMQ:
    SimplerServer or RobotArenaServer. The coupling is a per-observation metadata
    contract, not a server class: ``env_name``, ``robot_name``
    (widowx/google/franka/ur5), ``truncated`` (required, ends the episode) and
    ``done`` (success). Both servers honor it; RobotArena's SuccessChecker
    deliberately reuses SimplerEnv's ``done`` convention. A missing ``truncated``
    or unknown ``robot_name`` raises, and a missing ``done`` silently scores every
    episode as failure, so this drives sim eval, not a real robot.

    ``n_envs`` is discovered from the server's first RESET reply; the runner
    holds one ``Policy`` that takes ``n_envs`` observations and returns
    ``n_envs`` actions per step.
    """

    def __init__(self, config: SimplerEpisodeRunnerConfig, policy: Policy):
        super().__init__(config)
        self.policy = policy
        self.env: RecorderWrapper = RecorderWrapper(
            env=RobotEnv(config=self.config.env_config),
            config=self.config.recorder_config,
        )
        self.episodes: List[Episode] = []
        self.batch_count = 0
        self.hlc: Optional[Orchestrator] = AutoConfigurable(config.hlc_config) if config.hlc_config else None
        # Let a same-server HLC reuse the policy's client (no-op for own-client controllers).
        if self.hlc is not None:
            self.hlc.bind_policy(self.policy)
        self.prng = SEEDER.new_prng_np()

    def run(self) -> None:
        # A crash still saves whatever batches completed, but it must NOT exit 0: a launcher counts a
        # zero-status run as a valid result, so a swallowed crash would look like a policy that
        # failed every episode.
        try:
            while True:
                self.run_batch()
        except StopIteration:
            LOGGER.info("Episodes completed")
        except Exception as exc:
            LOGGER.exception("Episode runner crashed", exc)  # noqa: PLE1205
            self.save_stats()
            raise
        self.save_stats()

    def run_batch(self) -> None:  # noqa: C901, PLR0912, PLR0915
        self.batch_count += 1

        self.policy.reset()
        observations: List[RobotObservation] = self.env.reset_all(instruction="")
        n_envs = len(observations)

        env_name = observations[0].metadata["env_name"]
        robot_name = observations[0].metadata["robot_name"]
        if 'widowx' in robot_name:
            robot_name = 'widowx'
        elif 'google' in robot_name:
            robot_name = 'google_robot'
        elif 'franka' in robot_name:
            robot_name = 'franka'
        elif 'ur5' in robot_name:
            robot_name = 'ur5'
        elif 'aloha' in robot_name:
            robot_name = 'aloha-agilex'
        else:
            raise ValueError(f"Unknown robot name: {robot_name}")

        default_instruction = observations[0].metadata.get("instruction", "")
        per_env_instructions = self._assign_instructions(
            n_envs=n_envs,
            env_name=env_name,
            default_instruction=default_instruction,
        )
        if self.hlc is not None:
            # Per-env reset: the controller keys per-env state (task + progress) by env_idx, so one
            # instance drives the whole K-batch (get_commands returns one subtask per env). The scene's
            # default (unparaphrased) instruction is forwarded as ``canonical_instruction`` so an HLC
            # that parses task structure reads the canonical form, not a per-env paraphrase override.
            for env_idx, instruction in enumerate(per_env_instructions):
                self.hlc.reset(
                    task_description=instruction,
                    env_idx=env_idx,
                    env_name=env_name,
                    canonical_instruction=default_instruction,
                )

        unit = "episode" if n_envs == 1 else f"batch (K={n_envs})"
        LOGGER.info(f"Starting {unit} {self.batch_count}")

        per_env_done = [False] * n_envs
        per_env_finished = [
            False
        ] * n_envs  # episode over: reached success (if terminate_on_success) or truncated
        per_env_first_instruction = [""] * n_envs
        per_env_instruction_history: List[List[str]] = [[] for _ in range(n_envs)]
        per_env_num_steps = [0] * n_envs
        per_env_done_terminal = [False] * n_envs  # last recorded step's on-target state
        per_env_success_steps: List[List[int]] = [[] for _ in range(n_envs)]  # full per-step 0/1 trace
        # Composition per-goal progress, read from the ``oracle`` block every composition env
        # publishes (RobotArena's ConjunctionSuccessChecker, SimplerEnv's oracle_goal_keys,
        # RoboLab's task-spec checks). One contract, so no per-benchmark branch. Kept as change
        # points: the vector is near-constant, so a 1350-step RoboLab episode stores a handful of
        # rows instead of one per step.
        per_env_subtasks: List[List[str]] = [[] for _ in range(n_envs)]
        per_env_goal_trace: List[List[Tuple[int, List[bool]]]] = [[] for _ in range(n_envs)]
        per_env_motion_trace: List[Dict[str, List[Tuple[int, List[float]]]]] = [{} for _ in range(n_envs)]
        # Controller events (verifier verdicts, subtask advances), each stamped with its step.
        per_env_events: List[List[Dict[str, Any]]] = [[] for _ in range(n_envs)]
        # Trajectory metrics published by the env server at each env's terminal step (robolab).
        per_env_metrics: List[Optional[Dict[str, float]]] = [None] * n_envs
        # Object poses and grasp state from the privileged payload, from the reset observation on.
        per_env_scene = [SceneTrace() for _ in range(n_envs)]
        for env_i, obs in enumerate(observations):
            per_env_scene[env_i].record(0, obs.metadata)
        # Finished lanes are dropped from the policy call, skipping their (expensive) inference.
        # step_all always advances the whole batch, so we resend their last action and ignore the
        # result: the batched sim can't cheaply step a subset, so only inference is saved, not sim.
        prev_actions: List[RobotAction] = []

        step = 0
        while True:
            if self.hlc is not None:
                step_instructions = self.hlc.get_commands(observations, step)
            else:
                step_instructions = per_env_instructions

            for env_i, instruction in enumerate(step_instructions):
                if per_env_finished[env_i]:
                    continue  # frozen: episode already ended
                if step == 0:
                    per_env_first_instruction[env_i] = instruction
                hist = per_env_instruction_history[env_i]
                if not hist or hist[-1] != instruction:
                    hist.append(instruction)

            active = [env_i for env_i in range(n_envs) if not per_env_finished[env_i]]
            active_actions = self.policy.predict_actions(
                [observations[env_i] for env_i in active],
                [step_instructions[env_i] for env_i in active],
                env_indices=active,
            )
            if prev_actions:
                actions = list(prev_actions)
                for env_i, action in zip(active, active_actions, strict=True):
                    actions[env_i] = action
            else:
                actions = active_actions  # first step: every lane is active and in order
            prev_actions = actions
            step_meta = {
                "instructions": step_instructions,
                "step": step,
                "finished": list(per_env_finished),
            }
            events_by_env = dict(self.hlc.pop_events()) if self.hlc is not None else {}
            # A policy may log its own tool calls the same way (CapxPolicy.pop_events).
            policy_events = getattr(self.policy, "pop_events", None)
            if callable(policy_events):
                for env_i, evs in (policy_events() or {}).items():
                    events_by_env.setdefault(env_i, []).extend(evs)
            if events_by_env:  # finished envs keep stepping in lockstep; their episode log is closed
                stamped = {
                    env_i: [{"step": step, **e} for e in evs]
                    for env_i, evs in events_by_env.items()
                    if not per_env_finished[env_i]
                }
                step_meta["events"] = stamped
                for env_i, stamped_events in stamped.items():
                    per_env_events[env_i].extend(stamped_events)
            observations = self.env.step_all(actions, metadata=step_meta)
            step += 1

            for env_i, obs in enumerate(observations):
                if per_env_finished[env_i]:
                    continue  # frozen: episode already ended; ignore further batched steps
                step_done = bool(obs.metadata.get("done", False))
                if step_done:
                    per_env_done[env_i] = True
                per_env_done_terminal[env_i] = step_done
                per_env_success_steps[env_i].append(int(step_done))
                per_env_num_steps[env_i] = step
                metrics = obs.metadata.get("metrics")
                if metrics is not None:
                    per_env_metrics[env_i] = metrics
                record_motion(per_env_motion_trace[env_i], step, obs.state)
                per_env_scene[env_i].record(step, obs.metadata)
                oracle = obs.metadata.get("oracle")
                if oracle:
                    achieved = [bool(x) for x in oracle["achieved"]]
                    trace = per_env_goal_trace[env_i]
                    if not trace or trace[-1][1] != achieved:
                        trace.append((step, achieved))
                    per_env_subtasks[env_i] = [str(s) for s in oracle["subtasks"]]
                # End this env's episode on truncation, or on first success when terminate_on_success.
                if bool(obs.metadata["truncated"]) or (self.config.terminate_on_success and step_done):
                    per_env_finished[env_i] = True
            if all(per_env_finished):
                break

        for env_i in range(n_envs):
            self.episodes.append(
                Episode(
                    env_name=env_name,
                    robot_name=robot_name,
                    instruction=per_env_first_instruction[env_i],
                    instruction_history=per_env_instruction_history[env_i],
                    batch=self.batch_count,
                    # A single-env batch keeps the flat shape it always had (no index column).
                    batch_env_index=env_i if n_envs > 1 else None,
                    success=bool(per_env_done[env_i]),  # any step reached the goal state
                    success_terminal=bool(per_env_done_terminal[env_i]),  # the last step held it
                    success_per_step=per_env_success_steps[env_i],  # full trace -> post-hoc criterion
                    num_steps=per_env_num_steps[env_i],
                    subtasks=per_env_subtasks[env_i],
                    plan=self.hlc.plan_for(env_name, env_i) if self.hlc is not None else [],
                    goal_trace=per_env_goal_trace[env_i],
                    motion_trace=per_env_motion_trace[env_i],
                    events=per_env_events[env_i],
                    trajectory=per_env_metrics[env_i] or {},
                    **dataclasses.asdict(per_env_scene[env_i]),
                )
            )

        self.env.end_episode()
        LOGGER.info(
            f"Completed {unit} {self.batch_count} env={env_name} " f"successes={sum(per_env_done)}/{n_envs}"
        )

    def _assign_instructions(
        self,
        n_envs: int,
        env_name: str,
        default_instruction: str,
    ) -> List[str]:
        overrides = self.config.instruction_override.get(env_name)
        if n_envs == 1:
            if overrides:
                return [str(self.prng.choice(overrides))]
            return [default_instruction]
        return assign_block_instructions(
            n_envs=n_envs,
            default_instruction=default_instruction,
            overrides=overrides,
            default_block_size=self.config.default_block_size,
        )

    def save_stats(self) -> None:
        base_dir = os.path.dirname(self.config.recorder_config.output_path)
        results_path = os.path.join(base_dir, "results.yaml")
        lock_path = f"{results_path}.lock"
        lock = filelock.FileLock(lock_path, timeout=30)

        try:
            with lock:
                LOGGER.info(f"Saving stats ...")
                if os.path.exists(results_path):
                    with open(results_path, 'r') as file:
                        stats = yaml.load(file, Loader=yaml.UnsafeLoader)
                else:
                    stats = {}

                # One cell = one (robot, env). Lanes append to the same file, so a cell already
                # present is summed with this run's, not overwritten.
                by_cell: Dict[Tuple[str, str], List[Episode]] = {}
                for episode in self.episodes:
                    by_cell.setdefault((episode.robot_name, episode.env_name), []).append(episode)
                for (robot_name, env_name), episodes in by_cell.items():
                    previous = stats.setdefault(robot_name, {}).get(env_name) or {}
                    if isinstance(previous, str):
                        previous = {"success": previous}  # written before metrics were reported
                    stats[robot_name][env_name] = merge(previous, aggregate(episodes))

                with open(results_path, "w") as f:
                    yaml.dump(stats, f, sort_keys=True)

                log_path = os.path.join(base_dir, "episode_log.yaml")
                existing_log: List[Dict[str, Any]] = []
                if os.path.exists(log_path):
                    with open(log_path, 'r') as f:
                        existing_log = yaml.load(f, Loader=yaml.UnsafeLoader) or []
                existing_log.extend(episode.to_log_entry() for episode in self.episodes)
                with open(log_path, "w") as f:
                    yaml.dump(existing_log, f, default_flow_style=False, sort_keys=False)

                LOGGER.info(f"Stats saved to {results_path}")
        finally:
            if os.path.exists(lock_path):
                try:
                    os.remove(lock_path)
                except OSError as exc:
                    LOGGER.warn(f"Failed to remove lock file {lock_path}: {exc}")

    def __del__(self):
        self.close()

    def close(self):
        self.env.unwrapped.close()
