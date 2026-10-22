import time
import traceback
from typing import Any, Dict, List, Optional, TypeVar

import numpy as np

from ladder.components.inference.api.client.policy import Policy
from ladder.components.inference.api.client.recorder_wrapper import RecorderWrapper, RecorderWrapperConfig
from ladder.components.inference.api.client.robot_env import RobotEnv, RobotEnvConfig
from ladder.components.inference.api.core.action import RobotAction
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import AutoConfig, AutoConfigurable, Config, Configurable

LOGGER = get_logger(__name__)


class RobotEpisodeRunnerConfig(Config):
    env_config: RobotEnvConfig
    max_episode_steps: Optional[int] = 120  # Max number of steps to run per episode
    max_episode_secs: Optional[float] = None  # Max number of seconds to run per episode
    max_episodes: Optional[int] = None  # Max number of episodes to run. If None, run indefinitely.
    fixed_instruction: bool = False
    recorder_config: RecorderWrapperConfig
    # Hierarchical mode: a high-level controller (e.g. hlc.controller.SubtaskController) turns the
    # typed task into a per-step language command. None = the typed task drives every step.
    hlc_config: Optional[AutoConfig] = None

    def __post_init__(self):
        super().__post_init__()
        if self.max_episode_steps is None and self.max_episode_secs is None:
            raise ValueError("At least one of max_episode_steps or max_episode_secs must be specified.")


RunnerConfigT = TypeVar("RunnerConfigT", bound=RobotEpisodeRunnerConfig)


class EpisodeRunnerBase(Configurable[RunnerConfigT]):
    """The episode loop, the recorder, and the optional orchestrator. Subclasses say where the
    task comes from (``get_instruction``) and what to set up before it (``prepare_episode``)."""

    def __init__(self, config: RunnerConfigT, policy: Policy):
        super().__init__(config)
        self.policy = policy

        self.env = self.wrap_env(RobotEnv(config=self.config.env_config))

        self.instruction = ""
        self._reported_instruction: Optional[str] = None
        # Offered once so HLC pieces that share the policy's inference client can bind to it.
        self.hlc = AutoConfigurable(config.hlc_config) if config.hlc_config else None
        if self.hlc is not None:
            self.hlc.bind_policy(self.policy)

    def wrap_env(self, env: RobotEnv) -> RobotEnv:
        return RecorderWrapper(env=env, config=self.config.recorder_config)

    def run_episode(self, episode_index: int):
        self._reported_instruction = None
        input("Press Enter to reset robot")
        # Reset the robot to a neutral state. Bypass all wrappers, we only want to reset the robot.
        self.env.unwrapped.reset()

        self.instruction = self.get_instruction()

        self.prepare_episode()

        self.policy.reset()
        if self.hlc is not None:
            self.hlc.reset(task_description=self.instruction, env_idx=0)

        input(f"Press Enter to start episode {episode_index+1} ...")
        self.start_episode()
        print("Starting episode... Press Ctrl-C to terminate episode early!")

        observation: RobotObservation = self.env.reset(instruction=self.instruction)

        max_steps = self.episode_max_steps()
        max_secs = self.config.max_episode_secs or float("inf")
        steps = 0
        start_time = time.time()

        while steps < max_steps and (time.time() - start_time) < max_secs:
            try:
                instruction = self.instruction
                if self.hlc is not None:
                    instruction = self.hlc.get_commands([observation], steps)[0]
                action: RobotAction = self.policy.predict_actions([observation], [instruction])[0]

                # The recorder logs the commanded instruction per step and accumulates any events
                # the HLC recorded (e.g. subtask advances) into the episode metadata.
                step_metadata: Dict[str, Any] = {"instructions": [instruction], "step": steps}
                if self.hlc is not None:
                    events = {
                        env_i: [{"step": steps, **event} for event in env_events]
                        for env_i, env_events in self.hlc.pop_events().items()
                    }
                    if steps == 0:
                        # The plan exists once get_commands has run. It rides the recorder's generic
                        # event channel, so it lands in episode_meta.yaml with everything else.
                        plan = self.hlc.plan_for((observation.metadata or {}).get("env_name"))
                        if plan:
                            events.setdefault(0, []).insert(
                                0, {"step": 0, "kind": "plan", "subtasks": list(plan)}
                            )
                    self._report_hlc(steps, instruction, events)
                    if events:
                        step_metadata["events"] = events
                # A policy may log its own decisions the same way (HarnessPolicy.pop_events), as in sim.
                policy_events = getattr(self.policy, "pop_events", None)
                if callable(policy_events):
                    for env_i, env_events in (policy_events() or {}).items():
                        step_metadata.setdefault("events", {}).setdefault(env_i, []).extend(
                            {"step": steps, **event} for event in env_events
                        )
                observation: RobotObservation = self.env.step(action, metadata=step_metadata)

                steps += 1

            except KeyboardInterrupt:
                print("\nCaught KeyboardInterrupt: Terminating episode.")
                break
            except Exception:  # noqa: BLE001
                LOGGER.warn(f"\nCaught exception: {traceback.format_exc()}\nTerminating episode.")
                break

        self.env.end_episode()

    def episode_max_steps(self) -> float:
        """The step ceiling for the episode about to run."""
        return self.config.max_episode_steps or float("inf")

    def prepare_episode(self) -> None:
        """Set the scene up between the reset and the episode. Nothing to do by default."""

    def start_episode(self) -> None:
        """Run once the operator has confirmed, just before the first step."""

    def _report_hlc(self, step: int, instruction: str, events: Dict[int, List[Dict[str, Any]]]) -> None:
        """Print the controller's decisions as they happen, so a run is followable live."""
        flat = [event for env_events in events.values() for event in env_events]
        for event in flat:
            if event.get("kind") == "plan":
                subtasks = event.get("subtasks") or []
                LOGGER.info("plan: " + " | ".join(f"{i + 1}. {t}" for i, t in enumerate(subtasks)))
        if instruction != self._reported_instruction:
            self._reported_instruction = instruction
            LOGGER.info(f"step {step}: {instruction}")
        for event in flat:
            if event.get("kind") == "verify":
                LOGGER.info(
                    f"step {step}: verify subtask {event.get('subtask')} "
                    f"-> {event.get('verdict')}: {str(event.get('reply', ''))[:200]}"
                )

    def get_instruction(self) -> str:
        """Where this runner's task comes from."""
        raise NotImplementedError


class RobotEpisodeRunner(EpisodeRunnerBase[RobotEpisodeRunnerConfig]):
    """Runs inference on the robot in interactive mode: the operator types each task."""

    def run(self):
        max_episodes = self.config.max_episodes or np.iinfo(np.int32).max
        for episode_index in range(max_episodes):
            self.run_episode(episode_index)

    def get_instruction(self) -> str:
        """Prompt the user to input the next task."""
        instruction = self.instruction

        if self.config.fixed_instruction and instruction != "":
            return instruction

        if instruction == "":
            user_input = ""
            while user_input == "":
                user_input = input("Enter the language instruction: ")
            instruction = user_input
        else:
            user_input = input(
                "Enter the language instruction (or leave blank to repeat the previous instruction): "
            )
            if user_input == "":
                pass  # Do nothing -> Let instruction be the same
            else:
                instruction = user_input

        print(f"Instruction: {instruction}")

        return instruction
