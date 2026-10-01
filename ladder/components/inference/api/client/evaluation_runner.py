import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import PIL.Image
import yaml

from ladder.components.inference.api.client.policy import Policy
from ladder.components.inference.api.client.robot_episode_runner import (
    EpisodeRunnerBase,
    RobotEpisodeRunnerConfig,
)
from ladder.components.inference.api.core.pub_sub import ZmqPublisher, ZmqPublisherConfig
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import AutoConfig, AutoConfigurable, Config

LOGGER = get_logger(__name__)


class _Annotation(NamedTuple):
    """What the operator recorded for one episode, under the sim composition benchmark's names."""

    num_subgoals_complete: int
    num_goals: int
    distractors_stationary: Optional[bool]

    @property
    def score(self) -> float:
        return self.num_subgoals_complete / self.num_goals

    @property
    def success(self) -> bool:
        # As in sim, a disturbed distractor voids the episode without touching the goal count.
        return self.num_subgoals_complete == self.num_goals and self.distractors_stationary is not False


class EvaluationSystemConfig(Config):
    name: str
    hlc_config: Optional[AutoConfig] = None  # None = the flat policy, no orchestrator


class EvaluationRunnerConfig(RobotEpisodeRunnerConfig):
    evaluation_configuration_file: str
    num_trials: int = 3
    # Scored back to back on each restored scene. Empty = this runner's own hlc_config, alone.
    systems: List[EvaluationSystemConfig] = []
    resume_task: int = 0
    resume_config: int = 0
    resume_trial: int = 0
    resume_system: int = 0
    results_file: str = "evaluation_scores.csv"
    configuration_pub_config: ZmqPublisherConfig = ZmqPublisherConfig(port=5560)
    # Measured on this rig: a goal costs a median 49 steps and at most 84, the first one up to 122.
    steps_per_goal: int = 70
    extra_first_goal_steps: int = 60


class EvaluationRunner(EpisodeRunnerBase[EvaluationRunnerConfig]):
    """Runs every trial of an evaluation database instead of a typed instruction.

    Each trial publishes its reference image to the configuration visualizer, so the operator can
    restore the recorded scene before the episode starts, and records the goals it completed. Being
    a RobotEpisodeRunner, it takes `hlc_config` like any other run: the same database can be scored
    flat and then under each orchestrator, or under all of them per scene via `systems`.
    """

    def __init__(self, config: EvaluationRunnerConfig, policy: Policy):
        super().__init__(config, policy)
        self.dataset_file = Path(config.evaluation_configuration_file)
        with self.dataset_file.open() as f:
            self.dataset = json.load(f)
        self.configuration_pub = ZmqPublisher(config.configuration_pub_config)
        self._images: List[str] = []
        self._prompt = ""
        self._num_goals = 1
        self._num_distractors = 0
        self._systems = self._build_systems()
        self._system = ""
        self.results_file_path = Path(config.results_file)
        self._initialize_scores_csv()
        self._save_run_config()

    def _save_run_config(self) -> None:
        """Dump the resolved config beside the results, so a run can be reproduced from them alone."""
        path = self.results_file_path.parent / "eval_config.yaml"
        with open(path, "w") as f:
            yaml.dump(self.config.as_json(), f, default_flow_style=False, sort_keys=False)

    def _build_systems(self) -> List[Tuple[str, Optional[AutoConfigurable]]]:
        systems = []
        for entry in self.config.systems:
            hlc = AutoConfigurable(entry.hlc_config) if entry.hlc_config is not None else None
            if hlc is not None:
                hlc.bind_policy(self.policy)
            systems.append((entry.name, hlc))
        return systems or [("", self.hlc)]

    def get_instruction(self) -> str:
        return self._prompt

    def episode_max_steps(self) -> int:
        """The task's goal count sets the ceiling, so `max_episode_steps` does not apply here. The
        value used is saved per episode, next to the steps the episode actually took."""
        return self.config.extra_first_goal_steps + self.config.steps_per_goal * self._num_goals

    def prepare_episode(self) -> None:
        for img_path in self._images:
            self._show_image(img_path, Path(img_path).name.split("_", 1)[0])

    def start_episode(self) -> None:
        # Drop the reference blend now the scene matches, so the episode is watched unobstructed.
        self._show_image("", "clear")

    def run(self):
        # The subscriber drops the first messages it is sent, so burn them on a no-op clear.
        self._show_image("", "clear")
        episode_index = 0
        for task_index in range(self.config.resume_task, len(self.dataset)):
            entry = self.dataset[task_index]
            self._prompt = entry["prompt"]
            self._num_goals = entry.get("num_goals", 1)
            self._num_distractors = entry.get("num_distractors", 0)
            configurations = entry["configurations"]
            LOGGER.info(
                f"=== task {task_index}: {entry['task']} ({self._prompt}), "
                f"{self._num_goals} goals, {self.episode_max_steps()} steps ==="
            )

            first_config = self.config.resume_config if task_index == self.config.resume_task else 0
            for config_index in range(first_config, len(configurations)):
                self._images = configurations[config_index].get("images", [])
                resuming = task_index == self.config.resume_task and config_index == first_config
                first_trial = self.config.resume_trial if resuming else 0
                for trial in range(first_trial, self.config.num_trials):
                    first_system = self.config.resume_system if resuming and trial == first_trial else 0
                    for name, hlc in self._systems[first_system:]:
                        self._system, self.hlc = name, hlc
                        LOGGER.info(
                            f"configuration {config_index + 1}/{len(configurations)}, "
                            f"trial {trial + 1}/{self.config.num_trials}, system {name or 'default'}"
                        )
                        while True:
                            self.run_episode(episode_index)
                            answer = input("Trial finished. Enter to continue, 'r' to redo: ")
                            if answer.strip().lower() != "r":
                                break
                            LOGGER.info("redoing trial")
                        self._save_annotation(entry["task"], config_index, trial)
                        episode_index += 1

        LOGGER.info("all tasks completed")

    def _show_image(self, img_path: str, camera_name: str) -> None:
        """Publish a reference frame to the configuration visualizer, which blends it with the live
        camera. An empty path clears the overlay."""
        message: Dict[str, Any] = {"camera_name": camera_name}
        if img_path:
            message["image"] = np.array(PIL.Image.open((self.dataset_file.parent / img_path).resolve()))
        # The publisher does not block and the subscriber may not be up yet, so repeat.
        for _ in range(10):
            self.configuration_pub.publish(message, "configuration_image")

    def _initialize_scores_csv(self) -> None:
        self.results_file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.results_file_path.exists():
            with open(self.results_file_path, 'w', newline='') as csvfile:
                csv.writer(csvfile).writerow(
                    [
                        'system',
                        'task_description',
                        'configuration_index',
                        'trial_index',
                        'score',
                        'success',
                        'num_subgoals_complete',
                        'num_goals',
                        'distractors_stationary',
                    ]
                )

    def _prompt_for_annotation(self) -> _Annotation:
        while True:
            try:
                subgoals = int(input(f"Subgoals completed (0-{self._num_goals}): ").strip())
                if 0 <= subgoals <= self._num_goals:
                    break
                print(f"Please enter a whole number between 0 and {self._num_goals}.")
            except ValueError:
                print("Please enter a valid whole number.")

        stationary = None
        if self._num_distractors:
            stationary = input("Distractors left where they were? [Y/n]: ").strip().lower() != "n"
        return _Annotation(subgoals, self._num_goals, stationary)

    def _save_annotation(self, task: str, config_index: int, trial: int) -> None:
        annotation = self._prompt_for_annotation()
        with open(self.results_file_path, 'a', newline='') as csvfile:
            csv.writer(csvfile).writerow(
                [
                    self._system,
                    task,
                    config_index,
                    trial,
                    annotation.score,
                    annotation.success,
                    annotation.num_subgoals_complete,
                    annotation.num_goals,
                    annotation.distractors_stationary,
                ]
            )
        self._save_annotation_to_episode(annotation)
        LOGGER.info(f"scored {annotation.score} for '{task}' configuration {config_index} trial {trial}")

    def _save_annotation_to_episode(self, annotation: _Annotation) -> None:
        """Write the annotation into the episode just recorded."""
        paths = self.env.episode_paths
        if not paths:
            return
        path = os.path.join(paths[0], "episode_meta.yaml")
        with open(path) as f:
            metadata = yaml.safe_load(f) or {}
        metadata.update(
            system=self._system,
            max_episode_steps=self.episode_max_steps(),
            score=annotation.score,
            success=annotation.success,
            num_subgoals_complete=annotation.num_subgoals_complete,
            num_goals=annotation.num_goals,
            distractors_stationary=annotation.distractors_stationary,
        )
        with open(path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
