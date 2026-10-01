import collections
import concurrent.futures
import re
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing

from ladder.components.inference.api.client.hlc.base import Decomposition, SubtaskSource
from ladder.components.inference.api.client.hlc.vlm.base import VLM
from ladder.components.inference.api.client.policy import Policy
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import AutoConfig, AutoConfigurable, Config

LOGGER = get_logger(__name__)


class VLMSourceConfig(Config):
    vlm: AutoConfig  # the backend (OpenAIVLM, LocalVLM, ...) — any VLM works
    max_tokens: int = 256
    temperature: float = 0.3
    history_length: int = 3  # past (image, command) pairs kept as rolling context
    # How many of the most recent history entries also send their image. -1 = all, 0 = text-only.
    history_image_count: int = -1
    image_camera: str = "main"
    system_prompt: str = ""
    # Current-turn user prompt ({task} placeholder) and the question echoed on history turns.
    prompt_template: str = "Task: {task}\nWhat should the robot do next?"
    history_prompt: str = "What should the robot do next?"
    resolve_references: bool = True  # resolve <object> angle-bracket refs via a second call
    resolve_vlm: Optional[AutoConfig] = None  # backend for reference resolution (default: vlm)
    reasoning_effort: Optional[str] = None  # "none"|"low"|"medium"|"high" for thinking models
    # If set, the FIRST command per env returns this (with {task} filled) verbatim, no VLM call.
    initial_command_template: str = ""
    merge_system_into_user: bool = False  # for chat templates that require strict user/assistant turns
    max_parallel_calls: int = 32  # concurrent in-flight calls when servicing a K-env batch
    prepend_task: bool = False  # emit "<task>. <subtask>" instead of replacing the task
    lowercase_command: bool = False  # subtask LLs are trained on lowercased instructions


class VLMSource(SubtaskSource[VLMSourceConfig]):
    """On-the-fly subtasks from any :class:`VLM` backend: every index move generates a fresh
    command from the current frame plus a rolling per-env history of (image, command) pairs."""

    def __init__(self, config: VLMSourceConfig):
        super().__init__(config)
        self._histories: Dict[int, collections.deque] = {}
        self._vlm: VLM = AutoConfigurable(config.vlm)
        self._resolve_vlm: VLM = (
            AutoConfigurable(config.resolve_vlm) if config.resolve_vlm is not None else self._vlm
        )

    def bind_policy(self, policy: Policy) -> None:
        self._vlm.bind_policy(policy)
        if self._resolve_vlm is not self._vlm:
            self._resolve_vlm.bind_policy(policy)

    def reset(self, env_idx: int) -> None:
        self._get_history(env_idx).clear()

    def commands(
        self,
        observations: List[RobotObservation],
        tasks: List[str],
        env_indices: List[int],
        indices: List[int],
        step: int,
        decomposition: Optional[Decomposition],
    ) -> List[str]:
        del indices, decomposition
        # Each call may block on HTTPS; dispatch concurrently so K workers don't stall for K*latency
        # (per-env state is keyed by env_idx). Pre-warm on the main thread (first-touch isn't safe).
        self._vlm.warmup()
        for env_idx in env_indices:
            self._get_history(env_idx)
        max_workers = min(self.config.max_parallel_calls, len(env_indices))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(
                executor.map(
                    lambda pack: self._generate_command(pack[0], pack[1], step, pack[2]),
                    zip(observations, tasks, env_indices, strict=True),
                )
            )

    def _generate_command(self, observation: RobotObservation, task: str, step: int, env_idx: int) -> str:
        if step == 0 and self.config.initial_command_template != "":
            command = self.config.initial_command_template.format(task=task)
            LOGGER.info(f"VLM initial (no-API) command at step 0 env_idx={env_idx}: '{command}'")
            return command

        image = self._get_image(observation)
        messages = self._build_messages(image, task, env_idx)
        response_text = self._vlm.complete(
            messages,
            self.config.max_tokens,
            self.config.temperature,
            reasoning_effort=self.config.reasoning_effort,
        )
        command = self._extract_command(response_text)

        if self.config.resolve_references and re.search(r'<[^>]+>', command):
            command = self._resolve_references(command, image)

        self._get_history(env_idx).append((image, command))
        LOGGER.info(f"VLM command at step {step} env_idx={env_idx}: '{command}'")
        return self._prepend_task(task, command)

    def _prepend_task(self, task: str, command: str) -> str:
        if not self.config.prepend_task or command == "" or task == "" or task.lower() in command.lower():
            return command
        sep = "" if task.rstrip().endswith((".", "!", "?")) else "."
        return f"{task}{sep} {command}"

    def _get_history(self, env_idx: int) -> collections.deque:
        history = self._histories.get(env_idx)
        if history is None:
            history = collections.deque(maxlen=self.config.history_length)
            self._histories[env_idx] = history
        return history

    def _get_image(self, observation: RobotObservation) -> numpy.typing.NDArray[np.uint8]:
        camera = self.config.image_camera
        if observation.images is None or camera not in observation.images:
            raise ValueError(f"Camera '{camera}' not found in observation images")
        return observation.images[camera]

    def _build_messages(
        self, current_image: numpy.typing.NDArray[np.uint8], task: str, env_idx: int
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []

        if self.config.system_prompt != "" and not self.config.merge_system_into_user:
            messages.append({"role": "system", "content": self.config.system_prompt})

        history_entries = list(self._get_history(env_idx))
        keep_n = self.config.history_image_count
        for idx, (hist_image, hist_command) in enumerate(history_entries):
            include_image = keep_n < 0 or idx >= len(history_entries) - keep_n
            content: List[Dict[str, Any]] = []
            if include_image and hist_image is not None:
                content.append({"type": "image", "image": hist_image})
            content.append({"type": "text", "text": self.config.history_prompt})
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant", "content": hist_command})

        current_text = self.config.prompt_template.format(task=task)
        if self.config.merge_system_into_user and self.config.system_prompt != "":
            current_text = f"{self.config.system_prompt}\n\n{current_text}"
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": current_image},
                    {"type": "text", "text": current_text},
                ],
            }
        )

        return messages

    def _extract_command(self, response_text: str) -> str:
        # Reasoning models emit a long trace with no clean final line, so pull the LAST imperative
        # subtask (the LL clips at 512 tokens); fall back to the last non-empty line otherwise.
        imperative = re.findall(
            r"(?:Reach for|Move to|Pick up|Grasp|Move|Place|Put|Lower|Release|Open gripper|Close gripper)"
            r"[^.\n]{0,80}\.",
            response_text,
        )
        if imperative:
            return self._maybe_lowercase(imperative[-1].strip())
        lines = [line.strip() for line in response_text.strip().split('\n') if line.strip() != ""]
        if len(lines) == 0:
            LOGGER.info(f"VLM returned no usable command from response: '{response_text[:200]}'")
            return ""
        return self._maybe_lowercase(lines[-1])

    def _maybe_lowercase(self, command: str) -> str:
        return command.lower() if self.config.lowercase_command else command

    def _resolve_references(self, command: str, image: numpy.typing.NDArray[np.uint8]) -> str:
        prompt = (
            f"The following robot command contains object references in angle brackets: '{command}'\n"
            f"Look at the image and replace each <object> reference with its pixel coordinates "
            f"in the format (x, y). Return only the updated command, nothing else."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        resolved = self._resolve_vlm.complete(messages, max_tokens=128, temperature=0.0).strip()
        LOGGER.info(f"Resolved references: '{command}' -> '{resolved or command}'")
        return resolved or command
