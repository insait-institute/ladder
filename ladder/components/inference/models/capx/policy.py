import dataclasses
import functools
import io
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ladder.components.inference.api.client.hlc.vlm.base import VLM
from ladder.components.inference.api.client.program import EpisodeOver, ProgramEpisode, ProgramRunner
from ladder.components.inference.api.core.action import RobotAction
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.components.inference.models.capx.api.tiered import CapxApi, Motion
from ladder.components.inference.models.capx.perception.rgbd import RgbdScene
from ladder.components.inference.models.capx.perception.sam3 import Sam3Segmenter
from ladder.components.inference.models.capx.perception.scene import Scene
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import AutoConfig, AutoConfigurable, Config, Configurable

LOGGER = get_logger(__name__)

_SYSTEM_PROMPT = "You are a helpful assistant that generates Python code to directly solve the task."

_TASK_PROMPT = (
    "You are controlling a {robot} with API described below.\n"
    "Goal: {task}\n"
    "You may write python code comments for reasoning but ONLY write the executable Python code and "
    "do not write it in code fences.\n"
    "After this code executes, you will be able to get a new observation of the environment and "
    "write new code to complete the task if needed.\n"
    "The functions (APIs) below are already imported to the environment. If you want to use numpy, "
    "you need to import it explicitly.\n"
)

_MULTI_TURN_PROMPT = (
    "The following code blocks were just executed:\n"
    "```python\n{executed_code}\n```\n"
    "The current console stdout is:\n"
    "```\n{console_stdout}\n```\n"
    "The current console stderr is:\n"
    "```\n{console_stderr}\n```\n"
    "If there were no errors during compilation or execution, the console stderr will be empty. If "
    "there were errors, consider regenerating the code.\n"
    "Please respond with EXACTLY ONE of the following:\n"
    "- The word 'REGENERATE' followed immediately by new Python code in a fenced code block "
    "(```python...```) if you want to modify the code.\n"
    "- The word 'FINISH' if the task appears to be complete\n"
)

_DIFFERENCE_HEADER = (
    "Included below is the observed differences between the current state of the environment (afer "
    "the code above was executed) and the previous state of the environment (before the code above "
    "was executed):"
)
_DECISION_SUFFIX = (
    "Based on the code output, potential error messages, and the observation made above, carefully "
    "reason about the following:\n"
    "Please respond with EXACTLY ONE of the following:\n"
    "- The word 'REGENERATE' followed immediately by new Python code in a fenced code block "
    "(```python...```) if you want to modify the code.\n"
    "- The word 'FINISH' if the task appears to be complete"
)

_DESCRIBE_SYSTEM = (
    "You are a helpful assistant that describes the initial state of the environment with the goal of "
    "the task in mind. You should try to provide objective information and no assumptions. Do *NOT* "
    "write any code."
)
_DESCRIBE_INSTRUCTION = (
    "Describe the initial state of the environment with the goal of the task in mind. You should try "
    "to provide objective information and no assumptions. Do *NOT* write any code."
)

_DIFFERENCE_SYSTEM = (
    "You are a helpful assistant that describes the difference between the current state of the "
    "environment and the previous state of the environment with the goal of the task in mind and "
    "whether the task has been completed. You should try to provide objective information and no "
    "assumptions. Do *NOT* write any code."
)
_DIFFERENCE_INSTRUCTION = (
    "Describe the difference between the current state of the environment and the previous state of "
    "the environment with the goal of the task in mind and whether the task has been completed. You "
    "should try to provide objective information and no assumptions. Do *NOT* write any code.."
)

Views = List[Tuple[str, np.ndarray]]


def extract_code(content: str) -> str:
    fence_start, fence_end = "```python\n", "```"
    if fence_start in content:
        content = content[content.find(fence_start) + len(fence_start) :]
    if fence_end in content:
        content = content[: content.rfind(fence_end)]
    return content.strip()


def parse_decision(content: str) -> Tuple[str, str]:
    if content is not None and "REGENERATE" in content:
        return "regenerate", content.split("REGENERATE")[1].strip()
    return "finish", content or ""


class CapxPolicyConfig(Config):
    vlm: AutoConfig
    robot: str = "robot arm"
    max_turns: int = 10
    image_differencing: bool = True
    program_timeout_s: float = 1000.0
    max_tokens: int = 20480
    temperature: float = 1.0
    primary_camera: str = "main"
    extra_cameras: List[str] = []
    stop_on_success: bool = True
    oracle_code: Optional[str] = None
    scene: str = "camera"
    depth_camera: str = "main"
    support_z_m: Optional[float] = None
    tool_offset_m: float = 0.0
    max_translation_per_step_m: float = 0.025
    move_steps: int = 25
    arrive_tolerance_m: float = 0.01
    arrive_tolerance_rad: float = 0.15
    open_gripper_steps: int = 10
    close_gripper_steps: int = 15
    align_grasp_yaw: bool = False
    workspace_min_m: Optional[List[float]] = None
    workspace_max_m: Optional[List[float]] = None
    orientation_fallbacks: bool = True
    settle_tolerance_m: float = 0.02
    blocking_waypoint: bool = True
    segment_device: str = "cuda"
    box_fit: str = "obb"
    depth_range_m: Optional[List[float]] = [0.2, 1.8]
    reachability_clamp: bool = False


class CapxPolicy(Configurable[CapxPolicyConfig]):
    def __init__(self, config: CapxPolicyConfig) -> None:
        super().__init__(config)
        self._vlm: VLM = AutoConfigurable(config.vlm)
        self._vlm.warmup()
        fields = {f.name for f in dataclasses.fields(Motion)}
        self._motion = Motion(**{name: getattr(config, name) for name in fields})
        self._sam3 = Sam3Segmenter(device=config.segment_device)
        self._sam3.warmup()
        self._programs = ProgramRunner(self._run, "capx", config.program_timeout_s, self._episode_over)
        self._doc_logged = False

    def reset(self) -> None:
        self._programs.reset()

    def predict_actions(
        self,
        observations: List[RobotObservation],
        instructions: List[str],
        env_indices: Optional[List[int]] = None,
    ) -> List[RobotAction]:
        return self._programs.actions(observations, instructions, env_indices)

    def _episode_over(self, observation: RobotObservation) -> bool:
        metadata = observation.metadata or {}
        if bool(metadata.get("truncated", False)):
            return True
        return self.config.stop_on_success and bool(metadata.get("done", False))

    def _run(self, episode: ProgramEpisode) -> None:
        reason = self._trial(episode)
        LOGGER.info(f"CapxPolicy env={episode.env_idx}: program ended ({reason}) at step {episode.step}")

    def _trial(self, episode: ProgramEpisode) -> str:
        api = self._api(episode)
        if not self._doc_logged:
            self._doc_logged = True
            LOGGER.info(f"CapxPolicy API:\n{api.combined_doc()}")
        namespace: Dict[str, Any] = {"__name__": "__main__", **api.functions()}
        views = self._views(episode)
        task = (
            _TASK_PROMPT.format(robot=self.config.robot, task=episode.task)
            + "\nAPIs:\n\n"
            + api.combined_doc()
        )
        if self.config.oracle_code:
            prompt, code_blocks = [], [Path(self.config.oracle_code).read_text()]
        else:
            prompt = self._initial_prompt(task, views)
            code_blocks = [extract_code(self._ask(prompt))]
        for turn in range(self.config.max_turns + 1):
            code = code_blocks[-1]
            LOGGER.info(f"CapxPolicy env={episode.env_idx}: turn {turn} program:\n{code}")
            stdout = io.StringIO()
            namespace["print"] = functools.partial(print, file=stdout)
            stderr = ""
            calls = len(api.calls)
            reports = len(api.reports)
            try:
                exec(code, namespace)
            except EpisodeOver:
                return "episode over"
            except BaseException:  # noqa: BLE001 -- this is model-written code; nothing it
                # raises, SystemExit included, may escape into the runner
                stderr = traceback.format_exc()
            LOGGER.info(
                f"CapxPolicy env={episode.env_idx}: turn {turn} ran {len(api.calls) - calls} calls: "
                + "; ".join(api.calls[calls:])
            )
            episode.record(
                {
                    "kind": "tool",
                    "tool": "program",
                    "args": "; ".join(api.calls[calls:]),
                    "report": "; ".join(api.reports[reports:]),
                }
            )
            if episode.done:
                return "episode over"
            if turn == self.config.max_turns or self.config.oracle_code:
                return "max turns"
            before, views = views, self._views(episode)
            feedback = self._difference(task, before, views) if self.config.image_differencing else ""
            console = stdout.getvalue() + "".join(f"{r}\n" for r in api.reports[reports:])
            decision, reply = self._decide(prompt, "\n".join(code_blocks), console, stderr, feedback)
            LOGGER.info(
                f"CapxPolicy env={episode.env_idx}: turn {turn} -> {decision}\n"
                f"stdout: {console[-1000:]}\nstderr: {stderr[-1000:]}\nfeedback: {feedback[-1500:]}"
            )
            if decision == "finish":
                return "finish"
            code_blocks.append(extract_code(reply))
        return "max turns"

    def pop_events(self) -> Dict[int, List[Dict[str, object]]]:
        """The program's tool calls and executor reports since the last drain, by env, for the
        runner's episode event channel."""
        return self._programs.pop_events()

    def _api(self, episode: ProgramEpisode):
        return CapxApi(episode, self._motion, self._scene(episode), name=f"env={episode.env_idx}")

    def _scene(self, episode: ProgramEpisode) -> Scene:
        def observe():
            return episode.observation

        if self.config.scene != "camera":
            raise ValueError(f'unknown capx scene {self.config.scene!r}; use "camera"')
        return RgbdScene(
            observe,
            self._sam3,
            camera=self.config.depth_camera,
            segmenter=self._sam3,
            support_z_m=self.config.support_z_m,
            box_fit=self.config.box_fit,
            depth_range_m=self.config.depth_range_m,
        )

    def _views(self, episode: ProgramEpisode) -> Views:
        images = episode.observation.images or {}
        names = [self.config.primary_camera, *self.config.extra_cameras]
        return [(name, np.asarray(images[name])) for name in names if images.get(name) is not None]

    def _ask(self, messages: List[Dict[str, Any]]) -> str:
        return self._vlm.complete(messages, self.config.max_tokens, self.config.temperature)

    def _initial_prompt(self, task: str, views: Views) -> List[Dict[str, Any]]:
        text = task
        description = self._describe(task, views) if self.config.image_differencing else ""
        if description:
            text += f"\n\nThe initial state of the environment is described as follows:\n{description}"
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]

    def _describe(self, task: str, views: Views) -> str:
        if not views:
            return ""
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": task},
            {"type": "text", "text": _DESCRIBE_INSTRUCTION},
        ]
        for name, image in views:
            content += [
                {"type": "text", "text": f"{name.capitalize()} camera view:"},
                {"type": "image", "image": image},
            ]
        return self._ask(
            [{"role": "system", "content": _DESCRIBE_SYSTEM}, {"role": "user", "content": content}]
        )

    def _difference(self, task: str, before: Views, after: Views) -> str:
        if not before or not after:
            return ""
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": task},
            {"type": "text", "text": _DIFFERENCE_INSTRUCTION},
        ]
        for (name, previous), (_, current) in zip(before, after, strict=False):
            content += [
                {"type": "text", "text": f"Previous state ({name} camera):"},
                {"type": "image", "image": previous},
                {"type": "text", "text": f"Current state ({name} camera):"},
                {"type": "image", "image": current},
            ]
        return self._ask(
            [{"role": "system", "content": _DIFFERENCE_SYSTEM}, {"role": "user", "content": content}]
        )

    def _decide(
        self, prompt: List[Dict[str, Any]], executed_code: str, stdout: str, stderr: str, feedback: str
    ) -> Tuple[str, str]:
        parts = [
            prompt[-1]["content"][0]["text"],
            _MULTI_TURN_PROMPT.format(
                executed_code=executed_code, console_stdout=stdout, console_stderr=stderr
            ),
        ]
        if feedback:
            parts.append(f"{_DIFFERENCE_HEADER}\n{feedback}")
        parts.append(_DECISION_SUFFIX)
        messages = [
            prompt[0],
            {"role": "user", "content": [{"type": "text", "text": "".join(p + "\n" for p in parts)}]},
        ]
        return parse_decision(self._ask(messages))
