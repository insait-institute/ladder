import contextlib
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import gpt_policy.harness.models
import gpt_policy.input.request
import gpt_policy.runtime.runner
import gpt_policy.settings
import gpt_policy.tools.runtime
import numpy as np
import PIL.Image

from ladder.components.inference.api.client.program import ProgramEpisode, ProgramRunner, truncated
from ladder.components.inference.api.core.action import RobotAction
from ladder.components.inference.api.core.observation import RobotObservation
from ladder.components.inference.models.gpt_policy import protocol
from ladder.components.inference.models.gpt_policy.host import HostConfig, Localizer, SimRobot
from ladder.components.inference.models.gpt_policy.session import LadderVlmSession, Frame, VlmSessionConfig
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import AutoConfig, AutoConfigurable, Config, Configurable

LOGGER = get_logger(__name__)

_AGENT_TIMEOUT_S = 1800.0
_ARMS = ("left",)
_DOF = 6


class GptPolicyConfig(Config):
    vlm: AutoConfig
    session: VlmSessionConfig
    host: HostConfig
    cameras: List[str] = ["main", "wrist"]
    max_image_side: Optional[int] = 640
    point_grid: Optional[int] = None
    max_decisions: int = 40
    robot_name: str = "WidowX 250 6DoF"
    overload_recovery_s: float = 1800.0


class GptPolicyPolicy(Configurable[GptPolicyConfig]):
    def __init__(self, config: GptPolicyConfig) -> None:
        super().__init__(config)
        self._vlm = AutoConfigurable(config.vlm)
        self._vlm.warmup()
        self._catalog = protocol.load_catalog(config.point_grid)
        gpt_policy.runtime.runner.MODEL_RECOVERY_TIMEOUT_S = config.overload_recovery_s
        self._programs = ProgramRunner(self._run, "gpt-policy", _AGENT_TIMEOUT_S, over=_episode_over)

    def reset(self) -> None:
        self._programs.reset()

    def predict_actions(
        self,
        observations: List[RobotObservation],
        instructions: List[str],
        env_indices: Optional[List[int]] = None,
    ) -> List[RobotAction]:
        return self._programs.actions(observations, instructions, env_indices)

    def pop_events(self) -> Dict[int, List[Dict[str, object]]]:
        return self._programs.pop_events()

    def _run(self, episode: ProgramEpisode) -> None:
        robot = SimRobot(self.config.host, lambda: episode.observation, episode.act)
        cameras = _Cameras(self.config.cameras, self.config.max_image_side, lambda: episode.observation)
        grid = self.config.point_grid
        shown_shape = cameras.shown_shape if grid is None else lambda role: (grid + 1, grid + 1)
        localizer = Localizer(cameras.roles, lambda: episode.observation, shown_shape)
        session = LadderVlmSession(self.config.session, self._vlm)
        session.start(
            gpt_policy.harness.models.AgentContext(
                instructions=protocol.instructions(
                    self._catalog,
                    self.config.robot_name,
                    cameras.available(),
                    cameras.depth_roles(),
                    self.config.host.workspace_min_m,
                    self.config.host.workspace_max_m,
                    self.config.host.control_hz,
                    self.config.point_grid,
                ),
                tools=self._catalog.function_schemas(_DOF, _ARMS),
                output_schema=protocol.selection_schema(self._catalog, _DOF, _ARMS),
            )
        )
        outcome = gpt_policy.runtime.runner.run_loop(
            gpt_policy.settings.RuntimeConfig(
                robot_model=self.config.robot_name,
                interface="sim",
                right_interface="",
                gripper_open_readout=1.0,
                camera_width=0,
                camera_height=0,
                convert_camera_images_to_jpeg=True,
                camera_jpeg_quality=85,
                trajectory_hz=self.config.host.control_hz,
                task_name=episode.task,
                record_dir=None,
                max_decisions=self.config.max_decisions,
            ),
            gpt_policy.input.request.RunInput(
                instruction=episode.task, model=str(getattr(self._vlm.config, "model", "vlm"))
            ),
            robot,
            cameras,
            cameras,
            session,
            gpt_policy.tools.runtime.ToolExecutor(self._catalog, _ARMS, robot, localizer),
            _Recorder(episode.env_idx),
            display=_Display(episode.env_idx),
        )
        LOGGER.info(f"env={episode.env_idx}: gpt-policy finished '{episode.task}' -> {outcome}")


class _Cameras:
    def __init__(
        self, names: List[str], max_side: Optional[int], observation: Callable[[], RobotObservation]
    ) -> None:
        self._max_side = max_side
        self._observation = observation
        fixed = [name for name in names if "wrist" not in name]
        wrist = [name for name in names if "wrist" in name]
        self.roles: Dict[str, str] = {}
        if fixed:
            self.roles["top"] = fixed[0]
        if wrist:
            self.roles["left"] = wrist[0]

    def available(self) -> Dict[str, str]:
        images = self._observation().images or {}
        return {role: name for role, name in self.roles.items() if name in images}

    def depth_roles(self) -> Tuple[str, ...]:
        published = self._observation().cameras
        return tuple(role for role, name in self.available().items() if name in published)

    def shown_shape(self, role: str) -> Optional[Tuple[int, int]]:
        frame = self.snapshot().get(role)
        return None if frame is None else (frame.height, frame.width)

    def snapshot(self, after: float = 0.0) -> Dict[str, Frame]:
        del after
        images = self._observation().images or {}
        return {
            role: Frame(role, _shrink(images[name], self._max_side))
            for role, name in self.available().items()
        }

    def describe(self, images: Optional[Dict[str, Frame]] = None) -> List[Dict[str, Any]]:
        frames = images if images is not None else self.snapshot()
        return [
            {"name": role, "format": "rgb", "width": frame.width, "height": frame.height}
            for role, frame in frames.items()
        ]


class _Recorder:
    def __init__(self, env_idx: int) -> None:
        self._env = env_idx

    def observation(self, step: int, text: str, state: Any, cameras: Any, images: Any, **extra: Any) -> None:
        del text, state, cameras, images, extra
        LOGGER.info(f"env={self._env}: gpt-policy observation {step}")

    def write(self, kind: str, payload: Dict[str, Any]) -> None:
        if kind in ("model_decision", "terminal", "tool_error", "model_retry"):
            LOGGER.info(f"env={self._env}: gpt-policy {kind}: {payload}")
        elif kind == "execution_result" and payload.get("name") == "locate_point":
            result = payload.get("result", {})
            points = result.get("points", [])
            metric = sum(1 for p in points if p.get("metric_position_available"))
            LOGGER.info(
                f"env={self._env}: locate_point camera={result.get('camera')} "
                f"pixels={len(points)} metric={metric}/{len(points)} "
                f"xyz={[p.get('metric_position_base_xyz') for p in points]}"
            )
        elif kind == "execution_result" and "target_tcp_xyzrpy" in payload.get("result", {}).get(
            "execution_feedback", {}
        ):
            feedback = payload["result"]["execution_feedback"]
            target, measured = feedback["target_tcp_xyzrpy"], feedback["measured_tcp_xyzrpy"]
            LOGGER.info(
                f"env={self._env}: {payload.get('name')} "
                f"target_xyz={[round(v, 4) for v in target[:3]]} "
                f"measured_xyz={[round(v, 4) for v in measured[:3]]} "
                f"residual_mm={feedback['tcp_translation_error_m'] * 1000:.1f} "
                f"rot_err_rad={feedback['tcp_rotation_error_rad']:.3f} "
                f"gripper={feedback['gripper_measured_normalized']:.2f} "
                f"settled={feedback['settle']['settled']}"
            )


class _Display:
    def __init__(self, env_idx: int) -> None:
        self._env = env_idx

    @contextlib.contextmanager
    def waiting(self, message: str) -> Iterator[None]:
        del message
        yield

    def message(self, text: str, colour: str = "") -> None:
        del colour
        LOGGER.info(f"env={self._env}: gpt-policy {text}")

    def decision(self, step: int, name: str, arguments: Dict[str, Any]) -> None:
        LOGGER.info(f"env={self._env}: step {step} -> {name} {arguments.get('note', '')}")

    def error(self, step: int, name: str, error: Dict[str, Any]) -> None:
        LOGGER.warn(f"env={self._env}: step {step} {name} rejected: {error.get('error')}")

    def returned(self, step: int, name: str) -> None:
        del step, name

    def home(self, result: Any) -> None:
        del result


def _episode_over(observation: RobotObservation) -> bool:
    return truncated(observation) or bool((observation.metadata or {}).get("done", False))


def _shrink(image: np.ndarray, max_side: Optional[int]) -> np.ndarray:
    if max_side is None or max(image.shape[:2]) <= max_side:
        return image
    height, width = image.shape[:2]
    scale = max_side / float(max(height, width))
    resized = PIL.Image.fromarray(image).resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))), PIL.Image.BILINEAR
    )
    return np.asarray(resized)
