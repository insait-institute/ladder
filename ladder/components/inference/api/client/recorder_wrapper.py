import datetime
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, TypedDict

import yaml

from ladder.components.inference.api.client.env_wrapper import EnvWrapper, EnvWrapperConfig
from ladder.components.inference.api.client.rerun_writer import RerunWriter
from ladder.components.inference.api.client.robot_env import RobotEnv
from ladder.components.inference.api.client.video_writer import VideoWriterOpenCV
from ladder.components.inference.api.core.action import RobotAction
from ladder.components.inference.api.core.observation import RobotObservation


class InstructionLogEntry(TypedDict):
    step: int
    instruction: str


class RecorderWrapperConfig(EnvWrapperConfig):
    output_path: str
    fps: int = 5
    rerun_enabled: bool = True
    video_enabled: bool = True
    # Per-env video-encode thread-pool size; None => CPUs allocated to this process.
    video_encode_workers: Optional[int] = None


class RecorderWrapper(EnvWrapper[RecorderWrapperConfig]):
    """Records video, metadata, and per-env instruction history per episode."""

    def __init__(self, config: RecorderWrapperConfig, env: RobotEnv | EnvWrapper):
        super().__init__(config, env)
        self._n_envs: int = 1
        self._episode_indices: List[int] = []
        self._episode_paths: List[str] = []
        self._step_counts: List[int] = []
        self._last_obs: List[Optional[RobotObservation]] = []
        self._writers: List[Dict[str, VideoWriterOpenCV]] = []
        self._instructions: List[str] = []
        self._instruction_logs: List[List[InstructionLogEntry]] = []
        self._first_obs_metadata: List[Dict[str, Any]] = []
        self._events: Dict[int, List[Dict[str, Any]]] = {}
        self._rerun_writers: List[RerunWriter] = []

        os.makedirs(self.config.output_path, exist_ok=True)

    def reset(self, instruction: str = "", **kwargs) -> RobotObservation:  # type: ignore[override]
        observations = self.reset_all(instruction=instruction, **kwargs)
        self._require_single_env("reset")
        return observations[0]

    def step(
        self, action: RobotAction, metadata: Dict[str, Any] | None = None
    ) -> RobotObservation:  # type: ignore[override]
        self._require_single_env("step")
        return self.step_all([action], metadata=metadata)[0]

    def reset_all(self, instruction: str = "", **kwargs) -> List[RobotObservation]:
        observations = self.env.reset_all(instruction=instruction, **kwargs)
        self._n_envs = len(observations)

        base_index = self._next_episode_index()
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y_%m_%d_%H_%M_%S")

        self._episode_indices = []
        self._episode_paths = []
        self._step_counts = []
        self._last_obs = []
        self._writers = []
        self._instructions = []
        self._instruction_logs = []
        self._first_obs_metadata = []
        self._events = {}

        for env_i, observation in enumerate(observations):
            ep_idx = base_index + env_i
            instruction_eff = instruction or observation.metadata.get("instruction", "")
            self._episode_indices.append(ep_idx)
            self._episode_paths.append(
                os.path.join(
                    self.config.output_path, self._episode_id(ep_idx, timestamp, instruction_eff, env_i)
                )
            )
            self._step_counts.append(1)
            self._last_obs.append(observation)
            self._writers.append({})
            self._stream(env_i, observation)
            self._instructions.append(instruction_eff)
            self._instruction_logs.append([])
            self._first_obs_metadata.append(dict(observation.metadata))

        if self.config.rerun_enabled:
            self._rerun_writers = [
                RerunWriter(recording_id=os.path.basename(path), fps=self.config.fps)
                for path in self._episode_paths
            ]
        else:
            self._rerun_writers = []

        return observations

    def step_all(
        self,
        actions: List[RobotAction],
        metadata: Dict[str, Any] | None = None,
    ) -> List[RobotObservation]:
        self._log_step_instruction(metadata)
        # Generic per-episode event channel: ANY per-env events a client component put into the
        # step metadata are accumulated here and saved with the episode, so new event kinds need no
        # recorder changes.
        for env_i, events in ((metadata or {}).get("events") or {}).items():
            self._events.setdefault(env_i, []).extend(events)
        observations = self.env.step_all(actions, metadata=metadata)

        # Envs the runner has marked finished (e.g. terminate_on_success) keep stepping in lockstep
        # with the batch, but their episode is over. Stop recording them so each video/metadata ends
        # at that env's final step. When "finished" is absent, every env records to the batch end.
        finished = (metadata or {}).get("finished")
        for env_i, observation in enumerate(observations):
            if finished is not None and finished[env_i]:
                continue
            if env_i < len(self._rerun_writers):
                self._rerun_writers[env_i].capture_frame(
                    observation=self._last_obs[env_i], action=actions[env_i]
                )
            self._stream(env_i, observation)
            self._last_obs[env_i] = observation
            self._step_counts[env_i] += 1
        return observations

    def end_episode(self) -> None:
        # Each env writes only its own video + rerun files, so finalize them on a thread pool.
        # sched_getaffinity (not os.cpu_count) so the default respects the CPU affinity and K
        # never oversubscribes cores.
        max_workers = self.config.video_encode_workers or len(os.sched_getaffinity(os.getpid()))
        workers = min(self._n_envs, max_workers)
        if workers <= 1:
            for env_i in range(self._n_envs):
                self._finalize_episode(env_i)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(self._finalize_episode, range(self._n_envs)))
        self._rerun_writers = []

        self.env.end_episode()

    def _finalize_episode(self, env_i: int) -> None:
        last = self._last_obs[env_i]
        success: Optional[bool] = None
        self._close_videos(env_i)
        if last is not None and "done" in last.metadata:
            success = bool(last.metadata["done"])
            # Frames streamed into the pre-suffix directory, since success is only known now.
            final = self._episode_paths[env_i] + ("_success" if success else "_failure")
            if os.path.isdir(self._episode_paths[env_i]):
                os.replace(self._episode_paths[env_i], final)
            self._episode_paths[env_i] = final
        self._save_episode_metadata(env_i, success)
        # Save into the now-suffixed episode dir; parallelized across envs by the same pool.
        if env_i < len(self._rerun_writers):
            self._rerun_writers[env_i].close(
                output_path=os.path.join(self._episode_paths[env_i], "replay.rrd")
            )

    @property
    def episode_paths(self) -> List[str]:
        """Each env's recording directory for the episode just recorded."""
        return list(self._episode_paths)

    def _require_single_env(self, call: str) -> None:
        if self._n_envs > 1:
            raise RuntimeError(f"{call} requires n_envs=1; got {self._n_envs} — use {call}_all")

    def _log_step_instruction(self, metadata: Dict[str, Any] | None) -> None:
        if not metadata:
            return

        instructions: List[str]
        if "instructions" in metadata:
            raw = metadata["instructions"]
            if not isinstance(raw, list) or len(raw) != self._n_envs:
                raise ValueError(
                    f"metadata['instructions'] must be a list of length {self._n_envs}; got {raw!r}"
                )
            instructions = [str(x) for x in raw]
        elif "instruction" in metadata:
            instructions = [str(metadata["instruction"])] * self._n_envs
        else:
            return

        for env_i, instruction in enumerate(instructions):
            step_val = metadata.get("step", self._step_counts[env_i] - 1)
            log = self._instruction_logs[env_i]
            if not log or log[-1]["instruction"] != instruction:
                log.append({"step": step_val, "instruction": instruction})

    def _stream(self, env_i: int, observation: RobotObservation) -> None:
        """Write this observation's frames straight to disk. Retaining them until the episode ends
        costs n_envs x steps x frame of host RAM per lane, enough to OOM-kill a shared inference
        server on long episodes."""
        if not self.config.video_enabled or not observation.images:
            return
        writers = self._writers[env_i]
        for camera_name, frame in observation.images.items():
            if camera_name not in writers:
                writers[camera_name] = VideoWriterOpenCV(
                    output_path=os.path.join(self._episode_paths[env_i], f"{camera_name}.mp4"),
                    fps=self.config.fps,
                )
            writers[camera_name].append(frame)

    def _close_videos(self, env_i: int) -> None:
        for writer in self._writers[env_i].values():
            writer.close()
            if not writer.wrote_signal:  # all-zero placeholder camera
                os.remove(writer.output_path)

    def _save_episode_metadata(self, env_i: int, success: Optional[bool]) -> None:
        metadata: Dict[str, Any] = {
            "episode_index": self._episode_indices[env_i],
            "num_steps": max(0, self._step_counts[env_i] - 1),
            "success": success,
        }

        if self._events.get(env_i):
            metadata["events"] = self._events[env_i]

        first_meta = self._first_obs_metadata[env_i]
        if "env_name" in first_meta:
            metadata["env_name"] = first_meta["env_name"]
        if "instruction" in first_meta:
            metadata["default_instruction"] = first_meta["instruction"]

        log = self._instruction_logs[env_i]
        if log:
            metadata["instruction"] = log[0]["instruction"]
            if len(log) > 1:
                metadata["instruction_log"] = log
        else:
            metadata["instruction"] = self._instructions[env_i]

        if self._n_envs > 1:
            metadata["batch_env_index"] = env_i

        os.makedirs(self._episode_paths[env_i], exist_ok=True)
        path = os.path.join(self._episode_paths[env_i], "episode_meta.yaml")
        with open(path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    def _episode_id(self, ep_idx: int, timestamp: str, instruction: str, env_i: int) -> str:
        safe = instruction.replace(" ", "_").replace("/", "_")[:80]
        if self._n_envs > 1:
            return f"episode_{ep_idx}_{timestamp}_env{env_i}_{safe}"
        return f"episode_{ep_idx}_{timestamp}_{safe}"

    def _next_episode_index(self) -> int:
        existing = [name for name in os.listdir(self.config.output_path) if name.startswith("episode_")]
        return len(existing) + 1
