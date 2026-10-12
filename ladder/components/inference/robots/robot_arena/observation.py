"""Builds length-``n_envs`` ``RobotObservation`` lists from Genesis state.
One unbound camera; ``camera.render()`` returns ``(H, W, 3)`` at K=1 and
``(K, H, W, 3)`` at K>=2 (Madrona BatchRenderer)."""

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import genesis
import numpy as np
import torch

from ladder.components.inference.api.core.observation import (
    METADATA_CAMERAS_KEY,
    ManipulatorState,
    RobotObservation,
    encode_camera,
)
from ladder.components.inference.robots.robot_arena.gripper_controller import GripperController
from ladder.components.inference.robots.robot_arena.scene_loader import (
    ROBOT_CONFIGS,
    SceneData,
    composite_foreground_over_background,
)
from ladder.components.inference.robots.robot_arena.transforms import RobotName, eef_poses_world_to_canonical
from ladder.core.common.logger.logger import get_logger

LOGGER = get_logger(__name__)

# Genesis' camera is OpenGL (-Z forward, +Y up); pinhole intrinsics want +Z forward, +Y down.
_OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def _intrinsics_from_fov(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Pinhole intrinsics from a VERTICAL field of view, which is how Genesis names a camera."""
    focal = 0.5 * height / np.tan(0.5 * np.deg2rad(float(fov_deg)))
    return np.array(
        [[focal, 0.0, 0.5 * width], [0.0, focal, 0.5 * height], [0.0, 0.0, 1.0]], dtype=np.float64
    )


class ObservationBuilder:
    """Builds a length-``n_envs`` ``RobotObservation`` list from Genesis state."""

    def __init__(
        self,
        cameras: Sequence["genesis.Camera"],
        robot: "genesis.Entity",
        robot_name: RobotName,
        world_to_base: np.ndarray,
        gripper: GripperController,
        scene_data: SceneData,
        config_scene_name: str,
        control_freq: int,
        n_envs: int = 1,
        publish_depth: bool = False,
        render_images: bool = True,
        camera_fov: Optional[float] = None,
        camera_pose: Optional[np.ndarray] = None,
    ):
        self._cameras: List["genesis.Camera"] = list(cameras)
        assert len(self._cameras) == 1, f"expects 1 camera, got {len(self._cameras)}"
        self._camera_width, self._camera_height = (int(v) for v in self._cameras[0].res)

        self._robot = robot
        self._robot_name = robot_name
        self._robot_config = ROBOT_CONFIGS[robot_name]
        self._world_to_base = world_to_base
        self._gripper = gripper
        self._scene_data = scene_data
        self._config_scene_name = config_scene_name
        self._control_freq = control_freq
        self._n_envs = n_envs
        self._publish_depth = publish_depth
        self._render_images = render_images
        self._camera_payload_base: Optional[Dict[str, np.ndarray]] = None
        if publish_depth:
            assert (
                camera_fov is not None and camera_pose is not None
            ), "publish_depth needs the camera's fov and pose to build its calibration"
            self._camera_payload_base = {
                "intrinsics": _intrinsics_from_fov(
                    self._camera_width, self._camera_height, float(camera_fov)
                ),
                "base_from_camera": np.asarray(world_to_base, dtype=np.float64)
                @ np.asarray(camera_pose, dtype=np.float64)
                @ _OPENGL_TO_OPENCV,
            }
        self._background_image: Optional[np.ndarray] = None
        # Background image resized to render resolution, cached across steps and envs.
        self._background_resized_image: Optional[np.ndarray] = None
        self._background_gpu: Optional["torch.Tensor"] = None

    def build(
        self,
        step_count: int,
        start_timestamp: int,
        last_target_pose_canonical: Optional[np.ndarray],
    ) -> List[RobotObservation]:
        """``last_target_pose_canonical``: ``(4, 4)`` (broadcast), ``(K, 4, 4)``, or ``None``."""
        main_images, depth_all = self._render_main_images() if self._render_images else (None, None)

        # atleast_2d normalizes Genesis getter output to ``(n_envs, ...)`` — K=1
        # returns 1D shapes (n,), K>=2 already returns (K, n); both become (K, n).
        ee_link = self._robot.get_link(name=self._robot_config.ee_link)
        ee_pos_all = np.atleast_2d(ee_link.get_pos().cpu().numpy())
        ee_quat_all = np.atleast_2d(ee_link.get_quat().cpu().numpy())
        joints_all = np.atleast_2d(self._robot.get_dofs_position().cpu().numpy())

        gripper_state_all = self._gripper.get_normalized_state(joints_all)

        timestamps = start_timestamp + np.array([step_count / self._control_freq], dtype=np.float64)
        targets = _normalize_targets(last_target_pose_canonical, self._n_envs)

        eef_poses = eef_poses_world_to_canonical(
            self._world_to_base, self._robot_name, ee_pos_all, ee_quat_all
        )

        observations: List[RobotObservation] = []
        for env_i in range(self._n_envs):
            eef_pose = eef_poses[env_i]
            eef_pose_from_target = targets[env_i] if targets is not None else eef_pose

            metadata: Dict[str, object] = {
                "instruction": self._scene_data.task_instruction,
                "env_name": f"robot_arena_{self._config_scene_name}",
                "robot_name": self._robot_name,
            }
            if self._n_envs > 1:
                metadata["batch_env_index"] = env_i
            if depth_all is not None:
                metadata[METADATA_CAMERAS_KEY] = {
                    "main": encode_camera(
                        depth_m=depth_all[env_i],
                        intrinsics=self._camera_payload_base["intrinsics"],
                        base_from_camera=self._camera_payload_base["base_from_camera"],
                    )
                }

            observations.append(
                RobotObservation(
                    images=(
                        {"main": main_images[env_i], "wrist": np.zeros_like(main_images[env_i])}
                        if main_images is not None
                        else None
                    ),
                    state=ManipulatorState(
                        eef_pose=eef_pose,
                        eef_pose_from_target=eef_pose_from_target,
                        gripper_state=gripper_state_all[env_i].astype(np.float32),
                        joints=joints_all[env_i][:7].astype(np.float32),
                    ),
                    timestamps=timestamps,
                    metadata=metadata,
                )
            )

        return observations

    def clear_background_cache(self) -> None:
        self._background_image = None
        self._background_resized_image = None
        self._background_gpu = None

    def _render_main_images(self) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """Render + composite all K frames as ``[(H, W, 3) uint8]`` per env, plus metric depth.

        K>=2 (Madrona tensors) composites on the GPU; K=1 / numpy uses the CPU path. Depth comes
        from the same render call, so asking for it costs one extra target and no extra pass.
        """
        rgb_tensor, depth_tensor, seg_tensor, _ = self._cameras[0].render(
            rgb=True, depth=self._publish_depth, segmentation=True
        )
        depth_all = self._metric_depth(depth_tensor) if self._publish_depth else None

        if isinstance(rgb_tensor, torch.Tensor) and isinstance(seg_tensor, torch.Tensor):
            return self._render_main_images_gpu(rgb_tensor, seg_tensor), depth_all

        # CPU fallback (K=1 / numpy renderer).
        rgb_all = np.asarray(rgb_tensor)
        seg_all = np.asarray(seg_tensor)
        if rgb_all.ndim == 3:
            rgb_all = rgb_all[None]
        if seg_all.ndim == 2:
            seg_all = seg_all[None]
        if rgb_all.dtype != np.uint8:
            rgb_all = (np.clip(rgb_all, 0.0, 1.0) * 255).astype(np.uint8)
        rgb_all = rgb_all[..., :3]
        bg = self._background_resized_for(rgb_all.shape[1], rgb_all.shape[2])
        if bg is None:
            return (
                [np.ascontiguousarray(rgb_all[i], dtype=np.uint8) for i in range(rgb_all.shape[0])],
                depth_all,
            )
        composited = composite_foreground_over_background(rgb_all, seg_all, bg)
        return (
            [np.ascontiguousarray(composited[i], dtype=np.uint8) for i in range(composited.shape[0])],
            depth_all,
        )

    def _render_main_images_gpu(self, rgb_tensor: torch.Tensor, seg_tensor: torch.Tensor) -> List[np.ndarray]:
        """GPU composite for the K>=2 Madrona path. Bit-identical to the CPU path."""
        rgb = rgb_tensor[..., :3]
        if rgb.ndim == 3:
            rgb = rgb[None]
        if rgb.dtype != torch.uint8:
            rgb = (rgb.clamp(0.0, 1.0) * 255).to(torch.uint8)
        seg = seg_tensor
        if seg.ndim == 4:  # (K, H, W, 1)
            seg = seg[..., 0]
        elif seg.ndim == 2:  # (H, W)
            seg = seg[None]
        if rgb.shape[0] != self._n_envs or seg.shape[0] != self._n_envs:
            raise RuntimeError(f"render shape mismatch: rgb {tuple(rgb.shape)} seg {tuple(seg.shape)}")

        bg = self._background_resized_for(int(rgb.shape[1]), int(rgb.shape[2]))
        if bg is not None:
            bg_gpu = self._background_gpu_tensor(bg, rgb.device)
            composited = torch.where((seg > 0).unsqueeze(-1), rgb, bg_gpu)
        else:
            composited = rgb
        out = composited.contiguous().cpu().numpy()
        return [np.ascontiguousarray(out[i], dtype=np.uint8) for i in range(out.shape[0])]

    def _metric_depth(self, depth_tensor) -> np.ndarray:
        """Genesis' depth buffer as ``(K, H, W)`` metres, with the background marked invalid.

        Genesis reports metres along the optical axis (checked by plane-fitting the table: reading
        it as ray length instead bends the plane, 3.9 mm of residual against 3.6 mm). Everything
        the scene does not cover comes back at the far plane, and a RobotArena scene is mostly
        that: the backdrop is a photograph composited into the RGB, not geometry, so 87% of the
        frame has no surface at all. A real sensor reports no return there, so the far plane
        becomes NaN and every consumer drops it the same way it drops a sensor dropout.
        """
        if depth_tensor is None:
            raise RuntimeError("publish_depth is set but Genesis returned no depth buffer")
        depth = depth_tensor.detach().cpu().numpy() if hasattr(depth_tensor, "cpu") else depth_tensor
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim == 2:
            depth = depth[None]
        if depth.shape[0] != self._n_envs:
            raise RuntimeError(f"depth render gave {depth.shape[0]} frames for {self._n_envs} envs")
        far = float(self._cameras[0].far)
        return np.where(depth >= far - 1e-3, np.float32("nan"), depth)

    def _background_resized_for(self, h: int, w: int) -> Optional[np.ndarray]:
        """Load + resize the background to ``(h, w)`` once (cached). None if no bg."""
        if self._scene_data.background_path is None:
            return None
        if self._background_image is None:
            bg = cv2.imread(self._scene_data.background_path)
            if bg is None:
                LOGGER.warn(f"Could not load background: {self._scene_data.background_path}")
                return None
            self._background_image = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        if self._background_resized_image is None or self._background_resized_image.shape[:2] != (h, w):
            self._background_resized_image = cv2.resize(self._background_image, (w, h))
            self._background_gpu = None  # invalidate GPU copy
        return self._background_resized_image

    def _background_gpu_tensor(self, bg_resized: np.ndarray, device) -> torch.Tensor:
        """Upload the resized background to the GPU once (cached) as ``(H, W, 3)`` uint8."""
        if self._background_gpu is None or self._background_gpu.device != device:
            self._background_gpu = torch.as_tensor(bg_resized, device=device, dtype=torch.uint8)
        return self._background_gpu


def _normalize_targets(targets: Optional[np.ndarray], n_envs: int) -> Optional[np.ndarray]:
    """``(4, 4)`` → broadcast to ``(K, 4, 4)``; ``(K, 4, 4)`` → passthrough; ``None`` → ``None``."""
    if targets is None:
        return None
    if targets.ndim == 2:
        return np.broadcast_to(targets, (n_envs, 4, 4))
    return targets
