"""Genesis scene construction. K=1 builds without a batch dim (bit-identical
to the legacy path); K>=2 replicates via ``scene.build(n_envs=K)`` and
attaches one unbound camera under Madrona's BatchRenderer."""

import dataclasses
import importlib.metadata
import os
from typing import List, Optional, Tuple

import genesis
import numpy as np
import scipy.spatial.transform

from ladder.components.inference.robots.robot_arena.grasping import resolve_overlap, single_env_view
from ladder.components.inference.robots.robot_arena.scene_loader import (
    GROUND_CLIP_HEIGHT,
    OVERLAP_RESOLVE_ITERATIONS,
    ROBOT_CONFIGS,
    SETTLE_STEPS,
    TABLE_PLANE_HEIGHT,
    ObjectConfig,
    SceneData,
    compute_camera_fov_deg,
    compute_camera_to_world,
    resolve_genesis_surface,
)
from ladder.components.inference.robots.robot_arena.transforms import RobotName, quat_wxyz_to_xyzw
from ladder.core.common.logger.logger import get_logger
from ladder.core.compute.seeder import SEEDER

LOGGER = get_logger(__name__)


# Stamped by RobotArenaServer before the first env is built. genesis.init()
# must run in the same thread as scene.build(), so the server writes here
# instead of calling genesis.init(seed=...) directly from __init__. Default
# matches RobotArenaTestConfig.seed.
_GENESIS_SEED: int = 42


def set_genesis_seed(seed: int) -> None:
    global _GENESIS_SEED  # noqa: PLW0603
    _GENESIS_SEED = int(seed)


# gs_madrona iterates ``dist.files`` to locate libnvrtc.so.12, but rules_python
# 1.9.0 doesn't emit RECORD files so Distribution.files returns None and the
# iteration raises TypeError. Patch it to fall back to [].
_ORIGINAL_DISTRIBUTION_FILES = importlib.metadata.Distribution.files.fget  # type: ignore[attr-defined]
importlib.metadata.Distribution.files = property(  # type: ignore[assignment]
    lambda self: _ORIGINAL_DISTRIBUTION_FILES(self) or []
)


@dataclasses.dataclass
class SceneContext:
    """Genesis objects produced by scene construction. ``len(cameras) == 1``;
    the BatchRenderer returns ``(K, H, W, 3)`` from a single unbound camera."""

    scene: "genesis.Scene"
    robot: "genesis.Entity"
    cameras: List["genesis.Camera"]
    camera_fov: float
    camera_pose: np.ndarray
    object_entities: List[Tuple["genesis.Entity", ObjectConfig]]
    base_to_world: np.ndarray
    world_to_base: np.ndarray
    n_envs: int = 1

    @property
    def camera(self) -> "genesis.Camera":
        return self.cameras[0]


def build_genesis_scene(
    scene_data: SceneData,
    *,
    data_root: str,
    sim_freq: int,
    sim_substeps: int = 1,
    rigid_constraint_timeconst: Optional[float] = None,
    robot_name: RobotName,
    camera_width: int,
    camera_height: int,
    camera_offset: Optional[Tuple[float, float, float]] = None,
    render_only: bool = False,
    n_envs: int = 1,
    env_spacing: Tuple[float, float] = (5.0, 5.0),
) -> SceneContext:
    """K>=2 replicates the scene K times. ``render_only`` disables physics."""
    is_default = scene_data.is_default_scene
    robot_config = ROBOT_CONFIGS[robot_name]

    if not genesis._initialized:  # noqa: SLF001
        # genesis.init(seed=N) clobbers host RNGs; snapshot/restore via SEEDER.
        saved_state = SEEDER.get_state()
        genesis.init(backend=genesis.gpu, seed=_GENESIS_SEED)
        SEEDER.set_state(saved_state)

    renderer = genesis.renderers.BatchRenderer() if n_envs > 1 else None

    # Softening the constraint time constant (default 0.01s) bounds the forces a
    # deeply-interpenetrating settle or an over-constrained grasp weld can
    # generate, the other half of the "Invalid constraint forces causing 'nan'"
    # fix. None keeps the Genesis default.
    rigid_kwargs = {"dt": 1.0 / sim_freq}
    if rigid_constraint_timeconst is not None:
        rigid_kwargs["constraint_timeconst"] = rigid_constraint_timeconst

    # K=1 (pyrender Rasterizer) reads vis_options.lights/ambient_light; K>=2
    # (Madrona BatchRenderer) ignores them and instead reads scene.add_light()
    # calls plus a hardcoded 0.05 ambient term in the shader (set up below).
    scene = genesis.Scene(
        sim_options=genesis.options.SimOptions(dt=1.0 / sim_freq, substeps=sim_substeps),
        rigid_options=genesis.options.RigidOptions(**rigid_kwargs),
        vis_options=genesis.options.VisOptions(
            show_world_frame=False,
            ambient_light=(0.15, 0.15, 0.15),
            lights=[
                genesis.options.vis.DirectionalLight(
                    dir=(-0.3, -0.3, -1.0), color=(1.0, 1.0, 1.0), intensity=8.0
                ),
                genesis.options.vis.DirectionalLight(
                    dir=(0.4, -0.2, -1.0), color=(1.0, 1.0, 1.0), intensity=4.0
                ),
                genesis.options.vis.DirectionalLight(
                    dir=(-0.4, 0.6, -0.8), color=(1.0, 1.0, 1.0), intensity=3.0
                ),
            ],
        ),
        show_viewer=False,
        show_FPS=True,
        renderer=renderer,
    )

    if is_default:
        scene.add_entity(
            genesis.morphs.Plane(pos=(0, 0, TABLE_PLANE_HEIGHT), visualization=False, fixed=True)
        )
    else:
        scene.add_entity(genesis.morphs.Plane(visualization=False, fixed=True))

    arena_root = os.path.join(data_root, os.pardir, os.pardir)
    urdf_path = os.path.join(arena_root, robot_config.urdf_path)
    if is_default:
        urdf_morph = genesis.morphs.URDF(
            file=urdf_path,
            fixed=True,
            merge_fixed_links=True,
            links_to_keep=[robot_config.ee_link],
            pos=robot_config.initial_pos,
            quat=robot_config.initial_quat,
        )
    else:
        urdf_morph = genesis.morphs.URDF(
            file=urdf_path,
            fixed=True,
            merge_fixed_links=True,
            links_to_keep=[robot_config.ee_link],
        )
    # widowx_description URDF's OBJ meshes ship with a SimpleMaterial that
    # references interbotix_black.png — Genesis's trimesh loader picks that up
    # as the diffuse texture (utils/urdf.py:202+ falls through to the URDF
    # surface when no <material> tag is present). Don't override: the texture's
    # ~50 RGB matte-black pixels are exactly the WX250s look. The previous
    # K>=2 override of Default(color=(0.10,...)) flattened it to uniform gray.
    robot = scene.add_entity(urdf_morph)

    object_entities: List[Tuple["genesis.Entity", ObjectConfig]] = []
    for obj_config in scene_data.objects:
        entity = _add_object_to_scene(scene, obj_config, render_only=render_only)
        if entity is not None:
            object_entities.append((entity, obj_config))

    camera_fov = compute_camera_fov_deg(scene_data.camera_intrinsics)
    camera_pose = compute_camera_to_world(scene_data.camera_extrinsics, camera_offset=camera_offset)
    cameras = [
        scene.add_camera(
            pos=(0, 0, 0), lookat=(0, 0, 1), res=(camera_width, camera_height), fov=camera_fov, GUI=False
        )
    ]

    # Madrona ignores vis_options.lights/ambient_light — only scene.add_light()
    # plus its hardcoded ~0.05 ambient contribute. Reproduce the K=1 pyrender
    # key/fill/rim directionals (same dirs as vis_options above) scaled to Madrona
    # intensity units (pyrender 8/4/3 -> x0.2). No omnidirectional fill: it washes
    # out contrast and neutralises the warm scene hue, while the baked ambient lifts
    # shadows enough; castshadow restores the pyrender dark-shadow/bright-highlight
    # contrast. Calibrated to the pyrender reference by matching the segmentation-
    # masked foreground mean colour (dist ~8, vs ~37 for the old 6-cardinal fill).
    if n_envs > 1:
        for light_dir, intensity in (
            ((-0.3, -0.3, -1.0), 1.6),
            ((0.4, -0.2, -1.0), 0.8),
            ((-0.4, 0.6, -0.8), 0.6),
        ):
            scene.add_light(
                pos=(0.0, 0.0, 0.0),
                dir=light_dir,
                color=(1.0, 1.0, 1.0),
                intensity=intensity,
                directional=True,
                castshadow=True,
            )

    if n_envs > 1:
        scene.build(n_envs=n_envs, env_spacing=env_spacing)
    else:
        scene.build()

    for cam in cameras:
        cam.set_pose(camera_pose)

    if robot_config.initial_qpos is not None:
        robot.set_qpos(np.array(robot_config.initial_qpos, dtype=np.float32))

    if not render_only and not is_default:
        for entity, obj_cfg in object_entities:
            if not obj_cfg.fixed:
                entity.set_mass(obj_cfg.mass)
                entity.set_friction(obj_cfg.friction)

    # K>=2 envs share build-time poses, so overlap resolution on env 0 broadcasts.
    if not render_only:
        resolve_object_overlaps(object_entities)

    base_to_world, world_to_base = _compute_base_frame(robot, is_default, n_envs)

    return SceneContext(
        scene=scene,
        robot=robot,
        cameras=cameras,
        camera_fov=camera_fov,
        camera_pose=camera_pose,
        object_entities=object_entities,
        base_to_world=base_to_world,
        world_to_base=world_to_base,
        n_envs=n_envs,
    )


def resolve_object_overlaps(object_entities: List[Tuple["genesis.Entity", ObjectConfig]]) -> None:
    """Lift overlapping objects and clip below-ground ones (K=1 and K>=2 both)."""
    entity_list = [e for e, _ in object_entities]
    for _ in range(OVERLAP_RESOLVE_ITERATIONS):
        for i in range(len(entity_list)):
            for j in range(i + 1, len(entity_list)):
                resolve_overlap(entity_list[i], entity_list[j])

    for entity, _ in object_entities:
        pos = single_env_view(entity.get_pos())
        if pos[2] < 0:
            entity.set_pos([float(pos[0]), float(pos[1]), GROUND_CLIP_HEIGHT])


def settle_objects(
    scene: "genesis.Scene",
    object_entities: List[Tuple["genesis.Entity", ObjectConfig]],
) -> None:
    """Settle under gravity, restore each object's original XY."""
    xy_positions = []
    for entity, _ in object_entities:
        pos = single_env_view(entity.get_pos())
        xy_positions.append((float(pos[0]), float(pos[1])))

    for _ in range(SETTLE_STEPS):
        scene.step()

    for (entity, _), (original_x, original_y) in zip(object_entities, xy_positions, strict=True):
        pos = single_env_view(entity.get_pos())
        entity.set_pos([original_x, original_y, float(pos[2])])


def _compute_base_frame(
    robot: "genesis.Entity", is_default: bool, n_envs: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(base_to_world, world_to_base)``; identity for non-default scenes."""
    if not is_default:
        identity = np.eye(4, dtype=np.float64)
        return identity.copy(), identity.copy()

    base_link = robot.get_link(name="base_link")
    base_pos_raw = base_link.get_pos().cpu().numpy()
    base_quat_raw = base_link.get_quat().cpu().numpy()
    # K>=2 returns (K, 3) / (K, 4); envs identical at build time, so take env 0.
    base_pos = base_pos_raw[0].flatten() if n_envs > 1 else base_pos_raw.flatten()
    base_quat_wxyz = base_quat_raw[0].flatten() if n_envs > 1 else base_quat_raw.flatten()
    base_quat_xyzw = quat_wxyz_to_xyzw(base_quat_wxyz)
    base_rot = scipy.spatial.transform.Rotation.from_quat(base_quat_xyzw).as_matrix()
    base_to_world = np.eye(4, dtype=np.float64)
    base_to_world[:3, :3] = base_rot
    base_to_world[:3, 3] = base_pos
    world_to_base = np.linalg.inv(base_to_world)
    return base_to_world, world_to_base


def _add_object_to_scene(
    scene: "genesis.Scene", obj_config: ObjectConfig, *, render_only: bool = False
) -> Optional["genesis.Entity"]:
    if not os.path.exists(obj_config.mesh_path):
        LOGGER.warn(f"Skipping object '{obj_config.name}': mesh not found at {obj_config.mesh_path}")
        return None

    surface = resolve_genesis_surface(genesis, obj_config.surface)

    if render_only:
        mesh = genesis.morphs.Mesh(
            file=obj_config.mesh_path,
            pos=obj_config.position,
            quat=obj_config.rotation_quat,
            scale=obj_config.scale,
            fixed=True,
            collision=False,
            convexify=False,
            decimate=False,
        )
    else:
        mesh = genesis.morphs.Mesh(
            file=obj_config.mesh_path,
            pos=obj_config.position,
            quat=obj_config.rotation_quat,
            scale=obj_config.scale,
            fixed=obj_config.fixed,
            convexify=True,
            decimate=True,
        )
    return scene.add_entity(mesh, surface=surface)
