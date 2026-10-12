import dataclasses
import json
import os
import re
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import scipy.spatial.transform

from ladder.components.inference.robots.robot_arena.scene_corrections import (
    BACKGROUND_OVERRIDES,
    UPSTREAM_DATA_CORRECTIONS,
)
from ladder.components.inference.robots.robot_arena.transforms import RobotName, quat_xyzw_to_wxyz
from ladder.core.common.logger.logger import get_logger

LOGGER = get_logger(__name__)

# Keywords used to heuristically classify scene objects as fixed (non-movable).
# Objects whose name contains "_destination" or any of these keywords are treated as
# static geometry (table, receptacles, etc.) rather than manipulable objects.
# This is a fallback for scenes that don't provide explicit fixed/movable metadata.
FIXED_OBJECT_KEYWORDS = ("table", "tray", "plate", "bowl", "colander")

# Physics constants shared between single-env and batched-env implementations.
TABLE_PLANE_HEIGHT = 0.88  # Default-scene ground plane z offset (meters)
OVERLAP_RESOLVE_ITERATIONS = 5  # Passes to lift overlapping objects after scene build
SETTLE_STEPS = 400  # Physics steps to run after reset for settling (400 @ 200Hz = 2.0s).
# Raised from 100 (0.5s) on 2026-05-27: a drop-test found 42/74 scenes had objects still
# settling/sliding after 100 steps, so init positions (and the success-checker baseline)
# were captured mid-settle, causing 'floating' look + noisy initial positions.
GROUND_CLIP_HEIGHT = 0.01  # Minimum z position for objects (clip below-ground objects)
GRIPPER_OPEN_FORCE = 10.0  # Force magnitude for opening gripper (N)
GRIPPER_CLOSE_FORCE = -10.0  # Force magnitude for closing gripper (N)
GRIPPER_SAFETY_KP = 100.0  # Proportional gain for gripper safety limits
GRIPPER_SAFETY_KD = 0.5  # Derivative gain for gripper safety limits
GRIPPER_EFFORT_LIMIT = 10.0  # Maximum gripper effort (N)
GRIPPER_VELOCITY_LIMIT = 0.05  # Maximum gripper velocity (m/s)
GRIPPER_SOFT_POSITION_LOWER = 0.01  # Soft lower limit for gripper position (m)


@dataclasses.dataclass(frozen=True)
class RobotConfig:
    """Per-robot hardware configuration matching RobotArena's robot_config.yaml."""

    urdf_path: str
    ee_link: str
    initial_pos: List[float]
    initial_quat: List[float]
    initial_qpos: Optional[List[float]] = None
    kp: Optional[List[float]] = None
    kv: Optional[List[float]] = None
    gripper_dof_indices: Optional[List[int]] = None
    gripper_open: Optional[float] = None
    gripper_closed: Optional[float] = None


ROBOT_CONFIGS: Dict[RobotName, RobotConfig] = {
    RobotName.WIDOWX: RobotConfig(
        # Textured WidowX (OBJ+MTL) — shows matte surface detail under Madrona's
        # diffuse path, vs the flat-black STL in trossen_wx250s/.
        urdf_path="utils/assets/descriptions/widowx_description/wx250s.urdf",
        ee_link="ee_gripper_link",
        # Canonical Bridge V2 / SimplerEnv WidowX base position.
        initial_pos=[0.147, 0.028, 0.870],
        initial_quat=[0.0, 0.0, 0.0, 1.0],
        # Reference initial joint positions from RobotArena (6 arm + 2 finger DOFs)
        initial_qpos=[
            -0.02014876,
            0.04723017,
            0.22625704,
            -0.00307271,
            1.365988,
            -0.00168102,
            0.037,
            0.037,
        ],
        # Tuned PD gains from RobotArena reference implementation
        kp=[4345, 2019, 3184, 4424, 3140, 10000, 100, 100],
        kv=[224, 252, 593, 10, 893, 302, 10, 10],
        # Gripper DOF indices (left_finger, right_finger) in the joint array
        gripper_dof_indices=[6, 7],
        # Gripper range in meters (prismatic joint)
        gripper_open=0.037,
        gripper_closed=0.0,
    ),
    RobotName.FRANKA: RobotConfig(
        urdf_path="utils/assets/franka_emika_panda/panda_2f85.xml",
        ee_link="panda_hand",
        initial_pos=[0.127, 0.060, 0.870],
        initial_quat=[0.0, 0.0, 0.0, 1.0],
    ),
    RobotName.UR5: RobotConfig(
        urdf_path="utils/assets/ur5e/urdf/ur5e.urdf",
        ee_link="ee_virtual_link",
        initial_pos=[0.0, 0.0, 0.0],
        initial_quat=[0.0, 0.0, 0.0, 1.0],
    ),
}


@dataclasses.dataclass
class ObjectConfig:
    """Configuration for a single object in the scene."""

    name: str
    mesh_path: str
    position: List[float]
    rotation_quat: List[float]  # [w, x, y, z] quaternion (Genesis convention)
    scale: float  # Uniform scale factor
    mass: float
    friction: float
    surface: str
    fixed: bool = False


@dataclasses.dataclass
class SceneData:
    """Holds all data needed to reconstruct a RobotArena scene in Genesis."""

    objects: List[ObjectConfig]
    camera_extrinsics: np.ndarray  # [4, 4]
    camera_intrinsics: np.ndarray  # [3, 3]
    task_instruction: str
    background_path: Optional[str] = None
    is_default_scene: bool = False  # "defaultN" scenes have different setup


def _reproject_to_plane(
    position_world: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    z_plane: float,
) -> tuple[np.ndarray, float]:
    """Reproject a 3D point onto a horizontal plane through the camera.

    Follows the reference RobotArena implementation: projects the point through the
    camera to a pixel, back-projects the pixel ray, and intersects with z=z_plane.

    Returns:
        (new_position, scale_factor): the reprojected position and the depth ratio
        that should be applied to the object scale.
    """
    position_world = np.asarray(position_world, dtype=np.float64).reshape(3)
    rotation = extrinsics[:3, :3]
    translation = extrinsics[:3, 3]

    # World point → camera frame
    point_cam = rotation @ position_world + translation
    # Camera frame → pixel
    pixel_h = intrinsics @ point_cam
    pixel = pixel_h[:2] / pixel_h[2]

    # Back-project pixel to a camera-frame ray, then convert to world-frame ray
    pixel_homo = np.array([pixel[0], pixel[1], 1.0])
    ray_cam = np.linalg.inv(intrinsics) @ pixel_homo
    ray_world = rotation.T @ ray_cam
    cam_center_world = -rotation.T @ translation

    # Intersect ray with horizontal plane z = z_plane
    if abs(ray_world[2]) < 1e-12:
        LOGGER.warn("Ray parallel to z-plane; skipping reprojection")
        return position_world, 1.0
    t_scalar = (z_plane - cam_center_world[2]) / ray_world[2]
    new_position = cam_center_world + t_scalar * ray_world

    # Scale factor: ratio of new-to-old depth in camera frame
    z_orig = (rotation @ position_world + translation)[2]
    z_new = (rotation @ new_position + translation)[2]
    scale_factor = float(z_new / z_orig) if abs(z_orig) > 1e-12 else 1.0

    return new_position, scale_factor


def load_scene(  # noqa: PLR0915
    data_root: str,
    scene_name: str,
    background_path: Optional[str] = None,
    object_positions: Optional[List[List[float]]] = None,
    object_rotations: Optional[List[List[float]]] = None,
) -> SceneData:
    """
    Load a pre-generated RobotArena scene from disk.

    The scene directory structure (under data_root/bridge/<scene_name>/) contains:
        - extrinsics.npy: Camera extrinsic matrix [4, 4]
        - intrinsics.npy: Camera intrinsic matrix [3, 3]
        - masks/transformations.json: Per-object scale, translation [3], rotation [3x3]
        - masks/result.json: Per-object name, mass, friction, surface material
        - lang.txt: Task description(s)

    3D meshes are stored under data_root/assets/<scene_name>/<obj_key>.glb

    Args:
        data_root: Root path to the RobotArena data directory.
        scene_name: Scene directory name (e.g. "scene1").
        background_path: Optional override for background image path.
        object_positions: Optional override for object positions (for permutation/pose tests).
        object_rotations: Optional override for object rotations as quaternions (for pose tests).
    """
    scene_dir = os.path.join(data_root, "bridge", scene_name)

    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    # Detect scene type: "defaultN" scenes use a table-height plane and raw positions;
    # "sceneN" scenes use a ground-level plane with reprojected positions.
    is_default_scene = scene_name.startswith("default")

    # Load camera calibration
    extrinsics_path = os.path.join(scene_dir, "extrinsics.npy")
    intrinsics_path = os.path.join(scene_dir, "intrinsics.npy")

    if not os.path.exists(extrinsics_path):
        raise FileNotFoundError(f"Camera extrinsics not found: {extrinsics_path}")
    if not os.path.exists(intrinsics_path):
        raise FileNotFoundError(f"Camera intrinsics not found: {intrinsics_path}")

    camera_extrinsics = np.load(extrinsics_path).astype(np.float64)
    camera_intrinsics = np.load(intrinsics_path).astype(np.float64)

    assert camera_extrinsics.shape == (
        4,
        4,
    ), f"Expected extrinsics shape (4, 4), got {camera_extrinsics.shape}"
    assert camera_intrinsics.shape == (
        3,
        3,
    ), f"Expected intrinsics shape (3, 3), got {camera_intrinsics.shape}"

    # Load object transforms from masks/transformations.json
    # Format: {"image_XXXX_segmented": {"scale": float, "translation": [3], "rotation": [3] or [3x3]}, ...}
    transformations_path = os.path.join(scene_dir, "masks", "transformations.json")
    if not os.path.exists(transformations_path):
        raise FileNotFoundError(f"Transformations file not found: {transformations_path}")

    with open(transformations_path, "r") as f:
        transformations = json.load(f)

    # Load physical properties from masks/result.json
    # Format: [{"object_name": str, "count": [int], "mass": float, "friction": float, "surface": str}, ...]
    result_path = os.path.join(scene_dir, "masks", "result.json")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Result file not found: {result_path}")

    with open(result_path, "r") as f:
        result_entries = json.load(f)

    # Load physical_properties.json if present. For ~13 scenes this file carries the
    # CORRECT per-object names + _target/_destination roles, while masks/result.json
    # is garbled (wrong names / missing role tags). When present and non-empty it
    # takes precedence for naming; result.json remains the fallback (61 scenes have
    # an empty/absent physical_properties.json and are unaffected).
    phys_props_path = os.path.join(scene_dir, "physical_properties.json")
    phys_props_entries: List[Dict[str, Any]] = []
    if os.path.exists(phys_props_path):
        with open(phys_props_path, "r") as f:
            try:
                phys_props_entries = json.load(f)
            except json.JSONDecodeError:
                phys_props_entries = []

    # Load task instruction
    lang_path = os.path.join(scene_dir, "lang.txt")
    if not os.path.exists(lang_path):
        raise FileNotFoundError(f"Language file not found: {lang_path}")

    with open(lang_path, "r") as f:
        task_instructions = [line.strip() for line in f.readlines() if len(line.strip()) > 0]

    if len(task_instructions) == 0:
        raise ValueError(f"No task instructions found in {lang_path}")

    task_instruction = task_instructions[0]

    # Build object configs by joining transformations with physical properties via index
    objects = _build_object_configs(
        data_root=data_root,
        scene_name=scene_name,
        transformations=transformations,
        result_entries=result_entries,
        phys_props_entries=phys_props_entries,
        camera_extrinsics=camera_extrinsics if not is_default_scene else None,
        camera_intrinsics=camera_intrinsics if not is_default_scene else None,
        object_positions=object_positions,
        object_rotations=object_rotations,
    )

    # Resolve background path. The RobotArena data stores per-scene backgrounds
    # at scene_background/<scene_name>/background.png.
    # Some scenes have incorrect backgrounds in the upstream data; check for
    # overrides first.
    if background_path is None:
        bg_override = BACKGROUND_OVERRIDES.get(scene_name)
        if bg_override is not None:
            override_bg = os.path.join(
                data_root, "scene_background", bg_override.source_scene, "background.png"
            )
            if os.path.exists(override_bg):
                background_path = override_bg
                LOGGER.info(
                    f"Background override for '{scene_name}': using '{bg_override.source_scene}' "
                    f"background. Reason: {bg_override.reason}"
                )

    if background_path is None:
        default_bg = os.path.join(data_root, "scene_background", scene_name, "background.png")
        if os.path.exists(default_bg):
            background_path = default_bg

    if background_path is not None:
        LOGGER.info(
            f"Loaded scene '{scene_name}' (type={'default' if is_default_scene else 'standard'}): "
            f"{len(objects)} objects, background='{background_path}', instruction='{task_instruction}'"
        )
    else:
        LOGGER.warn(
            f"Loaded scene '{scene_name}' (type={'default' if is_default_scene else 'standard'}): "
            f"{len(objects)} objects, NO BACKGROUND FOUND (tried "
            f"{os.path.join(data_root, 'scene_background', scene_name, 'background.png')}), "
            f"instruction='{task_instruction}'"
        )

    return SceneData(
        objects=objects,
        camera_extrinsics=camera_extrinsics,
        camera_intrinsics=camera_intrinsics,
        task_instruction=task_instruction,
        background_path=background_path,
        is_default_scene=is_default_scene,
    )


def _build_object_configs(
    data_root: str,
    scene_name: str,
    transformations: Dict[str, Any],
    result_entries: List[Dict[str, Any]],
    phys_props_entries: Optional[List[Dict[str, Any]]] = None,
    camera_extrinsics: Optional[np.ndarray] = None,
    camera_intrinsics: Optional[np.ndarray] = None,
    object_positions: Optional[List[List[float]]] = None,
    object_rotations: Optional[List[List[float]]] = None,
) -> List[ObjectConfig]:
    """
    Build ObjectConfig list by joining transformations.json with result.json.

    The result.json "count" field maps each physics entry to its index in the
    sorted transformation keys list. For example, count=[2] means the physics
    entry corresponds to the 3rd object in transformations.json (image_0002_segmented).

    For non-default scenes (camera_extrinsics/intrinsics provided), object positions
    are reprojected onto a z=0.02 plane following the reference RobotArena implementation.
    """
    needs_reprojection = camera_extrinsics is not None and camera_intrinsics is not None
    transform_keys = sorted(transformations.keys())

    # Build index -> physics mapping from result.json
    physics_by_index: Dict[int, Dict[str, Any]] = {}
    for entry in result_entries:
        for idx in entry["count"]:
            physics_by_index[idx] = entry

    # Build index -> corrected name from physical_properties.json (when present): it
    # carries the right object names + _target/_destination roles for the scenes
    # where result.json is garbled. Takes precedence over result.json for naming.
    phys_name_by_index: Dict[int, str] = {}
    for entry in phys_props_entries or []:
        for idx in entry.get("count", []):
            if entry.get("object_name"):
                phys_name_by_index[idx] = entry["object_name"]

    objects: List[ObjectConfig] = []
    for i, obj_key in enumerate(transform_keys):
        transform = transformations[obj_key]

        # Apply upstream data corrections before any processing.
        correction = UPSTREAM_DATA_CORRECTIONS.get((scene_name, i))
        if correction is not None:
            LOGGER.info(f"Applying upstream correction for '{scene_name}' object {i}: {correction.reason}")

        # Mesh path: data_root/assets/<scene_name>/<obj_key>.glb
        mesh_path = os.path.join(data_root, "assets", scene_name, f"{obj_key}.glb")

        # Translation [3] (corrected if override present)
        if correction is not None and correction.translation is not None:
            position = list(correction.translation)
        else:
            position = list(transform["translation"])

        # Rotation: stored as either Euler angles [3] (deg) or a 3x3 rotation matrix.
        # Convert to wxyz quaternion (Genesis convention).
        if correction is not None and correction.rotation_euler_deg is not None:
            quat_xyzw = scipy.spatial.transform.Rotation.from_euler(
                "xyz", correction.rotation_euler_deg, degrees=True
            ).as_quat()
        else:
            rotation_raw = np.array(transform["rotation"], dtype=np.float64)
            if rotation_raw.shape == (3, 3):
                quat_xyzw = scipy.spatial.transform.Rotation.from_matrix(rotation_raw).as_quat()
            else:
                quat_xyzw = scipy.spatial.transform.Rotation.from_euler(
                    "xyz", rotation_raw, degrees=True
                ).as_quat()
        rotation_quat = quat_xyzw_to_wxyz(quat_xyzw).tolist()

        # Scale: uniform scalar (corrected if override present)
        if correction is not None and correction.scale is not None:
            scale = correction.scale
        else:
            scale = float(transform["scale"])

        # For non-default scenes, reproject positions onto a flat z=0.02 plane
        # and adjust scale to match the new depth. This follows the reference
        # RobotArena implementation which normalizes 3D-reconstructed positions.
        if needs_reprojection and object_positions is None:
            pos_array = np.array(position, dtype=np.float64)
            new_pos, scale_factor = _reproject_to_plane(
                pos_array, camera_intrinsics, camera_extrinsics, z_plane=0.02
            )
            position = new_pos.tolist()
            scale *= scale_factor

        # Physical properties from result.json; object name prefers the corrected
        # physical_properties.json name when available, else result.json, else key.
        physics = physics_by_index.get(i, {})
        obj_name = phys_name_by_index.get(i) or physics.get("object_name", obj_key)
        mass = float(physics.get("mass", 0.1))
        friction = float(physics.get("friction", 0.5))
        surface = str(physics.get("surface", "Default"))

        # Apply position/rotation overrides for perturbation tests
        if object_positions is not None and i < len(object_positions):
            position = list(object_positions[i])
        if object_rotations is not None and i < len(object_rotations):
            rotation_quat = list(object_rotations[i])

        # Determine if object is fixed (non-movable) based on its semantic name.
        # "_target" objects are always movable (they are the manipulation target).
        # "_destination" objects or keyword matches are static receptacles.
        obj_name_lower = obj_name.lower()
        is_fixed = "_target" not in obj_name_lower and (
            "_destination" in obj_name_lower or any(kw in obj_name_lower for kw in FIXED_OBJECT_KEYWORDS)
        )
        if is_fixed:
            LOGGER.debug(f"Classifying '{obj_name}' as fixed (heuristic keyword match)")

        objects.append(
            ObjectConfig(
                name=obj_name,
                mesh_path=mesh_path,
                position=position,
                rotation_quat=rotation_quat,
                scale=scale,
                mass=mass,
                friction=friction,
                surface=surface,
                fixed=is_fixed,
            )
        )

    return objects


def list_available_scenes(data_root: str, robot_name: str = "bridge") -> List[str]:
    """List all available scene names under data_root/<robot_name>/."""
    scenes_dir = os.path.join(data_root, robot_name)
    if not os.path.isdir(scenes_dir):
        raise FileNotFoundError(f"Scene data directory not found: {scenes_dir}")

    def _natural_sort_key(name: str) -> list:
        """Sort key that handles embedded numbers naturally (scene2 < scene19)."""
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]

    scenes = sorted(
        (entry for entry in os.listdir(scenes_dir) if os.path.isdir(os.path.join(scenes_dir, entry))),
        key=_natural_sort_key,
    )
    return scenes


def list_available_backgrounds(data_root: str) -> List[str]:
    """List all available generic background image paths."""
    bg_dir = os.path.join(data_root, "..", "background")
    if not os.path.isdir(bg_dir):
        return []

    return sorted(os.path.join(bg_dir, f) for f in os.listdir(bg_dir) if f.endswith(".png"))


def resolve_genesis_surface(gs_module: Any, surface_name: str) -> Any:
    """Resolve a surface name string to a Genesis surface object.

    Args:
        gs_module: The ``genesis`` module (passed to avoid top-level import).
        surface_name: One of "Default", "Glass", "Water", "Emission",
            "Plastic", "Rough", "Smooth".
    """
    surface_constructors = {
        "Default": gs_module.surfaces.Default,
        "Glass": gs_module.surfaces.Glass,
        "Water": gs_module.surfaces.Water,
        "Emission": gs_module.surfaces.Emission,
        "Plastic": gs_module.surfaces.Plastic,
        "Rough": gs_module.surfaces.Rough,
        "Smooth": gs_module.surfaces.Smooth,
    }
    constructor = surface_constructors.get(surface_name, gs_module.surfaces.Default)
    return constructor()


def compute_camera_fov_deg(intrinsics: np.ndarray) -> float:
    """Compute vertical FOV in degrees from a 3x3 camera intrinsics matrix.

    Uses the principal-point offset ``cy`` as a proxy for the original image
    half-height (``H = 2 * cy``) following the RobotArena convention.
    """
    fy = float(intrinsics[1, 1])
    cy = float(intrinsics[1, 2])
    original_height = 2.0 * cy
    return float(2.0 * np.degrees(np.arctan(original_height / (2.0 * fy))))


def compute_camera_to_world(
    extrinsics: np.ndarray,
    camera_offset: Optional[List[float]] = None,
) -> np.ndarray:
    """Compute the Genesis camera-to-world 4x4 matrix from extrinsics.

    Applies the Genesis camera rotation correction ``diag(1, -1, -1)`` and
    an optional translation offset.
    """
    camera_to_world = np.linalg.inv(extrinsics)
    rotation_correction = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    camera_to_world[:3, :3] = camera_to_world[:3, :3] @ rotation_correction
    if camera_offset is not None:
        camera_to_world[:3, 3] += np.array(camera_offset, dtype=np.float64)
    return camera_to_world


def composite_foreground_over_background(
    rgb: np.ndarray,
    seg: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    """Replace background pixels (segmentation <= 0) with *background*, resized to match.
    Handles a single ``(H, W, 3)`` frame or a batched ``(K, H, W, 3)`` stack."""
    single = rgb.ndim == 3
    if single:
        rgb, seg = rgb[None], seg[None]
    h, w = rgb.shape[1], rgb.shape[2]
    if background.shape[:2] != (h, w):
        background = cv2.resize(background, (w, h))
    seg_hw = seg[..., 0] if seg.ndim == 4 else seg  # (K, H, W)
    out = np.where((seg_hw > 0)[..., None], rgb, background[None]).astype(np.uint8)
    return out[0] if single else out
