import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import open3d as o3d
import trimesh
from curobo.types.base import TensorDeviceType
from scipy.spatial import cKDTree

from ladder.components.inference.models.tiptop.config import TiptopPolicyConfig
from ladder.components.inference.models.tiptop.perception import m2t2_to_tool_transform
from ladder.components.inference.models.tiptop.tamp import to_curobo_cuboid, to_curobo_mesh
from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)
TABLE_NAME = "table"


@dataclasses.dataclass
class ProcessedScene:
    table_cuboid: Any
    object_meshes: Dict[str, Any]
    object_clouds: Dict[str, o3d.geometry.PointCloud]
    grasps: Dict[str, Dict[str, Any]]


def _masked_points(
    mask: npt.NDArray[np.bool_],
    xyz_map: npt.NDArray[np.float32],
    rgb_map: npt.NDArray[np.float32],
    erode_pixels: int,
    label: str,
) -> Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    eroded = mask
    if erode_pixels > 0:
        kernel = np.ones((erode_pixels * 2 + 1, erode_pixels * 2 + 1), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    xyz, rgb = (xyz_map[eroded], rgb_map[eroded])
    valid = ~np.isnan(xyz).any(axis=1)
    if valid.sum() < 10 and erode_pixels > 0:
        xyz, rgb = (xyz_map[mask], rgb_map[mask])
        valid = ~np.isnan(xyz).any(axis=1)
    if valid.sum() < 10:
        LOGGER.warn(f"skipping {label}: only {int(valid.sum())} valid depth points")
        return None
    return (xyz[valid], rgb[valid])


def _contact_points(
    xyz_map: npt.NDArray[np.float32], masks: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    contacts = []
    for mask in masks.squeeze(1).astype(bool):
        points = xyz_map[mask]
        points = points[~np.isnan(points).any(axis=1)]
        if len(points) < 10:
            continue
        contacts.append(points[points[:, 2] <= np.percentile(points[:, 2], 10)].mean(axis=0))
    return np.array(contacts) if contacts else np.empty((0, 3))


def fit_table(
    xyz_map: npt.NDArray[np.float32],
    rgb_map: npt.NDArray[np.float32],
    masks: npt.NDArray[np.bool_],
    config: TiptopPolicyConfig,
    max_planes: int = 5,
    contact_threshold_m: float = 0.03,
) -> trimesh.primitives.Box:
    contacts = _contact_points(xyz_map, masks)
    if len(contacts) == 0:
        raise RuntimeError("no object contact points; nothing was detected to rest on a surface")
    valid = ~np.isnan(xyz_map).any(axis=2)
    voxel_size = 0.005
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz_map[valid])
    cloud.colors = o3d.utility.Vector3dVector(rgb_map[valid])
    cloud = cloud.voxel_down_sample(voxel_size=voxel_size)
    remaining, best_score, best = (cloud, -1, None)
    for _ in range(max_planes):
        if len(remaining.points) < 50:
            break
        model, inliers = remaining.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        normal = np.linalg.norm(model[:3])
        distances = np.abs(contacts @ np.asarray(model[:3]) + model[3]) / normal
        score = int((distances < contact_threshold_m).sum())
        if score > best_score:
            best_score, best = (score, remaining.select_by_index(inliers))
        remaining = remaining.select_by_index(inliers, invert=True)
    if best is None or best_score == 0:
        raise RuntimeError(f"no plane carries any of the {len(contacts)} detected objects")
    table, _ = best.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    labels = np.array(table.cluster_dbscan(eps=3 * voxel_size, min_points=10))
    if labels.size == 0 or labels.max() < 0:
        raise RuntimeError("the table plane has no dense cluster after outlier removal")
    table = table.select_by_index(np.where(labels == int(np.bincount(labels[labels >= 0]).argmax()))[0])
    points = np.asarray(table.points)
    xy_min = np.percentile(points[:, :2], 2, axis=0)
    xy_max = np.percentile(points[:, :2], 98, axis=0)
    top = points[:, 2].mean() + config.table_top_offset_m
    bounds = np.stack([np.append(xy_min, top - config.table_thickness_m), np.append(xy_max, top)])
    box = trimesh.primitives.Box(extents=bounds[1] - bounds[0])
    box.apply_translation(bounds.mean(0))
    box.visual.face_colors = np.append((np.asarray(table.colors).mean(0) * 255).astype(np.uint8), 255)
    LOGGER.info(f"table surface at z={points[:, 2].mean():.3f}, slab top {top:.3f}, dims {box.extents}")
    return box


def _hull_meshes(
    xyz_map: npt.NDArray[np.float32],
    rgb_map: npt.NDArray[np.float32],
    masks: npt.NDArray[np.bool_],
    bboxes: List[Dict[str, Any]],
    min_z: float,
    config: TiptopPolicyConfig,
) -> Tuple[Dict[str, trimesh.Trimesh], Dict[str, o3d.geometry.PointCloud]]:
    meshes: Dict[str, trimesh.Trimesh] = {}
    clouds: Dict[str, o3d.geometry.PointCloud] = {}
    for mask, bbox in zip(masks.squeeze(1).astype(bool), bboxes):
        label = bbox["label"]
        points = _masked_points(mask, xyz_map, rgb_map, config.mask_erosion_pixels, label)
        if points is None:
            continue
        xyz, rgb = points
        above = xyz[:, 2] > min_z
        if not above.any():
            LOGGER.warn(f"skipping {label}: no points above the table")
            continue
        xyz, rgb = (xyz[above], rgb[above])
        projected = xyz.copy()
        projected[:, 2] = xyz[:, 2].min()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.vstack([xyz, projected]))
        cloud.colors = o3d.utility.Vector3dVector(np.vstack([rgb, rgb]))
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
        clouds[label] = cloud
        try:
            hull, _ = cloud.compute_convex_hull()
            mesh = trimesh.Trimesh(
                vertices=np.asarray(hull.vertices), faces=np.asarray(hull.triangles), process=True
            )
            mesh.visual.face_colors = np.append(np.asarray(cloud.colors).mean(0) * 255, 255).astype(np.uint8)
            meshes[label] = mesh
        except Exception as error:  # noqa: BLE001
            LOGGER.warn(f"no convex hull for {label}: {error}")
            clouds.pop(label, None)
    return (meshes, clouds)


def _associate_grasps(
    grasps: Dict[str, Dict[str, Any]], clouds: Dict[str, o3d.geometry.PointCloud], threshold_m: float
) -> Dict[str, Dict[str, Any]]:
    labels = list(clouds)
    stacked = np.vstack([np.asarray(clouds[label].points) for label in labels])
    owner = np.concatenate([[label] * len(clouds[label].points) for label in labels])
    tree = cKDTree(stacked)
    poses, confidences, contacts, owners = ([], [], [], [])
    for group in grasps.values():
        if len(group["contacts"]) == 0:
            continue
        distances, nearest = tree.query(group["contacts"])
        near = distances < threshold_m
        poses.append(group["poses"][near])
        confidences.append(group["confidences"][near])
        contacts.append(group["contacts"][near])
        owners.append(owner[nearest][near])
    if not poses:
        return {
            label: {"poses": np.zeros((0, 4, 4)), "confidences": np.zeros(0), "contacts": np.zeros((0, 3))}
            for label in labels
        }
    poses = np.concatenate(poses)
    confidences = np.concatenate(confidences)
    contacts = np.concatenate(contacts)
    owners = np.concatenate(owners)
    associated = {}
    for label in labels:
        selected = owners == label
        associated[label] = {
            "poses": poses[selected],
            "confidences": confidences[selected],
            "contacts": contacts[selected],
        }
        LOGGER.info(f"{label}: {int(selected.sum())} grasps within {threshold_m * 100:.1f} cm")
    return associated


def build_scene(
    xyz_map: npt.NDArray[np.float32],
    rgb_map: npt.NDArray[np.float32],
    masks: npt.NDArray[np.bool_],
    bboxes: List[Dict[str, Any]],
    grasps: Dict[str, Dict[str, Any]],
    config: TiptopPolicyConfig,
) -> ProcessedScene:
    table_box = fit_table(xyz_map, rgb_map, masks, config)
    table_cuboid = to_curobo_cuboid(table_box, TABLE_NAME)
    min_z = table_box.bounds[1, 2] + config.object_z_filter_tolerance_m
    meshes, clouds = _hull_meshes(xyz_map, rgb_map, masks, bboxes, min_z, config)
    if not clouds:
        raise ValueError("no detected object has enough depth to reconstruct")
    associated = _associate_grasps(grasps, clouds, config.contact_threshold_m)
    tensor_args = TensorDeviceType()
    object_meshes = {}
    for label, mesh in meshes.items():
        curobo_mesh = to_curobo_mesh(mesh, label)
        object_meshes[label] = curobo_mesh
        group = associated[label]
        world_from_grasp = group["poses"] @ m2t2_to_tool_transform(
            config.grasp_depth_offset_m, tuple(config.m2t2_tool_rotation_deg)
        )
        origin = np.eye(4)
        origin[:3, 3] = np.asarray(curobo_mesh.pose)[:3]
        group["grasps_obj"] = tensor_args.to_device(np.linalg.inv(origin) @ world_from_grasp)
        group["confidences_pt"] = tensor_args.to_device(group["confidences"])
        associated[label] = group
    return ProcessedScene(table_cuboid, object_meshes, clouds, associated)


def world_bounds(obj: Any) -> Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    pose = np.asarray(getattr(obj, "pose", None), dtype=float).reshape(-1)
    if pose.size < 3:
        return None
    dims = getattr(obj, "dims", None)
    if dims is not None:
        half = np.asarray(dims, dtype=float).reshape(-1)[:3] / 2.0
        return (pose[:3] - half, pose[:3] + half)
    vertices = getattr(obj, "vertices", None)
    if vertices is None or not len(vertices):
        return None
    vertices = np.asarray(vertices, dtype=float)
    if np.abs(vertices).max() < 1.0:
        vertices = vertices + pose[:3]
    return (vertices.min(axis=0), vertices.max(axis=0))


def label_key(label: str) -> str:
    key = "".join((char if char.isalnum() else "_" for char in str(label).lower()))
    for article in ("the_", "an_", "a_"):
        if key.startswith(article):
            key = key[len(article) :]
            break
    return key.replace("_", "")


def goal_already_holds(
    atom: Dict[str, Any], object_meshes: Dict[str, Any], table_cuboid: Any, surface_gap_m: float = 0.012
) -> bool:
    if atom.get("predicate") != "on" or len(atom.get("args", ())) != 2:
        return False
    movable, surface = atom["args"]
    obj = object_meshes.get(movable)
    support = object_meshes.get(surface) or (table_cuboid if surface == table_cuboid.name else None)
    if obj is None or support is None:
        return False
    obj_bounds, support_bounds = (world_bounds(obj), world_bounds(support))
    if obj_bounds is None or support_bounds is None:
        return False
    centre = (obj_bounds[0][:2] + obj_bounds[1][:2]) / 2.0
    within = bool(np.all(centre >= support_bounds[0][:2]) and np.all(centre <= support_bounds[1][:2]))
    resting = -surface_gap_m <= obj_bounds[0][2] - support_bounds[1][2] <= surface_gap_m
    return within and resting
