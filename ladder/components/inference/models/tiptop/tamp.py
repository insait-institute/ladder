import dataclasses
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
import trimesh
from curobo.geom.types import Cuboid, Mesh, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from cutamp.algorithm import run_cutamp
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs import TAMPEnvironment
from cutamp.robots import load_fr3_robotiq_container, load_widowx_container
from cutamp.robots.franka_robotiq import fr3_robotiq_curobo_cfg, get_fr3_robotiq_ik_solver
from cutamp.robots.widowx import get_widowx_ik_solver, widowx_curobo_cfg
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
from cutamp.tamp_domain import HandEmpty, Holding, Near, On
from cutamp.task_planning.constraints import KinematicConstraint, StablePlacement
from cutamp.utils.common import sample_between_bounds

from ladder.components.inference.models.tiptop.config import TiptopPolicyConfig
from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)
SHRINK_MAX_FRACTION = 0.4


def to_curobo_cuboid(box: trimesh.primitives.Box, name: str) -> Any:
    centre, extents = (box.centroid, box.extents)
    colour = box.visual.main_color[:3] if hasattr(box.visual, "main_color") else [200, 200, 200]
    return Cuboid(
        name=name,
        dims=[float(extents[0]), float(extents[1]), float(extents[2])],
        pose=[float(centre[0]), float(centre[1]), float(centre[2]), 1.0, 0.0, 0.0, 0.0],
        color=list(colour),
    )


def to_curobo_mesh(mesh: trimesh.Trimesh, label: str) -> Any:
    vertices = np.asarray(mesh.vertices)
    centroid = vertices.mean(0)
    colours = getattr(mesh.visual, "vertex_colors", None)
    vertex_colors = (
        np.asarray(colours)[:, :3] / 255.0 if colours is not None else np.ones((len(vertices), 3)) * 0.5
    )
    return Mesh(
        name=label,
        vertices=(vertices - centroid).tolist(),
        faces=np.asarray(mesh.faces).tolist(),
        vertex_colors=vertex_colors.tolist(),
        pose=[float(centroid[0]), float(centroid[1]), float(centroid[2]), 1.0, 0.0, 0.0, 0.0],
    )


def _robot_parts(robot: str) -> Tuple[Any, Any, Any]:
    if robot == "widowx":
        return (get_widowx_ik_solver, load_widowx_container, widowx_curobo_cfg)
    return (get_fr3_robotiq_ik_solver, load_fr3_robotiq_container, fr3_robotiq_curobo_cfg)


def build_solvers(config: TiptopPolicyConfig) -> Tuple[Any, Any, Any]:
    make_ik, load_container, robot_cfg_of = _robot_parts(config.robot)
    world = WorldConfig(
        cuboid=[Cuboid(name="table", dims=[0.01, 0.01, 0.01], pose=[99.9, 99.9, 99.9, 1.0, 0.0, 0.0, 0.0])]
    )
    ik_solver = make_ik(world)
    container = load_container(TensorDeviceType())
    torch.cuda.synchronize()
    for _ in range(8):
        joints = sample_between_bounds(config.num_particles, bounds=container.joint_limits)
        ik_solver.solve_batch(container.kin_model.get_state(joints).ee_pose)
    torch.cuda.synchronize()
    robot_cfg = robot_cfg_of()
    robot_cfg["robot_cfg"]["kinematics"]["extra_collision_spheres"]["attached_object"] = config.coll_n_spheres
    motion_gen = MotionGen(
        MotionGenConfig.load_from_robot_config(
            robot_cfg=robot_cfg,
            world_model=world,
            use_cuda_graph=True,
            collision_activation_distance=config.collision_activation_distance_m,
            position_threshold=max(0.01, float(config.pos_err_tolerance_m or 0.0)),
            rotation_threshold=0.1,
        )
    )
    torch.cuda.synchronize()
    for _ in range(16):
        motion_gen.warmup()
    torch.cuda.synchronize()
    return (ik_solver, motion_gen, world)


def build_tamp_config(config: TiptopPolicyConfig) -> Any:
    return TAMPConfiguration(
        experiment_root=os.path.join(tempfile.gettempdir(), f"cutamp-{os.getpid()}"),
        num_particles=config.num_particles,
        max_loop_dur=config.max_planning_time_s,
        max_skeleton_sampling_dur=config.max_planning_time_s,
        num_opt_steps=config.opt_steps,
        m2t2_grasps=True,
        prop_satisfying_break=0.1,
        robot=config.robot,
        curobo_plan=True,
        max_motion_refine_attempts=config.max_motion_refine_attempts,
        warmup_ik=False,
        warmup_motion_gen=False,
        num_initial_plans=10,
        cache_subgraphs=True,
        world_activation_distance=config.collision_activation_distance_m,
        movable_activation_distance=0.01,
        time_dilation_factor=config.time_dilation_factor,
        placement_check="obb",
        placement_shrink_dist=config.placement_shrink_m,
        enable_visualizer=False,
        coll_sphere_radius=config.coll_sphere_radius_m,
        gripper_contact_allowance_m=config.gripper_contact_allowance_m,
        near_placement=False,
    )


def create_environment(
    object_meshes: Dict[str, Any],
    table_cuboid: Any,
    planned_atoms: List[Dict[str, Any]],
    task_atoms: List[Dict[str, Any]],
) -> Tuple[Any, List[Any]]:
    if table_cuboid.name in object_meshes:
        object_meshes = {k: v for k, v in object_meshes.items() if k != table_cuboid.name}
    known = set(object_meshes) | {table_cuboid.name}
    for atom in planned_atoms:
        for arg in atom.get("args", []):
            if arg not in known:
                raise ValueError(
                    f"goal {atom['predicate']}({', '.join(atom['args'])}) names an object that was not detected; known: {sorted(known)}"
                )
    surface_labels = {
        atom["args"][1] for atom in task_atoms if atom["predicate"] == "on" and len(atom["args"]) == 2
    }
    movables = [mesh for label, mesh in object_meshes.items() if label not in surface_labels]
    surfaces = [mesh for label, mesh in object_meshes.items() if label in surface_labels]
    goal_state = set()
    holding = False
    for atom in planned_atoms:
        predicate, args = (atom["predicate"], atom["args"])
        if predicate == "on" and len(args) == 2:
            goal_state.add(On.ground(args[0], args[1]))
        elif predicate == "holding" and len(args) == 1:
            holding = True
            goal_state.add(Holding.ground(args[0]))
        elif predicate == "near" and len(args) == 2:
            goal_state.add(Near.ground(args[0], args[1]))
        else:
            LOGGER.warn(f"ignoring unsupported goal atom {atom}")
    if not holding:
        goal_state.add(HandEmpty.ground())
    all_surfaces = [table_cuboid, *surfaces]
    env = TAMPEnvironment(
        name="tiptop_cutamp",
        movables=movables,
        statics=list(all_surfaces),
        type_to_objects={"Movable": movables, "Surface": all_surfaces},
        goal_state=frozenset(goal_state),
    )
    LOGGER.info(f"movables {[m.name for m in movables]}, surfaces {[s.name for s in all_surfaces]}")
    return (env, all_surfaces)


def _afforded_shrink(requested: float, all_surfaces: List[Any]) -> float:
    smallest = None
    for surface in all_surfaces:
        extents = getattr(surface, "extents", None)
        if extents is None or len(extents) < 2:
            continue
        half = min(float(extents[0]), float(extents[1])) / 2.0
        smallest = half if smallest is None else min(smallest, half)
    return requested if smallest is None else min(requested, SHRINK_MAX_FRACTION * smallest)


def run_planning(
    env: Any,
    tamp_config: Any,
    config: TiptopPolicyConfig,
    q_init: npt.NDArray[np.float32],
    ik_solver: Any,
    grasps: Dict[str, Any],
    motion_gen: Any,
    all_surfaces: List[Any],
) -> Tuple[Optional[List[Dict[str, Any]]], float, Optional[str]]:
    afforded = _afforded_shrink(float(tamp_config.placement_shrink_dist), all_surfaces)
    if afforded < float(tamp_config.placement_shrink_dist):
        LOGGER.info(f"placement shrink {tamp_config.placement_shrink_dist:.3f} -> {afforded:.3f} m")
        tamp_config = dataclasses.replace(tamp_config, placement_shrink_dist=afforded)
    constraint_to_tol = default_constraint_to_tol.copy()
    constraint_to_mult = default_constraint_to_mult.copy()
    if config.pos_err_tolerance_m is not None:
        constraint_to_tol[KinematicConstraint.type] = {
            **constraint_to_tol[KinematicConstraint.type],
            "pos_err": float(config.pos_err_tolerance_m),
        }
    in_xy = 0.01 if config.placement_in_xy_tolerance_m is None else float(config.placement_in_xy_tolerance_m)
    for surface in all_surfaces:
        constraint_to_tol[StablePlacement.type][f"{surface.name}_in_xy"] = in_xy
        constraint_to_tol[StablePlacement.type][f"{surface.name}_support"] = 0.01
        constraint_to_mult[StablePlacement.type][f"{surface.name}_support"] = 1.0
    start = time.perf_counter()
    plan, _, failure = run_cutamp(
        env,
        tamp_config,
        CostReducer(constraint_to_mult),
        ConstraintChecker(constraint_to_tol),
        q_init=q_init,
        ik_solver=ik_solver,
        grasps=grasps,
        motion_gen=motion_gen,
        experiment_dir=None,
    )
    elapsed = time.perf_counter() - start
    if plan is None:
        LOGGER.warn(f"cuTAMP found no plan in {elapsed:.1f}s: {failure}")
    else:
        LOGGER.info(f"cuTAMP found a {len(plan)}-step plan in {elapsed:.1f}s")
    return (plan, elapsed, failure)


def serialize_plan(
    plan: List[Dict[str, Any]], q_init: npt.NDArray[np.float32], kin_model: Any = None
) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []
    for step in plan:
        if step["type"] == "trajectory":
            serialized = {
                "type": "trajectory",
                "label": step["label"],
                "positions": step["plan"].position.cpu().numpy(),
                "velocities": step["plan"].velocity.cpu().numpy(),
                "dt": step["dt"],
            }
            if kin_model is not None:
                state = kin_model.get_state(step["plan"].position.contiguous())
                serialized["ee_poses"] = {
                    "position": state.ee_pose.position.cpu().numpy(),
                    "quaternion_wxyz": state.ee_pose.quaternion.cpu().numpy(),
                }
            steps.append(serialized)
        elif step["type"] == "gripper":
            steps.append({"type": "gripper", "label": step["label"], "action": step["action"]})
    return {"version": "1.1.0", "q_init": q_init, "steps": steps}
