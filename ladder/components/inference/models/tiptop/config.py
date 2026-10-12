from typing import List, Optional

from ladder.core.databib.config import AutoConfig, Config


class TiptopPolicyConfig(Config):
    vlm: AutoConfig
    robot: str = "widowx"
    arm_dof: int = 6
    vlm_max_tokens: int = 8192
    vlm_attempts: int = 4
    vlm_retry_base_s: float = 4.0
    vlm_temperature: float = 0.0
    vlm_reasoning_effort: Optional[str] = "low"
    depth_trunc_m: float = 5.0
    voxel_downsample_size_m: float = 0.0075
    contact_threshold_m: float = 0.01
    mask_erosion_pixels: int = 3
    sam2_model: str = "sam2.1_hiera_large.pt"
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    m2t2_apply_bounds: bool = False
    m2t2_num_points: int = 16384
    m2t2_num_runs: int = 5
    m2t2_mask_threshold: float = 0.035
    m2t2_max_grasps: int = 200
    m2t2_tool_rotation_deg: List[float] = [180.0, 0.0, -90.0]
    grasp_depth_offset_m: float = 0.0
    table_thickness_m: float = 0.05
    table_top_offset_m: float = -0.01
    object_z_filter_tolerance_m: float = -0.01
    num_particles: int = 256
    coll_n_spheres: int = 64
    coll_sphere_radius_m: float = 0.008
    opt_steps: int = 500
    max_planning_time_s: float = 60.0
    max_motion_refine_attempts: int = 128
    pos_err_tolerance_m: Optional[float] = 0.015
    placement_shrink_m: float = 0.0
    placement_in_xy_tolerance_m: Optional[float] = 0.02
    collision_activation_distance_m: float = 0.0
    gripper_contact_allowance_m: float = 0.0
    time_dilation_factor: float = 0.6
    goals_per_plan: Optional[int] = None
    execution: str = "end_effector"
    control_hz: float = 5.0
    waypoint_spacing: Optional[float] = None
    approach_refine: float = 3.0
    gripper_hold_steps: int = 5
    arrive_tolerance_m: float = 0.02
    arrive_tolerance_rad: float = 0.08
    arrive_budget_steps: int = 8
    plan_ee_rotation_deg: List[float] = [0.0, 0.0, 0.0]
    settle_steps: int = 0
    max_workers: int = 32

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.execution not in ("joint_angles", "end_effector"):
            raise ValueError(f"execution must be 'joint_angles' or 'end_effector', got {self.execution!r}")
        if self.robot not in ("widowx", "fr3_robotiq"):
            raise ValueError(f"robot must be 'widowx' or 'fr3_robotiq', got {self.robot!r}")
