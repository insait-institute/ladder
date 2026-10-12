import threading
from typing import Optional

import numpy as np

from ladder.components.inference.models.harness.arm import Arm
from ladder.components.inference.models.harness.perception import Detection, Scene
from ladder.components.inference.models.harness.tools.geometry import LIFT_M, column_radius, reach_windows
from ladder.components.inference.models.harness.tools.hands import Hands
from ladder.components.inference.models.harness.tools.report import Report
from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)

_CORRIDOR_RADIUS_M = 0.06


def clearance_to(hands: Hands, target_m: np.ndarray) -> float:
    hang = 0.5 * float(hands.held.extent_m[2]) if hands.held is not None else 0.0
    return (
        hands.scene.clearance_z(hands.arm.position_m, target_m, _CORRIDOR_RADIUS_M)
        + hands.arm.config.transit_clearance_m
        + hang
    )


def close_the_gap(arm: Arm, grasp_m: np.ndarray, support_z: float, gate_m: float, yaw_rad: float) -> None:
    short = float(arm.position_m[2] - grasp_m[2])
    if short < gate_m:
        return
    aim = grasp_m.copy()
    aim[2] = max(grasp_m[2] - short, support_z)
    arm.move_to(aim, yaw_rad, tolerance_m=short, ramped=True)


def holding(arm: Arm, scene: Scene, detection: Detection) -> bool:
    scene.refresh()
    radius = column_radius(float(np.min(detection.extent_m[:2])))
    left = float(scene.column_top(detection.center_m[None, :2], radius)[0])
    support = scene.support_z()
    held = left < 0.5 * (support + detection.top_m)
    LOGGER.info(
        f"[{threading.current_thread().name}] hold check {detection.name!r}: column "
        f"{left:.3f} vs support {support:.3f} top {detection.top_m:.3f} "
        f"[arm steps {arm.steps}] -> {'HELD' if held else 'EMPTY'}"
    )
    return held


def grasp(
    hands: Hands,
    detection: Detection,
    grasp_m: np.ndarray,
    yaw: float,
    object_name: str,
    width: Optional[float] = None,
) -> Report:
    arm, scene = hands.arm, hands.scene
    support = scene.support_z()
    if width is None:
        width = float(np.min(detection.extent_m[:2]))
    local_top = detection.top_near(grasp_m[:2], column_radius(width))
    precision, vertical = reach_windows(width, local_top, support)
    arm.open_gripper(wait=False)
    aim = grasp_m.copy()
    aim[2] = max(grasp_m[2] - arm.bias_z_m, support)
    arm.transit_to(aim, clearance_to(hands, grasp_m), yaw, tolerance_m=precision, ramped=True)
    close_the_gap(arm, grasp_m, support, vertical, yaw)
    miss = arm.position_m - grasp_m
    arm.close_gripper()
    arm.move_to(np.array([grasp_m[0], grasp_m[1], detection.top_m + LIFT_M]), yaw, ramped=True)
    # Where the jaws report their own width, believe them: the depth column under the hand is
    # wrong in both directions, calling a shut gripper full and a full one empty.
    held = (
        arm.openness() >= arm.config.min_grip_openness
        if arm.config.grip_from_openness
        else holding(arm, scene, detection)
    )
    LOGGER.info(
        f"[{threading.current_thread().name}] grasp {object_name!r}: extent "
        f"{np.round(detection.extent_m, 3).tolist()} yaw {yaw:.2f} at z {grasp_m[2]:.3f} "
        f"(support {support:.3f} local top {local_top:.3f}) missed by "
        f"{np.round(miss, 4).tolist()} asking {precision:.4f}/{vertical:.4f} -> openness "
        f"{arm.openness():.3f} [{arm.steps} steps] profile "
        f"{[(round(float(w), 3), round(float(h), 3)) for w, h in detection.cross_section_m()]} "
        f"-> {'HELD' if held else 'EMPTY'}"
    )
    if held:
        hands.held = detection
        hands.openness = float(arm.openness())
        return Report(True, f"picked {object_name!r}")
    return Report(
        False,
        f"failed to pick {object_name!r}: the gripper closed on nothing. It measures "
        f"{width * 1000:.0f} mm across and {(local_top - support) * 1000:.0f} mm tall.",
    )
