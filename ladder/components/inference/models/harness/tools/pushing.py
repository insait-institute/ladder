import math
import threading

import numpy as np

from ladder.components.inference.models.harness.perception import ObjectNotFoundError
from ladder.components.inference.models.harness.tools.geometry import (
    COLUMN_FLOOR_M,
    LIFT_M,
    MIN_ABOVE_SUPPORT_M,
)
from ladder.components.inference.models.harness.tools.grasping import clearance_to
from ladder.components.inference.models.harness.tools.hands import Hands
from ladder.components.inference.models.harness.tools.report import Report
from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)

_PUSH_HEIGHT_FRACTION = 0.3
_PUSH_STANDOFF_M = 0.04


def push(hands: Hands, object_name: str, destination_name: str) -> Report:
    arm, scene = hands.arm, hands.scene
    if hands.held is not None:
        return Report(False, f"cannot push while holding {hands.held.name!r}: place it first")
    scene.refresh()
    detection = scene.locate(object_name)
    destination = scene.locate(destination_name, surface=True)
    support = scene.support_z()
    travel = destination.center_m[:2] - detection.center_m[:2]
    distance = float(np.linalg.norm(travel))
    if distance < COLUMN_FLOOR_M:
        return Report(True, f"{object_name!r} is already on {destination_name!r}")
    heading = travel / distance
    radius = 0.5 * float(np.max(detection.extent_m[:2]))
    height = max(
        support + MIN_ABOVE_SUPPORT_M,
        min(
            support + _PUSH_HEIGHT_FRACTION * (detection.top_m - support),
            detection.foot_near(detection.center_m[:2], radius),
        ),
    )
    start = detection.center_m[:2] - heading * (radius + _PUSH_STANDOFF_M)
    finish = destination.center_m[:2] - heading * radius
    arm.close_gripper()
    arm.transit_to(np.array([start[0], start[1], height]), clearance_to(hands, detection.center_m))
    arm.move_to(np.array([finish[0], finish[1], height]))
    moved = arm.position_m
    LOGGER.info(
        f"[{threading.current_thread().name}] push {object_name!r} -> {destination_name!r}: "
        f"{distance * 1000:.0f} mm along {np.round(heading, 2).tolist()} at z {height:.3f} "
        f"(support {support:.3f} top {detection.top_m:.3f} radius {radius:.3f}), reached "
        f"{np.round(moved, 3).tolist()} [arm steps {arm.steps}]"
    )
    arm.move_to(np.array([moved[0], moved[1], moved[2] + LIFT_M]))
    scene.refresh()
    try:
        arrived = scene.locate(object_name)
    except ObjectNotFoundError:
        return Report(False, f"pushed {object_name!r}, but it can no longer be seen to check")
    offset = arrived.center_m[:2] - destination.center_m[:2]
    short = float(np.linalg.norm(offset))
    turn = np.array(
        [
            [math.cos(destination.yaw_rad), math.sin(destination.yaw_rad)],
            [-math.sin(destination.yaw_rad), math.cos(destination.yaw_rad)],
        ]
    )
    if np.all(np.abs(turn @ offset) <= 0.5 * destination.extent_m[:2]):
        return Report(True, f"pushed {object_name!r} onto {destination_name!r}")
    return Report(
        False,
        f"pushed {object_name!r}, and it is still {short * 1000:.0f} mm short of "
        f"{destination_name!r}. Push it again.",
    )
