import threading

import numpy as np

from ladder.components.inference.models.harness.tools.geometry import LIFT_M, PLACE_CLEARANCE_M, release_point
from ladder.components.inference.models.harness.tools.grasping import clearance_to
from ladder.components.inference.models.harness.tools.hands import Hands
from ladder.components.inference.models.harness.tools.report import Report
from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)


def place(hands: Hands, destination_name: str) -> Report:
    arm, scene = hands.arm, hands.scene
    if not arm.holding:
        return Report(False, "failed to place: nothing is held")
    destination = scene.locate(destination_name, surface=True)
    half_height = 0.5 * float(hands.held.extent_m[2]) if hands.held is not None else 0.0
    held_xy = hands.held.extent_m[:2] if hands.held is not None else np.zeros(2)
    spot = release_point(scene, destination, held_xy)
    target = np.array([spot[0], spot[1], spot[2] + half_height])
    if arm.config.release_above_rim:
        target[2] = max(target[2], destination.top_m + PLACE_CLEARANCE_M + half_height)
    clearance = clearance_to(hands, target)
    arm.transit_to(target, clearance)
    if hands.openness > 0.0 and arm.openness() < 0.5 * hands.openness:
        dropped, hands.held, hands.openness = hands.held, None, 0.0
        name = dropped.name if dropped is not None else "the object"
        return Report(
            False,
            f"failed to place {name!r}: it is no longer in the jaws, so it was "
            f"lost on the way. Pick it up again before placing it.",
        )
    LOGGER.info(
        f"[{threading.current_thread().name}] release over {destination_name!r}: clearance "
        f"{clearance:.3f} dest top "
        f"{destination.top_m:.3f} spot "
        f"{np.round(spot, 3).tolist()} half_height {half_height:.3f} target z {target[2]:.3f} "
        f"reached {arm.position_m[2]:.3f} -> openness {arm.openness():.3f} "
        f"[arm steps {arm.steps}]"
    )
    arm.open_gripper()
    held_height = 2.0 * half_height
    hands.held = None
    retreat = arm.position_m
    retreat[2] += LIFT_M
    arm.move_to(retreat)
    if arm.config.verify_place and held_height > 0.0:
        scene.refresh()
        # The column only shows an object that ends up on top. A taller destination hides it,
        # and the column keeps reading that destination's own rim.
        hidden = destination.top_m - scene.support_z() > held_height
        column = float(
            scene.column_top(np.asarray([[spot[0], spot[1]]]), max(half_height, PLACE_CLEARANCE_M))[0]
        )
        if not hidden and column < spot[2] + 0.5 * half_height:
            return Report(
                False,
                f"released over {destination_name!r} but it is not there: the column "
                f"reads {column:.3f} against {spot[2]:.3f}. Pick it up again.",
            )
    return Report(True, f"released over {destination_name!r}")
