import functools
import io
from typing import Any, Callable, Dict

import numpy as np

from ladder.components.inference.models.harness.perception import ObjectNotFoundError
from ladder.components.inference.models.harness.tools.hands import Hands
from ladder.components.inference.models.harness.tools.report import Report


def run(hands: Hands, tools: Dict[str, Callable[..., Report]], code: str) -> Report:
    arm, scene = hands.arm, hands.scene
    stdout = io.StringIO()
    namespace: Dict[str, Any] = {
        # Bound to this call, not the process: lanes share an interpreter, so redirecting
        # sys.stdout would capture what the other environments printed.
        "print": functools.partial(print, file=stdout),
        **tools,
        "locate": scene.locate,
        "move_to": arm.move_to,
        "open_gripper": arm.open_gripper,
        "close_gripper": arm.close_gripper,
        "support_z": scene.support_z,
        "np": np,
    }
    namespace.pop("run_code", None)
    start = arm.steps
    try:
        exec(code, namespace)  # noqa: S102 -- running model-written code is the point
    except ObjectNotFoundError as error:
        return Report(False, f"code stopped: {error}\n{stdout.getvalue()}")
    except Exception as error:  # noqa: BLE001 -- any error is a report, never a crashed episode
        return Report(False, f"code raised {type(error).__name__}: {error}\n{stdout.getvalue()}")
    spent = arm.steps - start
    return Report(True, f"code ran, {spent} control steps\n{stdout.getvalue()}".strip())
