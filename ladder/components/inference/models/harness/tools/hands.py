import dataclasses
from typing import Optional

from ladder.components.inference.models.harness.arm import Arm
from ladder.components.inference.models.harness.perception import Detection, Scene


@dataclasses.dataclass
class Hands:
    arm: Arm
    scene: Scene
    held: Optional[Detection] = None
    # What the jaws read right after they closed. A later reading below half of it means the
    # object left the gripper on the way.
    openness: float = 0.0
