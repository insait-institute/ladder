import dataclasses
from typing import Protocol

import numpy as np


@dataclasses.dataclass(frozen=True)
class Detection:
    center_m: np.ndarray  # [3]
    quaternion_wxyz: np.ndarray  # [4]
    extent_m: np.ndarray  # [3]


class Scene(Protocol):
    def locate(self, object_name: str) -> Detection:
        ...
