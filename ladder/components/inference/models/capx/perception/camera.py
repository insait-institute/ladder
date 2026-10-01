from typing import Any, Dict, List, Optional

import numpy as np

from ladder.components.inference.api.core.observation import METADATA_CAMERAS_KEY


def box_pixels(box_xyxy: np.ndarray, shape: tuple, stride: int = 1) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    x0, y0, x1, y1 = (float(v) for v in np.asarray(box_xyxy, dtype=np.float64).reshape(4))
    columns = np.arange(max(0, int(np.floor(x0))), min(width, int(np.ceil(x1)) + 1), max(1, stride))
    rows = np.arange(max(0, int(np.floor(y0))), min(height, int(np.ceil(y1)) + 1), max(1, stride))
    if columns.size == 0 or rows.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    grid = np.meshgrid(columns, rows, indexing="xy")
    return np.stack([grid[0].ravel(), grid[1].ravel()], axis=1).astype(np.float64)


def names(metadata: Optional[Dict[str, Any]]) -> List[str]:
    return sorted((metadata or {}).get(METADATA_CAMERAS_KEY) or {})
