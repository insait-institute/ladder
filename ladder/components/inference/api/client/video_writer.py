import os
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import PIL.Image

from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)


class VideoWriterOpenCV:
    def __init__(
        self,
        output_path: str,
        fps: int,
        codec: str = 'mp4v',
        transform: Callable[[np.ndarray | PIL.Image.Image], np.ndarray | PIL.Image.Image] = lambda x: x,
    ):
        self.output_path = output_path
        self.codec = codec
        self.fps = fps
        self.transform = transform
        self._writer: Optional[cv2.VideoWriter] = None
        # Whether any frame carried signal -- the streaming form of the all-zero placeholder check.
        self.wrote_signal = False

    def append(self, frame: np.ndarray | PIL.Image.Image) -> None:
        """Stream one frame. The writer opens on the first frame, which is when the size is known."""
        frame = self.transform(frame)
        if self._writer is None:
            width, height = image_wh(frame)
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self._writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
            if self._writer is None:
                raise RuntimeError(f"Failed to create video writer for {self.output_path}")
        array = np.asarray(frame)
        self.wrote_signal = self.wrote_signal or bool(np.any(array))
        self._writer.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self._writer is None:
            return
        self._writer.release()
        self._writer = None
        LOGGER.info(f"Video saved to {self.output_path}")


def image_wh(image: PIL.Image.Image | np.ndarray) -> Tuple[int, int]:
    if isinstance(image, PIL.Image.Image):
        return image.size
    if isinstance(image, np.ndarray):
        return image.shape[:2][::-1]
    raise ValueError(f"Expected PIL.Image.Image or np.ndarray, but got {type(image)}")
