import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)

_BPE_ASSET = "assets/bpe_simple_vocab_16e6.txt.gz"
_CONFIDENCE = 0.0
_BOX_MATCH_PX = 1.0
_MAX_RESULTS = 20


class Sam3Segmenter:
    def __init__(
        self, device: str = "cuda", confidence_threshold: float = _CONFIDENCE, interactive: bool = True
    ) -> None:
        self._device = device
        self._confidence = confidence_threshold
        self._interactive = interactive
        self._model: Any = None
        self._processor: Any = None
        self._lock = threading.Lock()
        self._cached: Optional[Tuple[int, np.ndarray, np.ndarray]] = None

    def warmup(self) -> None:
        import torch

        with self._lock:
            if self._model is not None:
                return
            import sam3
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            if self._device.startswith("cuda") and not torch.cuda.is_available():
                LOGGER.warn("Sam3Segmenter: no CUDA device, running SAM3 on the CPU")
                self._device = "cpu"
            bpe = Path(sam3.__file__).resolve().parent / _BPE_ASSET
            self._model = build_sam3_image_model(
                bpe_path=str(bpe),
                device=self._device,
                enable_inst_interactivity=self._interactive,
            )
            self._processor = Sam3Processor(
                self._model, device=self._device, confidence_threshold=self._confidence
            )
        LOGGER.info(f"Sam3Segmenter: sam3 on {self._device} at threshold {self._confidence}")

    def segment_text(self, image: np.ndarray, text_prompt: str) -> List[Dict[str, Any]]:
        import torch

        self.warmup()
        frame = _pil(image)
        with self._lock, torch.autocast(self._device.split(":")[0], dtype=torch.bfloat16):
            state = self._processor.set_image(frame)
            output = self._processor.set_text_prompt(prompt=str(text_prompt), state=state)
            found = _results(output, boxes=True, label=str(text_prompt))
        if found:
            self._cached = (id(image), np.asarray(found[0]["box"], dtype=np.float64), found[0]["mask"])
        return found

    def box(self, image: np.ndarray, object_name: str) -> np.ndarray:
        frame = np.asarray(image)
        height, width = frame.shape[:2]
        found = self.segment_text(frame, object_name)
        if not found:
            raise ValueError(f"SAM3 does not see a {object_name!r} in this view")
        pixels = np.asarray(found[0]["box"], dtype=np.float64)
        box = np.clip(pixels / np.array([width, height, width, height]), 0.0, 1.0)
        if box[2] - box[0] <= 0.0 or box[3] - box[1] <= 0.0:
            raise ValueError(f"SAM3 returned an empty box for {object_name!r}: {pixels.tolist()}")
        return box

    def mask(self, image: np.ndarray, box_xyxy: np.ndarray) -> Optional[np.ndarray]:
        import torch

        wanted = np.asarray(box_xyxy, dtype=np.float64).reshape(4)
        cached = self._cached
        if (
            cached is not None
            and id(image) == cached[0]
            and np.allclose(cached[1], wanted, atol=_BOX_MATCH_PX)
        ):
            return cached[2]
        self.warmup()
        frame = _pil(image)
        height, width = np.asarray(image).shape[:2]
        centre = [
            0.5 * (wanted[0] + wanted[2]) / width,
            0.5 * (wanted[1] + wanted[3]) / height,
            (wanted[2] - wanted[0]) / width,
            (wanted[3] - wanted[1]) / height,
        ]
        with self._lock, torch.autocast(self._device.split(":")[0], dtype=torch.bfloat16):
            state = self._processor.set_image(frame)
            output = self._processor.add_geometric_prompt(box=centre, label=True, state=state)
            found = _results(output, boxes=False, label="")
        return found[0]["mask"] if found else None


def _pil(image: np.ndarray):
    from PIL import Image

    return Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")


def _results(state: Dict[str, Any], boxes: bool, label: str) -> List[Dict[str, Any]]:
    masks = _numpy(state.get("masks"))
    scores = _numpy(state.get("scores"))
    if masks is None or scores is None or not len(scores):
        return []
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks.squeeze(1)
    boxed = _numpy(state.get("boxes")) if boxes else None
    order = np.argsort(scores.reshape(-1))[::-1][:_MAX_RESULTS]
    found: List[Dict[str, Any]] = []
    for index in order:
        entry: Dict[str, Any] = {"mask": masks[index] > 0, "score": float(scores[index])}
        if boxed is not None:
            entry["box"] = [float(v) for v in boxed[index]]
            entry["label"] = label
        found.append(entry)
    return found


def _numpy(tensor: Any) -> Optional[np.ndarray]:
    if tensor is None:
        return None
    detach = getattr(tensor, "detach", None)
    if detach is not None:
        tensor = detach().float().cpu()
    return np.asarray(tensor)
