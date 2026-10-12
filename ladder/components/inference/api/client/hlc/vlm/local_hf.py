import threading
from typing import Any, Dict, List, Optional

import numpy as np
import PIL.Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from ladder.components.inference.api.client.hlc.vlm.base import VLM, VLMConfig


class LocalVLMConfig(VLMConfig):
    # In-process HuggingFace backend (no server, no API key). Any image-text-to-text checkpoint works.
    model: str = "Qwen/Qwen3-VL-2B-Instruct"
    dtype: str = "bfloat16"
    device: str = "cuda"


class LocalVLM(VLM[LocalVLMConfig]):
    """A locally-loaded transformers image-text-to-text model behind the same ``complete`` interface
    as the API backends, so the verifier/decomposer swap a hosted model for a local one by config alone.
    Generation is serialized under a lock (transformers ``generate`` is not thread-safe)."""

    def __init__(self, config: LocalVLMConfig) -> None:
        super().__init__(config)
        self._model = None
        self._proc = None
        self._lock = threading.Lock()

    def warmup(self) -> None:
        if self._model is not None:
            return
        dtype = getattr(torch, self.config.dtype)
        self._proc = AutoProcessor.from_pretrained(self.config.model)
        self._model = (
            AutoModelForImageTextToText.from_pretrained(self.config.model, torch_dtype=dtype)
            .to(self.config.device)
            .eval()
        )

    def _to_chat(self, messages: List[Dict[str, Any]]):
        images: List[PIL.Image.Image] = []
        conv = []
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                conv.append({"role": m["role"], "content": [{"type": "text", "text": content}]})
                continue
            items = []
            for it in content:
                if it.get("type") == "image":
                    images.append(PIL.Image.fromarray(np.asarray(it["image"], dtype=np.uint8)))
                    items.append({"type": "image"})
                else:
                    items.append({"type": "text", "text": it["text"]})
            conv.append({"role": m["role"], "content": items})
        return conv, images

    def _complete(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        reasoning_effort: Optional[str],
    ) -> str:
        del reasoning_effort  # no thinking-budget knob on a local transformers generate
        self.warmup()
        conv, images = self._to_chat(messages)
        with self._lock:
            text = self._proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            inputs = self._proc(text=[text], images=images or None, return_tensors="pt").to(
                self._model.device
            )
            do_sample = temperature is not None and temperature > 0
            with torch.no_grad():
                gen = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                )
            new = gen[0][inputs["input_ids"].shape[1] :]
            return self._proc.decode(new, skip_special_tokens=True)
