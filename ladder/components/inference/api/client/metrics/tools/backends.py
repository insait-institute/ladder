import os
from typing import Any, Dict

from ladder.core.common.platform import REPO_ROOT
from ladder.core.databib.config import load_config

LOCAL_VLM = "ladder.components.inference.api.client.hlc.vlm.local_hf.LocalVLM"
# The eval's own configs, endpoint and cost-tracking headers included, so a scored planner is driven
# through the same client an eval calls. `gateway` is the Gemini path; the NVIDIA entries add a second
# provider with its own key, and Anthropic is separate because those models refuse `temperature`.
BACKEND_CONFIG = {
    "gateway": "ladder/components/inference/api/client/hlc/configs/vlm.yaml",
    "nvidia": "ladder/components/inference/api/client/hlc/configs/vlm_nvidia.yaml",
    "nvidia_anthropic": "ladder/components/inference/api/client/hlc/configs/vlm_nvidia_anthropic.yaml",
}
BACKENDS = ("local",) + tuple(BACKEND_CONFIG)


def vlm_config(model: str, backend: str) -> Dict[str, Any]:
    """The VLM backend a scorer drives. `load_config`, not `yaml.safe_load`: the Anthropic config is
    an `import:` of the shared one. The timeout is raised because scoring is not latency-bound and a
    timed-out call is indistinguishable in the result from a refusal."""
    if backend == "local":
        return {"autoclass": LOCAL_VLM, "model": model, "dtype": "bfloat16", "device": "cuda"}
    return load_config(os.path.join(REPO_ROOT, BACKEND_CONFIG[backend])) | {
        "model": model,
        "timeout": 600.0,
    }
