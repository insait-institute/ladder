import os
from typing import Any, Dict

from ladder.core.common.platform import REPO_ROOT
from ladder.core.databib.config import load_config

LOCAL_VLM = "ladder.components.inference.api.client.hlc.vlm.local_hf.LocalVLM"
# The eval's own endpoint config, so a scored model is driven through the same client an eval calls.
BACKEND_CONFIG = {
    "gateway": "ladder/components/inference/api/client/hlc/configs/vlm.yaml",
}
BACKENDS = ("local",) + tuple(BACKEND_CONFIG)


def vlm_config(model: str, backend: str) -> Dict[str, Any]:
    """The VLM backend a scorer drives. `load_config`, not `yaml.safe_load`: it expands the
    environment variables the config names. The timeout is raised because scoring is not
    latency-bound and a timed-out call is indistinguishable in the result from a refusal."""
    if backend == "local":
        return {"autoclass": LOCAL_VLM, "model": model, "dtype": "bfloat16", "device": "cuda"}
    return load_config(os.path.join(REPO_ROOT, BACKEND_CONFIG[backend])) | {
        "model": model,
        "timeout": 600.0,
    }
