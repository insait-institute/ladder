import glob
import json
import os
from typing import Any, Dict, List, Union

# A frame is either a path on disk (rendered locally) or the PNG bytes (read from the Hub).
Frame = Union[str, bytes]


def load_scenes(frames_dir: str) -> List[Dict[str, Any]]:
    """Every suite's export, concatenated. Each exporter writes one ``<suite>/scenes.json`` next to
    its frames, so a suite is added by adding an exporter and the frame names stay valid wherever the
    directory is copied to."""
    scenes = []
    for manifest in sorted(glob.glob(os.path.join(frames_dir, "*", "scenes.json"))):
        with open(manifest) as handle:
            exported = json.load(handle)
        for scene in exported:
            scene["frames"] = {
                camera: os.path.join(os.path.dirname(manifest), name)
                for camera, name in scene["frames"].items()
            }
            # A partial benchmark that looks complete silently invalidates every number from it.
            for path in scene["frames"].values():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{scene['id']}: {path}")
        scenes += exported
    return scenes
