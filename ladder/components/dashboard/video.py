import hashlib
import os
import pathlib
import subprocess
import threading
from typing import Optional

from ladder.core.common.logger import get_logger

LOGGER = get_logger(__name__)
PLAYABLE = ("h264", "vp8", "vp9", "av1")
_TRANSCODE_LOCK = threading.Lock()


def cache_dir() -> pathlib.Path:
    root = os.environ.get("LADDER_DASHBOARD_CACHE")
    if root:
        return pathlib.Path(root).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or str(pathlib.Path.home() / ".cache")
    return pathlib.Path(base) / "ladder-dashboard" / "video"


def _run(command: list) -> Optional[subprocess.CompletedProcess]:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        LOGGER.warn(f"{command[0]} failed: {exc}")
        return None


def _transcode(source: pathlib.Path, target: pathlib.Path) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(f".{os.getpid()}.tmp.mp4")
    done = _run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-threads",
            "2",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(staging),
        ]
    )
    if done is None or done.returncode != 0 or not staging.exists():
        staging.unlink(missing_ok=True)
        return False
    os.replace(staging, target)
    return True


def probe_codec(path: pathlib.Path) -> Optional[str]:
    probe = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ]
    )
    return probe.stdout.strip() if probe is not None and probe.returncode == 0 else None


def playable(path: pathlib.Path) -> pathlib.Path:
    """The same clip, in a codec a browser can play: the original when it is, else one transcoded
    clip per source under the cache dir. One lock covers probe and transcode, so concurrent
    requests for an uncached clip run a single ffmpeg."""
    stat = path.stat()
    key = hashlib.sha1(f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()
    target = cache_dir() / f"{key}.mp4"
    if target.exists():
        return target
    with _TRANSCODE_LOCK:
        if target.exists():
            return target
        if probe_codec(path) in PLAYABLE:
            return path
        return target if _transcode(path, target) else path
