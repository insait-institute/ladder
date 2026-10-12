import itertools
from pathlib import Path
from typing import Iterable

import av
import numpy as np
import PIL.Image

from ladder.core.common.np_utils import verify_np_array

from .common import tqdm_progress_bar


def save_frames_to_video(
    frames: np.ndarray | Iterable[np.ndarray | PIL.Image.Image],
    video_path: str | Path,
    fps: int,
    pix_fmt: str = "yuv420p",  # use yuv420p for higher compression, yuv444p for better quality
    codec: str = "libx265",  # lossy codec, use ffv1 for lossless encoding
    keyframe_interval: int = 20,  # keyframe at least every 20 frames
    preset: str = "superfast",  # encoding speed preset: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
) -> None:
    """
    Convert frames to an MP4 video using PyAV.

    Args:
        frames:
            - np.ndarray: shape (num_frames, H, W, C) with dtype uint8
            - iterable of PIL.Image.Image or np.ndarray where each item is one frame
        video_path (str | Path): Path to save the output mp4 file.
        fps (int): Frames per second for the video.
        pix_fmt (str): Pixel format for encoding (default: 'yuv420p').
        codec (str): Video codec for encoding (default: 'libx265').
        keyframe_interval (int): Keyframe interval (default: 20).
        preset (str): Encoding speed preset (default: 'superfast'). Options from fastest to slowest:
            ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow.
            Faster presets = larger files but much faster encoding.
    """
    if isinstance(frames, np.ndarray):
        if frames.ndim != 4:
            raise ValueError(f"Expected frames with shape (N, H, W, C), got {frames.shape}")
        frame_iterator = iter(frames)
    else:
        frame_iterator = iter(frames)

    try:
        first_frame = np.asarray(next(frame_iterator))
    except StopIteration as exc:
        raise ValueError("frames must contain at least one frame") from exc

    if first_frame.ndim != 3:
        raise ValueError(f"Expected frame shape (H, W, C), got {first_frame.shape}")

    height, width, channels = first_frame.shape

    if channels != 3:
        raise ValueError(f"Expected 3 channels for RGB frames, got {channels}")

    # Create the parent directory if it doesn't exist
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    # Open output container
    container = av.open(str(video_path), mode="w")

    # Add video stream
    stream = container.add_stream(codec, rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = pix_fmt
    stream.options = {"g": str(keyframe_interval)}  # GOP size (keyframe interval)

    # Codec-specific options
    if codec == "libx265":
        stream.options["preset"] = preset
        stream.options["x265-params"] = f"log-level=none:keyint={keyframe_interval}:bframes=0"
    elif codec == "libx264":
        stream.options["preset"] = preset
    elif codec == "libx264rgb":
        # RGB-native x264 variant used for lossless encoding (e.g. bit-exact depth): crf=0 avoids any
        # quantization and rgb24 avoids the YUV colorspace conversion that would otherwise lose precision.
        stream.options["preset"] = preset
        stream.options["crf"] = "0"

    try:
        total = len(frames)
    except TypeError:
        total = None

    # Encode frames
    with tqdm_progress_bar(
        desc=f"Saving {video_path}",
        total=total,
        unit="frame",
    ) as progress_bar:
        for frame in itertools.chain([first_frame], frame_iterator):
            frame_np = np.asarray(frame)

            verify_np_array(frame_np, shape=(height, width, channels), dtype=np.uint8)

            frame_av = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for packet in stream.encode(frame_av):
                container.mux(packet)
            progress_bar.update(1)

        # Flush encoder
        for packet in stream.encode():
            container.mux(packet)

    container.close()
