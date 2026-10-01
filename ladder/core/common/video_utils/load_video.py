from __future__ import annotations

import collections
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, NamedTuple, Optional

import av
import numpy as np
import PIL.Image


def read_frames_from_video(video_path: str) -> np.ndarray:
    """Return all frames of an RGB video as a uint8 array with shape (num_frames, H, W, 3)."""
    container = av.open(video_path)
    frames = []

    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))

    container.close()
    return np.stack(frames).astype(np.uint8)


def iter_frames_from_video(video_path: str) -> Iterator[np.ndarray]:
    """
    Yield RGB frames (H, W, 3 uint8) one at a time from a video.

    Unlike `read_frames_from_video`, this keeps only a single decoded frame in memory at a time, so a
    caller can stream frames straight into an encoder without materializing the whole clip.
    """
    with av.open(video_path) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def decode_frame_from_video(video_path: str, frame_index: int) -> PIL.Image.Image:
    """
    Decode a single frame at the given frame index from an RGB video using PyAV.

    This function provides frame-index-based access which can be faster for
    sequential access patterns or when precise frame numbers are known.

    Args:
        video_path (str): Path to the MP4 video file.
        frame_index (int): Zero-based frame index to decode.
        width (int): Width of the output frame.
        height (int): Height of the output frame.

    Returns:
        PIL.Image.Image: The frame as a PIL image.
    """
    with av.open(video_path) as container:  # av.container.input.InputContainer
        return decode_frame_from_container(container, frame_index=frame_index)


def decode_frame_from_container(
    container: av.container.input.InputContainer, frame_index: int
) -> PIL.Image.Image:
    """
    Decode the frame at `frame_index` from an already-open container.

    Seeking leaves no state behind that affects later reads, so one container can serve any number of
    frames in any order. `VideoFrameReader` relies on that to keep containers open across reads.
    """
    stream: av.video.stream.VideoStream = container.streams.video[0]

    # Make sure the frame index is within the bounds of the video
    assert frame_index < stream.frames, f"{frame_index} >= {stream.frames}"

    # Calculate timestamp from frame index for seeking
    fps = float(stream.average_rate)

    # Timestamp units explanation:
    # stream.time_base: The unit of time (in fractional seconds) in which timestamps
    #   in the stream are expressed, e.g 1/10240
    # frame.pts: Timestamp of the frame in `time_base` units
    # frame.time: Timestamp of the frame in seconds
    #
    # The relationship is `frame.time = frame.pts * stream.time_base`
    # Example: FPS=5, stream.time_base=1/10240
    #   - index=0 -> frame.pts=0,    frame.time = 0 * (1/10240) = 0.0
    #   - index=1 -> frame.pts=2048, frame.time = 2048 * (1/10240) = 0.2
    #   - index=2 -> frame.pts=4096, frame.time = 4096 * (1/10240) = 0.4

    # Correction offset in number of frames for seeking. Read the comment below for more details.
    seek_offset: int = 0
    max_tries = 10

    for i in range(max_tries):
        # Compute the target timestamp in seconds and then convert to `time_base` units
        seek_timestamp_sec: float = max(frame_index - seek_offset, 0) / fps
        seek_timestamp_pts = round(seek_timestamp_sec / stream.time_base)

        # Note that seek is O(1) with respect to the target keyframe - it jumps directly to it
        # container.seek finds the closest *previous keyframe* for a given timestamp.
        # https://pyav.basswood-io.com/docs/stable/api/stream.html#av.stream.Stream.time_base
        #
        #   `backward=True` -> look backwards for the nearest (key)frame if none exists at the offset
        #   `any_frame=False` -> seek to a keyframe only
        #
        # WHY seek_offset EXISTS — the B-frame / DTS-PTS problem:
        #
        # H.265/H.264 with B-frames (bidirectional predicted frames) encodes packets in a
        # different order than they are displayed. Each packet therefore carries two timestamps:
        #   - DTS (Decode Timestamp): order in which packets must be fed to the decoder
        #   - PTS (Presentation Timestamp): order in which decoded frames are displayed
        #
        # With B-frames the encoder buffers several frames before outputting them, so the first
        # few packets have *negative* DTS while PTS starts at 0. The gap between DTS and PTS at
        # keyframe boundaries is not constant — it depends on how many B-frames the encoder
        # placed before the next keyframe, which varies with lookahead-based encoders (x265
        # with lookahead=20, bframes=4 can produce a gap that ranges from 2 to 6 frames).
        #
        # container.seek is called with a PTS value, but FFmpeg seeks internally by DTS.
        # Because the DTS-PTS gap varies across keyframes, FFmpeg's translation from the
        # requested PTS to an internal DTS position is sometimes off by one keyframe, causing
        # seek to land on the *next* keyframe instead of the previous one.
        #
        # HOW THIS SHOULD BE PREVENTED: we encode all videos with `bframes=0` so that DTS == PTS for
        # every packet and seeking is always exact. The retry loop below is kept as a safety
        # net for incompatible legacy videos or videos produced by third-party tools that may still
        # use B-frames.
        container.seek(seek_timestamp_pts, backward=True, any_frame=False, stream=stream)

        # Starting from the keyframe found by seek, decode frames until we reach the desired frame index
        for frame in container.decode(stream):  # av.video.frame.VideoFrame
            decoded_index = round(frame.pts * stream.time_base * fps)

            if decoded_index > frame_index:
                seek_offset = (i + 1) ** 2
                break

            if decoded_index == frame_index:
                image: PIL.Image.Image = frame.to_image()
                return image

    raise ValueError(
        f"Could not find frame at index {frame_index} in video of length "
        f"{stream.frames} frames at {container.name}"
    )


class _CacheEntry(NamedTuple):
    container: av.container.input.InputContainer
    # A container holds decoder and demuxer state, so only one thread may drive it at a time
    lock: threading.Lock


class VideoFrameReader:
    """
    Reads frames by index, keeping the video containers it opens in an LRU cache.

    WHY THIS EXISTS: opening a container is not cheap. FFmpeg parses the whole seek index up front,
    one entry per frame at roughly 40 bytes each, so a container for a 300k-frame video costs ~11 MB
    of RAM and 20-100 ms to open. Datasets that pack many episodes into one file therefore spend
    almost all of their frame-read time re-parsing the same few indices: measured on DROID
    (berobot-v3, ~300k frames per file), reopening per frame put ~290 ms of the ~300 ms spent reading
    a frame into `av.open`. Keeping containers open trades that RAM for the parse.

    Cache size is a count of open containers, so budget it against ~40 bytes per frame per file. Size
    it at or above the number of files that will be read, or leave it unlimited: frames are sampled
    uniformly at random, so LRU evicts entries that are about to be needed again and a cache smaller
    than the working set never converges - it degrades sharply rather than gracefully.
    """

    def __init__(self, max_cached_videos: Optional[int] = None):
        """
        Args:
            max_cached_videos: Number of containers to keep open, or None for no limit. Only pass None
                when the total number of video files is known to be bounded.
        """
        if max_cached_videos is not None and max_cached_videos <= 0:
            raise ValueError(f"max_cached_videos must be positive or None, got {max_cached_videos}")

        self.max_cached_videos = max_cached_videos
        self._cache: collections.OrderedDict[str, _CacheEntry] = collections.OrderedDict()
        self._cache_lock = threading.Lock()
        # Containers must never be shared across a fork: parent and child would share one file
        # offset and corrupt each other's reads. Dataloader workers are forked, so drop anything
        # inherited from the parent process instead of using it.
        self._pid = os.getpid()

    def decode_frame(self, video_path: str, frame_index: int) -> PIL.Image.Image:
        entry = self._entry(str(video_path))

        # Held across the decode so two reader threads never drive the same container concurrently
        with entry.lock:
            return decode_frame_from_container(entry.container, frame_index=frame_index)

    def _entry(self, video_path: str) -> _CacheEntry:
        with self._cache_lock:
            if self._pid != os.getpid():
                # Inherited from a fork - abandon the containers, letting this process close its copies
                self._cache.clear()
                self._pid = os.getpid()

            entry = self._cache.get(video_path)
            if entry is not None:
                self._cache.move_to_end(video_path)
                return entry

            entry = _CacheEntry(container=av.open(video_path), lock=threading.Lock())
            self._cache[video_path] = entry
            self._evict()
            return entry

    def _evict(self) -> None:
        """Drop least-recently-used containers that no thread is currently decoding from."""
        if self.max_cached_videos is None:
            return

        for path in list(self._cache):  # least-recently-used first
            if len(self._cache) <= self.max_cached_videos:
                return

            entry = self._cache[path]
            # Skip entries in use; the cache may sit over its cap by at most the number of readers
            if entry.lock.acquire(blocking=False):
                try:
                    del self._cache[path]
                    entry.container.close()
                finally:
                    entry.lock.release()

    def close(self) -> None:
        with self._cache_lock:
            for entry in self._cache.values():
                entry.container.close()
            self._cache.clear()

    @property
    def num_cached_videos(self) -> int:
        with self._cache_lock:
            return len(self._cache)


def probe_video_stream(source_video_path: str | Path) -> Dict[str, Any]:
    """
    Return codec and frame geometry metadata for an encoded video file.

    Args:
        source_video_path: Path to the input video file.

    Returns:
        A dictionary containing:
            - codec: Codec name (e.g., 'h264', 'hevc')
            - pix_fmt: Pixel format (e.g., 'yuv420p', 'yuv444p')
            - fps: Average frames per second if available
            - width: Frame width in pixels
            - height: Frame height in pixels
            - time_base: Stream time base as (numerator, denominator)
            - extradata: Codec extradata bytes

    """
    with av.open(str(source_video_path)) as container:
        video_stream = next(stream for stream in container.streams if stream.type == 'video')
        codec_context = video_stream.codec_context
        codec_name = codec_context.name
        pix_fmt = codec_context.pix_fmt
        fps = None if video_stream.average_rate is None else float(video_stream.average_rate)
        width = int(video_stream.width)
        height = int(video_stream.height)
        time_base = None
        if video_stream.time_base is not None:
            time_base = (int(video_stream.time_base.numerator), int(video_stream.time_base.denominator))
        extradata = bytes(codec_context.extradata or b"")

    return {
        "codec": codec_name,
        "pix_fmt": pix_fmt,
        "fps": fps,
        "width": width,
        "height": height,
        "time_base": time_base,
        "extradata": extradata,
    }


# Map both encoder names and probed stream names onto a single canonical codec identity.
# A video can be tagged with the name of the encoder that produced it ("libx264", "libx265")
# or with the codec name reported by the decoder/ffprobe ("h264", "hevc"); both refer to the
# same stream codec. Downstream code that keys on the codec (e.g. file-size estimation) must
# normalize first so the different spellings collapse to the same key.
_CANONICAL_VIDEO_CODECS = {
    "libx264": "h264",
    "h264": "h264",
    "libx265": "h265",
    "hevc": "h265",
    "h265": "h265",
    "ffv1": "ffv1",
}


def normalize_codec(codec: str) -> str:
    """
    Normalize a video codec name to its canonical stream codec identity (e.g. "libx265" -> "h265").

    Unknown codecs are returned unchanged so callers can surface a clear, codec-specific error
    instead of silently masking an unsupported format.
    """
    return _CANONICAL_VIDEO_CODECS.get(codec, codec)
