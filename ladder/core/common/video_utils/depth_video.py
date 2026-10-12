import itertools
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence, Tuple, Type, Union

import numpy as np

from ladder.core.common.np_utils import verify_np_array

from .load_video import iter_frames_from_video, read_frames_from_video
from .save_video import save_frames_to_video

# Sentinel used for invalid depth (NaN / +-inf) so it survives the float<->uint16 roundtrip: 64.0m is
# well inside the 65.535m range representable in 16 bits of millimeters.
INVALID_DEPTH_VALUE = 64.0

_FLOAT_DTYPES = (np.float32, np.float64)
# A depth frame is single-channel (H, W); the trailing two axes are always spatial.
_DEPTH_FRAME_SHAPE = (-1, -1)
# A packed depth frame is an RGB image (H, W, 3).
_RGB_FRAME_SHAPE = (-1, -1, 3)


def _apply_framewise(
    transform: Callable[[np.ndarray], np.ndarray],
    frames: np.ndarray | Iterable[np.ndarray],
    *,
    dtype: Optional[Union[Type[np.generic], Sequence[Type[np.generic]]]] = None,
    frame_shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray | Iterator[np.ndarray]:
    """
    Apply `transform` while preserving the input container type.

    A whole `np.ndarray` is transformed in a single call (a single (H, W) frame or a stacked (N, H, W)
    clip); an iterable of frames is mapped lazily, keeping only one frame in memory at a time. When
    `frame_shape`/`dtype` are given, every array is validated first (per frame for iterables), accepting
    both a single frame and a stack of frames.
    """

    def _verify(array: np.ndarray) -> None:
        if frame_shape is None:
            return
        if array.ndim == len(frame_shape):
            expected_shape: Tuple[int, ...] = frame_shape
        elif array.ndim == len(frame_shape) + 1:
            expected_shape = (-1, *frame_shape)
        else:
            raise ValueError(
                f"Expected a frame with shape {frame_shape} or a stack (N, ...) thereof, "
                f"got shape {array.shape}"
            )
        verify_np_array(array, shape=expected_shape, dtype=dtype)

    if isinstance(frames, np.ndarray):
        _verify(frames)
        return transform(frames)

    if isinstance(frames, Iterable):

        def _generate() -> Iterator[np.ndarray]:
            for frame in frames:
                _verify(frame)
                yield transform(frame)

        return _generate()

    raise TypeError(f"Expected np.ndarray or an iterable of np.ndarray, got {type(frames)}")


def clip_and_replace_invalid_depth(
    depth: np.ndarray | Iterable[np.ndarray], invalid_value: float = INVALID_DEPTH_VALUE
) -> np.ndarray | Iterator[np.ndarray]:
    """
    Replace invalid float depth (NaN, +-inf) with `invalid_value` and clip values to [0, invalid_value].

    Input must be float depth in meters; output preserves the input container type.
    """

    def _transform(frame: np.ndarray) -> np.ndarray:
        frame = np.nan_to_num(frame, nan=invalid_value, posinf=invalid_value, neginf=invalid_value)
        return np.clip(frame, 0.0, invalid_value)

    return _apply_framewise(_transform, depth, dtype=_FLOAT_DTYPES, frame_shape=_DEPTH_FRAME_SHAPE)


def replace_invalid_depth_with_nan(
    depth: np.ndarray | Iterable[np.ndarray], invalid_value: float = INVALID_DEPTH_VALUE
) -> np.ndarray | Iterator[np.ndarray]:
    """
    Replace the `invalid_value` sentinel with NaN in float depth (inverse of clip_and_replace_invalid_depth).

    Input must be float depth in meters; output preserves the input container type.
    """

    def _transform(frame: np.ndarray) -> np.ndarray:
        return np.where(frame == invalid_value, np.nan, frame).astype(frame.dtype)

    return _apply_framewise(_transform, depth, dtype=_FLOAT_DTYPES, frame_shape=_DEPTH_FRAME_SHAPE)


def depth_meters_to_uint16(
    depth: np.ndarray | Iterable[np.ndarray],
) -> np.ndarray | Iterator[np.ndarray]:
    """
    Convert float depth in meters to uint16 millimeters (AB.CDE m -> ABCDE mm).

    Values must not exceed 65.535m to fit in 16 bits. Input must be float depth; output preserves the
    input container type.
    """

    def _transform(frame: np.ndarray) -> np.ndarray:
        return ((frame * 1000) % 100000).astype(np.uint16)

    return _apply_framewise(_transform, depth, dtype=_FLOAT_DTYPES, frame_shape=_DEPTH_FRAME_SHAPE)


def depth_uint16_to_meters(
    depth: np.ndarray | Iterable[np.ndarray],
) -> np.ndarray | Iterator[np.ndarray]:
    """
    Convert uint16 millimeter depth back to float32 meters (ABCDE mm -> AB.CDE m).

    Input must be uint16 depth; output preserves the input container type.
    """

    def _transform(frame: np.ndarray) -> np.ndarray:
        return frame.astype(np.float32) / 1000.0

    return _apply_framewise(_transform, depth, dtype=np.uint16, frame_shape=_DEPTH_FRAME_SHAPE)


def metric_depth_to_uint16(
    depth: np.ndarray | Iterable[np.ndarray],
) -> np.ndarray | Iterator[np.ndarray]:
    """
    Coerce metric depth to the uint16-millimeter storage form, accepting float meters or uint16 mm.

    This is the single entry point for "I have metric depth, give me the packable uint16":
    - float (float32/float64) frames are treated as metric depth in *meters* (invalid values ->
      sentinel, clipped to [0, INVALID_DEPTH_VALUE], scaled to millimeters);
    - uint16 frames are assumed to already be metric *millimeters* and passed through unchanged.

    Relative and inverse depth are NOT supported - the millimeter fixed-point scale is metric-only, and
    the two cannot be told apart from the values (their ranges overlap valid metric depth). Output
    preserves the input container type.
    """

    def _transform(frame: np.ndarray) -> np.ndarray:
        if frame.dtype == np.uint16:
            return frame
        if frame.dtype in _FLOAT_DTYPES:
            return depth_meters_to_uint16(clip_and_replace_invalid_depth(frame))
        raise TypeError(f"Expected uint16 (millimeters) or float (meters) depth, got {frame.dtype}")

    return _apply_framewise(_transform, depth)


def _subsample_frames(
    frames: np.ndarray | Iterable[np.ndarray], step: int
) -> np.ndarray | Iterable[np.ndarray]:
    if step == 1:
        return frames
    if isinstance(frames, np.ndarray):
        return frames[::step]
    return itertools.islice(frames, 0, None, step)


def _downsample_depth_frames(
    frames: np.ndarray | Iterable[np.ndarray], factor: int
) -> np.ndarray | Iterator[np.ndarray]:
    """Nearest-neighbour spatial downsampling of the trailing (H, W) axes by an integer factor."""
    return _apply_framewise(lambda frame: frame[..., ::factor, ::factor], frames)


def _upsample_depth_frames(
    frames: np.ndarray | Iterable[np.ndarray], factor: int
) -> np.ndarray | Iterator[np.ndarray]:
    """Nearest-neighbour spatial upsampling of the trailing (H, W) axes by an integer factor."""

    def _transform(frame: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(frame, factor, axis=-2), factor, axis=-1)

    return _apply_framewise(_transform, frames)


def _depth_uint16_to_rgb(
    depth: np.ndarray | Iterable[np.ndarray],
) -> np.ndarray | Iterator[np.ndarray]:
    """Pack uint16 depth into an RGB uint8 frame: R = high byte, G = low byte, B = 0 (padding)."""

    def _transform(frame: np.ndarray) -> np.ndarray:
        rgb = np.zeros(frame.shape + (3,), dtype=np.uint8)
        rgb[..., 0] = (frame >> 8).astype(np.uint8)
        rgb[..., 1] = (frame & 0xFF).astype(np.uint8)
        return rgb

    return _apply_framewise(_transform, depth, dtype=np.uint16, frame_shape=_DEPTH_FRAME_SHAPE)


def _rgb_to_depth_uint16(rgb: np.ndarray) -> np.ndarray:
    """Reconstruct uint16 depth from a packed RGB frame (inverse of _depth_uint16_to_rgb)."""
    high_byte = rgb[..., 0].astype(np.uint16)
    low_byte = rgb[..., 1].astype(np.uint16)
    return (high_byte << 8) | low_byte


def encode_depth_uint16_to_video(
    depth: np.ndarray | Iterable[np.ndarray], video_path: str | Path, fps: int
) -> None:
    """
    Losslessly encode uint16 depth frames to an mp4 by packing them into RGB and using libx264rgb.

    Depth is stored losslessly by packing the 16-bit millimeter value into the R (high byte) and G (low
    byte) channels of an RGB frame; the blue channel is padding. `depth` must be uint16 depth (single
    (H, W) frame is not accepted here - pass a stacked (N, H, W) array or an iterable of frames). Frames
    are streamed straight into the encoder for iterable inputs.
    """
    rgb_frames = _depth_uint16_to_rgb(depth)
    # crf=0 + libx264rgb keeps the packed bytes bit-exact; veryslow trades encode time for smaller files.
    save_frames_to_video(
        rgb_frames, video_path, fps=fps, codec="libx264rgb", pix_fmt="rgb24", preset="veryslow"
    )


def decode_video_to_depth_uint16(video_path: str | Path) -> np.ndarray:
    """Decode a depth mp4 written by encode_depth_uint16_to_video back to a uint16 (N, H, W) array."""
    rgb = read_frames_from_video(str(video_path))
    verify_np_array(rgb, shape=(-1, *_RGB_FRAME_SHAPE), dtype=np.uint8)
    return _rgb_to_depth_uint16(rgb)


def iter_depth_uint16_from_video(video_path: str | Path) -> Iterator[np.ndarray]:
    """Yield uint16 depth frames one at a time from a depth mp4, without materializing the whole clip."""
    for rgb in iter_frames_from_video(str(video_path)):
        verify_np_array(rgb, shape=_RGB_FRAME_SHAPE, dtype=np.uint8)
        yield _rgb_to_depth_uint16(rgb)


def save_depth_frames_to_video(
    depth: np.ndarray | Iterable[np.ndarray],
    video_path: str | Path,
    fps: int,
    downsample_factor: int,
    fps_reduction_factor: int = 1,
) -> None:
    """
    Save float depth (in meters) to a lossless depth mp4.

    Frames are temporally subsampled by `fps_reduction_factor`, spatially downsampled by
    `downsample_factor` (pass 1 to keep full resolution), cleaned (invalid depth -> sentinel, clipped),
    converted to uint16 millimeters and encoded. Iterable inputs are streamed frame by frame;
    `load_depth_frames_from_video` is the inverse.
    """
    frames = _subsample_frames(depth, fps_reduction_factor)
    frames = _downsample_depth_frames(frames, downsample_factor)
    frames = clip_and_replace_invalid_depth(frames)
    frames = depth_meters_to_uint16(frames)
    encode_depth_uint16_to_video(frames, video_path, fps=fps)


def load_depth_frames_from_video(video_path: str | Path, upsample_factor: int = 2) -> np.ndarray:
    """
    Load a depth mp4 as float32 depth in meters, with invalid depth restored to NaN.

    Frames are spatially upsampled by `upsample_factor` (inverse of `save_depth_frames_to_video`).
    Returns a stacked (N, H, W) array; use `iter_depth_frames_from_video` to stream instead.
    """
    depth = decode_video_to_depth_uint16(video_path)
    depth = depth_uint16_to_meters(depth)
    depth = replace_invalid_depth_with_nan(depth)
    return _upsample_depth_frames(depth, upsample_factor)


def iter_depth_frames_from_video(video_path: str | Path, upsample_factor: int = 2) -> Iterator[np.ndarray]:
    """Streaming counterpart of `load_depth_frames_from_video`, yielding float32 (H, W) frames in meters."""
    depth = iter_depth_uint16_from_video(video_path)
    depth = depth_uint16_to_meters(depth)
    depth = replace_invalid_depth_with_nan(depth)
    return _upsample_depth_frames(depth, upsample_factor)
