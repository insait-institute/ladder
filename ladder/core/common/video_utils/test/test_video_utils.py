import os
import tempfile
import unittest
from pathlib import Path
from typing import List

import av
import numpy as np
import PIL.Image

from ladder.core.common.path_utils import create_tempdir
from ladder.core.common.video_utils.depth_video import (
    decode_video_to_depth_uint16,
    encode_depth_uint16_to_video,
)
from ladder.core.common.video_utils.encode_video import combine_video_paths_into_chunk, transcode_video
from ladder.core.common.video_utils.load_video import decode_frame_from_video, normalize_codec
from ladder.core.common.video_utils.save_video import save_frames_to_video


class MockDataset:
    def __init__(self, frames: List[PIL.Image.Image], fps: int):
        self._frames = frames
        self.fps = fps
        self.episode_data_index = {
            'from': np.array([0], dtype=np.int64),
            'to': np.array([len(frames)], dtype=np.int64),
        }

    def get_value_at(self, column_name: str, index: int) -> PIL.Image.Image:
        assert column_name == 'observation.images.main_image'
        return self._frames[index]


def _make_rgb_frames(num_frames: int, height: int = 32, width: int = 48) -> np.ndarray:
    rows = np.arange(height, dtype=np.uint8)[:, None]
    cols = np.arange(width, dtype=np.uint8)[None, :]
    frames = []
    for i in range(num_frames):
        frame = np.empty((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = rows + (7 * i)
        frame[:, :, 1] = cols + (11 * i)
        frame[:, :, 2] = rows + cols + (13 * i)
        frames.append(frame)
    return np.stack(frames, axis=0)


def _read_rgb_frames(video_path: str | Path) -> np.ndarray:
    with av.open(str(video_path)) as container:
        stream: av.video.stream.VideoStream = container.streams.video[0]
        num_frames = stream.frames
    return np.stack(
        [np.asarray(decode_frame_from_video(str(video_path), i)) for i in range(num_frames)],
        axis=0,
    )


class TestDecodeFrameFromVideo(unittest.TestCase):
    def test_decode_video_frame(self):
        # NOTE: If you change the video file, make sure it has multiple keyframes for robust testing
        video_path = os.path.join(os.path.dirname(__file__), 'videos/video_with_keyframes.mp4')

        with av.open(video_path) as container:
            stream: av.video.stream.VideoStream = container.streams.video[0]
            images = np.stack(  # [B, H, W, C]
                [np.asarray(frame.to_image()) for frame in container.decode(stream)]
            )

        decoded_images = np.stack(  # [B, H, W, C]
            [np.asarray(decode_frame_from_video(video_path, frame_index=i)) for i in range(images.shape[0])]
        )

        np.testing.assert_array_equal(images, decoded_images)


class TestSaveFramesToVideo(unittest.TestCase):
    def setUp(self):
        self.output_path = create_tempdir(self)

    def test_save_frames_to_video_from_iterator(self):
        frames = _make_rgb_frames(num_frames=6)
        output_path = Path(self.output_path) / 'iterator_input.mp4'
        frame_iterator = (PIL.Image.fromarray(frame, mode='RGB') for frame in frames)

        save_frames_to_video(
            frames=frame_iterator,
            video_path=output_path,
            fps=5,
            pix_fmt='yuv444p',
            codec='libx264',
            keyframe_interval=100,
        )

        decoded_frames = _read_rgb_frames(output_path)
        self.assertEqual(decoded_frames.shape, frames.shape)

    def test_save_frames_to_video_from_numpy(self):
        output_path = Path(self.output_path) / 'output.mp4'
        frames = _make_rgb_frames(num_frames=6)

        save_frames_to_video(
            frames=frames,
            video_path=output_path,
            fps=5,
            pix_fmt='yuv444p',
            codec='libx264',
            keyframe_interval=100,
        )

        decoded_frames = _read_rgb_frames(output_path)
        self.assertEqual(decoded_frames.shape, frames.shape)


class TestCombineVideoPathsIntoChunk(unittest.TestCase):
    def test_combine_video_paths_into_chunk(self):
        video_path = os.path.join(os.path.dirname(__file__), 'videos/video_with_keyframes.mp4')

        with tempfile.TemporaryDirectory() as temp_dir:
            combined = Path(temp_dir) / 'combined.mp4'

            combine_video_paths_into_chunk([video_path, video_path], combined)

            input_frames = _read_rgb_frames(video_path)
            combined_frames = _read_rgb_frames(combined)

            expected_frames = np.concatenate([input_frames, input_frames], axis=0)
            np.testing.assert_array_equal(combined_frames, expected_frames)

    def test_combine_lossless_depth_videos_is_bit_exact(self):
        # Depth is stored as a lossless libx264rgb stream that probes as h264/gbrp. Remuxing must copy
        # packets without opening an encoder: the libx264 encoder rejects the RGB pixel format, so a
        # naive "open the encoder to write the header" path fails outright (and would silently re-encode
        # for the yuv codecs). This guards that depth concatenation stays bit-exact.
        rng = np.random.default_rng(0)
        depth_a = rng.integers(0, 65536, size=(6, 16, 24), dtype=np.uint16)
        depth_b = rng.integers(0, 65536, size=(5, 16, 24), dtype=np.uint16)

        with tempfile.TemporaryDirectory() as temp_dir:
            video_a = Path(temp_dir) / 'a.mp4'
            video_b = Path(temp_dir) / 'b.mp4'
            combined = Path(temp_dir) / 'combined.mp4'
            encode_depth_uint16_to_video(depth_a, video_a, fps=5)
            encode_depth_uint16_to_video(depth_b, video_b, fps=5)

            combine_video_paths_into_chunk([video_a, video_b], combined)

            np.testing.assert_array_equal(
                decode_video_to_depth_uint16(combined), np.concatenate([depth_a, depth_b], axis=0)
            )

    def test_combine_video_paths_into_chunk_with_frame_ranges(self):
        # video_with_keyframes.mp4 is encoded with b-pyramid, where frames near the end
        # of each GOP use the next keyframe as a forward reference. Packet-level remuxing
        # at a GOP boundary without re-encoding cannot include those frames (they need
        # the next keyframe to be decodable). To test exact frame equality we transcode
        # to a bframes=0 version first, so every GOP boundary is a clean cut.
        source_video = os.path.join(os.path.dirname(__file__), 'videos/video_with_keyframes.mp4')

        with tempfile.TemporaryDirectory() as temp_dir:
            nobframes_video = Path(temp_dir) / 'source_nobframes.mp4'
            transcode_video(
                source_video, nobframes_video, fps=5, pix_fmt='yuv444p', codec='libx265', keyframe_interval=20
            )

            combined = Path(temp_dir) / 'combined.mp4'
            combine_video_paths_into_chunk(
                video_paths=[nobframes_video, nobframes_video],
                output_video_path=combined,
                frame_ranges=[(0, 20), (60, 80)],
            )

            source_frames = _read_rgb_frames(nobframes_video)
            combined_frames = _read_rgb_frames(combined)

            expected_frames = np.concatenate([source_frames[:20], source_frames[60:80]], axis=0)
            np.testing.assert_array_equal(combined_frames, expected_frames)

    def test_combine_short_bframe_segment_keeps_dts_monotonic(self):
        # Regression: a single-frame clip has DTS == PTS (no room for B-frame reordering), while
        # multi-frame B-frame clips start with a negative DTS. When such a short clip sits between
        # B-frame clips, the naive per-segment timestamp offset used to make the following clip's
        # first DTS dip to or below the previous clip's last DTS, and the MP4 muxer rejected the
        # non-monotonic DTS with EINVAL. The fix nudges DTS to stay strictly increasing without ever
        # touching PTS, so the remux stays lossless and pixel-identical to the inputs.
        videos_dir = os.path.join(os.path.dirname(__file__), 'videos')
        multi = os.path.join(videos_dir, 'bframes_multi.mp4')
        single = os.path.join(videos_dir, 'bframes_single.mp4')

        with tempfile.TemporaryDirectory() as temp_dir:
            combined = Path(temp_dir) / 'combined.mp4'
            combine_video_paths_into_chunk([multi, single, multi], combined)

            with av.open(str(combined)) as container:
                stream = container.streams.video[0]
                dts_values = [int(packet.dts) for packet in container.demux(stream) if packet.dts is not None]
            self.assertTrue(
                all(earlier < later for earlier, later in zip(dts_values, dts_values[1:], strict=False)),
                f'DTS is not strictly increasing: {dts_values}',
            )

            combined_frames = _read_rgb_frames(combined)
            expected_frames = np.concatenate(
                [_read_rgb_frames(multi), _read_rgb_frames(single), _read_rgb_frames(multi)], axis=0
            )
            np.testing.assert_array_equal(combined_frames, expected_frames)

    def test_combine_without_bframes_produces_clean_dts_equals_pts(self):
        # Remuxing videos without B-frames must land at a clean stream where DTS == PTS for every
        # packet, i.e. none of the B-frame timestamp handling leaks in. This is the property that
        # keeps frame-index seeking exact (see load_video.py), so it is worth pinning down.
        video = os.path.join(os.path.dirname(__file__), 'videos/nobframes.mp4')

        with tempfile.TemporaryDirectory() as temp_dir:
            combined = Path(temp_dir) / 'combined.mp4'
            combine_video_paths_into_chunk([video, video], combined)

            with av.open(str(combined)) as container:
                stream = container.streams.video[0]
                timestamps = [
                    (packet.dts, packet.pts)
                    for packet in container.demux(stream)
                    if packet.dts is not None or packet.pts is not None
                ]
            self.assertTrue(
                all(dts == pts for dts, pts in timestamps),
                f'expected DTS == PTS for a remux without B-frames, got {timestamps}',
            )

            combined_frames = _read_rgb_frames(combined)
            expected_frames = np.concatenate([_read_rgb_frames(video), _read_rgb_frames(video)], axis=0)
            np.testing.assert_array_equal(combined_frames, expected_frames)

    def test_rejects_encoding_mismatch(self):
        frames = _make_rgb_frames(num_frames=4)

        with tempfile.TemporaryDirectory() as temp_dir:
            source_a = Path(temp_dir) / 'source_a.mp4'
            source_b = Path(temp_dir) / 'source_b.mp4'
            combined = Path(temp_dir) / 'combined.mp4'

            save_frames_to_video(
                frames=frames,
                video_path=source_a,
                fps=8,
                pix_fmt='yuv420p',
                codec='libx264',
                keyframe_interval=20,
            )
            save_frames_to_video(
                frames=frames,
                video_path=source_b,
                fps=12,
                pix_fmt='yuv420p',
                codec='libx264',
                keyframe_interval=20,
            )

            with self.assertRaises(ValueError):
                combine_video_paths_into_chunk([source_a, source_b], combined)


class TestNormalizeCodec(unittest.TestCase):
    def test_normalizes_encoder_and_probed_names(self):
        # Encoder names collapse onto the canonical stream name.
        self.assertEqual(normalize_codec("libx265"), "h265")
        self.assertEqual(normalize_codec("libx264"), "h264")
        # The ffprobe "hevc" spelling normalizes to the canonical "h265".
        self.assertEqual(normalize_codec("hevc"), "h265")
        # Canonical names (and ffv1) are returned unchanged.
        self.assertEqual(normalize_codec("h265"), "h265")
        self.assertEqual(normalize_codec("h264"), "h264")
        self.assertEqual(normalize_codec("ffv1"), "ffv1")
        # Unknown codecs pass through unchanged so callers can raise a clear error.
        self.assertEqual(normalize_codec("vp9"), "vp9")
