import collections.abc
import unittest
from pathlib import Path

import numpy as np

from ladder.core.common.path_utils import create_tempdir
from ladder.core.common.video_utils.depth_video import (
    INVALID_DEPTH_VALUE,
    clip_and_replace_invalid_depth,
    decode_video_to_depth_uint16,
    depth_meters_to_uint16,
    depth_uint16_to_meters,
    encode_depth_uint16_to_video,
    iter_depth_frames_from_video,
    iter_depth_uint16_from_video,
    load_depth_frames_from_video,
    metric_depth_to_uint16,
    replace_invalid_depth_with_nan,
    save_depth_frames_to_video,
)


def _valid_depth_meters(num_frames: int = 4, height: int = 16, width: int = 24) -> np.ndarray:
    """Float32 depth in [0, INVALID_DEPTH_VALUE) m, so nothing gets clipped/marked during a roundtrip."""
    rng = np.random.default_rng(0)
    return (rng.random((num_frames, height, width), dtype=np.float32) * (INVALID_DEPTH_VALUE - 4.0)).astype(
        np.float32
    )


def _depth_uint16(num_frames: int = 5, height: int = 16, width: int = 24) -> np.ndarray:
    rng = np.random.default_rng(1)
    depth = rng.integers(0, 65536, size=(num_frames, height, width), dtype=np.uint16)
    # Pin byte-boundary values so the high/low-byte packing is exercised at its extremes.
    depth[0, 0, :4] = np.array([0, 255, 256, 65535], dtype=np.uint16)
    return depth


class TestDepthTransforms(unittest.TestCase):
    def test_meters_uint16_roundtrip_within_millimeter(self):
        meters = _valid_depth_meters()
        recovered = depth_uint16_to_meters(depth_meters_to_uint16(meters))
        self.assertEqual(recovered.dtype, np.float32)
        self.assertLess(np.max(np.abs(recovered - meters)), 1.1e-3)

    def test_transforms_preserve_container_type(self):
        meters = _valid_depth_meters()

        for transform in (
            depth_meters_to_uint16,
            clip_and_replace_invalid_depth,
            replace_invalid_depth_with_nan,
        ):
            array_out = transform(meters)
            self.assertIsInstance(array_out, np.ndarray)

            iterable_out = transform(iter(meters))
            self.assertIsInstance(iterable_out, collections.abc.Iterator)
            self.assertNotIsInstance(iterable_out, np.ndarray)

            stacked = np.stack(list(iterable_out))
            np.testing.assert_array_equal(np.nan_to_num(stacked), np.nan_to_num(array_out))

    def test_clip_and_replace_invalid_depth(self):
        frame = np.array(
            [[np.nan, np.inf], [-np.inf, INVALID_DEPTH_VALUE + 10.0], [-5.0, 1.5]], dtype=np.float32
        )
        result = clip_and_replace_invalid_depth(frame)
        expected = np.array(
            [
                [INVALID_DEPTH_VALUE, INVALID_DEPTH_VALUE],
                [INVALID_DEPTH_VALUE, INVALID_DEPTH_VALUE],
                [0.0, 1.5],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(result, expected)

    def test_replace_invalid_depth_with_nan(self):
        frame = np.array([[INVALID_DEPTH_VALUE, 1.0], [2.5, INVALID_DEPTH_VALUE]], dtype=np.float32)
        result = replace_invalid_depth_with_nan(frame)
        self.assertTrue(np.isnan(result[0, 0]))
        self.assertTrue(np.isnan(result[1, 1]))
        np.testing.assert_array_equal(result[[0, 1], [1, 0]], np.array([1.0, 2.5], dtype=np.float32))

    def test_dtype_mismatch_raises(self):
        # Array inputs are validated eagerly.
        with self.assertRaises(TypeError):
            depth_meters_to_uint16(_depth_uint16())
        with self.assertRaises(TypeError):
            depth_uint16_to_meters(_valid_depth_meters())

        # Iterable inputs are validated lazily, when the offending frame is consumed.
        lazy = depth_uint16_to_meters(iter(_valid_depth_meters()))
        with self.assertRaises(TypeError):
            next(lazy)

    def test_metric_depth_to_uint16(self):
        # Float meters are cleaned + scaled to millimeters; uint16 is assumed to already be mm.
        meters = _valid_depth_meters()
        expected = depth_meters_to_uint16(clip_and_replace_invalid_depth(meters))
        np.testing.assert_array_equal(metric_depth_to_uint16(meters), expected)

        already_uint16 = _depth_uint16()
        np.testing.assert_array_equal(metric_depth_to_uint16(already_uint16), already_uint16)

        # Iterable input stays lazy and matches the array result.
        lazy = metric_depth_to_uint16(iter(meters))
        self.assertIsInstance(lazy, collections.abc.Iterator)
        np.testing.assert_array_equal(np.stack(list(lazy)), expected)


class TestDepthVideoRoundtrip(unittest.TestCase):
    def setUp(self):
        self.output_path = Path(create_tempdir(self))

    def test_encode_decode_uint16_is_lossless(self):
        depth = _depth_uint16()
        video_path = self.output_path / "depth.mp4"

        encode_depth_uint16_to_video(depth, video_path, fps=5)
        decoded = decode_video_to_depth_uint16(video_path)

        self.assertEqual(decoded.dtype, np.uint16)
        np.testing.assert_array_equal(decoded, depth)

    def test_iter_decode_matches_array_decode(self):
        depth = _depth_uint16()
        video_path = self.output_path / "depth.mp4"
        encode_depth_uint16_to_video(depth, video_path, fps=5)

        streamed = np.stack(list(iter_depth_uint16_from_video(video_path)))
        np.testing.assert_array_equal(streamed, decode_video_to_depth_uint16(video_path))

    def test_iterable_encode_matches_array_encode(self):
        depth = _depth_uint16()
        from_array = self.output_path / "from_array.mp4"
        from_iterable = self.output_path / "from_iterable.mp4"

        encode_depth_uint16_to_video(depth, from_array, fps=5)
        encode_depth_uint16_to_video((frame for frame in depth), from_iterable, fps=5)

        np.testing.assert_array_equal(
            decode_video_to_depth_uint16(from_iterable), decode_video_to_depth_uint16(from_array)
        )


class TestSaveLoadDepthFrames(unittest.TestCase):
    def setUp(self):
        self.output_path = Path(create_tempdir(self))

    def test_float_roundtrip_within_millimeter_and_restores_nan(self):
        meters = _valid_depth_meters()
        meters[0, 0, 0] = np.nan
        video_path = self.output_path / "depth.mp4"

        save_depth_frames_to_video(meters, video_path, fps=5, downsample_factor=1)
        loaded = load_depth_frames_from_video(video_path, upsample_factor=1)

        self.assertEqual(loaded.dtype, np.float32)
        self.assertEqual(loaded.shape, meters.shape)
        self.assertTrue(np.isnan(loaded[0, 0, 0]))
        finite = ~np.isnan(meters)
        self.assertLess(np.max(np.abs(loaded[finite] - meters[finite])), 1.1e-3)

    def test_iter_load_matches_array_load(self):
        meters = _valid_depth_meters()
        video_path = self.output_path / "depth.mp4"
        save_depth_frames_to_video(meters, video_path, fps=5, downsample_factor=1)

        streamed = np.stack(list(iter_depth_frames_from_video(video_path, upsample_factor=1)))
        np.testing.assert_array_equal(streamed, load_depth_frames_from_video(video_path, upsample_factor=1))

    def test_iterable_save_matches_array_save(self):
        meters = _valid_depth_meters()
        from_array = self.output_path / "from_array.mp4"
        from_iterable = self.output_path / "from_iterable.mp4"

        save_depth_frames_to_video(meters, from_array, fps=5, downsample_factor=2)
        save_depth_frames_to_video((frame for frame in meters), from_iterable, fps=5, downsample_factor=2)

        np.testing.assert_array_equal(
            load_depth_frames_from_video(from_iterable), load_depth_frames_from_video(from_array)
        )

    def test_downsample_upsample_preserves_shape(self):
        meters = _valid_depth_meters(height=16, width=24)
        video_path = self.output_path / "depth.mp4"

        save_depth_frames_to_video(meters, video_path, fps=5, downsample_factor=2)
        loaded = load_depth_frames_from_video(video_path, upsample_factor=2)

        self.assertEqual(loaded.shape, meters.shape)


if __name__ == "__main__":
    unittest.main()
