import os
import shutil
import threading
import unittest
from typing import List
from unittest import mock

import av
import numpy as np

from ladder.core.common.path_utils import create_tempdir
from ladder.core.common.video_utils.load_video import VideoFrameReader, decode_frame_from_video

# NOTE: this video has multiple keyframes, so reads exercise the seek-then-decode-forward path
VIDEO_PATH = os.path.join(os.path.dirname(__file__), 'videos/video_with_keyframes.mp4')


def _num_frames(video_path: str) -> int:
    with av.open(video_path) as container:
        return container.streams.video[0].frames


class TestVideoFrameReader(unittest.TestCase):
    """`VideoFrameReader` must return exactly what `decode_frame_from_video` returns while opening each
    video at most once, since the whole point is to stop re-parsing the seek index per frame."""

    def setUp(self) -> None:
        self.num_frames = _num_frames(VIDEO_PATH)
        self.reader = VideoFrameReader()
        self.addCleanup(self.reader.close)

    def test_frames_match_uncached_decode(self) -> None:
        # Out-of-order, with repeats, so a reused container has to survive backward seeks
        indices = [0, self.num_frames - 1, 3, 1, 3, self.num_frames // 2, 0]

        for index in indices:
            np.testing.assert_array_equal(
                np.asarray(self.reader.decode_frame(VIDEO_PATH, frame_index=index)),
                np.asarray(decode_frame_from_video(VIDEO_PATH, frame_index=index)),
                err_msg=f'frame {index} differs from the uncached decode',
            )

    def test_opens_each_video_once(self) -> None:
        with mock.patch(
            'ladder.core.common.video_utils.load_video.av.open', side_effect=av.open
        ) as spied_open:
            for index in range(self.num_frames):
                self.reader.decode_frame(VIDEO_PATH, frame_index=index)

        self.assertEqual(spied_open.call_count, 1)
        self.assertEqual(self.reader.num_cached_videos, 1)

    def test_evicts_least_recently_used(self) -> None:
        videos = self._copies(count=3)
        reader = VideoFrameReader(max_cached_videos=2)
        self.addCleanup(reader.close)

        for video_path in videos:
            reader.decode_frame(video_path, frame_index=0)

        self.assertEqual(reader.num_cached_videos, 2)
        # The first video was evicted, so it has to be reopened
        with mock.patch(
            'ladder.core.common.video_utils.load_video.av.open', side_effect=av.open
        ) as spied_open:
            reader.decode_frame(videos[0], frame_index=0)
            reader.decode_frame(videos[2], frame_index=0)
        self.assertEqual(spied_open.call_count, 1)

    def test_unlimited_cache_never_evicts(self) -> None:
        videos = self._copies(count=3)
        reader = VideoFrameReader(max_cached_videos=None)
        self.addCleanup(reader.close)

        for video_path in videos:
            reader.decode_frame(video_path, frame_index=0)

        self.assertEqual(reader.num_cached_videos, len(videos))

    def test_rejects_non_positive_cache_size(self) -> None:
        with self.assertRaises(ValueError):
            VideoFrameReader(max_cached_videos=0)

    def test_concurrent_reads_of_one_video(self) -> None:
        """Threads share one container, so it must be serialized rather than interleaved."""
        indices = list(range(self.num_frames)) * 2
        expected = {
            index: np.asarray(decode_frame_from_video(VIDEO_PATH, frame_index=index))
            for index in range(self.num_frames)
        }
        failures: List[str] = []

        def read_all() -> None:
            for index in indices:
                frame = np.asarray(self.reader.decode_frame(VIDEO_PATH, frame_index=index))
                if not np.array_equal(frame, expected[index]):
                    failures.append(f'frame {index} decoded incorrectly under concurrency')

        threads = [threading.Thread(target=read_all) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(failures, [])
        self.assertEqual(self.reader.num_cached_videos, 1)

    def _copies(self, count: int) -> List[str]:
        """Distinct paths for the same video, so cache behaviour can be tested by path."""
        tempdir = create_tempdir(self)
        paths = []
        for index in range(count):
            path = os.path.join(tempdir, f'video_{index}.mp4')
            shutil.copyfile(VIDEO_PATH, path)
            paths.append(path)
        return paths


if __name__ == '__main__':
    unittest.main()
