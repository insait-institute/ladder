import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import av

from ladder.core.common.logger import get_logger
from ladder.core.common.path_utils import hardlink_or_copy_file

from .common import tqdm_progress_bar
from .load_video import probe_video_stream

LOGGER = get_logger(__name__)


def transcode_video(
    source_video_path: str | Path,
    output_video_path: str | Path,
    fps: int,
    pix_fmt: str = "yuv420p",
    codec: str = "libx265",
    keyframe_interval: int = 20,
    preset: str = "superfast",
) -> None:
    """Transcode a video one frame at a time without materializing the full video in memory."""
    source_video_path = Path(source_video_path)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    with av.open(str(source_video_path)) as input_container:
        input_stream = next(stream for stream in input_container.streams if stream.type == 'video')

        output_container = av.open(str(output_video_path), mode="w")

        output_stream = output_container.add_stream(codec, rate=fps)
        output_stream.width = int(input_stream.width)
        output_stream.height = int(input_stream.height)
        output_stream.pix_fmt = pix_fmt
        output_stream.options = {"g": str(keyframe_interval)}

        if codec == "libx265":
            output_stream.options["preset"] = preset
            output_stream.options["x265-params"] = f"log-level=none:keyint={keyframe_interval}:bframes=0"
        elif codec == "libx264":
            output_stream.options["preset"] = preset

        with tqdm_progress_bar(
            desc=f"Transcoding {output_video_path}",
            unit="frame",
        ) as progress_bar:
            for decoded_frame in input_container.decode(video=0):
                input_frame_format = "gray" if decoded_frame.format.name.startswith("gray") else "rgb24"
                np_frame = decoded_frame.to_ndarray(format=input_frame_format)
                encoded_frame = av.VideoFrame.from_ndarray(np_frame, format=input_frame_format)
                for packet in output_stream.encode(encoded_frame):
                    output_container.mux(packet)
                    progress_bar.update(1)

            for packet in output_stream.encode():
                output_container.mux(packet)

        output_container.close()


def _seek_to_keyframe_packet(
    container: av.container.InputContainer,
    stream: av.video.stream.VideoStream,
    frame_index: int,
) -> av.Packet:
    """
    Seek to the keyframe at frame_index and return that keyframe packet.

    Uses the same retry logic as decode_frame_from_video: container.seek() with
    backward=True sometimes lands on the *next* keyframe instead of the previous one
    for B-frame encoded videos (see the detailed explanation in load_video.py).
    When that happens we move the seek target backwards by an increasing offset until
    we land before the target, then scan forward to the exact keyframe.
    """
    fps = float(stream.average_rate)

    seek_offset = 0
    max_tries = 10

    for i in range(max_tries):
        seek_pts = round(max(frame_index - seek_offset, 0) / fps / float(stream.time_base))
        container.seek(seek_pts, backward=True, any_frame=False, stream=stream)

        for packet in container.demux(stream):
            if packet.dts is None and packet.pts is None:
                continue
            if not packet.is_keyframe:
                continue
            packet_pts = int(packet.pts) if packet.pts is not None else int(packet.dts)
            packet_frame_index = round(packet_pts * float(stream.time_base) * fps)
            if packet_frame_index == frame_index:
                return packet
            if packet_frame_index > frame_index:
                seek_offset = (i + 1) ** 2
                break
            # packet_frame_index < frame_index: keep scanning forward for the target keyframe

    raise ValueError(
        f"Could not find keyframe at frame {frame_index} after {max_tries} attempts. "
        "Make sure start_frame is at a keyframe boundary."
    )


def combine_video_paths_into_chunk(  # noqa: PLR0912, PLR0915, C901
    video_paths: List[str | Path],
    output_video_path: str | Path,
    frame_ranges: Optional[List[Tuple[int, int]]] = None,
) -> None:
    """
    Concatenate encoded videos into one chunk without re-encoding.

    Requirements enforced:
      - every input has identical codec / pixel format / fps / geometry / time_base
      - the first frame of each segment (after seeking to start_frame) must be a keyframe

    Args:
        video_paths: Input video paths to concatenate.
        output_video_path: Path to write the combined output.
        frame_ranges: Optional list of (start_frame, end_frame) pairs, one per entry in video_paths.
            start_frame is inclusive, end_frame is exclusive. When None, all frames from each video are used.
            start_frame must land on a keyframe

    This function performs packet-level remuxing, preserving source quality and
    keeping output size close to the sum of the included source segments.
    """
    assert len(video_paths) > 0
    if frame_ranges is not None and len(frame_ranges) != len(video_paths):
        raise ValueError("frame_ranges must have the same length as video_paths")

    # Input and output videos are the same
    if len(video_paths) == 1 and frame_ranges is None:
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
        hardlink_or_copy_file(video_paths[0], output_video_path)
        return

    frame_ranges = frame_ranges or [(0, None)] * len(video_paths)

    input_paths = [Path(video_path) for video_path in video_paths]
    for video_path in input_paths:
        if not video_path.exists():
            raise FileNotFoundError(str(video_path))

    reference_video_encoding: Dict[str, Any] = probe_video_stream(input_paths[0])
    for video_path in input_paths[1:]:
        video_encoding: Dict[str, Any] = probe_video_stream(video_path)
        if video_encoding != reference_video_encoding:
            raise ValueError(
                f"Video encoding mismatch for {video_path}. "
                f"Expected {reference_video_encoding}, got {video_encoding}"
            )

    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    for path, (start, end) in zip(input_paths, frame_ranges, strict=True):
        if end is not None:
            total_frames += end - start
        else:
            with av.open(str(path)) as c:
                total_frames += max(0, (c.streams.video[0].frames or 0) - start)

    with (
        tqdm_progress_bar(
            desc=f"Muxing {output_video_path}",
            total=total_frames,
            unit="frame",
        ) as progress_bar,
        av.open(str(input_paths[0])) as reference_container,
    ):
        reference_stream: av.video.stream.VideoStream = reference_container.streams.video[0]
        with av.open(str(output_video_path), mode='w') as output_container:
            # opaque=True copies the codec object straight from the input (a decoder) instead of
            # constructing a fresh encoder for the output stream. This is a pure packet-copy remux, so an
            # encoder is never needed, and creating one is actively harmful: PyAV opens it in
            # start_encoding() before writing the header, which (a) wastes an encoder init and lets libx265
            # print its banner + "encoded 0 frames" to fd 2 over the tqdm bars, and (b) fails outright for
            # the lossless depth stream (probes as h264/gbrp), where the libx264 encoder rejects the RGB
            # pixel format with EINVAL even though no frame is ever encoded.
            output_stream = output_container.add_stream_from_template(reference_stream, opaque=True)

            # opaque=True skips the encoder that otherwise carried the fps-derived timescale, so the muxer
            # would fall back to its 90 kHz default. Copy the source time_base to keep the remux a faithful
            # copy. Otherwise re-chunking an already-chunked dataset mixes 90 kHz segments with the
            # source-timescale ones and trips the encoding-consistency check above.
            output_stream.time_base = reference_stream.time_base

            timestamp_offset = 0
            # DTS actually written for the previous packet. Concatenated segments can have different
            # B-frame reorder depths - e.g. a short clip whose only packet has DTS == PTS followed by a
            # B-frame clip whose first packet has a negative DTS. In that case the later segment's first
            # DTS can dip to or below the previous segment's last DTS, and the MP4 muxer rejects the
            # non-monotonic DTS with EINVAL ("Invalid argument"). We guard against that below by nudging
            # DTS up to stay strictly increasing. Only DTS is ever adjusted (never PTS), so PTS-based
            # frame-by-index lookup is unaffected and the output stays pixel-identical to the inputs.
            # For inputs without B-frames the guard never triggers (DTS is already strictly increasing
            # and equal to PTS), so those remuxes keep a clean DTS == PTS stream.
            last_muxed_dts: Optional[int] = None
            for video_path, (start_frame, end_frame) in zip(input_paths, frame_ranges, strict=True):
                with av.open(str(video_path)) as input_container:
                    input_stream: av.video.stream.VideoStream = input_container.streams.video[0]
                    if (start_frame, end_frame) != (0, None) and input_stream.codec_context.has_b_frames:
                        LOGGER.warn(
                            "Source video encoded with B-frames. Packed-level remuxing might produce faulty "
                            "frames if B-frames keyframes are cropped out of the video. Re-encode with "
                            "bframes=0 for clean results. Note that re-encoding might compress already "
                            f"compressed video. Source video: {video_path}"
                        )

                    fps = float(input_stream.average_rate)

                    first_packet = _seek_to_keyframe_packet(input_container, input_stream, start_frame)
                    packet_iterator = input_container.demux(input_stream)

                    first_packet_dts = 0 if first_packet.dts is None else int(first_packet.dts)
                    # Use PTS of the first packet as the base for both DTS and PTS adjustments.
                    # This preserves the DTS/PTS relationship (pts >= dts at all times) and keeps
                    # PTS values at the expected frame positions (i * frame_period) so that
                    # seek-based decoding (decode_frame_from_video) can locate frames by index.
                    # The first segment may produce negative DTS for B-frame-encoded sources;
                    # the MP4 muxer handles this via an edit list, the same way the source was created.
                    first_packet_pts = first_packet_dts if first_packet.pts is None else int(first_packet.pts)
                    segment_max_timestamp = timestamp_offset - 1

                    for packet in itertools.chain([first_packet], packet_iterator):
                        if packet.dts is None and packet.pts is None:
                            continue
                        if end_frame is not None and packet.pts is not None:
                            packet_frame_index = round(int(packet.pts) * float(input_stream.time_base) * fps)
                            if packet_frame_index >= end_frame:
                                break

                        if packet.dts is not None:
                            packet.dts = int(packet.dts) - first_packet_pts + timestamp_offset
                            if last_muxed_dts is not None and packet.dts <= last_muxed_dts:
                                packet.dts = last_muxed_dts + 1
                            last_muxed_dts = packet.dts
                            segment_max_timestamp = max(segment_max_timestamp, packet.dts)
                        if packet.pts is not None:
                            packet.pts = int(packet.pts) - first_packet_pts + timestamp_offset
                            segment_max_timestamp = max(segment_max_timestamp, packet.pts)

                        packet.stream = output_stream
                        output_container.mux(packet)
                        progress_bar.update(1)

                    if segment_max_timestamp < timestamp_offset:
                        raise ValueError(f"No valid video packets were muxed from {video_path}")

                    if end_frame is not None:
                        segment_duration = round(
                            (end_frame - start_frame) / fps / float(input_stream.time_base)
                        )
                    else:
                        segment_duration = input_stream.duration
                        if segment_duration is None:
                            segment_duration = segment_max_timestamp - timestamp_offset + 1
                    timestamp_offset += max(int(segment_duration), 1)
