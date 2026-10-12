#!/usr/bin/env python3
"""Flag RoboLab episodes whose scene stopped being rendered.

The composition rig composes every object and receptacle onto the base rig as a USD payload, and
Hydra can evict streamed payload geometry mid-episode: the reset frame shows the full scene, then
the table, robot and objects vanish while the static room backdrop keeps rendering. Physics is
unaffected, so nothing errors -- but the recorder writes the same observation the policy is served,
so the policy is blind for the rest of the episode and the cell scores as an ordinary failure.

Frame 0 is the reset frame and is rendered before any step, so it is the per-episode reference:
an episode is flagged when the table occupies less than `--min-ratio` of the area it occupied then.
"""
import argparse
import colorsys
import glob
import os
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple

from PIL import Image

# The tabletop, in HSV. The rig's table is the one large light-wood surface in frame; the office
# backdrop behind it has no comparable region, which is what makes presence separable at all.
_HUE = (0.03, 0.13)
_MIN_SAT = 0.30
_MIN_VAL = 0.30


def table_fraction(path: str) -> float:
    image = Image.open(path).convert("RGB").resize((96, 96))
    hits = 0
    for red, green, blue in image.getdata():
        hue, sat, val = colorsys.rgb_to_hsv(red / 255, green / 255, blue / 255)
        if _HUE[0] <= hue <= _HUE[1] and sat >= _MIN_SAT and val >= _MIN_VAL:
            hits += 1
    return hits / (96 * 96)


def _frame(video: str, index: int, out: str) -> Optional[str]:
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            video,
            "-vf",
            f"select=eq(n\\,{index})",
            "-vframes",
            "1",
            out,
        ],
        check=False,
    )
    return out if os.path.exists(out) and os.path.getsize(out) else None


def scan(video: str, samples: int, workdir: str) -> Tuple[float, List[float]]:
    """(reference table fraction at reset, table fraction at each sampled step)."""
    count = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            video,
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    total = int(count) if count.isdigit() else 0
    if total < 2:
        return 0.0, []
    reference = _frame(video, 0, os.path.join(workdir, "ref.png"))
    if reference is None:
        return 0.0, []
    # Skip the first few steps: the payloads are still resolving and a frame there is not evidence.
    indices = [int(total * (i + 1) / (samples + 1)) for i in range(samples)]
    later = []
    for position, index in enumerate(indices):
        path = _frame(video, index, os.path.join(workdir, f"f{position}.png"))
        if path is not None:
            later.append(table_fraction(path))
    return table_fraction(reference), later


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="campaign directory (scanned for episodes/*/main.mp4)")
    parser.add_argument("--samples", type=int, default=4, help="frames to check per episode")
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.30,
        help="flag when the table shrinks below this share of its reset area",
    )
    parser.add_argument(
        "--min-absolute",
        type=float,
        default=0.02,
        help="table area below this is no scene at all, whatever the reset frame showed",
    )
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    args = parser.parse_args()

    videos = sorted(glob.glob(os.path.join(args.root, "**", "episodes", "*", "main.mp4"), recursive=True))
    if not videos:
        print(f"no episode videos under {args.root}", file=sys.stderr)
        return 2

    flagged = []
    with tempfile.TemporaryDirectory() as workdir:
        for video in videos:
            reference, later = scan(video, args.samples, workdir)
            if not later:
                continue
            # Two ways to be blind, and the absolute test is not redundant: an episode whose payloads
            # were already gone at reset has a degenerate REFERENCE, so its ratio looks healthy.
            worst_absolute = min(later)
            worst_ratio = min(later) / reference if reference > 0 else 0.0
            if worst_absolute < args.min_absolute or worst_ratio < args.min_ratio:
                flagged.append((video, worst_absolute))
                if not args.quiet:
                    print(
                        f"BLIND table={worst_absolute:.3f} ratio={worst_ratio:4.2f}  "
                        f"{os.path.relpath(video, args.root)}"
                    )
    print(
        f"\n{len(flagged)}/{len(videos)} episodes lost their scene ({100 * len(flagged) / len(videos):.1f}%)"
    )
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
