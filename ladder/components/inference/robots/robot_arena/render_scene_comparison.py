#!/usr/bin/env python3
"""Render first frame of each RobotArena scene for visual comparison.

Renders scenes with optional object annotations, composites over backgrounds,
and optionally serves results via HTTP for remote viewing.

Requires GPU (Genesis).

Usage:
    bazel run //ladder/components/inference/robots/robot_arena:render_scene_comparison -- \
        --fast --annotate --serve --scenes default1,scene1,scene2
"""

import argparse
import functools
import http.server
import os
import re
import socket
import subprocess

import cv2
import numpy as np

from ladder.components.inference.robots.robot_arena import (
    scene_corrections as scene_corrections_module,
    scene_loader,
)
from ladder.components.inference.robots.robot_arena.robot_arena_env import RobotArenaEnvConfig
from ladder.components.inference.robots.robot_arena.scene_builder import (
    build_genesis_scene,
    resolve_object_overlaps,
)
from ladder.components.inference.robots.robot_arena.scene_loader import (
    SETTLE_STEPS,
    composite_foreground_over_background,
    list_available_scenes,
    load_scene,
)
from ladder.components.inference.robots.robot_arena.transforms import RobotName

# Genesis writes to $HOME/.cache on import; ensure HOME is set.
if "HOME" not in os.environ:
    os.environ["HOME"] = os.environ.get("TEST_TMPDIR", "/tmp")

_CORRECTIONS_RELPATH = "ladder/components/inference/robots/robot_arena/scene_corrections.py"
_LOCAL_CORRECTIONS_COPY = "/tmp/scene_corrections_live.py"


def annotate_objects_on_image(
    image: np.ndarray,
    seg: np.ndarray,
    objects: list,
    num_robot_links: int = 0,
) -> np.ndarray:
    """Overlay object names and indices on rendered image using segmentation IDs.

    Genesis assigns sequential segmentation IDs to entities/links:
      - Ground plane gets ID 0 (but has visualization=False, won't appear)
      - Robot URDF links get IDs 1..num_robot_links
      - Scene objects get IDs (num_robot_links+1), (num_robot_links+2), ...

    If num_robot_links is 0 (unknown), the offset is auto-detected from the
    segmentation mask by assuming the last len(objects) unique non-zero IDs
    are the scene objects in order.
    """
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thickness = 1

    if num_robot_links > 0:
        first_object_seg_id = num_robot_links + 1
    else:
        unique_ids = sorted(int(v) for v in np.unique(seg) if v > 0)
        if len(unique_ids) >= len(objects):
            first_object_seg_id = unique_ids[-len(objects)]
        else:
            first_object_seg_id = 2

    for object_index, object_config in enumerate(objects):
        seg_id = object_index + first_object_seg_id
        mask = seg == seg_id
        if int(np.sum(mask)) == 0:
            continue

        ys, xs = np.where(mask)
        centroid_x = int(np.mean(xs))
        centroid_y = int(np.mean(ys))

        label = f"[{object_index}] {object_config.name}"
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

        padding = 4
        overlay = annotated.copy()
        cv2.rectangle(
            overlay,
            (centroid_x - padding, centroid_y - text_height - 2 * padding),
            (centroid_x + text_width + padding, centroid_y + padding),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        cv2.putText(
            annotated, label, (centroid_x, centroid_y), font, font_scale, (255, 255, 255), font_thickness
        )
        cv2.circle(annotated, (centroid_x, centroid_y), 3, (0, 255, 0), cv2.FILLED)

    return annotated


# ---------------------------------------------------------------------------
# Corrections loading: fetch from login node via scp to bypass CephFS caching
# ---------------------------------------------------------------------------


def _resolve_corrections_path(explicit_path: str | None) -> str | None:
    """Find the source-tree scene_corrections.py."""
    if explicit_path is not None:
        return explicit_path if os.path.isfile(explicit_path) else None

    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace:
        candidate = os.path.join(workspace, _CORRECTIONS_RELPATH)
        if os.path.isfile(candidate):
            return candidate

    candidate = os.path.join(os.path.expanduser("~"), "ladder", _CORRECTIONS_RELPATH)
    if os.path.isfile(candidate):
        return candidate
    return None


def _fetch_corrections(source_path: str) -> str:
    """Fetch corrections file from the login node via scp to /tmp.

    If source_path is already on /tmp (local disk), reads it directly.
    Otherwise scps from SLURM_SUBMIT_HOST to bypass CephFS caching.

    Returns the local path to read from.
    """
    # If path is already on local disk (e.g. pre-copied by wrapper script), use it directly
    if source_path.startswith("/tmp/"):
        return source_path

    submit_host = os.environ.get("SLURM_SUBMIT_HOST", "")
    current_host = socket.gethostname()

    if not submit_host or current_host == submit_host:
        return source_path

    try:
        result = subprocess.run(
            ["scp", "-q", f"{submit_host}:{source_path}", _LOCAL_CORRECTIONS_COPY],
            capture_output=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            return _LOCAL_CORRECTIONS_COPY
        print(f"  WARNING: scp failed: {result.stderr.decode().strip()}")
    except Exception as error:  # noqa: BLE001
        print(f"  WARNING: scp error: {error}")
    return source_path


def load_and_apply_corrections(source_path: str) -> None:
    """Fetch, parse, and apply corrections to scene_loader in one call."""
    local_path = _fetch_corrections(source_path)

    namespace: dict = {}
    try:
        with open(local_path, "r", encoding="utf-8") as file_handle:
            exec(compile(file_handle.read(), local_path, "exec"), namespace)  # noqa: S102
    except Exception as error:  # noqa: BLE001
        print(f"  ERROR loading corrections: {error}")
        return

    corrections = namespace.get("UPSTREAM_DATA_CORRECTIONS")
    if corrections is None:
        print("  ERROR: UPSTREAM_DATA_CORRECTIONS not found in file")
        return

    scene_loader.UPSTREAM_DATA_CORRECTIONS = corrections

    # Also apply background overrides if present
    bg_overrides = namespace.get("BACKGROUND_OVERRIDES")
    if bg_overrides is not None:
        scene_corrections_module.BACKGROUND_OVERRIDES = bg_overrides
        print(
            f"  Applied {len(corrections)} corrections + {len(bg_overrides)} background overrides from {local_path}"
        )
    else:
        print(f"  Applied {len(corrections)} corrections from {local_path}")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_scene_first_frame(
    data_root: str,
    scene_name: str,
    output_dir: str,
    fast: bool = False,
    annotate: bool = False,
) -> None:
    """Render the first frame of a scene and save comparison image."""
    print(f"Rendering scene: {scene_name}")

    scene_data = load_scene(data_root, scene_name)
    config = RobotArenaEnvConfig(
        data_root=data_root,
        scene_name=scene_name,
        robot_name=RobotName.WIDOWX,
        max_episode_steps=100,
        control_freq=5,
        sim_freq=200,
        camera_width=640,
        camera_height=480,
    )

    ctx = build_genesis_scene(
        scene_data,
        data_root=config.data_root,
        sim_freq=config.sim_freq,
        robot_name=config.robot_name,
        camera_width=config.camera_width,
        camera_height=config.camera_height,
        camera_offset=config.camera_offset,
        render_only=fast,
    )

    if not fast:
        for _ in range(SETTLE_STEPS):
            ctx.scene.step()
        resolve_object_overlaps(ctx.object_entities)

    rgb, _, seg, _ = ctx.camera.render(rgb=True, depth=True, segmentation=True)
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    if hasattr(seg, "cpu"):
        seg = seg.cpu().numpy()

    # Check for background overrides (e.g. scene17 has wrong upstream background)
    bg_path = scene_data.background_path
    bg_override = scene_corrections_module.BACKGROUND_OVERRIDES.get(scene_name)
    if bg_override is not None:
        override_bg = os.path.join(data_root, "scene_background", bg_override.source_scene, "background.png")
        if os.path.exists(override_bg):
            bg_path = override_bg
            print(f"  Background override: using {bg_override.source_scene} ({bg_override.reason[:60]}...)")

    if bg_path and os.path.exists(bg_path):
        background = cv2.imread(bg_path)
        if background is not None:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
            final_image = composite_foreground_over_background(rgb, seg, background)
        else:
            final_image = rgb
    else:
        final_image = rgb

    if annotate:
        num_robot_links = len(ctx.robot.links)
        final_image = annotate_objects_on_image(final_image, seg, scene_data.objects, num_robot_links)
        print(f"  {scene_name} objects:")
        for object_index, object_config in enumerate(scene_data.objects):
            pos = object_config.position
            quat = object_config.rotation_quat
            print(
                f"    [{object_index}] {object_config.name:<30s} "
                f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
                f"quat=[{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]"
            )

    # Add title bar
    final_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
    h, w = final_bgr.shape[:2]
    title_h = 50
    titled = np.zeros((h + title_h, w, 3), dtype=np.uint8)
    titled[title_h:] = final_bgr
    cv2.putText(
        titled,
        f"{scene_name}: \"{scene_data.task_instruction}\"",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{scene_name}.png")
    cv2.imwrite(output_path, titled)
    print(f"  Saved: {output_path}")

    ctx.scene.destroy()


# ---------------------------------------------------------------------------
# HTML gallery + HTTP server
# ---------------------------------------------------------------------------


def _natural_sort_key(name: str) -> list:
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


def _write_gallery_html(output_dir: str) -> None:
    """Write an HTML gallery listing all rendered PNGs."""
    images = sorted((f for f in os.listdir(output_dir) if f.endswith(".png")), key=_natural_sort_key)
    html = (
        "<!DOCTYPE html><html><head><title>Scene Renders</title>\n"
        "<style>\n"
        "  body{background:#1a1a1a;color:#eee;font-family:monospace;margin:20px}\n"
        "  img{max-width:640px;margin:5px;border:1px solid #444}\n"
        "  .grid{display:flex;flex-wrap:wrap;gap:10px}\n"
        "  .cell{text-align:center}\n"
        "</style></head><body>\n"
        f"<h1>{len(images)} rendered scenes</h1>\n"
        "<div class='grid'>\n"
    )
    for image_name in images:
        scene_label = image_name.replace(".png", "")
        html += (
            f"<div class='cell'><a href='/{image_name}' target='_blank'>"
            f"<img src='/{image_name}'></a><br>{scene_label}</div>\n"
        )
    html += "</div></body></html>"

    with open(os.path.join(output_dir, "index.html"), "w") as file_handle:
        file_handle.write(html)


def _serve_directory(output_dir: str, port: int) -> None:
    """Start an HTTP server to browse rendered images. Blocks until Ctrl+C."""
    handler_class = functools.partial(http.server.SimpleHTTPRequestHandler, directory=output_dir)
    hostname = socket.getfqdn()
    print(f"\nServing at: http://{hostname}:{port}/index.html")
    print("Press Ctrl+C to stop.")
    server = http.server.HTTPServer(("0.0.0.0", port), handler_class)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Render RobotArena scene comparisons")
    parser.add_argument(
        "--data-root",
        default="ladder/components/inference/robots/robot_arena/robot_arena_data/examples/data",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.expanduser("~"), "ra_scene_comparisons"),
    )
    parser.add_argument("--scenes", default="", help="Comma-separated scene names (empty=all)")
    parser.add_argument("--fast", action="store_true", help="Render-only mode (no physics)")
    parser.add_argument("--annotate", action="store_true", help="Overlay object labels")
    parser.add_argument("--serve", action="store_true", help="Start HTTP server after rendering")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--corrections-path", default=None, help="Path to scene_corrections.py")
    args = parser.parse_args()

    # Load corrections from source tree (bypasses Bazel sandbox)
    corrections_path = _resolve_corrections_path(args.corrections_path)
    if corrections_path is not None:
        load_and_apply_corrections(corrections_path)

    if args.scenes:
        scene_names = [s.strip() for s in args.scenes.split(",")]
    else:
        scene_names = list_available_scenes(args.data_root)

    for scene_name in scene_names:
        try:
            render_scene_first_frame(args.data_root, scene_name, args.output_dir, args.fast, args.annotate)
        except Exception as exception:  # noqa: BLE001
            print(f"  ERROR rendering {scene_name}: {exception}")

    if args.serve:
        _write_gallery_html(args.output_dir)
        _serve_directory(args.output_dir, args.port)


if __name__ == "__main__":
    main()
