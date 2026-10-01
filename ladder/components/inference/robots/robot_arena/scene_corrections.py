"""
Upstream data corrections for RobotArena scenes.

The upstream RobotArena repository (offjangir/RobotArena) ships pre-generated scene
data that occasionally contains incorrect object transforms (e.g. cloth meshes rotated
90° so they stand vertically instead of lying flat) and occasionally incorrect background
images. Rather than forking the upstream repo or patching files at build time, we apply
targeted corrections at load time. Each entry documents what is wrong and why, so the
registry is self-auditing.

Correction methodology
----------------------
For **standard scenes** (non-default), the upstream 3D reconstruction pipeline often
produces object rotations with a ~90° X-axis tilt that causes objects to stand vertically
instead of lying flat. The fix is usually ``rotation_euler_deg=[0, 0, 0]`` (identity),
which resets the object to its native mesh orientation (as scanned). For objects whose
mesh is *natively* vertical (e.g. some utensils), ``[90, 0, Z]`` is used instead, where
Z preserves the original horizontal orientation from the upstream data.

For **default scenes**, the convention differs — the upstream rotation is Euler XYZ in
degrees and ``[90, 0, 0]`` is the typical fix for vertical cloths.

Verification workflow
---------------------
1. Render scenes:  ``./ladder/components/inference/robots/robot_arena/scripts/render_scenes.sh [scenes] --no-serve``
2. Create side-by-side comparison images:  ``python ladder/components/inference/robots/robot_arena/scripts/create_comparison.py``
3. Browse gallery:  serve ``~/ra_scene_comparisons/comparisons/all.html`` over HTTP
4. Inspect object transforms:  ``bazel run //.../inspect_scene_objects -- --scene <name>``

To add a new correction:
  1. Identify the scene name and the object index (0-based position in the sorted
     keys of transformations.json — i.e. image_0000_segmented = 0, etc.).
  2. Create a SceneCorrection with the fixed field(s) and a human-readable reason.
  3. Add it to UPSTREAM_DATA_CORRECTIONS keyed by (scene_name, object_index).
  4. Re-render and verify visually against ground truth images from
     ``data_root/bridge/<scene>/images0/0.jpg``.

Known upstream data issues (unfixable via corrections)
------------------------------------------------------
- **28 missing scenes**: scene13, scene26, scene51, scene52, scene56, scene58, scene59,
  scene61-69, scene73, scene74, scene78, scene83, scene86-89, scene91, scene92, scene95,
  scene96 — absent from the upstream dataset entirely.
- **scene17 background**: background.png shows a kitchen/sink instead of the actual table
  workspace. Overridden via BACKGROUND_OVERRIDES to use scene19's table background.
- **Thick cloth meshes (P7)**: Several cloth/towel objects were 3D-reconstructed as
  7-12cm thick solid blocks instead of thin sheets. Rotation fixes orient them correctly
  but the visual thickness is an upstream mesh quality limitation.
"""

import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.spatial.transform


@dataclasses.dataclass(frozen=True)
class SceneCorrection:
    """A targeted fix for a single object in an upstream RobotArena scene.

    Only non-None fields are applied; everything else is left as-is from the
    upstream data. All rotations are Euler XYZ in degrees, matching the upstream
    transformations.json convention for default scenes.
    """

    reason: str
    rotation_euler_deg: Optional[List[float]] = None
    translation: Optional[List[float]] = None
    scale: Optional[float] = None


# Key: (scene_name, object_index) where object_index is the 0-based position in
# the sorted keys of transformations.json (e.g. image_0000_segmented → 0).
UPSTREAM_DATA_CORRECTIONS: Dict[Tuple[str, int], SceneCorrection] = {
    ("default1", 0): SceneCorrection(
        reason="cloth_destination: upstream [90,0,0] X-rotation makes towel stand vertical; identity keeps it flat",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene1", 1): SceneCorrection(
        reason="red bell pepper: upstream [86.1,-19.9,-82.9] rotates pepper ~90° wrong; identity restores scan orientation",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene1", 2): SceneCorrection(
        reason="cloth_destination: upstream [93.7,-4.4,-83.7] tilts cloth ~90° vertical; identity keeps it flat",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene1", 3): SceneCorrection(
        reason="metal bowl_target: [0,0,90] (yaw) left it lying on its side; [90,0,0] "
        "settles it upright (opening up). Verified via --rotprobe render (2026-05-28).",
        rotation_euler_deg=[90, 0, 0],
    ),
    # --- scene2 ---
    ("scene2", 1): SceneCorrection(
        reason="blue cloth_target: upstream [88.3,-2.3,-100.1] tilts ~88°. Re-authored "
        "2026-05-27: target started in the 'upper_right' region the task moves it to "
        "(degenerate); shifted to lower-left so the task is doable.",
        rotation_euler_deg=[0, 0, 0],
        translation=[0.251, 0.07, 0.02],
    ),
    ("scene2", 2): SceneCorrection(
        reason="blue stuffed animal: upstream already [0,0,0]; kept for consistency",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene2", 3): SceneCorrection(
        reason="metal can: upstream [83.4,-6.8,-13.8] tilts can ~83° on its side",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene3 ---
    ("scene3", 0): SceneCorrection(
        reason="sauce_pan_target: upstream [106.9,-6.4,-5.9] tilts pan ~107°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene3", 1): SceneCorrection(
        reason="cloth_destination: upstream [81.4,-2.7,88.2] tilts cloth ~81°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene3", 2): SceneCorrection(
        reason="spoon: upstream already [0,0,0]; kept for consistency",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene3", 3): SceneCorrection(
        reason="can: upstream [108.1,-0.3,-29.5] tilts ~108°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene3", 4): SceneCorrection(
        reason="tomato: upstream [93.1,-2.6,-68.1] tilts ~93°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene4 ---
    ("scene4", 0): SceneCorrection(
        reason="small brush: upstream [106.6,-8.7,-3.7] tilts ~107°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene4", 1): SceneCorrection(
        reason="toy mushroom: upstream already [0,0,0]; kept for consistency",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene4", 2): SceneCorrection(
        reason="small metal bowl_target: upstream [106.7,-14.2,-108.3] tilts ~107°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene4", 3): SceneCorrection(
        reason="folded towel_destination: upstream [91.0,2.8,-90.9] tilts ~91°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene5 ---
    ("scene5", 1): SceneCorrection(
        reason="orange cup: upstream already [0,0,0]; kept for consistency",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene5", 2): SceneCorrection(
        reason="metal pot_destination: upstream [88.9,-6.8,-36.8] tilts pot ~89°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene6 ---
    ("scene6", 0): SceneCorrection(
        reason="Small metal pan: upstream [87.8,-13.3,-71.7] tilts pan ~88°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene6", 1): SceneCorrection(
        reason="Metal pan lid: upstream [93.9,6.7,86.4] tilts lid ~94°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene6", 2): SceneCorrection(
        reason="Yellow cloth: upstream already [0,0,0]; kept for consistency",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene6", 3): SceneCorrection(
        reason="Canned peaches_target: upstream [102.3,8.9,-93.6] tilts can ~102°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene7 ---
    ("scene7", 0): SceneCorrection(
        reason="metal pot with lid_destination: upstream already [0,0,0]; kept for consistency",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene7", 1): SceneCorrection(
        reason="spatula_target: upstream [139.0,-8.0,-30.8] tilts ~139°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene8 ---
    ("scene8", 0): SceneCorrection(
        reason="brown plastic pot_target: upstream [88.4,3.2,126.4] tilts ~88°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene8", 1): SceneCorrection(
        reason="brown plastic lid: upstream [108.5,-3.9,-67.1] tilts ~108°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene8", 2): SceneCorrection(
        reason="can of corn: upstream [107.0,-1.1,-71.1] tilts ~107°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene8", 3): SceneCorrection(
        reason="sushi: upstream [74.4,-6.5,-83.8] tilts ~74°",
        rotation_euler_deg=[0, 0, 90],
    ),
    # --- scene9 ---
    ("scene9", 0): SceneCorrection(
        reason="orange bowl_destination: upstream [67.6,-4.6,110.2] tilts ~68°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene9", 1): SceneCorrection(
        reason="metal can: upstream [69.1,-5.9,107.0] tilts ~69°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene9", 2): SceneCorrection(
        reason="blue folded towel: upstream already [0,0,0]; identity keeps flat",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene9", 4): SceneCorrection(
        reason="toy knife_target: upstream [102.9,-12.6,-94.9] tilts ~103°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene10 ---
    ("scene10", 0): SceneCorrection(
        reason="silver pot_target: upstream [88.3,-8.0,128.4] tilts ~88°. Re-authored "
        "2026-05-27: target started at the lower-right of the cooker (= the 'lower_right' "
        "goal region, user-confirmed degenerate); shifted to upper-left so the task is "
        "a genuine traversal. Verified settled + out-of-region.",
        rotation_euler_deg=[0, 0, 0],
        translation=[0.348, 0.113, 0.02],
    ),
    ("scene10", 1): SceneCorrection(
        reason="orange cloth: upstream [90.8,0.6,-87.8] tilts ~91°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene10", 2): SceneCorrection(
        reason="green spatula: upstream [108.9,2.7,-84.3] tilts ~109°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene10", 3): SceneCorrection(
        reason="yellow corn cob: upstream [104.9,11.0,98.8] tilts ~105°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- Remaining scenes: auto-generated corrections for objects with >30° rotation ---
    # --- scene11 ---
    ("scene11", 0): SceneCorrection(
        reason="toy food can: upstream [103.3,2.5,-60.7] tilts ~103°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene11", 1): SceneCorrection(
        reason="orange toy pan: upstream [72.5,-6.4,-21.6] tilts ~73°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene11", 2): SceneCorrection(
        reason="white toy spoon: upstream [58.2,20.2,-24.1] tilts ~58°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene11", 3): SceneCorrection(
        reason="white toy knife with yellow handle: upstream [145.9,6.6,-82.7] tilts ~146°",
        rotation_euler_deg=[0, 0, 90],
    ),
    # --- scene12 ---
    ("scene12", 0): SceneCorrection(
        reason="toy corn cob_target: upstream [77.2,-3.8,-66.4] tilts ~77°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene12", 1): SceneCorrection(
        reason="spoon: upstream [93.0,4.8,-79.4] tilts ~93°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene12", 2): SceneCorrection(
        reason="small metal pot: upstream [100.8,5.0,-105.2] tilts ~101°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene12", 3): SceneCorrection(
        reason="metal pot lid_destination: upstream [78.8,-11.0,67.0] tilts ~79°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene14 ---
    ("scene14", 0): SceneCorrection(
        reason="plastic tomato: upstream [75.3,-1.4,12.4] tilts ~75°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene14", 1): SceneCorrection(
        reason="white cloth: upstream [91.4,4.8,83.6] tilts cloth ~91° vertical",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene14", 2): SceneCorrection(
        reason="tin can: upstream [83.8,6.1,60.3] tilts ~84°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene14", 3): SceneCorrection(
        reason="tin can: upstream [85.8,-5.6,-54.2] tilts ~86°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene15 ---
    ("scene15", 0): SceneCorrection(
        reason="green block: upstream [86.9,18.9,-83.0] tilts ~87°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene15", 1): SceneCorrection(
        reason="green cube: upstream [103.0,0.1,-62.9] tilts ~103°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene15", 2): SceneCorrection(
        reason="red block: upstream [84.5,-3.1,86.5] tilts ~85°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene15", 3): SceneCorrection(
        reason="red cube: upstream [108.8,12.3,-85.6] tilts ~109°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene15", 4): SceneCorrection(
        reason="image_0004_segmented: upstream [90.7,-24.2,121.9] tilts ~91°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene16 ---
    ("scene16", 0): SceneCorrection(
        reason="wooden rectangular block with text_destination: upstream [6.6,10.1,-50.4] tilts ~7°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene16", 1): SceneCorrection(
        reason="wooden rectangular block with text_destination: upstream [96.0,4.2,-99.6] tilts ~96°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene16", 2): SceneCorrection(
        reason="wooden cylinder block: upstream [94.1,6.4,-96.0] tilts ~94°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene16", 3): SceneCorrection(
        reason="wooden L-shaped block: upstream [91.2,0.6,-33.1] tilts ~91°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene16", 4): SceneCorrection(
        reason="wooden cube block_target: upstream [29.5,21.0,-72.8] tilts ~29°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene17 ---
    ("scene17", 1): SceneCorrection(
        reason="toy corn cob: upstream [84.1,-5.1,59.4] tilts ~84°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene17", 2): SceneCorrection(
        reason="toy spatula_target: upstream [74.5,0.1,-69.6] tilts ~74°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene17", 3): SceneCorrection(
        reason="toy lemon: upstream [85.0,-36.6,-134.3] tilts ~85°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene17", 4): SceneCorrection(
        reason="toy mug: upstream [103.7,6.1,-47.9] tilts ~104°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene17", 5): SceneCorrection(
        reason="toy cutting board_destination: mesh natively vertical; [90,0,0] lays it flat. NOTE: background.png is wrong (kitchen instead of table) — upstream data issue",
        rotation_euler_deg=[90, 0, 0],
    ),
    # --- scene18 ---
    ("scene18", 0): SceneCorrection(
        reason="toy carrot: upstream [50.5,-2.6,-36.4] tilts ~51°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene18", 1): SceneCorrection(
        reason="toy potato: upstream [81.8,12.7,-28.3] tilts ~82°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene18", 2): SceneCorrection(
        reason="toy banana: upstream [15.0,2.6,-48.2] tilts ~15°",
        rotation_euler_deg=[90, 0, 0],
    ),
    ("scene18", 3): SceneCorrection(
        reason="blue cloth with black object: upstream [89.3,-4.9,-100.1] tilts ~89°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene19 ---
    ("scene19", 0): SceneCorrection(
        reason="toy corn: upstream [111.4,0.3,126.4] tilts ~111°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene19", 1): SceneCorrection(
        reason="toy broccoli: upstream [38.9,5.6,-56.2] tilts ~39°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene19", 2): SceneCorrection(
        reason="toy cutting board_destination: upstream [83.9,1.7,-62.0] tilts ~84°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene19", 3): SceneCorrection(
        reason="toy spatula_target: upstream [83.4,1.7,-17.0] tilts ~83° vertical",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene19", 4): SceneCorrection(
        reason="toy fork: upstream [85.1,-18.8,-80.4] tilts ~85°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene19", 5): SceneCorrection(
        reason="toy red fruit: upstream [78.7,-5.7,-48.1] tilts ~79°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene20 ---
    ("scene20", 0): SceneCorrection(
        reason="blue_object_target: upstream [87.5,2.7,-97.4] tilts ~87°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene20", 1): SceneCorrection(
        reason="pot_destination: upstream [91.7,0.8,-86.1] tilts ~92°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene20", 2): SceneCorrection(
        reason="image_0002_segmented: upstream [87.6,-2.0,161.3] tilts ~88°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene20", 3): SceneCorrection(
        reason="image_0003_segmented: upstream [61.3,5.9,-53.5] tilts ~61°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene21 ---
    ("scene21", 0): SceneCorrection(
        reason="can of tomatoes: upstream [97.8,-1.9,-81.0] tilts ~98°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene21", 1): SceneCorrection(
        reason="salt shaker_target: upstream [107.1,6.0,114.5] tilts ~107°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene21", 2): SceneCorrection(
        reason="yellow bell pepper: upstream [88.7,-4.6,-90.4] tilts ~89°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene21", 3): SceneCorrection(
        reason="small white cloth: upstream [-4.7,-4.7,-88.2] tilts ~5°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene22 ---
    ("scene22", 0): SceneCorrection(
        reason="Green cloth_destination: upstream [99.8,3.7,-90.4] tilts ~100°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene22", 1): SceneCorrection(
        reason="Toy donut: upstream [83.2,2.0,105.5] tilts ~83°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene22", 2): SceneCorrection(
        reason="Small metal bowl: upstream [81.0,-2.5,95.2] tilts ~81°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene22", 3): SceneCorrection(
        reason="Toy spoon_target: mesh natively vertical; [90,0,0] lays it flat",
        rotation_euler_deg=[90, 0, 0],
    ),
    ("scene22", 4): SceneCorrection(
        reason="Tomato sauce can: upstream [74.5,-1.9,97.2] tilts ~75°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene22", 5): SceneCorrection(
        reason="Pineapple slices can: upstream [107.5,2.8,-86.0] tilts ~108°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene23 ---
    ("scene23", 0): SceneCorrection(
        reason="metal strainer with lid: upstream [89.9,-16.5,-62.2] tilts ~90°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene23", 1): SceneCorrection(
        reason="tomato sauce can: upstream [94.0,-0.1,-70.8] tilts ~94°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene23", 2): SceneCorrection(
        reason="pineapple slices can: upstream [70.9,0.9,59.4] tilts ~71°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene23", 3): SceneCorrection(
        reason="green toy pickle: upstream [125.2,-0.4,-85.8] tilts ~125°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene23", 4): SceneCorrection(
        reason="toy ice cream tub_target: upstream [106.2,-4.3,-78.7] tilts ~106°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene24 ---
    ("scene24", 0): SceneCorrection(
        reason="red handled spoon: upstream [79.6,6.7,-98.8] tilts ~80°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene24", 1): SceneCorrection(
        reason="orange plastic bowl_target: upstream [40.2,1.2,-74.5] tilts ~40°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene24", 2): SceneCorrection(
        reason="yellow toy banana_destination: upstream [93.3,2.5,-36.6] tilts ~93°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene24", 3): SceneCorrection(
        reason="purple cloth: upstream [93.7,0.3,-88.8] tilts ~94°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene25 ---
    ("scene25", 0): SceneCorrection(
        reason="bottle: upstream [78.5,12.1,2.1] tilts ~78°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene25", 1): SceneCorrection(
        reason="salt shaker: upstream [117.3,4.8,126.3] tilts ~117°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene25", 2): SceneCorrection(
        reason="cloth: upstream [85.5,-0.1,-46.5] tilts ~86°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene25", 3): SceneCorrection(
        reason="cube_target: upstream [4.1,-16.6,-67.2] tilts ~4°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene27 ---
    ("scene27", 0): SceneCorrection(
        reason="small metal saucepan: upstream [62.6,22.7,-33.2] tilts ~63°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene27", 1): SceneCorrection(
        reason="spoon with blue handle: upstream [55.1,5.2,-87.4] tilts ~55°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene27", 2): SceneCorrection(
        reason="toy orange half: upstream [2.5,-1.8,-56.3] tilts ~2°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene27", 3): SceneCorrection(
        reason="green cloth: upstream [88.2,1.0,-92.1] tilts ~88°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene28 ---
    ("scene28", 0): SceneCorrection(
        reason="red cone: upstream [55.7,-15.1,-74.0] tilts ~56°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene28", 1): SceneCorrection(
        reason="red pawn_destination: upstream [74.1,-2.5,-54.8] tilts ~74°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene28", 2): SceneCorrection(
        reason="red disk: upstream [89.7,-3.5,-69.6] tilts ~90°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene28", 3): SceneCorrection(
        reason="red cylinder_target: upstream [97.8,1.6,-88.0] tilts ~98°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene28", 4): SceneCorrection(
        reason="red layered cone: upstream [81.2,-20.8,-102.3] tilts ~81°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene28", 5): SceneCorrection(
        reason="red pawn_destination: upstream [87.2,3.3,-36.6] tilts ~87°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene29 ---
    ("scene29", 0): SceneCorrection(
        reason="wooden game piece: upstream [92.4,1.6,101.3] tilts ~92°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene29", 1): SceneCorrection(
        reason="wooden block_target: upstream [99.6,-9.1,-29.8] tilts ~100°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene29", 2): SceneCorrection(
        reason="wooden block_target: upstream [99.9,7.7,-103.3] tilts ~100°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene29", 3): SceneCorrection(
        reason="wooden egg cup with ball: upstream [91.5,9.5,-164.7] tilts ~92°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene29", 4): SceneCorrection(
        reason="stack of wooden blocks: upstream [5.7,-17.3,-96.3] tilts ~6°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene30 ---
    ("scene30", 0): SceneCorrection(
        reason="can of peas and carrots: upstream [95.8,2.9,-98.4] tilts ~96°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene30", 1): SceneCorrection(
        reason="pink spoon_target: upstream [100.5,-6.0,-25.8] tilts ~101°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene30", 2): SceneCorrection(
        reason="orange cloth_destination: upstream [40.3,12.8,-62.6] tilts ~40°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene30", 3): SceneCorrection(
        reason="small metal pan: upstream [102.5,-4.4,-2.0] tilts ~102°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene31 ---
    ("scene31", 0): SceneCorrection(
        reason="plastic cup_target (the 'orange-white object'): upstream [84.5,15.7,-47.2] "
        "tilts ~84°. Re-authored 2026-05-27: cup started on the right side (= the goal, "
        "user-confirmed degenerate); shifted to the left so the move is real. Verified "
        "settled + out-of-region.",
        rotation_euler_deg=[90, 0, 0],
        translation=[0.272, 0.117, 0.02],
    ),
    ("scene31", 1): SceneCorrection(
        reason="toy knife: upstream [96.0,-1.0,-73.4] tilts ~96°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene31", 2): SceneCorrection(
        reason="toy food: upstream [31.8,13.3,-54.8] tilts ~32°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene31", 3): SceneCorrection(
        reason="blue cloth: upstream [81.8,-7.4,73.2] tilts ~82°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene32 ---
    ("scene32", 0): SceneCorrection(
        reason="green toy cucumber: upstream [53.0,-1.8,-5.0] tilts ~53°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene32", 1): SceneCorrection(
        reason="small metal pot_target: upstream [89.0,3.3,-39.4] tilts ~89°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene32", 2): SceneCorrection(
        reason="purple cloth_destination: upstream [90.2,3.7,-91.8] tilts ~90°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene32", 3): SceneCorrection(
        reason="toy spatula: upstream [98.6,3.3,-91.8] tilts ~99°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene33 ---
    ("scene33", 0): SceneCorrection(
        reason="toy knife: upstream [93.4,1.1,-172.8] tilts ~93°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene33", 1): SceneCorrection(
        reason="purple towel: upstream [-2.4,19.1,-38.4] tilts ~2°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene33", 2): SceneCorrection(
        reason="cheese wedge: upstream [85.3,-2.8,-89.3] tilts ~85°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene33", 3): SceneCorrection(
        reason="can: upstream [78.4,-3.1,-65.8] tilts ~78°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene34 ---
    ("scene34", 0): SceneCorrection(
        reason="toy potato: upstream [21.0,-3.7,-50.4] tilts ~21°. Re-authored 2026-05-27: "
        "target started in the 'lower'/bottom-center region the task moves it to "
        "(degenerate); shifted toward upper so the task is doable.",
        rotation_euler_deg=[0, 0, 0],
        translation=[0.446, 0.254, 0.02],
    ),
    ("scene34", 1): SceneCorrection(
        reason="toy can of mushrooms: upstream [86.6,-22.8,50.2] tilts ~87°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene34", 2): SceneCorrection(
        reason="toy salt shaker: upstream [94.2,-2.8,87.6] tilts ~94°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene34", 3): SceneCorrection(
        reason="toy sushi: upstream [61.5,9.2,-35.6] tilts ~62°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene34", 4): SceneCorrection(
        reason="toy can of tomato sauce: upstream [93.0,-6.5,-80.5] tilts ~93°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene34", 5): SceneCorrection(
        reason="toy can of pineapple slices: upstream [93.8,-4.8,-77.3] tilts ~94°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene34", 6): SceneCorrection(
        reason="toy microwave: upstream [89.2,-2.5,-93.2] tilts ~89°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene35 ---
    ("scene35", 0): SceneCorrection(
        reason="cup_destination: upstream [83.3,0.8,-22.7] tilts ~83°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene35", 1): SceneCorrection(
        reason="toy carrot_target: upstream [49.6,-5.2,-56.8] tilts ~50°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene35", 2): SceneCorrection(
        reason="towel: upstream [90.2,-1.7,-86.1] tilts ~90°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene35", 3): SceneCorrection(
        reason="can of cherries: upstream [98.1,-4.4,-90.7] tilts ~98°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene36 ---
    ("scene36", 0): SceneCorrection(
        reason="toy corn cob: upstream [91.7,7.4,-40.7] tilts ~92°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene36", 1): SceneCorrection(
        reason="toy fork_target: upstream [77.2,-6.6,-91.5] tilts ~77°",
        rotation_euler_deg=[0, 0, 90],
    ),
    ("scene36", 2): SceneCorrection(
        reason="toy pan: upstream [84.4,-0.5,-38.5] tilts ~84°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene37 ---
    ("scene37", 0): SceneCorrection(
        reason="metal pot: upstream [86.4,-1.6,-47.5] tilts ~86°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene37", 1): SceneCorrection(
        reason="yellow toy knife_target: upstream [99.6,41.5,-86.3] tilts ~100°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene37", 2): SceneCorrection(
        reason="stuffed monkey toy: upstream [23.1,12.8,-69.8] tilts ~23°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene37", 3): SceneCorrection(
        reason="metal pot lid: mesh natively vertical; [90,0,0] lays it flat",
        rotation_euler_deg=[90, 0, 0],
    ),
    # --- scene38 ---
    ("scene38", 0): SceneCorrection(
        reason="Yellow object: upstream [88.6,2.1,89.1] tilts ~89°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene38", 1): SceneCorrection(
        reason="Green cloth: upstream [85.0,-4.5,-132.5] tilts ~85°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene38", 2): SceneCorrection(
        reason="Blue brush_target: upstream [21.0,2.9,-61.4] tilts ~21°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene39 ---
    ("scene39", 0): SceneCorrection(
        reason="yellow brush: upstream [89.7,6.8,-63.5] tilts ~90°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene39", 1): SceneCorrection(
        reason="toy sushi_target: upstream [72.5,-3.1,-87.8] tilts ~72°",
        rotation_euler_deg=[90, 0, 0],
    ),
    ("scene39", 2): SceneCorrection(
        reason="metal pot_destination: upstream [87.6,-12.8,-121.3] tilts ~88°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene39", 3): SceneCorrection(
        reason="blue cloth: upstream [97.2,-11.3,163.9] tilts ~97°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene40 ---
    ("scene40", 0): SceneCorrection(
        reason="metal pot: upstream [97.2,3.0,-43.4] tilts ~97°",
        rotation_euler_deg=[0, 0, 90],
    ),
    ("scene40", 1): SceneCorrection(
        reason="red toy tomato: upstream [48.9,-15.3,-72.7] tilts ~49°",
        rotation_euler_deg=[0, 0, 90],
    ),
    # --- scene41 ---
    ("scene41", 0): SceneCorrection(
        reason="metal pot_target: upstream [75.7,5.0,-43.7] tilts ~76°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene41", 1): SceneCorrection(
        reason="red spoon: upstream [82.1,14.2,-63.9] tilts ~82°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene41", 2): SceneCorrection(
        reason="green cloth_destination: upstream [78.9,-5.2,-75.9] tilts ~79°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene41", 3): SceneCorrection(
        reason="toy chicken wing: upstream [5.0,-15.4,-69.3] tilts ~5°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene42 ---
    ("scene42", 0): SceneCorrection(
        reason="red block: upstream [83.6,2.4,93.7] tilts ~84°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene42", 1): SceneCorrection(
        reason="yellow block: upstream [87.5,-9.9,-89.2] tilts ~87°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene42", 2): SceneCorrection(
        reason="blue block: upstream [94.8,9.2,-107.3] tilts ~95°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene42", 3): SceneCorrection(
        reason="green block: upstream [70.3,6.2,-2.2] tilts ~70°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene43 ---
    ("scene43", 0): SceneCorrection(
        reason="metal bowl: upstream [96.0,4.8,-113.3] tilts ~96°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene43", 1): SceneCorrection(
        reason="green cloth_destination: upstream [97.5,3.1,-93.9] tilts ~98°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene43", 2): SceneCorrection(
        reason="toy croissant: upstream [101.1,-8.7,-72.1] tilts ~101°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene43", 3): SceneCorrection(
        reason="toy knife_target: upstream [98.5,3.7,-96.3] tilts ~99°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene44 ---
    ("scene44", 0): SceneCorrection(
        reason="toy knife: upstream [117.2,22.8,-54.3] tilts ~117°",
        rotation_euler_deg=[0, 90, 0],
    ),
    ("scene44", 1): SceneCorrection(
        reason="red capped bottle: upstream [92.4,-4.7,-36.0] tilts ~92°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene44", 2): SceneCorrection(
        reason="toy strawberry: upstream [108.8,-5.8,-66.6] tilts ~109°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene45 ---
    ("scene45", 0): SceneCorrection(
        reason="blue bowl_target: upstream [109.2,5.1,113.3] tilts ~109°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene45", 1): SceneCorrection(
        reason="purple cloth: upstream [88.7,-0.7,-31.6] tilts ~89°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene45", 2): SceneCorrection(
        reason="sushi toy: upstream [104.2,34.1,-104.9] tilts ~104°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene45", 3): SceneCorrection(
        reason="small can: upstream [94.8,-6.0,-76.5] tilts ~95°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene46 ---
    ("scene46", 0): SceneCorrection(
        reason="white sushi roll toy: upstream [77.4,-1.9,-65.6] tilts ~77°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene46", 1): SceneCorrection(
        reason="green broccoli toy_target: upstream [66.0,-30.2,-115.2] tilts ~66°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene46", 2): SceneCorrection(
        reason="orange salmon sushi toy: upstream [90.4,-33.9,-140.9] tilts ~90°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene46", 3): SceneCorrection(
        reason="yellow duck plush toy: upstream [66.6,14.6,-17.2] tilts ~67°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene47 ---
    ("scene47", 0): SceneCorrection(
        reason="Toy sushi: upstream [65.1,0.6,-56.2] tilts ~65°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene47", 1): SceneCorrection(
        reason="Toy frying pan_target: upstream [81.8,3.1,-97.4] tilts ~82°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene48 ---
    ("scene48", 0): SceneCorrection(
        reason="metal bowl: upstream [89.3,2.3,126.5] tilts ~89°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene48", 1): SceneCorrection(
        reason="purple towel_destination: upstream [86.7,2.4,-91.7] tilts ~87°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene48", 2): SceneCorrection(
        reason="yellow brush_target: upstream [89.9,8.3,-91.7] tilts ~90°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene48", 3): SceneCorrection(
        reason="orange toy food: upstream [97.7,-0.3,-90.2] tilts ~98°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene49 ---
    ("scene49", 0): SceneCorrection(
        reason="yellow toy cheese wedge_target: upstream [67.4,-22.3,-105.5] tilts ~67°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene49", 1): SceneCorrection(
        reason="toy metal pot_destination: upstream [62.0,19.3,82.5] tilts ~62°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene49", 2): SceneCorrection(
        reason="stovetop knob: upstream [87.1,-4.3,-77.5] tilts ~87°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene49", 3): SceneCorrection(
        reason="stovetop knob: upstream [82.9,-7.7,-65.3] tilts ~83°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene49", 4): SceneCorrection(
        reason="stovetop knob: upstream [92.5,-3.9,98.5] tilts ~92°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene50 ---
    ("scene50", 0): SceneCorrection(
        reason="toy hammer: upstream [25.5,-39.9,-89.0] tilts ~25°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene50", 1): SceneCorrection(
        reason="blue block: upstream [96.3,-7.2,-94.9] tilts ~96°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene50", 2): SceneCorrection(
        reason="metal lid: upstream [78.9,-4.1,73.5] tilts ~79°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene50", 3): SceneCorrection(
        reason="metal pot_destination: upstream [88.3,-3.2,-33.4] tilts ~88°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene50", 4): SceneCorrection(
        reason="stuffed animal_target: upstream [99.0,-1.8,-84.0] tilts ~99°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene53 ---
    ("scene53", 0): SceneCorrection(
        reason="milk carton_target: upstream [65.0,18.1,-4.3] tilts ~65°. Re-authored "
        "2026-05-27: target started at the 'back'/upper region the push task moves it "
        "to (degenerate); shifted to the front so it must be pushed back.",
        rotation_euler_deg=[0, 0, 0],
        translation=[0.236, 0.178, 0.02],
    ),
    ("scene53", 1): SceneCorrection(
        reason="can of beans: upstream [100.0,-6.4,-10.0] tilts ~100°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene53", 2): SceneCorrection(
        reason="yellow cloth: upstream [90.7,-0.8,-88.0] tilts ~91°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene53", 3): SceneCorrection(
        reason="potato: upstream [32.2,3.2,-59.7] tilts ~32°",
        rotation_euler_deg=[90, 0, 0],
    ),
    # --- scene54 ---
    ("scene54", 0): SceneCorrection(
        reason=(
            "spoon_target: mesh natively vertical (same as scene22 spoon); "
            "[90,0,0] lays it flat. Identity caused NaN during pre-grasped "
            "release subtask — vertical spoon welded to horizontal EE creates "
            "ill-conditioned moment coupling that blows up on first step."
        ),
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene54", 1): SceneCorrection(
        reason="pot: upstream [88.0,-2.1,90.3] tilts ~88°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene54", 2): SceneCorrection(
        reason="cloth_destination: upstream [94.9,-1.9,-82.5] tilts ~95°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene54", 3): SceneCorrection(
        reason="toy chicken drumstick: upstream [87.6,9.9,-35.8] tilts ~88°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene55 ---
    ("scene55", 0): SceneCorrection(
        reason="yellow bell pepper toy: upstream [87.2,-9.9,68.3] tilts ~87°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene55", 1): SceneCorrection(
        reason="green spatula toy_target: upstream [96.1,15.7,-86.3] tilts ~96°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene55", 2): SceneCorrection(
        reason="small metal pan toy: upstream [107.6,-8.9,-45.4] tilts ~108°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene55", 3): SceneCorrection(
        reason="green cloth: upstream [97.8,4.9,-96.4] tilts ~98°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene57 ---
    ("scene57", 0): SceneCorrection(
        reason="green cucumber toy: upstream [56.4,-37.2,-97.5] tilts ~56°",
        rotation_euler_deg=[90, 0, 0],
    ),
    ("scene57", 1): SceneCorrection(
        reason="yellow sponge: upstream [104.3,7.3,-82.4] tilts ~104°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene57", 2): SceneCorrection(
        reason="blue fork: upstream [99.8,-5.2,-31.5] tilts ~100°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene57", 3): SceneCorrection(
        reason="metal pot: upstream [102.1,3.8,-84.3] tilts ~102°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene60 ---
    ("scene60", 0): SceneCorrection(
        reason="toy mushroom_target: upstream [55.4,-12.1,-68.7] tilts ~55°",
        rotation_euler_deg=[0, 0, 90],
    ),
    ("scene60", 1): SceneCorrection(
        reason="toy pot_destination: upstream [75.2,-9.9,90.1] tilts ~75°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene70 ---
    ("scene70", 0): SceneCorrection(
        reason="orange plastic cup: upstream [90.3,3.6,-38.8] tilts ~90°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene70", 1): SceneCorrection(
        reason="red strawberry: upstream [48.0,1.3,-45.6] tilts ~48°",
        rotation_euler_deg=[90, 0, 0],
    ),
    ("scene70", 2): SceneCorrection(
        reason="metal colander_destination: upstream [93.7,-4.8,-96.0] tilts ~94°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene70", 3): SceneCorrection(
        reason="yellow cheese wedge: upstream [72.5,-4.8,-94.3] tilts ~73°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene70", 4): SceneCorrection(
        reason="yellow banana_target: upstream [31.2,8.3,-39.3] tilts ~31°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene70", 5): SceneCorrection(
        reason="red sausage: upstream [78.9,-12.1,-57.6] tilts ~79°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene71 ---
    ("scene71", 0): SceneCorrection(
        reason="spice jar: upstream [104.0,10.6,-73.7] tilts ~104°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene71", 1): SceneCorrection(
        reason="toy cabbage: upstream [85.3,-9.9,67.1] tilts ~85°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene71", 2): SceneCorrection(
        reason="toy chicken leg: upstream [102.2,16.4,-122.2] tilts ~102°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene71", 3): SceneCorrection(
        reason="plastic cup_destination: upstream [100.4,-8.9,-40.0] tilts ~100°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene71", 4): SceneCorrection(
        reason="toy cucumber_target: upstream [6.3,-7.2,-44.9] tilts ~6°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene72 ---
    ("scene72", 0): SceneCorrection(
        reason="utility knife: upstream [99.9,-6.9,-29.0] tilts ~100°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene72", 1): SceneCorrection(
        reason="radish half (toy)_target: upstream [64.9,3.4,-53.0] tilts ~65°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene72", 2): SceneCorrection(
        reason="pear (toy): upstream [25.0,10.0,-36.9] tilts ~25°",
        rotation_euler_deg=[0, 90, 0],
    ),
    ("scene72", 3): SceneCorrection(
        reason="plate (toy)_destination: upstream [80.8,9.2,176.9] tilts ~81°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene72", 4): SceneCorrection(
        reason="strawberry (toy): upstream [69.3,-29.6,-114.3] tilts ~69°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene72", 5): SceneCorrection(
        reason="carrot (toy): upstream [24.9,19.1,-46.6] tilts ~25°",
        rotation_euler_deg=[90, 0, 0],
    ),
    # --- scene75 ---
    ("scene75", 0): SceneCorrection(
        reason="cheese block_target: upstream [95.6,-6.9,-10.4] tilts ~96°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene75", 1): SceneCorrection(
        reason="green cloth: upstream [99.3,10.0,-79.3] tilts ~99°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene75", 2): SceneCorrection(
        reason="yellow brush: upstream [94.6,4.3,-82.8] tilts ~95°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene75", 3): SceneCorrection(
        reason="metal pot_destination: upstream [83.8,18.8,179.8] tilts ~84°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene76 ---
    ("scene76", 0): SceneCorrection(
        reason="yellow cloth: upstream [93.7,-0.5,-29.9] tilts ~94°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene76", 1): SceneCorrection(
        reason="metal pot_target: upstream [88.5,-7.3,-40.2] tilts ~88°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene76", 2): SceneCorrection(
        reason="mushroom toy: upstream [36.2,-0.2,-58.9] tilts ~36°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene77 ---
    ("scene77", 0): SceneCorrection(
        reason="folded blue towel_destination: upstream [89.0,-1.8,-84.5] tilts ~89°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene77", 1): SceneCorrection(
        reason="toy mushroom: upstream [80.4,-27.9,-89.2] tilts ~80°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene77", 2): SceneCorrection(
        reason="toy pot_target: upstream [83.8,-3.2,-74.1] tilts ~84°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene77", 3): SceneCorrection(
        reason="toy corn with spoon: upstream [98.9,-6.2,-92.2] tilts ~99°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene77", 4): SceneCorrection(
        reason="can of tomato sauce: upstream [83.7,-4.7,95.9] tilts ~84°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene77", 5): SceneCorrection(
        reason="can of pineapple slices: upstream [73.2,-9.7,50.6] tilts ~73°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene79 ---
    ("scene79", 0): SceneCorrection(
        reason="small metal pot: upstream [93.3,-9.3,-75.6] tilts ~93°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene79", 1): SceneCorrection(
        reason="folded green cloth: upstream [95.5,2.6,160.7] tilts ~95°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene79", 2): SceneCorrection(
        reason="yellow toy pepper: upstream [82.4,0.1,-37.3] tilts ~82°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene79", 3): SceneCorrection(
        reason="can of tomato sauce: upstream [92.8,-3.3,-59.4] tilts ~93°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene79", 4): SceneCorrection(
        reason="can of pineapple slices: upstream [75.4,-7.2,45.7] tilts ~75°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene79", 5): SceneCorrection(
        reason="toy knife_target: mesh natively vertical; [90,0,0] lays it flat",
        rotation_euler_deg=[90, 0, 0],
    ),
    # --- scene80 ---
    ("scene80", 0): SceneCorrection(
        reason="yellow cloth: upstream [86.8,8.6,90.5] tilts ~87°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene80", 1): SceneCorrection(
        reason="metal pot_destination: upstream [94.9,-2.3,-40.0] tilts ~95°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene80", 2): SceneCorrection(
        reason="patterned container with lid_target: upstream [93.1,-2.6,-68.1] tilts ~93°. "
        "Re-authored 2026-05-27: container started already to the right of the metal pot "
        "(= the goal, user-confirmed degenerate); shifted to the left of the pot so the "
        "task requires moving it right. Verified settled + out-of-region.",
        rotation_euler_deg=[0, 0, 0],
        translation=[0.193, 0.194, 0.02],
    ),
    ("scene80", 3): SceneCorrection(
        reason="blue fork: mesh natively vertical; [90,0,-92] lays flat + restores upstream Z orientation to avoid overlap with spoon",
        rotation_euler_deg=[90, 0, -92],
    ),
    ("scene80", 4): SceneCorrection(
        reason="red spoon: mesh natively vertical; [90,0,-88] lays flat + restores upstream Z orientation to avoid overlap with fork",
        rotation_euler_deg=[90, 0, -88],
    ),
    # --- scene81 ---
    ("scene81", 0): SceneCorrection(
        reason="Can of food_target: upstream [94.5,-9.8,-2.2] tilts ~95°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene81", 1): SceneCorrection(
        reason="Small pot_destination: upstream [86.3,7.2,134.8] tilts ~86°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene81", 2): SceneCorrection(
        reason="Blue towel: upstream [96.8,0.8,-99.8] tilts ~97°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene81", 3): SceneCorrection(
        reason="Piece of fruit/vegetable: upstream [28.2,-28.4,-88.5] tilts ~28°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene81", 4): SceneCorrection(
        reason="Small knife: upstream [62.4,-2.7,-80.4] tilts ~62°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene82 ---
    ("scene82", 0): SceneCorrection(
        reason="red plastic pot: upstream [83.3,-2.8,-91.8] tilts ~83°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene82", 1): SceneCorrection(
        reason="blue microfiber cloth: upstream [92.3,-5.5,-81.5] tilts ~92°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene82", 2): SceneCorrection(
        reason="toy food can with blue and green label: upstream [93.5,-9.8,-49.2] tilts ~94°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene82", 3): SceneCorrection(
        reason="toy food can with red and yellow label_target: upstream [95.0,-8.2,-43.0] tilts ~95°. "
        "Re-authored 2026-05-27: target started in the 'lower'/bottom region the task moves it "
        "to (degenerate, 72% false-success); shifted toward upper so the task is doable.",
        rotation_euler_deg=[0, 0, 0],
        translation=[0.443, 0.012, 0.02],
    ),
    ("scene82", 4): SceneCorrection(
        reason="sushi toy: upstream [97.8,26.8,-79.4] tilts ~98°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene84 ---
    ("scene84", 0): SceneCorrection(
        reason="metal bowl_destination: upstream [98.6,-4.0,-33.9] tilts ~99°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene84", 1): SceneCorrection(
        reason="plastic scoop: upstream [87.8,-1.9,-39.8] tilts ~88°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene84", 2): SceneCorrection(
        reason="blue cloth_target: upstream [100.1,-1.4,-59.3] tilts ~100°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene84", 3): SceneCorrection(
        reason="sushi toy: upstream [82.6,-8.4,-90.9] tilts ~83°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene85 ---
    ("scene85", 0): SceneCorrection(
        reason="folded cloth_destination: upstream [93.4,8.4,-81.9] tilts ~93°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene85", 1): SceneCorrection(
        reason="metal pan: upstream [92.4,4.9,-106.5] tilts ~92°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene85", 2): SceneCorrection(
        reason="toy vegetable: upstream [95.8,-9.9,-7.4] tilts ~96°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene85", 3): SceneCorrection(
        reason="fork_target: upstream [89.6,5.2,-94.0] tilts ~90°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene90 ---
    ("scene90", 0): SceneCorrection(
        reason="toy tomato: upstream [65.3,1.1,-41.4] tilts ~65°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene90", 1): SceneCorrection(
        reason="toy potato: upstream [65.6,-21.5,-98.9] tilts ~66°",
        rotation_euler_deg=[0, 90, 0],
    ),
    ("scene90", 2): SceneCorrection(
        reason="toy pot_destination: upstream [100.4,-3.8,-65.5] tilts ~100°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene90", 3): SceneCorrection(
        reason="toy pot lid_target: upstream [81.3,4.3,104.8] tilts ~81°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene90", 4): SceneCorrection(
        reason="table: upstream [59.4,9.5,-141.2] tilts ~59°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene93 ---
    ("scene93", 0): SceneCorrection(
        reason="dark red wooden bowling pin: upstream [60.5,0.4,-39.7] tilts ~61°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene93", 1): SceneCorrection(
        reason="dark red wooden mushroom shape: upstream [80.2,2.3,-41.3] tilts ~80°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene93", 2): SceneCorrection(
        reason="light brown wooden cylindrical block with groove: upstream [82.4,0.3,-54.6] tilts ~82°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene93", 3): SceneCorrection(
        reason="light brown wooden cube with dowel: upstream [52.9,-29.1,-88.3] tilts ~53°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene93", 4): SceneCorrection(
        reason="light brown wooden rectangular block_target: upstream [77.7,-5.8,-62.9] tilts ~78°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene93", 5): SceneCorrection(
        reason="light brown wooden stick: upstream [88.0,10.7,86.5] tilts ~88°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene93", 6): SceneCorrection(
        reason="light brown wooden stick: upstream [113.4,-0.2,150.6] tilts ~113°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene94 ---
    ("scene94", 0): SceneCorrection(
        reason="toy milk carton_target: upstream [78.3,5.3,-24.8] tilts ~78°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene94", 1): SceneCorrection(
        reason="blue cloth: upstream [96.2,3.2,91.1] tilts ~96°; identity restores scan orientation",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene94", 2): SceneCorrection(
        reason="green toy vegetable: upstream [27.8,-7.2,-76.3] tilts ~28°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene97 ---
    ("scene97", 0): SceneCorrection(
        reason="green toy cutting board_destination: upstream [96.9,0.1,-89.5] tilts ~97°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene97", 1): SceneCorrection(
        reason="toy knife_target: mesh natively vertical; [90,0,-77] lays flat + restores upstream Z orientation to avoid overlap with cutting board",
        rotation_euler_deg=[90, 0, -77],
    ),
    ("scene97", 2): SceneCorrection(
        reason="red toy tomato: upstream [59.8,-3.1,-54.5] tilts ~60°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene97", 3): SceneCorrection(
        reason="red toy pepper: upstream [40.0,12.3,-43.2] tilts ~40°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene97", 4): SceneCorrection(
        reason="green toy sphere: upstream [-168.3,-2.1,114.7] tilts ~168°",
        rotation_euler_deg=[0, 0, 0],
    ),
    # --- scene98 ---
    ("scene98", 0): SceneCorrection(
        reason="blue bowl_target: upstream [82.1,1.5,121.2] tilts ~82°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene98", 1): SceneCorrection(
        reason="blue plate_destination: upstream [86.5,-0.6,77.2] tilts ~87°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene98", 2): SceneCorrection(
        # green blob is a 1mm-thin flat sheet (mesh_ext ~[4.8,4.0,0.1]cm). The old
        # [90,0,0] stood it vertical -> the contact solver blew up to NaN constraint
        # forces on the first settle step (scene98 crashed at prebuild for every model).
        # It is a distractor (task is blue bowl -> blue plate), so lay it flat (thin
        # axis vertical = stable patch); pose does not affect task success.
        reason="green blob: 1mm-thin sheet; lay flat to avoid NaN constraint forces (was [90,0,0] vertical)",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene98", 3): SceneCorrection(
        reason="yellow banana toy: upstream [93.0,-2.2,-92.6] tilts ~93°",
        rotation_euler_deg=[0, 0, 0],
    ),
    ("scene5", 0): SceneCorrection(reason="pose-search candidate", rotation_euler_deg=[90, 0, 0]),
}


# ---------------------------------------------------------------------------
# Scene-level corrections: background image overrides
# ---------------------------------------------------------------------------
# Some scenes have incorrect background images in the upstream data (e.g. the
# background.png shows a completely different workspace than the actual scene).
# This dict maps scene_name -> (replacement_scene, reason) where the background
# from replacement_scene is used instead.
#
# The replacement is applied in scene_loader.load_scene() after the default
# background path is resolved.


# ---------------------------------------------------------------------------
# Orientation validation
# ---------------------------------------------------------------------------

_FLAT_OBJECT_KEYWORDS = ("cloth", "towel")
_FLAT_OBJECT_MAX_TILT_DEG = 45.0


@dataclasses.dataclass(frozen=True)
class SceneValidationWarning:
    """A single validation issue found in a loaded scene."""

    scene_name: str
    object_index: int
    object_name: str
    message: str
    tilt_angle_deg: float


def validate_object_orientations(
    scene_name: str,
    object_names: List[str],
    object_rotation_quats_wxyz: List[List[float]],
) -> List[SceneValidationWarning]:
    """Check scene objects for physically implausible orientations.

    Inspects objects whose names match _FLAT_OBJECT_KEYWORDS and verifies that
    their rotation keeps the mesh approximately horizontal (local Z near world Z).
    """
    warnings: List[SceneValidationWarning] = []

    for i, (name, quat_wxyz) in enumerate(zip(object_names, object_rotation_quats_wxyz, strict=True)):
        name_lower = name.lower()
        if not any(kw in name_lower for kw in _FLAT_OBJECT_KEYWORDS):
            continue

        quat_wxyz_arr = np.array(quat_wxyz, dtype=np.float64)
        quat_xyzw = np.array([quat_wxyz_arr[1], quat_wxyz_arr[2], quat_wxyz_arr[3], quat_wxyz_arr[0]])
        rotation_matrix = scipy.spatial.transform.Rotation.from_quat(quat_xyzw).as_matrix()

        local_z_in_world = rotation_matrix[:, 2]
        cos_angle = np.clip(np.dot(local_z_in_world, np.array([0.0, 0.0, 1.0])), -1.0, 1.0)
        tilt_angle_deg = float(np.degrees(np.arccos(np.abs(cos_angle))))

        if tilt_angle_deg > _FLAT_OBJECT_MAX_TILT_DEG:
            warnings.append(
                SceneValidationWarning(
                    scene_name=scene_name,
                    object_index=i,
                    object_name=name,
                    message=(
                        f"Flat object '{name}' (index {i}) has its Z-axis tilted "
                        f"{tilt_angle_deg:.1f}\u00b0 from vertical (threshold: "
                        f"{_FLAT_OBJECT_MAX_TILT_DEG}\u00b0). This likely means the "
                        f"object is standing upright instead of lying flat."
                    ),
                    tilt_angle_deg=tilt_angle_deg,
                )
            )

    return warnings


# ---------------------------------------------------------------------------
# Scene-level corrections: background image overrides
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BackgroundOverride:
    """Override a scene's background image with one from another scene."""

    source_scene: str
    reason: str


BACKGROUND_OVERRIDES: Dict[str, BackgroundOverride] = {
    "scene17": BackgroundOverride(
        source_scene="scene19",
        reason=(
            "Upstream background.png shows a kitchen/sink workspace, but the actual "
            "scene17 ground truth (images0/) shows objects on a brown wooden table — "
            "the same table used by scene19. The camera extrinsics differ slightly "
            "between scene17 and scene19 so the compositing won't be pixel-perfect, "
            "but it is far closer than the wrong kitchen background."
        ),
    ),
}
