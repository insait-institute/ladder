import inspect
import math
from typing import Callable, Dict, Optional

import numpy as np

from ladder.components.inference.models.harness.arm import Arm
from ladder.components.inference.models.harness.perception import Detection, ObjectNotFoundError, Scene
from ladder.components.inference.models.harness.tools import code as run_code_tool, grasping, placing, pushing
from ladder.components.inference.models.harness.tools.geometry import grasp_height
from ladder.components.inference.models.harness.tools.hands import Hands
from ladder.components.inference.models.harness.tools.report import Report

_GRIP_MARGIN = 0.85


# The docstring and signature of every method named in `by_name` are pasted into the prompt
# verbatim by `catalog`. EDITING ONE OF THEM CHANGES WHAT THE MODEL IS TOLD.
class Tools:
    def __init__(self, arm: Arm, scene: Scene) -> None:
        self.hands = Hands(arm, scene)

    @property
    def arm(self) -> Arm:
        return self.hands.arm

    @property
    def scene(self) -> Scene:
        return self.hands.scene

    @property
    def held(self) -> Optional[Detection]:
        return self.hands.held

    def by_name(self) -> Dict[str, Callable[..., Report]]:
        return {
            name: getattr(self, name)
            for name in (
                "pick_and_place",
                "pick",
                "place",
                "push",
                "measure",
                "grasp_at",
                "run_code",
            )
        }

    def catalog(self) -> str:
        lines = []
        for name, tool in self.by_name().items():
            # Drop the return annotation: rendering it would write this package's import path into
            # the prompt, so moving a file would silently reword what the model is told.
            signature = inspect.signature(tool).replace(return_annotation=inspect.Signature.empty)
            lines.append(f"{name}{signature}")
            lines.extend(f"    {line}" for line in (inspect.getdoc(tool) or "").splitlines())
            lines.append("")
        return "\n".join(lines).strip()

    def measure(self, object_name: str, surface: bool = False) -> Report:
        """What the cameras and depth say about one thing, so you can decide where to take hold of
        it yourself. Reports where it is, how big it is, how wide and how thick it is along its
        length section by section, and what the hand can actually close on. The jaws hold a prism,
        not a scoop, so the widest part of a thing is not always the part to take it by. Pass surface=True for something to put
        things ON, like a plate or a cloth, which is too flat to ground otherwise."""
        self.scene.refresh()
        detection = self.scene.locate(object_name, surface=surface)
        support = self.scene.support_z()
        span = self.arm.config.max_span_m
        sections = ", ".join(
            f"{w * 1000:.0f}x{h * 1000:.0f}" for w, h in detection.cross_section_m() if w > 0.0
        )
        foot = detection.foot_near(detection.center_m[:2], 0.5 * float(np.min(detection.extent_m[:2])))
        limit = f"the jaws open to {span * 1000:.0f} mm" if span is not None else "the jaw span is unknown"
        return Report(
            True,
            f"{object_name!r} is at [{detection.center_m[0]:.3f}, {detection.center_m[1]:.3f}], "
            f"{float(np.min(detection.extent_m[:2])) * 1000:.0f} mm across at its narrowest, "
            f"visible from z {foot:.3f} up to z {detection.top_m:.3f} ({(detection.top_m - foot) * 1000:.0f} "
            f"mm of it), standing {(foot - support) * 1000:.0f} mm above the table, lying at "
            f"{math.degrees(detection.yaw_rad):.0f} degrees. Below z {foot:.3f} the cameras see "
            f"whatever it rests on, not it. Along its length, section by section from the nearest "
            f"end, it measures {sections} mm (width x height). For scale, {limit}.",
        )

    def grasp_at(
        self, object_name: str, x_m: float, y_m: float, z_m: float, yaw_deg: Optional[float] = None
    ) -> Report:
        """Take hold of an object at a point YOU choose, then lift it. Use it when `pick` reports it
        cannot take something -- a pot too wide to span has a handle, a bowl has a rim, a spoon has
        a stem -- and `measure` to find where those are. `yaw_deg` turns the hand about the vertical;
        leave it out to keep the object's own angle. Everything else is as `pick`: the jaws open on
        the way, close on the point, and the object is checked and lifted."""
        self.scene.refresh()
        detection = self.scene.locate(object_name)
        yaw = detection.yaw_rad if yaw_deg is None else math.radians(yaw_deg)
        return grasping.grasp(
            self.hands, detection, np.array([x_m, y_m, z_m], dtype=np.float64), yaw, object_name
        )

    def pick(self, object_name: str) -> Report:
        """Grasp an object and lift it clear of the surface. Name it the way it looks, with its
        colour ("the yellow lemon"), so the right one is grounded when several are similar. The
        jaws close on the widest part of it they can span, since that is the part a slightly wide
        approach still catches; use `grasp_at` to take hold somewhere else."""
        self.scene.refresh()
        detection = self.scene.locate(object_name)
        support = self.scene.support_z()
        span = self.arm.config.max_span_m
        if span is None:
            grasp_xy, width = detection.center_m[:2], float(np.min(detection.extent_m[:2]))
        else:
            grasp_xy, width = detection.grip_point_m(_GRIP_MARGIN * span)
        if span is not None and width <= 0.0:
            return Report(
                False,
                f"cannot pick {object_name!r}: it measures "
                f"{float(np.min(detection.extent_m[:2])) * 1000:.0f} mm across at its narrowest and "
                f"nowhere along it fits jaws that open to {span * 1000:.0f} mm. `push` slides it "
                f"along the surface without lifting it, which is the usual answer. Otherwise use "
                f"`measure` to find a narrower part -- a handle, a stem -- and `grasp_at` that "
                f"point. If it is a container or a surface, put things ON or IN it instead.",
            )
        height = grasp_height(detection, grasp_xy, width, support)
        return grasping.grasp(
            self.hands,
            detection,
            np.array([grasp_xy[0], grasp_xy[1], height]),
            detection.yaw_rad,
            object_name,
            width,
        )

    def place(self, destination_name: str) -> Report:
        """Put the object already in the gripper onto or into a destination, named the same way.
        This needs a pick to have succeeded first; to move something still on the table, use
        pick_and_place. Releases above the destination's own top surface and on a clear part of it,
        so it works for a plate, for a cloth, and for the inside of a bin."""
        return placing.place(self.hands, destination_name)

    def push(self, object_name: str, destination_name: str) -> Report:
        """Slide something along the surface to a destination with the closed hand, without lifting
        it. This is how to move what the jaws cannot take hold of: a pot or a bowl too wide to span,
        something too heavy, something flat with nothing to grip. It cannot put one thing INTO
        another or stack it on top, and it drags whatever stands in the way, so lift when you can
        and push when you cannot."""
        return pushing.push(self.hands, object_name, destination_name)

    def pick_and_place(self, object_name: str, destination_name: str) -> Report:
        """Move one object to one destination. This is the whole of most goals; prefer it over
        calling pick and place separately."""
        picked = self.pick(object_name)
        if not picked.ok:
            return picked
        placed = self.place(destination_name)
        return Report(placed.ok, f"{picked}, then {placed}")

    def go_home(self) -> Report:
        home = self.arm.home_m
        try:
            clearance = grasping.clearance_to(self.hands, home)
        except ObjectNotFoundError:
            clearance = self.arm.position_m[2]
        here = self.arm.position_m
        if here[2] < clearance:
            self.arm.move_to(np.array([here[0], here[1], clearance]))
        self.arm.go_home()
        return Report(True, "returned home")

    def run_code(self, code: str) -> Report:
        """Run Python against this same API when a goal needs something the other tools do not
        cover. `pick`, `place`, `pick_and_place`, `measure`, `grasp_at`, `go_home`,
        `locate(name, surface=False)` (returns an object with `center_m`, `extent_m`, `top_m`,
        `yaw_rad` and `cross_section_m()`, the width and height along its length; pass surface=True
        for something to put things ON, like a plate or a cloth, which is too flat to ground
        otherwise), `move_to(position_m, yaw_rad)`, `open_gripper`, `close_gripper` and `np` are
        already defined. Every skill returns a report with `ok`, True when it finished, and `text`.
        Print what you learn."""
        return run_code_tool.run(self.hands, {**self.by_name(), "go_home": self.go_home}, code)
