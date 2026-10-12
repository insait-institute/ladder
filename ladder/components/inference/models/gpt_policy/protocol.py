import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import gpt_policy.harness.protocol
import gpt_policy.tools.catalog

_TAIL_MARKER = "Orientation retention and reachability:"
# Upstream was written for a Chinese-speaking lab and asks for the notes in Chinese. The benchmark is
# English end to end, so the notes are too.
_NOTE_LANGUAGE = ("1-2 short Chinese sentences, usually 20-60 characters", "1-2 short English sentences")
# Upstream's validation errors, which reach the model through the repair prompt.
_ERROR_TRANSLATIONS = (
    ("Agent 工具选择无效: ", ""),
    ("工具选择必须是 JSON 对象", "the tool selection must be a JSON object"),
    ("工具选择必须包含 name 字符串和 arguments 对象", "the tool selection needs a name string and an arguments object"),
    ("未启用的工具: ", "tool not enabled: "),
)

_SUPPORTED_HANDLERS = frozenset(
    {"robot.move_to", "robot.move_eef_chunk", "robot.set_gripper", "vision.locate_point"}
)


_LOCATE_PARAMETERS = {
    "type": "object",
    "properties": {
        "camera": {"$template": "camera_name"},
        "pixels_xy": {
            "type": "array",
            "items": {"$schema": "pixel"},
            "minItems": 1,
            "maxItems": 8,
            "description": "Image coordinates [x,y], origin top left, in the image you were shown.",
        },
        "note": {"type": "string"},
    },
    "required": ["camera", "pixels_xy", "note"],
    "additionalProperties": False,
}
_LOCATE_PROMPT = (
    "pixels_xy is a list of image coordinates [x,y] with the origin at the top left, in the image "
    "you were shown; the host rescales them to the depth frame. Ask for every pixel you need in ONE "
    "call: each call is a whole turn and returns no motion. Results come back in the same order, one "
    "per pixel. Use a position only when its metric_position_available is true. The point returned "
    "is the surface the pixel falls on, so a pixel on top of an object gives that object's top."
)


_SCHEMA_LEAKS = frozenset({"required", "additionalProperties", "type", "properties", "description"})


def normalize_selection(reply: str, schemas: Dict[str, Any]) -> Any:
    selection = _loads(reply)
    if not isinstance(selection, dict):
        return reply
    name = selection.get("name")
    arguments = selection.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {
            key: value
            for key, value in selection.items()
            if key not in ("name", "arguments") and value is not None
        }
        if "poses" in arguments and arguments.get("target") == _last(arguments["poses"]):
            arguments.pop("target")
        if not isinstance(name, str):
            name = _infer(arguments, schemas)
    if not isinstance(name, str) or name not in schemas:
        return reply
    return {"name": name, "arguments": _coerce(arguments, schemas[name]["parameters"])}


def _last(poses: Any) -> Any:
    return poses[-1] if isinstance(poses, list) and poses else None


def _loads(reply: str) -> Any:
    text = reply.strip()
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1] == "```":
        text = "\n".join(lines[1:-1])
    try:
        return json.loads(text)
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[start : end + 1]) if 0 <= start < end else None
    except ValueError:
        return None


def _infer(arguments: Dict[str, Any], schemas: Dict[str, Any]) -> Optional[str]:
    keys = set(arguments)
    fits = [
        name
        for name, schema in schemas.items()
        if set(schema["parameters"].get("required", ())) <= keys
        and keys <= set(schema["parameters"].get("properties", {})) | _SCHEMA_LEAKS
    ]
    return fits[0] if len(fits) == 1 else None


def _coerce(arguments: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
    properties = parameters.get("properties", {})
    coerced = {
        key: value for key, value in arguments.items() if key in properties or key not in _SCHEMA_LEAKS
    }
    pixels = coerced.get("pixels_xy")
    if isinstance(pixels, list) and len(pixels) == 2 and all(isinstance(v, (int, float)) for v in pixels):
        coerced["pixels_xy"] = [pixels]
    return coerced


def load_catalog(point_grid: Optional[int] = None) -> gpt_policy.tools.catalog.ToolCatalog:
    path = pathlib.Path(gpt_policy.tools.catalog.__file__).resolve().parents[3] / "configs" / "tools.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for tool in data["tools"]:
        tool["enabled"] = tool.get("terminal", False) or tool["handler"] in _SUPPORTED_HANDLERS
        if tool["name"] == "locate_point":
            tool["parameters"] = json.loads(_on_grid(json.dumps(_LOCATE_PARAMETERS), point_grid))
            tool["prompt"] = _on_grid(_LOCATE_PROMPT, point_grid)
    return gpt_policy.tools.catalog.ToolCatalog(data, f"{path} (ladder host)")


def selection_schema(
    catalog: gpt_policy.tools.catalog.ToolCatalog, dof: int, arms: Tuple[str, ...]
) -> Dict[str, Any]:
    tools = catalog.function_schemas(dof, arms)
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": [tool["function"]["name"] for tool in tools]},
            "arguments": {"anyOf": [tool["function"]["parameters"] for tool in tools]},
        },
        "required": ["name", "arguments"],
        "additionalProperties": False,
    }


def guidance(catalog: gpt_policy.tools.catalog.ToolCatalog) -> str:
    text = gpt_policy.harness.protocol.instructions(
        "probe", "probe", 6, ("left",), settings={"scene": {"safety_notes": ["probe"]}}, catalog=catalog
    )
    marker = text.find(_TAIL_MARKER)
    assert marker >= 0, "upstream instructions() no longer carries the shared behaviour guidance"
    assert _NOTE_LANGUAGE[0] in text, "upstream instructions() no longer asks for Chinese notes"
    return text[marker:].replace(*_NOTE_LANGUAGE)


def english_error(message: str) -> str:
    for chinese, english in _ERROR_TRANSLATIONS:
        message = message.replace(chinese, english)
    return message


def instructions(
    catalog: gpt_policy.tools.catalog.ToolCatalog,
    robot_name: str,
    camera_roles: Dict[str, str],
    depth_cameras: Tuple[str, ...],
    workspace_min_m: Optional[List[float]],
    workspace_max_m: Optional[List[float]],
    control_hz: float,
    point_grid: Optional[int] = None,
) -> str:
    views = "\n".join(
        f"- {role}: {'wrist camera, moving with the gripper' if role != 'top' else 'fixed camera looking at the workspace'}"
        f"{'; publishes depth, so locate_point returns a metric position' if role in depth_cameras else '; RGB only'}."
        for role in camera_roles
    )
    workspace = (
        f"- Commanded positions are clamped to x{_span(workspace_min_m, workspace_max_m, 0)}, "
        f"y{_span(workspace_min_m, workspace_max_m, 1)}, z{_span(workspace_min_m, workspace_max_m, 2)} metres."
        if workspace_min_m and workspace_max_m
        else "- No workspace clamp is applied; an unreachable target simply fails to arrive."
    )
    return f"""You are a {robot_name} controller. One 6-DoF arm with a parallel gripper, driven through the GPT-Policy tool protocol.

Robot and frame conventions:
- The arm is commanded in its own base_link frame: +x forward, +y left, +z up, metres.
- TCP is the midpoint between the fingertips. pose_xyzrpy reports it as [x,y,z,roll,pitch,yaw]; command pose_xyzquat=[x,y,z,qx,qy,qz,qw] with a unit quaternion mapping the tool axes into base_link.
- Tool +z points from the wrist towards the fingertips, tool +y is the jaw opening axis. xyzw=[1,0,0,0] points tool +z along base -z and tool +x along base +x, which is the arm's reset orientation.
- The host samples the straight-line/SLERP path from the current TCP, commands each sample to the controller and then holds the final pose until motion stops. It preserves your targets and their order and changes nothing else.
- The controller is a position controller running at {control_hz:g} Hz and it stops short of a command under load; every result reports the target, the measured pose and the residual.
{workspace}
- Gripper commands are a normalized opening: 0 fully closed, 1 fully open.

Cameras:
{views}
- Images arrive in observation.images in the order listed above, labelled with the camera name.
- Read a position off an image with locate_point rather than guessing it. Give it the pixel {_on_grid("in the image you were shown", point_grid)}; the host scales it to the depth frame itself. Use the result only when metric_position_available is true.
- locate_point returns the surface point the pixel falls on, so a pixel on the top of an object gives that object's top, and a pixel on the table gives the table.

Fixed scene facts:
- The table surface is the lowest safe height; do not command the fingertips below it.
- To grasp, the fingertips must close AROUND an object, not on top of it. locate_point gives you the object's TOP, so a grasp commanded at that height closes on the upper edge and the object slips out as soon as you lift. Command the grasp below the located top, near the middle of the object's visible height, while staying above the table.
- Objects the task does not name are obstacles, not targets.

Return exactly one tool selection per turn: {{"name": "...", "arguments": {{...}}}}. No Markdown or text outside this object. The host executes the full motion segment, then supplies a fresh observation.

Tools:
{catalog.prompt_catalog(6, ("left",))}

State joint_pos/tcp_pose are live joint and calibrated TCP feedback. Only one arm is controlled.

{guidance(catalog)}"""


def _on_grid(text: str, point_grid: Optional[int]) -> str:
    grid = f"on a 0..{point_grid} grid over the image you were shown"
    return text if point_grid is None else text.replace("in the image you were shown", grid)


def _span(low: Optional[List[float]], high: Optional[List[float]], axis: int) -> str:
    return f"[{low[axis]:.3f}, {high[axis]:.3f}]"
