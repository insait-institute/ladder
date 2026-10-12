import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ladder.components.inference.api.client.hlc.vlm.base import VLM
from ladder.components.inference.models.harness import prompts
from ladder.components.inference.models.harness.arm import Arm, ArmConfig, Robot
from ladder.components.inference.models.harness.perception import ObjectNotFoundError, Scene, SceneConfig
from ladder.components.inference.models.harness.tools import Report, Tools
from ladder.core.common.logger import get_logger
from ladder.core.databib.config import AutoConfig, Config, Configurable

LOGGER = get_logger(__name__)

_MAX_ATTEMPTS_PER_GOAL = 2
_MAX_QUESTIONS_PER_GOAL = 4
_MAX_TOKENS = 2048
_TEMPERATURE = 0.0


def _as_goal(entry: Any) -> Optional[Tuple[str, Optional[Tuple[str, str]]]]:
    # Keep the planner's sentence rather than rebuilding it from the pair: it is what the planner
    # reads back as attempted, and a task visiting one pair twice must not collapse to one line.
    if isinstance(entry, dict):
        text = str(entry.get("goal", "")).strip()
        moved = str(entry.get("object", "")).strip()
        destination = str(entry.get("destination", "")).strip()
        if moved and destination:
            return text or f"move {moved} to {destination}", (moved, destination)
        entry = text
    text = str(entry).strip()
    return (text, None) if text else None


def parse_json(reply: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", re.sub(r"```(?:json)?|```", "", reply or ""), re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


class AgentConfig(Config):
    vlm: AutoConfig
    scene: SceneConfig = SceneConfig()
    arm: ArmConfig = ArmConfig()
    robot: str = "robot arm"


class Agent(Configurable[AgentConfig]):
    def __init__(self, config: AgentConfig, vlm: VLM) -> None:
        super().__init__(config)
        self._vlm = vlm

    def run(self, robot: Robot, task: str, events: List[Dict[str, object]]) -> None:
        arm = Arm(self.config.arm, robot)
        scene = Scene(self.config.scene, self._vlm, arm)
        tools = Tools(arm, scene)
        done: List[str] = []
        while True:
            spent = arm.steps
            scene.refresh()
            goals = self._plan(scene, task, done)
            if not goals:
                goals = self._plan(scene, task, done, recheck=True)
            events.append({"type": "plan", "goals": [goal for goal, _ in goals]})
            if not goals:
                return
            for goal, planned in goals:
                outcome = self._pursue(tools, scene, goal, planned, events)
                done.append(f"{goal} -- {outcome}" if outcome else goal)
                # A scene that parks the arm before every look clears the view itself.
                if scene.config.clear_view_position_m is None:
                    tools.go_home()
            if arm.steps == spent:
                return

    def _pursue(
        self,
        tools: Tools,
        scene: Scene,
        goal: str,
        planned: Optional[Tuple[str, str]],
        events: List[Dict[str, object]],
    ) -> Optional[str]:
        history: List[str] = []
        repeated: Set[str] = set()
        attempt, asked = 0, 0
        while attempt < _MAX_ATTEMPTS_PER_GOAL and asked < _MAX_QUESTIONS_PER_GOAL:
            spent = tools.arm.steps
            if planned is not None:
                # An object and a destination ARE a pick_and_place. The choice still runs on retry.
                tool: Optional[str] = "pick_and_place"
                args: Dict[str, Any] = {"object_name": planned[0], "destination_name": planned[1]}
                planned = None
            else:
                tool, args = self._choose(scene, tools, goal, history)
            if tool is None:
                events.append(
                    {"type": "tool", "goal": goal, "tool": None, "report": "no tool chosen", "ok": False}
                )
                return history[-1] if history else None
            call = f"{tool}({sorted(args.items())})"
            report = self._invoke(tools, tool, args)
            LOGGER.info(f"goal {goal!r} attempt {attempt}.{asked}: {tool}({args}) -> {report}")
            history.append(f"{tool}({args}) -> {report}")
            events.append(
                {
                    "type": "tool",
                    "goal": goal,
                    "tool": tool,
                    "args": args,
                    "report": str(report),
                    "ok": report.ok,
                }
            )
            if tools.arm.steps == spent:
                if call not in repeated:
                    repeated.add(call)
                    asked += 1
                    continue
                attempt += 1
                continue
            if report.ok:
                return str(report)
            attempt += 1
        return history[-1] if history else None

    def _invoke(self, tools: Tools, tool: str, args: Dict[str, Any]) -> Report:
        function = tools.by_name().get(tool)
        if function is None:
            return Report(False, f"no such tool {tool!r}")
        try:
            return function(**args)
        except ObjectNotFoundError as error:
            return Report(False, f"failed: {error}")
        except TypeError as error:
            return Report(False, f"failed: wrong arguments for {tool} ({error})")

    def _plan(
        self, scene: Scene, task: str, done: List[str], recheck: bool = False
    ) -> List[Tuple[str, Optional[Tuple[str, str]]]]:
        attempted = (
            prompts.ATTEMPTED_HEADER.format(lines="\n".join(f"- {line}" for line in done)) if done else ""
        )
        prompt = prompts.PLAN.format(robot=self.config.robot, task=task, attempted=attempted)
        reply = self._ask(scene, prompt + prompts.RECHECK if recheck else prompt)
        payload = parse_json(reply) or {}
        goals = [goal for goal in map(_as_goal, payload.get("goals", [])) if goal is not None]
        if not goals and not done:
            LOGGER.warn(f"no plan parsed from {reply[:200]!r}; falling back to the task itself")
            goals = [(task, None)]
        return goals

    def _choose(
        self, scene: Scene, tools: Tools, goal: str, history: List[str]
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        scene.refresh()
        rendered = (
            prompts.HISTORY_HEADER.format(lines="\n".join(f"- {line}" for line in history)) if history else ""
        )
        reply = self._ask(
            scene,
            prompts.CHOOSE.format(
                robot=self.config.robot,
                goal=goal,
                hand=f"holding {tools.held.name!r}" if tools.held is not None else "empty",
                catalog=tools.catalog(),
                history=rendered,
            ),
        )
        payload = parse_json(reply) or {}
        tool = payload.get("tool")
        args = payload.get("args")
        if not isinstance(tool, str) or not isinstance(args, dict):
            return None, {}
        return tool, {str(key): value for key, value in args.items()}

    def _ask(self, scene: Scene, text: str) -> str:
        return self._vlm.complete(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "image", "image": scene.image()}, {"type": "text", "text": text}],
                }
            ],
            max_tokens=_MAX_TOKENS,
            temperature=_TEMPERATURE,
        )
