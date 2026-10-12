from typing import Any, Dict, List, Optional

from ladder.components.inference.api.client import scene_events
from ladder.components.inference.api.client.episode import Episode

SUCCESS, FAILURE, NEUTRAL = "success", "failure", "neutral"
FIELD_CHARS = 400
GRASP_SEVERITY = {
    scene_events.PLACED: SUCCESS,
    scene_events.MISPLACED: FAILURE,
    scene_events.DROPPED: FAILURE,
    scene_events.SLIPPED: FAILURE,
    scene_events.HELD: NEUTRAL,
}


def _event(step: int, kind: str, name: str, info: str = "", severity: str = NEUTRAL) -> Dict[str, Any]:
    return {"step": int(step), "kind": kind, "name": name, "info": info, "severity": severity}


def _line(value: Any) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= FIELD_CHARS else text[: FIELD_CHARS - 1] + "…"


def _verify(raw: Dict[str, Any]) -> Dict[str, Any]:
    verdict, truth, subtask = raw.get("verdict"), raw.get("gt"), raw.get("subtask")
    where = "" if subtask is None else f" subtask {subtask}"
    if verdict is None:
        return _event(raw["step"], "verify", f"verify{where} failed", _line(raw.get("reply") or ""))
    if truth is None:
        severity = NEUTRAL
    elif bool(verdict) != bool(truth):
        severity = FAILURE
    else:
        severity = SUCCESS if verdict else NEUTRAL
    truth_note = "" if truth is None else f"truth: {'done' if truth else 'not done'}"
    info = " · ".join(p for p in (truth_note, _line(raw.get("reply") or "")) if p)
    return _event(
        raw["step"], "verify", f"verified{where} {'done' if verdict else 'not done'}", info, severity
    )


def _advance(raw: Dict[str, Any]) -> Dict[str, Any]:
    truth = raw.get("gt")
    note = "" if truth is None else f"it was {'done' if truth else 'not done'}"
    return _event(
        raw["step"],
        "advance",
        f"advance to subtask {raw.get('subtask')}",
        " · ".join(p for p in (f"left subtask {raw.get('from')}", note) if p),
        NEUTRAL if truth is not False else FAILURE,
    )


def _grasp(raw: Dict[str, Any]) -> Dict[str, Any]:
    truth = raw.get("gt_held_done")
    note = "" if truth is None else f"its subtask was {'done' if truth else 'not done'}"
    return _event(
        raw["step"],
        "grasp",
        f"identified held object: {raw.get('held') or 'nothing'}",
        " · ".join(p for p in (note, _line(raw.get("reply") or "")) if p),
    )


def _controller(episode: Optional[Episode], recorded: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for raw in recorded or (episode.events if episode else []):
        if raw.get("step") is None:
            continue
        kind = str(raw.get("kind") or "event")
        if kind == "advance":
            out.append(_advance(raw))
        elif kind == "verify":
            out.append(_verify(raw))
        elif kind == "grasp":
            out.append(_grasp(raw))
        elif kind == "plan":
            subtasks = list(raw.get("subtasks") or [])
            out.append(
                _event(raw["step"], kind, f"planned {len(subtasks)} subtasks", "; ".join(map(str, subtasks)))
            )
        elif kind == "code":
            turn = raw.get("turn")
            name = "generated program" + ("" if turn is None else f" (turn {turn})")
            out.append(
                {
                    **_event(raw["step"], kind, name, _line(raw.get("code") or "")),
                    # The full text rides along for the episode page's program panel.
                    "code": str(raw.get("code") or ""),
                }
            )
        elif kind == "tool":
            report = raw.get("report")
            out.append(
                _event(
                    raw["step"],
                    kind,
                    f"{raw.get('tool') or 'tool'}: {raw.get('args') or ''}"[-FIELD_CHARS:],
                    _line(report),
                    NEUTRAL if raw.get("ok") is not False else FAILURE,
                )
            )
        else:
            fields = ", ".join(f"{k}={_line(v)}" for k, v in raw.items() if k not in ("step", "kind"))
            out.append(_event(raw["step"], kind, kind, fields))
    return out


def _goals(episode: Episode) -> List[Dict[str, Any]]:
    out = []
    for goal in range(len(episode.goals)):
        label = episode.subtasks[goal] if goal < len(episode.subtasks) else f"goal {goal}"
        for step, achieved in scene_events.goal_flips(episode, goal):
            name = f"goal {goal} {'achieved' if achieved else 'lost'}"
            out.append(_event(step, "goal", name, label, SUCCESS if achieved else FAILURE))
    return out


def _objects(episode: Episode) -> List[Dict[str, Any]]:
    out = []
    for grasp in scene_events.grasps(episode):
        off = "" if grasp.intent == scene_events.ON_TARGET else f" ({grasp.intent.replace('_', ' ')})"
        severity = NEUTRAL if grasp.intent == scene_events.ON_TARGET else FAILURE
        out.append(_event(grasp.start, "grasp", f"grasped {grasp.object}{off}", "", severity))
        if grasp.end is not None:
            where = f" on {grasp.resting_on}" if grasp.resting_on else ""
            name = f"{grasp.outcome} {grasp.object}{where}"
            out.append(_event(grasp.end, "release", name, "", GRASP_SEVERITY.get(grasp.outcome, NEUTRAL)))
    for event in scene_events.displacements(episode):
        if event.regressed_goal is None and not event.fell:
            continue
        name = f"{'dropped off the table' if event.fell else 'knocked'} {event.object}"
        undone = "" if event.regressed_goal is None else f", undid goal {event.regressed_goal}"
        out.append(_event(event.moved_at, "knock", name, f"moved {event.distance_m:.2f} m{undone}", FAILURE))
    return out


def events(episode: Optional[Episode], recorded: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = _controller(episode, recorded)
    if episode is not None:
        out += _goals(episode) + _objects(episode)
    return sorted(out, key=lambda e: (e["step"], e["kind"]))


def failure_cause(episode: Episode) -> Optional[str]:
    return scene_events.failure_cause(episode)
