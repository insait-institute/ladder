import dataclasses
import functools
import os
import pathlib
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml

from ladder.components.inference.api.client.episode import Episode
from ladder.components.inference.api.client.metrics import aggregate
from ladder.components.inference.api.client.metrics.registry import merge

CELL = re.compile(r"^(?P<system>.+)_c\d+(?:\.a\d+)*$")
EPISODE = re.compile(r"^episode_(?P<index>\d+)_(?P<stamp>\d{4}(?:_\d{2}){5})(?:_env(?P<env>\d+))?_")
SUITES = ("robolab", "robot_arena", "simpler")
REAL_SUITE = "real robot"
REAL_MARKER = "eval_config.yaml"  # what a real-robot eval writes into its run directory
UNIT_DEPTH = 4
LANE = re.compile(r"^lane\d+$")
CAMERA_ORDER = {"main": 0, "wrist": 1}


@dataclasses.dataclass
class Run:
    run_id: str
    campaign: str
    benchmark: str
    suite: str
    model: str
    system: str
    paths: List[str]
    real: bool = False
    scored: bool = False
    episodes: int = 0
    successes: int = 0


@dataclasses.dataclass
class Unit:
    unit: str
    env_name: str
    robot: str
    episodes: int
    successes: int
    success_rate: float
    goal_progress: Optional[float]
    metrics: Dict[str, str]
    scored: bool


@dataclasses.dataclass
class Rollout:
    unit: str
    index: int
    batch: int
    env_index: int
    success: bool
    num_steps: int
    instruction: str
    subtasks: List[str]
    plan: List[str]
    commands: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    goals: List[bool]
    metrics: Dict[str, str]
    videos: List[Dict[str, str]]
    path: str
    cell: str
    system: str = ""


def _load(path: pathlib.Path) -> Any:
    try:
        return yaml.load(path.read_text(errors="replace"), Loader=yaml.UnsafeLoader)
    except (OSError, yaml.YAMLError):
        return None


def _mtime(path: pathlib.Path) -> int:
    try:
        return int(path.stat().st_mtime)
    except OSError:
        return 0


def pair(value: Any) -> Tuple[float, float]:
    try:
        numerator, denominator = (float(part) for part in str(value).split("/"))
    except (TypeError, ValueError):
        return 0.0, 0.0
    return numerator, denominator


def ratio(value: Any) -> Optional[float]:
    numerator, denominator = pair(value)
    return numerator / denominator if denominator else None


def cells(root: pathlib.Path) -> Iterator[Tuple[pathlib.Path, bool]]:
    """Cell directories under ``root``, marked False, and real-robot run directories, marked True.

    A sim cell is a ``<system>_c<job>[.a<n>]`` directory, and nothing below a cell yields again.
    A real-robot run is a directory that holds its episode recordings and the runner's eval config
    directly, with no sim scores or episode log above it, so the run dir is its own unit.
    Maintenance trees (``_archive``, ``_discarded``, ...) never count as either.
    """
    if _direct_recordings(root) and (root / REAL_MARKER).is_file() and not _sim_above(root, root):
        yield root, True
        return
    for dirpath, dirnames, _ in os.walk(root, followlinks=True):
        base = pathlib.Path(dirpath)
        matched = [name for name in dirnames if CELL.match(name) and not name.startswith("_")]
        if matched:
            for name in matched:
                yield base / name, False
            dirnames[:] = [
                name
                for name in dirnames
                if name not in matched and not name.startswith(("_", "episode_", "."))
            ]
            continue
        for name in list(dirnames):
            if name.startswith(("_", "episode_")) or LANE.fullmatch(name):
                dirnames.remove(name)
                continue
            child = base / name
            if _direct_recordings(child) and (child / REAL_MARKER).is_file() and not _sim_above(root, child):
                yield child, True
                dirnames.remove(name)
        dirnames[:] = [name for name in dirnames if not name.startswith((".", "_", "episode_"))]


def _direct_recordings(directory: pathlib.Path) -> bool:
    return any(child.is_dir() and EPISODE.match(child.name) for child in directory.iterdir())


def _sim_above(root: pathlib.Path, directory: pathlib.Path) -> bool:
    """Whether anything along ``directory``'s part of ``root`` looks like a sim run's tree: a
    results.yaml or an episode log, which real-robot runs never produce."""
    for ancestor in [*directory.parents, directory]:
        if not ancestor.is_relative_to(root):
            break
        if any((ancestor / kind).is_file() for kind in ("results.yaml", "episode_log.yaml")):
            return True
    return False


def _run_of(root: pathlib.Path, cell: pathlib.Path, real: bool = False) -> Run:
    above = cell.relative_to(root).parts[:-1]
    if real:
        # <root>/<session>/<eval dir>/<checkpoint>: the parent directory rides as the benchmark,
        # the session as the campaign.
        model, system, suite = cell.name, "", REAL_SUITE
        benchmark = above[-1] if above else root.name
        campaign = "/".join(above[:-1]) if len(above) > 1 else root.name
    else:
        system = CELL.match(cell.name).group("system")
        model = above[-1] if above else ""
        benchmark = above[-2] if len(above) > 1 else ""
        campaign = "/".join(above[:-2]) or root.name
        suite = next((s for s in SUITES if benchmark.startswith(s)), benchmark)
    return Run(
        run_id="/".join(part for part in (campaign, benchmark, model, system) if part),
        campaign=campaign,
        benchmark=benchmark,
        suite=suite,
        model=model,
        system=system,
        paths=[str(cell)],
        real=real,
    )


def _unit_dirs(cell: pathlib.Path) -> Dict[str, List[pathlib.Path]]:
    """Unit directories of one cell, grouped by unit name.

    A unit dir holds a results.yaml, or when no writer ever saved one, an episode log or the
    recordings parked one level below it. Deeper than the canonical
    ``<cell>/<checkpoint>/<unit>`` shape sit RoboLab's per-lane leftovers (``lane*/output/.../all``),
    which carry the launcher's whole-slice tallies beside the split per-unit copies the harness
    reads, so they are skipped here to keep a run's tallies equal to the harness's.
    """
    found: Dict[str, Dict[pathlib.Path, None]] = {}
    for depth in range(1, UNIT_DEPTH + 1):
        prefix = "/".join(["*"] * depth)
        for kind in ("results.yaml", "episode_log.yaml"):
            for path in cell.glob(f"{prefix}/{kind}"):
                if any(LANE.fullmatch(part) for part in path.relative_to(cell).parts[:-1]):
                    continue
                found.setdefault(path.parent.name, {})[path.parent] = None
        # One discovered without either marker: the directory holding episode recordings, which is
        # one level below its unit.
        for holder in cell.glob(prefix):
            if not holder.is_dir() or holder.name.startswith(("episode_", "lane")):
                continue
            if any(LANE.fullmatch(part) for part in holder.relative_to(cell).parts):
                continue
            if any(child.is_dir() and EPISODE.match(child.name) for child in holder.iterdir()):
                unit = holder.parent
                if unit != cell:
                    found.setdefault(unit.name, {})[unit] = None
    return {unit: sorted(dirs) for unit, dirs in found.items()}


def _real_units(cell: pathlib.Path) -> Dict[str, List[pathlib.Path]]:
    """Units of one real-robot run: the run directory itself is the unit."""
    if _direct_recordings(cell):
        return {cell.name: [cell]}
    return {}


@functools.lru_cache(maxsize=4096)
def _scores(unit_dir: str, _key: int) -> Optional[Tuple[str, str, Dict[str, str]]]:
    stats = _load(pathlib.Path(unit_dir) / "results.yaml")
    if not isinstance(stats, dict):
        return None
    robots, envs, maps = [], [], []
    for robot, per_env in stats.items():
        if not isinstance(per_env, dict):
            continue
        robots.append(str(robot))
        for env_name, metrics in per_env.items():
            envs.append(str(env_name))
            metrics = metrics if isinstance(metrics, dict) else {"success": metrics}
            maps.append({str(k): str(v) for k, v in metrics.items()})
    if not maps:
        return None
    return ", ".join(dict.fromkeys(robots)), ", ".join(dict.fromkeys(envs)), merge(*maps)


def _summary(unit: str, dirs: List[pathlib.Path], real: bool = False) -> Unit:
    scored = [s for s in (_scores(str(d), _mtime(d / "results.yaml")) for d in dirs) if s]
    if not scored:
        # A real-robot run tallies from the recorded episodes; a cell unit keeps its recorded
        # episodes watchable but unscored, so tallies still match the harness's.
        if real:
            tally = _real_tally(dirs)
            if tally is None:
                return Unit(unit, "", "", 0, 0, 0.0, None, {}, False)
            successes, annotated, episodes = tally
            if annotated:
                return Unit(
                    unit=unit,
                    env_name="",
                    robot="",
                    episodes=episodes,
                    successes=successes,
                    success_rate=successes / episodes,
                    goal_progress=None,
                    metrics={"success": f"{successes}/{episodes}"},
                    scored=True,
                )
            return Unit(unit, "", "", episodes, 0, 0.0, None, {}, False)
        episodes = sum(len(_recordings(unit_dir)) for unit_dir in dirs)
        if episodes:
            return Unit(unit, "", "", episodes, 0, 0.0, None, {}, False)
        return Unit(unit, "", "", 0, 0, 0.0, None, {}, False)
    metrics = merge(*[m for _, _, m in scored])
    successes, episodes = pair(metrics.get("success"))
    return Unit(
        unit=unit,
        env_name=", ".join(dict.fromkeys(env for _, env, _ in scored)),
        robot=", ".join(dict.fromkeys(robot for robot, _, _ in scored)),
        episodes=int(episodes),
        successes=int(successes),
        success_rate=successes / episodes if episodes else 0.0,
        goal_progress=ratio(metrics.get("goal_progress")),
        metrics=metrics,
        scored=True,
    )


def _real_tally(dirs: List[pathlib.Path]) -> Optional[Tuple[int, bool, int]]:
    """(successes, annotated, episodes) over the episode metas of dirs without a scores map."""
    episodes, successes, annotated = 0, 0, False
    for unit_dir in dirs:
        for _, _, directory in _recordings(unit_dir):
            meta = _load(directory / "episode_meta.yaml")
            if not isinstance(meta, dict):
                continue
            episodes += 1
            value = meta.get("success")
            if value is not None:
                annotated = True
            else:
                continue
            successes += int(bool(value))
    return (successes, annotated, episodes) if episodes else None


def _recordings(unit_dir: pathlib.Path) -> List[Tuple[int, int, pathlib.Path]]:
    """(batch, env, directory) per episode recording under ``unit_dir``, or [] when there are none.

    Batched runners park recordings in a child of the unit dir (``episodes/`` or ``rec/``); a
    real-robot run writes them directly into the run directory. The first candidate with matching
    episode directories wins.
    """
    if not unit_dir.is_dir():
        return []
    children = sorted(p for p in unit_dir.iterdir() if p.is_dir())
    for child in [unit_dir, *children]:
        parsed = [(EPISODE.match(e.name), e) for e in child.iterdir()]
        parsed = [(m, e) for m, e in parsed if m]
        if not parsed:
            continue
        stamps = sorted(
            {m.group("stamp") for m, _ in parsed},
            key=lambda s: min(int(m.group("index")) for m, _ in parsed if m.group("stamp") == s),
        )
        return sorted((stamps.index(m.group("stamp")) + 1, int(m.group("env") or 0), e) for m, e in parsed)
    return []


@functools.lru_cache(maxsize=8)
def _log_rows(unit_dir: str, _key: int) -> List[Dict[str, Any]]:
    rows = _load(pathlib.Path(unit_dir) / "episode_log.yaml") or []
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


@functools.lru_cache(maxsize=512)
def _rollouts(unit_dir: str, _key: int) -> List[Rollout]:
    path = pathlib.Path(unit_dir)
    logged = {
        (int(r.get("batch") or 1), int(r.get("batch_env_index") or 0)): r for r in _log_rows(unit_dir, _key)
    }
    recorded = {(batch, env): directory for batch, env, directory in _recordings(path)}
    out = []
    for index, key in enumerate(sorted(set(logged) | set(recorded))):
        row, directory = logged.get(key), recorded.get(key)
        meta = (_load(directory / "episode_meta.yaml") if directory else None) or {}
        meta = meta if isinstance(meta, dict) else {}
        episode = Episode.from_log_entry(row) if row else None
        # Without a log row (a real-robot episode), the recorder's metadata is all there is.
        events = list(meta.get("events") or (episode.events if episode else []))
        planned = [e.get("subtasks") or [] for e in events if e.get("kind") == "plan"]
        out.append(
            Rollout(
                unit=path.name,
                index=index,
                batch=key[0],
                env_index=key[1],
                success=bool(row.get("success") if row else meta.get("success")),
                num_steps=int((row or meta).get("num_steps") or 0),
                instruction=str((row or meta).get("instruction") or meta.get("default_instruction") or ""),
                subtasks=list(episode.subtasks) if episode else [],
                plan=list(episode.plan) if episode else (planned[-1] if planned else []),
                commands=_commands(meta, episode),
                events=events,
                goals=list(episode.goals) if episode else [],
                metrics=aggregate([episode]) if episode else {},
                videos=_videos(directory) if directory else [],
                path=str(directory or ""),
                cell=unit_dir,
                system=str(meta.get("system") or ""),
            )
        )
    return out


def _rev(unit_dir: pathlib.Path) -> int:
    """Cache key for a unit's rollouts: the episode log's stamp, or the unit dir itself, which
    changes whenever a recording leaves or joins it (a run with no episode log)."""
    return max(_mtime(unit_dir / "episode_log.yaml"), _mtime(unit_dir))


def _videos(episode_dir: pathlib.Path) -> List[Dict[str, str]]:
    found = [{"name": p.stem, "path": str(p)} for p in sorted(episode_dir.glob("*.mp4"))]
    return sorted(found, key=lambda c: (CAMERA_ORDER.get(c["name"], 2), c["name"]))


def _commands(meta: Dict[str, Any], episode: Optional[Episode]) -> List[Dict[str, Any]]:
    log = meta.get("instruction_log")
    if isinstance(log, list) and log:
        return [{"step": int(r.get("step", 0)), "instruction": str(r.get("instruction", ""))} for r in log]
    history = episode.instruction_history[:1] if episode else []
    single = meta.get("instruction") or (history[0] if history else "")
    return [{"step": 0, "instruction": str(single)}] if single else []


def record(unit_dir: pathlib.Path, batch: int, env_index: int) -> Optional[Episode]:
    for row in _log_rows(str(unit_dir), _mtime(unit_dir / "episode_log.yaml")):
        if (int(row.get("batch") or 1), int(row.get("batch_env_index") or 0)) == (batch, env_index):
            return Episode.from_log_entry(row)
    return None


def _cell_tally(cell: pathlib.Path) -> Tuple[int, int]:
    """(successes, episodes) over one cell's canonical per-unit results.yaml, the harness's count."""
    successes = episodes = 0
    for results_path in cell.glob("*/*/results.yaml"):
        if any(LANE.fullmatch(part) for part in results_path.relative_to(cell).parts[:-1]):
            continue
        stats = _load(results_path)
        for envs in (stats or {}).values():
            if not isinstance(envs, dict):
                continue
            for metrics in envs.values():
                got = pair(metrics.get("success") if isinstance(metrics, dict) else metrics)
                successes, episodes = successes + got[0], episodes + got[1]
    return int(successes), int(episodes)


def run_tally(run: Run) -> Tuple[int, int]:
    """(successes, episodes) pooled over a run's canonical tallies, so the sidebar can hide
    campaigns without results without paying a per-run units walk."""
    paths = [pathlib.Path(cell) for cell in run.paths]
    if run.real:
        tally = _real_tally(paths) or (0, False, 0)
        return int(tally[0]), int(tally[2])
    successes = episodes = 0
    for cell in paths:
        got = _cell_tally(cell)
        successes, episodes = successes + got[0], episodes + got[1]
    return successes, episodes


class Store:
    def __init__(self, roots: List[pathlib.Path]):
        self.roots = [pathlib.Path(r).expanduser().resolve() for r in roots]
        self._runs: Optional[List[Run]] = None

    def refresh(self) -> None:
        self._runs = None

    def runs(self) -> List[Run]:
        if self._runs is not None:
            return self._runs
        merged: Dict[str, Run] = {}
        for root in (r for r in self.roots if r.is_dir()):
            for cell, real in cells(root):
                run = _run_of(root, cell, real)
                if run.run_id in merged:
                    merged[run.run_id].paths.append(str(cell))
                else:
                    merged[run.run_id] = run
        # The tallies come from one results.yaml per unit per cell; a cached first load reads
        # thousands of small files over the store's filesystem, so they are fetched in parallel.
        pending: List[Tuple[Run, List[pathlib.Path]]] = []
        for run in merged.values():
            run.paths.sort()
            if run.real:
                successes, episodes = run_tally(run)
                run.episodes, run.successes, run.scored = episodes, successes, successes > 0
            else:
                pending.append((run, [pathlib.Path(cell) for cell in run.paths]))
        cells_dir = dict.fromkeys((cell for _, paths in pending for cell in paths))
        with ThreadPoolExecutor(max_workers=32) as pool:
            tallied = dict(zip(cells_dir, pool.map(_cell_tally, cells_dir), strict=True))
        for run, paths in pending:
            for cell in paths:
                succeeds, total = tallied[cell]
                run.successes, run.episodes = run.successes + succeeds, run.episodes + total
            run.scored = run.successes > 0
        self._runs = sorted(merged.values(), key=lambda r: (r.campaign, r.benchmark, r.model, r.system))
        return self._runs

    def run(self, run_id: str) -> Optional[Run]:
        return next((r for r in self.runs() if r.run_id == run_id), None)

    def index(self, run_id: str) -> Dict[str, Tuple[List[pathlib.Path], Unit]]:
        run = self.run(run_id)
        found: Dict[str, Tuple[List[pathlib.Path], Unit]] = {}
        for cell in sorted(run.paths) if run else []:
            land = (_real_units if run.real else _unit_dirs)(pathlib.Path(cell))
            for unit, dirs in land.items():
                summary = _summary(unit, dirs, real=run.real)
                previous = found.get(unit)
                # A fuller measurement wins, and a scored one beats merely recorded episodes of the
                # same count; ties keep the later attempt, which is the rerun.
                better = (summary.scored, summary.episodes)
                if previous is None or better >= (previous[1].scored, previous[1].episodes):
                    found[unit] = (dirs, summary)
        return found

    def units(self, run_id: str) -> List[Unit]:
        return sorted((u for _, u in self.index(run_id).values()), key=lambda u: u.unit)

    def unit_dirs(self, run_id: str, unit: str) -> List[pathlib.Path]:
        entry = self.index(run_id).get(unit)
        return entry[0] if entry else []

    def episodes(self, run_id: str, unit: str) -> List[Rollout]:
        rows: List[Rollout] = []
        for unit_dir in self.unit_dirs(run_id, unit):
            rows.extend(_rollouts(str(unit_dir), _rev(unit_dir)))
        return [dataclasses.replace(row, index=i) for i, row in enumerate(rows)]

    def episode(self, run_id: str, unit: str, index: int) -> Optional[Rollout]:
        rows = self.episodes(run_id, unit)
        return rows[index] if 0 <= index < len(rows) else None

    def logs(self, run_id: str, unit: str = "") -> List[pathlib.Path]:
        run = self.run(run_id)
        starts = (
            self.unit_dirs(run_id, unit) if unit else [pathlib.Path(c) for c in (run.paths if run else [])]
        )
        found: List[pathlib.Path] = []
        for start in starts:
            for directory in [start, *_up_to_cell(start)]:
                found.extend(sorted(directory.glob("*.log")))
                found.extend(sorted(directory.glob("lane*/*.log")))
        return list(dict.fromkeys(found))

    def provenance(self, run_id: str) -> List[Dict[str, str]]:
        run = self.run(run_id)
        out = []
        for cell in sorted(run.paths) if run else []:
            path = pathlib.Path(cell) / "run_info.txt"
            if not path.is_file():
                continue
            parts = (line.partition(":") for line in path.read_text().splitlines())
            out.append({"cell": pathlib.Path(cell).name, **{k.strip(): v.strip() for k, _, v in parts if v}})
        return out


def _up_to_cell(start: pathlib.Path) -> List[pathlib.Path]:
    out = []
    for parent in start.parents:
        out.append(parent)
        if CELL.match(parent.name):
            break
    return out
