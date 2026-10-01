import argparse
import dataclasses
import pathlib
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from ladder.components.dashboard import store, timeline, video
from ladder.components.inference.api.client.metrics.registry import merge

STATIC = pathlib.Path(__file__).parent / "static"
LOG_TAIL_BYTES = 256 * 1024


def _totals(units: List[store.Unit]) -> Dict[str, Any]:
    metrics = merge(*[u.metrics for u in units]) if units else {}
    successes, episodes = store.pair(metrics.get("success"))
    return {
        "units": len(units),
        "episodes": int(episodes),
        "successes": int(successes),
        "success_rate": successes / episodes if episodes else 0.0,
        "goal_progress": store.ratio(metrics.get("goal_progress")),
        "metrics": metrics,
    }


def _mount_browse(app: FastAPI, results: store.Store) -> None:
    @app.get("/api/runs")
    def runs() -> Dict[str, Any]:
        return {
            "roots": [str(p) for p in results.roots],
            "runs": [dataclasses.asdict(r) for r in results.runs()],
        }

    @app.post("/api/refresh")
    def refresh() -> Dict[str, Any]:
        results.refresh()
        return {"runs": len(results.runs())}

    @app.get("/api/units")
    def units(run: str) -> Dict[str, Any]:
        found = results.run(run)
        if found is None:
            raise HTTPException(status_code=404, detail=f"unknown run: {run!r}")
        rows = results.units(run)
        return {
            "run": dataclasses.asdict(found),
            **_totals(rows),
            "unit_rows": [dataclasses.asdict(u) for u in rows],
            "provenance": results.provenance(run),
            "logs": [{"name": p.name, "path": str(p)} for p in results.logs(run)],
        }

    @app.get("/api/episodes")
    def episodes(run: str, unit: str) -> Dict[str, Any]:
        rows = results.episodes(run, unit)
        summary = next((u for u in results.units(run) if u.unit == unit), None)
        return {
            "unit": unit,
            "summary": dataclasses.asdict(summary) if summary else None,
            "episodes": [dataclasses.asdict(row) for row in rows],
            "logs": [{"name": p.name, "path": str(p)} for p in results.logs(run, unit)],
        }


def _mount_episode(app: FastAPI, results: store.Store) -> None:
    def _row(run: str, unit: str, index: int) -> store.Rollout:
        found = results.episode(run, unit, index)
        if found is None:
            raise HTTPException(status_code=404, detail="unknown episode")
        return found

    @app.get("/api/episode")
    def episode(run: str, unit: str, index: int) -> Dict[str, Any]:
        row = _row(run, unit, index)
        record = store.record(pathlib.Path(row.cell), row.batch, row.env_index) if row.cell else None
        return {
            "episode": dataclasses.asdict(row),
            # The row already holds what was recorded; the log row is not required.
            "events": timeline.events(record, row.events),
            "failure_cause": timeline.failure_cause(record) if record else None,
            "goal_objects": record.goal_objects if record else [],
        }

    @app.get("/api/traces")
    def traces(run: str, unit: str, index: int) -> Dict[str, Any]:
        row = _row(run, unit, index)
        record = store.record(pathlib.Path(row.cell), row.batch, row.env_index) if row.cell else None
        if record is None:
            return {}
        return {
            "num_steps": record.num_steps,
            "subtasks": record.subtasks,
            "motion": {
                arm: [[step, values] for step, values in rows] for arm, rows in record.motion_trace.items()
            },
            "goals": [[step, [bool(x) for x in a]] for step, a in record.goal_trace],
            "objects": {n: [[s, c] for s, c in rows] for n, rows in record.object_trace.items()},
        }

    @app.get("/api/video")
    def clip(run: str, unit: str, index: int, name: str) -> FileResponse:
        camera = next((c for c in _row(run, unit, index).videos if c["name"] == name), None)
        if camera is None:
            raise HTTPException(status_code=404, detail=f"no camera {name!r}")
        return FileResponse(video.playable(pathlib.Path(camera["path"])), media_type="video/mp4")

    @app.get("/api/log", response_class=PlainTextResponse)
    def log(run: str, path: str, unit: str = "") -> str:
        if pathlib.Path(path) not in results.logs(run, unit):
            raise HTTPException(status_code=404, detail="unknown log for this run")
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - LOG_TAIL_BYTES))
            return handle.read().decode("utf-8", errors="replace")


def create_app(roots: Optional[List[pathlib.Path]] = None) -> FastAPI:
    app = FastAPI(title="Ladder eval dashboard")
    results = store.Store(roots or [])
    app.mount("/static", StaticFiles(directory=str(STATIC), follow_symlink=True), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        stamps = [int(p.stat().st_mtime) for p in STATIC.iterdir() if p.is_file()]
        return (STATIC / "index.html").read_text().replace("{{version}}", str(max(stamps, default=0)))

    _mount_browse(app, results)
    _mount_episode(app, results)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse robot-policy eval results in a browser.")
    parser.add_argument(
        "--results",
        type=pathlib.Path,
        nargs="+",
        action="append",
        required=True,
        help="one or more result directories; the flag may repeat",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run(
        create_app([path for group in args.results for path in group]), host=args.host, port=args.port
    )


if __name__ == "__main__":
    main()
