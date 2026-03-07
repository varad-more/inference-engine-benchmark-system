"""
FastAPI dashboard server for the benchmark system.

Endpoints:
  GET  /api/results               — list saved result files
  GET  /api/results/{id}          — load a specific result
  GET  /api/compare/{scenario}    — latest vllm+sglang comparison for scenario
  POST /api/run                   — trigger a new benchmark run
  GET  /api/run/{job_id}/status   — poll run progress
  WS   /ws/live                   — real-time metrics stream during a live run
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

import structlog
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Inference Engine Benchmark Dashboard",
    description="Real-time benchmark comparison for vLLM vs SGLang",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory job registry
# ---------------------------------------------------------------------------

class JobStatus(BaseModel):
    job_id: str
    scenario: str
    engines: list[str]
    status: str = "pending"  # pending | running | done | error
    progress: int = 0
    total: int = 0
    message: str = ""
    result_paths: list[str] = Field(default_factory=list)
    started_at: float = Field(default_factory=time.time)
    finished_at: float | None = None


_jobs: dict[str, JobStatus] = {}

# Active WebSocket connections for live streaming
_live_connections: set[WebSocket] = set()


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    scenario: str
    engines: list[str] = Field(default=["vllm", "sglang"])
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    vllm_host: str = "localhost"
    vllm_port: int = 8000
    sglang_host: str = "localhost"
    sglang_port: int = 8001


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/results")
async def list_results() -> JSONResponse:
    """List all saved benchmark result files."""
    files = sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return JSONResponse([
        {
            "id": f.stem,
            "filename": f.name,
            "size_bytes": f.stat().st_size,
            "modified": f.stat().st_mtime,
        }
        for f in files
    ])


@app.get("/api/results/{result_id}")
async def get_result(result_id: str) -> JSONResponse:
    """Load and return a specific result by its stem (filename without .json)."""
    # Allow both with and without .json extension
    candidates = list(RESULTS_DIR.glob(f"{result_id}*.json"))
    if not candidates:
        raise HTTPException(status_code=404, detail=f"Result '{result_id}' not found")
    path = candidates[0]
    return JSONResponse(json.loads(path.read_text()))


@app.get("/api/compare/{scenario}")
async def compare_scenario(scenario: str) -> JSONResponse:
    """
    Load the latest vllm and sglang result files for the given scenario
    and compute comparative deltas.
    """
    def _latest(engine: str) -> dict[str, Any] | None:
        pattern = f"{scenario}_{engine}_*.json"
        files = sorted(
            RESULTS_DIR.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not files:
            return None
        return json.loads(files[0].read_text())

    vllm_data = _latest("VLLMClient")
    sglang_data = _latest("SGLangClient")

    if not vllm_data and not sglang_data:
        # Try comparison file directly
        comp_files = sorted(
            RESULTS_DIR.glob(f"comparison_{scenario}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if comp_files:
            return JSONResponse(json.loads(comp_files[0].read_text()))
        raise HTTPException(status_code=404, detail=f"No results found for scenario '{scenario}'")

    response: dict[str, Any] = {"scenario": scenario}
    if vllm_data:
        response["vllm"] = vllm_data.get("metrics", {})
    if sglang_data:
        response["sglang"] = sglang_data.get("metrics", {})

    # Compute delta if both are available
    if vllm_data and sglang_data:
        vllm_m = vllm_data.get("metrics", {})
        sglang_m = sglang_data.get("metrics", {})

        def _delta(va: float, vb: float) -> float:
            return round((va - vb) / max(abs(va), 1e-9) * 100, 2)

        vllm_ttft = vllm_m.get("ttft", {}).get("p95", 0)
        sglang_ttft = sglang_m.get("ttft", {}).get("p95", 0)
        vllm_tps = vllm_m.get("throughput", {}).get("tokens_per_sec", 0)
        sglang_tps = sglang_m.get("throughput", {}).get("tokens_per_sec", 0)

        response["delta"] = {
            "ttft_p95_pct": _delta(vllm_ttft, sglang_ttft),
            "tokens_per_sec_pct": _delta(vllm_tps, sglang_tps),
            "note": "positive = vllm higher",
        }

    return JSONResponse(response)


@app.post("/api/run")
async def start_run(req: RunRequest, background_tasks: BackgroundTasks) -> JSONResponse:
    """Trigger a new benchmark run in the background."""
    job_id = str(uuid.uuid4())[:8]
    job = JobStatus(
        job_id=job_id,
        scenario=req.scenario,
        engines=req.engines,
        total=len(req.engines) * _estimate_requests(req.scenario),
    )
    _jobs[job_id] = job
    background_tasks.add_task(_run_benchmark_job, job_id, req)
    return JSONResponse({"job_id": job_id, "status": "pending"}, status_code=202)


@app.get("/api/run/{job_id}/status")
async def get_run_status(job_id: str) -> JSONResponse:
    """Poll the status of a background benchmark run."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    job = _jobs[job_id]
    return JSONResponse(job.model_dump())


@app.websocket("/ws/live")
async def live_metrics_ws(websocket: WebSocket) -> None:
    """
    WebSocket endpoint that streams real-time engine metrics and run progress.

    Clients receive JSON messages of the form:
      {"type": "metrics", "data": {...}} — engine metrics snapshot
      {"type": "progress", "data": {...}} — job progress update
      {"type": "done", "data": {...}}     — run completed
    """
    await websocket.accept()
    _live_connections.add(websocket)
    try:
        while True:
            # Keep connection alive; actual pushes come from _broadcast()
            await asyncio.sleep(1)
            # Echo heartbeat
            await websocket.send_json({"type": "heartbeat", "ts": time.time()})
    except WebSocketDisconnect:
        pass
    finally:
        _live_connections.discard(websocket)


# ---------------------------------------------------------------------------
# Background job runner
# ---------------------------------------------------------------------------

async def _broadcast(message: dict[str, Any]) -> None:
    """Push a message to all active WebSocket clients."""
    dead: set[WebSocket] = set()
    for ws in _live_connections:
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)
    _live_connections.difference_update(dead)


async def _run_benchmark_job(job_id: str, req: RunRequest) -> None:
    """Background task: run the requested scenario and update job status."""
    from benchmarks.runner import BenchmarkRunner, RequestResult
    from benchmarks.scenarios import SCENARIOS
    from engines.vllm_client import VLLMClient
    from engines.sglang_client import SGLangClient

    job = _jobs[job_id]
    job.status = "running"
    await _broadcast({"type": "progress", "data": job.model_dump()})

    scenario = SCENARIOS.get(req.scenario)
    if scenario is None:
        job.status = "error"
        job.message = f"Unknown scenario: {req.scenario}"
        await _broadcast({"type": "error", "data": job.model_dump()})
        return

    runner = BenchmarkRunner()
    completed = 0

    def _progress_cb(done: int, total: int, result: RequestResult) -> None:
        nonlocal completed
        completed = done
        job.progress = completed
        asyncio.create_task(_broadcast({
            "type": "progress",
            "data": {
                "job_id": job_id,
                "done": done,
                "total": total,
                "last_ttft_ms": result.ttft_ms,
                "last_success": result.success,
            },
        }))

    try:
        clients: dict[str, Any] = {}
        if "vllm" in req.engines:
            clients["vllm"] = VLLMClient(req.vllm_host, req.vllm_port, req.model)
        if "sglang" in req.engines:
            clients["sglang"] = SGLangClient(req.sglang_host, req.sglang_port, req.model)

        for engine_name, client in clients.items():
            results = await runner.run_scenario(scenario, client, _progress_cb)
            path = results.save()
            job.result_paths.append(str(path))

            # Stream per-engine metrics
            await _broadcast({
                "type": "metrics",
                "data": {
                    "engine": engine_name,
                    "ttft_p95": results.metrics.ttft.p95,
                    "tokens_per_sec": results.metrics.throughput.tokens_per_sec,
                    "kv_cache_timeline": results.metrics.kv_cache_timeline[-10:],
                },
            })

            if hasattr(client, "aclose"):
                await client.aclose()

        job.status = "done"
        job.finished_at = time.time()
        await _broadcast({"type": "done", "data": job.model_dump()})
        logger.info("job completed", job_id=job_id, paths=job.result_paths)

    except Exception as exc:
        logger.exception("job failed", job_id=job_id, error=str(exc))
        job.status = "error"
        job.message = str(exc)
        job.finished_at = time.time()
        await _broadcast({"type": "error", "data": job.model_dump()})


def _estimate_requests(scenario_name: str) -> int:
    from benchmarks.scenarios import SCENARIOS
    s = SCENARIOS.get(scenario_name)
    if s is None:
        return 100
    return getattr(s, "num_requests", getattr(s, "requests_per_level", 100))


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "timestamp": time.time()})


if __name__ == "__main__":
    uvicorn.run(
        "dashboard.app:app",
        host="0.0.0.0",
        port=3000,
        reload=False,
        log_level="info",
    )
