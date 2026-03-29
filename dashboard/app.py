"""
FastAPI dashboard server for the benchmark system.

Endpoints:
  GET  /                        — browser-friendly dashboard page
  GET  /api/results             — list saved result files
  GET  /api/results/{id}        — load a specific result
  GET  /api/compare/{scenario}  — latest vllm+sglang comparison for scenario
  GET  /api/current             — detect the currently running benchmark/test
  POST /api/run                 — trigger a new benchmark run
  GET  /api/run/{job_id}/status — poll run progress
  WS   /ws/live                 — real-time metrics stream during a live run
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import structlog
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from analysis import get_result_model, load_results, select_model_results
from engines.base_client import DEFAULT_MODEL

logger = structlog.get_logger(__name__)

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
RESULTS_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Inference Engine Benchmark Dashboard",
    description="Real-time benchmark comparison for vLLM vs SGLang",
    version="0.2.0",
)

_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
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

# Lock protecting _jobs and _live_connections from concurrent mutation
_state_lock = asyncio.Lock()


MODEL_LABELS = {
    "qwen": "Qwen",
    "qwen7b": "Qwen 7B",
    "mistral7b": "Mistral 7B",
    "gemma2b": "Gemma 2B",
    "gemma9b": "Gemma 9B",
    "phi3": "Phi-3 mini",
}
ENGINE_LABELS = {
    "VLLMClient": "vLLM",
    "SGLangClient": "SGLang",
}


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


# Pattern that rejects shell metacharacters and allows only safe identifiers
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")
# Model names can include org/model-name patterns (e.g. Qwen/Qwen2.5-1.5B-Instruct)
_SAFE_MODEL_RE = re.compile(r"^[a-zA-Z0-9_\-./]+$")


class RunRequest(BaseModel):
    scenario: str
    engines: list[str] = Field(default=["vllm", "sglang"])
    model: str = DEFAULT_MODEL
    vllm_host: str = "localhost"
    vllm_port: int = 8000
    sglang_host: str = "localhost"

    @field_validator("scenario")
    @classmethod
    def validate_scenario(cls, v: str) -> str:
        if not _SAFE_NAME_RE.match(v):
            raise ValueError(f"Invalid scenario name: {v!r}")
        return v

    @field_validator("engines")
    @classmethod
    def validate_engines(cls, v: list[str]) -> list[str]:
        allowed = {"vllm", "sglang"}
        for engine in v:
            if engine not in allowed:
                raise ValueError(f"Invalid engine: {engine!r}. Must be one of {allowed}")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if not _SAFE_MODEL_RE.match(v) or len(v) > 200:
            raise ValueError(f"Invalid model name: {v!r}")
        return v

    @field_validator("vllm_host", "sglang_host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        if not _SAFE_NAME_RE.match(v.replace(".", "")) or len(v) > 253:
            raise ValueError(f"Invalid host: {v!r}")
        return v

    sglang_port: int = 8001


# ---------------------------------------------------------------------------
# Process / system helpers
# ---------------------------------------------------------------------------


def _run_cmd(argv: list[str], timeout_sec: int = 8) -> str:
    """Run a command with explicit argv (no shell) and return stdout text."""
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except Exception as exc:  # pragma: no cover
        logger.debug("command failed", argv=argv, error=str(exc))
        return ""
    return completed.stdout.strip()


def _strip_pgrep_prefix(line: str) -> str:
    parts = line.strip().split(maxsplit=1)
    if parts and parts[0].isdigit():
        return parts[1] if len(parts) > 1 else ""
    return line.strip()


def _arg_value(tokens: list[str], *flags: str) -> str | None:
    for flag in flags:
        if flag in tokens:
            idx = tokens.index(flag)
            if idx + 1 < len(tokens):
                return tokens[idx + 1]
    return None


def _parse_active_benchmark_command(command: str) -> dict[str, Any] | None:
    """Parse a foreground benchmark CLI command into structured data."""
    command = _strip_pgrep_prefix(command)
    if "run_experiment.py" not in command or " run " not in f" {command} ":
        return None

    tokens = shlex.split(command)
    if "run_experiment.py" not in tokens or "run" not in tokens:
        return None

    scenario = _arg_value(tokens, "--scenario", "-s")
    engines = _arg_value(tokens, "--engines", "-e")
    model = _arg_value(tokens, "--model", "-m")
    prompt_pack = _arg_value(tokens, "--prompt-pack")

    return {
        "state": "running",
        "source": "cli-process",
        "scenario": scenario,
        "engines": engines.split(",") if engines else [],
        "engine": engines,
        "model": model,
        "prompt_pack": prompt_pack,
        "raw_command": command,
    }


def _parse_helper_script_command(command: str) -> dict[str, Any] | None:
    """Parse helper shell script names like run_gemma2b_vllm_single.sh."""
    command = _strip_pgrep_prefix(command)
    match = re.search(r"/(run_[^\s]+\.sh)\b", command)
    if not match:
        return None

    basename = match.group(1)
    name = basename.removeprefix("run_").removesuffix(".sh")
    parts = name.split("_")
    if not parts:
        return None

    model_key = parts[0]
    engine = next((part for part in parts if part in {"vllm", "sglang"}), None)
    if "throughput" in parts:
        scenario = "throughput_ramp"
    elif "single" in parts:
        scenario = "single_request_latency"
    else:
        scenario = None

    return {
        "state": "queued_or_cooldown",
        "source": "helper-script",
        "scenario": scenario,
        "engines": [engine] if engine else [],
        "engine": engine,
        "model": MODEL_LABELS.get(model_key, model_key),
        "script": basename,
        "raw_command": command,
    }


def _parse_server_command(command: str) -> dict[str, Any] | None:
    command = _strip_pgrep_prefix(command)
    if "pgrep -af" in command:
        return None
    tokens = shlex.split(command)

    if "vllm" in tokens and "serve" in tokens:
        return {
            "engine": "vllm",
            "model": _arg_value(tokens, "--model"),
            "raw_command": command,
        }

    if "sglang.launch_server" in command:
        return {
            "engine": "sglang",
            "model": _arg_value(tokens, "--model-path"),
            "raw_command": command,
        }

    if "run_experiment.py" in tokens and "serve" in tokens:
        return {
            "engine": "dashboard",
            "model": None,
            "raw_command": command,
        }

    return None


def _result_file_payload(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": path.stem,
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "modified": path.stat().st_mtime,
        "type": "unknown",
    }
    try:
        data = json.loads(path.read_text())
    except Exception:
        return payload

    if "metrics" in data and "scenario_name" in data and "engine_name" in data:
        metrics = data.get("metrics", {})
        throughput = metrics.get("throughput", {})
        payload.update(
            {
                "type": "result",
                "scenario": data.get("scenario_name"),
                "engine": ENGINE_LABELS.get(data.get("engine_name"), data.get("engine_name")),
                "model": data.get("run_metadata", {}).get("model"),
                "prompt_pack": data.get("workload_metadata", {}).get("prompt_pack"),
                "success_rate_pct": round((1 - metrics.get("error_rate", 0.0)) * 100, 1),
                "ttft_p95_ms": metrics.get("ttft", {}).get("p95"),
                "latency_p95_ms": metrics.get("latency", {}).get("p95"),
                "tokens_per_sec": throughput.get("tokens_per_sec"),
                "requests_per_sec": throughput.get("requests_per_sec"),
            }
        )
    elif "scenario_name" in data and "vllm" in data and "sglang" in data:
        payload.update(
            {
                "type": "comparison",
                "scenario": data.get("scenario_name"),
                "model": _comparison_file_model(data),
            }
        )
    elif "tasks" in data and "model" in data:
        payload.update({"type": "matrix_manifest", "model": data.get("model")})

    return payload


def _payload_matches_model(payload: dict[str, Any], model: str | None) -> bool:
    if model is None:
        return True
    return payload.get("model") == model


def _latest_results_payload(limit: int = 8, model: str | None = None) -> list[dict[str, Any]]:
    files = sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    payloads: list[dict[str, Any]] = []
    for file_path in files:
        payload = _result_file_payload(file_path)
        if not _payload_matches_model(payload, model):
            continue
        payloads.append(payload)
        if len(payloads) >= limit:
            break
    return payloads


def _latest_engine_result(results: list[dict[str, Any]], engine_name: str) -> dict[str, Any] | None:
    matches = [r for r in results if r.get("engine_name") == engine_name]
    if not matches:
        return None
    return max(matches, key=lambda item: float(item.get("timestamp", 0.0)))


def _comparison_file_model(data: dict[str, Any]) -> str | None:
    for key in ("vllm", "sglang"):
        payload = data.get(key)
        if isinstance(payload, dict):
            model = get_result_model(payload)
            if model:
                return model
    return None


def _system_status_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {}

    # GPU via nvidia-smi (no shell, direct argv)
    gpu_line = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if gpu_line:
        first_line = gpu_line.splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        if len(parts) == 4:
            payload["gpu"] = {
                "temperature_c": float(parts[0]),
                "utilization_pct": float(parts[1]),
                "memory_used_mib": float(parts[2]),
                "memory_total_mib": float(parts[3]),
            }

    # Load average via Python stdlib
    try:
        load = os.getloadavg()
        payload["load_avg"] = ", ".join(f"{x:.2f}" for x in load)
    except OSError:
        pass

    # Memory via /proc/meminfo (Linux) — no subprocess needed
    try:
        meminfo: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, rest = line.split(":", 1)
                # Values are in kB
                meminfo[key.strip()] = int(rest.strip().split()[0])
        total = meminfo.get("MemTotal", 0) // 1024
        available = meminfo.get("MemAvailable", 0) // 1024
        used = total - available
        payload["memory"] = {"used_mib": used, "available_mib": available}
    except (OSError, ValueError, KeyError):
        pass

    return payload


def _latest_completed_result_payload(model: str | None = None) -> dict[str, Any] | None:
    for item in _latest_results_payload(limit=30, model=model):
        if item.get("type") == "result":
            return item
    return None


def _current_activity_payload(model: str | None = None) -> dict[str, Any]:
    active_job = next((job for job in _jobs.values() if job.status == "running"), None)
    if active_job is not None:
        current: dict[str, Any] | None = {
            "state": "running",
            "source": "dashboard-job",
            "scenario": active_job.scenario,
            "engines": active_job.engines,
            "engine": ",".join(active_job.engines),
            "model": None,
            "prompt_pack": None,
            "job_id": active_job.job_id,
            "progress": active_job.progress,
            "total": active_job.total,
        }
    else:
        current = None

    if current is None:
        run_lines = _run_cmd(["pgrep", "-af", "run_experiment.py run"]).splitlines()
        for line in run_lines:
            parsed = _parse_active_benchmark_command(line)
            if parsed is not None:
                current = parsed
                break

    if current is None:
        script_lines = _run_cmd(["pgrep", "-af", r"run_.*\.sh"]).splitlines()
        for line in script_lines:
            parsed = _parse_helper_script_command(line)
            if parsed is not None:
                current = parsed
                break

    server_lines = _run_cmd(
        ["pgrep", "-af", "vllm serve|sglang.launch_server|run_experiment.py serve"]
    ).splitlines()
    active_servers = []
    for line in server_lines:
        parsed = _parse_server_command(line)
        if parsed is not None:
            active_servers.append(parsed)

    return {
        "timestamp": time.time(),
        "current": current or {"state": "idle", "source": "none"},
        "active_servers": active_servers,
        "result_filter_model": model,
        "latest_results": _latest_results_payload(limit=5, model=model),
        "latest_completed_result": _latest_completed_result_payload(model=model),
        "system": _system_status_payload(),
    }


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def dashboard_home() -> HTMLResponse:
    """Browser-friendly landing page for the benchmark dashboard."""
    html = """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Inference Benchmark Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
          :root {
            --bg: #0b1020;
            --panel: #121933;
            --panel-2: #172043;
            --panel-3: #0f1530;
            --border: #27335a;
            --text: #e8eefc;
            --muted: #9fb0d0;
            --link: #7cc4ff;
            --ok: #7ef0a2;
            --warn: #ffd166;
            --bad: #ff7b7b;
          }
          body { font-family: system-ui, -apple-system, sans-serif; margin: 2rem; background: var(--bg); color: var(--text); }
          a { color: var(--link); }
          code { background: #11182c; padding: 0.15rem 0.35rem; border-radius: 6px; }
          .muted { color: var(--muted); }
          .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin-top: 1rem; }
          .card { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 1rem; }
          .wide { grid-column: 1 / -1; }
          .hero { display:grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap: 1rem; margin-top: 1rem; }
          .stat { background: var(--panel-3); border: 1px solid var(--border); border-radius: 12px; padding: 0.9rem; }
          .stat .label { color: var(--muted); font-size: 0.85rem; margin-bottom: 0.35rem; }
          .stat .value { font-size: 1.2rem; font-weight: 700; }
          .pill { display: inline-block; padding: 0.2rem 0.55rem; border-radius: 999px; background: var(--panel-2); border: 1px solid var(--border); font-size: 0.9rem; }
          .status-ok { color: var(--ok); }
          .status-warn { color: var(--warn); }
          .status-bad { color: var(--bad); }
          table { width: 100%; border-collapse: collapse; }
          th, td { padding: 0.6rem; border-bottom: 1px solid var(--border); text-align: left; font-size: 0.93rem; vertical-align: top; }
          .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92rem; }
          .small { font-size: 0.88rem; }
          .compare-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap:0.75rem; }
          .compare-item { background: var(--panel-3); border:1px solid var(--border); border-radius: 10px; padding: 0.75rem; }
          .chart-container { position: relative; height: 350px; width: 100%; background: var(--panel-3); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; padding-bottom: 2rem; box-sizing: border-box; }
        </style>
      </head>
      <body>
        <h1>Inference Benchmark Dashboard</h1>
        <p class="muted">Live benchmark status, latest results, and quick comparisons. Use <code>http://</code>, not <code>https://</code>, unless this is behind a TLS reverse proxy.</p>
        <p class="muted" id="filter-note" style="display:none"></p>

        <div class="hero">
          <div class="stat">
            <div class="label">Current model</div>
            <div class="value" id="hero-model">—</div>
          </div>
          <div class="stat">
            <div class="label">Current test</div>
            <div class="value" id="hero-scenario">—</div>
          </div>
          <div class="stat">
            <div class="label">GPU status</div>
            <div class="value" id="hero-gpu">—</div>
          </div>
          <div class="stat">
            <div class="label">Latest result</div>
            <div class="value" id="hero-latest">—</div>
          </div>
        </div>

        <div class="grid">
          <div class="card">
            <h2>Current activity</h2>
            <div id="current">Loading…</div>
          </div>
          <div class="card">
            <h2>Machine + services</h2>
            <div id="services">Loading…</div>
          </div>
          <div class="card">
            <h2>Latest completed result</h2>
            <div id="latest-result">Loading…</div>
          </div>
          <div class="card wide">
            <h2>Latest result files</h2>
            <div id="results">Loading…</div>
          </div>
          <div class="card wide">
            <h2>Visualizations</h2>
            <div class="compare-grid" style="gap: 1.5rem">
              <div class="chart-container"><canvas id="latencyChart"></canvas></div>
              <div class="chart-container"><canvas id="throughputChart"></canvas></div>
            </div>
          </div>
          <div class="card">
            <h2>single_request_latency compare</h2>
            <div id="compare-latency">Loading…</div>
          </div>
          <div class="card">
            <h2>throughput_ramp compare</h2>
            <div id="compare-throughput">Loading…</div>
          </div>
          <div class="card">
            <h2>Quick links</h2>
            <ul>
              <li><a href="/docs">API docs</a></li>
              <li><a href="/api/current">Current activity JSON</a></li>
              <li><a href="/api/results">Raw results JSON</a></li>
              <li><a href="/api/compare/single_request_latency">Latency compare JSON</a></li>
              <li><a href="/api/compare/throughput_ramp">Throughput compare JSON</a></li>
            </ul>
          </div>
        </div>

        <script>
          function esc(value) {
            return String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
          }
          function fmtMs(value) { return value == null ? '—' : `${Number(value).toFixed(1)} ms`; }
          function fmtNum(value, digits = 1) { return value == null ? '—' : Number(value).toFixed(digits); }
          const urlModel = new URLSearchParams(window.location.search).get('model');

          function withModel(url) {
            if (!urlModel) return url;
            const sep = url.includes('?') ? '&' : '?';
            return `${url}${sep}model=${encodeURIComponent(urlModel)}`;
          }

          function statusClass(state) {
            if (state === 'running') return 'status-ok';
            if (state === 'queued_or_cooldown') return 'status-warn';
            return 'status-bad';
          }

          function renderFilterNote(payload) {
            const node = document.getElementById('filter-note');
            const model = payload.result_filter_model || urlModel;
            if (!model) {
              node.style.display = 'none';
              node.textContent = '';
              return;
            }
            node.style.display = 'block';
            node.innerHTML = `Filtering result summaries and comparisons to <code>${esc(model)}</code>.`;
          }

          function renderHero(payload) {
            const current = payload.current || {};
            const system = payload.system || {};
            const latest = payload.latest_completed_result || {};
            const filterModel = payload.result_filter_model || null;
            document.getElementById('hero-model').textContent = current.model || filterModel || 'Idle';
            document.getElementById('hero-scenario').textContent = current.scenario || current.state || 'idle';
            if (system.gpu) {
              document.getElementById('hero-gpu').textContent = `${system.gpu.utilization_pct}% • ${system.gpu.temperature_c}°C`;
            } else {
              document.getElementById('hero-gpu').textContent = 'n/a';
            }
            document.getElementById('hero-latest').textContent = latest.model ? `${latest.model}` : 'No results';
          }

          function renderCurrent(payload) {
            const current = payload.current || { state: 'idle' };
            const parts = [];
            parts.push(`<p><span class="pill ${statusClass(current.state)}">${esc(current.state)}</span></p>`);
            if (current.model) parts.push(`<p><strong>Model:</strong> ${esc(current.model)}</p>`);
            if (current.engine) parts.push(`<p><strong>Engine:</strong> ${esc(current.engine)}</p>`);
            if (current.scenario) parts.push(`<p><strong>Scenario:</strong> ${esc(current.scenario)}</p>`);
            if (current.prompt_pack) parts.push(`<p><strong>Prompt pack:</strong> ${esc(current.prompt_pack)}</p>`);
            if (current.progress != null && current.total) parts.push(`<p><strong>Progress:</strong> ${current.progress}/${current.total}</p>`);
            if (current.script) parts.push(`<p><strong>Runner script:</strong> <span class="mono">${esc(current.script)}</span></p>`);
            if (current.raw_command) parts.push(`<p class="small muted"><strong>Command:</strong> ${esc(current.raw_command)}</p>`);
            if (current.state === 'idle') parts.push('<p class="muted">No active benchmark process detected right now.</p>');
            document.getElementById('current').innerHTML = parts.join('');
          }

          function renderServices(payload) {
            const parts = [];
            const system = payload.system || {};
            if (system.gpu) {
              parts.push(`<p><strong>GPU:</strong> ${system.gpu.utilization_pct}% util, ${system.gpu.memory_used_mib}/${system.gpu.memory_total_mib} MiB, ${system.gpu.temperature_c}°C</p>`);
            }
            if (system.load_avg) {
              parts.push(`<p><strong>Load avg:</strong> ${esc(system.load_avg)}</p>`);
            }
            if (system.memory) {
              parts.push(`<p><strong>RAM:</strong> ${fmtNum(system.memory.used_mib, 0)} MiB used, ${fmtNum(system.memory.available_mib, 0)} MiB available</p>`);
            }
            const services = payload.active_servers || [];
            if (services.length) {
              parts.push('<ul>' + services.map(item => `<li><strong>${esc(item.engine)}</strong>${item.model ? ` — ${esc(item.model)}` : ''}</li>`).join('') + '</ul>');
            } else {
              parts.push('<p class="muted">No engine/dashboard processes detected.</p>');
            }
            document.getElementById('services').innerHTML = parts.join('');
          }

          function renderLatestResult(item) {
            if (!item) {
              document.getElementById('latest-result').innerHTML = '<p class="muted">No completed result file found yet.</p>';
              return;
            }
            document.getElementById('latest-result').innerHTML = `
              <p><strong>${esc(item.model || 'Unknown model')}</strong></p>
              <p>${esc(item.engine || '—')} • ${esc(item.scenario || '—')}</p>
              <p><strong>TTFT p95:</strong> ${fmtMs(item.ttft_p95_ms)}</p>
              <p><strong>Latency p95:</strong> ${fmtMs(item.latency_p95_ms)}</p>
              <p><strong>Tok/s:</strong> ${fmtNum(item.tokens_per_sec)}</p>
              <p><strong>Req/s:</strong> ${fmtNum(item.requests_per_sec, 2)}</p>
              <p><strong>Success:</strong> ${fmtNum(item.success_rate_pct, 1)}%</p>
              <p class="small"><a href="/api/results/${encodeURIComponent(item.id)}">Open JSON</a></p>
            `;
          }

          function renderResults(results) {
            if (!results.length) {
              document.getElementById('results').innerHTML = '<p class="muted">No result files yet.</p>';
              return;
            }
            const rows = results.filter(item => item.type === 'result').slice(0, 12).map(item => `
              <tr>
                <td><a href="/api/results/${encodeURIComponent(item.id)}">${esc(item.model || item.filename)}</a></td>
                <td>${esc(item.engine || '—')}</td>
                <td>${esc(item.scenario || '—')}</td>
                <td>${fmtMs(item.ttft_p95_ms)}</td>
                <td>${fmtNum(item.tokens_per_sec)}</td>
                <td>${fmtNum(item.success_rate_pct, 1)}%</td>
                <td>${new Date(item.modified * 1000).toLocaleString()}</td>
              </tr>
            `).join('');
            document.getElementById('results').innerHTML = `
              <table>
                <thead><tr><th>Model</th><th>Engine</th><th>Scenario</th><th>TTFT p95</th><th>Tok/s</th><th>Success</th><th>Modified</th></tr></thead>
                <tbody>${rows}</tbody>
              </table>
            `;
          }

          function renderCompare(targetId, payload, emptyText) {
            if (!payload || (!payload.vllm && !payload.sglang)) {
              document.getElementById(targetId).innerHTML = `<p class="muted">${emptyText}</p>`;
              return;
            }
            const chunks = [];
            if (payload.vllm) {
              chunks.push(`<div class="compare-item"><strong>vLLM</strong><br>TTFT p95: ${fmtMs(payload.vllm.ttft?.p95)}<br>Tok/s: ${fmtNum(payload.vllm.throughput?.tokens_per_sec)}<br>Req/s: ${fmtNum(payload.vllm.throughput?.requests_per_sec, 2)}</div>`);
            }
            if (payload.sglang) {
              chunks.push(`<div class="compare-item"><strong>SGLang</strong><br>TTFT p95: ${fmtMs(payload.sglang.ttft?.p95)}<br>Tok/s: ${fmtNum(payload.sglang.throughput?.tokens_per_sec)}<br>Req/s: ${fmtNum(payload.sglang.throughput?.requests_per_sec, 2)}</div>`);
            }
            if (payload.delta) {
              chunks.push(`<div class="compare-item"><strong>Delta</strong><br>TTFT p95 Δ: ${esc(payload.delta.ttft_p95_pct)}%<br>Tok/s Δ: ${esc(payload.delta.tokens_per_sec_pct)}%</div>`);
            }
            const modelNote = payload.model ? `<p class="small muted"><strong>Model:</strong> ${esc(payload.model)}</p>` : '';
            document.getElementById(targetId).innerHTML = `${modelNote}<div class="compare-grid">${chunks.join('')}</div>`;
          }

          async function loadAll() {
            try {
              const [currentRes, resultsRes, latencyRes, throughputRes] = await Promise.all([
                fetch(withModel('/api/current')),
                fetch(withModel('/api/results')),
                fetch(withModel('/api/compare/single_request_latency')).catch(() => null),
                fetch(withModel('/api/compare/throughput_ramp')).catch(() => null),
              ]);
              const current = await currentRes.json();
              const results = await resultsRes.json();
              renderFilterNote(current);
              renderHero(current);
              renderCurrent(current);
              renderServices(current);
              renderLatestResult(current.latest_completed_result);
              renderResults(results);

              let latency = null;
              let throughput = null;
              try { if (latencyRes && latencyRes.ok) latency = await latencyRes.json(); } catch (_) {}
              try { if (throughputRes && throughputRes.ok) throughput = await throughputRes.json(); } catch (_) {}
              renderCompare('compare-latency', latency, 'No latency comparison yet.');
              renderCompare('compare-throughput', throughput, 'No throughput comparison yet.');
              renderCharts(results);
            } catch (err) {
              document.getElementById('current').innerHTML = `<pre>Failed to load dashboard data: ${esc(err)}</pre>`;
            }
          }

          let latencyChartInstance = null;
          let throughputChartInstance = null;

          function renderCharts(results) {
            const chartData = results.filter(r => r.type === 'result');
            const latencyData = chartData.filter(r => r.scenario === 'single_request_latency' && r.ttft_p95_ms);
            const tpsData = chartData.filter(r => r.scenario === 'throughput_ramp' && r.tokens_per_sec);
            const models = [...new Set(chartData.map(r => r.model).filter(Boolean))];

            const ctxL = document.getElementById('latencyChart');
            if (ctxL && models.length > 0) {
              if (latencyChartInstance) latencyChartInstance.destroy();
              latencyChartInstance = new Chart(ctxL, {
                type: 'bar',
                data: {
                  labels: models,
                  datasets: [
                    { label: 'vLLM', backgroundColor: '#5B8DEF', data: models.map(m => latencyData.find(d => d.model === m && d.engine === 'vLLM')?.ttft_p95_ms || 0) },
                    { label: 'SGLang', backgroundColor: '#F5A524', data: models.map(m => latencyData.find(d => d.model === m && d.engine === 'SGLang')?.ttft_p95_ms || 0) }
                  ]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { title: { display: true, text: 'Single Request TTFT p95 (ms) [Lower is Better]', color: '#e8eefc' }, legend: { labels: { color: '#e8eefc' } } }, scales: { y: { ticks: { color: '#9fb0d0' }, grid: { color: '#27335a' } }, x: { ticks: { color: '#9fb0d0' }, grid: { color: '#27335a' } } } }
              });
            }

            const ctxT = document.getElementById('throughputChart');
            if (ctxT && models.length > 0) {
              if (throughputChartInstance) throughputChartInstance.destroy();
              throughputChartInstance = new Chart(ctxT, {
                type: 'bar',
                data: {
                  labels: models,
                  datasets: [
                    { label: 'vLLM', backgroundColor: '#5B8DEF', data: models.map(m => tpsData.find(d => d.model === m && d.engine === 'vLLM')?.tokens_per_sec || 0) },
                    { label: 'SGLang', backgroundColor: '#F5A524', data: models.map(m => tpsData.find(d => d.model === m && d.engine === 'SGLang')?.tokens_per_sec || 0) }
                  ]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { title: { display: true, text: 'Throughput (Tokens / sec) [Higher is Better]', color: '#e8eefc' }, legend: { labels: { color: '#e8eefc' } } }, scales: { y: { ticks: { color: '#9fb0d0' }, grid: { color: '#27335a' } }, x: { ticks: { color: '#9fb0d0' }, grid: { color: '#27335a' } } } }
              });
            }
          }

          loadAll();
          setInterval(loadAll, 5000);
        </script>
      </body>
    </html>
    """
    return HTMLResponse(html)


@app.get("/api/results")
async def list_results(
    model: str | None = Query(default=None, description="Optional model filter"),
) -> JSONResponse:
    """List saved benchmark result files, optionally filtered by model."""
    return JSONResponse(_latest_results_payload(limit=200, model=model))


@app.get("/api/results/{result_id}")
async def get_result(result_id: str) -> JSONResponse:
    """Load and return a specific result by its stem (filename without .json)."""
    if not _SAFE_NAME_RE.match(result_id):
        raise HTTPException(status_code=400, detail="Invalid result ID")
    candidates = list(RESULTS_DIR.glob(f"{result_id}*.json"))
    if not candidates:
        raise HTTPException(status_code=404, detail=f"Result '{result_id}' not found")
    path = candidates[0]
    if not path.resolve().is_relative_to(RESULTS_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid path")
    return JSONResponse(json.loads(path.read_text()))


@app.get("/api/current")
async def current_activity(
    model: str | None = Query(
        default=None, description="Optional model filter for result summaries"
    ),
) -> JSONResponse:
    """Return active benchmark info and optionally filtered result summaries."""
    return JSONResponse(_current_activity_payload(model=model))


@app.get("/api/compare/{scenario}")
async def compare_scenario(
    scenario: str,
    model: str | None = Query(default=None, description="Optional model filter"),
) -> JSONResponse:
    """Load model-consistent vLLM and SGLang results for the given scenario and compare them."""
    scenario_results = [r for r in load_results(RESULTS_DIR) if r.get("scenario_name") == scenario]

    selected_model: str | None = None
    available_models: list[str] = []
    if scenario_results:
        try:
            selected_model, scenario_results, selection_meta = select_model_results(
                scenario_results,
                preferred_model=model,
                require_engines={"VLLMClient", "SGLangClient"},
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        available_models = selection_meta.get("available_models", [])

        vllm_data = _latest_engine_result(scenario_results, "VLLMClient")
        sglang_data = _latest_engine_result(scenario_results, "SGLangClient")

        response: dict[str, Any] = {
            "scenario": scenario,
            "model": selected_model,
            "available_models": available_models,
        }
        if vllm_data:
            response["vllm"] = vllm_data.get("metrics", {})
        if sglang_data:
            response["sglang"] = sglang_data.get("metrics", {})

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

    comp_files = sorted(
        RESULTS_DIR.glob(f"comparison_{scenario}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    comparison_payloads: list[dict[str, Any]] = []
    for path in comp_files:
        try:
            comparison_payloads.append(json.loads(path.read_text()))
        except Exception:
            continue

    if model is not None:
        comparison_payloads = [
            payload for payload in comparison_payloads if _comparison_file_model(payload) == model
        ]

    if comparison_payloads:
        selected = comparison_payloads[0]
        selected_model = _comparison_file_model(selected)
        selected.setdefault("model", selected_model)
        return JSONResponse(selected)

    raise HTTPException(status_code=404, detail=f"No results found for scenario '{scenario}'")


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
    async with _state_lock:
        _jobs[job_id] = job
    background_tasks.add_task(_run_benchmark_job, job_id, req)
    return JSONResponse({"job_id": job_id, "status": "pending"}, status_code=202)


@app.get("/api/run/{job_id}/status")
async def get_run_status(job_id: str) -> JSONResponse:
    """Poll the status of a background benchmark run."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JSONResponse(_jobs[job_id].model_dump())


@app.websocket("/ws/live")
async def live_metrics_ws(websocket: WebSocket) -> None:
    """WebSocket endpoint that streams real-time engine metrics and run progress."""
    await websocket.accept()
    async with _state_lock:
        _live_connections.add(websocket)
    try:
        while True:
            await asyncio.sleep(1)
            await websocket.send_json({"type": "heartbeat", "ts": time.time()})
    except WebSocketDisconnect:
        pass
    finally:
        async with _state_lock:
            _live_connections.discard(websocket)


# ---------------------------------------------------------------------------
# Background job runner
# ---------------------------------------------------------------------------


async def _broadcast(message: dict[str, Any]) -> None:
    """Push a message to all active WebSocket clients."""
    async with _state_lock:
        dead: set[WebSocket] = set()
        for ws in list(_live_connections):
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)
        _live_connections.difference_update(dead)


async def _run_benchmark_job(job_id: str, req: RunRequest) -> None:
    """Background task: run the requested scenario and update job status."""
    from benchmarks.runner import BenchmarkRunner
    from benchmarks.scenarios import SCENARIOS
    from engines.sglang_client import SGLangClient
    from engines.vllm_client import VLLMClient

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

    def _progress_cb(done: int, total: int, result: Any) -> None:
        job.progress = done
        asyncio.create_task(
            _broadcast(
                {
                    "type": "progress",
                    "data": {
                        "job_id": job_id,
                        "done": done,
                        "total": total,
                        "last_ttft_ms": result.ttft_ms,
                        "last_success": result.success,
                    },
                }
            )
        )

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

            await _broadcast(
                {
                    "type": "metrics",
                    "data": {
                        "engine": engine_name,
                        "ttft_p95": results.metrics.ttft.p95,
                        "tokens_per_sec": results.metrics.throughput.tokens_per_sec,
                        "kv_cache_timeline": results.metrics.kv_cache_timeline[-10:],
                    },
                }
            )

            await client.aclose()

        job.status = "done"
        job.finished_at = time.time()
        await _broadcast({"type": "done", "data": job.model_dump()})
        logger.info("job completed", job_id=job_id, paths=job.result_paths)

    except Exception as exc:  # pragma: no cover - live-path defensive branch
        logger.exception("job failed", job_id=job_id, error=str(exc))
        job.status = "error"
        job.message = str(exc)
        job.finished_at = time.time()
        await _broadcast({"type": "error", "data": job.model_dump()})


# ---------------------------------------------------------------------------
# Misc helpers / health
# ---------------------------------------------------------------------------


def _estimate_requests(scenario_name: str) -> int:
    from benchmarks.scenarios import SCENARIOS

    scenario = SCENARIOS.get(scenario_name)
    if scenario is None:
        return 100
    return getattr(scenario, "num_requests", getattr(scenario, "requests_per_level", 100))


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
