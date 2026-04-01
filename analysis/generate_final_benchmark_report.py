from __future__ import annotations

import html
import json
import re
from collections.abc import Iterable
from pathlib import Path

REPORT_DATE = "2026-03-31"
RESULTS_DIR = Path("results")
RESULT_ROOTS = [RESULTS_DIR]
OUTPUT_DIR = Path("reports")
FIGURES_DIR = OUTPUT_DIR / "figures"

TARGET_MODELS = [
    {"id": "google/gemma-2-2b-it", "dir": "gemma-2-2b-it", "name": "Gemma 2 2B", "size_b": 2},
    {"id": "HuggingFaceTB/SmolLM3-3B", "dir": "smollm3-3b", "name": "SmolLM3 3B", "size_b": 3},
    {"id": "meta-llama/Llama-3.2-3B-Instruct", "dir": "llama-3-2-3b-instruct", "name": "Llama 3.2 3B", "size_b": 3},
    {"id": "microsoft/Phi-3-mini-4k-instruct", "dir": "phi-3-mini-4k-instruct", "name": "Phi-3 mini", "size_b": 4},
    {"id": "microsoft/Phi-4-mini-instruct", "dir": "phi-4-mini-instruct", "name": "Phi-4 mini", "size_b": 4},
    {"id": "google/gemma-3-4b-it", "dir": "gemma-3-4b-it", "name": "Gemma 3 4B", "size_b": 4},
    {"id": "Qwen/Qwen2.5-7B-Instruct", "dir": "qwen2-5-7b-instruct", "name": "Qwen 2.5 7B", "size_b": 7},
    {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "dir": "deepseek-r1-distill-qwen-7b", "name": "DS-R1 Qwen 7B", "size_b": 7},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3", "dir": "mistral-7b-instruct-v0-3", "name": "Mistral 7B", "size_b": 7},
    {"id": "meta-llama/Llama-3.1-8B-Instruct", "dir": "llama-3-1-8b-instruct", "name": "Llama 3.1 8B", "size_b": 8},
    {"id": "Qwen/Qwen3-8B", "dir": "qwen3-8b", "name": "Qwen3 8B", "size_b": 8},
    {"id": "ibm-granite/granite-3.3-8b-instruct", "dir": "granite-3-3-8b-instruct", "name": "Granite 8B", "size_b": 8},
    {"id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "dir": "deepseek-r1-distill-llama-8b", "name": "DS-R1 Llama 8B", "size_b": 8},
    {"id": "google/gemma-2-9b-it", "dir": "gemma-2-9b-it", "name": "Gemma 2 9B", "size_b": 9},
    {"id": "google/gemma-3-12b-it", "dir": "gemma-3-12b-it", "name": "Gemma 3 12B", "size_b": 12},
    {"id": "Qwen/Qwen3-30B-A3B", "dir": "qwen3-30b-a3b", "name": "Qwen3 30B-A3B", "size_b": 30},
]
TARGET_MODEL_MAP = {entry["id"]: entry for entry in TARGET_MODELS}
DIR_NAME_TO_MODEL_ID = {entry["dir"]: entry["id"] for entry in TARGET_MODELS}
MODEL_ORDER = [entry["name"] for entry in TARGET_MODELS]
SCENARIO_ORDER = ["single_request_latency", "throughput_ramp"]
ENGINE_ORDER = ["vLLM", "SGLang"]
ENGINE_LABELS = {"VLLMClient": "vLLM", "SGLangClient": "SGLang"}
COLORS = {"vLLM": "#5B8DEF", "SGLang": "#F5A524"}

MODEL_NOTES = {
    "google/gemma-2-9b-it": [
        "vLLM required tuned launch settings on the single A10G: `max_model_len=2048`, `gpu_memory_utilization=0.95`, `--disable-frontend-multiprocessing`, and `--enforce-eager`.",
    ],
}


def _scenario_rank(name: str) -> int:
    return SCENARIO_ORDER.index(name) if name in SCENARIO_ORDER else len(SCENARIO_ORDER)


def _engine_rank(name: str) -> int:
    return ENGINE_ORDER.index(name) if name in ENGINE_ORDER else len(ENGINE_ORDER)


def _safe_float(value: float | int | None) -> float | None:
    return None if value is None else float(value)


def _load_result_model_map_from_logs() -> dict[str, str]:
    """Map result file names to model IDs by parsing matrix log files."""
    mapping: dict[str, str] = {}
    pattern = re.compile(r"path=(results/[^\s]+\.json)")
    done_pattern = re.compile(r"results/[^\s]+\.json")

    log_paths: list[Path] = []
    for root in RESULT_ROOTS:
        if root.exists():
            log_paths.extend(sorted(root.rglob("*.log")))

    for log_path in log_paths:
        current_model: str | None = None
        for raw_line in log_path.read_text(errors="ignore").splitlines():
            line = raw_line.strip()
            if line.startswith("===== MODEL:") and line.endswith("====="):
                current_model = line.removeprefix("===== MODEL:").removesuffix("=====").strip()
                continue
            if not current_model:
                continue

            match = pattern.search(line)
            if match:
                mapping[Path(match.group(1)).name] = current_model
                continue

            # Fallback for summary lines such as: "✓ sglang done → results/...json"
            if "results/" in line and line.endswith(".json"):
                maybe = done_pattern.search(line)
                if maybe:
                    mapping[Path(maybe.group(0)).name] = current_model

    return mapping


def _load_snapshot_hints() -> list[dict]:
    """Load prior snapshot rows as model-id hints for legacy result files."""
    hints: list[dict] = []
    for snapshot in sorted(OUTPUT_DIR.glob("benchmark_snapshot_*.json")):
        try:
            rows = json.loads(snapshot.read_text())
        except Exception:
            continue
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and row.get("model_id") in TARGET_MODEL_MAP:
                    hints.append(row)
    return hints


def _infer_model_from_snapshot_hints(data: dict, snapshot_hints: list[dict]) -> str | None:
    scenario = data.get("scenario_name")
    engine = ENGINE_LABELS.get(data.get("engine_name"), data.get("engine_name", "unknown"))
    metrics = data.get("metrics", {})
    ttft_p95 = _safe_float(metrics.get("ttft", {}).get("p95"))
    latency_p95 = _safe_float(metrics.get("latency", {}).get("p95"))

    candidates = [
        row
        for row in snapshot_hints
        if row.get("scenario") == scenario and row.get("engine") == engine
    ]
    if not candidates or ttft_p95 is None or latency_p95 is None:
        return None

    scored: list[tuple[float, str]] = []
    for row in candidates:
        row_ttft = _safe_float(row.get("ttft_p95"))
        row_latency = _safe_float(row.get("latency_p95"))
        if row_ttft is None or row_latency is None:
            continue
        dist = abs(row_ttft - ttft_p95) + abs(row_latency - latency_p95)
        scored.append((dist, row["model_id"]))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0])
    # Require a close match to avoid accidental cross-model assignment.
    return scored[0][1] if scored[0][0] <= 5.0 else None


def _extract_model_id(
    data: dict,
    path: Path,
    model_map_from_logs: dict[str, str],
    snapshot_hints: list[dict],
) -> str | None:
    run_metadata = data.get("run_metadata") or {}
    scenario_config = data.get("scenario_config") or {}

    candidates = [
        run_metadata.get("model"),
        run_metadata.get("model_id"),
        scenario_config.get("model"),
        scenario_config.get("model_id"),
        data.get("model"),
        data.get("model_id"),
        model_map_from_logs.get(path.name),
        DIR_NAME_TO_MODEL_ID.get(path.parent.name),
    ]
    for candidate in candidates:
        if candidate in TARGET_MODEL_MAP:
            return candidate

    return _infer_model_from_snapshot_hints(data, snapshot_hints)


def _normalized_throughput(data: dict, metrics: dict) -> tuple[float, float, float]:
    """Return (tokens_per_sec, requests_per_sec, wall_time_sec) normalized by full run wall-clock time."""
    throughput = metrics.get("throughput", {})

    successful_requests = int(throughput.get("successful_requests", 0) or 0)
    if successful_requests <= 0:
        successful_requests = sum(
            1
            for req in data.get("requests", [])
            if isinstance(req, dict) and req.get("success", False)
        )

    total_tokens = float(throughput.get("total_tokens_generated", 0.0) or 0.0)
    if total_tokens <= 0:
        total_tokens = float(
            sum(
                int(req.get("output_tokens", 0) or 0)
                for req in data.get("requests", [])
                if isinstance(req, dict) and req.get("success", False)
            )
        )

    wall_time_sec = 0.0
    timeline = data.get("engine_metrics_timeline") or []
    if len(timeline) >= 2:
        timestamps = [
            _safe_float(entry.get("timestamp"))
            for entry in timeline
            if isinstance(entry, dict) and entry.get("timestamp") is not None
        ]
        timestamps = [t for t in timestamps if t is not None]
        if len(timestamps) >= 2:
            wall_time_sec = max(timestamps) - min(timestamps)

    if wall_time_sec <= 0 and data.get("scenario_name") == "single_request_latency":
        wall_time_sec = (
            sum(
                float(req.get("total_ms", 0.0) or 0.0)
                for req in data.get("requests", [])
                if isinstance(req, dict) and req.get("success", True)
            )
            / 1000.0
        )

    if wall_time_sec <= 0:
        wall_time_sec = float(throughput.get("wall_time_sec", 0.0) or 0.0)

    if wall_time_sec <= 0:
        req_times = [
            float(req.get("total_ms", 0.0) or 0.0)
            for req in data.get("requests", [])
            if isinstance(req, dict)
        ]
        if req_times:
            wall_time_sec = max(req_times) / 1000.0

    if wall_time_sec > 0 and successful_requests > 0:
        return (
            total_tokens / wall_time_sec,
            successful_requests / wall_time_sec,
            wall_time_sec,
        )

    return (
        float(throughput.get("tokens_per_sec", 0.0) or 0.0),
        float(throughput.get("requests_per_sec", 0.0) or 0.0),
        wall_time_sec,
    )


def load_latest_rows() -> list[dict]:
    latest: dict[tuple[str, str, str], tuple[float, dict]] = {}
    model_map_from_logs = _load_result_model_map_from_logs()
    snapshot_hints = _load_snapshot_hints()

    result_paths: list[Path] = []
    for root in RESULT_ROOTS:
        if root.exists():
            result_paths.extend(sorted(root.rglob("*.json")))

    for path in result_paths:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not {"scenario_name", "engine_name", "metrics"}.issubset(data):
            continue

        model_id = _extract_model_id(data, path, model_map_from_logs, snapshot_hints)
        if model_id not in TARGET_MODEL_MAP:
            continue

        engine = ENGINE_LABELS.get(data.get("engine_name"), data.get("engine_name", "unknown"))
        scenario = data.get("scenario_name")
        timestamp = float(data.get("timestamp", path.stat().st_mtime))
        key = (model_id, scenario, engine)
        prev = latest.get(key)
        if prev is None or timestamp > prev[0]:
            latest[key] = (timestamp, {"path": str(path), "payload": data})

    rows: list[dict] = []
    for (model_id, scenario, engine), (_, container) in latest.items():
        data = container["payload"]
        metrics = data["metrics"]
        tokens_per_sec, requests_per_sec, wall_time_sec = _normalized_throughput(data, metrics)

        row = {
            "model_id": model_id,
            "model": TARGET_MODEL_MAP[model_id]["name"],
            "size_b": TARGET_MODEL_MAP[model_id]["size_b"],
            "scenario": scenario,
            "engine": engine,
            "ttft_p50": _safe_float(metrics.get("ttft", {}).get("p50")),
            "ttft_p95": _safe_float(metrics.get("ttft", {}).get("p95")),
            "ttft_p99": _safe_float(metrics.get("ttft", {}).get("p99")),
            "latency_p95": _safe_float(metrics.get("latency", {}).get("p95")),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "requests_per_sec": round(requests_per_sec, 2),
            "success_pct": round((1 - float(metrics.get("error_rate", 0.0))) * 100, 1),
            "wall_time_sec": round(wall_time_sec, 3),
            "path": container["path"],
        }
        rows.append(row)

    rows.sort(key=lambda r: (r["size_b"], _scenario_rank(r["scenario"]), _engine_rank(r["engine"])))
    return rows


def rows_for(
    rows: Iterable[dict], *, scenario: str | None = None, model_id: str | None = None
) -> list[dict]:
    data = list(rows)
    if scenario is not None:
        data = [r for r in data if r["scenario"] == scenario]
    if model_id is not None:
        data = [r for r in data if r["model_id"] == model_id]
    return data


def best_by(
    rows: Iterable[dict], metric: str, *, scenario: str, lower_is_better: bool = False
) -> dict | None:
    candidates = [r for r in rows if r["scenario"] == scenario and r.get(metric) is not None]
    if not candidates:
        return None
    return sorted(candidates, key=lambda r: r[metric], reverse=not lower_is_better)[0]


def grouped_metric(rows: Iterable[dict], scenario: str, metric: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {model: {} for model in MODEL_ORDER}
    for row in rows:
        if row["scenario"] != scenario:
            continue
        value = row.get(metric)
        if value is not None:
            model_name = row["model"]
            if model_name not in grouped:
                grouped[model_name] = {}
            grouped[model_name][row["engine"]] = value
    return grouped


def render_grouped_bar_svg(
    title: str,
    subtitle: str,
    grouped: dict[str, dict[str, float]],
    y_label: str,
    output: Path,
    *,
    lower_is_better: bool = False,
) -> None:
    # Only include models that have data
    active_models = [m for m in MODEL_ORDER if grouped.get(m)]
    n_models = max(len(active_models), 1)

    # Dynamic width: ~110px per model, min 900, max 1800
    width = max(900, min(1800, 120 + n_models * 110))
    height = 720
    left, right, top, bottom = 100, 40, 90, 150
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_val = (
        max((value for m in active_models for value in grouped[m].values()), default=1.0)
        * 1.18
    )
    ticks = 5
    group_w = plot_w / n_models
    bar_w = min(group_w * 0.30, 36)

    svg: list[str] = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    svg.append(
        "<style>text{font-family:system-ui,-apple-system,sans-serif;fill:#e8eefc}"
        ".muted{fill:#9fb0d0}.small{font-size:11px}.label{font-size:13px}"
        ".title{font-size:26px;font-weight:700}.subtitle{font-size:13px}"
        ".axis{stroke:#5c6b91;stroke-width:1}.grid{stroke:#27335a;stroke-width:1}"
        ".value{font-size:10px;font-weight:700}.legend{font-size:13px}"
        ".xlbl{font-size:12px}</style>"
    )
    svg.append(f'<rect width="{width}" height="{height}" fill="#0b1020"/>')
    svg.append(f'<text x="{left}" y="40" class="title">{html.escape(title)}</text>')
    svg.append(f'<text x="{left}" y="64" class="subtitle muted">{html.escape(subtitle)}</text>')

    for i in range(ticks + 1):
        y = top + plot_h - (plot_h * i / ticks)
        value = max_val * i / ticks
        svg.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid"/>'
        )
        svg.append(
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="small muted">{value:.0f}</text>'
        )

    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" class="axis"/>')
    svg.append(
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" class="axis"/>'
    )

    for gi, model in enumerate(active_models):
        gx = left + gi * group_w + group_w * 0.15
        bar_gap = group_w * 0.06
        for ei, engine in enumerate(ENGINE_ORDER):
            if engine not in grouped[model]:
                continue
            value = grouped[model][engine]
            bh = (value / max_val) * plot_h if max_val else 0
            x = gx + ei * (bar_w + bar_gap)
            y = top + plot_h - bh
            svg.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="4" fill="{COLORS[engine]}"/>'
            )
            # Value label above bar
            fmt = f"{value:.0f}" if value >= 100 else f"{value:.1f}"
            svg.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" class="value">{fmt}</text>'
            )
        # Angled model name label
        label_x = gx + (bar_w + bar_gap) * 0.5 + bar_w * 0.5
        label_y = height - bottom + 16
        svg.append(
            f'<text x="{label_x:.1f}" y="{label_y}" text-anchor="end" '
            f'transform="rotate(-35 {label_x:.1f} {label_y})" class="xlbl">'
            f'{html.escape(model)}</text>'
        )

    lx = width - right - 180
    ly = 44
    for idx, engine in enumerate(ENGINE_ORDER):
        yy = ly + idx * 22
        svg.append(
            f'<rect x="{lx}" y="{yy - 10}" width="14" height="14" rx="3" fill="{COLORS[engine]}"/>'
        )
        svg.append(f'<text x="{lx + 24}" y="{yy + 1}" class="legend">{engine}</text>')

    svg.append(
        f'<text x="24" y="{top + plot_h / 2:.1f}" transform="rotate(-90 24 {top + plot_h / 2:.1f})" class="label muted">{html.escape(y_label)}</text>'
    )
    direction = "Lower is better \u2193" if lower_is_better else "Higher is better \u2191"
    svg.append(
        f'<text x="{left}" y="{height - 16}" class="small muted">{direction} \u2022 {n_models} models with data</text>'
    )
    svg.append("</svg>")
    output.write_text("".join(svg))


def render_scatter_svg(rows: Iterable[dict], output: Path) -> None:
    data = [
        r
        for r in rows
        if r["scenario"] == "throughput_ramp"
        and r.get("latency_p95") is not None
        and r.get("tokens_per_sec") is not None
    ]
    data.sort(key=lambda r: (r["size_b"], _engine_rank(r["engine"]), r["model"]))

    # Dynamic height: accommodate legend entries (~22px each)
    legend_rows = len(data)
    min_legend_h = legend_rows * 22 + 120
    plot_min_h = 500
    height = max(860, 96 + 90 + max(plot_min_h, min_legend_h))

    width = 1520
    left, right, top, bottom = 110, 380, 96, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_x_raw = min((r["latency_p95"] for r in data), default=0.0)
    max_x_raw = max((r["latency_p95"] for r in data), default=1.0)
    min_x = max(0.0, min_x_raw * 0.9)
    max_x = max_x_raw * 1.08 if max_x_raw else 1.0
    min_y = 0.0
    max_y_raw = max((r["tokens_per_sec"] for r in data), default=1.0)
    max_y = max_y_raw * 1.12 if max_y_raw else 1.0

    def x_pos(v: float) -> float:
        span = max(max_x - min_x, 1.0)
        return left + ((v - min_x) / span) * plot_w

    def y_pos(v: float) -> float:
        span = max(max_y - min_y, 1.0)
        return top + plot_h - ((v - min_y) / span) * plot_h

    x_ticks = 6
    y_ticks = 6

    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    svg.append(
        "<style>text{font-family:system-ui,-apple-system,sans-serif;fill:#e8eefc}.muted{fill:#9fb0d0}.title{font-size:28px;font-weight:700}.subtitle{font-size:14px}.axis{stroke:#5c6b91;stroke-width:1}.grid{stroke:#27335a;stroke-width:1}.small{font-size:12px}.label{font-size:14px}.legend-title{font-size:16px;font-weight:700}.legend-row{font-size:12px}.point-id{font-size:11px;font-weight:700;fill:#08101e}.hint{font-size:13px;fill:#9fb0d0}</style>"
    )
    svg.append(f'<rect width="{width}" height="{height}" fill="#0b1020"/>')
    svg.append(f'<text x="{left}" y="40" class="title">Throughput tradeoff map</text>')
    svg.append(
        f'<text x="{left}" y="64" class="subtitle muted">Throughput-ramp runs only. Top-left is ideal: lower latency p95, higher tokens/sec. Numbered points map to the legend on the right.</text>'
    )

    for i in range(x_ticks + 1):
        x = left + plot_w * i / x_ticks
        xv = min_x + (max_x - min_x) * i / x_ticks
        svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" class="grid"/>')
        svg.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 24:.1f}" text-anchor="middle" class="small muted">{xv:.0f}</text>'
        )

    for i in range(y_ticks + 1):
        y = top + plot_h - (plot_h * i / y_ticks)
        yv = min_y + (max_y - min_y) * i / y_ticks
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        svg.append(
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="small muted">{yv:.0f}</text>'
        )

    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    svg.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>'
    )

    svg.append(
        f'<path d="M {left + 40} {top + 70} L {left + 10} {top + 40}" fill="none" stroke="#9fb0d0" stroke-width="1.5" stroke-dasharray="4 4"/>'
    )
    svg.append(
        f'<text x="{left + 48}" y="{top + 74}" class="hint">better</text>'
    )

    for idx, row in enumerate(data, start=1):
        x = x_pos(row["latency_p95"])
        y = y_pos(row["tokens_per_sec"])
        color = COLORS[row["engine"]]
        svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="{color}" stroke="#ffffff" stroke-width="1.8"/>'
        )
        svg.append(
            f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" class="point-id">{idx}</text>'
        )

    legend_x = left + plot_w + 28
    legend_y = top
    svg.append(
        f'<rect x="{legend_x - 18}" y="{legend_y - 26}" width="{right - 12}" height="{plot_h + 36}" rx="14" fill="#101933" stroke="#27335a" stroke-width="1"/>'
    )
    svg.append(f'<text x="{legend_x}" y="{legend_y}" class="legend-title">Legend</text>')
    svg.append(
        f'<text x="{legend_x}" y="{legend_y + 24}" class="small muted"># • model • engine • latency p95 • tok/s</text>'
    )

    for idx, row in enumerate(data, start=1):
        yy = legend_y + 52 + (idx - 1) * 28
        color = COLORS[row["engine"]]
        svg.append(
            f'<circle cx="{legend_x + 10}" cy="{yy - 4}" r="10" fill="{color}" stroke="#ffffff" stroke-width="1.3"/>'
        )
        svg.append(
            f'<text x="{legend_x + 10}" y="{yy}" text-anchor="middle" class="point-id">{idx}</text>'
        )
        label = f"{row['model']} • {row['engine']} • {row['latency_p95']:.0f} ms • {row['tokens_per_sec']:.1f} tok/s"
        svg.append(
            f'<text x="{legend_x + 30}" y="{yy}" class="legend-row">{html.escape(label)}</text>'
        )

    engine_legend_y = top + plot_h - 42
    for offset, engine in enumerate(ENGINE_ORDER):
        yy = engine_legend_y + offset * 22
        svg.append(
            f'<rect x="{legend_x}" y="{yy - 11}" width="14" height="14" rx="3" fill="{COLORS[engine]}"/>'
        )
        svg.append(f'<text x="{legend_x + 22}" y="{yy}" class="small">{engine}</text>')

    svg.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 18}" text-anchor="middle" class="label muted">Latency p95 (ms)</text>'
    )
    svg.append(
        f'<text x="26" y="{top + plot_h / 2:.1f}" transform="rotate(-90 26 {top + plot_h / 2:.1f})" class="label muted">Tokens / sec</text>'
    )
    svg.append("</svg>")
    output.write_text("".join(svg))


def render_markdown_table(rows: Iterable[dict]) -> str:
    lines = [
        "| Model | Scenario | Engine | TTFT p50 | TTFT p95 | Latency p95 | Tok/s | Req/s | Success |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | `{r['scenario']}` | {r['engine']} | {r['ttft_p50']:.1f} ms | {r['ttft_p95']:.1f} ms | {r['latency_p95']:.1f} ms | {r['tokens_per_sec']:.1f} | {r['requests_per_sec']:.2f} | {r['success_pct']:.1f}% |"
        )
    return "\n".join(lines)


def generate_takeaways(rows: Iterable[dict]) -> list[str]:
    takeaways = []
    rows = list(rows)
    for model in TARGET_MODELS:
        model_rows = rows_for(rows, model_id=model["id"])
        single = rows_for(model_rows, scenario="single_request_latency")
        throughput = rows_for(model_rows, scenario="throughput_ramp")
        notes = MODEL_NOTES.get(model["id"], [])
        lines = []
        if len(single) >= 2:
            best_single = sorted(single, key=lambda r: r["ttft_p95"])[0]
            lines.append(f"{best_single['engine']} won the single-request TTFT comparison.")
        elif len(single) == 1:
            lines.append(
                f"Only {single[0]['engine']} completed the single-request benchmark on this setup."
            )
        if len(throughput) >= 2:
            best_tps = sorted(throughput, key=lambda r: r["tokens_per_sec"], reverse=True)[0]
            best_rps = sorted(throughput, key=lambda r: r["requests_per_sec"], reverse=True)[0]
            lines.append(
                f"For throughput, {best_tps['engine']} led on tok/s while {best_rps['engine']} led on req/s."
                if best_tps["engine"] != best_rps["engine"]
                else f"For throughput, {best_tps['engine']} led on both tok/s and req/s."
            )
        elif len(throughput) == 1:
            lines.append(
                f"Only {throughput[0]['engine']} completed the throughput ramp on this setup."
            )
        lines.extend(notes)
        if lines:
            takeaways.append(f"### {model['name']}\n" + "\n".join(f"- {line}" for line in lines))
    return takeaways


def build_markdown(rows: list[dict]) -> str:
    best_single = best_by(rows, "ttft_p95", scenario="single_request_latency", lower_is_better=True)
    best_tps = best_by(rows, "tokens_per_sec", scenario="throughput_ramp")
    best_rps = best_by(rows, "requests_per_sec", scenario="throughput_ramp")
    takeaways = generate_takeaways(rows)
    notes = sorted({note for model_id, notes in MODEL_NOTES.items() for note in notes})
    models_included = ", ".join(MODEL_ORDER)

    return f"""# Final Multi-Model Benchmark Report ({REPORT_DATE})

## Executive summary

This report consolidates the completed benchmark matrix collected on an **AWS g5.2xlarge** host with a single **NVIDIA A10G (24 GB)** GPU. All engine runs were executed **sequentially** on the same machine to avoid VRAM contention and to keep the comparison fair on one GPU.

### Headline findings

- **Fastest single-request TTFT p95:** {best_single["model"]} on **{best_single["engine"]}** at **{best_single["ttft_p95"]:.1f} ms**.
- **Highest throughput (tokens/sec):** {best_tps["model"]} on **{best_tps["engine"]}** at **{best_tps["tokens_per_sec"]:.1f} tok/s**.
- **Highest throughput (requests/sec):** {best_rps["model"]} on **{best_rps["engine"]}** at **{best_rps["requests_per_sec"]:.2f} req/s**.
- **Broad pattern:** vLLM consistently won the low-latency single-request TTFT tests, while throughput leadership depended on the model family.

## Environment

- Instance: **AWS g5.2xlarge**
- GPU: **NVIDIA A10G, 24 GB VRAM**
- Execution policy: **one engine at a time**
- Models included: {models_included}

## Important notes

{chr(10).join(f"- {note}" for note in notes)}

## Visual summary

### Single-request latency (TTFT p95)
![Single request TTFT p95](figures/single_request_ttft_p95.svg)

### Throughput tokens/sec
![Throughput tokens per second](figures/throughput_tokens_per_sec.svg)

### Throughput requests/sec
![Throughput requests per second](figures/throughput_requests_per_sec.svg)

### Throughput latency p95
![Throughput latency p95](figures/throughput_latency_p95.svg)

### Throughput tradeoff map
![Throughput tradeoff map](figures/throughput_tradeoff.svg)

## Single-request latency results

{render_markdown_table(rows_for(rows, scenario="single_request_latency"))}

## Throughput-ramp results

{render_markdown_table(rows_for(rows, scenario="throughput_ramp"))}

## Model-by-model takeaways

{chr(10).join(takeaways)}

## Interpretation

This matrix shows why model/engine benchmarking should not be reduced to a single winner. Across this run:

- **vLLM** repeatedly delivered the lowest TTFT in single-request tests.
- **SGLang** remained very competitive and in some cases won or matched throughput on mid-sized models.
- **Larger models** on a single A10G can require engine-specific tuning to fit and behave well.

The data is therefore best used as an **engineering decision aid**, not a blanket statement that one engine dominates all workloads.

## Generated artifacts

- `reports/final_benchmark_report_{REPORT_DATE}.md`
- `reports/final_benchmark_report_{REPORT_DATE}.html`
- `reports/benchmark_snapshot_{REPORT_DATE}.json`
- `reports/figures/single_request_ttft_p95.svg`
- `reports/figures/throughput_tokens_per_sec.svg`
- `reports/figures/throughput_requests_per_sec.svg`
- `reports/figures/throughput_latency_p95.svg`
- `reports/figures/throughput_tradeoff.svg`
"""


def build_html(rows: list[dict]) -> str:
    best_single = best_by(rows, "ttft_p95", scenario="single_request_latency", lower_is_better=True)
    best_tps = best_by(rows, "tokens_per_sec", scenario="throughput_ramp")
    best_rps = best_by(rows, "requests_per_sec", scenario="throughput_ramp")
    notes = sorted({note for model_id, notes in MODEL_NOTES.items() for note in notes})
    models_included = ", ".join(MODEL_ORDER)

    def render_table(table_rows: list[dict]) -> str:
        body = "".join(
            f"<tr><td>{html.escape(r['model'])}</td><td><code>{html.escape(r['scenario'])}</code></td><td>{html.escape(r['engine'])}</td><td>{r['ttft_p50']:.1f} ms</td><td>{r['ttft_p95']:.1f} ms</td><td>{r['latency_p95']:.1f} ms</td><td>{r['tokens_per_sec']:.1f}</td><td>{r['requests_per_sec']:.2f}</td><td>{r['success_pct']:.1f}%</td></tr>"
            for r in table_rows
        )
        return f"<table><thead><tr><th>Model</th><th>Scenario</th><th>Engine</th><th>TTFT p50</th><th>TTFT p95</th><th>Latency p95</th><th>Tok/s</th><th>Req/s</th><th>Success</th></tr></thead><tbody>{body}</tbody></table>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Final Multi-Model Benchmark Report ({REPORT_DATE})</title>
  <style>
    :root {{
      --bg: #0b1020; --panel: #121933; --panel2: #172043; --border: #27335a;
      --text: #e8eefc; --muted: #9fb0d0; --link: #7cc4ff;
    }}
    body {{ font-family: system-ui, -apple-system, sans-serif; margin: 2rem; background: var(--bg); color: var(--text); }}
    a {{ color: var(--link); }}
    code {{ background:#11182c; padding:0.15rem 0.35rem; border-radius:6px; }}
    .hero {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap:1rem; margin:1rem 0 1.5rem; }}
    .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 1rem; }}
    .label {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 0.35rem; }}
    .value {{ font-size: 1.25rem; font-weight: 700; }}
    h1, h2, h3 {{ margin-top: 1.5rem; }}
    img {{ width: 100%; max-width: 1100px; display:block; margin: 0.75rem 0 1.25rem; border:1px solid var(--border); border-radius: 12px; background: #0f1530; }}
    table {{ width:100%; border-collapse: collapse; margin: 1rem 0 1.5rem; }}
    th, td {{ padding: 0.7rem; border-bottom: 1px solid var(--border); text-align:left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; }}
    ul {{ line-height: 1.6; }}
  </style>
</head>
<body>
  <h1>Final Multi-Model Benchmark Report ({REPORT_DATE})</h1>
  <p>This is the polished report for the completed single-A10G benchmark matrix comparing <strong>vLLM</strong> and <strong>SGLang</strong> across multiple open models.</p>

  <div class="hero">
    <div class="card"><div class="label">Fastest TTFT p95</div><div class="value">{html.escape(best_single["model"])}</div><div>{html.escape(best_single["engine"])} • {best_single["ttft_p95"]:.1f} ms</div></div>
    <div class="card"><div class="label">Best tok/s</div><div class="value">{html.escape(best_tps["model"])}</div><div>{html.escape(best_tps["engine"])} • {best_tps["tokens_per_sec"]:.1f} tok/s</div></div>
    <div class="card"><div class="label">Best req/s</div><div class="value">{html.escape(best_rps["model"])}</div><div>{html.escape(best_rps["engine"])} • {best_rps["requests_per_sec"]:.2f} req/s</div></div>
  </div>

  <div class="card">
    <h2>Environment</h2>
    <ul>
      <li>AWS <strong>g5.2xlarge</strong></li>
      <li>NVIDIA <strong>A10G 24 GB</strong></li>
      <li>Sequential engine execution on a single GPU</li>
      <li>Models included: <strong>{html.escape(models_included)}</strong></li>
    </ul>
  </div>

  <div class="card">
    <h2>Important notes</h2>
    <ul>
      {"".join(f"<li>{html.escape(note)}</li>" for note in notes)}
    </ul>
  </div>

  <h2>Visual summary</h2>
  <h3>Single-request latency (TTFT p95)</h3>
  <img src="figures/single_request_ttft_p95.svg" alt="Single request TTFT p95" />
  <h3>Throughput tokens/sec</h3>
  <img src="figures/throughput_tokens_per_sec.svg" alt="Throughput tokens per second" />
  <h3>Throughput requests/sec</h3>
  <img src="figures/throughput_requests_per_sec.svg" alt="Throughput requests per second" />
  <h3>Throughput latency p95</h3>
  <img src="figures/throughput_latency_p95.svg" alt="Throughput latency p95" />
  <h3>Throughput tradeoff map</h3>
  <img src="figures/throughput_tradeoff.svg" alt="Throughput tradeoff map" />

  <h2>Single-request latency results</h2>
  {render_table(rows_for(rows, scenario="single_request_latency"))}

  <h2>Throughput-ramp results</h2>
  {render_table(rows_for(rows, scenario="throughput_ramp"))}
</body>
</html>
"""


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    rows = load_latest_rows()
    if not rows:
        raise SystemExit("No matching benchmark result files found in results/")

    grouped_single_ttft = grouped_metric(rows, "single_request_latency", "ttft_p95")
    grouped_throughput_tps = grouped_metric(rows, "throughput_ramp", "tokens_per_sec")
    grouped_throughput_rps = grouped_metric(rows, "throughput_ramp", "requests_per_sec")
    grouped_throughput_latency = grouped_metric(rows, "throughput_ramp", "latency_p95")

    render_grouped_bar_svg(
        title="Single-request latency: TTFT p95",
        subtitle="Lower is better • sequential runs on AWS g5.2xlarge / A10G",
        grouped=grouped_single_ttft,
        y_label="TTFT p95 (ms)",
        output=FIGURES_DIR / "single_request_ttft_p95.svg",
        lower_is_better=True,
    )
    render_grouped_bar_svg(
        title="Throughput ramp: tokens/sec",
        subtitle="Higher is better • completed model/engine pairs only",
        grouped=grouped_throughput_tps,
        y_label="Tokens / sec",
        output=FIGURES_DIR / "throughput_tokens_per_sec.svg",
    )
    render_grouped_bar_svg(
        title="Throughput ramp: requests/sec",
        subtitle="Higher is better • completed model/engine pairs only",
        grouped=grouped_throughput_rps,
        y_label="Requests / sec",
        output=FIGURES_DIR / "throughput_requests_per_sec.svg",
    )
    render_grouped_bar_svg(
        title="Throughput ramp: latency p95",
        subtitle="Lower is better • note the Gemma 9B SGLang tail-latency spike",
        grouped=grouped_throughput_latency,
        y_label="Latency p95 (ms)",
        output=FIGURES_DIR / "throughput_latency_p95.svg",
        lower_is_better=True,
    )
    render_scatter_svg(rows, FIGURES_DIR / "throughput_tradeoff.svg")

    snapshot = OUTPUT_DIR / f"benchmark_snapshot_{REPORT_DATE}.json"
    snapshot.write_text(json.dumps(rows, indent=2))
    markdown = OUTPUT_DIR / f"final_benchmark_report_{REPORT_DATE}.md"
    markdown.write_text(build_markdown(rows))
    html_path = OUTPUT_DIR / f"final_benchmark_report_{REPORT_DATE}.html"
    html_path.write_text(build_html(rows))
    (OUTPUT_DIR / "index.html").write_text(build_html(rows))


if __name__ == "__main__":
    main()
