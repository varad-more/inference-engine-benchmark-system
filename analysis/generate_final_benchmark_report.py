from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable

REPORT_DATE = "2026-03-22"
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("reports")
FIGURES_DIR = OUTPUT_DIR / "figures"

TARGET_MODELS = [
    {"id": "google/gemma-2-2b-it", "name": "Gemma 2B", "size_b": 2},
    {"id": "microsoft/Phi-3-mini-4k-instruct", "name": "Phi-3 mini", "size_b": 3},
    {"id": "Qwen/Qwen2.5-7B-Instruct", "name": "Qwen 7B", "size_b": 7},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3", "name": "Mistral 7B", "size_b": 7},
    {"id": "google/gemma-2-9b-it", "name": "Gemma 9B", "size_b": 9},
]
TARGET_MODEL_MAP = {entry["id"]: entry for entry in TARGET_MODELS}
MODEL_ORDER = [entry["name"] for entry in TARGET_MODELS]
SCENARIO_ORDER = ["single_request_latency", "throughput_ramp"]
ENGINE_ORDER = ["vLLM", "SGLang"]
ENGINE_LABELS = {"VLLMClient": "vLLM", "SGLangClient": "SGLang"}
COLORS = {"vLLM": "#5B8DEF", "SGLang": "#F5A524"}

MODEL_NOTES = {
    "microsoft/Phi-3-mini-4k-instruct": [
        "SGLang could not be included on this setup because the FlashInfer/CUDA graph path failed on unsupported `head_dim=96`.",
    ],
    "google/gemma-2-9b-it": [
        "vLLM required tuned launch settings on the single A10G: `context=4096` and `gpu_memory_utilization=0.92`.",
    ],
}


def _scenario_rank(name: str) -> int:
    return SCENARIO_ORDER.index(name) if name in SCENARIO_ORDER else len(SCENARIO_ORDER)


def _engine_rank(name: str) -> int:
    return ENGINE_ORDER.index(name) if name in ENGINE_ORDER else len(ENGINE_ORDER)


def _safe_float(value: float | int | None) -> float | None:
    return None if value is None else float(value)


def load_latest_rows() -> list[dict]:
    latest: dict[tuple[str, str, str], tuple[float, dict]] = {}

    for path in sorted(RESULTS_DIR.glob("*Client_*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not {"scenario_name", "engine_name", "metrics"}.issubset(data):
            continue

        model_id = data.get("run_metadata", {}).get("model")
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
        throughput = metrics.get("throughput", {})
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
            "tokens_per_sec": _safe_float(throughput.get("tokens_per_sec")),
            "requests_per_sec": _safe_float(throughput.get("requests_per_sec")),
            "success_pct": round((1 - float(metrics.get("error_rate", 0.0))) * 100, 1),
            "path": container["path"],
        }
        rows.append(row)

    rows.sort(key=lambda r: (r["size_b"], _scenario_rank(r["scenario"]), _engine_rank(r["engine"])))
    return rows


def rows_for(rows: Iterable[dict], *, scenario: str | None = None, model_id: str | None = None) -> list[dict]:
    data = list(rows)
    if scenario is not None:
        data = [r for r in data if r["scenario"] == scenario]
    if model_id is not None:
        data = [r for r in data if r["model_id"] == model_id]
    return data


def best_by(rows: Iterable[dict], metric: str, *, scenario: str, lower_is_better: bool = False) -> dict | None:
    candidates = [r for r in rows if r["scenario"] == scenario and r.get(metric) is not None]
    if not candidates:
        return None
    return sorted(candidates, key=lambda r: r[metric], reverse=not lower_is_better)[0]


def grouped_metric(rows: Iterable[dict], scenario: str, metric: str) -> dict[str, dict[str, float]]:
    grouped = {model: {} for model in MODEL_ORDER}
    for row in rows:
        if row["scenario"] != scenario:
            continue
        value = row.get(metric)
        if value is not None:
            grouped[row["model"]][row["engine"]] = value
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
    width, height = 1200, 650
    left, right, top, bottom = 100, 40, 90, 110
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_val = max((value for engines in grouped.values() for value in engines.values()), default=1.0) * 1.15
    ticks = 5
    group_w = plot_w / max(len(MODEL_ORDER), 1)
    bar_w = group_w * 0.26

    svg: list[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append('<style>text{font-family:system-ui,-apple-system,sans-serif;fill:#e8eefc}.muted{fill:#9fb0d0}.small{font-size:12px}.label{font-size:14px}.title{font-size:28px;font-weight:700}.subtitle{font-size:14px}.axis{stroke:#5c6b91;stroke-width:1}.grid{stroke:#27335a;stroke-width:1}.value{font-size:12px;font-weight:700}.legend{font-size:13px}</style>')
    svg.append(f'<rect width="{width}" height="{height}" fill="#0b1020"/>')
    svg.append(f'<text x="{left}" y="40" class="title">{html.escape(title)}</text>')
    svg.append(f'<text x="{left}" y="64" class="subtitle muted">{html.escape(subtitle)}</text>')

    for i in range(ticks + 1):
        y = top + plot_h - (plot_h * i / ticks)
        value = max_val * i / ticks
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        svg.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="small muted">{value:.0f}</text>')

    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>')
    svg.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>')

    for gi, model in enumerate(MODEL_ORDER):
        gx = left + gi * group_w + group_w * 0.18
        for ei, engine in enumerate(ENGINE_ORDER):
            if engine not in grouped[model]:
                continue
            value = grouped[model][engine]
            bh = (value / max_val) * plot_h if max_val else 0
            x = gx + ei * (bar_w + group_w * 0.08)
            y = top + plot_h - bh
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="6" fill="{COLORS[engine]}"/>')
            svg.append(f'<text x="{x + bar_w/2:.1f}" y="{y-8:.1f}" text-anchor="middle" class="value">{value:.1f}</text>')
        svg.append(f'<text x="{gx + group_w*0.28:.1f}" y="{height-bottom+28}" text-anchor="middle" class="label">{html.escape(model)}</text>')

    lx = width - right - 180
    ly = 44
    for idx, engine in enumerate(ENGINE_ORDER):
        yy = ly + idx * 22
        svg.append(f'<rect x="{lx}" y="{yy-10}" width="14" height="14" rx="3" fill="{COLORS[engine]}"/>')
        svg.append(f'<text x="{lx+24}" y="{yy+1}" class="legend">{engine}</text>')

    svg.append(f'<text x="24" y="{top + plot_h/2:.1f}" transform="rotate(-90 24 {top + plot_h/2:.1f})" class="label muted">{html.escape(y_label)}</text>')
    svg.append(f'<text x="{left}" y="{height-20}" class="small muted">{"Lower is better" if lower_is_better else "Higher is better"}</text>')
    svg.append('</svg>')
    output.write_text(''.join(svg))


def render_scatter_svg(rows: Iterable[dict], output: Path) -> None:
    data = [r for r in rows if r["scenario"] == "throughput_ramp" and r.get("latency_p95") is not None and r.get("tokens_per_sec") is not None]
    width, height = 1200, 680
    left, right, top, bottom = 110, 50, 90, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_x = min((r["latency_p95"] for r in data), default=0)
    max_x = max((r["latency_p95"] for r in data), default=1)
    min_y = 0.0
    max_y = max((r["tokens_per_sec"] for r in data), default=1.0) * 1.1

    def x_pos(v: float) -> float:
        span = max(max_x - min_x, 1.0)
        return left + ((v - min_x) / span) * plot_w

    def y_pos(v: float) -> float:
        span = max(max_y - min_y, 1.0)
        return top + plot_h - ((v - min_y) / span) * plot_h

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append('<style>text{font-family:system-ui,-apple-system,sans-serif;fill:#e8eefc}.muted{fill:#9fb0d0}.title{font-size:28px;font-weight:700}.subtitle{font-size:14px}.axis{stroke:#5c6b91;stroke-width:1}.grid{stroke:#27335a;stroke-width:1}.small{font-size:12px}.label{font-size:14px}</style>')
    svg.append(f'<rect width="{width}" height="{height}" fill="#0b1020"/>')
    svg.append(f'<text x="{left}" y="40" class="title">Throughput tradeoff map</text>')
    svg.append(f'<text x="{left}" y="64" class="subtitle muted">Each point is a completed throughput-ramp run. Top-left is ideal: lower latency p95, higher tokens/sec.</text>')
    for i in range(6):
        x = left + plot_w * i / 5
        y = top + plot_h * i / 5
        xv = min_x + (max_x - min_x) * i / 5
        yv = max_y * i / 5
        svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" class="grid"/>')
        svg.append(f'<line x1="{left}" y1="{top+plot_h-y+top:.1f}" x2="{left+plot_w}" y2="{top+plot_h-y+top:.1f}" class="grid"/>')
        svg.append(f'<text x="{x:.1f}" y="{top+plot_h+22:.1f}" text-anchor="middle" class="small muted">{xv:.0f}</text>')
        svg.append(f'<text x="{left-12}" y="{top+plot_h - (plot_h*i/5)+4:.1f}" text-anchor="end" class="small muted">{yv:.0f}</text>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>')
    svg.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>')
    for row in data:
        x = x_pos(row["latency_p95"])
        y = y_pos(row["tokens_per_sec"])
        color = COLORS[row["engine"]]
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
        label = f"{row['model']} • {row['engine']}"
        svg.append(f'<text x="{x+12:.1f}" y="{y-10:.1f}" class="small">{html.escape(label)}</text>')
    svg.append(f'<text x="{left + plot_w/2:.1f}" y="{height-18}" text-anchor="middle" class="label muted">Latency p95 (ms)</text>')
    svg.append(f'<text x="26" y="{top + plot_h/2:.1f}" transform="rotate(-90 26 {top + plot_h/2:.1f})" class="label muted">Tokens / sec</text>')
    svg.append('</svg>')
    output.write_text(''.join(svg))


def render_markdown_table(rows: Iterable[dict]) -> str:
    lines = [
        '| Model | Scenario | Engine | TTFT p50 | TTFT p95 | Latency p95 | Tok/s | Req/s | Success |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | `{r['scenario']}` | {r['engine']} | {r['ttft_p50']:.1f} ms | {r['ttft_p95']:.1f} ms | {r['latency_p95']:.1f} ms | {r['tokens_per_sec']:.1f} | {r['requests_per_sec']:.2f} | {r['success_pct']:.1f}% |"
        )
    return '\n'.join(lines)


def generate_takeaways(rows: Iterable[dict]) -> list[str]:
    takeaways = []
    rows = list(rows)
    for model in TARGET_MODELS:
        model_rows = rows_for(rows, model_id=model['id'])
        single = rows_for(model_rows, scenario='single_request_latency')
        throughput = rows_for(model_rows, scenario='throughput_ramp')
        notes = MODEL_NOTES.get(model['id'], [])
        lines = []
        if len(single) >= 2:
            best_single = sorted(single, key=lambda r: r['ttft_p95'])[0]
            lines.append(f"{best_single['engine']} won the single-request TTFT comparison.")
        elif len(single) == 1:
            lines.append(f"Only {single[0]['engine']} completed the single-request benchmark on this setup.")
        if len(throughput) >= 2:
            best_tps = sorted(throughput, key=lambda r: r['tokens_per_sec'], reverse=True)[0]
            best_rps = sorted(throughput, key=lambda r: r['requests_per_sec'], reverse=True)[0]
            lines.append(
                f"For throughput, {best_tps['engine']} led on tok/s while {best_rps['engine']} led on req/s."
                if best_tps['engine'] != best_rps['engine']
                else f"For throughput, {best_tps['engine']} led on both tok/s and req/s."
            )
        elif len(throughput) == 1:
            lines.append(f"Only {throughput[0]['engine']} completed the throughput ramp on this setup.")
        lines.extend(notes)
        if lines:
            takeaways.append(f"### {model['name']}\n" + '\n'.join(f"- {line}" for line in lines))
    return takeaways


def build_markdown(rows: list[dict]) -> str:
    best_single = best_by(rows, 'ttft_p95', scenario='single_request_latency', lower_is_better=True)
    best_tps = best_by(rows, 'tokens_per_sec', scenario='throughput_ramp')
    best_rps = best_by(rows, 'requests_per_sec', scenario='throughput_ramp')
    takeaways = generate_takeaways(rows)
    notes = sorted({note for model_id, notes in MODEL_NOTES.items() for note in notes})

    return f"""# Final Multi-Model Benchmark Report ({REPORT_DATE})

## Executive summary

This report consolidates the completed benchmark matrix collected on an **AWS g5.2xlarge** host with a single **NVIDIA A10G (24 GB)** GPU. All engine runs were executed **sequentially** on the same machine to avoid VRAM contention and to keep the comparison fair on one GPU.

### Headline findings

- **Fastest single-request TTFT p95:** {best_single['model']} on **{best_single['engine']}** at **{best_single['ttft_p95']:.1f} ms**.
- **Highest throughput (tokens/sec):** {best_tps['model']} on **{best_tps['engine']}** at **{best_tps['tokens_per_sec']:.1f} tok/s**.
- **Highest throughput (requests/sec):** {best_rps['model']} on **{best_rps['engine']}** at **{best_rps['requests_per_sec']:.2f} req/s**.
- **Broad pattern:** vLLM consistently won the low-latency single-request TTFT tests, while throughput leadership depended on the model family.

## Environment

- Instance: **AWS g5.2xlarge**
- GPU: **NVIDIA A10G, 24 GB VRAM**
- Execution policy: **one engine at a time**
- Models included: Gemma 2B, Phi-3 mini, Qwen 7B, Mistral 7B, Gemma 9B

## Important notes

{chr(10).join(f'- {note}' for note in notes)}

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

{render_markdown_table(rows_for(rows, scenario='single_request_latency'))}

## Throughput-ramp results

{render_markdown_table(rows_for(rows, scenario='throughput_ramp'))}

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
    best_single = best_by(rows, 'ttft_p95', scenario='single_request_latency', lower_is_better=True)
    best_tps = best_by(rows, 'tokens_per_sec', scenario='throughput_ramp')
    best_rps = best_by(rows, 'requests_per_sec', scenario='throughput_ramp')
    notes = sorted({note for model_id, notes in MODEL_NOTES.items() for note in notes})

    def render_table(table_rows: list[dict]) -> str:
        body = ''.join(
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
    <div class="card"><div class="label">Fastest TTFT p95</div><div class="value">{html.escape(best_single['model'])}</div><div>{html.escape(best_single['engine'])} • {best_single['ttft_p95']:.1f} ms</div></div>
    <div class="card"><div class="label">Best tok/s</div><div class="value">{html.escape(best_tps['model'])}</div><div>{html.escape(best_tps['engine'])} • {best_tps['tokens_per_sec']:.1f} tok/s</div></div>
    <div class="card"><div class="label">Best req/s</div><div class="value">{html.escape(best_rps['model'])}</div><div>{html.escape(best_rps['engine'])} • {best_rps['requests_per_sec']:.2f} req/s</div></div>
  </div>

  <div class="card">
    <h2>Environment</h2>
    <ul>
      <li>AWS <strong>g5.2xlarge</strong></li>
      <li>NVIDIA <strong>A10G 24 GB</strong></li>
      <li>Sequential engine execution on a single GPU</li>
    </ul>
  </div>

  <div class="card">
    <h2>Important notes</h2>
    <ul>
      {''.join(f'<li>{html.escape(note)}</li>' for note in notes)}
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
  {render_table(rows_for(rows, scenario='single_request_latency'))}

  <h2>Throughput-ramp results</h2>
  {render_table(rows_for(rows, scenario='throughput_ramp'))}
</body>
</html>
"""


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    rows = load_latest_rows()
    if not rows:
        raise SystemExit("No matching benchmark result files found in results/")

    grouped_single_ttft = grouped_metric(rows, 'single_request_latency', 'ttft_p95')
    grouped_throughput_tps = grouped_metric(rows, 'throughput_ramp', 'tokens_per_sec')
    grouped_throughput_rps = grouped_metric(rows, 'throughput_ramp', 'requests_per_sec')
    grouped_throughput_latency = grouped_metric(rows, 'throughput_ramp', 'latency_p95')

    render_grouped_bar_svg(
        title='Single-request latency: TTFT p95',
        subtitle='Lower is better • sequential runs on AWS g5.2xlarge / A10G',
        grouped=grouped_single_ttft,
        y_label='TTFT p95 (ms)',
        output=FIGURES_DIR / 'single_request_ttft_p95.svg',
        lower_is_better=True,
    )
    render_grouped_bar_svg(
        title='Throughput ramp: tokens/sec',
        subtitle='Higher is better • completed model/engine pairs only',
        grouped=grouped_throughput_tps,
        y_label='Tokens / sec',
        output=FIGURES_DIR / 'throughput_tokens_per_sec.svg',
    )
    render_grouped_bar_svg(
        title='Throughput ramp: requests/sec',
        subtitle='Higher is better • completed model/engine pairs only',
        grouped=grouped_throughput_rps,
        y_label='Requests / sec',
        output=FIGURES_DIR / 'throughput_requests_per_sec.svg',
    )
    render_grouped_bar_svg(
        title='Throughput ramp: latency p95',
        subtitle='Lower is better • note the Gemma 9B SGLang tail-latency spike',
        grouped=grouped_throughput_latency,
        y_label='Latency p95 (ms)',
        output=FIGURES_DIR / 'throughput_latency_p95.svg',
        lower_is_better=True,
    )
    render_scatter_svg(rows, FIGURES_DIR / 'throughput_tradeoff.svg')

    snapshot = OUTPUT_DIR / f'benchmark_snapshot_{REPORT_DATE}.json'
    snapshot.write_text(json.dumps(rows, indent=2))
    markdown = OUTPUT_DIR / f'final_benchmark_report_{REPORT_DATE}.md'
    markdown.write_text(build_markdown(rows))
    html_path = OUTPUT_DIR / f'final_benchmark_report_{REPORT_DATE}.html'
    html_path.write_text(build_html(rows))
    (OUTPUT_DIR / 'index.html').write_text(build_html(rows))


if __name__ == '__main__':
    main()
