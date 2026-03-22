from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

REPORT_DATE = "2026-03-22"
OUTPUT_DIR = Path("reports")
FIGURES_DIR = OUTPUT_DIR / "figures"

DATA = [
    {
        "model": "Gemma 2B",
        "model_id": "google/gemma-2-2b-it",
        "size_b": 2,
        "scenario": "single_request_latency",
        "engine": "SGLang",
        "ttft_p50": 34.1,
        "ttft_p95": 35.0,
        "ttft_p99": 39.0,
        "latency_p95": 1683.0,
        "tokens_per_sec": 3785.5,
        "requests_per_sec": 29.57,
        "success_pct": 100.0,
    },
    {
        "model": "Gemma 2B",
        "model_id": "google/gemma-2-2b-it",
        "size_b": 2,
        "scenario": "single_request_latency",
        "engine": "vLLM",
        "ttft_p50": 19.8,
        "ttft_p95": 20.3,
        "ttft_p99": 43.5,
        "latency_p95": 1661.8,
        "tokens_per_sec": 3749.2,
        "requests_per_sec": 29.29,
        "success_pct": 100.0,
    },
    {
        "model": "Gemma 2B",
        "model_id": "google/gemma-2-2b-it",
        "size_b": 2,
        "scenario": "throughput_ramp",
        "engine": "SGLang",
        "ttft_p50": 53.6,
        "ttft_p95": 159.9,
        "ttft_p99": 177.1,
        "latency_p95": 4898.1,
        "tokens_per_sec": 36459.2,
        "requests_per_sec": 142.43,
        "success_pct": 100.0,
    },
    {
        "model": "Gemma 2B",
        "model_id": "google/gemma-2-2b-it",
        "size_b": 2,
        "scenario": "throughput_ramp",
        "engine": "vLLM",
        "ttft_p50": 44.0,
        "ttft_p95": 189.9,
        "ttft_p99": 239.2,
        "latency_p95": 2301.1,
        "tokens_per_sec": 33875.2,
        "requests_per_sec": 289.16,
        "success_pct": 100.0,
    },
    {
        "model": "Phi-3 mini",
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "size_b": 3,
        "scenario": "single_request_latency",
        "engine": "vLLM",
        "ttft_p50": 25.4,
        "ttft_p95": 25.9,
        "ttft_p99": 53.4,
        "latency_p95": 2243.6,
        "tokens_per_sec": 2786.4,
        "requests_per_sec": 21.77,
        "success_pct": 100.0,
        "note": "SGLang blocked on this setup due to FlashInfer/CUDA graph incompatibility (unsupported head_dim=96).",
    },
    {
        "model": "Phi-3 mini",
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "size_b": 3,
        "scenario": "throughput_ramp",
        "engine": "vLLM",
        "ttft_p50": 55.5,
        "ttft_p95": 188.6,
        "ttft_p99": 241.1,
        "latency_p95": 8645.5,
        "tokens_per_sec": 20533.9,
        "requests_per_sec": 80.21,
        "success_pct": 100.0,
        "note": "SGLang blocked on this setup due to FlashInfer/CUDA graph incompatibility (unsupported head_dim=96).",
    },
    {
        "model": "Qwen 7B",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "size_b": 7,
        "scenario": "single_request_latency",
        "engine": "SGLang",
        "ttft_p50": 67.9,
        "ttft_p95": 68.2,
        "ttft_p99": 69.0,
        "latency_p95": 4178.4,
        "tokens_per_sec": 1531.6,
        "requests_per_sec": 11.97,
        "success_pct": 100.0,
    },
    {
        "model": "Qwen 7B",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "size_b": 7,
        "scenario": "single_request_latency",
        "engine": "vLLM",
        "ttft_p50": 40.4,
        "ttft_p95": 40.7,
        "ttft_p99": 115.2,
        "latency_p95": 4202.4,
        "tokens_per_sec": 1451.3,
        "requests_per_sec": 11.34,
        "success_pct": 100.0,
    },
    {
        "model": "Qwen 7B",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "size_b": 7,
        "scenario": "throughput_ramp",
        "engine": "SGLang",
        "ttft_p50": 68.7,
        "ttft_p95": 194.2,
        "ttft_p99": 425.3,
        "latency_p95": 9581.5,
        "tokens_per_sec": 18667.3,
        "requests_per_sec": 72.92,
        "success_pct": 100.0,
    },
    {
        "model": "Qwen 7B",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "size_b": 7,
        "scenario": "throughput_ramp",
        "engine": "vLLM",
        "ttft_p50": 89.9,
        "ttft_p95": 311.9,
        "ttft_p99": 439.9,
        "latency_p95": 10091.5,
        "tokens_per_sec": 15140.9,
        "requests_per_sec": 68.98,
        "success_pct": 100.0,
    },
    {
        "model": "Mistral 7B",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "size_b": 7,
        "scenario": "single_request_latency",
        "engine": "SGLang",
        "ttft_p50": 66.0,
        "ttft_p95": 66.4,
        "ttft_p99": 69.9,
        "latency_p95": 4057.3,
        "tokens_per_sec": 1574.9,
        "requests_per_sec": 12.30,
        "success_pct": 100.0,
    },
    {
        "model": "Mistral 7B",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "size_b": 7,
        "scenario": "single_request_latency",
        "engine": "vLLM",
        "ttft_p50": 41.4,
        "ttft_p95": 41.7,
        "ttft_p99": 99.6,
        "latency_p95": 4044.0,
        "tokens_per_sec": 1539.7,
        "requests_per_sec": 12.03,
        "success_pct": 100.0,
    },
    {
        "model": "Mistral 7B",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "size_b": 7,
        "scenario": "throughput_ramp",
        "engine": "SGLang",
        "ttft_p50": 69.7,
        "ttft_p95": 353.6,
        "ttft_p99": 412.1,
        "latency_p95": 10332.4,
        "tokens_per_sec": 17294.3,
        "requests_per_sec": 67.56,
        "success_pct": 100.0,
    },
    {
        "model": "Mistral 7B",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "size_b": 7,
        "scenario": "throughput_ramp",
        "engine": "vLLM",
        "ttft_p50": 92.3,
        "ttft_p95": 240.5,
        "ttft_p99": 432.7,
        "latency_p95": 10342.1,
        "tokens_per_sec": 17175.6,
        "requests_per_sec": 67.09,
        "success_pct": 100.0,
    },
    {
        "model": "Gemma 9B",
        "model_id": "google/gemma-2-9b-it",
        "size_b": 9,
        "scenario": "single_request_latency",
        "engine": "SGLang",
        "ttft_p50": 86.3,
        "ttft_p95": 86.9,
        "ttft_p99": 106.7,
        "latency_p95": 333.4,
        "tokens_per_sec": 676.2,
        "requests_per_sec": 135.24,
        "success_pct": 100.0,
    },
    {
        "model": "Gemma 9B",
        "model_id": "google/gemma-2-9b-it",
        "size_b": 9,
        "scenario": "single_request_latency",
        "engine": "vLLM",
        "ttft_p50": 120.8,
        "ttft_p95": 122.2,
        "ttft_p99": 124.5,
        "latency_p95": 381.6,
        "tokens_per_sec": 648.6,
        "requests_per_sec": 129.73,
        "success_pct": 100.0,
        "note": "vLLM required tuned launch settings on the A10G: context=4096, gpu_memory_utilization=0.92.",
    },
    {
        "model": "Gemma 9B",
        "model_id": "google/gemma-2-9b-it",
        "size_b": 9,
        "scenario": "throughput_ramp",
        "engine": "SGLang",
        "ttft_p50": 91.4,
        "ttft_p95": 3666.6,
        "ttft_p99": 5389.8,
        "latency_p95": 5277.1,
        "tokens_per_sec": 3595.1,
        "requests_per_sec": 99.86,
        "success_pct": 100.0,
    },
    {
        "model": "Gemma 9B",
        "model_id": "google/gemma-2-9b-it",
        "size_b": 9,
        "scenario": "throughput_ramp",
        "engine": "vLLM",
        "ttft_p50": 82.7,
        "ttft_p95": 362.5,
        "ttft_p99": 525.0,
        "latency_p95": 2483.7,
        "tokens_per_sec": 9619.6,
        "requests_per_sec": 267.21,
        "success_pct": 100.0,
        "note": "vLLM required tuned launch settings on the A10G: context=4096, gpu_memory_utilization=0.92.",
    },
]

COLORS = {"vLLM": "#5B8DEF", "SGLang": "#F5A524"}
ENGINES = ["vLLM", "SGLang"]
MODEL_ORDER = ["Gemma 2B", "Phi-3 mini", "Qwen 7B", "Mistral 7B", "Gemma 9B"]


def records_for(scenario: str) -> list[dict]:
    return [r for r in DATA if r["scenario"] == scenario]


def grouped_metric(scenario: str, metric: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {model: {} for model in MODEL_ORDER}
    for row in records_for(scenario):
        out[row["model"]][row["engine"]] = row[metric]
    return out


def render_grouped_bar_svg(title: str, subtitle: str, grouped: dict[str, dict[str, float]], y_label: str, output: Path, *, lower_is_better: bool = False) -> None:
    width, height = 1200, 640
    left, right, top, bottom = 100, 40, 90, 110
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_val = max((value for engines in grouped.values() for value in engines.values()), default=1.0)
    max_val *= 1.15
    ticks = 5
    group_w = plot_w / max(len(MODEL_ORDER), 1)
    bar_w = group_w * 0.26
    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append('<style>text{font-family:system-ui,-apple-system,sans-serif;fill:#e8eefc}.muted{fill:#9fb0d0}.small{font-size:12px}.label{font-size:14px}.title{font-size:28px;font-weight:700}.subtitle{font-size:14px}.axis{stroke:#5c6b91;stroke-width:1}.grid{stroke:#27335a;stroke-width:1}.value{font-size:12px;font-weight:700}.legend{font-size:13px}</style>')
    svg.append(f'<rect width="{width}" height="{height}" fill="#0b1020"/>')
    svg.append(f'<text x="{left}" y="40" class="title">{title}</text>')
    svg.append(f'<text x="{left}" y="64" class="subtitle muted">{subtitle}</text>')
    # grid + y ticks
    for i in range(ticks + 1):
        y = top + plot_h - (plot_h * i / ticks)
        value = max_val * i / ticks
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        svg.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="small muted">{value:.0f}</text>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>')
    svg.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>')
    # bars
    for gi, model in enumerate(MODEL_ORDER):
        gx = left + gi * group_w + group_w * 0.18
        for ei, engine in enumerate(ENGINES):
            if engine not in grouped[model]:
                continue
            value = grouped[model][engine]
            bh = (value / max_val) * plot_h if max_val else 0
            x = gx + ei * (bar_w + group_w * 0.08)
            y = top + plot_h - bh
            color = COLORS[engine]
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="6" fill="{color}"/>')
            svg.append(f'<text x="{x + bar_w/2:.1f}" y="{y-8:.1f}" text-anchor="middle" class="value">{value:.1f}</text>')
        svg.append(f'<text x="{gx + group_w*0.28:.1f}" y="{height-bottom+28}" text-anchor="middle" class="label">{model}</text>')
    # legend
    lx = width - right - 180
    ly = 44
    for idx, engine in enumerate(ENGINES):
        yy = ly + idx * 22
        svg.append(f'<rect x="{lx}" y="{yy-10}" width="14" height="14" rx="3" fill="{COLORS[engine]}"/>')
        svg.append(f'<text x="{lx+24}" y="{yy+1}" class="legend">{engine}</text>')
    svg.append(f'<text x="24" y="{top + plot_h/2:.1f}" transform="rotate(-90 24 {top + plot_h/2:.1f})" class="label muted">{y_label}</text>')
    if lower_is_better:
        svg.append(f'<text x="{left}" y="{height-20}" class="small muted">Lower is better</text>')
    else:
        svg.append(f'<text x="{left}" y="{height-20}" class="small muted">Higher is better</text>')
    svg.append('</svg>')
    output.write_text(''.join(svg))


def render_markdown_tables(rows: Iterable[dict]) -> str:
    rows = list(rows)
    out = []
    out.append('| Model | Scenario | Engine | TTFT p50 | TTFT p95 | Latency p95 | Tok/s | Req/s | Success |')
    out.append('|---|---|---|---:|---:|---:|---:|---:|---:|')
    for r in rows:
        out.append(
            f"| {r['model']} | `{r['scenario']}` | {r['engine']} | {r['ttft_p50']:.1f} ms | {r['ttft_p95']:.1f} ms | {r['latency_p95']:.1f} ms | {r['tokens_per_sec']:.1f} | {r['requests_per_sec']:.2f} | {r['success_pct']:.1f}% |"
        )
    return '\n'.join(out)


def best_by(rows: Iterable[dict], key: str, *, scenario: str, lower_is_better: bool = False) -> dict | None:
    candidates = [r for r in rows if r['scenario'] == scenario]
    if not candidates:
        return None
    return sorted(candidates, key=lambda r: r[key], reverse=not lower_is_better)[0]


def build_report() -> str:
    single = [r for r in DATA if r['scenario'] == 'single_request_latency']
    throughput = [r for r in DATA if r['scenario'] == 'throughput_ramp']
    best_single_ttft = best_by(DATA, 'ttft_p95', scenario='single_request_latency', lower_is_better=True)
    best_throughput_tps = best_by(DATA, 'tokens_per_sec', scenario='throughput_ramp')
    best_throughput_rps = best_by(DATA, 'requests_per_sec', scenario='throughput_ramp')

    return f"""# Final Multi-Model Benchmark Report ({REPORT_DATE})

## Executive summary

This report consolidates the completed benchmark matrix collected on the AWS `g5.2xlarge` single-GPU host (NVIDIA A10G 24 GB) using **sequential engine execution**. The goal was to compare **vLLM** and **SGLang** across a representative mix of open models while avoiding VRAM contention by running only one engine at a time.

### Headline findings

- **Best single-request TTFT p95:** {best_single_ttft['model']} on {best_single_ttft['engine']} at **{best_single_ttft['ttft_p95']:.1f} ms**.
- **Best throughput (tokens/sec):** {best_throughput_tps['model']} on {best_throughput_tps['engine']} at **{best_throughput_tps['tokens_per_sec']:.1f} tok/s**.
- **Best throughput (requests/sec):** {best_throughput_rps['model']} on {best_throughput_rps['engine']} at **{best_throughput_rps['requests_per_sec']:.2f} req/s**.
- **Phi-3 mini** was benchmarked on **vLLM only** because the SGLang FlashInfer/CUDA graph path crashed on `unsupported head_dim=96`.
- **Gemma 9B + vLLM** required tuned launch parameters on the A10G: `context=4096` and `gpu_memory_utilization=0.92`.

## Test environment

- Instance: **AWS g5.2xlarge**
- GPU: **NVIDIA A10G (24 GB VRAM)**
- Execution policy: **sequential only** (one engine at a time)
- Cooling policy: cooldown between engine switches / heavier model transitions
- Result source: consolidated metrics captured during the orchestration run on {REPORT_DATE}

## Visual summary

### 1) Single-request TTFT p95

![Single request TTFT p95](figures/single_request_ttft_p95.svg)

### 2) Throughput tokens/sec

![Throughput tokens per second](figures/throughput_tokens_per_sec.svg)

### 3) Throughput requests/sec

![Throughput requests per second](figures/throughput_requests_per_sec.svg)

### 4) Throughput latency p95

![Throughput latency p95](figures/throughput_latency_p95.svg)

## Single-request latency results

{render_markdown_tables(single)}

## Throughput-ramp results

{render_markdown_tables(throughput)}

## Model-by-model takeaways

### Gemma 2B
- vLLM had the lower TTFT on the single-request case.
- vLLM dominated requests/sec on throughput ramp.
- SGLang posted slightly higher tokens/sec on the ramp.

### Phi-3 mini
- vLLM results are solid and competitive for a small model.
- SGLang could not be included on this hardware/software combination due to a reproducible compatibility failure.

### Qwen 7B
- vLLM won the low-latency single-request scenario.
- SGLang won the throughput-ramp tokens/sec and requests/sec for this model.

### Mistral 7B
- vLLM again won the single-request TTFT.
- Throughput-ramp performance between vLLM and SGLang ended up very close on this model.

### Gemma 9B
- SGLang fit and ran with default-style settings, but had very poor tail latency in throughput ramp.
- vLLM needed tuning to fit on the A10G, but once tuned it substantially improved p95 latency and throughput for the ramp scenario.

## Caveats

- These numbers are for a **single A10G host**, not a multi-GPU cluster.
- Sequential execution improves fairness on this box, but still reflects one-machine constraints.
- Later work should add:
  - multi-run variance / repeated trials,
  - richer prompt-pack driven workloads,
  - structured-output validity metrics,
  - and larger hardware tiers for 9B+ models.

## Files generated

- `reports/final_benchmark_report_{REPORT_DATE}.md`
- `reports/benchmark_snapshot_{REPORT_DATE}.json`
- `reports/figures/single_request_ttft_p95.svg`
- `reports/figures/throughput_tokens_per_sec.svg`
- `reports/figures/throughput_requests_per_sec.svg`
- `reports/figures/throughput_latency_p95.svg`
"""


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    (OUTPUT_DIR / f"benchmark_snapshot_{REPORT_DATE}.json").write_text(json.dumps(DATA, indent=2))

    render_grouped_bar_svg(
        title="Single-request latency: TTFT p95",
        subtitle="Lower is better • sequential runs on AWS g5.2xlarge / A10G",
        grouped=grouped_metric("single_request_latency", "ttft_p95"),
        y_label="TTFT p95 (ms)",
        output=FIGURES_DIR / "single_request_ttft_p95.svg",
        lower_is_better=True,
    )
    render_grouped_bar_svg(
        title="Throughput ramp: tokens/sec",
        subtitle="Higher is better • completed model/engine pairs only",
        grouped=grouped_metric("throughput_ramp", "tokens_per_sec"),
        y_label="Tokens / sec",
        output=FIGURES_DIR / "throughput_tokens_per_sec.svg",
    )
    render_grouped_bar_svg(
        title="Throughput ramp: requests/sec",
        subtitle="Higher is better • completed model/engine pairs only",
        grouped=grouped_metric("throughput_ramp", "requests_per_sec"),
        y_label="Requests / sec",
        output=FIGURES_DIR / "throughput_requests_per_sec.svg",
    )
    render_grouped_bar_svg(
        title="Throughput ramp: latency p95",
        subtitle="Lower is better • note the Gemma 9B SGLang tail-latency spike",
        grouped=grouped_metric("throughput_ramp", "latency_p95"),
        y_label="Latency p95 (ms)",
        output=FIGURES_DIR / "throughput_latency_p95.svg",
        lower_is_better=True,
    )

    (OUTPUT_DIR / f"final_benchmark_report_{REPORT_DATE}.md").write_text(build_report())


if __name__ == "__main__":
    main()
