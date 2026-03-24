"""Aggregate saved benchmark JSON files into a final markdown report."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _iter_result_files(results_dir: Path) -> list[Path]:
    return sorted(path for path in results_dir.glob("*.json") if path.is_file())


def _load_result_files(results_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _iter_result_files(results_dir):
        data = json.loads(path.read_text())
        if "metrics" in data and "scenario_name" in data and "engine_name" in data:
            data["_source_path"] = str(path)
            records.append(data)
    return records


def aggregate_results(results_dir: Path) -> dict[str, Any]:
    records = _load_result_files(results_dir)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        run_meta = record.get("run_metadata", {})
        model = run_meta.get("model", "unknown-model")
        grouped[(model, record["scenario_name"], record["engine_name"])].append(record)

    summary_rows: list[dict[str, Any]] = []
    for (model, scenario_name, engine_name), items in sorted(grouped.items()):
        metrics_list = [item["metrics"] for item in items]
        summary_rows.append(
            {
                "model": model,
                "scenario_name": scenario_name,
                "engine_name": engine_name,
                "runs": len(items),
                "ttft_p50_ms": mean(m["ttft"]["p50"] for m in metrics_list),
                "ttft_p95_ms": mean(m["ttft"]["p95"] for m in metrics_list),
                "latency_p95_ms": mean(m["latency"]["p95"] for m in metrics_list),
                "tokens_per_sec": mean(m["throughput"]["tokens_per_sec"] for m in metrics_list),
                "requests_per_sec": mean(m["throughput"]["requests_per_sec"] for m in metrics_list),
                "success_rate_pct": mean((1 - m["error_rate"]) * 100 for m in metrics_list),
                "prompt_packs": sorted(
                    {
                        item.get("workload_metadata", {}).get("prompt_pack", "unknown")
                        for item in items
                    }
                ),
                "sources": [item["_source_path"] for item in items],
            }
        )

    return {
        "results_dir": str(results_dir),
        "total_result_files": len(records),
        "rows": summary_rows,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    rows = summary["rows"]
    models = sorted({row["model"] for row in rows})

    lines = [
        "# Final Benchmark Summary",
        "",
        f"Source directory: `{summary['results_dir']}`",
        f"Result files considered: **{summary['total_result_files']}**",
        "",
    ]

    if not rows:
        lines.append("No result files found.")
        return "\n".join(lines)

    for model in models:
        lines.extend(
            [
                f"## {model}",
                "",
                "| Scenario | Engine | Runs | Prompt pack(s) | TTFT p50 | TTFT p95 | Latency p95 | Tok/s | Req/s | Success |",
                "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        model_rows = [row for row in rows if row["model"] == model]
        for row in sorted(
            model_rows, key=lambda item: (item["scenario_name"], item["engine_name"])
        ):
            lines.append(
                "| {scenario} | {engine} | {runs} | {packs} | {ttft_p50:.1f} ms | {ttft_p95:.1f} ms | "
                "{latency_p95:.1f} ms | {tps:.1f} | {rps:.2f} | {success:.1f}% |".format(
                    scenario=row["scenario_name"],
                    engine=row["engine_name"],
                    runs=row["runs"],
                    packs=", ".join(row["prompt_packs"]),
                    ttft_p50=row["ttft_p50_ms"],
                    ttft_p95=row["ttft_p95_ms"],
                    latency_p95=row["latency_p95_ms"],
                    tps=row["tokens_per_sec"],
                    rps=row["requests_per_sec"],
                    success=row["success_rate_pct"],
                )
            )
        lines.append("")

    return "\n".join(lines)


def generate_final_report(results_dir: Path, output_path: Path) -> dict[str, Any]:
    summary = aggregate_results(results_dir)
    output_path.write_text(render_markdown(summary))
    return summary
