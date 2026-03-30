"""Aggregate saved benchmark JSON files into a final markdown report."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from analysis import get_engine_variant, get_spec_method, load_results, select_model_results


def aggregate_results(results_dir: Path, model: str | None = None) -> dict[str, Any]:
    records = load_results(results_dir)
    available_models = sorted(
        {
            str(record.get("run_metadata", {}).get("model", "unknown-model"))
            for record in records
            if record.get("run_metadata", {}).get("model")
        }
    )
    selection_mode = "all-results"
    selected_model = model

    if model is not None:
        selected_model, records, selection = select_model_results(records, preferred_model=model)
        selection_mode = str(selection.get("selection_mode", "explicit-model"))

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        run_meta = record.get("run_metadata", {})
        record_model = run_meta.get("model", "unknown-model")
        # Use engine_variant when present (spec-dec runs); fall back to engine_name
        variant = get_engine_variant(record)
        grouped[(record_model, record["scenario_name"], variant)].append(record)

    summary_rows: list[dict[str, Any]] = []
    for (row_model, scenario_name, engine_variant), items in sorted(grouped.items()):
        metrics_list = [item["metrics"] for item in items]
        spec_method = get_spec_method(items[0]) if items else None
        summary_rows.append(
            {
                "model": row_model,
                "scenario_name": scenario_name,
                "engine_name": engine_variant,
                "engine_variant": engine_variant,
                "spec_method": spec_method,
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
        "available_models": available_models,
        "selected_model": selected_model,
        "selection_mode": selection_mode,
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

    if summary.get("selected_model"):
        lines.extend(
            [
                f"Model filter: `{summary['selected_model']}`",
                f"Selection mode: `{summary.get('selection_mode', 'explicit-model')}`",
                "",
            ]
        )
    elif summary.get("available_models"):
        lines.extend(
            [
                f"Models detected: {', '.join(f'`{model}`' for model in summary['available_models'])}",
                "",
            ]
        )

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


def generate_final_report(
    results_dir: Path, output_path: Path, model: str | None = None
) -> dict[str, Any]:
    summary = aggregate_results(results_dir, model=model)
    output_path.write_text(render_markdown(summary))
    return summary
