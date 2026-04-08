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
        "_raw_records": records,  # kept for saturation analysis; not serialised to JSON
    }


def _compute_saturation(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    For throughput_ramp and throughput_ramp_extended result files, partition requests
    by concurrency level (using scenario_config) and compute per-level throughput.

    Throughput per level ≈ (total output tokens × concurrency) / sum(total_ms / 1000)
    This is an approximation for a semaphore-bounded concurrent workload.

    Returns list of rows: {model, engine, scenario, concurrency, tokens_per_sec, requests}
    """
    rows: list[dict[str, Any]] = []
    ramp_scenarios = {"throughput_ramp", "throughput_ramp_extended"}

    for rec in records:
        if rec.get("scenario_name") not in ramp_scenarios:
            continue
        cfg = rec.get("scenario_config", {})
        concurrency_levels = cfg.get("concurrency_levels", [])
        rpl = cfg.get("requests_per_level", 100)
        if not concurrency_levels or not rpl:
            continue

        model_raw = rec.get("run_metadata", {}).get("model", "")
        from analysis import get_engine_variant
        engine = get_engine_variant(rec)
        model = model_raw.split("/")[-1] if model_raw else "unknown"
        scenario = rec.get("scenario_name", "unknown")

        sorted_requests = sorted(rec.get("requests", []), key=lambda r: r["request_id"])

        for i, concurrency in enumerate(concurrency_levels):
            level_reqs = sorted_requests[i * rpl : (i + 1) * rpl]
            successful = [r for r in level_reqs if r.get("success")]
            if not successful:
                continue
            total_tokens = sum(r.get("output_tokens", 0) for r in successful)
            total_ms = sum(r.get("total_ms", 0.0) for r in successful)
            if total_ms <= 0:
                continue
            # Approximation: concurrent throughput = (tokens × concurrency) / total_latency
            approx_tps = total_tokens * concurrency / (total_ms / 1000.0)
            rows.append(
                {
                    "model": model,
                    "engine": engine,
                    "scenario": scenario,
                    "concurrency": concurrency,
                    "tokens_per_sec": approx_tps,
                    "n_requests": len(successful),
                }
            )
    return rows


def _render_saturation(saturation_rows: list[dict[str, Any]]) -> list[str]:
    """Render a 'Saturation Analysis' section from per-level throughput rows."""
    if not saturation_rows:
        return []

    from collections import defaultdict

    lines: list[str] = [
        "## Saturation Analysis",
        "",
        "Per-level throughput from `throughput_ramp` and `throughput_ramp_extended` runs. "
        "Throughput is approximated as `(output tokens × concurrency) / Σ(total_ms)`. "
        "The **saturation point** is the lowest concurrency level at which throughput is within 5% of the run maximum.",
        "",
    ]

    # Group by (model, engine, scenario)
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in saturation_rows:
        grouped[(row["model"], row["engine"], row["scenario"])].append(row)

    # Collect all concurrency levels seen
    all_concurrencies: list[int] = sorted({r["concurrency"] for r in saturation_rows})
    conc_header = " | ".join(f"C={c}" for c in all_concurrencies)
    conc_sep = " | ".join("------" for _ in all_concurrencies)

    lines.append(f"| Model | Engine | Scenario | {conc_header} | Saturation point |")
    lines.append(f"|-------|--------|----------|{conc_sep}|-----------------|")

    # Aggregate multiple runs by mean
    for key in sorted(grouped.keys()):
        model, engine, scenario = key
        level_rows = grouped[key]

        # Average across multiple runs at the same concurrency
        from statistics import mean as _mean
        by_conc: dict[int, list[float]] = defaultdict(list)
        for r in level_rows:
            by_conc[r["concurrency"]].append(r["tokens_per_sec"])
        avg_tps = {c: _mean(vs) for c, vs in by_conc.items()}

        max_tps = max(avg_tps.values()) if avg_tps else 1.0
        sat_point = "—"
        for c in all_concurrencies:
            tps = avg_tps.get(c, 0.0)
            if tps >= max_tps * 0.95:
                sat_point = f"C={c}"
                break

        cells = []
        for c in all_concurrencies:
            tps = avg_tps.get(c)
            if tps is None:
                cells.append("—")
            else:
                cells.append(f"{tps:.0f}")
        cells_str = " | ".join(cells)
        lines.append(f"| {model} | {engine} | {scenario} | {cells_str} | {sat_point} |")

    lines.append("")
    return lines


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

    # Saturation analysis (requires access to raw records stored in summary)
    raw_records = summary.get("_raw_records", [])
    if raw_records:
        saturation_rows = _compute_saturation(raw_records)
        lines.extend(_render_saturation(saturation_rows))

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
