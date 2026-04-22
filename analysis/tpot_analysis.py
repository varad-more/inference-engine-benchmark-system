"""
TPOT (Time Per Output Token) analysis across all benchmark result files.

TPOT is computed per request as:
    tpot_ms = (total_ms - ttft_ms) / max(output_tokens - 1, 1)

This represents the average inter-token latency after the first token, which
is the standard definition used in LLM serving benchmarks.

Usage:
    python -m analysis.tpot_analysis [--results-dir results] [--output reports/tpot_analysis.md]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentile(sorted_vals: list[float], p: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    idx = (p / 100.0) * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def _tpot(total_ms: float, ttft_ms: float, output_tokens: int) -> float:
    """Inter-token latency after first token, in ms."""
    decode_ms = total_ms - ttft_ms
    decode_tokens = max(output_tokens - 1, 1)
    return decode_ms / decode_tokens


def _normalise_engine(raw: str) -> str:
    """Collapse VLLMClient/vllm -> vllm, SGLangClient/sglang -> sglang."""
    low = raw.lower()
    if "vllm" in low:
        return "vllm"
    if "sglang" in low or "sgl" in low:
        return "sglang"
    return raw


def _model_slug(model_str: str) -> str:
    """google/gemma-2-2b-it -> gemma-2-2b-it."""
    return model_str.split("/")[-1] if model_str else "unknown"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_results(results_dir: Path) -> list[dict]:
    """Return list of parsed JSON dicts from all non-manifest result files."""
    records = []
    for path in sorted(results_dir.rglob("*.json")):
        name = path.stem
        # Skip manifests and comparison rollups — they lack per-request data
        if name.startswith("matrix_manifest") or name.startswith("comparison_"):
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        # Must have scenario_name and requests array
        if "requests" not in data or "scenario_name" not in data:
            continue
        data["_source_path"] = str(path)
        records.append(data)
    return records


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def compute_tpot_stats(
    records: list[dict],
) -> dict[tuple[str, str, str], dict]:
    """
    Returns a dict keyed by (model_slug, engine, scenario_name) ->
      {p50, p95, p99, mean, stdev, count, sample_output_tokens_mean}
    """
    # Accumulate per-request TPOT values per group
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    token_counts: dict[tuple[str, str, str], list[int]] = defaultdict(list)

    for rec in records:
        model_raw = rec.get("run_metadata", {}).get("model", "")
        # Fall back to directory-level slug if model not in metadata
        if not model_raw:
            src = Path(rec["_source_path"])
            model_raw = src.parent.name  # directory slug
        model = _model_slug(model_raw)
        engine = _normalise_engine(rec.get("engine_name", "unknown"))
        scenario = rec.get("scenario_name", "unknown")
        key = (model, engine, scenario)

        for req in rec.get("requests", []):
            if not req.get("success", False):
                continue
            total_ms = float(req.get("total_ms", 0.0))
            ttft_ms = float(req.get("ttft_ms", 0.0))
            output_tokens = int(req.get("output_tokens", 0))
            if output_tokens < 1 or total_ms <= 0:
                continue
            groups[key].append(_tpot(total_ms, ttft_ms, output_tokens))
            token_counts[key].append(output_tokens)

    result = {}
    for key, tpots in groups.items():
        if not tpots:
            continue
        s = sorted(tpots)
        result[key] = {
            "p50": _percentile(s, 50),
            "p95": _percentile(s, 95),
            "p99": _percentile(s, 99),
            "mean": statistics.mean(s),
            "stdev": statistics.stdev(s) if len(s) > 1 else 0.0,
            "count": len(s),
            "mean_output_tokens": statistics.mean(token_counts[key]),
        }
    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

SCENARIO_ORDER = [
    "single_request_latency",
    "throughput_ramp",
    "long_context_stress",
    "prefix_sharing_benefit",
    "structured_generation_speed",
]


def _scenario_sort_key(s: str) -> int:
    try:
        return SCENARIO_ORDER.index(s)
    except ValueError:
        return len(SCENARIO_ORDER)


def render_markdown(stats: dict[tuple[str, str, str], dict]) -> str:
    """Render a markdown report from TPOT stats."""
    lines: list[str] = []
    lines.append("# TPOT Analysis Report")
    lines.append("")
    lines.append(
        "**TPOT** (Time Per Output Token) is the average inter-token decode latency "
        "after the first token: `(total_ms − ttft_ms) / max(output_tokens − 1, 1)`."
    )
    lines.append("")
    lines.append("Lower is better. P99 is the most conservative SLO-relevant metric.")
    lines.append("")

    # Group by scenario for readability
    by_scenario: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
    for (model, engine, scenario), v in stats.items():
        by_scenario[scenario].append((model, engine, v))

    scenarios_sorted = sorted(by_scenario.keys(), key=_scenario_sort_key)

    for scenario in scenarios_sorted:
        entries = by_scenario[scenario]
        # Sort by model then engine
        entries.sort(key=lambda x: (x[0], x[1]))

        lines.append(f"## {scenario.replace('_', ' ').title()}")
        lines.append("")
        lines.append(
            "| Model | Engine | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | Stdev | N | Mean Output Tokens |"
        )
        lines.append(
            "|-------|--------|----------|----------|----------|-----------|-------|---|--------------------|"
        )
        for model, engine, v in entries:
            lines.append(
                f"| {model} | {engine} "
                f"| {v['p50']:.2f} | {v['p95']:.2f} | {v['p99']:.2f} "
                f"| {v['mean']:.2f} | {v['stdev']:.2f} "
                f"| {v['count']} | {v['mean_output_tokens']:.1f} |"
            )
        lines.append("")

    # Cross-scenario engine comparison (aggregate across all scenarios)
    lines.append("## Engine Comparison (All Scenarios Aggregated)")
    lines.append("")
    lines.append("Aggregates TPOT samples from all scenarios for a high-level engine comparison.")
    lines.append("")

    engine_agg: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    for (model, engine, _scenario), v in stats.items():
        # We don't have raw samples here, but we can aggregate means weighted by count
        # as a proxy. Recomputing from scratch would need raw data passed through.
        engine_agg[(model, engine)].append((v["mean"], v["count"]))

    lines.append("| Model | Engine | Weighted Mean TPOT (ms) | Total Requests |")
    lines.append("|-------|--------|------------------------|----------------|")
    agg_rows = []
    for (model, engine), pairs in engine_agg.items():
        total_n = sum(n for _, n in pairs)
        weighted_mean = sum(m * n for m, n in pairs) / max(total_n, 1)
        agg_rows.append((model, engine, weighted_mean, total_n))
    agg_rows.sort(key=lambda r: (r[0], r[1]))
    for model, engine, wm, n in agg_rows:
        lines.append(f"| {model} | {engine} | {wm:.2f} | {n} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute P50/P95/P99 TPOT per (model, engine, scenario)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root directory containing benchmark result JSON files (default: results/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/tpot_analysis.md"),
        help="Output markdown file path (default: reports/tpot_analysis.md)",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        raise SystemExit(f"Results directory not found: {args.results_dir}")

    print(f"Loading results from: {args.results_dir}")
    records = load_results(args.results_dir)
    print(f"  Loaded {len(records)} result files")

    stats = compute_tpot_stats(records)
    print(f"  Computed TPOT stats for {len(stats)} (model, engine, scenario) groups")

    report = render_markdown(stats)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Report written to: {args.output}")

    # Print summary to stdout
    print()
    print("=== TPOT Summary (P50 / P95 / P99 ms) ===")
    rows = sorted(stats.items(), key=lambda kv: (_scenario_sort_key(kv[0][2]), kv[0][0], kv[0][1]))
    for (model, engine, scenario), v in rows:
        print(
            f"  {model:35s} {engine:8s} {scenario:35s}  "
            f"P50={v['p50']:7.2f}  P95={v['p95']:7.2f}  P99={v['p99']:7.2f}  n={v['count']}"
        )


if __name__ == "__main__":
    main()
