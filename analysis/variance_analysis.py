"""
Variance analysis over repeated benchmark runs.

Reads all result JSON files from a directory (default: results_variance/),
groups by (model, engine, scenario), and computes for each of:
  - TTFT P50, TTFT P95
  - Throughput (tok/s)
  - TPOT P95  (derived: (total_ms - ttft_ms) / max(output_tokens - 1, 1))

Per group statistics:
  - mean, std, 95% CI via t-distribution (scipy.stats.t), coefficient of variation (CV)
  - CV > 5% flagged as "HIGH VARIANCE — claim unreliable"

Outputs:
  - reports/variance_analysis.md  (full tables + CI-annotated headline table)

Usage:
    python -m analysis.variance_analysis
    python -m analysis.variance_analysis --results-dir results_variance --output reports/variance_analysis.md
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scipy import stats as scipy_stats


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


def _tpot_p95_from_requests(requests: list[dict]) -> float:
    """Compute TPOT P95 from a single run's request list."""
    tpots = []
    for req in requests:
        if not req.get("success", False):
            continue
        total_ms = float(req.get("total_ms", 0.0))
        ttft_ms = float(req.get("ttft_ms", 0.0))
        output_tokens = int(req.get("output_tokens", 0))
        if total_ms <= 0 or output_tokens < 1:
            continue
        tpots.append((total_ms - ttft_ms) / max(output_tokens - 1, 1))
    if not tpots:
        return 0.0
    return _percentile(sorted(tpots), 95)


def _normalise_engine(raw: str) -> str:
    low = raw.lower()
    if "vllm" in low:
        return "vllm"
    if "sglang" in low or "sgl" in low:
        return "sglang"
    return raw


def _model_slug(model_str: str) -> str:
    return model_str.split("/")[-1] if model_str else "unknown"


def _ci95(values: list[float]) -> float:
    """Return half-width of 95% CI using t-distribution. Returns 0 if n < 2."""
    n = len(values)
    if n < 2:
        return 0.0
    se = statistics.stdev(values) / math.sqrt(n)
    t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
    return t_crit * se


def _cv(values: list[float]) -> float:
    """Coefficient of variation (std/mean). Returns 0 if mean ≈ 0."""
    if len(values) < 2:
        return 0.0
    m = statistics.mean(values)
    if abs(m) < 1e-9:
        return 0.0
    return statistics.stdev(values) / m * 100.0  # as a percentage


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> list[dict]:
    records = []
    for path in sorted(results_dir.rglob("*.json")):
        name = path.stem
        if name.startswith("matrix_manifest") or name.startswith("comparison_"):
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if "requests" not in data or "scenario_name" not in data:
            continue
        data["_source_path"] = str(path)
        records.append(data)
    return records


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

_METRICS = ["ttft_p50_ms", "ttft_p95_ms", "tokens_per_sec", "tpot_p95_ms"]
_METRIC_LABELS = {
    "ttft_p50_ms": "TTFT P50 (ms)",
    "ttft_p95_ms": "TTFT P95 (ms)",
    "tokens_per_sec": "Throughput (tok/s)",
    "tpot_p95_ms": "TPOT P95 (ms)",
}
_HIGH_VARIANCE_THRESHOLD = 5.0  # CV%


def extract_run_metrics(record: dict) -> dict[str, float]:
    """Pull the four key metric values from a single result file."""
    m = record.get("metrics", {})
    return {
        "ttft_p50_ms": m.get("ttft", {}).get("p50", 0.0),
        "ttft_p95_ms": m.get("ttft", {}).get("p95", 0.0),
        "tokens_per_sec": m.get("throughput", {}).get("tokens_per_sec", 0.0),
        "tpot_p95_ms": _tpot_p95_from_requests(record.get("requests", [])),
    }


def compute_variance_stats(
    records: list[dict],
) -> dict[tuple[str, str, str], dict[str, dict]]:
    """
    Returns dict keyed by (model_slug, engine, scenario_name) ->
      {metric_name -> {mean, std, ci95, cv_pct, n, values, high_variance}}
    """
    # Group per-run metric observations
    groups: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {m: [] for m in _METRICS}
    )

    for rec in records:
        model_raw = rec.get("run_metadata", {}).get("model", "")
        if not model_raw:
            model_raw = Path(rec["_source_path"]).parent.name
        model = _model_slug(model_raw)
        engine = _normalise_engine(rec.get("engine_name", "unknown"))
        scenario = rec.get("scenario_name", "unknown")
        key = (model, engine, scenario)

        run_metrics = extract_run_metrics(rec)
        for metric_name, value in run_metrics.items():
            if value > 0:  # only count successful runs
                groups[key][metric_name].append(value)

    result: dict[tuple[str, str, str], dict[str, dict]] = {}
    for key, metric_lists in groups.items():
        result[key] = {}
        for metric_name, values in metric_lists.items():
            if not values:
                result[key][metric_name] = {
                    "mean": 0.0, "std": 0.0, "ci95": 0.0,
                    "cv_pct": 0.0, "n": 0, "values": [],
                    "high_variance": False,
                }
                continue
            m = statistics.mean(values)
            s = statistics.stdev(values) if len(values) > 1 else 0.0
            ci = _ci95(values)
            cv = _cv(values)
            result[key][metric_name] = {
                "mean": m,
                "std": s,
                "ci95": ci,
                "cv_pct": cv,
                "n": len(values),
                "values": values,
                "high_variance": cv > _HIGH_VARIANCE_THRESHOLD,
            }
    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

SCENARIO_ORDER = [
    "single_request_latency",
    "throughput_ramp",
    "throughput_ramp_extended",
    "long_context_stress",
    "prefix_sharing_benefit",
    "structured_generation_speed",
]


def _sc_key(s: str) -> int:
    try:
        return SCENARIO_ORDER.index(s)
    except ValueError:
        return len(SCENARIO_ORDER)


def _fmt_ci(v: dict) -> str:
    if v["n"] == 0:
        return "—"
    ci_str = f"± {v['ci95']:.1f}" if v["ci95"] > 0 else "—"
    hv = " ⚠" if v["high_variance"] else ""
    return f"{v['mean']:.1f} {ci_str}{hv}"


def _fmt_tps_ci(v: dict) -> str:
    """Throughput with ± CI, formatted as integer."""
    if v["n"] == 0:
        return "—"
    ci_str = f"± {v['ci95']:.1f}" if v["ci95"] > 0 else "—"
    hv = " ⚠" if v["high_variance"] else ""
    return f"{v['mean']:.1f} {ci_str}{hv}"


def render_markdown(
    stats: dict[tuple[str, str, str], dict[str, dict]],
) -> str:
    lines: list[str] = []
    lines.append("# Variance Analysis Report")
    lines.append("")
    lines.append(
        "**Methodology:** Each metric is sampled once per benchmark run. "
        "95% CIs use the t-distribution (`scipy.stats.t`). "
        f"**⚠ = CV > {_HIGH_VARIANCE_THRESHOLD}%** — claim unreliable, needs more iterations or investigation."
    )
    lines.append("")
    lines.append(
        "| Metric | Formula |"
    )
    lines.append("|--------|---------|")
    lines.append("| TTFT P50/P95 | from `metrics.ttft.p50/p95` |")
    lines.append("| Throughput | from `metrics.throughput.tokens_per_sec` |")
    lines.append("| TPOT P95 | `(total_ms − ttft_ms) / max(output_tokens − 1, 1)`, P95 across requests |")
    lines.append("")

    # Per-scenario tables
    by_scenario: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
    for (model, engine, scenario), metric_stats in stats.items():
        by_scenario[scenario].append((model, engine, metric_stats))

    for scenario in sorted(by_scenario.keys(), key=_sc_key):
        entries = by_scenario[scenario]
        entries.sort(key=lambda x: (x[0], x[1]))

        lines.append(f"## {scenario.replace('_', ' ').title()}")
        lines.append("")
        header = (
            "| Model | Engine | N runs "
            "| TTFT P50 (ms) | CV% "
            "| TTFT P95 (ms) | CV% "
            "| Tok/s | CV% "
            "| TPOT P95 (ms) | CV% |"
        )
        sep = "|-------|--------|--------|" + "---------------|-----|" * 4
        lines.append(header)
        lines.append(sep)

        for model, engine, ms in entries:
            n = max(v["n"] for v in ms.values()) if ms else 0
            tp50 = ms.get("ttft_p50_ms", {})
            tp95 = ms.get("ttft_p95_ms", {})
            tps = ms.get("tokens_per_sec", {})
            tpot = ms.get("tpot_p95_ms", {})

            def _cell(v: dict, fmt: str = ".1f") -> str:
                if v.get("n", 0) == 0:
                    return "—"
                hv = " ⚠" if v["high_variance"] else ""
                ci = f" ± {v['ci95']:{fmt}}" if v.get("ci95", 0) > 0 else ""
                return f"{v['mean']:{fmt}}{ci}{hv}"

            def _cv_cell(v: dict) -> str:
                if v.get("n", 0) < 2:
                    return "—"
                hv = "⚠" if v["high_variance"] else ""
                return f"{v['cv_pct']:.1f}%{hv}"

            lines.append(
                f"| {model} | {engine} | {n} "
                f"| {_cell(tp50)} | {_cv_cell(tp50)} "
                f"| {_cell(tp95)} | {_cv_cell(tp95)} "
                f"| {_cell(tps)} | {_cv_cell(tps)} "
                f"| {_cell(tpot)} | {_cv_cell(tpot)} |"
            )
        lines.append("")

    # Headline comparison table with CI bounds
    lines.append("## Headline Comparison Table (with 95% CI)")
    lines.append("")
    lines.append(
        "Replaces bare point estimates in the main README. "
        "Values from `single_request_latency` (TTFT) and `throughput_ramp` (tok/s)."
    )
    lines.append("")
    lines.append(
        "| Model | vLLM TTFT P50 | SGLang TTFT P50 | vLLM Peak tok/s | SGLang Peak tok/s |"
    )
    lines.append("|-------|---------------|-----------------|-----------------|-------------------|")

    # Collect per-model data for the headline table
    models = sorted({model for (model, _, _) in stats.keys()})
    for model in models:
        def _get(scenario: str, engine: str, metric: str) -> dict:
            return stats.get((model, engine, scenario), {}).get(metric, {"n": 0})

        vllm_ttft = _get("single_request_latency", "vllm", "ttft_p50_ms")
        sgl_ttft = _get("single_request_latency", "sglang", "ttft_p50_ms")
        vllm_tps = _get("throughput_ramp", "vllm", "tokens_per_sec")
        sgl_tps = _get("throughput_ramp", "sglang", "tokens_per_sec")

        def _headline(v: dict, fmt: str = ".1f") -> str:
            if v.get("n", 0) == 0:
                return "—"
            hv = " ⚠" if v.get("high_variance") else ""
            if v.get("ci95", 0) > 0:
                return f"{v['mean']:{fmt}} ± {v['ci95']:{fmt}}{hv}"
            return f"{v['mean']:{fmt}}{hv}"

        lines.append(
            f"| {model} "
            f"| {_headline(vllm_ttft)} ms "
            f"| {_headline(sgl_ttft)} ms "
            f"| {_headline(vllm_tps)} tok/s "
            f"| {_headline(sgl_tps)} tok/s |"
        )
    lines.append("")

    # Stability summary
    high_var_claims: list[str] = []
    for (model, engine, scenario), ms in stats.items():
        for metric_name, v in ms.items():
            if v.get("high_variance") and v.get("n", 0) >= 2:
                high_var_claims.append(
                    f"- **{model} / {engine} / {scenario}** — "
                    f"{_METRIC_LABELS.get(metric_name, metric_name)}: "
                    f"CV = {v['cv_pct']:.1f}% (mean {v['mean']:.1f}, std {v['std']:.1f})"
                )

    lines.append("## High-Variance Claims (CV > 5%) — Require Asterisk")
    lines.append("")
    if high_var_claims:
        lines.extend(sorted(high_var_claims))
    else:
        lines.append("_No metrics exceeded 5% CV — all headline claims are stable._")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute mean/std/95%CI/CV for benchmark metrics across repeated runs"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_variance"),
        help="Directory containing variance-run result JSON files (default: results_variance/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/variance_analysis.md"),
        help="Output markdown path (default: reports/variance_analysis.md)",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        raise SystemExit(
            f"Results directory not found: {args.results_dir}\n"
            "Run scripts/run_new_benchmarks.sh --phase1 first."
        )

    print(f"Loading results from: {args.results_dir}")
    records = load_results(args.results_dir)
    print(f"  Loaded {len(records)} result files")

    if not records:
        raise SystemExit("No result files found.")

    stats = compute_variance_stats(records)
    print(f"  Computed variance stats for {len(stats)} (model, engine, scenario) groups")

    report = render_markdown(stats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Report written to: {args.output}")

    # Stdout summary
    print()
    print("=== Variance Summary ===")
    high_var = [
        (key, mn, v)
        for key, ms in stats.items()
        for mn, v in ms.items()
        if v.get("high_variance") and v.get("n", 0) >= 2
    ]
    if high_var:
        print(f"  ⚠  {len(high_var)} high-variance metrics (CV > {_HIGH_VARIANCE_THRESHOLD}%):")
        for (model, engine, scenario), mn, v in sorted(high_var):
            print(f"     {model}/{engine}/{scenario} — {mn}: CV={v['cv_pct']:.1f}%")
    else:
        print("  All metrics stable (CV ≤ 5%)")
    print()
    print("=== Headline CI Table ===")
    models = sorted({m for (m, _, _) in stats.keys()})
    for model in models:
        vllm_ttft = stats.get((model, "vllm", "single_request_latency"), {}).get("ttft_p50_ms", {})
        sgl_ttft = stats.get((model, "sglang", "single_request_latency"), {}).get("ttft_p50_ms", {})
        vllm_tps = stats.get((model, "vllm", "throughput_ramp"), {}).get("tokens_per_sec", {})
        sgl_tps = stats.get((model, "sglang", "throughput_ramp"), {}).get("tokens_per_sec", {})

        def _s(v: dict) -> str:
            if not v or v.get("n", 0) == 0:
                return "n/a"
            ci = f" ± {v['ci95']:.1f}" if v.get("ci95", 0) > 0 else ""
            return f"{v['mean']:.1f}{ci}"

        print(f"  {model:<35}  TTFT vllm={_s(vllm_ttft)} sgl={_s(sgl_ttft)}  "
              f"tok/s vllm={_s(vllm_tps)} sgl={_s(sgl_tps)}")


if __name__ == "__main__":
    main()
