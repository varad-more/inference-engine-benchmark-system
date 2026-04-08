"""
Decode-length sweep analysis.

Reads results from results_decode_sweep/ (or --results-dir), groups by
(model, engine, max_output_tokens), and computes:
  - Mean tok/s ± 95% CI across iterations
  - Mean TTFT P50 ± 95% CI
  - Mean TPOT P50 ± 95% CI

Identifies the crossover point (if any) where one engine overtakes the other
as max_output_tokens increases, and whether TTFT advantages are preserved at
long-decode (>1K tokens) or engines converge.

Output: reports/decode_length_analysis.md

Usage:
    python -m analysis.decode_length_analysis
    python -m analysis.decode_length_analysis --results-dir results_decode_sweep
"""

from __future__ import annotations

import argparse
import json
import math
import re
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


def _tpot_p50_from_requests(requests: list[dict]) -> float:
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
    return _percentile(sorted(tpots), 50) if tpots else 0.0


def _ci95(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    se = statistics.stdev(values) / math.sqrt(n)
    return scipy_stats.t.ppf(0.975, df=n - 1) * se


def _normalise_engine(raw: str) -> str:
    low = raw.lower()
    if "vllm" in low:
        return "vllm"
    if "sglang" in low or "sgl" in low:
        return "sglang"
    return raw


def _model_slug(model_str: str) -> str:
    return model_str.split("/")[-1] if model_str else "unknown"


def _max_tokens_from_scenario(scenario_name: str, cfg: dict) -> int | None:
    """Extract max_output_tokens from scenario name or config."""
    # Try config first (most reliable)
    v = cfg.get("max_output_tokens")
    if v:
        return int(v)
    # Fall back to name suffix: decode_length_sweep_1024
    m = re.search(r"decode_length_sweep_(\d+)", scenario_name)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_DECODE_SWEEP_PREFIX = "decode_length_sweep"


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
        if _DECODE_SWEEP_PREFIX not in data.get("scenario_name", ""):
            continue
        data["_source_path"] = str(path)
        records.append(data)
    return records


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_sweep_stats(
    records: list[dict],
) -> dict[tuple[str, str, int], dict[str, dict]]:
    """
    Returns dict keyed by (model_slug, engine, max_output_tokens) ->
      {
        "tokens_per_sec": {mean, ci95, n, values},
        "ttft_p50_ms":    {mean, ci95, n, values},
        "tpot_p50_ms":    {mean, ci95, n, values},
      }
    """
    groups: dict[tuple[str, str, int], dict[str, list[float]]] = defaultdict(
        lambda: {"tokens_per_sec": [], "ttft_p50_ms": [], "tpot_p50_ms": []}
    )

    for rec in records:
        model_raw = rec.get("run_metadata", {}).get("model", "")
        if not model_raw:
            model_raw = Path(rec["_source_path"]).parent.name
        model = _model_slug(model_raw)
        engine = _normalise_engine(rec.get("engine_name", "unknown"))

        max_tokens = _max_tokens_from_scenario(
            rec.get("scenario_name", ""), rec.get("scenario_config", {})
        )
        if max_tokens is None:
            continue

        key = (model, engine, max_tokens)
        m = rec.get("metrics", {})
        tps = m.get("throughput", {}).get("tokens_per_sec", 0.0)
        ttft = m.get("ttft", {}).get("p50", 0.0)
        tpot = _tpot_p50_from_requests(rec.get("requests", []))

        if tps > 0:
            groups[key]["tokens_per_sec"].append(tps)
        if ttft > 0:
            groups[key]["ttft_p50_ms"].append(ttft)
        if tpot > 0:
            groups[key]["tpot_p50_ms"].append(tpot)

    result = {}
    for key, metric_lists in groups.items():
        result[key] = {}
        for metric, values in metric_lists.items():
            if not values:
                result[key][metric] = {"mean": 0.0, "ci95": 0.0, "n": 0, "values": []}
            else:
                result[key][metric] = {
                    "mean": statistics.mean(values),
                    "ci95": _ci95(values),
                    "n": len(values),
                    "values": values,
                }
    return result


def _find_crossover(
    stats: dict[tuple[str, str, int], dict],
    model: str,
    metric: str,
) -> str:
    """
    Identify the max_output_tokens value at which sglang overtakes vllm
    (or vice versa) for a given metric. Returns a human-readable string.
    """
    sweep_lengths = sorted(
        {k[2] for k in stats if k[0] == model}
    )
    if len(sweep_lengths) < 2:
        return "insufficient data"

    results_by_len: dict[int, dict[str, float]] = {}
    for length in sweep_lengths:
        vllm_v = stats.get((model, "vllm", length), {}).get(metric, {})
        sgl_v = stats.get((model, "sglang", length), {}).get(metric, {})
        v_mean = vllm_v.get("mean", 0.0)
        s_mean = sgl_v.get("mean", 0.0)
        if v_mean > 0 and s_mean > 0:
            results_by_len[length] = {"vllm": v_mean, "sglang": s_mean}

    if not results_by_len:
        return "insufficient data"

    # Determine who leads at the shortest length
    first = min(results_by_len)
    leader_at_first = "vllm" if results_by_len[first]["vllm"] >= results_by_len[first]["sglang"] else "sglang"

    for length in sweep_lengths[1:]:
        if length not in results_by_len:
            continue
        current_leader = (
            "vllm"
            if results_by_len[length]["vllm"] >= results_by_len[length]["sglang"]
            else "sglang"
        )
        if current_leader != leader_at_first:
            return f"{leader_at_first} → {current_leader} at max_tokens={length}"

    # Check convergence: if gap < 3% at largest length, report convergence
    last = max(results_by_len)
    v_last = results_by_len[last]["vllm"]
    s_last = results_by_len[last]["sglang"]
    gap_pct = abs(v_last - s_last) / max(v_last, s_last) * 100
    if gap_pct < 3.0:
        return f"converge at max_tokens={last} (gap {gap_pct:.1f}%)"

    winner = "vllm" if v_last > s_last else "sglang"
    return f"no crossover — {winner} leads throughout (gap {gap_pct:.1f}% at max_tokens={last})"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_SWEEP_LENGTHS = [64, 256, 1024, 4096]


def _fmt(v: dict, unit: str = "") -> str:
    if v.get("n", 0) == 0:
        return "—"
    ci = f" ± {v['ci95']:.1f}" if v.get("ci95", 0) > 0 else ""
    return f"{v['mean']:.1f}{ci}{unit}"


def render_markdown(stats: dict[tuple[str, str, int], dict]) -> str:
    lines: list[str] = []
    lines.append("# Decode-Length Sweep Analysis")
    lines.append("")
    lines.append(
        "Fixed ~512-token prompts, `max_output_tokens` swept across "
        f"{_SWEEP_LENGTHS}. "
        "Separates prefill-bound from decode-bound behaviour. "
        "Values show mean ± 95% CI across iterations."
    )
    lines.append("")

    models = sorted({k[0] for k in stats})

    for model in models:
        lines.append(f"## {model}")
        lines.append("")

        # Throughput table
        lines.append("### Throughput (tok/s)")
        lines.append("")
        header = "| max_tokens | vLLM tok/s | SGLang tok/s | Winner |"
        sep    = "|-----------|-----------|-------------|--------|"
        lines.append(header)
        lines.append(sep)
        for length in _SWEEP_LENGTHS:
            vllm_v = stats.get((model, "vllm", length), {}).get("tokens_per_sec", {})
            sgl_v  = stats.get((model, "sglang", length), {}).get("tokens_per_sec", {})
            v_mean = vllm_v.get("mean", 0.0)
            s_mean = sgl_v.get("mean", 0.0)
            if v_mean > 0 and s_mean > 0:
                gap = abs(v_mean - s_mean) / max(v_mean, s_mean) * 100
                if gap < 2:
                    winner = "Tie"
                elif v_mean > s_mean:
                    winner = f"vLLM +{gap:.0f}%"
                else:
                    winner = f"SGLang +{gap:.0f}%"
            else:
                winner = "—"
            lines.append(
                f"| {length} | {_fmt(vllm_v)} | {_fmt(sgl_v)} | {winner} |"
            )
        lines.append("")

        # TTFT table
        lines.append("### TTFT P50 (ms)")
        lines.append("")
        lines.append("| max_tokens | vLLM TTFT | SGLang TTFT |")
        lines.append("|-----------|-----------|-------------|")
        for length in _SWEEP_LENGTHS:
            vllm_v = stats.get((model, "vllm", length), {}).get("ttft_p50_ms", {})
            sgl_v  = stats.get((model, "sglang", length), {}).get("ttft_p50_ms", {})
            lines.append(f"| {length} | {_fmt(vllm_v, ' ms')} | {_fmt(sgl_v, ' ms')} |")
        lines.append("")

        # TPOT table
        lines.append("### TPOT P50 (ms/token)")
        lines.append("")
        lines.append("| max_tokens | vLLM TPOT | SGLang TPOT |")
        lines.append("|-----------|-----------|-------------|")
        for length in _SWEEP_LENGTHS:
            vllm_v = stats.get((model, "vllm", length), {}).get("tpot_p50_ms", {})
            sgl_v  = stats.get((model, "sglang", length), {}).get("tpot_p50_ms", {})
            lines.append(f"| {length} | {_fmt(vllm_v, ' ms')} | {_fmt(sgl_v, ' ms')} |")
        lines.append("")

        # Crossover / convergence findings
        tps_cross = _find_crossover(stats, model, "tokens_per_sec")
        ttft_cross = _find_crossover(stats, model, "ttft_p50_ms")

        lines.append("### Findings")
        lines.append("")
        lines.append(f"- **Throughput crossover:** {tps_cross}")
        lines.append(f"- **TTFT crossover:** {ttft_cross}")

        # TTFT advantage at long decode
        vllm_ttft_long = stats.get((model, "vllm", 4096), {}).get("ttft_p50_ms", {})
        sgl_ttft_long  = stats.get((model, "sglang", 4096), {}).get("ttft_p50_ms", {})
        if vllm_ttft_long.get("n", 0) > 0 and sgl_ttft_long.get("n", 0) > 0:
            v_ttft = vllm_ttft_long["mean"]
            s_ttft = sgl_ttft_long["mean"]
            gap = abs(v_ttft - s_ttft) / max(v_ttft, s_ttft) * 100
            leader = "vLLM" if v_ttft < s_ttft else "SGLang"
            if gap < 3:
                lines.append(
                    f"- **TTFT at max_tokens=4096:** converged (gap {gap:.1f}%) — "
                    "TTFT advantage does not persist at long decode."
                )
            else:
                lines.append(
                    f"- **TTFT at max_tokens=4096:** {leader} still leads by {gap:.1f}% — "
                    "TTFT advantage is preserved at long decode."
                )
        lines.append("")

    # Cross-model summary
    lines.append("## Summary: TTFT Advantage at Long Decode (max_tokens=4096)")
    lines.append("")
    lines.append(
        "Does vLLM's TTFT advantage (seen at concurrency=1) survive high output-token budgets? "
        "TTFT is determined at the prefill stage, so it should be independent of max_tokens — "
        "any divergence here indicates system-level effects (KV memory pressure, scheduler backpressure)."
    )
    lines.append("")
    lines.append("| Model | vLLM TTFT @4096 | SGLang TTFT @4096 | Preserved? |")
    lines.append("|-------|----------------|------------------|------------|")
    for model in models:
        vllm_v = stats.get((model, "vllm", 4096), {}).get("ttft_p50_ms", {})
        sgl_v  = stats.get((model, "sglang", 4096), {}).get("ttft_p50_ms", {})
        if vllm_v.get("n", 0) > 0 and sgl_v.get("n", 0) > 0:
            gap = abs(vllm_v["mean"] - sgl_v["mean"]) / max(vllm_v["mean"], sgl_v["mean"]) * 100
            preserved = "Yes" if gap >= 3 else "No (converged)"
        else:
            preserved = "—"
        lines.append(f"| {model} | {_fmt(vllm_v, ' ms')} | {_fmt(sgl_v, ' ms')} | {preserved} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse decode-length sweep results (tok/s, TTFT, TPOT vs max_output_tokens)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_decode_sweep"),
        help="Directory containing decode-sweep result JSON files (default: results_decode_sweep/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/decode_length_analysis.md"),
        help="Output markdown path (default: reports/decode_length_analysis.md)",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        raise SystemExit(
            f"Results directory not found: {args.results_dir}\n"
            "Run scripts/run_decode_sweep.sh first."
        )

    print(f"Loading results from: {args.results_dir}")
    records = load_results(args.results_dir)
    print(f"  Loaded {len(records)} decode-sweep result files")

    if not records:
        raise SystemExit("No decode-length-sweep result files found.")

    stats = compute_sweep_stats(records)
    print(f"  Computed stats for {len(stats)} (model, engine, max_tokens) groups")

    report = render_markdown(stats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Report written to: {args.output}")

    # Stdout summary
    print()
    print("=== Crossover Summary ===")
    models = sorted({k[0] for k in stats})
    for model in models:
        cross = _find_crossover(stats, model, "tokens_per_sec")
        print(f"  {model:<35}  throughput: {cross}")


if __name__ == "__main__":
    main()
