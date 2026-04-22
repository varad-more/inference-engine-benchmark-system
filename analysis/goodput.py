"""
Goodput analysis: requests/sec that meet both a TTFT SLO and a TPOT SLO.

Goodput = (requests satisfying both SLOs) / wall_time_sec

TPOT per request = (total_ms - ttft_ms) / max(output_tokens - 1, 1)

Usage:
    python -m analysis.goodput --ttft-slo-ms 200 --tpot-slo-ms 30
    python -m analysis.goodput --ttft-slo-ms 500 --tpot-slo-ms 50 --results-dir results
    python -m analysis.goodput --ttft-slo-ms 200 --tpot-slo-ms 30 --scenario throughput_ramp
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers (shared with tpot_analysis but kept self-contained)
# ---------------------------------------------------------------------------


def _tpot(total_ms: float, ttft_ms: float, output_tokens: int) -> float:
    return (total_ms - ttft_ms) / max(output_tokens - 1, 1)


def _normalise_engine(raw: str) -> str:
    low = raw.lower()
    if "vllm" in low:
        return "vllm"
    if "sglang" in low or "sgl" in low:
        return "sglang"
    return raw


def _model_slug(model_str: str) -> str:
    return model_str.split("/")[-1] if model_str else "unknown"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results(results_dir: Path, scenario_filter: str | None = None) -> list[dict]:
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
        if scenario_filter and data.get("scenario_name") != scenario_filter:
            continue
        data["_source_path"] = str(path)
        records.append(data)
    return records


# ---------------------------------------------------------------------------
# Goodput computation
# ---------------------------------------------------------------------------


def compute_goodput(
    records: list[dict],
    ttft_slo_ms: float,
    tpot_slo_ms: float,
) -> dict[tuple[str, str], dict]:
    """
    Returns dict keyed by (model_slug, engine) ->
      {
        goodput_rps: float,          # qualifying requests / total wall time
        qualifying_requests: int,
        total_requests: int,
        total_wall_time_sec: float,
        slo_pass_rate: float,        # qualifying / total (successful)
        per_scenario: {scenario: {qualifying, total, wall_time_sec, goodput_rps}}
      }
    """
    # Accumulate per (model, engine) across scenarios
    agg: dict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "qualifying_requests": 0,
            "total_successful": 0,
            "total_wall_time_sec": 0.0,
            "per_scenario": defaultdict(
                lambda: {"qualifying": 0, "total": 0, "wall_time_sec": 0.0}
            ),
        }
    )

    for rec in records:
        model_raw = rec.get("run_metadata", {}).get("model", "")
        if not model_raw:
            model_raw = Path(rec["_source_path"]).parent.name
        model = _model_slug(model_raw)
        engine = _normalise_engine(rec.get("engine_name", "unknown"))
        scenario = rec.get("scenario_name", "unknown")
        key = (model, engine)

        wall_time_sec = rec.get("metrics", {}).get("throughput", {}).get("wall_time_sec", 0.0)
        # Guard against zero/negative wall times (shouldn't happen in valid files)
        if wall_time_sec <= 0:
            continue

        qualifying = 0
        total_successful = 0
        for req in rec.get("requests", []):
            if not req.get("success", False):
                continue
            total_ms = float(req.get("total_ms", 0.0))
            ttft_ms = float(req.get("ttft_ms", 0.0))
            output_tokens = int(req.get("output_tokens", 0))
            if total_ms <= 0 or output_tokens < 1:
                continue

            total_successful += 1
            tpot_ms = _tpot(total_ms, ttft_ms, output_tokens)
            if ttft_ms <= ttft_slo_ms and tpot_ms <= tpot_slo_ms:
                qualifying += 1

        agg[key]["qualifying_requests"] += qualifying
        agg[key]["total_successful"] += total_successful
        agg[key]["total_wall_time_sec"] += wall_time_sec

        ps = agg[key]["per_scenario"][scenario]
        ps["qualifying"] += qualifying
        ps["total"] += total_successful
        ps["wall_time_sec"] += wall_time_sec

    result = {}
    for key, v in agg.items():
        wall = v["total_wall_time_sec"]
        q = v["qualifying_requests"]
        tot = v["total_successful"]
        per_scenario = {}
        for sc, sv in v["per_scenario"].items():
            sw = sv["wall_time_sec"]
            per_scenario[sc] = {
                "qualifying": sv["qualifying"],
                "total": sv["total"],
                "wall_time_sec": sw,
                "goodput_rps": sv["qualifying"] / sw if sw > 0 else 0.0,
                "slo_pass_rate": sv["qualifying"] / sv["total"] if sv["total"] > 0 else 0.0,
            }
        result[key] = {
            "goodput_rps": q / wall if wall > 0 else 0.0,
            "qualifying_requests": q,
            "total_successful": tot,
            "total_wall_time_sec": wall,
            "slo_pass_rate": q / tot if tot > 0 else 0.0,
            "per_scenario": per_scenario,
        }
    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_table(
    goodput: dict[tuple[str, str], dict],
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    scenario_filter: str | None,
) -> str:
    lines: list[str] = []
    lines.append("# Goodput Analysis")
    lines.append("")
    lines.append(f"**SLO thresholds:** TTFT ≤ {ttft_slo_ms:.0f} ms, TPOT ≤ {tpot_slo_ms:.1f} ms")
    lines.append("")
    if scenario_filter:
        lines.append(f"**Scenario filter:** `{scenario_filter}`")
        lines.append("")
    lines.append(
        "**Goodput** = qualifying requests (meeting both SLOs) ÷ total wall-clock time. "
        "Higher is better."
    )
    lines.append("")

    # Aggregate table
    lines.append("## Aggregate Goodput per (Model, Engine)")
    lines.append("")
    lines.append(
        "| Model | Engine | Goodput (req/s) | SLO Pass Rate | Qualifying | Total Successful | Wall Time (s) |"
    )
    lines.append(
        "|-------|--------|-----------------|---------------|-----------|-----------------|---------------|"
    )

    rows = sorted(goodput.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    for (model, engine), v in rows:
        lines.append(
            f"| {model} | {engine} "
            f"| {v['goodput_rps']:.4f} "
            f"| {v['slo_pass_rate'] * 100:.1f}% "
            f"| {v['qualifying_requests']} "
            f"| {v['total_successful']} "
            f"| {v['total_wall_time_sec']:.1f} |"
        )
    lines.append("")

    # Per-scenario breakdown
    lines.append("## Per-Scenario Breakdown")
    lines.append("")

    # Collect all scenarios present
    all_scenarios: set[str] = set()
    for v in goodput.values():
        all_scenarios.update(v["per_scenario"].keys())

    SCENARIO_ORDER = [
        "single_request_latency",
        "throughput_ramp",
        "long_context_stress",
        "prefix_sharing_benefit",
        "structured_generation_speed",
    ]

    def _sc_key(s: str) -> int:
        try:
            return SCENARIO_ORDER.index(s)
        except ValueError:
            return len(SCENARIO_ORDER)

    for scenario in sorted(all_scenarios, key=_sc_key):
        lines.append(f"### {scenario.replace('_', ' ').title()}")
        lines.append("")
        lines.append(
            "| Model | Engine | Goodput (req/s) | SLO Pass Rate | Qualifying | Total | Wall Time (s) |"
        )
        lines.append(
            "|-------|--------|-----------------|---------------|-----------|-------|---------------|"
        )
        sc_rows = []
        for (model, engine), v in goodput.items():
            if scenario not in v["per_scenario"]:
                continue
            sv = v["per_scenario"][scenario]
            sc_rows.append((model, engine, sv))
        sc_rows.sort(key=lambda r: (r[0], r[1]))
        for model, engine, sv in sc_rows:
            lines.append(
                f"| {model} | {engine} "
                f"| {sv['goodput_rps']:.4f} "
                f"| {sv['slo_pass_rate'] * 100:.1f}% "
                f"| {sv['qualifying']} "
                f"| {sv['total']} "
                f"| {sv['wall_time_sec']:.1f} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute goodput (req/s meeting TTFT + TPOT SLOs) per (model, engine)"
    )
    parser.add_argument(
        "--ttft-slo-ms",
        type=float,
        required=True,
        help="Maximum acceptable TTFT in milliseconds",
    )
    parser.add_argument(
        "--tpot-slo-ms",
        type=float,
        required=True,
        help="Maximum acceptable TPOT (inter-token latency) in milliseconds",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root directory containing benchmark result JSON files (default: results/)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Filter to a single scenario name (e.g. throughput_ramp)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write a markdown report (e.g. reports/goodput.md)",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        raise SystemExit(f"Results directory not found: {args.results_dir}")

    print(f"Loading results from: {args.results_dir}")
    records = load_results(args.results_dir, scenario_filter=args.scenario)
    print(f"  Loaded {len(records)} result files")

    goodput = compute_goodput(records, args.ttft_slo_ms, args.tpot_slo_ms)

    print()
    print(f"=== Goodput (TTFT ≤ {args.ttft_slo_ms:.0f} ms, TPOT ≤ {args.tpot_slo_ms:.1f} ms) ===")
    print()
    header = f"  {'Model':<35} {'Engine':<8}  {'Goodput (rps)':>14}  {'SLO%':>7}  {'Qualifying':>10}  {'Total':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for (model, engine), v in sorted(goodput.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        print(
            f"  {model:<35} {engine:<8}  {v['goodput_rps']:>14.4f}  "
            f"{v['slo_pass_rate'] * 100:>6.1f}%  {v['qualifying_requests']:>10}  {v['total_successful']:>8}"
        )

    if args.output:
        report = render_table(goodput, args.ttft_slo_ms, args.tpot_slo_ms, args.scenario)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"\nReport written to: {args.output}")


if __name__ == "__main__":
    main()
