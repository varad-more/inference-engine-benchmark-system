"""Generate the variance / CV figure.

Horizontal bar chart of coefficient of variation (CV%) per (model, engine,
scenario, metric) for the 4-model variance subset. The CV threshold above
which a claim is flagged "unreliable" is drawn as a vertical line.

Output: reports/figures/variance_cv.svg
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis import _figure_style as style
from analysis.variance_analysis import compute_variance_stats, load_results

RESULTS_DIR = Path("results_variance")
OUTPUT = Path("reports/figures/variance_cv.svg")
CV_THRESHOLD = 5.0  # matches variance_analysis._HIGH_VARIANCE_THRESHOLD

METRICS = [
    ("ttft_p50_ms", "TTFT p50"),
    ("ttft_p95_ms", "TTFT p95"),
    ("tokens_per_sec", "Tokens/s"),
    ("tpot_p95_ms", "TPOT p95"),
]
SCENARIO_ORDER = [
    ("single_request_latency", "Single Req"),
    ("throughput_ramp", "Throughput Ramp"),
    ("long_context_stress", "Long Context"),
    ("prefix_sharing_benefit", "Prefix Share"),
    ("structured_generation_speed", "Structured"),
]
ENGINE_COLOURS = {"vllm": style.VLLM, "sglang": style.SGLANG}


def main() -> None:
    style.apply()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    records = load_results(RESULTS_DIR)
    stats = compute_variance_stats(records)

    n_cols = len(METRICS)
    n_rows = len(SCENARIO_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 2.4 * n_rows), dpi=120, sharex=False)
    if n_rows == 1:
        axes = [axes]

    max_cv = 1.0  # seed min scale
    for (scenario_key, _scenario_label), row in zip(SCENARIO_ORDER, axes):
        for (metric_key, _metric_label), ax in zip(METRICS, row):
            for _, (model, engine, sc), v in ((i, k, stats[k]) for i, k in enumerate(stats)):
                if sc != scenario_key:
                    continue
                cv = v.get(metric_key, {}).get("cv_pct", 0.0)
                max_cv = max(max_cv, cv)

    for ri, (scenario_key, scenario_label) in enumerate(SCENARIO_ORDER):
        row_keys = [k for k in stats if k[2] == scenario_key]
        # Sort by (model, engine) for stable layout
        row_keys.sort(key=lambda k: (k[0].lower(), k[1]))
        labels = [f"{model} · {engine}" for (model, engine, _) in row_keys]

        for ci, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[ri][ci]
            if not row_keys:
                ax.set_visible(False)
                continue

            values = [stats[k].get(metric_key, {}).get("cv_pct", 0.0) for k in row_keys]
            colours = [ENGINE_COLOURS.get(k[1], style.MUTED) for k in row_keys]

            y = np.arange(len(row_keys))
            ax.barh(y, values, color=colours, edgecolor=style.BG, linewidth=0.5, height=0.7)
            for yi, val in zip(y, values):
                if val > 0:
                    ax.text(
                        val + max_cv * 0.01,
                        yi,
                        f"{val:.1f}%",
                        va="center",
                        fontsize=7,
                        color=style.FG,
                    )

            ax.axvline(CV_THRESHOLD, color=style.EAGLE, linestyle="--", linewidth=1.0, alpha=0.7)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.grid(axis="x", linestyle="--", alpha=0.3)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.set_xlim(0, max(CV_THRESHOLD * 1.4, max_cv * 1.15))

            if ri == 0:
                ax.set_title(metric_label, fontsize=10, pad=6)
            if ci == 0:
                ax.set_ylabel(scenario_label, fontsize=10, labelpad=10)
            if ri == n_rows - 1:
                ax.set_xlabel("CV % (std / mean × 100)", fontsize=9)

    fig.suptitle(
        f"Variance across 5 iterations · CV% by scenario × metric  (⚠ threshold {CV_THRESHOLD:.0f}%)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.962,
        "Bars left of the dashed line → claim is reproducible; bars right of it → needs more iterations.  "
        "Source: results_variance/ (4 models × 5 iterations × 2 engines).",
        ha="center",
        fontsize=9,
        color=style.MUTED,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(OUTPUT, format="svg")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
