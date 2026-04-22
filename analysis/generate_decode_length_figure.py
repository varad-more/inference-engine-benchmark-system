"""Generate the decode-length sweep figure.

Plots tokens/sec and TTFT p50 as a function of max_output_tokens for each
(model, engine) across the 4 sweep points {64, 256, 1024, 4096}. Iterations
are aggregated by mean with 95% CI error bars.

Output: reports/figures/decode_length_sweep.svg
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from analysis import _figure_style as style
from analysis.decode_length_analysis import compute_sweep_stats, load_results

RESULTS_DIR = Path("results_decode_sweep")
OUTPUT = Path("reports/figures/decode_length_sweep.svg")

MODEL_DISPLAY = {
    "gemma-2-2b-it": "Gemma 2 2B",
    "phi-4-mini-instruct": "Phi-4 mini",
    "gemma-3-4b-it": "Gemma 3 4B",
    "llama-3-1-8b-instruct": "Llama 3.1 8B",
    "gemma-4-e2b-it": "Gemma 4 E2B",
    "gemma-4-e4b-it": "Gemma 4 E4B",
}
MODEL_ORDER = [
    "gemma-2-2b-it",
    "phi-4-mini-instruct",
    "gemma-3-4b-it",
    "llama-3-1-8b-instruct",
    "gemma-4-e2b-it",
    "gemma-4-e4b-it",
]
SWEEP_LENGTHS = [64, 256, 1024, 4096]
ENGINE_COLOURS = {"vllm": style.VLLM, "sglang": style.SGLANG}


def _ci95(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    se = statistics.stdev(values) / math.sqrt(n)
    return float(scipy_stats.t.ppf(0.975, df=n - 1) * se)


def _series(stats: dict, model: str, engine: str, metric: str) -> tuple[list[float], list[float]]:
    means, cis = [], []
    for length in SWEEP_LENGTHS:
        cell = stats.get((model, engine, length), {}).get(metric, {})
        means.append(float(cell.get("mean") or math.nan))
        vals = cell.get("values", []) or []
        cis.append(_ci95(list(vals)))
    return means, cis


def main() -> None:
    style.apply()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    records = load_results(RESULTS_DIR)
    stats = compute_sweep_stats(records)

    models_with_data = [
        m
        for m in MODEL_ORDER
        if any((m, e, length) in stats for e in ENGINE_COLOURS for length in SWEEP_LENGTHS)
    ]

    n_cols = 3
    n_rows = (len(models_with_data) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.8 * n_rows), dpi=120, sharex=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, model in enumerate(models_with_data):
        ax = axes[i]
        for engine, colour in ENGINE_COLOURS.items():
            tps, ci = _series(stats, model, engine, "tokens_per_sec")
            if all(math.isnan(v) for v in tps):
                continue
            ax.errorbar(
                SWEEP_LENGTHS,
                tps,
                yerr=ci,
                marker="o",
                linewidth=1.8,
                markersize=6,
                capsize=3,
                color=colour,
                label=engine,
            )
            for x_val, y_val in zip(SWEEP_LENGTHS, tps):
                if not math.isnan(y_val):
                    ax.annotate(
                        f"{y_val:.0f}",
                        (x_val, y_val),
                        textcoords="offset points",
                        xytext=(0, 6),
                        ha="center",
                        fontsize=7.5,
                        color=colour,
                    )

        ax.set_xscale("log")
        ax.set_xticks(SWEEP_LENGTHS)
        ax.set_xticklabels([str(x) for x in SWEEP_LENGTHS])
        ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if i % n_cols == 0:
            ax.set_ylabel("Tokens / sec")
        if i >= len(models_with_data) - n_cols:
            ax.set_xlabel("max_output_tokens")
        ax.legend(loc="best", fontsize=8, framealpha=0.9, facecolor=style.BG, labelcolor=style.FG)

    # Hide any unused axes
    for j in range(len(models_with_data), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Decode-Length Sweep: Tokens/sec vs max_output_tokens",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.96,
        "Prompt ≈ 512 tokens · concurrency 8 · 180 requests/run · mean of n=3 iterations with 95% CI error bars",
        ha="center",
        fontsize=9,
        color=style.MUTED,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT, format="svg")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
