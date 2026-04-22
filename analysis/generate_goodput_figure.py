"""Generate the goodput comparison figure.

Horizontal grouped bar chart of goodput (req/s) per model, comparing vLLM vs
SGLang under a pair of default SLOs (TTFT ≤ 100 ms, TPOT ≤ 35 ms). Matches
the thresholds used in `reports/goodput_slo100_35.md`.

Output: reports/figures/goodput.svg
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis import _figure_style as style
from analysis.goodput import compute_goodput, load_results

RESULTS_DIR = Path("results")
OUTPUT = Path("reports/figures/goodput.svg")

TTFT_SLO_MS = 100.0
TPOT_SLO_MS = 35.0


def main() -> None:
    style.apply()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    records = load_results(RESULTS_DIR)
    goodput = compute_goodput(records, ttft_slo_ms=TTFT_SLO_MS, tpot_slo_ms=TPOT_SLO_MS)

    # Group per model
    models: dict[str, dict[str, float]] = {}
    pass_rates: dict[str, dict[str, float]] = {}
    for (model, engine), v in goodput.items():
        models.setdefault(model, {})[engine] = float(v.get("goodput_rps", 0.0))
        pass_rates.setdefault(model, {})[engine] = float(v.get("slo_pass_rate", 0.0)) * 100.0

    # Drop models that have zero for both engines (usually oversized / non-qualifying)
    ordered = sorted(
        models.items(),
        key=lambda kv: max(kv[1].values() or [0.0]),
        reverse=True,
    )

    labels = [m for m, _ in ordered]
    vllm_vals = [models[m].get("vllm", 0.0) for m, _ in ordered]
    sgl_vals = [models[m].get("sglang", 0.0) for m, _ in ordered]
    vllm_pass = [pass_rates[m].get("vllm", 0.0) for m, _ in ordered]
    sgl_pass = [pass_rates[m].get("sglang", 0.0) for m, _ in ordered]

    y = np.arange(len(labels))
    h = 0.4

    fig, ax = plt.subplots(figsize=(12, max(6, 0.42 * len(labels) + 2)), dpi=120)

    b_vllm = ax.barh(
        y - h / 2, vllm_vals, h, label="vLLM", color=style.VLLM, edgecolor=style.BG, linewidth=0.5
    )
    b_sgl = ax.barh(
        y + h / 2,
        sgl_vals,
        h,
        label="SGLang",
        color=style.SGLANG,
        edgecolor=style.BG,
        linewidth=0.5,
    )

    xmax = max([*vllm_vals, *sgl_vals, 0.01]) * 1.2
    ax.set_xlim(0, xmax)

    for bar, val, pass_rate in zip(b_vllm, vllm_vals, vllm_pass):
        if val > 0:
            ax.text(
                val + xmax * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f} rps · {pass_rate:.0f}%",
                va="center",
                fontsize=8,
                color=style.FG,
            )
    for bar, val, pass_rate in zip(b_sgl, sgl_vals, sgl_pass):
        if val > 0:
            ax.text(
                val + xmax * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f} rps · {pass_rate:.0f}%",
                va="center",
                fontsize=8,
                color=style.FG,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Goodput (qualifying requests / sec, summed across all scenarios)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.set_title(
        f"Goodput — TTFT ≤ {TTFT_SLO_MS:.0f} ms AND TPOT ≤ {TPOT_SLO_MS:.0f} ms",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.text(
        0.5,
        1.04,
        "Percentage after each bar is the SLO pass-rate: qualifying ÷ total successful requests.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.5,
        color=style.MUTED,
    )
    ax.legend(
        loc="lower right", fontsize=9, framealpha=0.9, facecolor=style.BG, labelcolor=style.FG
    )

    fig.tight_layout()
    fig.savefig(OUTPUT, format="svg")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
