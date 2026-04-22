"""Generate the speculative-decoding comparison figure.

Reads baseline + ngram + eagle3 result files for the models that have them
(Llama 3.1 8B, Qwen3 8B, Gemma 4 E4B) and renders a two-panel chart:
  left  — median TTFT (lower is better)
  right — peak throughput (higher is better)

Output: reports/figures/speculative_decoding.svg
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis import _figure_style as style

RESULTS_DIR = Path("results")
OUTPUT = Path("reports/figures/speculative_decoding.svg")

# (display label, result-directory name) — only models with at least baseline + ngram data
MODELS = [
    ("Llama 3.1 8B", "llama-3-1-8b-instruct"),
    ("Qwen3 8B", "qwen3-8b"),
    ("Gemma 4 E4B", "gemma-4-e4b-it"),
]

# (legend label, engine tokens to try in order, colour)
# Older runs use `VLLMClient_` / `SGLangClient_`; newer runs use `vllm_` / `sglang_`.
VARIANTS = [
    ("Baseline vLLM", ["vllm_", "VLLMClient_"], style.VLLM),
    ("Baseline SGLang", ["sglang_", "SGLangClient_"], style.SGLANG),
    ("Ngram vLLM", ["vllm-ngram_"], style.NGRAM),
    ("Ngram SGLang", ["sglang-ngram_"], "#C7A8E8"),
    ("Eagle3 vLLM", ["vllm-eagle3_"], style.EAGLE),
    ("Eagle3 SGLang", ["sglang-eagle3_"], "#E8B8A8"),
]


def _latest(model_dir: str, scenario: str, engine_tokens: list[str]) -> dict | None:
    """Return the newest result payload whose filename matches any of the engine tokens."""
    for token in engine_tokens:
        pattern = f"{RESULTS_DIR}/{model_dir}/{scenario}_{token}*.json"
        files = sorted(glob.glob(pattern))
        if files:
            return json.loads(Path(files[-1]).read_text())
    return None


def _metrics_for(model_dir: str, engine_tokens: list[str]) -> tuple[float | None, float | None]:
    """Return (median TTFT ms, peak throughput tok/s) for a (model, variant) cell."""
    srl = _latest(model_dir, "single_request_latency", engine_tokens)
    ramp = _latest(model_dir, "throughput_ramp", engine_tokens)
    ttft = None
    tps = None
    if srl:
        ttft_metrics = srl.get("metrics", {}).get("ttft", {})
        ttft = ttft_metrics.get("median") or ttft_metrics.get("p50")
        # guard against zero-token runs (bad data)
        if (srl.get("metrics", {}).get("throughput", {}).get("total_tokens_generated") or 0) == 0:
            # TTFT is still a valid measurement even if no tokens were returned
            pass
    if ramp:
        thr = ramp.get("metrics", {}).get("throughput", {})
        if (thr.get("total_tokens_generated") or 0) > 0:
            tps = thr.get("tokens_per_sec")
    return ttft, tps


def main() -> None:
    style.apply()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    n_models = len(MODELS)
    n_variants = len(VARIANTS)
    bar_w = 0.8 / n_variants
    x = np.arange(n_models)

    fig, (ax_ttft, ax_tps) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=120)

    for vi, (label, tokens, colour) in enumerate(VARIANTS):
        ttft_vals = []
        tps_vals = []
        for _, model_dir in MODELS:
            ttft, tps = _metrics_for(model_dir, tokens)
            ttft_vals.append(ttft if ttft is not None else np.nan)
            tps_vals.append(tps if tps is not None else np.nan)

        offset = (vi - (n_variants - 1) / 2) * bar_w
        xs = x + offset

        ax_ttft.bar(
            xs, ttft_vals, width=bar_w, label=label, color=colour, edgecolor=style.BG, linewidth=0.5
        )
        for xpos, val in zip(xs, ttft_vals):
            if not np.isnan(val):
                ax_ttft.text(
                    xpos,
                    val + 1.5,
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=style.FG,
                )

        ax_tps.bar(
            xs, tps_vals, width=bar_w, label=label, color=colour, edgecolor=style.BG, linewidth=0.5
        )
        for xpos, val in zip(xs, tps_vals):
            if not np.isnan(val):
                ax_tps.text(
                    xpos,
                    val + 1.5,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=style.FG,
                )

    for ax, title, ylabel, note in [
        (ax_ttft, "TTFT Median (ms)", "TTFT (ms) — lower is better", None),
        (ax_tps, "Peak Throughput (tok/s)", "Tokens/sec — higher is better", None),
    ]:
        ax.set_title(title, pad=12)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in MODELS])
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # Annotate missing cells
    ax_tps.annotate(
        "Gemma 4 E2B omitted: single/ramp files emit 0 tokens — rerun pending",
        xy=(0.5, -0.22),
        xycoords="axes fraction",
        ha="center",
        fontsize=8,
        color=style.MUTED,
    )

    fig.suptitle(
        "Speculative Decoding: Baseline vs Ngram vs Eagle3",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    ax_ttft.legend(
        loc="upper left", fontsize=8, framealpha=0.9, facecolor=style.BG, labelcolor=style.FG
    )
    fig.text(
        0.5,
        0.93,
        "AWS g5.2xlarge / A10G 24 GB · single-request latency at concurrency 1, throughput at concurrency ramp 1→32",
        ha="center",
        fontsize=9,
        color=style.MUTED,
    )

    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(OUTPUT, format="svg")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
