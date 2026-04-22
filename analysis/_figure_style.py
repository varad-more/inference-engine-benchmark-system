"""Shared matplotlib style for benchmark figures.

Dark theme aligned with the hand-rolled SVGs in generate_final_benchmark_report.py.
"""

from __future__ import annotations

import matplotlib as mpl

BG = "#0b1020"
PANEL = "#0b1020"
GRID = "#27335a"
AXIS = "#5c6b91"
FG = "#e8eefc"
MUTED = "#9fb0d0"

VLLM = "#5B8DEF"
SGLANG = "#F5A524"
NGRAM = "#7BD389"
EAGLE = "#E05D78"


def apply() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": FG,
            "axes.titlecolor": FG,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": FG,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "savefig.facecolor": BG,
            "savefig.edgecolor": BG,
            "font.family": ["system-ui", "-apple-system", "DejaVu Sans"],
        }
    )
