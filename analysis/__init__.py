"""Analysis utilities for benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

RESULTS_DIR = Path("results")


def load_results(results_dir: Path = RESULTS_DIR) -> list[dict[str, Any]]:
    """Load all benchmark result JSON files from a directory.

    Each returned dict has an extra ``_filename`` key with the source filename
    and is guaranteed to contain ``scenario_name``, ``engine_name``, and ``metrics``.
    """
    files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    data: list[dict[str, Any]] = []
    for f in files:
        try:
            d = json.loads(f.read_text())
        except Exception as exc:
            logger.warning("failed to load result", file=str(f), error=str(exc))
            continue
        if "metrics" not in d or "scenario_name" not in d or "engine_name" not in d:
            continue
        d["_filename"] = f.name
        d["_source_path"] = str(f)
        data.append(d)
    return data


def extract_model_name(results: list[dict[str, Any]]) -> str:
    """Return the model name from the first result that has one, or a fallback."""
    for r in results:
        model = r.get("run_metadata", {}).get("model")
        if model:
            return model
    return "unknown"
