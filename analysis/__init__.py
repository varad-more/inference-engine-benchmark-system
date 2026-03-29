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
    def safe_mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except PermissionError:
            return 0.0

    files = sorted(results_dir.glob("*.json"), key=safe_mtime)
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


def get_result_model(result: dict[str, Any]) -> str | None:
    """Return the canonical model name for a result payload when available."""
    model = result.get("run_metadata", {}).get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return None


def extract_model_name(results: list[dict[str, Any]]) -> str:
    """Return the model name from the first result that has one, or a fallback."""
    for r in results:
        model = get_result_model(r)
        if model:
            return model
    return "unknown"


def select_model_results(
    results: list[dict[str, Any]],
    preferred_model: str | None = None,
    *,
    require_engines: set[str] | None = None,
) -> tuple[str | None, list[dict[str, Any]], dict[str, Any]]:
    """Choose a model-consistent slice of results.

    When multiple models are present, prefer the latest model that has all of the
    required engines represented. Falls back to the latest model overall.
    """
    available_models = sorted({model for r in results if (model := get_result_model(r))})
    metadata: dict[str, Any] = {
        "available_models": available_models,
        "selection_mode": "all-results",
    }

    if preferred_model is not None:
        selected = [r for r in results if get_result_model(r) == preferred_model]
        if not selected:
            raise ValueError(f"Model '{preferred_model}' not found in results")
        metadata["selection_mode"] = "explicit-model"
        return preferred_model, selected, metadata

    if len(available_models) <= 1:
        selected_model = available_models[0] if available_models else None
        selected = (
            [r for r in results if get_result_model(r) == selected_model]
            if selected_model is not None
            else list(results)
        )
        metadata["selection_mode"] = "single-model" if selected_model else "legacy-no-model"
        return selected_model, selected, metadata

    required = require_engines or set()
    chosen_model: str | None = None
    chosen_key: tuple[int, float] | None = None

    for model_name in available_models:
        model_results = [r for r in results if get_result_model(r) == model_name]
        engine_names = {str(r.get("engine_name", "")) for r in model_results}
        has_required = required.issubset(engine_names) if required else True
        latest_ts = max(float(r.get("timestamp", 0.0)) for r in model_results)
        candidate_key = (1 if has_required else 0, latest_ts)
        if chosen_key is None or candidate_key > chosen_key:
            chosen_model = model_name
            chosen_key = candidate_key

    selected = [r for r in results if get_result_model(r) == chosen_model]
    metadata["selection_mode"] = (
        "latest-complete-model"
        if chosen_key is not None and chosen_key[0] == 1 and required
        else "latest-model"
    )
    return chosen_model, selected, metadata
