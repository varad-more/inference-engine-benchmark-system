"""Tests for ScenarioResults result filename and metadata behaviour."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.metrics import LatencyStats, ScenarioMetrics, ThroughputStats
from benchmarks.runner import ScenarioResults


def _make_results(engine_name: str = "vllm", run_metadata: dict | None = None) -> ScenarioResults:
    lat = LatencyStats(
        mean=10.0,
        median=10.0,
        p50=10.0,
        p90=14.0,
        p95=15.0,
        p99=20.0,
        stdev=1.0,
        min=8.0,
        max=25.0,
        count=1,
    )
    tput = ThroughputStats(
        total_requests=1,
        successful_requests=1,
        failed_requests=0,
        total_tokens_generated=100,
        wall_time_sec=1.0,
        requests_per_sec=1.0,
        tokens_per_sec=100.0,
        mean_tokens_per_request=100.0,
    )
    metrics = ScenarioMetrics(
        scenario_name="single_request_latency",
        engine_name=engine_name,
        ttft=lat,
        latency=lat,
        throughput=tput,
    )
    return ScenarioResults(
        scenario_name="single_request_latency",
        engine_name=engine_name,
        run_id="test-run-id",
        timestamp=1711700000.0,
        requests=[],
        metrics=metrics,
        run_metadata=run_metadata or {},
    )


def test_save_with_engine_variant_uses_variant_in_filename(tmp_path: Path) -> None:
    results = _make_results(
        engine_name="vllm",
        run_metadata={
            "engine": "vllm",
            "engine_variant": "vllm-eagle3",
            "spec_method": "eagle3",
        },
    )
    saved = results.save(tmp_path)

    assert saved.name == "single_request_latency_vllm-eagle3_1711700000.json"
    assert saved.exists()


def test_save_without_variant_falls_back_to_engine_name(tmp_path: Path) -> None:
    results = _make_results(engine_name="vllm", run_metadata={"model": "Qwen/Qwen3-8B"})
    saved = results.save(tmp_path)

    assert saved.name == "single_request_latency_vllm_1711700000.json"
    assert saved.exists()


def test_saved_json_contains_run_metadata(tmp_path: Path) -> None:
    run_meta = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "engine": "vllm",
        "engine_variant": "vllm-eagle3",
        "spec_method": "eagle3",
    }
    results = _make_results(engine_name="vllm", run_metadata=run_meta)
    saved = results.save(tmp_path)

    payload = json.loads(saved.read_text())
    assert payload["run_metadata"] == run_meta
    assert payload["engine_name"] == "vllm"


def test_save_ngram_variant_filename(tmp_path: Path) -> None:
    results = _make_results(
        engine_name="sglang",
        run_metadata={"engine_variant": "sglang-ngram", "spec_method": "ngram"},
    )
    saved = results.save(tmp_path)

    assert saved.name == "single_request_latency_sglang-ngram_1711700000.json"


def test_save_baseline_variant_no_spec_method(tmp_path: Path) -> None:
    results = _make_results(
        engine_name="sglang",
        run_metadata={"engine_variant": "sglang", "spec_method": None},
    )
    saved = results.save(tmp_path)

    assert saved.name == "single_request_latency_sglang_1711700000.json"
    payload = json.loads(saved.read_text())
    assert payload["run_metadata"]["spec_method"] is None
