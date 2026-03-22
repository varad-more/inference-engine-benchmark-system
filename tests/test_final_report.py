"""Tests for analysis/final_report.py."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.final_report import aggregate_results, generate_final_report, render_markdown


def _write_result(path: Path, *, model: str, scenario: str, engine: str, tps: float) -> None:
    payload = {
        "scenario_name": scenario,
        "engine_name": engine,
        "metrics": {
            "ttft": {"p50": 20.0, "p95": 50.0},
            "latency": {"p95": 800.0},
            "throughput": {
                "tokens_per_sec": tps,
                "requests_per_sec": tps / 100.0,
            },
            "error_rate": 0.0,
        },
        "run_metadata": {"model": model},
        "workload_metadata": {"prompt_pack": "short_chat"},
    }
    path.write_text(json.dumps(payload))


def test_aggregate_results(tmp_path: Path) -> None:
    _write_result(
        tmp_path / "single_request_latency_VLLMClient_1.json",
        model="Qwen/Qwen2.5-7B-Instruct",
        scenario="single_request_latency",
        engine="VLLMClient",
        tps=5000.0,
    )
    _write_result(
        tmp_path / "single_request_latency_VLLMClient_2.json",
        model="Qwen/Qwen2.5-7B-Instruct",
        scenario="single_request_latency",
        engine="VLLMClient",
        tps=5100.0,
    )

    summary = aggregate_results(tmp_path)
    assert summary["total_result_files"] == 2
    assert len(summary["rows"]) == 1
    row = summary["rows"][0]
    assert row["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert row["runs"] == 2
    assert row["tokens_per_sec"] == 5050.0


def test_generate_final_report(tmp_path: Path) -> None:
    _write_result(
        tmp_path / "throughput_ramp_SGLangClient_1.json",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        scenario="throughput_ramp",
        engine="SGLangClient",
        tps=42000.0,
    )
    output = tmp_path / "final.md"
    summary = generate_final_report(tmp_path, output)
    assert summary["total_result_files"] == 1
    content = output.read_text()
    assert "Final Benchmark Summary" in content
    assert "mistralai/Mistral-7B-Instruct-v0.3" in content


def test_render_markdown_empty() -> None:
    md = render_markdown({"results_dir": "x", "total_result_files": 0, "rows": []})
    assert "No result files found." in md
