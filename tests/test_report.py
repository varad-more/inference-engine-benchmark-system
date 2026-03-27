from __future__ import annotations

import json
from pathlib import Path

from analysis import select_model_results
from analysis.report import generate_report


def _write_result(
    path: Path,
    *,
    model: str,
    scenario: str,
    engine: str,
    timestamp: float,
    ttft_p95: float,
    tokens_per_sec: float,
) -> None:
    payload = {
        "scenario_name": scenario,
        "engine_name": engine,
        "timestamp": timestamp,
        "requests": [
            {"success": True, "ttft_ms": ttft_p95 / 2},
            {"success": True, "ttft_ms": ttft_p95},
        ],
        "metrics": {
            "ttft": {"p50": ttft_p95 / 2, "p95": ttft_p95, "p99": ttft_p95},
            "latency": {"p95": ttft_p95 * 10},
            "throughput": {
                "tokens_per_sec": tokens_per_sec,
                "requests_per_sec": tokens_per_sec / 100.0,
                "total_requests": 2,
            },
            "kv_cache_timeline": [0.1, 0.2],
            "error_rate": 0.0,
        },
        "run_metadata": {"model": model},
        "workload_metadata": {"prompt_pack": "short_chat"},
    }
    path.write_text(json.dumps(payload))


def test_select_model_results_prefers_latest_complete_model(tmp_path: Path) -> None:
    qwen = "Qwen/Qwen2.5-7B-Instruct"
    gemma = "google/gemma-2-2b-it"
    _write_result(
        tmp_path / "single_request_latency_VLLMClient_qwen.json",
        model=qwen,
        scenario="single_request_latency",
        engine="VLLMClient",
        timestamp=100,
        ttft_p95=10.0,
        tokens_per_sec=1000.0,
    )
    _write_result(
        tmp_path / "single_request_latency_SGLangClient_qwen.json",
        model=qwen,
        scenario="single_request_latency",
        engine="SGLangClient",
        timestamp=110,
        ttft_p95=20.0,
        tokens_per_sec=900.0,
    )
    _write_result(
        tmp_path / "single_request_latency_VLLMClient_gemma.json",
        model=gemma,
        scenario="single_request_latency",
        engine="VLLMClient",
        timestamp=200,
        ttft_p95=99.0,
        tokens_per_sec=500.0,
    )

    selected_model, selected_results, metadata = select_model_results(
        [json.loads(path.read_text()) for path in sorted(tmp_path.glob("*.json"))],
        require_engines={"VLLMClient", "SGLangClient"},
    )

    assert selected_model == qwen
    assert len(selected_results) == 2
    assert metadata["selection_mode"] == "latest-complete-model"


def test_generate_report_filters_to_latest_complete_model(tmp_path: Path) -> None:
    qwen = "Qwen/Qwen2.5-7B-Instruct"
    gemma = "google/gemma-2-2b-it"
    _write_result(
        tmp_path / "single_request_latency_VLLMClient_qwen.json",
        model=qwen,
        scenario="single_request_latency",
        engine="VLLMClient",
        timestamp=100,
        ttft_p95=10.0,
        tokens_per_sec=1000.0,
    )
    _write_result(
        tmp_path / "single_request_latency_SGLangClient_qwen.json",
        model=qwen,
        scenario="single_request_latency",
        engine="SGLangClient",
        timestamp=110,
        ttft_p95=20.0,
        tokens_per_sec=900.0,
    )
    _write_result(
        tmp_path / "single_request_latency_VLLMClient_gemma.json",
        model=gemma,
        scenario="single_request_latency",
        engine="VLLMClient",
        timestamp=200,
        ttft_p95=99.0,
        tokens_per_sec=500.0,
    )

    output = tmp_path / "report.html"
    generate_report(results_dir=tmp_path, output_path=output)
    html = output.read_text()

    assert qwen in html
    assert "Detected 2 models" in html
    assert gemma not in html


def test_generate_report_explicit_model_filter(tmp_path: Path) -> None:
    qwen = "Qwen/Qwen2.5-7B-Instruct"
    gemma = "google/gemma-2-2b-it"
    _write_result(
        tmp_path / "single_request_latency_VLLMClient_qwen.json",
        model=qwen,
        scenario="single_request_latency",
        engine="VLLMClient",
        timestamp=100,
        ttft_p95=10.0,
        tokens_per_sec=1000.0,
    )
    _write_result(
        tmp_path / "single_request_latency_SGLangClient_qwen.json",
        model=qwen,
        scenario="single_request_latency",
        engine="SGLangClient",
        timestamp=110,
        ttft_p95=20.0,
        tokens_per_sec=900.0,
    )
    _write_result(
        tmp_path / "single_request_latency_VLLMClient_gemma.json",
        model=gemma,
        scenario="single_request_latency",
        engine="VLLMClient",
        timestamp=200,
        ttft_p95=30.0,
        tokens_per_sec=800.0,
    )
    _write_result(
        tmp_path / "single_request_latency_SGLangClient_gemma.json",
        model=gemma,
        scenario="single_request_latency",
        engine="SGLangClient",
        timestamp=210,
        ttft_p95=25.0,
        tokens_per_sec=850.0,
    )

    output = tmp_path / "report_gemma.html"
    generate_report(results_dir=tmp_path, output_path=output, model=gemma)
    html = output.read_text()

    assert gemma in html
    assert "Report explicitly filtered" in html
    assert qwen not in html
