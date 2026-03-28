"""Regression tests for wall-time normalization in BenchmarkRunner metrics."""

from __future__ import annotations

import pytest

from benchmarks.runner import BenchmarkRunner, RequestResult


def _successful_request(i: int, total_ms: float, output_tokens: int = 20) -> RequestResult:
    return RequestResult(
        request_id=i,
        prompt_len=10,
        success=True,
        ttft_ms=50.0,
        total_ms=total_ms,
        output_tokens=output_tokens,
        tokens_per_sec=output_tokens / (total_ms / 1000.0),
    )


def test_compute_metrics_uses_timeline_wall_time() -> None:
    runner = BenchmarkRunner()
    requests = [_successful_request(i, total_ms=1000.0) for i in range(5)]
    # Full run lasted 10s even though each request took ~1s.
    timeline = [
        {
            "timestamp": 100.0,
            "gpu_memory_used_gb": 0.0,
            "kv_cache_usage_pct": 0.0,
            "pending_requests": 0,
            "running_requests": 0,
        },
        {
            "timestamp": 110.0,
            "gpu_memory_used_gb": 0.0,
            "kv_cache_usage_pct": 0.0,
            "pending_requests": 0,
            "running_requests": 0,
        },
    ]

    metrics = runner._compute_metrics("throughput_ramp", "VLLMClient", requests, timeline)

    assert metrics.throughput.wall_time_sec == pytest.approx(10.0, rel=1e-6)
    assert metrics.throughput.requests_per_sec == pytest.approx(0.5, rel=1e-6)
    assert metrics.throughput.tokens_per_sec == pytest.approx(10.0, rel=1e-6)


def test_compute_metrics_single_request_fallback_sums_latency() -> None:
    runner = BenchmarkRunner()
    requests = [_successful_request(i, total_ms=1000.0) for i in range(5)]

    metrics = runner._compute_metrics("single_request_latency", "SGLangClient", requests, [])

    # 5 sequential 1s requests => ~5s wall time.
    assert metrics.throughput.wall_time_sec == pytest.approx(5.0, rel=1e-6)
    assert metrics.throughput.requests_per_sec == pytest.approx(1.0, rel=1e-6)
    assert metrics.throughput.tokens_per_sec == pytest.approx(20.0, rel=1e-6)
