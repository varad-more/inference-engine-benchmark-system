"""Tests for engines/base_client.py and VLLMClient with httpx mocking."""

from __future__ import annotations

import json
import time

import httpx
import pytest
import respx

from engines.base_client import EngineMetrics, GenerationResult

# ---------------------------------------------------------------------------
# GenerationResult
# ---------------------------------------------------------------------------

class TestGenerationResult:
    def test_fields(self) -> None:
        r = GenerationResult(
            text="hello",
            ttft_ms=10.5,
            total_ms=200.0,
            prompt_tokens=8,
            output_tokens=1,
            tokens_per_sec=5.0,
        )
        assert r.text == "hello"
        assert r.ttft_ms == 10.5
        assert r.total_ms == 200.0
        assert r.prompt_tokens == 8
        assert r.output_tokens == 1
        assert r.tokens_per_sec == 5.0

    def test_from_timing(self) -> None:
        t0 = time.monotonic()
        r = GenerationResult.from_timing(
            text="test output",
            start_time=t0,
            first_token_time=t0 + 0.05,  # 50ms TTFT
            end_time=t0 + 0.5,           # 500ms total
            prompt_tokens=12,
            output_tokens=25,
        )
        assert r.text == "test output"
        assert abs(r.ttft_ms - 50.0) < 5.0
        assert abs(r.total_ms - 500.0) < 5.0
        assert r.prompt_tokens == 12
        assert r.output_tokens == 25
        assert abs(r.tokens_per_sec - 50.0) < 5.0  # 25 tokens / 0.5s

    def test_from_timing_ttft_equals_end(self) -> None:
        """When first_token_time == end_time (instant), TTFT == total."""
        t0 = time.monotonic()
        r = GenerationResult.from_timing(
            text="x",
            start_time=t0,
            first_token_time=t0 + 0.1,
            end_time=t0 + 0.1,
            prompt_tokens=5,
            output_tokens=1,
        )
        assert abs(r.ttft_ms - r.total_ms) < 1.0

    def test_tokens_per_sec_nonzero(self) -> None:
        t0 = 0.0
        r = GenerationResult.from_timing(
            text="a b c",
            start_time=t0,
            first_token_time=t0 + 0.01,
            end_time=t0 + 1.0,
            prompt_tokens=3,
            output_tokens=100,
        )
        assert r.tokens_per_sec == pytest.approx(100.0, abs=1.0)


# ---------------------------------------------------------------------------
# EngineMetrics
# ---------------------------------------------------------------------------

class TestEngineMetrics:
    def test_defaults(self) -> None:
        m = EngineMetrics()
        assert m.gpu_memory_used_gb == 0.0
        assert m.kv_cache_usage_pct == 0.0
        assert m.pending_requests == 0
        assert m.running_requests == 0
        assert m.timestamp > 0

    def test_custom_values(self) -> None:
        m = EngineMetrics(
            gpu_memory_used_gb=18.5,
            kv_cache_usage_pct=0.72,
            pending_requests=5,
            running_requests=12,
        )
        assert m.gpu_memory_used_gb == 18.5
        assert m.kv_cache_usage_pct == 0.72
        assert m.pending_requests == 5
        assert m.running_requests == 12


# ---------------------------------------------------------------------------
# VLLMClient with respx mocking
# ---------------------------------------------------------------------------

@pytest.fixture()
def vllm_base_url() -> str:
    return "http://localhost:8000"


def _sse_body(tokens: list[str], model: str = "test-model") -> bytes:
    """Build a fake SSE response body matching vLLM's format."""
    lines: list[str] = []
    for i, tok in enumerate(tokens):
        chunk = {
            "id": f"cmpl-{i}",
            "object": "text_completion",
            "choices": [{"text": tok, "finish_reason": None, "index": 0}],
        }
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


@respx.mock
@pytest.mark.asyncio
async def test_vllm_health_check_ok(vllm_base_url: str) -> None:
    from engines.vllm_client import VLLMClient

    respx.get(f"{vllm_base_url}/health").mock(return_value=httpx.Response(200))
    client = VLLMClient(host="localhost", port=8000)
    assert await client.health_check() is True
    await client.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_vllm_health_check_fail(vllm_base_url: str) -> None:
    from engines.vllm_client import VLLMClient

    respx.get(f"{vllm_base_url}/health").mock(side_effect=httpx.ConnectError("refused"))
    client = VLLMClient(host="localhost", port=8000)
    assert await client.health_check() is False
    await client.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_vllm_generate_stream(vllm_base_url: str) -> None:
    from engines.vllm_client import VLLMClient

    tokens = [" The", " sky", " is", " blue", "."]
    body = _sse_body(tokens)

    respx.post(f"{vllm_base_url}/v1/completions").mock(
        return_value=httpx.Response(
            200,
            content=body,
            headers={"Content-Type": "text/event-stream"},
        )
    )

    client = VLLMClient(host="localhost", port=8000)
    collected: list[str] = []
    async for tok in client.generate_stream("What colour is the sky?", max_tokens=20):
        collected.append(tok)
    await client.aclose()

    assert collected == tokens


@respx.mock
@pytest.mark.asyncio
async def test_vllm_generate_returns_result(vllm_base_url: str) -> None:
    from engines.vllm_client import VLLMClient

    tokens = ["Paris", " is", " the", " capital", "."]
    body = _sse_body(tokens)

    respx.post(f"{vllm_base_url}/v1/completions").mock(
        return_value=httpx.Response(
            200,
            content=body,
            headers={"Content-Type": "text/event-stream"},
        )
    )
    # Mock tokenize endpoint
    respx.post(f"{vllm_base_url}/tokenize").mock(
        return_value=httpx.Response(200, json={"count": 7})
    )

    client = VLLMClient(host="localhost", port=8000)
    result = await client.generate("What is the capital of France?", max_tokens=20)
    await client.aclose()

    assert result.text == "Paris is the capital."
    assert result.output_tokens == 5
    assert result.ttft_ms > 0
    assert result.total_ms >= result.ttft_ms


@respx.mock
@pytest.mark.asyncio
async def test_vllm_get_metrics_parses_prometheus(vllm_base_url: str) -> None:
    from engines.vllm_client import VLLMClient

    prometheus_body = """
# HELP vllm:gpu_cache_usage_perc GPU KV-Cache usage
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="test"} 42.5
# HELP vllm:num_running_seqs Running sequences
vllm:num_running_seqs{model_name="test"} 8
# HELP vllm:num_waiting_seqs Waiting sequences
vllm:num_waiting_seqs{model_name="test"} 3
vllm:gpu_memory_usage_perc{model_name="test"} 70.0
"""
    respx.get(f"{vllm_base_url}/metrics").mock(
        return_value=httpx.Response(200, text=prometheus_body)
    )

    client = VLLMClient(host="localhost", port=8000)
    metrics = await client.get_metrics()
    await client.aclose()

    assert abs(metrics.kv_cache_usage_pct - 0.425) < 0.01
    assert metrics.running_requests == 8
    assert metrics.pending_requests == 3
    assert abs(metrics.gpu_memory_used_gb - 70.0 * 24.0 / 100.0) < 0.1


@respx.mock
@pytest.mark.asyncio
async def test_vllm_get_metrics_empty_response(vllm_base_url: str) -> None:
    from engines.vllm_client import VLLMClient

    respx.get(f"{vllm_base_url}/metrics").mock(
        return_value=httpx.Response(500)
    )

    client = VLLMClient(host="localhost", port=8000)
    metrics = await client.get_metrics()
    await client.aclose()

    # Should return zeroed metrics, not raise
    assert metrics.kv_cache_usage_pct == 0.0
    assert metrics.running_requests == 0


# ---------------------------------------------------------------------------
# SGLangClient with respx mocking
# ---------------------------------------------------------------------------

@respx.mock
@pytest.mark.asyncio
async def test_sglang_health_check() -> None:
    from engines.sglang_client import SGLangClient

    respx.get("http://localhost:8001/health").mock(return_value=httpx.Response(200))
    client = SGLangClient(host="localhost", port=8001)
    assert await client.health_check() is True
    await client.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_sglang_get_metrics() -> None:
    from engines.sglang_client import SGLangClient

    server_info = {
        "gpu_memory_used_gb": 12.3,
        "kv_cache_usage": 0.55,
        "waiting_queue_size": 2,
        "num_running_reqs": 6,
    }
    respx.get("http://localhost:8001/get_server_info").mock(
        return_value=httpx.Response(200, json=server_info)
    )

    client = SGLangClient(host="localhost", port=8001)
    metrics = await client.get_metrics()
    await client.aclose()

    assert abs(metrics.gpu_memory_used_gb - 12.3) < 0.01
    assert abs(metrics.kv_cache_usage_pct - 0.55) < 0.01
    assert metrics.pending_requests == 2
    assert metrics.running_requests == 6


@respx.mock
@pytest.mark.asyncio
async def test_sglang_generate_stream() -> None:
    from engines.sglang_client import SGLangClient

    tokens = ["Hello", " world", "!"]
    body = _sse_body(tokens)

    respx.post("http://localhost:8001/v1/completions").mock(
        return_value=httpx.Response(
            200,
            content=body,
            headers={"Content-Type": "text/event-stream"},
        )
    )

    client = SGLangClient(host="localhost", port=8001)
    collected: list[str] = []
    async for tok in client.generate_stream("Say hello", max_tokens=10):
        collected.append(tok)
    await client.aclose()

    assert collected == tokens


# ---------------------------------------------------------------------------
# Prometheus parser unit test (private function)
# ---------------------------------------------------------------------------

def test_prometheus_parser() -> None:
    from engines.vllm_client import _parse_prometheus

    body = """
vllm:gpu_cache_usage_perc{model_name="m"} 42.5
vllm:num_running_seqs{model_name="m"} 8
vllm:num_waiting_seqs{model_name="m"} 3
some_other_metric 99.9
"""
    assert _parse_prometheus(body, "vllm:gpu_cache_usage_perc") == pytest.approx(42.5)
    assert _parse_prometheus(body, "vllm:num_running_seqs") == pytest.approx(8.0)
    assert _parse_prometheus(body, "vllm:num_waiting_seqs") == pytest.approx(3.0)
    assert _parse_prometheus(body, "nonexistent") is None


def test_prometheus_parser_undecorated() -> None:
    from engines.vllm_client import _parse_prometheus

    body = "vllm:gpu_cache_usage_perc 55.0\n"
    assert _parse_prometheus(body, "vllm:gpu_cache_usage_perc") == pytest.approx(55.0)
