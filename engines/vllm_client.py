"""vLLM inference client using OpenAI-compatible API with SSE streaming."""

from __future__ import annotations

import re

import structlog

from engines.base_client import (
    DEFAULT_MODEL,
    BaseInferenceClient,
    EngineMetrics,
    GenerationResult,
)

logger = structlog.get_logger(__name__)

# Prometheus metric names emitted by vLLM
_METRIC_GPU_CACHE = "vllm:gpu_cache_usage_perc"
_METRIC_RUNNING = "vllm:num_running_seqs"
_METRIC_WAITING = "vllm:num_waiting_seqs"
_METRIC_GPU_MEM = "vllm:gpu_memory_usage_perc"


def _parse_prometheus(text: str, metric_name: str) -> float | None:
    """Extract the first numeric value for a given Prometheus metric name."""
    pattern = rf"^{re.escape(metric_name)}\{{[^}}]*\}}\s+([\d.eE+\-]+)"
    for line in text.splitlines():
        m = re.match(pattern, line)
        if m:
            return float(m.group(1))
    # Fallback: undecorated metric
    pattern2 = rf"^{re.escape(metric_name)}\s+([\d.eE+\-]+)"
    for line in text.splitlines():
        m = re.match(pattern2, line)
        if m:
            return float(m.group(1))
    return None


class VLLMClient(BaseInferenceClient):
    """Client for a running vLLM server (OpenAI-compatible REST API)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        gpu_memory_gb: float = 24.0,
    ) -> None:
        super().__init__(host, port, model, timeout)
        self._gpu_memory_gb = gpu_memory_gb

    # ------------------------------------------------------------------
    # Metrics (vLLM-specific: Prometheus /metrics endpoint)
    # ------------------------------------------------------------------

    async def get_metrics(self) -> EngineMetrics:
        try:
            r = await self._http.get("/metrics")
            r.raise_for_status()
            body = r.text
        except Exception as exc:
            self._log.warning("failed to fetch vllm /metrics", error=str(exc))
            return EngineMetrics()

        kv_pct = _parse_prometheus(body, _METRIC_GPU_CACHE) or 0.0
        running = int(_parse_prometheus(body, _METRIC_RUNNING) or 0)
        waiting = int(_parse_prometheus(body, _METRIC_WAITING) or 0)
        gpu_mem_pct = _parse_prometheus(body, _METRIC_GPU_MEM) or 0.0

        gpu_gb = gpu_mem_pct * self._gpu_memory_gb / 100.0

        return EngineMetrics(
            gpu_memory_used_gb=gpu_gb,
            kv_cache_usage_pct=kv_pct / 100.0,  # normalise 0-1
            pending_requests=waiting,
            running_requests=running,
        )

    # ------------------------------------------------------------------
    # Prefix caching benchmark helper
    # ------------------------------------------------------------------

    async def test_prefix_caching(
        self,
        shared_prefix: str,
        suffixes: list[str],
        max_tokens: int = 50,
    ) -> dict[str, object]:
        """
        Send N requests that share the same prefix and measure latency drop
        as vLLM's prefix cache warms up.

        Returns dict with:
          - cold_ttft_ms: TTFT of first (cache-miss) request
          - warm_ttft_ms: mean TTFT of subsequent (cache-hit) requests
          - speedup: cold / warm ratio
          - hit_rate_estimate: fraction of requests that benefited
        """
        results: list[GenerationResult] = []
        for i, suffix in enumerate(suffixes):
            prompt = shared_prefix + suffix
            self._log.debug("prefix_cache_test", request=i, prompt_len=len(prompt))
            res = await self.generate(prompt, max_tokens=max_tokens, temperature=0.0)
            results.append(res)

        if not results:
            return {}

        cold_ttft = results[0].ttft_ms
        warm_ttfts = [r.ttft_ms for r in results[1:]] if len(results) > 1 else [cold_ttft]
        mean_warm = sum(warm_ttfts) / len(warm_ttfts)
        speedup = cold_ttft / max(mean_warm, 1e-3)

        # Heuristic: if warm is < 60% of cold, consider it a cache hit
        hit_count = sum(1 for t in warm_ttfts if t < cold_ttft * 0.6)
        hit_rate = hit_count / max(len(warm_ttfts), 1)

        return {
            "cold_ttft_ms": cold_ttft,
            "warm_ttft_ms": mean_warm,
            "speedup": speedup,
            "hit_rate_estimate": hit_rate,
            "n_requests": len(results),
        }


if __name__ == "__main__":
    import asyncio

    async def smoke() -> None:
        client = VLLMClient(host="localhost", port=8000)
        healthy = await client.health_check()
        print(f"vLLM healthy: {healthy}")
        if healthy:
            result = await client.generate("The capital of France is", max_tokens=10)
            print(f"Result: {result}")
            metrics = await client.get_metrics()
            print(f"Metrics: {metrics}")
        await client.aclose()

    asyncio.run(smoke())
