"""vLLM inference client using OpenAI-compatible API with SSE streaming."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator

import httpx
import structlog

from engines.base_client import BaseInferenceClient, EngineMetrics, GenerationResult, retry_async

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
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        timeout: float = 120.0,
        gpu_memory_gb: float = 24.0,
    ) -> None:
        super().__init__(host, port, model)
        self._timeout = timeout
        self._gpu_memory_gb = gpu_memory_gb
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Non-streaming generation; measures TTFT via the first SSE chunk."""
        tokens: list[str] = []
        first_token_time: float | None = None
        start = time.monotonic()

        async for token in self.generate_stream(prompt, max_tokens, temperature):
            if first_token_time is None:
                first_token_time = time.monotonic()
            tokens.append(token)

        end = time.monotonic()
        first_token_time = first_token_time or end
        text = "".join(tokens)
        output_tokens = len(tokens)  # approximate: one chunk ≈ one token from vLLM stream

        # Fetch actual token counts from a non-streaming call if needed.
        # For benchmark accuracy we use a quick non-stream call to get usage.
        prompt_tokens = await self._count_prompt_tokens(prompt)

        return GenerationResult.from_timing(
            text=text,
            start_time=start,
            first_token_time=first_token_time,
            end_time=end,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Stream tokens via SSE; yields each decoded token text fragment."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        async with self._http.stream(
            "POST",
            "/v1/completions",
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    text = chunk["choices"][0].get("text", "")
                    if text:
                        yield text
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    async def health_check(self) -> bool:
        try:
            async def _check() -> bool:
                r = await self._http.get("/health")
                return r.status_code == 200
            return await retry_async(_check, retries=2, backoff=1.0, logger_ctx=self._log)
        except Exception as exc:
            self._log.warning("vllm health check failed", error=str(exc))
            return False

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _count_prompt_tokens(self, prompt: str) -> int:
        """Use vLLM tokenize endpoint if available, else rough estimate."""
        try:
            r = await self._http.post(
                "/tokenize",
                json={"model": self.model, "prompt": prompt},
            )
            if r.status_code == 200:
                return r.json().get("count", len(prompt.split()))
        except Exception:
            pass
        return len(prompt.split())  # rough fallback

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> VLLMClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


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
