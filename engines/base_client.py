"""Abstract base class for inference engine clients."""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

import httpx
import structlog

logger = structlog.get_logger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-8B"

T = TypeVar("T")


async def retry_async(
    fn: Callable[..., Awaitable[T]],
    *args: object,
    retries: int = 2,
    backoff: float = 0.5,
    logger_ctx: structlog.stdlib.BoundLogger | None = None,
) -> T:
    """Retry an async callable with exponential backoff. Not for benchmark-critical paths."""
    last_exc: Exception | None = None
    for attempt in range(1 + retries):
        try:
            return await fn(*args)
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                wait = backoff * (2**attempt)
                if logger_ctx:
                    logger_ctx.debug("retrying", attempt=attempt + 1, wait=wait, error=str(exc))
                await asyncio.sleep(wait)
    raise last_exc  # type: ignore[misc]


@dataclass
class GenerationResult:
    """Result from a single generation request."""

    text: str
    ttft_ms: float  # Time to first token in milliseconds
    total_ms: float  # Total generation time in milliseconds
    prompt_tokens: int
    output_tokens: int
    tokens_per_sec: float

    @classmethod
    def from_timing(
        cls,
        text: str,
        start_time: float,
        first_token_time: float,
        end_time: float,
        prompt_tokens: int,
        output_tokens: int,
    ) -> GenerationResult:
        ttft_ms = (first_token_time - start_time) * 1000
        total_ms = (end_time - start_time) * 1000
        tokens_per_sec = output_tokens / max((end_time - start_time), 1e-9)
        return cls(
            text=text,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            tokens_per_sec=tokens_per_sec,
        )


@dataclass
class EngineMetrics:
    """Real-time metrics from an inference engine."""

    gpu_memory_used_gb: float = 0.0
    kv_cache_usage_pct: float = 0.0
    pending_requests: int = 0
    running_requests: int = 0
    timestamp: float = field(default_factory=time.time)


class BaseInferenceClient(ABC):
    """Abstract base for vLLM and SGLang inference clients."""

    def __init__(self, host: str, port: int, model: str, timeout: float = 120.0) -> None:
        self.host = host
        self.port = port
        self.model = model
        self.base_url = f"http://{host}:{port}"
        self._timeout = timeout
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
        )
        self._log = structlog.get_logger(self.__class__.__name__)

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Collect a full streamed generation, measuring TTFT and total time."""
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
        prompt_tokens = await self._count_prompt_tokens(prompt)

        return GenerationResult.from_timing(
            text=text,
            start_time=start,
            first_token_time=first_token_time,
            end_time=end,
            prompt_tokens=prompt_tokens,
            output_tokens=len(tokens),
        )

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Stream tokens via SSE from the OpenAI compatible /v1/completions endpoint."""
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
                data_str = line[len("data:") :].strip()
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
        """Return True if the engine is healthy and ready to serve."""
        try:

            async def _check() -> bool:
                r = await self._http.get("/health")
                return r.status_code == 200

            return await retry_async(_check, retries=2, backoff=1.0, logger_ctx=self._log)
        except Exception as exc:
            self._log.warning("health check failed", error=str(exc))
            return False

    @abstractmethod
    async def get_metrics(self) -> EngineMetrics:
        """Fetch current engine metrics (GPU memory, KV cache, queue depths)."""
        ...

    async def _count_prompt_tokens(self, prompt: str) -> int:
        """Use the /tokenize endpoint if available, else rough word count estimate."""
        try:
            r = await self._http.post(
                "/tokenize",
                json={"model": self.model, "prompt": prompt},
            )
            if r.status_code == 200:
                return r.json().get("count", len(prompt.split()))
        except Exception:
            pass
        return len(prompt.split())

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()

    async def __aenter__(self) -> BaseInferenceClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(host={self.host}, port={self.port}, model={self.model})"


if __name__ == "__main__":
    import asyncio
    import dataclasses

    # Smoke test: verify dataclasses are instantiable
    result = GenerationResult(
        text="hello world",
        ttft_ms=12.3,
        total_ms=145.7,
        prompt_tokens=8,
        output_tokens=2,
        tokens_per_sec=13.7,
    )
    metrics = EngineMetrics(
        gpu_memory_used_gb=14.2,
        kv_cache_usage_pct=0.41,
        pending_requests=3,
        running_requests=8,
    )
    print("GenerationResult:", dataclasses.asdict(result))
    print("EngineMetrics:", dataclasses.asdict(metrics))

    # Verify from_timing factory
    t0 = time.monotonic()
    r2 = GenerationResult.from_timing(
        text="test",
        start_time=t0,
        first_token_time=t0 + 0.05,
        end_time=t0 + 0.5,
        prompt_tokens=10,
        output_tokens=20,
    )
    assert abs(r2.ttft_ms - 50) < 10, f"Expected ~50ms TTFT, got {r2.ttft_ms:.1f}ms"
    print("from_timing OK, ttft_ms ≈", round(r2.ttft_ms, 1))
    print("Smoke test passed.")
