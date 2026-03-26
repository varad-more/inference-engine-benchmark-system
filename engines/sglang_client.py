"""SGLang inference client — REST API + optional in-process sgl.Runtime."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from engines.base_client import (
    DEFAULT_MODEL,
    BaseInferenceClient,
    EngineMetrics,
)

if TYPE_CHECKING:
    pass  # sglang imported lazily to keep import optional

logger = structlog.get_logger(__name__)


@dataclass
class SGLangProgramResult:
    """Result from running a native SGLang sgl.function program."""

    variables: dict[str, str]  # named generation slots
    total_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


class SGLangClient(BaseInferenceClient):
    """Client for a running SGLang server (OpenAI-compatible + /get_server_info)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8001,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        runtime_model_path: str | None = None,
    ) -> None:
        super().__init__(host, port, model, timeout)
        self._runtime_model_path = runtime_model_path or model
        self._sgl_runtime: Any = None  # lazy initialised sglang.Runtime

    # ------------------------------------------------------------------
    # Metrics (SGLang-specific: /get_server_info endpoint)
    # ------------------------------------------------------------------

    async def get_metrics(self) -> EngineMetrics:
        """Fetch metrics from SGLang's /get_server_info endpoint."""
        try:
            r = await self._http.get("/get_server_info")
            r.raise_for_status()
            info: dict[str, Any] = r.json()
        except Exception as exc:
            self._log.warning("failed to fetch sglang /get_server_info", error=str(exc))
            return EngineMetrics()

        # SGLang server info keys (subject to version changes)
        gpu_mem = float(info.get("gpu_memory_used_gb", 0.0))
        kv_pct = float(info.get("kv_cache_usage", 0.0))
        if kv_pct > 1.0:
            kv_pct /= 100.0  # some versions return 0-100

        return EngineMetrics(
            gpu_memory_used_gb=gpu_mem,
            kv_cache_usage_pct=kv_pct,
            pending_requests=int(info.get("waiting_queue_size", 0)),
            running_requests=int(info.get("num_running_reqs", 0)),
        )

    # ------------------------------------------------------------------
    # Native SGLang program execution (in-process via sglang.Runtime)
    # ------------------------------------------------------------------

    async def _get_runtime(self) -> Any:
        """Lazily initialise a sglang.Runtime bound to the remote server."""
        if self._sgl_runtime is None:
            try:
                import sglang as sgl  # type: ignore[import]

                # Connect to the already-running server rather than launching a new one
                runtime = sgl.Runtime(
                    model_path=self._runtime_model_path,
                    port=self.port,
                )
                self._sgl_runtime = runtime
                sgl.set_default_backend(runtime)
            except ImportError as exc:
                raise RuntimeError(
                    "sglang package not installed. Install with: pip install sglang"
                ) from exc
        return self._sgl_runtime

    async def run_sgl_program(
        self,
        program_fn: Callable[..., Any],
        **kwargs: Any,
    ) -> SGLangProgramResult:
        """
        Execute a native @sgl.function program in-process and return results.

        The program_fn must be decorated with @sgl.function and accept (s, **kwargs).
        Execution is dispatched to a thread pool to avoid blocking the event loop.
        """
        await self._get_runtime()

        loop = asyncio.get_running_loop()
        start = time.monotonic()

        def _run() -> Any:
            return program_fn(**kwargs)

        state = await loop.run_in_executor(None, _run)
        end = time.monotonic()

        # Extract all generated text variables from the SGLang state object
        variables: dict[str, str] = {}
        try:
            for key in state.get_var_names():  # type: ignore[union-attr]
                variables[key] = state[key]
        except Exception:
            # Fallback: try common attribute patterns
            for attr in ("reasoning", "answer", "hypothesis_0", "output", "result"):
                try:
                    variables[attr] = str(state[attr])
                except Exception:
                    pass

        meta: dict[str, Any] = {}
        try:
            meta["usage"] = state.get_usage()  # type: ignore[union-attr]
        except Exception:
            pass

        return SGLangProgramResult(
            variables=variables,
            total_ms=(end - start) * 1000,
            meta=meta,
        )

    async def aclose(self) -> None:
        await super().aclose()
        if self._sgl_runtime is not None:
            try:
                self._sgl_runtime.shutdown()
            except Exception:
                self._log.debug("sglang runtime shutdown error", exc_info=True)


if __name__ == "__main__":
    import asyncio

    async def smoke() -> None:
        client = SGLangClient(host="localhost", port=8001)
        healthy = await client.health_check()
        print(f"SGLang healthy: {healthy}")
        if healthy:
            result = await client.generate("The capital of France is", max_tokens=10)
            print(f"Result: {result}")
            metrics = await client.get_metrics()
            print(f"Metrics: {metrics}")
        await client.aclose()

    asyncio.run(smoke())
