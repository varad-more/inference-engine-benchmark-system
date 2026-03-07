"""
Benchmark runner: executes scenarios against inference engine clients,
collects per-request metrics, polls engine health, and saves JSON results.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import structlog

from benchmarks.metrics import (
    ConcurrencyPoint,
    LatencyStats,
    ScenarioMetrics,
    ThroughputStats,
    compare_metrics,
)
from benchmarks.scenarios import (
    BenchmarkScenario,
    LongContextStress,
    PrefixSharingBenefit,
    ScenarioType,
    SingleRequestLatency,
    StructuredGenerationSpeed,
    ThroughputRamp,
    make_json_extraction_prompt,
    make_long_prompt,
    make_short_prompt,
    make_system_prompt,
)
from engines.base_client import BaseInferenceClient, EngineMetrics, GenerationResult

logger = structlog.get_logger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Per-request result
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    request_id: int
    prompt_len: int
    success: bool
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    output_tokens: int = 0
    tokens_per_sec: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Scenario result container
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResults:
    scenario_name: str
    engine_name: str
    run_id: str
    timestamp: float
    requests: list[RequestResult]
    metrics: ScenarioMetrics
    engine_metrics_timeline: list[dict[str, Any]] = field(default_factory=list)
    scenario_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "engine_name": self.engine_name,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "requests": [asdict(r) for r in self.requests],
            "metrics": self.metrics.to_dict(),
            "engine_metrics_timeline": self.engine_metrics_timeline,
            "scenario_config": self.scenario_config,
        }

    def save(self, results_dir: Path = RESULTS_DIR) -> Path:
        ts = int(self.timestamp)
        fname = f"{self.scenario_name}_{self.engine_name}_{ts}.json"
        path = results_dir / fname
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("saved results", path=str(path))
        return path


# ---------------------------------------------------------------------------
# Comparison result
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    scenario_name: str
    vllm_results: ScenarioResults
    sglang_results: ScenarioResults
    delta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "vllm": self.vllm_results.to_dict(),
            "sglang": self.sglang_results.to_dict(),
            "delta": self.delta,
        }

    def save(self, results_dir: Path = RESULTS_DIR) -> Path:
        ts = int(time.time())
        fname = f"comparison_{self.scenario_name}_{ts}.json"
        path = results_dir / fname
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("saved comparison", path=str(path))
        return path


# ---------------------------------------------------------------------------
# Progress callback type
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[int, int, RequestResult], None]


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Runs benchmark scenarios against an inference client.

    Usage:
        runner = BenchmarkRunner()
        results = await runner.run_scenario(scenario, client)
        results.save()
    """

    def __init__(
        self,
        results_dir: Path = RESULTS_DIR,
        metrics_poll_interval: float = 2.0,
    ) -> None:
        self.results_dir = results_dir
        self.metrics_poll_interval = metrics_poll_interval
        self.results_dir.mkdir(exist_ok=True)
        self._log = structlog.get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run_scenario(
        self,
        scenario: BenchmarkScenario,
        client: BaseInferenceClient,
        progress_cb: ProgressCallback | None = None,
    ) -> ScenarioResults:
        """Dispatch to the appropriate scenario handler."""
        self._log.info(
            "starting scenario",
            scenario=scenario.name,
            engine=client.__class__.__name__,
        )
        run_id = str(uuid.uuid4())[:8]
        ts = time.time()

        handler = {
            ScenarioType.SINGLE_REQUEST_LATENCY: self._run_single_latency,
            ScenarioType.THROUGHPUT_RAMP: self._run_throughput_ramp,
            ScenarioType.LONG_CONTEXT_STRESS: self._run_long_context,
            ScenarioType.PREFIX_SHARING_BENEFIT: self._run_prefix_sharing,
            ScenarioType.STRUCTURED_GENERATION_SPEED: self._run_structured_gen,
        }.get(scenario.scenario_type)

        if handler is None:
            raise ValueError(f"Unknown scenario type: {scenario.scenario_type}")

        requests, engine_timeline = await handler(scenario, client, progress_cb)
        metrics = self._compute_metrics(scenario.name, client.__class__.__name__, requests, engine_timeline)

        return ScenarioResults(
            scenario_name=scenario.name,
            engine_name=client.__class__.__name__,
            run_id=run_id,
            timestamp=ts,
            requests=requests,
            metrics=metrics,
            engine_metrics_timeline=engine_timeline,
            scenario_config=scenario.to_dict(),
        )

    # ------------------------------------------------------------------
    # Scenario handlers
    # ------------------------------------------------------------------

    async def _run_single_latency(
        self,
        scenario: BenchmarkScenario,
        client: BaseInferenceClient,
        progress_cb: ProgressCallback | None,
    ) -> tuple[list[RequestResult], list[dict[str, Any]]]:
        assert isinstance(scenario, SingleRequestLatency)
        prompt = make_short_prompt(scenario.prompt_tokens)
        semaphore = asyncio.Semaphore(scenario.concurrency)
        results: list[RequestResult] = []
        engine_timeline: list[dict[str, Any]] = []

        async def _single(idx: int) -> RequestResult:
            async with semaphore:
                return await self._timed_request(
                    client, idx, prompt, scenario.max_output_tokens, scenario.temperature
                )

        poll_task = asyncio.create_task(
            self._poll_metrics(client, engine_timeline, stop_event := asyncio.Event())
        )

        tasks = [_single(i) for i in range(scenario.num_requests)]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            r = await coro
            results.append(r)
            if progress_cb:
                progress_cb(i + 1, scenario.num_requests, r)

        stop_event.set()
        await poll_task
        results.sort(key=lambda x: x.request_id)
        return results, engine_timeline

    async def _run_throughput_ramp(
        self,
        scenario: BenchmarkScenario,
        client: BaseInferenceClient,
        progress_cb: ProgressCallback | None,
    ) -> tuple[list[RequestResult], list[dict[str, Any]]]:
        assert isinstance(scenario, ThroughputRamp)
        prompt = make_short_prompt(scenario.prompt_tokens)
        all_results: list[RequestResult] = []
        engine_timeline: list[dict[str, Any]] = []

        global_idx = 0
        for concurrency in scenario.concurrency_levels:
            self._log.info("throughput ramp level", concurrency=concurrency)
            semaphore = asyncio.Semaphore(concurrency)
            level_results: list[RequestResult] = []

            stop_event = asyncio.Event()
            poll_task = asyncio.create_task(
                self._poll_metrics(client, engine_timeline, stop_event)
            )

            async def _req(idx: int) -> RequestResult:
                async with semaphore:
                    return await self._timed_request(
                        client, idx, prompt, scenario.max_output_tokens, scenario.temperature
                    )

            tasks = [_req(global_idx + i) for i in range(scenario.requests_per_level)]
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                r = await coro
                level_results.append(r)
                if progress_cb:
                    progress_cb(i + 1, scenario.requests_per_level, r)

            stop_event.set()
            await poll_task

            global_idx += scenario.requests_per_level
            all_results.extend(level_results)

        return all_results, engine_timeline

    async def _run_long_context(
        self,
        scenario: BenchmarkScenario,
        client: BaseInferenceClient,
        progress_cb: ProgressCallback | None,
    ) -> tuple[list[RequestResult], list[dict[str, Any]]]:
        assert isinstance(scenario, LongContextStress)
        prompt = make_long_prompt(scenario.prompt_tokens)
        semaphore = asyncio.Semaphore(scenario.concurrency)
        results: list[RequestResult] = []
        engine_timeline: list[dict[str, Any]] = []

        stop_event = asyncio.Event()
        poll_task = asyncio.create_task(
            self._poll_metrics(client, engine_timeline, stop_event)
        )

        async def _req(idx: int) -> RequestResult:
            async with semaphore:
                return await self._timed_request(
                    client, idx, prompt, scenario.max_output_tokens, scenario.temperature
                )

        tasks = [_req(i) for i in range(scenario.num_requests)]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            r = await coro
            results.append(r)
            if progress_cb:
                progress_cb(i + 1, scenario.num_requests, r)

        stop_event.set()
        await poll_task
        results.sort(key=lambda x: x.request_id)
        return results, engine_timeline

    async def _run_prefix_sharing(
        self,
        scenario: BenchmarkScenario,
        client: BaseInferenceClient,
        progress_cb: ProgressCallback | None,
    ) -> tuple[list[RequestResult], list[dict[str, Any]]]:
        assert isinstance(scenario, PrefixSharingBenefit)
        shared_prefix = make_system_prompt(scenario.shared_prefix_tokens)
        semaphore = asyncio.Semaphore(scenario.concurrency)
        results: list[RequestResult] = []
        engine_timeline: list[dict[str, Any]] = []

        stop_event = asyncio.Event()
        poll_task = asyncio.create_task(
            self._poll_metrics(client, engine_timeline, stop_event)
        )

        async def _req(idx: int) -> RequestResult:
            async with semaphore:
                suffix = make_short_prompt(scenario.user_suffix_tokens) + f" (variant {idx})"
                prompt = shared_prefix + "\n\n" + suffix
                return await self._timed_request(
                    client, idx, prompt, scenario.max_output_tokens, scenario.temperature
                )

        tasks = [_req(i) for i in range(scenario.num_requests)]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            r = await coro
            results.append(r)
            if progress_cb:
                progress_cb(i + 1, scenario.num_requests, r)

        stop_event.set()
        await poll_task
        results.sort(key=lambda x: x.request_id)
        return results, engine_timeline

    async def _run_structured_gen(
        self,
        scenario: BenchmarkScenario,
        client: BaseInferenceClient,
        progress_cb: ProgressCallback | None,
    ) -> tuple[list[RequestResult], list[dict[str, Any]]]:
        assert isinstance(scenario, StructuredGenerationSpeed)
        semaphore = asyncio.Semaphore(scenario.concurrency)
        results: list[RequestResult] = []
        engine_timeline: list[dict[str, Any]] = []

        # Varied entity extraction inputs
        base_texts = [
            "Apple CEO Tim Cook announced Q4 revenue beat expectations.",
            "Microsoft Azure cloud grew 29% YoY, Satya Nadella praised teams.",
            "The Federal Reserve raised rates by 25bps, markets fell sharply.",
            "NASA's Artemis mission successfully landed near the lunar south pole.",
            "OpenAI released GPT-5, causing Google DeepMind to accelerate Gemini.",
        ]

        stop_event = asyncio.Event()
        poll_task = asyncio.create_task(
            self._poll_metrics(client, engine_timeline, stop_event)
        )

        async def _req(idx: int) -> RequestResult:
            async with semaphore:
                text = base_texts[idx % len(base_texts)]
                prompt = make_json_extraction_prompt(text)
                return await self._timed_request(
                    client, idx, prompt, scenario.max_output_tokens, scenario.temperature
                )

        tasks = [_req(i) for i in range(scenario.num_requests)]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            r = await coro
            results.append(r)
            if progress_cb:
                progress_cb(i + 1, scenario.num_requests, r)

        stop_event.set()
        await poll_task
        results.sort(key=lambda x: x.request_id)
        return results, engine_timeline

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _timed_request(
        self,
        client: BaseInferenceClient,
        idx: int,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> RequestResult:
        try:
            gen: GenerationResult = await client.generate(
                prompt, max_tokens=max_tokens, temperature=temperature
            )
            return RequestResult(
                request_id=idx,
                prompt_len=len(prompt.split()),
                success=True,
                ttft_ms=gen.ttft_ms,
                total_ms=gen.total_ms,
                output_tokens=gen.output_tokens,
                tokens_per_sec=gen.tokens_per_sec,
            )
        except Exception as exc:
            self._log.error("request failed", idx=idx, error=str(exc))
            return RequestResult(
                request_id=idx,
                prompt_len=len(prompt.split()),
                success=False,
                error=str(exc)[:200],
            )

    async def _poll_metrics(
        self,
        client: BaseInferenceClient,
        timeline: list[dict[str, Any]],
        stop_event: asyncio.Event,
    ) -> None:
        while not stop_event.is_set():
            try:
                m: EngineMetrics = await client.get_metrics()
                timeline.append({
                    "timestamp": m.timestamp,
                    "gpu_memory_used_gb": m.gpu_memory_used_gb,
                    "kv_cache_usage_pct": m.kv_cache_usage_pct,
                    "pending_requests": m.pending_requests,
                    "running_requests": m.running_requests,
                })
            except Exception as exc:
                self._log.debug("metrics poll failed", error=str(exc))
            try:
                await asyncio.wait_for(
                    asyncio.shield(asyncio.ensure_future(stop_event.wait())),
                    timeout=self.metrics_poll_interval,
                )
            except asyncio.TimeoutError:
                pass

    def _compute_metrics(
        self,
        scenario_name: str,
        engine_name: str,
        requests: list[RequestResult],
        engine_timeline: list[dict[str, Any]],
    ) -> ScenarioMetrics:
        successful = [r for r in requests if r.success]
        ttft_samples = [r.ttft_ms for r in successful]
        total_samples = [r.total_ms for r in successful]
        total_tokens = sum(r.output_tokens for r in successful)
        wall_time = (
            (max(r.total_ms for r in successful) / 1000.0) if successful else 0.0
        )

        latency_stats = LatencyStats.from_samples(total_samples)
        ttft_stats = LatencyStats.from_samples(ttft_samples)
        throughput_stats = ThroughputStats.compute(
            total_requests=len(requests),
            successful_requests=len(successful),
            total_tokens=total_tokens,
            wall_time_sec=wall_time,
        )

        kv_timeline = [e["kv_cache_usage_pct"] for e in engine_timeline]
        gpu_timeline = [e["gpu_memory_used_gb"] for e in engine_timeline]
        error_rate = (len(requests) - len(successful)) / max(len(requests), 1)

        return ScenarioMetrics(
            scenario_name=scenario_name,
            engine_name=engine_name,
            latency=latency_stats,
            throughput=throughput_stats,
            ttft=ttft_stats,
            kv_cache_timeline=kv_timeline,
            gpu_memory_timeline=gpu_timeline,
            error_rate=error_rate,
        )


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

async def run_comparison(
    scenario: BenchmarkScenario,
    vllm_client: BaseInferenceClient,
    sglang_client: BaseInferenceClient,
    results_dir: Path = RESULTS_DIR,
    progress_cb: ProgressCallback | None = None,
) -> ComparisonResult:
    """
    Run the same scenario on both engines sequentially and produce a ComparisonResult.
    """
    runner = BenchmarkRunner(results_dir=results_dir)

    logger.info("running vLLM", scenario=scenario.name)
    vllm_results = await runner.run_scenario(scenario, vllm_client, progress_cb)
    vllm_results.save(results_dir)

    logger.info("running SGLang", scenario=scenario.name)
    sglang_results = await runner.run_scenario(scenario, sglang_client, progress_cb)
    sglang_results.save(results_dir)

    delta = compare_metrics(vllm_results.metrics, sglang_results.metrics)

    comparison = ComparisonResult(
        scenario_name=scenario.name,
        vllm_results=vllm_results,
        sglang_results=sglang_results,
        delta=delta,
    )
    comparison.save(results_dir)
    return comparison


if __name__ == "__main__":
    import asyncio
    from benchmarks.scenarios import SingleRequestLatency
    from engines.vllm_client import VLLMClient
    from engines.sglang_client import SGLangClient

    async def smoke() -> None:
        scenario = SingleRequestLatency(
            name="single_request_latency",
            num_requests=5,
            concurrency=1,
        )
        client = VLLMClient(host="localhost", port=8000)
        healthy = await client.health_check()
        if not healthy:
            print("vLLM not running, skipping live smoke test")
            await client.aclose()
            return

        runner = BenchmarkRunner()
        results = await runner.run_scenario(scenario, client)
        print(f"Completed {len(results.requests)} requests")
        print(f"TTFT p50: {results.metrics.ttft.p50:.1f}ms")
        print(f"Throughput: {results.metrics.throughput.tokens_per_sec:.1f} tok/s")
        path = results.save()
        print(f"Saved to: {path}")
        await client.aclose()

    asyncio.run(smoke())
