"""
Benchmark runner: executes scenarios against inference engine clients,
collects per-request metrics, polls engine health, and saves JSON results.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog

from benchmarks.metrics import (
    CompareMetricsResult,
    LatencyStats,
    ScenarioMetrics,
    ThroughputStats,
    compare_metrics,
)
from benchmarks.prompt_packs import (
    PromptRecord,
    cycle_prompt_pack,
    default_prompt_pack_for_scenario,
    load_shared_prefix_pack,
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


@dataclass
class RequestResult:
    request_id: int
    prompt_len: int
    success: bool
    prompt_id: str = ""
    prompt_category: str = ""
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    output_tokens: int = 0
    tokens_per_sec: float = 0.0
    error: str = ""


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
    workload_metadata: dict[str, Any] = field(default_factory=dict)
    run_metadata: dict[str, Any] = field(default_factory=dict)

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
            "workload_metadata": self.workload_metadata,
            "run_metadata": self.run_metadata,
        }

    def save(self, results_dir: Path = RESULTS_DIR) -> Path:
        ts = int(self.timestamp)
        fname = f"{self.scenario_name}_{self.engine_name}_{ts}.json"
        path = results_dir / fname
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("saved results", path=str(path))
        return path


@dataclass
class ComparisonResult:
    scenario_name: str
    vllm_results: ScenarioResults
    sglang_results: ScenarioResults
    delta: CompareMetricsResult

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


ProgressCallback = Callable[[int, int, RequestResult], None]


class BenchmarkRunner:
    """Runs benchmark scenarios against an inference client."""

    def __init__(
        self,
        results_dir: Path = RESULTS_DIR,
        metrics_poll_interval: float = 2.0,
    ) -> None:
        self.results_dir = results_dir
        self.metrics_poll_interval = metrics_poll_interval
        self.results_dir.mkdir(exist_ok=True)
        self._log = structlog.get_logger(self.__class__.__name__)

    async def run_scenario(
        self,
        scenario: BenchmarkScenario,
        client: BaseInferenceClient,
        progress_cb: ProgressCallback | None = None,
        run_metadata: dict[str, Any] | None = None,
    ) -> ScenarioResults:
        """Dispatch to the appropriate scenario handler."""
        self._log.info(
            "starting scenario",
            scenario=scenario.name,
            engine=client.__class__.__name__,
            prompt_pack=scenario.prompt_pack,
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

        requests, engine_timeline, workload_metadata = await handler(scenario, client, progress_cb)
        metrics = self._compute_metrics(
            scenario.name,
            client.__class__.__name__,
            requests,
            engine_timeline,
        )

        merged_run_metadata = dict(run_metadata or {})
        merged_run_metadata.setdefault("model", getattr(client, "model", "unknown"))
        merged_run_metadata.setdefault("host", getattr(client, "host", "unknown"))
        merged_run_metadata.setdefault("port", getattr(client, "port", "unknown"))

        return ScenarioResults(
            scenario_name=scenario.name,
            engine_name=client.__class__.__name__,
            run_id=run_id,
            timestamp=ts,
            requests=requests,
            metrics=metrics,
            engine_metrics_timeline=engine_timeline,
            scenario_config=scenario.to_dict(),
            workload_metadata=workload_metadata,
            run_metadata=merged_run_metadata,
        )

    async def _run_concurrent_from_records(
        self,
        client: BaseInferenceClient,
        prompt_records: list[PromptRecord],
        num_requests: int,
        concurrency: int,
        max_output_tokens: int,
        temperature: float,
        progress_cb: ProgressCallback | None,
    ) -> tuple[list[RequestResult], list[dict[str, Any]]]:
        """Shared fan-out/gather logic used by most scenario handlers."""
        semaphore = asyncio.Semaphore(concurrency)
        results: list[RequestResult] = []
        engine_timeline: list[dict[str, Any]] = []

        stop_event = asyncio.Event()
        poll_task = asyncio.create_task(self._poll_metrics(client, engine_timeline, stop_event))

        async def _req(idx: int) -> RequestResult:
            async with semaphore:
                record = prompt_records[idx]
                return await self._timed_request(
                    client,
                    idx,
                    record.prompt,
                    max_output_tokens,
                    temperature,
                    prompt_id=record.id,
                    prompt_category=record.category,
                )

        tasks = [_req(i) for i in range(num_requests)]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            r = await coro
            results.append(r)
            if progress_cb:
                progress_cb(i + 1, num_requests, r)

        stop_event.set()
        await poll_task
        results.sort(key=lambda x: x.request_id)
        return results, engine_timeline

    async def _run_single_latency(
        self,
        scenario: BenchmarkScenario,
        client: BaseInferenceClient,
        progress_cb: ProgressCallback | None,
    ) -> tuple[list[RequestResult], list[dict[str, Any]], dict[str, Any]]:
        assert isinstance(scenario, SingleRequestLatency)
        prompt_records, workload_metadata = self._prompt_records_for_scenario(
            scenario_name=scenario.name,
            prompt_pack=scenario.prompt_pack,
            count=scenario.num_requests,
            fallback_builder=lambda: make_short_prompt(scenario.prompt_tokens),
        )
        results, engine_timeline = await self._run_concurrent_from_records(
            client,
            prompt_records,
            scenario.num_requests,
            scenario.concurrency,
            scenario.max_output_tokens,
            scenario.temperature,
            progress_cb,
        )
        return results, engine_timeline, workload_metadata

    async def _run_throughput_ramp(
        self,
        scenario: BenchmarkScenario,
        client: BaseInferenceClient,
        progress_cb: ProgressCallback | None,
    ) -> tuple[list[RequestResult], list[dict[str, Any]], dict[str, Any]]:
        assert isinstance(scenario, ThroughputRamp)
        total_requests = scenario.requests_per_level * len(scenario.concurrency_levels)
        prompt_records, workload_metadata = self._prompt_records_for_scenario(
            scenario_name=scenario.name,
            prompt_pack=scenario.prompt_pack,
            count=total_requests,
            fallback_builder=lambda: make_short_prompt(scenario.prompt_tokens),
        )
        all_results: list[RequestResult] = []
        engine_timeline: list[dict[str, Any]] = []

        global_idx = 0
        for concurrency in scenario.concurrency_levels:
            self._log.info("throughput ramp level", concurrency=concurrency)
            level_records = prompt_records[global_idx : global_idx + scenario.requests_per_level]
            level_results, level_timeline = await self._run_concurrent_from_records(
                client,
                level_records,
                scenario.requests_per_level,
                concurrency,
                scenario.max_output_tokens,
                scenario.temperature,
                progress_cb,
            )
            engine_timeline.extend(level_timeline)
            global_idx += scenario.requests_per_level
            all_results.extend(level_results)

        return all_results, engine_timeline, workload_metadata

    async def _run_long_context(
        self,
        scenario: BenchmarkScenario,
        client: BaseInferenceClient,
        progress_cb: ProgressCallback | None,
    ) -> tuple[list[RequestResult], list[dict[str, Any]], dict[str, Any]]:
        assert isinstance(scenario, LongContextStress)
        prompt_records, workload_metadata = self._prompt_records_for_scenario(
            scenario_name=scenario.name,
            prompt_pack=scenario.prompt_pack,
            count=scenario.num_requests,
            fallback_builder=lambda: make_long_prompt(scenario.prompt_tokens),
        )
        results, engine_timeline = await self._run_concurrent_from_records(
            client,
            prompt_records,
            scenario.num_requests,
            scenario.concurrency,
            scenario.max_output_tokens,
            scenario.temperature,
            progress_cb,
        )
        return results, engine_timeline, workload_metadata

    async def _run_prefix_sharing(
        self,
        scenario: BenchmarkScenario,
        client: BaseInferenceClient,
        progress_cb: ProgressCallback | None,
    ) -> tuple[list[RequestResult], list[dict[str, Any]], dict[str, Any]]:
        assert isinstance(scenario, PrefixSharingBenefit)
        pack = self._shared_prefix_pack_for_scenario(scenario)
        workload_metadata = {
            "prompt_pack": scenario.prompt_pack,
            "prompt_source": "prompt_pack",
            "shared_prefix_id": pack.id,
            "shared_prefix_length_words": len(pack.shared_prefix.split()),
            "prompt_ids": [
                f"{pack.id}:{idx % len(pack.suffixes)}" for idx in range(scenario.num_requests)
            ],
        }
        # Build PromptRecord list from shared prefix + suffixes
        prompt_records = [
            PromptRecord(
                id=f"{pack.id}:{idx % len(pack.suffixes)}",
                category=pack.category,
                prompt=pack.shared_prefix + "\n\n" + pack.suffixes[idx % len(pack.suffixes)],
            )
            for idx in range(scenario.num_requests)
        ]
        results, engine_timeline = await self._run_concurrent_from_records(
            client,
            prompt_records,
            scenario.num_requests,
            scenario.concurrency,
            scenario.max_output_tokens,
            scenario.temperature,
            progress_cb,
        )
        return results, engine_timeline, workload_metadata

    async def _run_structured_gen(
        self,
        scenario: BenchmarkScenario,
        client: BaseInferenceClient,
        progress_cb: ProgressCallback | None,
    ) -> tuple[list[RequestResult], list[dict[str, Any]], dict[str, Any]]:
        assert isinstance(scenario, StructuredGenerationSpeed)
        prompt_records, workload_metadata = self._prompt_records_for_scenario(
            scenario_name=scenario.name,
            prompt_pack=scenario.prompt_pack,
            count=scenario.num_requests,
            fallback_builder=lambda: make_json_extraction_prompt(
                "Apple CEO Tim Cook announced Q4 revenue beat expectations."
            ),
        )
        workload_metadata["schemas_used"] = sorted(
            {record.schema for record in prompt_records if record.schema}
        )
        results, engine_timeline = await self._run_concurrent_from_records(
            client,
            prompt_records,
            scenario.num_requests,
            scenario.concurrency,
            scenario.max_output_tokens,
            scenario.temperature,
            progress_cb,
        )
        return results, engine_timeline, workload_metadata

    async def _timed_request(
        self,
        client: BaseInferenceClient,
        idx: int,
        prompt: str,
        max_tokens: int,
        temperature: float,
        prompt_id: str = "",
        prompt_category: str = "",
    ) -> RequestResult:
        try:
            gen: GenerationResult = await client.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return RequestResult(
                request_id=idx,
                prompt_id=prompt_id,
                prompt_category=prompt_category,
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
                prompt_id=prompt_id,
                prompt_category=prompt_category,
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
                timeline.append(
                    {
                        "timestamp": m.timestamp,
                        "gpu_memory_used_gb": m.gpu_memory_used_gb,
                        "kv_cache_usage_pct": m.kv_cache_usage_pct,
                        "pending_requests": m.pending_requests,
                        "running_requests": m.running_requests,
                    }
                )
            except Exception as exc:
                self._log.debug("metrics poll failed", error=str(exc))
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.metrics_poll_interval)
            except TimeoutError:
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

        wall_time = 0.0
        if len(engine_timeline) >= 2:
            timestamps = [entry["timestamp"] for entry in engine_timeline if "timestamp" in entry]
            if len(timestamps) >= 2:
                wall_time = max(timestamps) - min(timestamps)

        if wall_time <= 0 and scenario_name == "single_request_latency" and successful:
            # Sequential run: summing per-request totals approximates full wall-clock duration.
            wall_time = sum(r.total_ms for r in successful) / 1000.0

        if wall_time <= 0 and successful:
            # Last-resort fallback.
            wall_time = max(r.total_ms for r in successful) / 1000.0

        latency_stats = LatencyStats.from_samples(total_samples)
        ttft_stats = LatencyStats.from_samples(ttft_samples)
        throughput_stats = ThroughputStats.compute(
            total_requests=len(requests),
            successful_requests=len(successful),
            total_tokens=total_tokens,
            wall_time_sec=wall_time,
        )

        kv_timeline = [entry["kv_cache_usage_pct"] for entry in engine_timeline]
        gpu_timeline = [entry["gpu_memory_used_gb"] for entry in engine_timeline]
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

    def _prompt_records_for_scenario(
        self,
        scenario_name: str,
        prompt_pack: str | None,
        count: int,
        fallback_builder: Callable[[], str],
    ) -> tuple[list[PromptRecord], dict[str, Any]]:
        pack_name = prompt_pack or default_prompt_pack_for_scenario(scenario_name)
        try:
            prompt_records = cycle_prompt_pack(pack_name, count)
            return prompt_records, {
                "prompt_pack": pack_name,
                "prompt_source": "prompt_pack",
                "prompt_ids": [record.id for record in prompt_records],
                "categories": sorted({record.category for record in prompt_records}),
            }
        except Exception as exc:
            self._log.warning(
                "prompt pack unavailable, using synthetic fallback",
                scenario=scenario_name,
                prompt_pack=pack_name,
                error=str(exc),
            )
            prompt_records = [
                PromptRecord(
                    id=f"synthetic_{scenario_name}_{idx:04d}",
                    category="synthetic",
                    prompt=fallback_builder(),
                )
                for idx in range(count)
            ]
            return prompt_records, {
                "prompt_pack": pack_name,
                "prompt_source": "synthetic_fallback",
                "prompt_ids": [record.id for record in prompt_records],
                "categories": ["synthetic"],
            }

    def _shared_prefix_pack_for_scenario(self, scenario: PrefixSharingBenefit):
        pack_name = scenario.prompt_pack or default_prompt_pack_for_scenario(scenario.name)
        if pack_name != "shared_prefix":
            self._log.warning(
                "prefix-sharing scenario ignores non-shared pack override",
                prompt_pack=pack_name,
            )
        try:
            return load_shared_prefix_pack()
        except Exception as exc:
            self._log.warning(
                "shared prefix pack unavailable, using synthetic fallback", error=str(exc)
            )
            return type(
                "SyntheticSharedPrefix",
                (),
                {
                    "id": "synthetic_shared_prefix",
                    "category": "shared_prefix",
                    "shared_prefix": make_system_prompt(scenario.shared_prefix_tokens),
                    "suffixes": tuple(
                        make_short_prompt(scenario.user_suffix_tokens) + f" (variant {idx})"
                        for idx in range(5)
                    ),
                },
            )()


async def run_comparison(
    scenario: BenchmarkScenario,
    vllm_client: BaseInferenceClient,
    sglang_client: BaseInferenceClient,
    results_dir: Path = RESULTS_DIR,
    progress_cb: ProgressCallback | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> ComparisonResult:
    """Run the same scenario on both engines sequentially."""
    runner = BenchmarkRunner(results_dir=results_dir)

    logger.info("running vLLM", scenario=scenario.name)
    vllm_results = await runner.run_scenario(scenario, vllm_client, progress_cb, run_metadata)
    vllm_results.save(results_dir)

    logger.info("running SGLang", scenario=scenario.name)
    sglang_results = await runner.run_scenario(scenario, sglang_client, progress_cb, run_metadata)
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

    async def smoke() -> None:
        scenario = SingleRequestLatency(
            name="single_request_latency", num_requests=5, concurrency=1
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
