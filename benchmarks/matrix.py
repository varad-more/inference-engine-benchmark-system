"""Sequential scenario x engine x iteration matrix execution helpers."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from benchmarks.runner import BenchmarkRunner, ScenarioResults
from benchmarks.scenarios import BenchmarkScenario
from engines.base_client import BaseInferenceClient


@dataclass
class MatrixTask:
    model: str
    scenario_name: str
    engine: str
    iteration: int
    prompt_pack: str | None = None


@dataclass
class MatrixManifest:
    started_at: float
    finished_at: float
    model: str
    tasks: list[dict[str, Any]]
    cooldown_seconds: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, output_path: Path) -> Path:
        output_path.write_text(json.dumps(self.to_dict(), indent=2))
        return output_path


ClientFactory = Callable[[str, str], BaseInferenceClient]


async def run_matrix(
    *,
    model: str,
    scenarios: list[BenchmarkScenario],
    engines: list[str],
    iterations: int,
    cooldown_seconds: int,
    runner: BenchmarkRunner,
    client_factory: ClientFactory,
    results_dir: Path,
) -> MatrixManifest:
    started_at = time.time()
    tasks_log: list[dict[str, Any]] = []

    for scenario in scenarios:
        for iteration in range(1, iterations + 1):
            for engine in engines:
                client = client_factory(engine, model)
                healthy = await client.health_check()
                task_info = {
                    "model": model,
                    "scenario_name": scenario.name,
                    "engine": engine,
                    "iteration": iteration,
                    "healthy_before_run": healthy,
                    "started_at": time.time(),
                }
                if not healthy:
                    task_info["status"] = "skipped_unhealthy"
                    tasks_log.append(task_info)
                    if hasattr(client, "aclose"):
                        await client.aclose()
                    continue

                run_metadata = {
                    "model": model,
                    "iteration": iteration,
                    "matrix_execution": True,
                }
                results: ScenarioResults = await runner.run_scenario(
                    scenario,
                    client,
                    run_metadata=run_metadata,
                )
                result_path = results.save(results_dir)
                task_info.update(
                    {
                        "status": "completed",
                        "finished_at": time.time(),
                        "result_path": str(result_path),
                    }
                )
                tasks_log.append(task_info)
                if hasattr(client, "aclose"):
                    await client.aclose()
                if cooldown_seconds > 0:
                    await asyncio.sleep(cooldown_seconds)

    return MatrixManifest(
        started_at=started_at,
        finished_at=time.time(),
        model=model,
        tasks=tasks_log,
        cooldown_seconds=cooldown_seconds,
    )
