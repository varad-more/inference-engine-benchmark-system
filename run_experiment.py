"""CLI for the vLLM vs SGLang Benchmark System."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import sys
from pathlib import Path

import structlog
import typer
import uvicorn
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from engines.base_client import DEFAULT_MODEL


def _configure_logging() -> None:
    """Configure structlog for JSON (production) or colored console (development)."""
    log_format = os.environ.get("LOG_FORMAT", "console").lower()
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if log_format == "json":
        shared_processors.append(structlog.processors.format_exc_info)
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))


_configure_logging()


def _version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import version as pkg_version

        try:
            v = pkg_version("inference-engine-benchmark")
        except Exception:
            v = "0.1.0-dev"
        typer.echo(f"inference-engine-benchmark {v}")
        raise typer.Exit()


app = typer.Typer(
    name="benchmark",
    help="vLLM vs SGLang comparative inference benchmark system.",
    add_completion=False,
)
console = Console()
logger = structlog.get_logger(__name__)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """vLLM vs SGLang comparative inference benchmark system."""


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _make_client(engine: str, model: str, host_override: str | None = None):
    from engines.sglang_client import SGLangClient
    from engines.vllm_client import VLLMClient

    host = host_override or "localhost"
    if engine == "vllm":
        return VLLMClient(host=host, port=8000, model=model)
    if engine == "sglang":
        return SGLangClient(host=host, port=8001, model=model)

    console.print(f"[red]Unknown engine: {engine}[/red]")
    raise typer.Exit(1)


def _get_scenario(scenario_name: str):
    from benchmarks.scenarios import SCENARIOS

    scenario = SCENARIOS.get(scenario_name)
    if scenario is None:
        console.print(f"[red]Unknown scenario: {scenario_name}[/red]")
        console.print(f"Available: {', '.join(SCENARIOS.keys())}")
        raise typer.Exit(1)
    return copy.deepcopy(scenario)


@app.command()
def run(
    scenario: str = typer.Option(..., "--scenario", "-s", help="Scenario name"),
    engines: str = typer.Option("vllm,sglang", "--engines", "-e", help="Comma-separated engines"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    prompt_pack: str | None = typer.Option(
        None, "--prompt-pack", help="Optional prompt-pack override"
    ),
    vllm_host: str = typer.Option("localhost", "--vllm-host"),
    sglang_host: str = typer.Option("localhost", "--sglang-host"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    strict: bool = typer.Option(False, "--strict", help="Abort if engine health check fails"),
) -> None:
    """Run a benchmark scenario on one or more engines."""
    from benchmarks.runner import BenchmarkRunner, RequestResult

    results_dir = output_dir or RESULTS_DIR
    results_dir.mkdir(exist_ok=True)
    engine_list = [engine.lower() for engine in _parse_csv(engines)]
    bench_scenario = _get_scenario(scenario)
    if prompt_pack:
        bench_scenario.prompt_pack = prompt_pack

    console.rule(f"[bold blue]Benchmark: {scenario}")
    console.print(f"  Engines     : [cyan]{', '.join(engine_list)}[/cyan]")
    console.print(f"  Model       : [cyan]{model}[/cyan]")
    console.print(f"  Prompt pack : [cyan]{bench_scenario.prompt_pack}[/cyan]")
    console.print(f"  Results     : [cyan]{results_dir}[/cyan]")

    async def _run() -> None:
        runner = BenchmarkRunner(results_dir=results_dir)
        active_client = None

        try:
            for engine_name in engine_list:
                host = vllm_host if engine_name == "vllm" else sglang_host
                client = _make_client(engine_name, model, host)
                active_client = client

                healthy = await client.health_check()
                if not healthy:
                    if strict:
                        console.print(
                            f"[red]Error: {engine_name} health check failed (--strict mode).[/red]"
                        )
                        await client.aclose()
                        raise typer.Exit(1)
                    console.print(
                        f"[yellow]Warning: {engine_name} health check failed. Proceeding anyway.[/yellow]"
                    )

                total_reqs = getattr(bench_scenario, "num_requests", None) or getattr(
                    bench_scenario,
                    "requests_per_level",
                    100,
                )

                with Progress(
                    SpinnerColumn(),
                    TextColumn(f"[bold]{engine_name}[/bold] {{task.description}}"),
                    BarColumn(),
                    TextColumn("{task.completed}/{task.total}"),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    task_id = progress.add_task("running...", total=total_reqs)

                    def _cb(done: int, total: int, result: RequestResult) -> None:
                        progress.update(task_id, completed=done)

                    results = await runner.run_scenario(
                        bench_scenario,
                        client,
                        _cb,
                        run_metadata={
                            "model": model,
                            "engine": engine_name,
                            "prompt_pack": bench_scenario.prompt_pack,
                        },
                    )

                path = results.save(results_dir)
                console.print(f"\n[green]✓[/green] {engine_name} done → [dim]{path}[/dim]")
                _print_summary_table(engine_name, results)

                await client.aclose()
                active_client = None
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted — cleaning up...[/yellow]")
        finally:
            if active_client is not None:
                await active_client.aclose()

    asyncio.run(_run())


@app.command()
def compare(
    scenario: str = typer.Option(..., "--scenario", "-s"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    prompt_pack: str | None = typer.Option(None, "--prompt-pack"),
    vllm_host: str = typer.Option("localhost", "--vllm-host"),
    sglang_host: str = typer.Option("localhost", "--sglang-host"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
) -> None:
    """Run the same scenario on both engines and print a comparison table."""
    from benchmarks.runner import run_comparison
    from engines.sglang_client import SGLangClient
    from engines.vllm_client import VLLMClient

    results_dir = output_dir or RESULTS_DIR
    results_dir.mkdir(exist_ok=True)
    bench_scenario = _get_scenario(scenario)
    if prompt_pack:
        bench_scenario.prompt_pack = prompt_pack

    vllm_client = VLLMClient(host=vllm_host, port=8000, model=model)
    sglang_client = SGLangClient(host=sglang_host, port=8001, model=model)

    console.rule(f"[bold blue]Comparison: {scenario}")
    console.print(f"  Model       : [cyan]{model}[/cyan]")
    console.print(f"  Prompt pack : [cyan]{bench_scenario.prompt_pack}[/cyan]")

    async def _run() -> None:
        comparison = await run_comparison(
            scenario=bench_scenario,
            vllm_client=vllm_client,
            sglang_client=sglang_client,
            results_dir=results_dir,
            run_metadata={"model": model, "prompt_pack": bench_scenario.prompt_pack},
        )
        await vllm_client.aclose()
        await sglang_client.aclose()

        _print_comparison_table(comparison)
        path = comparison.save(results_dir)
        console.print(f"\n[green]Comparison saved → {path}[/green]")

    asyncio.run(_run())


@app.command()
def matrix(
    scenarios: str = typer.Option(
        "single_request_latency,throughput_ramp",
        "--scenarios",
        help="Comma-separated scenarios for the matrix",
    ),
    engines: str = typer.Option("sglang,vllm", "--engines", help="Comma-separated engines"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    prompt_pack: str | None = typer.Option(
        None, "--prompt-pack", help="Optional override for all scenarios"
    ),
    iterations: int = typer.Option(1, "--iterations", min=1),
    cooldown_seconds: int = typer.Option(60, "--cooldown-seconds", min=0),
    vllm_host: str = typer.Option("localhost", "--vllm-host"),
    sglang_host: str = typer.Option("localhost", "--sglang-host"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
) -> None:
    """Run a sequential scenario x engine x iteration matrix for one active model."""
    import json
    import time

    from benchmarks.runner import BenchmarkRunner

    results_dir = output_dir or RESULTS_DIR
    results_dir.mkdir(exist_ok=True)
    scenario_list = [_get_scenario(name) for name in _parse_csv(scenarios)]
    if prompt_pack:
        for scenario_obj in scenario_list:
            scenario_obj.prompt_pack = prompt_pack
    engine_list = [engine.lower() for engine in _parse_csv(engines)]

    console.rule("[bold blue]Matrix execution")
    console.print(f"  Model          : [cyan]{model}[/cyan]")
    console.print(f"  Scenarios      : [cyan]{', '.join(s.name for s in scenario_list)}[/cyan]")
    console.print(f"  Engines        : [cyan]{', '.join(engine_list)}[/cyan]")
    console.print(f"  Iterations     : [cyan]{iterations}[/cyan]")
    console.print(f"  Cooldown (sec) : [cyan]{cooldown_seconds}[/cyan]")

    async def _run() -> None:
        runner = BenchmarkRunner(results_dir=results_dir)
        started_at = time.time()
        tasks_log: list[dict] = []

        for bench_scenario in scenario_list:
            for iteration in range(1, iterations + 1):
                for engine_name in engine_list:
                    host = vllm_host if engine_name == "vllm" else sglang_host
                    client = _make_client(engine_name, model, host)
                    healthy = await client.health_check()
                    task_info: dict = {
                        "model": model,
                        "scenario_name": bench_scenario.name,
                        "engine": engine_name,
                        "iteration": iteration,
                        "healthy_before_run": healthy,
                        "started_at": time.time(),
                    }
                    if not healthy:
                        task_info["status"] = "skipped_unhealthy"
                        tasks_log.append(task_info)
                        await client.aclose()
                        continue

                    results = await runner.run_scenario(
                        bench_scenario,
                        client,
                        run_metadata={
                            "model": model,
                            "iteration": iteration,
                            "matrix_execution": True,
                        },
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
                    await client.aclose()
                    if cooldown_seconds > 0:
                        await asyncio.sleep(cooldown_seconds)

        manifest = {
            "started_at": started_at,
            "finished_at": time.time(),
            "model": model,
            "tasks": tasks_log,
            "cooldown_seconds": cooldown_seconds,
        }
        manifest_path = results_dir / f"matrix_manifest_{int(started_at)}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        console.print(f"[green]✓[/green] Matrix manifest saved to {manifest_path}")

    asyncio.run(_run())


@app.command()
def report(
    output: Path = typer.Option(Path("report.html"), "--output", "-o"),
    results_dir: Path | None = typer.Option(None, "--results-dir"),
) -> None:
    """Generate an HTML report from all saved results."""
    from analysis.report import generate_report

    src_dir = results_dir or RESULTS_DIR
    console.print(f"Generating report from [cyan]{src_dir}[/cyan] → [cyan]{output}[/cyan]")
    generate_report(results_dir=src_dir, output_path=output)
    console.print(f"[green]✓[/green] Report written to {output}")


@app.command("final-report")
def final_report(
    output: Path = typer.Option(Path("final_report.md"), "--output", "-o"),
    results_dir: Path | None = typer.Option(None, "--results-dir"),
) -> None:
    """Generate an aggregated markdown summary from all saved result files."""
    from analysis.final_report import generate_final_report

    src_dir = results_dir or RESULTS_DIR
    summary = generate_final_report(results_dir=src_dir, output_path=output)
    console.print(
        f"[green]✓[/green] Final report written to {output} from {summary['total_result_files']} result files"
    )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(3000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the benchmark dashboard server."""
    console.print(f"[bold]Starting dashboard on http://{host}:{port}[/bold]")
    uvicorn.run("dashboard.app:app", host=host, port=port, reload=reload, log_level="info")


@app.command("list-scenarios")
def list_scenarios() -> None:
    """List all available benchmark scenarios."""
    from benchmarks.scenarios import SCENARIOS

    table = Table(title="Available Scenarios", show_lines=True)
    table.add_column("Name", style="cyan")
    table.add_column("Type")
    table.add_column("Prompt pack")
    table.add_column("Description")
    for name, scenario in SCENARIOS.items():
        table.add_row(
            name,
            scenario.scenario_type.value,
            str(scenario.prompt_pack),
            scenario.description[:80],
        )
    console.print(table)


@app.command("list-prompt-packs")
def list_prompt_packs() -> None:
    """List prompt packs shipped with the repo."""
    from benchmarks.prompt_packs import list_prompt_packs

    table = Table(title="Available Prompt Packs")
    table.add_column("Prompt pack", style="cyan")
    for pack_name in list_prompt_packs():
        table.add_row(pack_name)
    console.print(table)


@app.command()
def health(
    vllm_host: str = typer.Option("localhost", "--vllm-host"),
    sglang_host: str = typer.Option("localhost", "--sglang-host"),
    model: str = typer.Option(DEFAULT_MODEL, "--model"),
) -> None:
    """Check health of both inference engines."""
    from engines.sglang_client import SGLangClient
    from engines.vllm_client import VLLMClient

    async def _check() -> None:
        vllm = VLLMClient(host=vllm_host, port=8000, model=model)
        sglang = SGLangClient(host=sglang_host, port=8001, model=model)

        v_ok = await vllm.health_check()
        s_ok = await sglang.health_check()
        await vllm.aclose()
        await sglang.aclose()

        table = Table(title="Engine Health")
        table.add_column("Engine")
        table.add_column("Status")
        table.add_column("URL")
        table.add_row(
            "vLLM",
            "[green]✓ healthy[/green]" if v_ok else "[red]✗ unreachable[/red]",
            f"http://{vllm_host}:8000",
        )
        table.add_row(
            "SGLang",
            "[green]✓ healthy[/green]" if s_ok else "[red]✗ unreachable[/red]",
            f"http://{sglang_host}:8001",
        )
        console.print(table)

    asyncio.run(_check())


def _print_summary_table(engine_name: str, results) -> None:
    metrics = results.metrics
    table = Table(title=f"[bold]{engine_name}[/bold] — {results.scenario_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Requests", str(metrics.throughput.total_requests))
    table.add_row("Success rate", f"{(1 - metrics.error_rate) * 100:.1f}%")
    table.add_row("TTFT p50", f"{metrics.ttft.p50:.1f} ms")
    table.add_row("TTFT p95", f"{metrics.ttft.p95:.1f} ms")
    table.add_row("TTFT p99", f"{metrics.ttft.p99:.1f} ms")
    table.add_row("Total latency p95", f"{metrics.latency.p95:.1f} ms")
    table.add_row("Tokens/sec", f"{metrics.throughput.tokens_per_sec:.1f}")
    table.add_row("Requests/sec", f"{metrics.throughput.requests_per_sec:.2f}")
    if metrics.kv_cache_timeline:
        avg_kv = sum(metrics.kv_cache_timeline) / len(metrics.kv_cache_timeline)
        table.add_row("Avg KV cache usage", f"{avg_kv * 100:.1f}%")
    console.print(table)


def _print_comparison_table(comparison) -> None:
    vr = comparison.vllm_results.metrics
    sr = comparison.sglang_results.metrics

    table = Table(title=f"Comparison — {comparison.scenario_name}", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("vLLM", justify="right")
    table.add_column("SGLang", justify="right")
    table.add_column("Δ (vllm−sglang)", justify="right")

    def _row(label: str, va: float, vb: float, unit: str = "", lower_better: bool = True) -> None:
        diff = va - vb
        color = "green" if (diff < 0) == lower_better else "red"
        sign = "+" if diff > 0 else ""
        table.add_row(
            label,
            f"{va:.2f}{unit}",
            f"{vb:.2f}{unit}",
            f"[{color}]{sign}{diff:.2f}{unit}[/{color}]",
        )

    _row("TTFT p50 (ms)", vr.ttft.p50, sr.ttft.p50, "ms")
    _row("TTFT p95 (ms)", vr.ttft.p95, sr.ttft.p95, "ms")
    _row("TTFT p99 (ms)", vr.ttft.p99, sr.ttft.p99, "ms")
    _row("Latency p95 (ms)", vr.latency.p95, sr.latency.p95, "ms")
    _row(
        "Tokens/sec",
        vr.throughput.tokens_per_sec,
        sr.throughput.tokens_per_sec,
        "",
        lower_better=False,
    )
    _row(
        "Requests/sec",
        vr.throughput.requests_per_sec,
        sr.throughput.requests_per_sec,
        "",
        lower_better=False,
    )
    _row("Error rate", vr.error_rate * 100, sr.error_rate * 100, "%")

    console.print(table)


if __name__ == "__main__":
    app()
