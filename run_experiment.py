"""
CLI for the vLLM vs SGLang Benchmark System.

Usage:
    python run_experiment.py run --scenario throughput_ramp --engines vllm,sglang
    python run_experiment.py compare --scenario prefix_sharing_benefit
    python run_experiment.py report --output report.html
    python run_experiment.py serve
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

import structlog
import typer
import uvicorn
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

app = typer.Typer(
    name="benchmark",
    help="vLLM vs SGLang comparative inference benchmark system.",
    add_completion=False,
)
console = Console()
logger = structlog.get_logger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_engines(engines_str: str) -> list[str]:
    return [e.strip().lower() for e in engines_str.split(",")]


def _make_client(engine: str, model: str, host_override: str | None = None):
    from engines.vllm_client import VLLMClient
    from engines.sglang_client import SGLangClient

    if engine == "vllm":
        host = host_override or "localhost"
        return VLLMClient(host=host, port=8000, model=model)
    elif engine == "sglang":
        host = host_override or "localhost"
        return SGLangClient(host=host, port=8001, model=model)
    else:
        console.print(f"[red]Unknown engine: {engine}[/red]")
        raise typer.Exit(1)


def _get_scenario(scenario_name: str):
    from benchmarks.scenarios import SCENARIOS
    s = SCENARIOS.get(scenario_name)
    if s is None:
        console.print(f"[red]Unknown scenario: {scenario_name}[/red]")
        console.print(f"Available: {', '.join(SCENARIOS.keys())}")
        raise typer.Exit(1)
    return s


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def run(
    scenario: str = typer.Option(..., "--scenario", "-s", help="Scenario name"),
    engines: str = typer.Option("vllm,sglang", "--engines", "-e", help="Comma-separated engines"),
    model: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model", "-m"),
    vllm_host: str = typer.Option("localhost", "--vllm-host"),
    sglang_host: str = typer.Option("localhost", "--sglang-host"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
) -> None:
    """Run a benchmark scenario on one or more engines."""
    from benchmarks.runner import BenchmarkRunner, RequestResult

    results_dir = output_dir or RESULTS_DIR
    results_dir.mkdir(exist_ok=True)
    engine_list = _parse_engines(engines)
    bench_scenario = _get_scenario(scenario)

    console.rule(f"[bold blue]Benchmark: {scenario}")
    console.print(f"  Engines : [cyan]{', '.join(engine_list)}[/cyan]")
    console.print(f"  Model   : [cyan]{model}[/cyan]")
    console.print(f"  Results : [cyan]{results_dir}[/cyan]")

    async def _run() -> None:
        runner = BenchmarkRunner(results_dir=results_dir)

        for engine_name in engine_list:
            host = vllm_host if engine_name == "vllm" else sglang_host
            client = _make_client(engine_name, model, host)

            healthy = await client.health_check()
            if not healthy:
                console.print(f"[yellow]Warning: {engine_name} health check failed. Proceeding anyway.[/yellow]")

            total_reqs = getattr(bench_scenario, "num_requests", None) or getattr(bench_scenario, "requests_per_level", 100)
            completed = {"n": 0}

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

                results = await runner.run_scenario(bench_scenario, client, _cb)

            path = results.save(results_dir)
            console.print(f"\n[green]✓[/green] {engine_name} done → [dim]{path}[/dim]")
            _print_summary_table(engine_name, results)

            if hasattr(client, "aclose"):
                await client.aclose()

    asyncio.run(_run())


@app.command()
def compare(
    scenario: str = typer.Option(..., "--scenario", "-s"),
    model: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model", "-m"),
    vllm_host: str = typer.Option("localhost", "--vllm-host"),
    sglang_host: str = typer.Option("localhost", "--sglang-host"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
) -> None:
    """Run the same scenario on both engines and print a comparison table."""
    from benchmarks.runner import run_comparison
    from engines.vllm_client import VLLMClient
    from engines.sglang_client import SGLangClient

    results_dir = output_dir or RESULTS_DIR
    results_dir.mkdir(exist_ok=True)
    bench_scenario = _get_scenario(scenario)

    vllm_client = VLLMClient(host=vllm_host, port=8000, model=model)
    sglang_client = SGLangClient(host=sglang_host, port=8001, model=model)

    console.rule(f"[bold blue]Comparison: {scenario}")

    async def _run() -> None:
        comparison = await run_comparison(
            scenario=bench_scenario,
            vllm_client=vllm_client,
            sglang_client=sglang_client,
            results_dir=results_dir,
        )
        await vllm_client.aclose()
        await sglang_client.aclose()

        _print_comparison_table(comparison)
        path = comparison.save(results_dir)
        console.print(f"\n[green]Comparison saved → {path}[/green]")

    asyncio.run(_run())


@app.command()
def report(
    output: Path = typer.Option(Path("report.html"), "--output", "-o"),
    results_dir: Optional[Path] = typer.Option(None, "--results-dir"),
) -> None:
    """Generate an HTML report from all saved results."""
    from analysis.report import generate_report

    src_dir = results_dir or RESULTS_DIR
    console.print(f"Generating report from [cyan]{src_dir}[/cyan] → [cyan]{output}[/cyan]")
    generate_report(results_dir=src_dir, output_path=output)
    console.print(f"[green]✓[/green] Report written to {output}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(3000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the benchmark dashboard server."""
    console.print(f"[bold]Starting dashboard on http://{host}:{port}[/bold]")
    uvicorn.run(
        "dashboard.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@app.command()
def list_scenarios() -> None:
    """List all available benchmark scenarios."""
    from benchmarks.scenarios import SCENARIOS

    table = Table(title="Available Scenarios", show_lines=True)
    table.add_column("Name", style="cyan")
    table.add_column("Type")
    table.add_column("Description")
    for name, s in SCENARIOS.items():
        table.add_row(name, s.scenario_type.value, s.description[:80])
    console.print(table)


@app.command()
def health(
    vllm_host: str = typer.Option("localhost", "--vllm-host"),
    sglang_host: str = typer.Option("localhost", "--sglang-host"),
    model: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model"),
) -> None:
    """Check health of both inference engines."""
    from engines.vllm_client import VLLMClient
    from engines.sglang_client import SGLangClient

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
        table.add_row("vLLM", "[green]✓ healthy[/green]" if v_ok else "[red]✗ unreachable[/red]", f"http://{vllm_host}:8000")
        table.add_row("SGLang", "[green]✓ healthy[/green]" if s_ok else "[red]✗ unreachable[/red]", f"http://{sglang_host}:8001")
        console.print(table)

    asyncio.run(_check())


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def _print_summary_table(engine_name: str, results) -> None:
    m = results.metrics
    table = Table(title=f"[bold]{engine_name}[/bold] — {results.scenario_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Requests", str(m.throughput.total_requests))
    table.add_row("Success rate", f"{(1 - m.error_rate)*100:.1f}%")
    table.add_row("TTFT p50", f"{m.ttft.p50:.1f} ms")
    table.add_row("TTFT p95", f"{m.ttft.p95:.1f} ms")
    table.add_row("TTFT p99", f"{m.ttft.p99:.1f} ms")
    table.add_row("Total latency p95", f"{m.latency.p95:.1f} ms")
    table.add_row("Tokens/sec", f"{m.throughput.tokens_per_sec:.1f}")
    table.add_row("Requests/sec", f"{m.throughput.requests_per_sec:.2f}")
    if m.kv_cache_timeline:
        avg_kv = sum(m.kv_cache_timeline) / len(m.kv_cache_timeline)
        table.add_row("Avg KV cache usage", f"{avg_kv*100:.1f}%")
    console.print(table)


def _print_comparison_table(comparison) -> None:
    vr = comparison.vllm_results.metrics
    sr = comparison.sglang_results.metrics
    d = comparison.delta

    table = Table(title=f"Comparison — {comparison.scenario_name}", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("vLLM", justify="right")
    table.add_column("SGLang", justify="right")
    table.add_column("Δ (vllm−sglang)", justify="right")

    def _row(label: str, va: float, vb: float, unit: str = "", lower_better: bool = True) -> None:
        diff = va - vb
        color = "green" if (diff < 0) == lower_better else "red"
        sign = "+" if diff > 0 else ""
        table.add_row(label, f"{va:.2f}{unit}", f"{vb:.2f}{unit}", f"[{color}]{sign}{diff:.2f}{unit}[/{color}]")

    _row("TTFT p50 (ms)", vr.ttft.p50, sr.ttft.p50, "ms")
    _row("TTFT p95 (ms)", vr.ttft.p95, sr.ttft.p95, "ms")
    _row("TTFT p99 (ms)", vr.ttft.p99, sr.ttft.p99, "ms")
    _row("Latency p95 (ms)", vr.latency.p95, sr.latency.p95, "ms")
    _row("Tokens/sec", vr.throughput.tokens_per_sec, sr.throughput.tokens_per_sec, "", lower_better=False)
    _row("Requests/sec", vr.throughput.requests_per_sec, sr.throughput.requests_per_sec, "", lower_better=False)
    _row("Error rate", vr.error_rate * 100, sr.error_rate * 100, "%")

    console.print(table)


if __name__ == "__main__":
    app()
