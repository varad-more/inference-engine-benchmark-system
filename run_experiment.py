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

# ---------------------------------------------------------------------------
# Engine variant registry
# ---------------------------------------------------------------------------
# Each key is a CLI-addressable engine variant.  Speculative-decoding variants
# share the same port as their baseline (only one runs at a time on one GPU).
_ENGINE_VARIANTS: dict[str, dict] = {
    "vllm":          {"label": "vLLM",          "port": 8000, "base": "vllm",  "spec_method": None},
    "vllm-eagle3":   {"label": "vLLM+Eagle3",   "port": 8000, "base": "vllm",  "spec_method": "eagle3"},
    "vllm-ngram":    {"label": "vLLM+Ngram",    "port": 8000, "base": "vllm",  "spec_method": "ngram"},
    "sglang":        {"label": "SGLang",         "port": 8001, "base": "sglang","spec_method": None},
    "sglang-eagle3": {"label": "SGLang+Eagle3",  "port": 8001, "base": "sglang","spec_method": "eagle3"},
    "sglang-ngram":  {"label": "SGLang+Ngram",   "port": 8001, "base": "sglang","spec_method": "ngram"},
}
_BASELINE_ENGINES = ["vllm", "sglang"]


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_engines(value: str, *, allow_group_aliases: bool = False) -> list[str]:
    items = [item.lower() for item in _parse_csv(value)]
    if not items:
        raise typer.BadParameter("Specify at least one engine.")

    normalized: list[str] = []
    for item in items:
        if allow_group_aliases and item in {"all", "both"}:
            normalized.extend(_BASELINE_ENGINES)
        elif allow_group_aliases and item == "all-spec":
            normalized.extend(_ENGINE_VARIANTS.keys())
        elif item in _ENGINE_VARIANTS:
            normalized.append(item)
        else:
            valid_opts = list(_ENGINE_VARIANTS.keys())
            if allow_group_aliases:
                valid_opts.extend(["both", "all-spec"])
            raise typer.BadParameter(
                f"Unknown engine '{item}'. Valid options: {', '.join(valid_opts)}."
            )

    deduped: list[str] = []
    for engine in normalized:
        if engine not in deduped:
            deduped.append(engine)
    return deduped


def _make_client(engine: str, model: str, host_override: str | None = None):
    from engines.sglang_client import SGLangClient
    from engines.vllm_client import VLLMClient

    info = _ENGINE_VARIANTS.get(engine)
    if info is None:
        console.print(f"[red]Unknown engine variant: {engine}[/red]")
        raise typer.Exit(1)

    host = host_override or "localhost"
    port = info["port"]
    base = info["base"]
    if base == "vllm":
        return VLLMClient(host=host, port=port, model=model)
    return SGLangClient(host=host, port=port, model=model)


def _variant_metadata(engine_name: str) -> dict[str, str | None]:
    """Return run_metadata fields for an engine variant."""
    info = _ENGINE_VARIANTS[engine_name]
    return {
        "engine": info["base"],
        "engine_variant": engine_name,
        "spec_method": info["spec_method"],
    }


def _resolve_host(engine_name: str, vllm_host: str, sglang_host: str) -> str:
    """Pick the right host based on the variant's base engine."""
    base = _ENGINE_VARIANTS[engine_name]["base"]
    return vllm_host if base == "vllm" else sglang_host


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
    engine_list = _parse_engines(engines)
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
                host = _resolve_host(engine_name, vllm_host, sglang_host)
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
                            **_variant_metadata(engine_name),
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
    engines: str = typer.Option(
        "vllm,sglang",
        "--engines",
        "-e",
        help="Two comma-separated engine variants to compare (e.g. vllm,vllm-eagle3)",
    ),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    prompt_pack: str | None = typer.Option(None, "--prompt-pack"),
    vllm_host: str = typer.Option("localhost", "--vllm-host"),
    sglang_host: str = typer.Option("localhost", "--sglang-host"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
) -> None:
    """Run the same scenario on two engine variants and print a comparison table."""
    from benchmarks.runner import run_comparison

    engine_list = _parse_engines(engines)
    if len(engine_list) != 2:
        console.print("[red]Error: compare requires exactly 2 engines.[/red]")
        raise typer.Exit(1)

    results_dir = output_dir or RESULTS_DIR
    results_dir.mkdir(exist_ok=True)
    bench_scenario = _get_scenario(scenario)
    if prompt_pack:
        bench_scenario.prompt_pack = prompt_pack

    engine_a, engine_b = engine_list
    host_a = _resolve_host(engine_a, vllm_host, sglang_host)
    host_b = _resolve_host(engine_b, vllm_host, sglang_host)
    client_a = _make_client(engine_a, model, host_a)
    client_b = _make_client(engine_b, model, host_b)
    label_a = _ENGINE_VARIANTS[engine_a]["label"]
    label_b = _ENGINE_VARIANTS[engine_b]["label"]

    console.rule(f"[bold blue]Comparison: {scenario}")
    console.print(f"  Engines     : [cyan]{label_a} vs {label_b}[/cyan]")
    console.print(f"  Model       : [cyan]{model}[/cyan]")
    console.print(f"  Prompt pack : [cyan]{bench_scenario.prompt_pack}[/cyan]")

    async def _run() -> None:
        comparison = await run_comparison(
            scenario=bench_scenario,
            vllm_client=client_a,
            sglang_client=client_b,
            results_dir=results_dir,
            run_metadata={
                "model": model,
                "prompt_pack": bench_scenario.prompt_pack,
                "engine_a": engine_a,
                "engine_b": engine_b,
            },
        )
        await client_a.aclose()
        await client_b.aclose()

        _print_comparison_table(comparison, label_a=label_a, label_b=label_b)
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
    engine_list = _parse_engines(engines)

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
                    host = _resolve_host(engine_name, vllm_host, sglang_host)
                    client = _make_client(engine_name, model, host)
                    healthy = await client.health_check()
                    task_info: dict = {
                        "model": model,
                        "scenario_name": bench_scenario.name,
                        **_variant_metadata(engine_name),
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
                            **_variant_metadata(engine_name),
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
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Restrict the HTML report to a specific model when multiple models exist",
    ),
) -> None:
    """Generate an HTML report from saved results."""
    from analysis.report import generate_report

    src_dir = results_dir or RESULTS_DIR
    console.print(f"Generating report from [cyan]{src_dir}[/cyan] → [cyan]{output}[/cyan]")
    if model:
        console.print(f"  Model filter : [cyan]{model}[/cyan]")
    generate_report(results_dir=src_dir, output_path=output, model=model)
    console.print(f"[green]✓[/green] Report written to {output}")


@app.command("final-report")
def final_report(
    output: Path = typer.Option(Path("final_report.md"), "--output", "-o"),
    results_dir: Path | None = typer.Option(None, "--results-dir"),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Restrict the markdown summary to a specific model when multiple models exist",
    ),
) -> None:
    """Generate an aggregated markdown summary from saved result files."""
    from analysis.final_report import generate_final_report

    src_dir = results_dir or RESULTS_DIR
    console.print(f"Generating final report from [cyan]{src_dir}[/cyan] → [cyan]{output}[/cyan]")
    if model:
        console.print(f"  Model filter : [cyan]{model}[/cyan]")
    summary = generate_final_report(results_dir=src_dir, output_path=output, model=model)
    console.print(
        f"[green]✓[/green] Final report written to {output} from {summary['total_result_files']} result files"
    )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(3000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
    results_dir: Path | None = typer.Option(
        None,
        "--results-dir",
        help="Directory of benchmark JSON files to expose in the dashboard",
    ),
) -> None:
    """Start the benchmark dashboard server."""
    if results_dir is not None:
        os.environ["RESULTS_DIR"] = str(results_dir)
    console.print(f"[bold]Starting dashboard on http://{host}:{port}[/bold]")
    if results_dir is not None:
        console.print(f"  Results dir  : [cyan]{results_dir}[/cyan]")
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
    engines: str = typer.Option(
        "both",
        "--engines",
        "-e",
        help="Comma-separated engines to check: vllm, sglang, or both",
    ),
    vllm_host: str = typer.Option("localhost", "--vllm-host"),
    sglang_host: str = typer.Option("localhost", "--sglang-host"),
    model: str = typer.Option(DEFAULT_MODEL, "--model"),
) -> None:
    """Check health of one or more inference engines."""
    engine_list = _parse_engines(engines, allow_group_aliases=True)

    async def _check() -> None:
        statuses: dict[str, bool] = {}

        for engine_name in engine_list:
            host = _resolve_host(engine_name, vllm_host, sglang_host)
            client = _make_client(engine_name, model, host)
            try:
                statuses[engine_name] = await client.health_check()
            finally:
                await client.aclose()

        table = Table(title="Engine Health")
        table.add_column("Engine")
        table.add_column("Variant")
        table.add_column("Status")
        table.add_column("URL")
        for engine_name in engine_list:
            info = _ENGINE_VARIANTS[engine_name]
            host = _resolve_host(engine_name, vllm_host, sglang_host)
            url = f"http://{host}:{info['port']}"
            spec = info["spec_method"] or "baseline"
            if engine_name in statuses:
                status = (
                    "[green]✓ healthy[/green]"
                    if statuses[engine_name]
                    else "[red]✗ unreachable[/red]"
                )
            else:
                status = "[dim]- skipped[/dim]"
            table.add_row(info["label"], spec, status, url)
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


def _print_comparison_table(comparison, label_a: str = "vLLM", label_b: str = "SGLang") -> None:
    vr = comparison.vllm_results.metrics
    sr = comparison.sglang_results.metrics

    table = Table(title=f"Comparison — {comparison.scenario_name}", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column(label_a, justify="right")
    table.add_column(label_b, justify="right")
    table.add_column(f"Δ ({label_a}−{label_b})", justify="right")

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
