from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import analysis.final_report
import run_experiment

runner = CliRunner()


class FakeClient:
    def __init__(self, healthy: bool) -> None:
        self.healthy = healthy
        self.closed = False

    async def health_check(self) -> bool:
        return self.healthy

    async def aclose(self) -> None:
        self.closed = True


def test_health_checks_selected_engine_and_skips_others(monkeypatch) -> None:
    calls: list[tuple[str, str, str | None]] = []

    def fake_make_client(engine: str, model: str, host_override: str | None = None):
        calls.append((engine, model, host_override))
        return FakeClient(healthy=True)

    monkeypatch.setattr(run_experiment, "_make_client", fake_make_client)

    result = runner.invoke(run_experiment.app, ["health", "--engines", "sglang"])

    assert result.exit_code == 0
    assert calls == [("sglang", run_experiment.DEFAULT_MODEL, "localhost")]
    assert "SGLang" in result.output
    assert "skipped" in result.output.lower()
    assert "healthy" in result.output.lower()


def test_health_supports_both_alias(monkeypatch) -> None:
    calls: list[str] = []

    def fake_make_client(engine: str, model: str, host_override: str | None = None):
        calls.append(engine)
        return FakeClient(healthy=engine == "vllm")

    monkeypatch.setattr(run_experiment, "_make_client", fake_make_client)

    result = runner.invoke(run_experiment.app, ["health", "--engines", "both"])

    assert result.exit_code == 0
    assert calls == ["vllm", "sglang"]
    assert "vLLM" in result.output
    assert "SGLang" in result.output
    assert "healthy" in result.output.lower()
    assert "unreachable" in result.output.lower()


def test_health_rejects_unknown_engine() -> None:
    result = runner.invoke(run_experiment.app, ["health", "--engines", "llamacpp"])

    assert result.exit_code != 0
    assert "Unknown engine 'llamacpp'" in result.output


def test_final_report_passes_model_filter(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_generate_final_report(results_dir: Path, output_path: Path, model: str | None = None):
        calls["results_dir"] = results_dir
        calls["output_path"] = output_path
        calls["model"] = model
        output_path.write_text("# stub")
        return {"total_result_files": 2}

    monkeypatch.setattr(analysis.final_report, "generate_final_report", fake_generate_final_report)

    result = runner.invoke(
        run_experiment.app,
        [
            "final-report",
            "--results-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "summary.md"),
            "--model",
            "google/gemma-2-2b-it",
        ],
    )

    assert result.exit_code == 0
    assert calls == {
        "results_dir": tmp_path,
        "output_path": tmp_path / "summary.md",
        "model": "google/gemma-2-2b-it",
    }
    assert "Model filter" in result.output


def test_serve_accepts_results_dir(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_uvicorn_run(app: str, host: str, port: int, reload: bool, log_level: str) -> None:
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port
        captured["reload"] = reload
        captured["log_level"] = log_level
        captured["results_dir_env"] = run_experiment.os.environ.get("RESULTS_DIR")

    monkeypatch.delenv("RESULTS_DIR", raising=False)
    monkeypatch.setattr(run_experiment.uvicorn, "run", fake_uvicorn_run)

    result = runner.invoke(
        run_experiment.app,
        ["serve", "--host", "0.0.0.0", "--port", "3010", "--results-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert captured == {
        "app": "dashboard.app:app",
        "host": "0.0.0.0",
        "port": 3010,
        "reload": False,
        "log_level": "info",
        "results_dir_env": str(tmp_path),
    }
    assert "Results dir" in result.output
