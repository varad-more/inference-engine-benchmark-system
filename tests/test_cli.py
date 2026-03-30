from __future__ import annotations

from pathlib import Path

import pytest
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


def test_health_accepts_spec_dec_variants(monkeypatch) -> None:
    calls: list[str] = []

    def fake_make_client(engine: str, model: str, host_override: str | None = None):
        calls.append(engine)
        return FakeClient(healthy=True)

    monkeypatch.setattr(run_experiment, "_make_client", fake_make_client)

    result = runner.invoke(run_experiment.app, ["health", "--engines", "vllm-eagle3"])
    assert result.exit_code == 0
    assert calls == ["vllm-eagle3"]
    assert "Eagle3" in result.output


def test_parse_engines_accepts_spec_dec_variants() -> None:
    from run_experiment import _parse_engines

    assert _parse_engines("vllm-eagle3") == ["vllm-eagle3"]
    assert _parse_engines("vllm-ngram") == ["vllm-ngram"]
    assert _parse_engines("sglang-eagle3") == ["sglang-eagle3"]
    assert _parse_engines("sglang-ngram") == ["sglang-ngram"]


def test_parse_engines_rejects_unknown_variant() -> None:
    from run_experiment import _parse_engines

    with pytest.raises(Exception):  # typer.BadParameter
        _parse_engines("vllm-medusa")


def test_parse_engines_all_spec_alias() -> None:
    from run_experiment import _ENGINE_VARIANTS, _parse_engines

    result = _parse_engines("all-spec", allow_group_aliases=True)
    assert set(result) == set(_ENGINE_VARIANTS.keys())


def test_variant_metadata_eagle3() -> None:
    from run_experiment import _variant_metadata

    meta = _variant_metadata("vllm-eagle3")
    assert meta["engine"] == "vllm"
    assert meta["engine_variant"] == "vllm-eagle3"
    assert meta["spec_method"] == "eagle3"


def test_variant_metadata_baseline_has_no_spec_method() -> None:
    from run_experiment import _variant_metadata

    meta = _variant_metadata("vllm")
    assert meta["spec_method"] is None


def test_make_client_eagle3_creates_vllm_client() -> None:
    from engines.vllm_client import VLLMClient
    from run_experiment import _make_client

    client = _make_client("vllm-eagle3", "meta-llama/Llama-3.1-8B-Instruct")
    assert isinstance(client, VLLMClient)
    assert client.port == 8000


def test_make_client_sglang_eagle3_creates_sglang_client() -> None:
    from engines.sglang_client import SGLangClient
    from run_experiment import _make_client

    client = _make_client("sglang-eagle3", "meta-llama/Llama-3.1-8B-Instruct")
    assert isinstance(client, SGLangClient)
    assert client.port == 8001


def test_resolve_host_routes_by_base_engine() -> None:
    from run_experiment import _resolve_host

    assert _resolve_host("vllm-eagle3", "vllm-host", "sglang-host") == "vllm-host"
    assert _resolve_host("sglang-ngram", "vllm-host", "sglang-host") == "sglang-host"


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
