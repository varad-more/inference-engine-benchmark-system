from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import dashboard.app as dash


def test_parse_active_benchmark_command() -> None:
    parsed = dash._parse_active_benchmark_command(
        "1234 python run_experiment.py run --scenario throughput_ramp --engines vllm --model google/gemma-2-2b-it --prompt-pack long_generation"
    )
    assert parsed is not None
    assert parsed["state"] == "running"
    assert parsed["scenario"] == "throughput_ramp"
    assert parsed["engine"] == "vllm"
    assert parsed["model"] == "google/gemma-2-2b-it"
    assert parsed["prompt_pack"] == "long_generation"


def test_parse_helper_script_command() -> None:
    parsed = dash._parse_helper_script_command(
        "4321 /bin/bash /home/ubuntu/run_mistral7b_sglang_single.sh"
    )
    assert parsed is not None
    assert parsed["state"] == "queued_or_cooldown"
    assert parsed["engine"] == "sglang"
    assert parsed["scenario"] == "single_request_latency"
    assert parsed["model"] == "Mistral 7B"


def test_current_activity_endpoint(monkeypatch) -> None:
    dash._jobs.clear()

    def fake_run_shell(command: str, timeout_sec: int = 8) -> str:
        if "pgrep -af 'run_experiment.py run'" in command:
            return "1234 python run_experiment.py run --scenario single_request_latency --engines sglang --model Qwen/Qwen2.5-7B-Instruct"
        if "pgrep -af 'vllm serve|sglang.launch_server|run_experiment.py serve'" in command:
            return "5678 python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct"
        return ""

    monkeypatch.setattr(dash, "_run_shell", fake_run_shell)
    monkeypatch.setattr(dash, "_latest_results_payload", lambda limit=8: [])

    client = TestClient(dash.app)
    response = client.get("/api/current")
    payload = response.json()

    assert response.status_code == 200
    assert payload["current"]["state"] == "running"
    assert payload["current"]["scenario"] == "single_request_latency"
    assert payload["active_servers"][0]["engine"] == "sglang"


def test_result_file_payload(tmp_path: Path) -> None:
    path = tmp_path / "single_request_latency_VLLMClient_123.json"
    path.write_text(
        json.dumps(
            {
                "scenario_name": "single_request_latency",
                "engine_name": "VLLMClient",
                "metrics": {
                    "ttft": {"p95": 12.3},
                    "latency": {"p95": 456.7},
                    "throughput": {"tokens_per_sec": 999.0, "requests_per_sec": 7.5},
                    "error_rate": 0.0,
                },
                "run_metadata": {"model": "google/gemma-2-2b-it"},
                "workload_metadata": {"prompt_pack": "short_chat"},
            }
        )
    )
    payload = dash._result_file_payload(path)
    assert payload["type"] == "result"
    assert payload["engine"] == "vLLM"
    assert payload["model"] == "google/gemma-2-2b-it"
    assert payload["ttft_p95_ms"] == 12.3


def test_dashboard_home_renders() -> None:
    client = TestClient(dash.app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Current activity" in response.text
    assert "Latest completed result" in response.text
    assert "/api/current" in response.text
