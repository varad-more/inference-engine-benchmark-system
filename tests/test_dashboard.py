from __future__ import annotations

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


def test_dashboard_home_renders() -> None:
    client = TestClient(dash.app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Current activity" in response.text
    assert "/api/current" in response.text
