# Getting Started — Inference Engine Benchmark System

A practical guide covering environment setup, running your first benchmark, extending to speculative decoding, and interpreting results. Written for a single A10G 24GB GPU (AWS `g5.2xlarge`), but works on any GPU with ≥16GB VRAM.

---

## Prerequisites

- Docker + Docker Compose v2
- Python 3.11+ with conda (or venv)
- NVIDIA GPU with ≥16GB VRAM and the NVIDIA Container Toolkit installed
- HuggingFace account (free) for gated models — Qwen3-8B does **not** need a token

Verify your GPU is reachable by Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

---

## 1. Install

```bash
git clone <repo-url>
cd inference-engine-benchmark-system

# Install Python dependencies
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Edit .env — add HUGGING_FACE_HUB_TOKEN for gated models (Llama, Gemma)

# Create model cache directory (weights download here)
mkdir -p model-cache
```

---

## 2. Choose a Model

| Model | HF ID | VRAM | Token needed | Best for |
|---|---|---|---|---|
| **Qwen3-8B** (default) | `Qwen/Qwen3-8B` | ~16GB | No | General benchmarking, spec-dec |
| Gemma 3 4B | `google/gemma-3-4b-it` | ~8GB | Yes | Lightweight / fast iteration |
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` | ~16GB | Yes | Eagle3 spec-dec (best draft support) |
| DeepSeek-R1 Distill 7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | ~14GB | No | Reasoning model latency profile |
| Llama 3.2 3B | `meta-llama/Llama-3.2-3B-Instruct` | ~6GB | Yes | Throughput ceiling / concurrency tests |

Set your target model:

```bash
export MODEL=Qwen/Qwen3-8B   # change to any model above
```

---

## 3. Start an Inference Engine

Run **one engine at a time** on a single GPU — both engines share the same GPU and will contend for VRAM if started simultaneously.

```bash
# vLLM (port 8000)
docker compose --profile vllm up -d vllm
sleep 120   # wait for model to load into GPU memory

# Verify it's ready
curl http://localhost:8000/health
```

Or for SGLang:

```bash
# SGLang (port 8001)
docker compose --profile sglang up -d sglang
sleep 120

curl http://localhost:8001/health
```

Use the CLI health check to see formatted status:

```bash
python run_experiment.py health --engines vllm
python run_experiment.py health --engines sglang
python run_experiment.py health --engines both
```

---

## 4. Run Your First Benchmark

```bash
# Single-request latency — measures TTFT and end-to-end latency
python run_experiment.py run \
  --scenario single_request_latency \
  --engines vllm \
  --model $MODEL

# Throughput ramp — sweeps concurrency levels (1 → 32) to find the knee
python run_experiment.py run \
  --scenario throughput_ramp \
  --engines vllm \
  --model $MODEL
```

Results are saved to `results/` as JSON files named `{scenario}_{engine}_{timestamp}.json`.

---

## 5. Run All Scenarios (Matrix Mode)

The `matrix` command runs every scenario × engine combination in one shot, with configurable iterations and cooldown:

```bash
python run_experiment.py matrix \
  --scenarios single_request_latency,throughput_ramp,long_context_stress,prefix_sharing_benefit,structured_generation_speed \
  --engines vllm \
  --model $MODEL \
  --iterations 2 \
  --cooldown-seconds 120
```

Switch engines and repeat:

```bash
docker compose --profile vllm down
sleep 60

docker compose --profile sglang up -d sglang && sleep 120
python run_experiment.py matrix \
  --scenarios single_request_latency,throughput_ramp,long_context_stress,prefix_sharing_benefit,structured_generation_speed \
  --engines sglang \
  --model $MODEL \
  --iterations 2 \
  --cooldown-seconds 120
docker compose --profile sglang down
```

---

## 6. View Available Scenarios and Prompts

```bash
python run_experiment.py list-scenarios
python run_experiment.py list-prompt-packs
```

| Scenario | What it measures |
|---|---|
| `single_request_latency` | TTFT and e2e latency at 1-4 concurrent requests |
| `throughput_ramp` | Tokens/sec and req/sec across concurrency sweep |
| `long_context_stress` | Performance with 6k–8k token prompts |
| `prefix_sharing_benefit` | Cache hit rate benefit from shared prompt prefixes |
| `structured_generation_speed` | JSON-constrained decoding overhead |

---

## 7. Speculative Decoding Benchmarks

Speculative decoding is configured at engine startup — not a separate scenario. See [`SPECULATIVE_DECODING.md`](SPECULATIVE_DECODING.md) for a full runbook.

**Quick start (Eagle3 on vLLM, requires Llama 3.1 8B):**

```bash
export MODEL=meta-llama/Llama-3.1-8B-Instruct

# 1. Baseline
docker compose --profile vllm up -d vllm && sleep 120
python run_experiment.py run -s single_request_latency -e vllm --model $MODEL
docker compose --profile vllm down

# 2. Eagle3 (loads two models — wait longer)
docker compose --profile vllm-eagle3 up -d vllm-eagle3 && sleep 180
python run_experiment.py run -s single_request_latency -e vllm-eagle3 --model $MODEL
docker compose --profile vllm-eagle3 down

# 3. Ngram (no draft model needed)
docker compose --profile vllm-ngram up -d vllm-ngram && sleep 120
python run_experiment.py run -s single_request_latency -e vllm-ngram --model $MODEL
docker compose --profile vllm-ngram down
```

Results from all three runs feed into the same report — the engine variant is tracked in the result filename and metadata.

**Available engine variants:**

| Variant | Description |
|---|---|
| `vllm` | vLLM baseline |
| `vllm-eagle3` | vLLM + Eagle3 speculative decoding |
| `vllm-ngram` | vLLM + Ngram speculative decoding |
| `sglang` | SGLang baseline |
| `sglang-eagle3` | SGLang + Eagle3 speculative decoding |
| `sglang-ngram` | SGLang + Ngram speculative decoding |

---

## 8. Generate Reports

```bash
# Aggregated markdown summary (all result files in results/)
python run_experiment.py final-report --output summary.md

# Filter to a specific model
python run_experiment.py final-report --model $MODEL --output summary.md

# HTML report with charts
python run_experiment.py report --output report.html
```

### Dashboard (live view)

```bash
# Start the dashboard (reads from results/ directory)
python run_experiment.py serve --results-dir results/

# Or via docker compose (runs on port 3000)
docker compose --profile dashboard up -d dashboard
# Open http://localhost:3000
```

---

## 9. Direct Engine Inference (curl)

Test the engines directly without the benchmark harness:

```bash
# vLLM
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [{"role": "user", "content": "What is speculative decoding?"}],
    "max_tokens": 256
  }'

# SGLang
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [{"role": "user", "content": "What is speculative decoding?"}],
    "max_tokens": 256
  }'
```

Check available models on a running engine:

```bash
curl http://localhost:8000/v1/models | python -m json.tool
```

---

## 10. Side-by-Side Comparison

Compare two engine variants directly on the same scenario:

```bash
# Classic baseline comparison (requires both engines running simultaneously — not recommended on single GPU)
python run_experiment.py compare \
  --scenario single_request_latency \
  --engines vllm,sglang

# Compare two variants from saved results (run sequentially, then compare)
python run_experiment.py compare \
  --scenario single_request_latency \
  --engines vllm,vllm-eagle3
```

---

## 11. CI / Automated Runs

Run the test suite to verify the harness is healthy before a benchmark session:

```bash
python -m pytest tests/ -v
```

Key test files:
- `tests/test_cli.py` — CLI commands and engine variant parsing
- `tests/test_result_metadata.py` — Result filename and metadata correctness
- `tests/test_scenarios.py` — Scenario definitions
- `tests/test_metrics.py` — Metrics calculation

---

## Common Issues

| Symptom | Fix |
|---|---|
| `docker: Error response from daemon: could not select device driver "nvidia"` | Install NVIDIA Container Toolkit: `nvidia-ctk runtime configure --runtime=docker` |
| Engine health check returns 503 for >2 min | Model still loading — `docker logs vllm-server -f` to watch progress |
| OOM on Eagle3 startup | Reduce `gpu-memory-utilization` to `0.75` in `.env` or reduce `MAX_MODEL_LEN` |
| `HfHubHTTPError: 401` | Add `HUGGING_FACE_HUB_TOKEN` to `.env` and accept model license on HuggingFace |
| `unsupported head_dim` on SGLang | Known limitation for some models (e.g. Phi-3 mini) — use vLLM only for that model |
| Results not showing in report | Check `results/*.json` exist and contain `scenario_name`, `engine_name`, `metrics` keys |

---

## Next Steps

- **Speculative decoding runbook**: [`docs/SPECULATIVE_DECODING.md`](SPECULATIVE_DECODING.md)
- **Single-GPU operation guide**: [`docs/SINGLE_GPU_OPERATION.md`](SINGLE_GPU_OPERATION.md)
- **Known limitations**: [`docs/KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md)
- **Validated benchmark results**: [`docs/VALIDATED_BENCHMARK_RUNBOOK.md`](VALIDATED_BENCHMARK_RUNBOOK.md)
