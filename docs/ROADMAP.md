# Roadmap — Next Implementation Steps

Current state: A10G 24GB single-GPU benchmark harness, 7 baseline models, speculative decoding (Eagle3 + Ngram) infrastructure complete, 89 tests passing.

This document describes the next concrete phases in priority order, with enough detail to implement each one without further research.

Full model catalog with VRAM requirements: [`docs/MODEL_CATALOG.md`](MODEL_CATALOG.md)

---

## Phase A — 2025/2026 Model Benchmarks (Immediate)

**Goal:** Run the existing 5-scenario matrix against the new model set and publish updated results.

### A.1 — Baseline matrix (A10G 24GB, no spec-dec, no quantization)

All models below fit unquantized on a single A10G 24GB. Run sequentially, one engine at a time.

| Model | HF ID | VRAM | Token | Priority | What it adds |
|---|---|---|---|---|---|
| **Qwen3-8B** | `Qwen/Qwen3-8B` | ~16GB | No | 1 — default | Hybrid thinking mode; Eagle3 support |
| **DeepSeek-R1 Distill 7B** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | ~14GB | No | 2 | Reasoning profile; very different latency shape |
| **SmolLM3 3B** | `HuggingFaceTB/SmolLM3-3B` | ~6GB | No | 3 | Throughput ceiling; beats Llama 3.2 3B |
| **Phi-4-mini** | `microsoft/Phi-4-mini-instruct` | ~7GB | No | 4 | Best small-model quality in 2025 |
| **Qwen3-30B-A3B** *(MoE)* | `Qwen/Qwen3-30B-A3B` | ~17GB | No | 5 | 30B total weights, 3B active — unique TTFT/quality tradeoff |
| **Granite 3.3 8B** | `ibm-granite/granite-3.3-8b-instruct` | ~16GB | No | 6 | Apache 2.0; 128K context; enterprise/coding |
| **Gemma 3 4B** | `google/gemma-3-4b-it` | ~8GB | Yes | 7 | Multimodal; Google 2025 flagship small model |
| **Gemma 3 12B** | `google/gemma-3-12b-it` | ~24GB | Yes | 8 | Tight A10G fit; step up in quality from 4B |
| **DeepSeek-R1 Distill 8B** | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | ~16GB | No | 9 | Llama-based distill; compare vs Qwen-based R1 |
| **Llama 3.1 8B** | `meta-llama/Llama-3.1-8B-Instruct` | ~16GB | Yes | 10 | Best Eagle3 draft support for spec-dec runs |

```bash
MODEL=Qwen/Qwen3-8B

# vLLM
docker compose --profile vllm up -d && sleep 120
python run_experiment.py matrix \
  --scenarios single_request_latency,throughput_ramp,long_context_stress,prefix_sharing_benefit,structured_generation_speed \
  --engines vllm --model $MODEL --iterations 2 --cooldown-seconds 120
docker compose --profile vllm down && sleep 60

# SGLang
docker compose --profile sglang up -d && sleep 120
python run_experiment.py matrix \
  --scenarios single_request_latency,throughput_ramp,long_context_stress,prefix_sharing_benefit,structured_generation_speed \
  --engines sglang --model $MODEL --iterations 2 --cooldown-seconds 120
docker compose --profile sglang down
```

Repeat for each model. Generate a combined report when all models are done:

```bash
python run_experiment.py final-report --output reports/2026_baseline_summary.md
python run_experiment.py report --output reports/2026_baseline_report.html
```

### A.2 — Speculative decoding matrix (Llama 3.1 8B + Qwen3-8B)

Run the spec-dec variants for models with Eagle3 draft support:

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct
SCENARIOS=single_request_latency,throughput_ramp

for PROFILE in vllm vllm-eagle3 vllm-ngram sglang sglang-eagle3 sglang-ngram; do
  docker compose --profile $PROFILE up -d
  sleep $([ "$PROFILE" = *eagle3* ] && echo 180 || echo 120)
  python run_experiment.py matrix \
    --scenarios $SCENARIOS --engines $PROFILE \
    --model $MODEL --iterations 2 --cooldown-seconds 60
  docker compose --profile $PROFILE down && sleep 60
done
```

### A.3 — Reasoning model scenario (DeepSeek-R1)

DeepSeek-R1 Distill generates long chain-of-thought output before the final answer — this completely changes the latency and throughput profile compared to chat models. Add a dedicated reasoning scenario:

**File: `benchmarks/scenarios.py`** — add `ReasoningLatency` scenario:
- 20 requests, concurrency 1–4
- Prompt pack: `reasoning.jsonl`
- Track: TTFT, time-to-last-token, output token count (for thinking vs answer ratio)
- `max_tokens: 2048` (reasoning chains need headroom)

This scenario runs on all models but is most informative for R1 distill variants.

---

## Phase B — Quantization Benchmarks

**Goal:** Benchmark AWQ and GPTQ quantized models to measure accuracy/speed tradeoffs on A10G.

### B.1 — Why quantization matters on A10G

The A10G has 24GB VRAM. At fp16/bf16:
- 8B models: ~16GB — fits, but leaves no room for Eagle3 draft
- 14B models: ~28GB — **does not fit** without quantization
- 32B models: ~64GB — requires multi-GPU or quantization

AWQ int4 roughly halves VRAM usage with <5% quality degradation on most tasks.

### B.2 — Models to quantize

| Target | Quantized HF ID | VRAM (int4) | Fits A10G |
|---|---|---|---|
| Qwen3-8B | `Qwen/Qwen3-8B-AWQ` *(if available)* | ~8GB | Yes |
| Llama 3.1 8B | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | ~8GB | Yes |
| Qwen3-14B | `Qwen/Qwen3-14B-AWQ` *(if available)* | ~14GB | Yes |
| Mistral Small 24B | `bartowski/Mistral-Small-3.1-22B-Instruct-2503-exl2` | ~12GB | Yes (exl2) |

### B.3 — Engine support

- **vLLM**: pass `--quantization awq` or `--quantization gptq` to the server command
- **SGLang**: pass `--quantization awq` to `sglang.launch_server`

### B.4 — Implementation

1. Add quantization variants to `_ENGINE_VARIANTS` in `run_experiment.py`:
   ```python
   "vllm-awq":  {"label": "vLLM+AWQ",  "port": 8000, "base": "vllm",  "spec_method": None, "quantization": "awq"},
   "sglang-awq": {"label": "SGLang+AWQ", "port": 8001, "base": "sglang", "spec_method": None, "quantization": "awq"},
   ```

2. Add `vllm-awq` and `sglang-awq` profiles to `docker-compose.yml` with `--quantization awq` flag.

3. Add `quantization` field to `run_metadata` so reports can separate fp16 vs int4 results.

4. New benchmark question: "Does AWQ int4 on a 14B model beat fp16 on an 8B model for throughput? For TTFT?"

---

## Phase C — Multi-GPU Inferencing

**Goal:** Enable benchmarking on A100/H100 multi-GPU instances for large models (32B–70B).

### C.1 — What changes with multiple GPUs

Both vLLM and SGLang support tensor parallelism — the model weights are split across GPUs. This is controlled by a single flag:
- **vLLM**: `--tensor-parallel-size N`
- **SGLang**: `--tp-size N`

No code changes are needed in the benchmark harness itself — engines still expose the same OpenAI-compatible HTTP API. The only changes are in `docker-compose.yml` and the deployment scripts.

### C.2 — Docker Compose changes

Add multi-GPU profiles to `docker-compose.yml`:

```yaml
vllm-multi:
  profiles: ["vllm-multi"]
  image: ${VLLM_IMAGE:-vllm/vllm-openai:v0.18.0-cu130}
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all          # expose all GPUs
            capabilities: [gpu]
  command:
    - "--model"
    - "${MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
    - "--tensor-parallel-size"
    - "${TP_SIZE:-2}"
    - "--gpu-memory-utilization"
    - "0.90"
    ...

sglang-multi:
  profiles: ["sglang-multi"]
  # same pattern with --tp-size ${TP_SIZE:-2}
```

### C.3 — New engine variants

```python
"vllm-tp2":   {"label": "vLLM TP=2",   "port": 8000, "base": "vllm",  "spec_method": None},
"sglang-tp2": {"label": "SGLang TP=2",  "port": 8001, "base": "sglang", "spec_method": None},
```

### C.4 — Target models for multi-GPU runs

| Model | TP size | GPU config | Est. VRAM/GPU |
|---|---|---|---|
| `Qwen/Qwen3-32B` | 2 | 2× A100 40GB | ~32GB |
| `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | 2 | 2× A100 40GB | ~24GB |
| `meta-llama/Llama-3.3-70B-Instruct` | 2–4 | 2–4× A100 80GB | ~35–70GB |

### C.5 — Deployment script update

Add `--tp-size` flag to `deploy/ec2_deploy.sh` to support `p4d.24xlarge` (8× A100 40GB) and `p3.16xlarge` (8× V100) instance types alongside the existing `g5.2xlarge`.

```bash
./deploy/ec2_deploy.sh \
  --mode single \
  --instance p4d.24xlarge \
  --key my-key \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tp-size 4
```

### C.6 — New benchmark question

At TP=2 vs TP=4 on a 70B model: throughput scales sub-linearly (communication overhead). Measuring where the throughput-per-GPU-dollar peaks is genuinely useful.

---

## Phase D — Continuous Benchmarking (CI/CD Integration)

**Goal:** Run a lightweight benchmark suite automatically on every model/engine update.

### D.1 — Fast CI profile

Add a `ci` scenario that runs in < 5 minutes on any GPU:
- 10 requests, concurrency 1
- Checks: TTFT < 2× baseline, success_rate == 1.0, no regressions > 20%

```bash
python run_experiment.py run --scenario ci_smoke --engines vllm --strict
```

### D.2 — GitHub Actions workflow

```yaml
# .github/workflows/benchmark-ci.yml
on:
  push:
    paths: ["engines/**", "benchmarks/**", "docker-compose.yml"]
  schedule:
    - cron: "0 2 * * 1"   # weekly, Monday 2am

jobs:
  benchmark:
    runs-on: [self-hosted, gpu]   # requires a self-hosted runner with GPU
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[dev]"
      - run: docker compose --profile vllm up -d && sleep 120
      - run: python run_experiment.py run --scenario ci_smoke --engines vllm --strict
      - run: python run_experiment.py final-report --output ci_report.md
      - uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: |
            results/
            ci_report.md
```

### D.3 — Regression detection

Add a `--baseline-dir` flag to `final-report` that compares current results to a saved baseline and fails if any metric regresses by more than a configurable threshold:

```bash
python run_experiment.py final-report \
  --baseline-dir results/baseline_2026_04 \
  --regression-threshold 0.15 \
  --output regression_report.md
```

---

## Phase E — Production Inference API

**Goal:** Expose a stable, load-balanced inference API on top of the benchmarked engines.

### E.1 — What this means

The benchmark harness currently talks directly to vLLM/SGLang. A production layer adds:
- A stable API endpoint that abstracts the underlying engine
- Request routing (send to whichever engine is healthy)
- Rate limiting and auth
- Observability (latency histograms, error rates)

### E.2 — Minimal implementation (no new frameworks)

The dashboard already has a FastAPI app (`dashboard/app.py`). Extend it with an inference proxy endpoint:

```python
# dashboard/app.py — new endpoint
@app.post("/v1/chat/completions")
async def proxy_inference(request: Request, body: dict):
    """Route to the healthy engine, fall back to the other on error."""
    engine = pick_healthy_engine()   # checks last health poll result
    target = f"http://{engine_host(engine)}/v1/chat/completions"
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(target, json=body)
        return resp.json()
```

This gives you a single endpoint at `:3000/v1/chat/completions` that automatically routes to whichever engine is running — useful when switching engines mid-benchmark-session.

### E.3 — Proper production stack (if needed)

For real production traffic, run a gateway in front of vLLM/SGLang:

| Option | Best for | Notes |
|---|---|---|
| **litellm proxy** | Multi-model routing, OpenAI-compat drop-in | `pip install litellm[proxy]`; config-file driven |
| **nginx upstream** | Simple load balancing, no Python dep | Round-robin across engine replicas |
| **Kubernetes + Gateway API** | Multi-replica, autoscaling | Requires k8s; vLLM has a Helm chart |

A `docker-compose.yml` profile for litellm proxy is the lowest-effort path:

```yaml
litellm:
  profiles: ["litellm"]
  image: ghcr.io/berriai/litellm:main-latest
  ports:
    - "4000:4000"
  volumes:
    - ./deploy/litellm_config.yaml:/app/config.yaml
  command: ["--config", "/app/config.yaml", "--port", "4000"]
```

### E.4 — Observability

Both vLLM and SGLang expose Prometheus metrics. Add a Prometheus + Grafana stack to `docker-compose.yml`:

```yaml
prometheus:
  profiles: ["observability"]
  image: prom/prometheus:latest
  volumes:
    - ./deploy/prometheus.yml:/etc/prometheus/prometheus.yml
  ports: ["9090:9090"]

grafana:
  profiles: ["observability"]
  image: grafana/grafana:latest
  ports: ["3001:3000"]
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=admin
```

Scrape targets: `vllm:8000/metrics` (Prometheus native), `sglang:8001/get_server_info` (needs a sidecar exporter).

---

## Phase F — Reasoning Model Benchmarks

**Goal:** Benchmark models with chain-of-thought reasoning (DeepSeek-R1, Qwen3 thinking mode).

### F.1 — Why reasoning models are different

Standard chat models: prompt → short output (50–200 tokens)
Reasoning models: prompt → `<think>...</think>` (500–3000 tokens) → final answer (50–200 tokens)

This fundamentally changes the benchmark profile:
- TTFT is similar to chat models (first token of the thinking chain)
- End-to-end latency is 5–20× higher
- Throughput in tokens/sec may be similar but requests/sec collapses
- Memory pressure is higher (long KV cache per request)

### F.2 — Qwen3-8B thinking mode toggle

Qwen3-8B supports both thinking-on and thinking-off via a system prompt:

```python
# Thinking on (default)
{"role": "system", "content": "/think"}

# Thinking off (faster, less accurate on complex tasks)
{"role": "system", "content": "/no_think"}
```

Add a `--thinking` flag to the `run` command that injects the appropriate system prompt. This lets you benchmark the latency cost of reasoning on identical hardware.

### F.3 — New prompt pack: `reasoning_hard.jsonl`

Current `reasoning.jsonl` is for multi-step prompts. Add a harder pack with:
- Math problems requiring multi-step arithmetic
- Code debugging tasks
- Logic puzzles

These elicit longer thinking chains from R1/Qwen3 and produce more meaningful latency measurements.

---

## Implementation Priority

| Phase | Impact | Effort | Do first |
|---|---|---|---|
| **A** — Run new model benchmarks | High — new results to publish | Low — run existing harness | **Yes, now** |
| **B** — Quantization | High — unlocks 14B+ on A10G | Medium — compose + variant changes | After A |
| **C** — Multi-GPU | High — unlocks 70B models | Medium — compose + deploy script | After A, parallel with B |
| **D** — CI benchmarks | Medium — prevents regressions | Medium — needs self-hosted runner | After B/C |
| **E** — Production inference API | High if serving real traffic | Low (proxy) to High (k8s) | Start with litellm proxy |
| **F** — Reasoning benchmarks | Medium — interesting story | Low — new scenario + prompt pack | After A |

---

## Hardware Reference

### Instance types and what they unlock

| Instance | GPU | VRAM | TP max | Cost/hr (on-demand) | Spot saving | Best for |
|---|---|---|---|---|---|---|
| `g5.2xlarge` | 1× A10G 24GB | 24GB | 1 | ~$1.21 | ~70% | Current baseline; all ≤12B models |
| `g5.12xlarge` | 4× A10G 24GB | 96GB | 4 | ~$5.67 | ~60% | Qwen3-30B MoE, spec-dec at scale |
| `g6.2xlarge` | 1× L4 24GB | 24GB | 1 | ~$0.98 | ~70% | Cheaper A10G alternative; same VRAM |
| `g6e.2xlarge` | 1× L40S 48GB | 48GB | 1 | ~$2.20 | ~65% | 24B models unquantized, Llama 4 Scout |
| `p3.2xlarge` | 1× V100 16GB | 16GB | 1 | ~$3.06 | ~70% | Budget A100 alternative; older CUDA |
| `p4d.24xlarge` | 8× A100 40GB | 320GB | 8 | ~$32.77 | ~50% | Llama 3.3 70B, Qwen3-32B, full sweep |
| `p4de.24xlarge` | 8× A100 80GB | 640GB | 8 | ~$40.96 | ~50% | 70B fp16 + Eagle3; Qwen3-235B-A22B |
| `p5.48xlarge` | 8× H100 80GB | 640GB | 8 | ~$98.32 | ~40% | Maximum throughput; Llama 4 Maverick |

> Use spot instances for all non-time-critical benchmark runs. For 8–24hr benchmark sessions, spot is 40–70% cheaper and rarely interrupted if you bid the on-demand price.

### Model-to-GPU mapping

| GPU tier | Models (no quantization) | Models (with AWQ/INT4) |
|---|---|---|
| **1× A10G 24GB** | Qwen3-8B, Llama 3.1 8B, DeepSeek-R1-7B, Phi-4-mini, SmolLM3-3B, Granite-3.3-8B, Gemma 3 4B, Llama 3.2 3B | Phi-4 14B, DeepSeek-R1-14B, Qwen3-14B |
| **1× A10G 24GB (MoE)** | Qwen3-30B-A3B (~17GB total weights) | — |
| **1× L40S 48GB** | All A10G models + Gemma 3 12B, Mistral Small 24B (tight) | Gemma 3 27B, Qwen3-32B |
| **1× A100 80GB** | Gemma 3 27B, Mistral Small 24B, Qwen3-32B, DeepSeek-R1-32B, Llama 4 Scout (MoE) | DeepSeek-R1-70B |
| **2× A100 80GB** | Llama 3.3 70B (TP=2), Llama 4 Scout (TP=2 headroom) | Qwen3-235B-A22B (MoE) |
| **4× A100 80GB** | Llama 4 Maverick 400B (MoE, Q8) | — |
| **8× H100 80GB** | Llama 4 Maverick 400B (bf16), DeepSeek-V3 (MoE, Q4) | — |

### Recommended procurement order

1. **Start here (already have):** `g5.2xlarge` — run Phase A benchmarks now, zero cost increase
2. **Next:** `g6e.2xlarge` (1× L40S 48GB) — unlocks Gemma 3 27B and Mistral Small 24B unquantized for ~$2.20/hr
3. **Then:** `p4d.24xlarge` spot — full 70B model sweep for ~$16/hr spot; schedule overnight

---

## Files to Create/Modify Per Phase

| Phase | New files | Modified files |
|---|---|---|
| A | `reports/2026_baseline_*` | None (run existing harness) |
| B | `deploy/litellm_config.yaml` (optional) | `run_experiment.py`, `docker-compose.yml` |
| C | — | `docker-compose.yml`, `deploy/ec2_deploy.sh`, `run_experiment.py` |
| D | `.github/workflows/benchmark-ci.yml` | `benchmarks/scenarios.py`, `run_experiment.py` |
| E | `deploy/litellm_config.yaml`, `deploy/prometheus.yml` | `docker-compose.yml`, `dashboard/app.py` |
| F | `prompts/reasoning_hard.jsonl` | `benchmarks/scenarios.py`, `run_experiment.py` |
