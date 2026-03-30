# Speculative Decoding Benchmarks

Speculative decoding reduces time-to-first-token (TTFT) and end-to-end latency by letting a small "draft" model predict several tokens ahead, which the main model then verifies in a single forward pass. When the draft is correct, you get multiple tokens for the cost of one verification step.

This guide covers how to run speculative decoding benchmarks using the two supported methods — **Eagle3** and **Ngram** — on both vLLM and SGLang.

---

## Supported Methods

| Method | How it works | Draft model needed | VRAM overhead | Best for |
|---|---|---|---|---|
| **Eagle3** | Trained speculative head predicts next tokens | Yes (~1-2 GB) | ~3-5 GB extra | Greedy/low-temp generation, chat |
| **Ngram** | Matches token sequences from the prompt | No | Minimal | Documents with repeated phrases, RAG |

**Expected speedup (single request, A10G 24GB):**
- Eagle3 on Llama 3.1 8B: ~1.8-2.4x TTFT reduction
- Ngram on Llama 3.1 8B: ~1.2-1.5x TTFT reduction
- Speedup degrades at high concurrency (>16 concurrent requests)

---

## Draft Model Reference

| Target Model | Eagle3 Draft (vLLM) | Eagle3 Draft (SGLang) |
|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | `RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3` | `jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B` |
| `Qwen/Qwen3-8B` | `RedHatAI/Qwen3-8B-speculator.eagle3` | Not yet available |
| `google/gemma-3-4b-it` | Not yet available | Not yet available |
| `meta-llama/Llama-3.3-70B-Instruct` | `RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3` | `yuhuili/EAGLE3-LLaMA3.3-Instruct-70B` |

> **Note:** Eagle3 speculators must share the same tokenizer/vocabulary as the target model. Do not mix model families (e.g., a Llama draft with a Qwen target).

---

## 2025-2026 Models — Hardware Requirements

### A10G 24GB (current benchmark hardware)

| Model | HF ID | Est. VRAM | Spec-Dec Compatible |
|---|---|---|---|
| Qwen 3 8B | `Qwen/Qwen3-8B` | ~16 GB | Eagle3 (vLLM), Ngram both engines |
| Gemma 3 4B | `google/gemma-3-4b-it` | ~8 GB | Ngram only (no Eagle3 draft yet) |
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` | ~16 GB | Eagle3 + Ngram, both engines |
| DeepSeek-R1 Distill 7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | ~14 GB | Ngram only |

> For Eagle3 on 8B models, set `gpu-memory-utilization=0.80` (vLLM) or `mem-fraction-static=0.70` (SGLang) to leave room for the ~1-2 GB draft model.

### A100/H100 (larger hardware)

| Model | HF ID | Est. VRAM | Notes |
|---|---|---|---|
| Mistral Small 3.2 24B | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | ~48 GB | A100 80GB or 2× A100 40GB |
| Qwen 3 32B | `Qwen/Qwen3-32B` | ~64 GB | A100 80GB |
| Llama 3.3 70B | `meta-llama/Llama-3.3-70B-Instruct` | ~140 GB | 2× A100 80GB or H100 SXM |

---

## Running Speculative Decoding Benchmarks

### Prerequisites

1. Edit `.env` with your HuggingFace token (needed for gated models like Llama):
   ```bash
   HUGGING_FACE_HUB_TOKEN=hf_YOUR_TOKEN
   EAGLE3_VLLM_DRAFT=RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3
   EAGLE3_SGLANG_DRAFT=jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B
   ```

2. Set target model (Eagle3 works best with Llama 3.1 8B — has full draft model support):
   ```bash
   export MODEL=meta-llama/Llama-3.1-8B-Instruct
   ```

---

### Step-by-step: Baseline vs Eagle3 vs Ngram (vLLM)

```bash
# 1. Run baseline
docker compose --profile vllm up -d
sleep 120  # wait for model to load
python run_experiment.py run -s single_request_latency -e vllm --model meta-llama/Llama-3.1-8B-Instruct
python run_experiment.py run -s throughput_ramp -e vllm --model meta-llama/Llama-3.1-8B-Instruct
docker compose --profile vllm down

# 2. Run Eagle3
docker compose --profile vllm-eagle3 up -d
sleep 180  # two models load — takes longer
python run_experiment.py run -s single_request_latency -e vllm-eagle3 --model meta-llama/Llama-3.1-8B-Instruct
python run_experiment.py run -s throughput_ramp -e vllm-eagle3 --model meta-llama/Llama-3.1-8B-Instruct
docker compose --profile vllm-eagle3 down

# 3. Run Ngram
docker compose --profile vllm-ngram up -d
sleep 120
python run_experiment.py run -s single_request_latency -e vllm-ngram --model meta-llama/Llama-3.1-8B-Instruct
python run_experiment.py run -s throughput_ramp -e vllm-ngram --model meta-llama/Llama-3.1-8B-Instruct
docker compose --profile vllm-ngram down

# 4. Generate comparison report
python run_experiment.py report --output spec_dec_report.html
```

---

### Step-by-step: Baseline vs Eagle3 vs Ngram (SGLang)

```bash
# 1. Baseline
docker compose --profile sglang up -d && sleep 120
python run_experiment.py run -s single_request_latency -e sglang --model meta-llama/Llama-3.1-8B-Instruct
docker compose --profile sglang down

# 2. Eagle3
docker compose --profile sglang-eagle3 up -d && sleep 180
python run_experiment.py run -s single_request_latency -e sglang-eagle3 --model meta-llama/Llama-3.1-8B-Instruct
docker compose --profile sglang-eagle3 down

# 3. Ngram
docker compose --profile sglang-ngram up -d && sleep 120
python run_experiment.py run -s single_request_latency -e sglang-ngram --model meta-llama/Llama-3.1-8B-Instruct
docker compose --profile sglang-ngram down
```

---

### Using the matrix command (recommended for full run)

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct

# Baseline vLLM
docker compose --profile vllm up -d && sleep 120
python run_experiment.py matrix \
  --scenarios single_request_latency,throughput_ramp,long_context_stress,prefix_sharing_benefit,structured_generation_speed \
  --engines vllm --model $MODEL --iterations 2 --cooldown-seconds 120
docker compose --profile vllm down

# vLLM Eagle3
docker compose --profile vllm-eagle3 up -d && sleep 180
python run_experiment.py matrix \
  --scenarios single_request_latency,throughput_ramp \
  --engines vllm-eagle3 --model $MODEL --iterations 2 --cooldown-seconds 120
docker compose --profile vllm-eagle3 down

# vLLM Ngram
docker compose --profile vllm-ngram up -d && sleep 120
python run_experiment.py matrix \
  --scenarios single_request_latency,throughput_ramp \
  --engines vllm-ngram --model $MODEL --iterations 2 --cooldown-seconds 120
docker compose --profile vllm-ngram down
```

---

### Direct head-to-head comparison

```bash
# Compare baseline vs Eagle3 on the same scenario
# (requires both services to be running on different hosts, or run sequentially and use saved results)
python run_experiment.py compare \
  --scenario single_request_latency \
  --engines vllm,vllm-eagle3 \
  --model meta-llama/Llama-3.1-8B-Instruct
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| OOM on Eagle3 startup | Draft model + main model exceed VRAM | Reduce `gpu-memory-utilization` to 0.75 or `mem-fraction-static` to 0.65 |
| Eagle3 much slower than baseline | High concurrency saturates draft verification | Reduce concurrency or test at lower request rates |
| SGLang Eagle3 crashes on startup | Context length too large for draft tree | Add `--speculative-num-draft-tokens 8` to reduce tree size |
| Ngram shows no speedup | Prompts have no repeated token patterns | Ngram benefits only repetitive/document-style prompts; try `long_context` scenario |
| `EAGLE3_VLLM_DRAFT` not found on HF Hub | Draft model not available for this target | Use Ngram instead, or check RedHatAI HF page for latest speculators |

---

## Known Limitations

- Eagle3 speculative decoding has a context length cap of **2048 tokens** for the speculative portion in some vLLM versions. Long-context benchmarks may fall back to normal decoding.
- Speculative decoding speedup is **highest at low concurrency** (1-4 concurrent requests). At concurrency ≥ 32, benefits shrink significantly as the GPU is already saturated.
- The `structured_generation_speed` scenario uses constrained decoding, which may interact with speculative decoding in engine-specific ways. Treat those results with care.
- Eagle3 draft models are specific to a model family and version. `Qwen3-8B` speculators may not yet be available for SGLang.
