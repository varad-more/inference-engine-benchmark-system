# Blog Companion Guide Pack

This document is written as a companion asset for readers who discover the repository through the blog post. It is designed to answer the most practical follow-up questions:

- What does this repo actually do?
- How do I run inference myself?
- How do I interpret the benchmark outputs?
- When should I choose vLLM vs SGLang?
- What are the common failure modes on a single-GPU setup?

---

## 1. What this repository helps you do

`inference-engine-benchmark-system` is a practical benchmark harness for comparing **vLLM** and **SGLang** under realistic inference workloads.

It helps you answer questions like:

- Which engine gives better **time to first token (TTFT)**?
- Which one handles **throughput ramp** better?
- How much does model family change the result?
- What breaks when you try to serve larger models on a single GPU?

It is not only a benchmark runner — it is also a reproducible systems workflow with:

- engine startup and health checks,
- saved JSON result artifacts,
- dashboard/API views,
- AWS deployment guidance,
- and reporting/visualization support.

---

## 2. Quickstart for a new user

### Goal
Run one model on one engine, verify health, execute a benchmark, and inspect results.

### Minimum requirements

| Requirement | Recommendation |
|---|---|
| GPU | NVIDIA A10G 24 GB or better |
| Instance | AWS `g5.2xlarge` for 7B-class models |
| Docker | Required |
| Hugging Face token | Required for most model pulls |
| Execution mode | Sequential on single-GPU hosts |

### First steps

1. Clone the repo.
2. Add your Hugging Face token to `.env`.
3. Start **one engine only**.
4. Wait for `/health` to go green.
5. Run `single_request_latency` first.
6. Run `throughput_ramp` second.
7. Inspect `results/*.json`.

---

## 3. Single-engine inference guide

### Example: run inference on vLLM

```bash
cd ~/repos/inference-engine-benchmark-system
source .venv/bin/activate
sudo docker compose down
sudo docker compose up -d vllm
curl http://localhost:8000/health
```

If healthy, send a request:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-2-2b-it",
    "prompt": "Explain cache invalidation in simple terms.",
    "max_tokens": 120,
    "temperature": 0.0
  }'
```

### Example: run inference on SGLang

```bash
cd ~/repos/inference-engine-benchmark-system
source .venv/bin/activate
sudo docker compose down
sudo docker compose up -d sglang
curl http://localhost:8001/health
```

Then:

```bash
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-2-2b-it",
    "prompt": "Explain cache invalidation in simple terms.",
    "max_tokens": 120,
    "temperature": 0.0
  }'
```

---

## 4. Benchmark guide

### Run single-request latency

```bash
python run_experiment.py run \
  --scenario single_request_latency \
  --engines vllm \
  --model google/gemma-2-2b-it
```

### Run throughput ramp

```bash
python run_experiment.py run \
  --scenario throughput_ramp \
  --engines vllm \
  --model google/gemma-2-2b-it
```

### Sequential benchmark rule

On a single A10G, do **not** run vLLM and SGLang together. Use this pattern:

1. Start engine A
2. Run benchmark(s)
3. Stop engine A
4. Cool down GPU
5. Start engine B
6. Run benchmark(s)

---

## 5. How to interpret the results

### Metrics guide

| Metric | Meaning | Better direction | Why it matters |
|---|---|---|---|
| TTFT p50 | Typical time to first token | Lower | Fastest signal of responsiveness |
| TTFT p95 | Tail time to first token | Lower | Shows user-facing jitter/slowness |
| Total latency p95 | Tail end-to-end response time | Lower | Better view of full request experience |
| Tokens/sec | Generation throughput | Higher | Important for heavy decode workloads |
| Requests/sec | Total request handling rate | Higher | Important for concurrent serving |
| Success rate | Fraction of successful requests | Higher | Tells you whether the run is stable |

### Practical reading pattern

| Scenario | What to focus on first |
|---|---|
| `single_request_latency` | TTFT p50 / TTFT p95 |
| `throughput_ramp` | Tokens/sec, Requests/sec, Latency p95 |
| Large model fits | Whether the engine starts at all without tuning |
| Structured tasks | Validity + latency, not just speed |

---

## 6. Example benchmark output in tabular form

### Example: completed benchmark snapshot

| Model | Scenario | Engine | TTFT p95 | Latency p95 | Tok/s | Req/s | Success |
|---|---|---|---:|---:|---:|---:|---:|
| Gemma 2B | single_request_latency | vLLM | 20.3 ms | 1661.8 ms | 3749.2 | 29.29 | 100.0% |
| Gemma 2B | throughput_ramp | SGLang | 159.9 ms | 4898.1 ms | 36459.2 | 142.43 | 100.0% |
| Qwen 7B | single_request_latency | vLLM | 40.7 ms | 4202.4 ms | 1451.3 | 11.34 | 100.0% |
| Qwen 7B | throughput_ramp | SGLang | 194.2 ms | 9581.5 ms | 18667.3 | 72.92 | 100.0% |
| Mistral 7B | throughput_ramp | vLLM | 240.5 ms | 10342.1 ms | 17175.6 | 67.09 | 100.0% |
| Gemma 9B | throughput_ramp | vLLM | 362.5 ms | 2483.7 ms | 9619.6 | 267.21 | 100.0% |

### Example: “who won what?” table

| Question | Winner | Why |
|---|---|---|
| Best single-request TTFT p95 | Gemma 2B on vLLM | Lowest observed TTFT p95 |
| Best throughput (tok/s) | Gemma 2B on SGLang | Highest aggregate decode throughput |
| Best throughput (req/s) | Gemma 2B on vLLM | Highest request-rate handling |
| Best 9B throughput result | Gemma 9B on vLLM (tuned) | Much stronger req/s and latency p95 than SGLang |

---

## 7. Common failure modes and what they mean

| Symptom | Likely cause | What to try |
|---|---|---|
| Engine won’t start | model too large / config mismatch | lower context length, increase memory utilization, verify model path |
| `curl ... /health` never returns | model still downloading/loading | wait, check logs |
| GPU OOM / cache block errors | KV cache cannot fit | reduce context, adjust engine memory settings |
| One engine works, other fails | engine/model compatibility issue | document it, pivot, don’t waste retries |
| Throughput tail latency explodes | model near memory/compute limits | reduce concurrency or move to larger GPU |

### Real examples from this benchmark series

| Model | Engine | Issue |
|---|---|---|
| Phi-3 mini | SGLang | FlashInfer/CUDA graph incompatibility (`unsupported head_dim=96`) |
| Gemma 9B | vLLM | Needed tuned fit settings on A10G (`context=4096`, `gpu_memory_utilization=0.92`) |

---

## 8. Recommended user journey from the blog

If a reader lands from the blog, the best progression is:

| Step | User action | Why |
|---|---|---|
| 1 | Read the final benchmark report | Understand the outcome before running anything |
| 2 | Start with Gemma 2B or Qwen 7B | Easier first-time experience |
| 3 | Run `single_request_latency` first | Fastest validation path |
| 4 | Run `throughput_ramp` second | Understand scale behavior |
| 5 | Compare results via dashboard/API | Easier interpretation |
| 6 | Only then move to larger models | Avoid unnecessary failures early |

---

## 9. Recommended callout boxes for the blog

### Callout: What surprised us

- vLLM consistently won low-latency single-request TTFT.
- SGLang stayed very competitive on throughput for some mid-sized models.
- Larger models on a single A10G often require practical tuning, not just bigger expectations.

### Callout: What this teaches

- You cannot declare a single universal engine winner.
- Model family matters.
- Hardware fit matters.
- Tail latency matters just as much as average speed.

### Callout: What users should do next

- reproduce one smaller model first,
- validate engine health,
- benchmark sequentially,
- inspect results in tabular form,
- then expand to heavier models.

---

## 10. FAQ for blog readers

### Which model should I try first?
Start with **Gemma 2B** or **Qwen 7B**.

### Can I run both engines together?
Not on a single A10G if you want stable, fair results.

### Why do some model/engine combinations fail?
Because serving compatibility is not only about the model — it also depends on CUDA kernels, memory fit, attention implementation, and engine-specific graph paths.

### Why do you care about TTFT p95 so much?
Because tail latency is what users feel when systems start to struggle.

### Is tokens/sec enough to choose an engine?
No. You should look at TTFT, p95 latency, req/s, and stability together.

---

## 11. Best companion assets to link from the blog

- `reports/final_benchmark_report_2026-03-22.md`
- `reports/final_benchmark_report_2026-03-22.html`
- `reports/benchmark_snapshot_2026-03-22.json`
- dashboard homepage
- dashboard `/api/current`
- dashboard `/api/results`

---

## 12. Suggested next upgrades for readers who want more rigor

| Upgrade | Why it helps |
|---|---|
| Multi-run variance / repeated trials | makes conclusions more defensible |
| Better prompt-pack workloads | improves realism |
| Structured output validity metrics | more production-relevant |
| Larger hardware tier | makes 9B+ behavior less distorted by fit constraints |
| More model families | reduces bias from one architecture |
