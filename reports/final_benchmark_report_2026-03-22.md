# Final Multi-Model Benchmark Report (2026-03-22)

## Executive summary

This report consolidates the completed benchmark matrix collected on the AWS `g5.2xlarge` single-GPU host (NVIDIA A10G 24 GB) using **sequential engine execution**. The goal was to compare **vLLM** and **SGLang** across a representative mix of open models while avoiding VRAM contention by running only one engine at a time.

### Headline findings

- **Best single-request TTFT p95:** Gemma 2B on vLLM at **20.3 ms**.
- **Best throughput (tokens/sec):** Gemma 2B on SGLang at **36459.2 tok/s**.
- **Best throughput (requests/sec):** Gemma 2B on vLLM at **289.16 req/s**.
- **Phi-3 mini** was benchmarked on **vLLM only** because the SGLang FlashInfer/CUDA graph path crashed on `unsupported head_dim=96`.
- **Gemma 9B + vLLM** required tuned launch parameters on the A10G: `context=4096` and `gpu_memory_utilization=0.92`.

## Test environment

- Instance: **AWS g5.2xlarge**
- GPU: **NVIDIA A10G (24 GB VRAM)**
- Execution policy: **sequential only** (one engine at a time)
- Cooling policy: cooldown between engine switches / heavier model transitions
- Result source: consolidated metrics captured during the orchestration run on 2026-03-22

## Visual summary

### 1) Single-request TTFT p95

![Single request TTFT p95](figures/single_request_ttft_p95.svg)

### 2) Throughput tokens/sec

![Throughput tokens per second](figures/throughput_tokens_per_sec.svg)

### 3) Throughput requests/sec

![Throughput requests per second](figures/throughput_requests_per_sec.svg)

### 4) Throughput latency p95

![Throughput latency p95](figures/throughput_latency_p95.svg)

## Single-request latency results

| Model | Scenario | Engine | TTFT p50 | TTFT p95 | Latency p95 | Tok/s | Req/s | Success |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Gemma 2B | `single_request_latency` | SGLang | 34.1 ms | 35.0 ms | 1683.0 ms | 3785.5 | 29.57 | 100.0% |
| Gemma 2B | `single_request_latency` | vLLM | 19.8 ms | 20.3 ms | 1661.8 ms | 3749.2 | 29.29 | 100.0% |
| Phi-3 mini | `single_request_latency` | vLLM | 25.4 ms | 25.9 ms | 2243.6 ms | 2786.4 | 21.77 | 100.0% |
| Qwen 7B | `single_request_latency` | SGLang | 67.9 ms | 68.2 ms | 4178.4 ms | 1531.6 | 11.97 | 100.0% |
| Qwen 7B | `single_request_latency` | vLLM | 40.4 ms | 40.7 ms | 4202.4 ms | 1451.3 | 11.34 | 100.0% |
| Mistral 7B | `single_request_latency` | SGLang | 66.0 ms | 66.4 ms | 4057.3 ms | 1574.9 | 12.30 | 100.0% |
| Mistral 7B | `single_request_latency` | vLLM | 41.4 ms | 41.7 ms | 4044.0 ms | 1539.7 | 12.03 | 100.0% |
| Gemma 9B | `single_request_latency` | SGLang | 86.3 ms | 86.9 ms | 333.4 ms | 676.2 | 135.24 | 100.0% |
| Gemma 9B | `single_request_latency` | vLLM | 120.8 ms | 122.2 ms | 381.6 ms | 648.6 | 129.73 | 100.0% |

## Throughput-ramp results

| Model | Scenario | Engine | TTFT p50 | TTFT p95 | Latency p95 | Tok/s | Req/s | Success |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Gemma 2B | `throughput_ramp` | SGLang | 53.6 ms | 159.9 ms | 4898.1 ms | 36459.2 | 142.43 | 100.0% |
| Gemma 2B | `throughput_ramp` | vLLM | 44.0 ms | 189.9 ms | 2301.1 ms | 33875.2 | 289.16 | 100.0% |
| Phi-3 mini | `throughput_ramp` | vLLM | 55.5 ms | 188.6 ms | 8645.5 ms | 20533.9 | 80.21 | 100.0% |
| Qwen 7B | `throughput_ramp` | SGLang | 68.7 ms | 194.2 ms | 9581.5 ms | 18667.3 | 72.92 | 100.0% |
| Qwen 7B | `throughput_ramp` | vLLM | 89.9 ms | 311.9 ms | 10091.5 ms | 15140.9 | 68.98 | 100.0% |
| Mistral 7B | `throughput_ramp` | SGLang | 69.7 ms | 353.6 ms | 10332.4 ms | 17294.3 | 67.56 | 100.0% |
| Mistral 7B | `throughput_ramp` | vLLM | 92.3 ms | 240.5 ms | 10342.1 ms | 17175.6 | 67.09 | 100.0% |
| Gemma 9B | `throughput_ramp` | SGLang | 91.4 ms | 3666.6 ms | 5277.1 ms | 3595.1 | 99.86 | 100.0% |
| Gemma 9B | `throughput_ramp` | vLLM | 82.7 ms | 362.5 ms | 2483.7 ms | 9619.6 | 267.21 | 100.0% |

## Model-by-model takeaways

### Gemma 2B
- vLLM had the lower TTFT on the single-request case.
- vLLM dominated requests/sec on throughput ramp.
- SGLang posted slightly higher tokens/sec on the ramp.

### Phi-3 mini
- vLLM results are solid and competitive for a small model.
- SGLang could not be included on this hardware/software combination due to a reproducible compatibility failure.

### Qwen 7B
- vLLM won the low-latency single-request scenario.
- SGLang won the throughput-ramp tokens/sec and requests/sec for this model.

### Mistral 7B
- vLLM again won the single-request TTFT.
- Throughput-ramp performance between vLLM and SGLang ended up very close on this model.

### Gemma 9B
- SGLang fit and ran with default-style settings, but had very poor tail latency in throughput ramp.
- vLLM needed tuning to fit on the A10G, but once tuned it substantially improved p95 latency and throughput for the ramp scenario.

## Caveats

- These numbers are for a **single A10G host**, not a multi-GPU cluster.
- Sequential execution improves fairness on this box, but still reflects one-machine constraints.
- Later work should add:
  - multi-run variance / repeated trials,
  - richer prompt-pack driven workloads,
  - structured-output validity metrics,
  - and larger hardware tiers for 9B+ models.

## Files generated

- `reports/final_benchmark_report_2026-03-22.md`
- `reports/benchmark_snapshot_2026-03-22.json`
- `reports/figures/single_request_ttft_p95.svg`
- `reports/figures/throughput_tokens_per_sec.svg`
- `reports/figures/throughput_requests_per_sec.svg`
- `reports/figures/throughput_latency_p95.svg`
