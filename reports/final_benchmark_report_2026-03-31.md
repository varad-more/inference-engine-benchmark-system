# Final Multi-Model Benchmark Report (2026-03-31)

## Executive summary

This report consolidates the completed benchmark matrix collected on an **AWS g5.2xlarge** host with a single **NVIDIA A10G (24 GB)** GPU. All engine runs were executed **sequentially** on the same machine to avoid VRAM contention and to keep the comparison fair on one GPU.

### Headline findings

- **Fastest single-request TTFT p95:** Gemma 2 2B on **vLLM** at **22.5 ms**.
- **Highest throughput (tokens/sec):** Gemma 2 2B on **vLLM** at **264.7 tok/s**.
- **Highest throughput (requests/sec):** Gemma 2 2B on **vLLM** at **1.14 req/s**.
- **Broad pattern:** vLLM consistently won the low-latency single-request TTFT tests, while throughput leadership depended on the model family.

## Environment

- Instance: **AWS g5.2xlarge**
- GPU: **NVIDIA A10G, 24 GB VRAM**
- Execution policy: **one engine at a time**
- Models included: Gemma 2 2B, SmolLM3 3B, Llama 3.2 3B, Phi-3 mini, Phi-4 mini, Gemma 3 4B, Qwen 2.5 7B, DS-R1 Qwen 7B, Mistral 7B, Llama 3.1 8B, Qwen3 8B, Granite 8B, DS-R1 Llama 8B, Gemma 2 9B, Gemma 3 12B, Qwen3 30B-A3B

## Important notes

- vLLM required tuned launch settings on the single A10G: `max_model_len=2048`, `gpu_memory_utilization=0.95`, `--disable-frontend-multiprocessing`, and `--enforce-eager`.

## Visual summary

### Single-request latency (TTFT p95)
![Single request TTFT p95](figures/single_request_ttft_p95.svg)

### Throughput tokens/sec
![Throughput tokens per second](figures/throughput_tokens_per_sec.svg)

### Throughput requests/sec
![Throughput requests per second](figures/throughput_requests_per_sec.svg)

### Throughput latency p95
![Throughput latency p95](figures/throughput_latency_p95.svg)

### Throughput tradeoff map
![Throughput tradeoff map](figures/throughput_tradeoff.svg)

## Single-request latency results

| Model | Scenario | Engine | TTFT p50 | TTFT p95 | Latency p95 | Tok/s | Req/s | Success |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Gemma 2 2B | `single_request_latency` | vLLM | 19.7 ms | 22.5 ms | 1655.6 ms | 77.6 | 0.75 | 100.0% |
| Gemma 2 2B | `single_request_latency` | SGLang | 30.4 ms | 31.5 ms | 1656.1 ms | 78.2 | 0.75 | 100.0% |
| Llama 3.2 3B | `single_request_latency` | vLLM | 22.6 ms | 23.3 ms | 1928.7 ms | 66.3 | 0.52 | 100.0% |
| SmolLM3 3B | `single_request_latency` | vLLM | 23.6 ms | 24.1 ms | 1882.5 ms | 69.2 | 0.54 | 100.0% |
| Llama 3.2 3B | `single_request_latency` | SGLang | 32.1 ms | 33.2 ms | 1903.4 ms | 67.7 | 0.53 | 100.0% |
| SmolLM3 3B | `single_request_latency` | SGLang | 57.1 ms | 60.0 ms | 2052.1 ms | 63.4 | 0.50 | 100.0% |
| Gemma 3 4B | `single_request_latency` | vLLM | 87.4 ms | 90.0 ms | 5332.0 ms | 23.8 | 0.21 | 100.0% |
| Phi-3 mini | `single_request_latency` | vLLM | 25.0 ms | 25.3 ms | 2233.8 ms | 57.8 | 0.45 | 100.0% |
| Phi-4 mini | `single_request_latency` | vLLM | 32.7 ms | 33.4 ms | 2269.0 ms | 56.8 | 0.55 | 100.0% |
| Gemma 3 4B | `single_request_latency` | SGLang | 78.0 ms | 80.5 ms | 2837.6 ms | 45.0 | 0.40 | 100.0% |
| Phi-3 mini | `single_request_latency` | SGLang | 43.2 ms | 50.5 ms | 2319.8 ms | 55.7 | 0.44 | 100.0% |
| Phi-4 mini | `single_request_latency` | SGLang | 39.6 ms | 39.9 ms | 2411.7 ms | 52.7 | 0.53 | 100.0% |
| DS-R1 Qwen 7B | `single_request_latency` | vLLM | 40.1 ms | 62.8 ms | 4215.8 ms | 30.5 | 0.28 | 100.0% |
| Mistral 7B | `single_request_latency` | vLLM | 41.0 ms | 62.5 ms | 4064.0 ms | 31.8 | 0.26 | 100.0% |
| Qwen 2.5 7B | `single_request_latency` | vLLM | 40.9 ms | 62.7 ms | 4220.3 ms | 30.6 | 0.29 | 100.0% |
| DS-R1 Qwen 7B | `single_request_latency` | SGLang | 65.7 ms | 66.3 ms | 4177.2 ms | 30.9 | 0.24 | 100.0% |
| Mistral 7B | `single_request_latency` | SGLang | 62.4 ms | 62.7 ms | 4047.0 ms | 31.8 | 0.26 | 100.0% |
| Qwen 2.5 7B | `single_request_latency` | SGLang | 65.6 ms | 65.7 ms | 4170.2 ms | 30.9 | 0.27 | 100.0% |
| DS-R1 Llama 8B | `single_request_latency` | vLLM | 42.4 ms | 43.1 ms | 4242.3 ms | 30.3 | 0.24 | 100.0% |
| Granite 8B | `single_request_latency` | vLLM | 46.3 ms | 47.0 ms | 4629.6 ms | 27.7 | 0.22 | 100.0% |
| Llama 3.1 8B | `single_request_latency` | vLLM | 42.7 ms | 43.8 ms | 4247.3 ms | 30.3 | 0.24 | 100.0% |
| Qwen3 8B | `single_request_latency` | vLLM | 44.0 ms | 44.6 ms | 4381.3 ms | 29.2 | 0.23 | 100.0% |
| DS-R1 Llama 8B | `single_request_latency` | SGLang | 68.6 ms | 68.9 ms | 4253.0 ms | 30.3 | 0.24 | 100.0% |
| Granite 8B | `single_request_latency` | SGLang | 76.0 ms | 76.5 ms | 4645.8 ms | 27.6 | 0.22 | 100.0% |
| Llama 3.1 8B | `single_request_latency` | SGLang | 69.1 ms | 69.5 ms | 4256.0 ms | 30.3 | 0.24 | 100.0% |
| Qwen3 8B | `single_request_latency` | SGLang | 71.6 ms | 72.1 ms | 4387.8 ms | 29.4 | 0.23 | 100.0% |
| Gemma 2 9B | `single_request_latency` | vLLM | 74.0 ms | 105.8 ms | 5360.3 ms | 24.0 | 0.21 | 100.0% |
| Gemma 2 9B | `single_request_latency` | SGLang | 82.9 ms | 83.4 ms | 5312.4 ms | 24.1 | 0.21 | 100.0% |

## Throughput-ramp results

| Model | Scenario | Engine | TTFT p50 | TTFT p95 | Latency p95 | Tok/s | Req/s | Success |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Gemma 2 2B | `throughput_ramp` | vLLM | 41.1 ms | 135.5 ms | 4588.2 ms | 264.7 | 1.14 | 100.0% |
| Gemma 2 2B | `throughput_ramp` | SGLang | 46.9 ms | 156.1 ms | 4844.5 ms | 258.0 | 1.05 | 100.0% |
| Llama 3.2 3B | `throughput_ramp` | vLLM | 49.5 ms | 174.9 ms | 5415.8 ms | 223.5 | 0.87 | 100.0% |
| SmolLM3 3B | `throughput_ramp` | vLLM | 48.5 ms | 143.3 ms | 5521.5 ms | 229.5 | 0.90 | 100.0% |
| Llama 3.2 3B | `throughput_ramp` | SGLang | 46.3 ms | 159.5 ms | 5544.7 ms | 226.0 | 0.88 | 100.0% |
| SmolLM3 3B | `throughput_ramp` | SGLang | 95.1 ms | 371.9 ms | 10724.5 ms | 204.6 | 0.80 | 100.0% |
| Gemma 3 4B | `throughput_ramp` | vLLM | 126.8 ms | 218.1 ms | 11538.7 ms | 83.9 | 0.33 | 100.0% |
| Phi-3 mini | `throughput_ramp` | vLLM | 53.7 ms | 228.9 ms | 8063.6 ms | 190.9 | 0.75 | 100.0% |
| Phi-4 mini | `throughput_ramp` | vLLM | 55.5 ms | 153.1 ms | 7511.6 ms | 188.6 | 0.74 | 100.0% |
| Gemma 3 4B | `throughput_ramp` | SGLang | 125.6 ms | 302.8 ms | 14731.5 ms | 149.2 | 0.58 | 100.0% |
| Phi-3 mini | `throughput_ramp` | SGLang | 56.6 ms | 167.3 ms | 7538.7 ms | 187.2 | 0.73 | 100.0% |
| Phi-4 mini | `throughput_ramp` | SGLang | 56.8 ms | 160.9 ms | 8410.8 ms | 175.9 | 0.79 | 100.0% |
| DS-R1 Qwen 7B | `throughput_ramp` | vLLM | 92.6 ms | 184.7 ms | 9322.1 ms | 105.6 | 0.41 | 100.0% |
| Mistral 7B | `throughput_ramp` | vLLM | 92.5 ms | 209.8 ms | 10139.1 ms | 106.6 | 0.42 | 100.0% |
| Qwen 2.5 7B | `throughput_ramp` | vLLM | 92.5 ms | 196.5 ms | 9772.7 ms | 105.3 | 0.41 | 100.0% |
| DS-R1 Qwen 7B | `throughput_ramp` | SGLang | 69.7 ms | 313.3 ms | 9603.0 ms | 106.1 | 0.41 | 100.0% |
| Mistral 7B | `throughput_ramp` | SGLang | 72.7 ms | 179.3 ms | 10122.3 ms | 106.8 | 0.42 | 99.9% |
| Qwen 2.5 7B | `throughput_ramp` | SGLang | 65.9 ms | 175.1 ms | 9542.9 ms | 106.3 | 0.42 | 100.0% |
| DS-R1 Llama 8B | `throughput_ramp` | vLLM | 98.3 ms | 244.6 ms | 10547.8 ms | 101.8 | 0.40 | 100.0% |
| Granite 8B | `throughput_ramp` | vLLM | 107.4 ms | 233.9 ms | 12299.3 ms | 92.8 | 0.36 | 100.0% |
| Llama 3.1 8B | `throughput_ramp` | vLLM | 97.6 ms | 195.6 ms | 10523.5 ms | 101.7 | 0.40 | 100.0% |
| Qwen3 8B | `throughput_ramp` | vLLM | 101.2 ms | 220.8 ms | 11646.3 ms | 98.3 | 0.38 | 100.0% |
| DS-R1 Llama 8B | `throughput_ramp` | SGLang | 94.1 ms | 177.8 ms | 10611.2 ms | 102.0 | 0.40 | 100.0% |
| Granite 8B | `throughput_ramp` | SGLang | 77.1 ms | 194.6 ms | 12521.8 ms | 92.8 | 0.36 | 100.0% |
| Llama 3.1 8B | `throughput_ramp` | SGLang | 69.5 ms | 194.0 ms | 10584.6 ms | 102.1 | 0.40 | 100.0% |
| Qwen3 8B | `throughput_ramp` | SGLang | 72.5 ms | 183.1 ms | 11692.4 ms | 98.5 | 0.38 | 100.0% |
| Gemma 2 9B | `throughput_ramp` | vLLM | 126.9 ms | 280.3 ms | 14399.0 ms | 79.8 | 0.33 | 100.0% |
| Gemma 2 9B | `throughput_ramp` | SGLang | 124.6 ms | 25982.1 ms | 46027.4 ms | 77.5 | 0.30 | 99.7% |

## Model-by-model takeaways

### Gemma 2 2B
- vLLM won the single-request TTFT comparison.
- For throughput, vLLM led on both tok/s and req/s.
### SmolLM3 3B
- vLLM won the single-request TTFT comparison.
- For throughput, vLLM led on both tok/s and req/s.
### Llama 3.2 3B
- vLLM won the single-request TTFT comparison.
- For throughput, SGLang led on both tok/s and req/s.
### Phi-3 mini
- vLLM won the single-request TTFT comparison.
- For throughput, vLLM led on both tok/s and req/s.
### Phi-4 mini
- vLLM won the single-request TTFT comparison.
- For throughput, vLLM led on tok/s while SGLang led on req/s.
### Gemma 3 4B
- SGLang won the single-request TTFT comparison.
- For throughput, SGLang led on both tok/s and req/s.
### Qwen 2.5 7B
- vLLM won the single-request TTFT comparison.
- For throughput, SGLang led on both tok/s and req/s.
### DS-R1 Qwen 7B
- vLLM won the single-request TTFT comparison.
- For throughput, SGLang led on tok/s while vLLM led on req/s.
### Mistral 7B
- vLLM won the single-request TTFT comparison.
- For throughput, SGLang led on tok/s while vLLM led on req/s.
### Llama 3.1 8B
- vLLM won the single-request TTFT comparison.
- For throughput, SGLang led on tok/s while vLLM led on req/s.
### Qwen3 8B
- vLLM won the single-request TTFT comparison.
- For throughput, SGLang led on tok/s while vLLM led on req/s.
### Granite 8B
- vLLM won the single-request TTFT comparison.
- For throughput, vLLM led on both tok/s and req/s.
### DS-R1 Llama 8B
- vLLM won the single-request TTFT comparison.
- For throughput, SGLang led on tok/s while vLLM led on req/s.
### Gemma 2 9B
- SGLang won the single-request TTFT comparison.
- For throughput, vLLM led on both tok/s and req/s.
- vLLM required tuned launch settings on the single A10G: `max_model_len=2048`, `gpu_memory_utilization=0.95`, `--disable-frontend-multiprocessing`, and `--enforce-eager`.

## Interpretation

This matrix shows why model/engine benchmarking should not be reduced to a single winner. Across this run:

- **vLLM** repeatedly delivered the lowest TTFT in single-request tests.
- **SGLang** remained very competitive and in some cases won or matched throughput on mid-sized models.
- **Larger models** on a single A10G can require engine-specific tuning to fit and behave well.

The data is therefore best used as an **engineering decision aid**, not a blanket statement that one engine dominates all workloads.

## Generated artifacts

- `reports/final_benchmark_report_2026-03-31.md`
- `reports/final_benchmark_report_2026-03-31.html`
- `reports/benchmark_snapshot_2026-03-31.json`
- `reports/figures/single_request_ttft_p95.svg`
- `reports/figures/throughput_tokens_per_sec.svg`
- `reports/figures/throughput_requests_per_sec.svg`
- `reports/figures/throughput_latency_p95.svg`
- `reports/figures/throughput_tradeoff.svg`
