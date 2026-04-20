# Variance Analysis Report

**Methodology:** Each metric is sampled once per benchmark run. 95% CIs use the t-distribution (`scipy.stats.t`). **⚠ = CV > 5.0%** — claim unreliable, needs more iterations or investigation.

| Metric | Formula |
|--------|---------|
| TTFT P50/P95 | from `metrics.ttft.p50/p95` |
| Throughput | from `metrics.throughput.tokens_per_sec` |
| TPOT P95 | `(total_ms − ttft_ms) / max(output_tokens − 1, 1)`, P95 across requests |

## Single Request Latency

| Model | Engine | N runs | TTFT P50 (ms) | CV% | TTFT P95 (ms) | CV% | Tok/s | CV% | TPOT P95 (ms) | CV% |
|-------|--------|--------|---------------|-----|---------------|-----|---------------|-----|---------------|-----|
| Llama-3.1-8B-Instruct | sglang | 5 | 69.1 ± 0.0 | 0.0% | 69.4 ± 0.1 | 0.1% | 30.3 ± 0.0 | 0.0% | 33.0 ± 0.0 | 0.0% |
| Llama-3.1-8B-Instruct | vllm | 5 | 42.2 ± 0.0 | 0.1% | 42.5 ± 0.0 | 0.1% | 30.3 ± 0.0 | 0.0% | 33.1 ± 0.0 | 0.0% |
| Phi-4-mini-instruct | sglang | 5 | 40.1 ± 0.0 | 0.1% | 40.4 ± 0.1 | 0.1% | 52.8 ± 0.1 | 0.1% | 19.3 ± 0.0 | 0.0% |
| Phi-4-mini-instruct | vllm | 5 | 33.3 ± 0.0 | 0.1% | 33.8 ± 0.1 | 0.2% | 56.8 ± 0.0 | 0.0% | 18.0 ± 0.0 | 0.0% |
| gemma-2-2b-it | sglang | 5 | 26.3 ± 0.0 | 0.1% | 26.7 ± 0.1 | 0.2% | 77.5 ± 0.1 | 0.1% | 12.9 ± 0.0 | 0.0% |
| gemma-2-2b-it | vllm | 5 | 19.4 ± 0.1 | 0.2% | 23.3 ± 0.2 | 0.5% | 77.6 ± 0.0 | 0.0% | 13.1 ± 0.0 | 0.0% |
| gemma-3-4b-it | sglang | 5 | 78.6 ± 0.5 | 0.5% | 80.6 ± 0.8 | 0.8% | 45.0 ± 0.0 | 0.0% | 22.0 ± 0.0 | 0.0% |
| gemma-3-4b-it | vllm | 5 | 87.0 ± 0.4 | 0.3% | 89.5 ± 0.8 | 0.7% | 24.2 ± 0.3 | 1.1% | 41.9 ± 0.2 | 0.4% |

## Throughput Ramp

| Model | Engine | N runs | TTFT P50 (ms) | CV% | TTFT P95 (ms) | CV% | Tok/s | CV% | TPOT P95 (ms) | CV% |
|-------|--------|--------|---------------|-----|---------------|-----|---------------|-----|---------------|-----|
| Llama-3.1-8B-Instruct | sglang | 6 | 70.8 ± 0.7 | 0.9% | 169.4 ± 2.5 | 1.4% | 102.1 ± 0.0 | 0.0% | 40.7 ± 0.1 | 0.1% |
| Llama-3.1-8B-Instruct | vllm | 5 | 97.1 ± 1.1 | 0.9% | 181.8 ± 24.3 ⚠ | 10.8%⚠ | 101.8 ± 0.0 | 0.0% | 40.4 ± 0.1 | 0.1% |
| Phi-4-mini-instruct | sglang | 5 | 57.3 ± 0.1 | 0.2% | 168.9 ± 127.7 ⚠ | 60.9%⚠ | 175.8 ± 0.8 | 0.4% | 32.5 ± 0.2 | 0.5% |
| Phi-4-mini-instruct | vllm | 5 | 54.6 ± 2.3 | 3.5% | 135.6 ± 14.3 ⚠ | 8.5%⚠ | 188.8 ± 0.1 | 0.0% | 28.8 ± 0.0 | 0.1% |
| gemma-2-2b-it | sglang | 5 | 41.0 ± 2.4 | 4.6% | 125.9 ± 13.6 ⚠ | 8.7%⚠ | 263.5 ± 0.8 | 0.2% | 18.7 ± 0.4 | 1.7% |
| gemma-2-2b-it | vllm | 5 | 35.0 ± 0.3 | 0.6% | 107.3 ± 22.1 ⚠ | 16.6%⚠ | 264.9 ± 0.3 | 0.1% | 17.6 ± 0.0 | 0.1% |
| gemma-3-4b-it | sglang | 5 | 126.8 ± 1.8 | 1.1% | 260.6 ± 17.4 ⚠ | 5.4%⚠ | 149.1 ± 0.4 | 0.2% | 54.5 ± 0.6 | 0.8% |
| gemma-3-4b-it | vllm | 5 | 127.0 ± 0.3 | 0.2% | 186.2 ± 6.0 | 2.6% | 83.5 ± 0.2 | 0.2% | 44.1 ± 0.1 | 0.2% |

## Long Context Stress

| Model | Engine | N runs | TTFT P50 (ms) | CV% | TTFT P95 (ms) | CV% | Tok/s | CV% | TPOT P95 (ms) | CV% |
|-------|--------|--------|---------------|-----|---------------|-----|---------------|-----|---------------|-----|
| Llama-3.1-8B-Instruct | sglang | 5 | 69.5 ± 0.8 | 0.9% | 85.8 ± 6.6 ⚠ | 6.2%⚠ | 115.5 ± 0.0 | 0.0% | 34.7 ± 0.0 | 0.0% |
| Llama-3.1-8B-Instruct | vllm | 5 | 97.5 ± 1.9 | 1.6% | 100.0 ± 0.8 | 0.6% | 115.8 ± 0.0 | 0.0% | 34.8 ± 0.0 | 0.0% |
| Phi-4-mini-instruct | sglang | 5 | 37.2 ± 1.9 | 4.0% | 55.4 ± 6.4 ⚠ | 9.2%⚠ | 211.7 ± 0.0 | 0.0% | 19.8 ± 0.0 | 0.0% |
| Phi-4-mini-instruct | vllm | 5 | 49.7 ± 0.8 | 1.2% | 53.4 ± 2.1 | 3.2% | 212.2 ± 0.1 | 0.0% | 18.9 ± 0.0 | 0.0% |
| gemma-2-2b-it | sglang | 5 | 37.3 ± 0.9 | 2.0% | 44.7 ± 1.8 | 3.2% | 301.4 ± 0.6 | 0.2% | 13.4 ± 0.1 | 0.3% |
| gemma-2-2b-it | vllm | 5 | 35.0 ± 1.9 | 4.5% | 44.7 ± 0.9 | 1.6% | 305.5 ± 4.5 | 1.2% | 13.0 ± 0.0 | 0.2% |
| gemma-3-4b-it | sglang | 5 | 98.5 ± 1.1 | 0.9% | 128.6 ± 3.8 | 2.4% | 170.0 ± 7.9 | 3.7% | 23.3 ± 0.5 | 1.7% |
| gemma-3-4b-it | vllm | 5 | 127.6 ± 2.7 | 1.7% | 138.0 ± 2.6 | 1.5% | 97.2 ± 2.0 | 1.7% | 42.4 ± 0.3 | 0.5% |

## Prefix Sharing Benefit

| Model | Engine | N runs | TTFT P50 (ms) | CV% | TTFT P95 (ms) | CV% | Tok/s | CV% | TPOT P95 (ms) | CV% |
|-------|--------|--------|---------------|-----|---------------|-----|---------------|-----|---------------|-----|
| Llama-3.1-8B-Instruct | sglang | 5 | 63.6 ± 0.3 | 0.3% | 107.4 ± 56.3 ⚠ | 42.2%⚠ | 225.4 ± 0.0 | 0.0% | 34.9 ± 0.0 | 0.0% |
| Llama-3.1-8B-Instruct | vllm | 5 | 91.6 ± 2.2 | 1.9% | 100.3 ± 0.1 | 0.1% | 219.5 ± 0.1 | 0.0% | 35.2 ± 0.0 | 0.1% |
| Phi-4-mini-instruct | sglang | 5 | 55.6 ± 1.0 | 1.4% | 73.3 ± 29.0 ⚠ | 31.8%⚠ | 376.2 ± 0.4 | 0.1% | 20.9 ± 0.2 | 0.8% |
| Phi-4-mini-instruct | vllm | 5 | 52.0 ± 0.5 | 0.8% | 61.3 ± 3.0 | 4.0% | 403.1 ± 8.8 | 1.8% | 19.6 ± 0.1 | 0.3% |
| gemma-2-2b-it | sglang | 5 | 36.0 ± 0.4 | 1.0% | 52.9 ± 16.8 ⚠ | 25.6%⚠ | 550.2 ± 3.1 | 0.4% | 14.5 ± 0.1 | 0.6% |
| gemma-2-2b-it | vllm | 5 | 34.1 ± 0.2 | 0.4% | 46.5 ± 2.8 | 4.8% | 613.9 ± 36.7 | 4.8% | 13.3 ± 0.0 | 0.0% |
| gemma-3-4b-it | sglang | 5 | 117.8 ± 18.1 ⚠ | 12.4%⚠ | 150.6 ± 3.6 | 1.9% | 324.4 ± 10.0 | 2.5% | 24.8 ± 0.5 | 1.6% |
| gemma-3-4b-it | vllm | 5 | 122.4 ± 0.5 | 0.3% | 141.9 ± 2.1 | 1.2% | 183.9 ± 3.8 | 1.7% | 43.4 ± 0.1 | 0.3% |

## Structured Generation Speed

| Model | Engine | N runs | TTFT P50 (ms) | CV% | TTFT P95 (ms) | CV% | Tok/s | CV% | TPOT P95 (ms) | CV% |
|-------|--------|--------|---------------|-----|---------------|-----|---------------|-----|---------------|-----|
| Llama-3.1-8B-Instruct | sglang | 5 | 89.3 ± 0.9 | 0.8% | 129.5 ± 39.6 ⚠ | 24.6%⚠ | 422.7 ± 0.3 | 0.1% | 36.6 ± 0.0 | 0.0% |
| Llama-3.1-8B-Instruct | vllm | 5 | 97.5 ± 2.3 | 1.9% | 120.5 ± 12.7 ⚠ | 8.5%⚠ | 426.4 ± 0.1 | 0.0% | 36.6 ± 0.0 | 0.0% |
| Phi-4-mini-instruct | sglang | 5 | 55.6 ± 1.1 | 1.5% | 78.4 ± 23.2 ⚠ | 23.8%⚠ | 694.2 ± 43.7 ⚠ | 5.1%⚠ | 23.9 ± 0.4 | 1.5% |
| Phi-4-mini-instruct | vllm | 5 | 55.1 ± 0.4 | 0.5% | 70.7 ± 0.9 | 1.0% | 773.2 ± 28.9 | 3.0% | 20.9 ± 0.0 | 0.2% |
| gemma-2-2b-it | sglang | 5 | 41.9 ± 0.7 | 1.4% | 62.2 ± 14.8 ⚠ | 19.1%⚠ | 1023.1 ± 3.9 | 0.3% | 16.9 ± 0.3 | 1.6% |
| gemma-2-2b-it | vllm | 5 | 41.1 ± 4.7 ⚠ | 9.2%⚠ | 52.8 ± 6.4 ⚠ | 9.7%⚠ | 1226.2 ± 0.7 | 0.0% | 13.9 ± 0.0 | 0.0% |
| gemma-3-4b-it | sglang | 5 | 120.1 ± 20.9 ⚠ | 14.0%⚠ | 183.9 ± 69.4 ⚠ | 30.4%⚠ | 616.8 ± 10.3 | 1.3% | 25.3 ± 0.3 | 0.9% |
| gemma-3-4b-it | vllm | 5 | 124.5 ± 0.9 | 0.6% | 149.4 ± 2.0 | 1.1% | 341.3 ± 6.2 | 1.5% | 44.5 ± 0.2 | 0.3% |

## Headline Comparison Table (with 95% CI)

Replaces bare point estimates in the main README. Values from `single_request_latency` (TTFT) and `throughput_ramp` (tok/s).

| Model | vLLM TTFT P50 | SGLang TTFT P50 | vLLM Peak tok/s | SGLang Peak tok/s |
|-------|---------------|-----------------|-----------------|-------------------|
| Llama-3.1-8B-Instruct | 42.2 ± 0.0 ms | 69.1 ± 0.0 ms | 101.8 ± 0.0 tok/s | 102.1 ± 0.0 tok/s |
| Phi-4-mini-instruct | 33.3 ± 0.0 ms | 40.1 ± 0.0 ms | 188.8 ± 0.1 tok/s | 175.8 ± 0.8 tok/s |
| gemma-2-2b-it | 19.4 ± 0.1 ms | 26.3 ± 0.0 ms | 264.9 ± 0.3 tok/s | 263.5 ± 0.8 tok/s |
| gemma-3-4b-it | 87.0 ± 0.4 ms | 78.6 ± 0.5 ms | 83.5 ± 0.2 tok/s | 149.1 ± 0.4 tok/s |

## High-Variance Claims (CV > 5%) — Require Asterisk

- **Llama-3.1-8B-Instruct / sglang / long_context_stress** — TTFT P95 (ms): CV = 6.2% (mean 85.8, std 5.3)
- **Llama-3.1-8B-Instruct / sglang / prefix_sharing_benefit** — TTFT P95 (ms): CV = 42.2% (mean 107.4, std 45.3)
- **Llama-3.1-8B-Instruct / sglang / structured_generation_speed** — TTFT P95 (ms): CV = 24.6% (mean 129.5, std 31.9)
- **Llama-3.1-8B-Instruct / vllm / structured_generation_speed** — TTFT P95 (ms): CV = 8.5% (mean 120.5, std 10.3)
- **Llama-3.1-8B-Instruct / vllm / throughput_ramp** — TTFT P95 (ms): CV = 10.8% (mean 181.8, std 19.5)
- **Phi-4-mini-instruct / sglang / long_context_stress** — TTFT P95 (ms): CV = 9.2% (mean 55.4, std 5.1)
- **Phi-4-mini-instruct / sglang / prefix_sharing_benefit** — TTFT P95 (ms): CV = 31.8% (mean 73.3, std 23.3)
- **Phi-4-mini-instruct / sglang / structured_generation_speed** — TTFT P95 (ms): CV = 23.8% (mean 78.4, std 18.7)
- **Phi-4-mini-instruct / sglang / structured_generation_speed** — Throughput (tok/s): CV = 5.1% (mean 694.2, std 35.2)
- **Phi-4-mini-instruct / sglang / throughput_ramp** — TTFT P95 (ms): CV = 60.9% (mean 168.9, std 102.8)
- **Phi-4-mini-instruct / vllm / throughput_ramp** — TTFT P95 (ms): CV = 8.5% (mean 135.6, std 11.5)
- **gemma-2-2b-it / sglang / prefix_sharing_benefit** — TTFT P95 (ms): CV = 25.6% (mean 52.9, std 13.6)
- **gemma-2-2b-it / sglang / structured_generation_speed** — TTFT P95 (ms): CV = 19.1% (mean 62.2, std 11.9)
- **gemma-2-2b-it / sglang / throughput_ramp** — TTFT P95 (ms): CV = 8.7% (mean 125.9, std 11.0)
- **gemma-2-2b-it / vllm / structured_generation_speed** — TTFT P50 (ms): CV = 9.2% (mean 41.1, std 3.8)
- **gemma-2-2b-it / vllm / structured_generation_speed** — TTFT P95 (ms): CV = 9.7% (mean 52.8, std 5.1)
- **gemma-2-2b-it / vllm / throughput_ramp** — TTFT P95 (ms): CV = 16.6% (mean 107.3, std 17.8)
- **gemma-3-4b-it / sglang / prefix_sharing_benefit** — TTFT P50 (ms): CV = 12.4% (mean 117.8, std 14.5)
- **gemma-3-4b-it / sglang / structured_generation_speed** — TTFT P50 (ms): CV = 14.0% (mean 120.1, std 16.9)
- **gemma-3-4b-it / sglang / structured_generation_speed** — TTFT P95 (ms): CV = 30.4% (mean 183.9, std 55.9)
- **gemma-3-4b-it / sglang / throughput_ramp** — TTFT P95 (ms): CV = 5.4% (mean 260.6, std 14.0)
