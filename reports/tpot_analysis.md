# TPOT Analysis Report

**TPOT** (Time Per Output Token) is the average inter-token decode latency after the first token: `(total_ms − ttft_ms) / max(output_tokens − 1, 1)`.

Lower is better. P99 is the most conservative SLO-relevant metric.

## Single Request Latency

| Model | Engine | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | Stdev | N | Mean Output Tokens |
|-------|--------|----------|----------|----------|-----------|-------|---|--------------------|
| Llama-3.1-8B-Instruct | sglang | 32.96 | 32.96 | 32.96 | 32.96 | 0.00 | 250 | 128.0 |
| Llama-3.1-8B-Instruct | vllm | 33.07 | 33.08 | 33.08 | 33.07 | 0.00 | 250 | 128.0 |
| Phi-4-mini-instruct | sglang | 18.82 | 19.29 | 19.30 | 18.89 | 0.21 | 250 | 100.2 |
| Phi-4-mini-instruct | vllm | 17.74 | 17.99 | 18.00 | 17.76 | 0.13 | 250 | 102.8 |
| gemma-2-2b-it | sglang | 12.82 | 12.90 | 12.90 | 12.84 | 0.04 | 250 | 109.2 |
| gemma-2-2b-it | vllm | 12.88 | 13.13 | 13.13 | 12.96 | 0.10 | 250 | 103.0 |
| gemma-3-4b-it | sglang | 21.79 | 21.95 | 21.95 | 21.85 | 0.08 | 250 | 112.6 |
| gemma-3-4b-it | vllm | 40.94 | 41.88 | 42.15 | 40.98 | 0.57 | 250 | 115.6 |

## Throughput Ramp

| Model | Engine | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | Stdev | N | Mean Output Tokens |
|-------|--------|----------|----------|----------|-----------|-------|---|--------------------|
| Llama-3.1-8B-Instruct | sglang | 35.06 | 40.73 | 41.05 | 36.00 | 2.33 | 4200 | 256.0 |
| Llama-3.1-8B-Instruct | vllm | 35.24 | 40.37 | 40.46 | 35.92 | 2.07 | 3500 | 256.0 |
| Phi-4-mini-instruct | sglang | 20.36 | 32.56 | 32.82 | 23.15 | 5.27 | 3500 | 220.9 |
| Phi-4-mini-instruct | vllm | 19.37 | 28.84 | 28.91 | 20.65 | 3.20 | 3500 | 256.0 |
| gemma-2-2b-it | sglang | 13.48 | 18.62 | 19.24 | 14.55 | 2.03 | 3493 | 244.7 |
| gemma-2-2b-it | vllm | 13.21 | 17.55 | 17.59 | 13.96 | 1.41 | 3500 | 232.0 |
| gemma-3-4b-it | sglang | 22.73 | 54.41 | 55.20 | 31.38 | 14.09 | 3497 | 255.9 |
| gemma-3-4b-it | vllm | 42.24 | 44.16 | 44.32 | 42.32 | 1.04 | 3500 | 256.0 |

## Long Context Stress

| Model | Engine | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | Stdev | N | Mean Output Tokens |
|-------|--------|----------|----------|----------|-----------|-------|---|--------------------|
| Llama-3.1-8B-Instruct | sglang | 34.72 | 34.74 | 34.84 | 34.73 | 0.03 | 100 | 256.0 |
| Llama-3.1-8B-Instruct | vllm | 34.81 | 34.81 | 34.82 | 34.81 | 0.00 | 100 | 256.0 |
| Phi-4-mini-instruct | sglang | 19.74 | 19.76 | 19.80 | 19.74 | 0.02 | 100 | 256.0 |
| Phi-4-mini-instruct | vllm | 18.90 | 18.91 | 18.91 | 18.90 | 0.00 | 100 | 256.0 |
| gemma-2-2b-it | sglang | 13.34 | 13.47 | 13.50 | 13.34 | 0.09 | 100 | 242.8 |
| gemma-2-2b-it | vllm | 12.98 | 13.05 | 13.05 | 12.99 | 0.03 | 100 | 245.7 |
| gemma-3-4b-it | sglang | 22.87 | 23.38 | 23.77 | 22.91 | 0.41 | 100 | 239.8 |
| gemma-3-4b-it | vllm | 42.03 | 42.53 | 42.65 | 41.96 | 0.39 | 100 | 256.0 |

## Prefix Sharing Benefit

| Model | Engine | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | Stdev | N | Mean Output Tokens |
|-------|--------|----------|----------|----------|-----------|-------|---|--------------------|
| Llama-3.1-8B-Instruct | sglang | 34.94 | 34.95 | 34.96 | 34.93 | 0.07 | 500 | 128.0 |
| Llama-3.1-8B-Instruct | vllm | 35.16 | 35.18 | 35.21 | 35.15 | 0.09 | 500 | 128.0 |
| Phi-4-mini-instruct | sglang | 20.52 | 21.03 | 21.16 | 20.55 | 0.27 | 500 | 113.9 |
| Phi-4-mini-instruct | vllm | 19.27 | 19.55 | 19.70 | 19.29 | 0.12 | 500 | 119.9 |
| gemma-2-2b-it | sglang | 14.12 | 14.49 | 14.68 | 14.07 | 0.30 | 499 | 111.1 |
| gemma-2-2b-it | vllm | 13.12 | 13.28 | 13.30 | 13.15 | 0.07 | 500 | 115.8 |
| gemma-3-4b-it | sglang | 23.73 | 25.17 | 25.26 | 23.78 | 0.76 | 499 | 125.8 |
| gemma-3-4b-it | vllm | 42.67 | 43.46 | 43.67 | 42.59 | 0.66 | 500 | 108.7 |

## Structured Generation Speed

| Model | Engine | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | Stdev | N | Mean Output Tokens |
|-------|--------|----------|----------|----------|-----------|-------|---|--------------------|
| Llama-3.1-8B-Instruct | sglang | 36.36 | 36.57 | 36.82 | 36.38 | 0.30 | 1000 | 150.0 |
| Llama-3.1-8B-Instruct | vllm | 36.55 | 36.58 | 36.60 | 36.49 | 0.28 | 1000 | 150.0 |
| Phi-4-mini-instruct | sglang | 22.60 | 23.93 | 24.53 | 22.53 | 0.92 | 1000 | 72.7 |
| Phi-4-mini-instruct | vllm | 20.20 | 20.93 | 21.06 | 20.34 | 0.34 | 1000 | 110.3 |
| gemma-2-2b-it | sglang | 15.78 | 16.93 | 17.40 | 15.60 | 0.94 | 999 | 61.8 |
| gemma-2-2b-it | vllm | 13.73 | 13.91 | 13.94 | 13.75 | 0.11 | 1000 | 61.6 |
| gemma-3-4b-it | sglang | 24.55 | 25.39 | 25.76 | 24.46 | 0.61 | 984 | 148.1 |
| gemma-3-4b-it | vllm | 43.99 | 44.65 | 44.76 | 43.89 | 0.56 | 1000 | 75.5 |

## Engine Comparison (All Scenarios Aggregated)

Aggregates TPOT samples from all scenarios for a high-level engine comparison.

| Model | Engine | Weighted Mean TPOT (ms) | Total Requests |
|-------|--------|------------------------|----------------|
| Llama-3.1-8B-Instruct | sglang | 35.83 | 6050 |
| Llama-3.1-8B-Instruct | vllm | 35.80 | 5350 |
| Phi-4-mini-instruct | sglang | 22.53 | 5350 |
| Phi-4-mini-instruct | vllm | 20.30 | 5350 |
| gemma-2-2b-it | sglang | 14.60 | 5341 |
| gemma-2-2b-it | vllm | 13.78 | 5350 |
| gemma-3-4b-it | sglang | 28.79 | 5330 |
| gemma-3-4b-it | vllm | 42.57 | 5350 |
