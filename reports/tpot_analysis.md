# TPOT Analysis Report

**TPOT** (Time Per Output Token) is the average inter-token decode latency after the first token: `(total_ms − ttft_ms) / max(output_tokens − 1, 1)`.

Lower is better. P99 is the most conservative SLO-relevant metric.

## Single Request Latency

| Model | Engine | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | Stdev | N | Mean Output Tokens |
|-------|--------|----------|----------|----------|-----------|-------|---|--------------------|
| DeepSeek-R1-Distill-Llama-8B | sglang | 32.94 | 32.95 | 32.95 | 32.94 | 0.01 | 50 | 128.0 |
| DeepSeek-R1-Distill-Llama-8B | vllm | 33.07 | 33.07 | 33.07 | 33.07 | 0.01 | 50 | 128.0 |
| DeepSeek-R1-Distill-Qwen-7B | sglang | 32.37 | 32.37 | 32.38 | 32.37 | 0.00 | 50 | 128.0 |
| DeepSeek-R1-Distill-Qwen-7B | vllm | 32.71 | 33.13 | 33.14 | 32.87 | 0.20 | 50 | 109.2 |
| Llama-3.1-8B-Instruct | sglang | 32.96 | 38.92 | 38.99 | 34.89 | 2.78 | 150 | 92.0 |
| Llama-3.1-8B-Instruct | vllm | 35.72 | 40.30 | 40.30 | 36.39 | 2.98 | 150 | 109.9 |
| Llama-3.2-3B-Instruct | sglang | 14.72 | 14.73 | 14.73 | 14.72 | 0.00 | 50 | 128.0 |
| Llama-3.2-3B-Instruct | vllm | 15.00 | 15.00 | 15.01 | 15.00 | 0.00 | 50 | 128.0 |
| Mistral-7B-Instruct-v0.3 | sglang | 31.37 | 31.65 | 31.66 | 31.43 | 0.11 | 50 | 124.0 |
| Mistral-7B-Instruct-v0.3 | vllm | 31.51 | 31.80 | 31.81 | 31.57 | 0.12 | 50 | 124.0 |
| Phi-3-mini-4k-instruct | sglang | 17.72 | 17.92 | 18.01 | 17.75 | 0.10 | 50 | 128.0 |
| Phi-3-mini-4k-instruct | vllm | 17.39 | 17.39 | 17.39 | 17.39 | 0.00 | 50 | 128.0 |
| Phi-4-mini-instruct | sglang | 18.83 | 19.30 | 19.31 | 18.89 | 0.21 | 50 | 100.2 |
| Phi-4-mini-instruct | vllm | 17.75 | 18.01 | 18.02 | 17.77 | 0.13 | 50 | 102.8 |
| Qwen2.5-7B-Instruct | sglang | 32.32 | 32.80 | 32.80 | 32.41 | 0.20 | 50 | 115.4 |
| Qwen2.5-7B-Instruct | vllm | 32.74 | 33.25 | 33.25 | 32.92 | 0.23 | 50 | 105.8 |
| Qwen3-8B | sglang | 36.99 | 40.34 | 40.87 | 37.10 | 3.14 | 100 | 73.3 |
| Qwen3-8B | vllm | 35.59 | 37.16 | 37.24 | 35.65 | 1.54 | 100 | 115.1 |
| SmolLM3-3B | sglang | 15.69 | 15.69 | 15.69 | 15.69 | 0.00 | 50 | 128.0 |
| SmolLM3-3B | vllm | 14.63 | 14.63 | 14.71 | 14.63 | 0.02 | 50 | 128.0 |
| gemma-2-2b-it | sglang | 12.80 | 12.92 | 12.93 | 12.84 | 0.06 | 50 | 104.0 |
| gemma-2-2b-it | vllm | 12.86 | 13.11 | 13.11 | 12.93 | 0.10 | 50 | 103.0 |
| gemma-2-9b-it | sglang | 41.15 | 42.15 | 42.16 | 41.35 | 0.40 | 50 | 116.2 |
| gemma-2-9b-it | vllm | 41.38 | 42.35 | 42.38 | 41.57 | 0.39 | 50 | 116.0 |
| gemma-3-4b-it | sglang | 21.72 | 21.89 | 21.89 | 21.78 | 0.08 | 50 | 112.7 |
| gemma-3-4b-it | vllm | 40.96 | 41.93 | 42.28 | 41.07 | 0.51 | 50 | 115.6 |
| granite-3.3-8b-instruct | sglang | 35.98 | 35.98 | 35.98 | 35.98 | 0.00 | 50 | 128.0 |
| granite-3.3-8b-instruct | vllm | 36.08 | 36.09 | 36.09 | 36.08 | 0.00 | 50 | 128.0 |

## Throughput Ramp

| Model | Engine | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | Stdev | N | Mean Output Tokens |
|-------|--------|----------|----------|----------|-----------|-------|---|--------------------|
| DeepSeek-R1-Distill-Llama-8B | sglang | 34.99 | 40.95 | 41.10 | 36.00 | 2.35 | 700 | 256.0 |
| DeepSeek-R1-Distill-Llama-8B | vllm | 35.23 | 40.40 | 40.44 | 35.92 | 2.08 | 700 | 256.0 |
| DeepSeek-R1-Distill-Qwen-7B | sglang | 33.05 | 36.68 | 36.78 | 33.62 | 1.27 | 700 | 256.0 |
| DeepSeek-R1-Distill-Qwen-7B | vllm | 33.14 | 35.76 | 35.83 | 33.39 | 0.86 | 700 | 256.0 |
| Llama-3.1-8B-Instruct | sglang | 37.96 | 160.42 | 220.83 | 53.81 | 44.42 | 2100 | 178.7 |
| Llama-3.1-8B-Instruct | vllm | 37.54 | 53.18 | 92.70 | 41.52 | 10.90 | 2100 | 218.6 |
| Llama-3.2-3B-Instruct | sglang | 15.75 | 21.11 | 21.32 | 16.67 | 1.91 | 700 | 256.0 |
| Llama-3.2-3B-Instruct | vllm | 16.11 | 20.55 | 20.56 | 16.66 | 1.55 | 700 | 256.0 |
| Mistral-7B-Instruct-v0.3 | sglang | 33.51 | 39.05 | 39.12 | 34.41 | 2.25 | 699 | 256.0 |
| Mistral-7B-Instruct-v0.3 | vllm | 33.75 | 38.92 | 38.98 | 34.38 | 2.05 | 700 | 256.0 |
| Phi-3-mini-4k-instruct | sglang | 19.39 | 28.87 | 29.10 | 20.71 | 3.17 | 700 | 256.0 |
| Phi-3-mini-4k-instruct | vllm | 19.24 | 30.72 | 30.74 | 20.77 | 3.79 | 700 | 256.0 |
| Phi-4-mini-instruct | sglang | 20.45 | 32.51 | 32.66 | 23.15 | 5.26 | 700 | 221.9 |
| Phi-4-mini-instruct | vllm | 19.37 | 28.85 | 28.91 | 20.66 | 3.20 | 700 | 255.9 |
| Qwen2.5-7B-Instruct | sglang | 33.10 | 36.75 | 36.91 | 33.55 | 1.26 | 700 | 256.0 |
| Qwen2.5-7B-Instruct | vllm | 33.36 | 37.47 | 37.62 | 33.84 | 1.36 | 700 | 256.0 |
| Qwen3-8B | sglang | 40.75 | 241.97 | 256.23 | 67.70 | 58.32 | 1400 | 140.2 |
| Qwen3-8B | vllm | 38.18 | 46.58 | 54.26 | 39.50 | 4.45 | 1400 | 247.5 |
| SmolLM3-3B | sglang | 16.94 | 41.23 | 41.93 | 23.40 | 10.52 | 700 | 256.0 |
| SmolLM3-3B | vllm | 15.56 | 21.14 | 21.16 | 16.80 | 2.18 | 700 | 256.0 |
| gemma-2-2b-it | sglang | 13.63 | 18.36 | 18.43 | 14.67 | 1.96 | 700 | 245.7 |
| gemma-2-2b-it | vllm | 13.21 | 17.55 | 17.63 | 13.97 | 1.42 | 700 | 232.2 |
| gemma-2-9b-it | sglang | 44.72 | 46.96 | 80.14 | 46.08 | 11.90 | 698 | 255.0 |
| gemma-2-9b-it | vllm | 45.51 | 56.06 | 72.09 | 47.22 | 5.66 | 700 | 240.2 |
| gemma-3-4b-it | sglang | 22.69 | 56.79 | 56.83 | 31.76 | 14.82 | 700 | 256.0 |
| gemma-3-4b-it | vllm | 42.25 | 44.53 | 44.61 | 42.41 | 1.39 | 700 | 256.0 |
| granite-3.3-8b-instruct | sglang | 38.72 | 48.43 | 48.65 | 40.29 | 3.57 | 700 | 256.0 |
| granite-3.3-8b-instruct | vllm | 38.91 | 47.33 | 47.45 | 39.98 | 3.15 | 700 | 256.0 |

## Long Context Stress

| Model | Engine | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | Stdev | N | Mean Output Tokens |
|-------|--------|----------|----------|----------|-----------|-------|---|--------------------|
| DeepSeek-R1-Distill-Llama-8B | sglang | 34.65 | 34.67 | 34.81 | 34.66 | 0.04 | 20 | 256.0 |
| DeepSeek-R1-Distill-Llama-8B | vllm | 34.81 | 34.82 | 34.83 | 34.81 | 0.01 | 20 | 256.0 |
| DeepSeek-R1-Distill-Qwen-7B | sglang | 32.79 | 32.80 | 32.91 | 32.79 | 0.03 | 20 | 256.0 |
| DeepSeek-R1-Distill-Qwen-7B | vllm | 32.86 | 32.86 | 32.87 | 32.86 | 0.01 | 20 | 256.0 |
| Llama-3.1-8B-Instruct | sglang | 34.70 | 34.72 | 34.77 | 34.71 | 0.02 | 20 | 256.0 |
| Llama-3.1-8B-Instruct | vllm | 34.81 | 34.82 | 34.82 | 34.81 | 0.01 | 20 | 256.0 |
| Llama-3.2-3B-Instruct | sglang | 15.64 | 15.65 | 15.65 | 15.64 | 0.01 | 20 | 256.0 |
| Llama-3.2-3B-Instruct | vllm | 15.81 | 15.82 | 15.82 | 15.81 | 0.00 | 20 | 256.0 |
| Mistral-7B-Instruct-v0.3 | sglang | 33.45 | 33.60 | 33.70 | 33.45 | 0.14 | 20 | 228.2 |
| Mistral-7B-Instruct-v0.3 | vllm | 33.29 | 33.45 | 33.45 | 33.33 | 0.08 | 20 | 236.1 |
| Phi-3-mini-4k-instruct | sglang | 18.87 | 18.91 | 18.94 | 18.87 | 0.03 | 20 | 256.0 |
| Phi-3-mini-4k-instruct | vllm | 18.55 | 18.56 | 18.57 | 18.55 | 0.01 | 20 | 256.0 |
| Phi-4-mini-instruct | sglang | 19.74 | 19.80 | 19.83 | 19.75 | 0.03 | 20 | 256.0 |
| Phi-4-mini-instruct | vllm | 18.92 | 18.93 | 18.93 | 18.92 | 0.01 | 20 | 256.0 |
| Qwen2.5-7B-Instruct | sglang | 32.77 | 32.78 | 32.83 | 32.77 | 0.02 | 20 | 256.0 |
| Qwen2.5-7B-Instruct | vllm | 32.97 | 33.10 | 33.11 | 33.01 | 0.07 | 20 | 250.2 |
| Qwen3-8B | sglang | 35.99 | 36.07 | 36.15 | 36.01 | 0.05 | 20 | 256.0 |
| Qwen3-8B | vllm | 36.05 | 36.07 | 36.07 | 36.05 | 0.01 | 20 | 256.0 |
| SmolLM3-3B | sglang | 17.08 | 17.46 | 17.47 | 17.07 | 0.19 | 20 | 236.4 |
| SmolLM3-3B | vllm | 15.40 | 15.47 | 15.48 | 15.40 | 0.08 | 20 | 239.7 |
| gemma-2-2b-it | sglang | 13.45 | 13.58 | 13.61 | 13.42 | 0.09 | 20 | 234.0 |
| gemma-2-2b-it | vllm | 12.97 | 13.02 | 13.02 | 12.98 | 0.02 | 20 | 250.7 |
| gemma-2-9b-it | sglang | 44.58 | 45.39 | 45.39 | 44.61 | 0.53 | 20 | 209.2 |
| gemma-2-9b-it | vllm | 46.12 | 47.08 | 47.12 | 45.86 | 0.91 | 20 | 105.8 |
| gemma-3-4b-it | sglang | 22.52 | 22.53 | 22.57 | 22.52 | 0.01 | 20 | 254.3 |
| gemma-3-4b-it | vllm | 41.84 | 42.19 | 42.29 | 41.85 | 0.31 | 20 | 256.0 |
| granite-3.3-8b-instruct | sglang | 38.65 | 38.73 | 38.86 | 38.54 | 0.26 | 20 | 241.3 |
| granite-3.3-8b-instruct | vllm | 38.41 | 38.57 | 38.57 | 38.44 | 0.10 | 20 | 242.8 |

## Prefix Sharing Benefit

| Model | Engine | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | Stdev | N | Mean Output Tokens |
|-------|--------|----------|----------|----------|-----------|-------|---|--------------------|
| DeepSeek-R1-Distill-Llama-8B | sglang | 34.89 | 34.90 | 34.92 | 34.88 | 0.12 | 100 | 128.0 |
| DeepSeek-R1-Distill-Llama-8B | vllm | 35.17 | 35.18 | 35.19 | 35.15 | 0.08 | 100 | 128.0 |
| DeepSeek-R1-Distill-Qwen-7B | sglang | 33.16 | 33.31 | 33.32 | 33.10 | 0.31 | 100 | 128.0 |
| DeepSeek-R1-Distill-Qwen-7B | vllm | 33.10 | 33.13 | 33.13 | 33.09 | 0.05 | 100 | 128.0 |
| Llama-3.1-8B-Instruct | sglang | 34.95 | 35.09 | 35.10 | 34.97 | 0.09 | 100 | 128.0 |
| Llama-3.1-8B-Instruct | vllm | 35.19 | 35.21 | 35.21 | 35.17 | 0.09 | 100 | 128.0 |
| Llama-3.2-3B-Instruct | sglang | 15.73 | 15.77 | 15.78 | 15.73 | 0.03 | 100 | 128.0 |
| Llama-3.2-3B-Instruct | vllm | 16.05 | 16.07 | 16.07 | 16.04 | 0.06 | 100 | 128.0 |
| Mistral-7B-Instruct-v0.3 | sglang | 35.26 | 36.01 | 36.38 | 35.14 | 0.68 | 100 | 107.4 |
| Mistral-7B-Instruct-v0.3 | vllm | 34.03 | 34.12 | 34.14 | 33.92 | 0.19 | 100 | 105.6 |
| Phi-3-mini-4k-instruct | sglang | 19.09 | 19.26 | 19.28 | 19.10 | 0.11 | 100 | 128.0 |
| Phi-3-mini-4k-instruct | vllm | 19.08 | 19.10 | 19.10 | 19.05 | 0.14 | 100 | 128.0 |
| Phi-4-mini-instruct | sglang | 20.78 | 21.14 | 21.50 | 20.74 | 0.27 | 100 | 114.0 |
| Phi-4-mini-instruct | vllm | 19.28 | 19.65 | 19.70 | 19.33 | 0.15 | 100 | 113.8 |
| Qwen2.5-7B-Instruct | sglang | 33.83 | 34.37 | 34.71 | 33.81 | 0.44 | 100 | 122.1 |
| Qwen2.5-7B-Instruct | vllm | 33.35 | 33.65 | 33.67 | 33.40 | 0.13 | 100 | 124.4 |
| Qwen3-8B | sglang | 36.22 | 36.34 | 36.37 | 36.23 | 0.12 | 100 | 128.0 |
| Qwen3-8B | vllm | 36.47 | 36.49 | 36.50 | 36.45 | 0.10 | 100 | 128.0 |
| SmolLM3-3B | sglang | 18.69 | 19.30 | 19.58 | 18.56 | 0.64 | 100 | 104.7 |
| SmolLM3-3B | vllm | 15.61 | 15.90 | 15.94 | 15.67 | 0.14 | 100 | 105.0 |
| gemma-2-2b-it | sglang | 14.43 | 16.11 | 30.18 | 14.99 | 2.85 | 100 | 101.0 |
| gemma-2-2b-it | vllm | 13.12 | 13.28 | 13.30 | 13.16 | 0.08 | 100 | 114.1 |
| gemma-2-9b-it | sglang | 49.06 | 50.57 | 50.84 | 48.81 | 1.28 | 100 | 80.1 |
| gemma-2-9b-it | vllm | 46.21 | 47.94 | 47.99 | 46.33 | 0.98 | 100 | 87.5 |
| gemma-3-4b-it | sglang | 23.21 | 23.22 | 23.24 | 23.16 | 0.18 | 99 | 126.9 |
| gemma-3-4b-it | vllm | 42.45 | 43.43 | 43.87 | 42.50 | 0.53 | 100 | 104.1 |
| granite-3.3-8b-instruct | sglang | 38.59 | 38.61 | 38.63 | 38.60 | 0.15 | 100 | 128.0 |
| granite-3.3-8b-instruct | vllm | 38.83 | 38.88 | 38.89 | 38.82 | 0.10 | 100 | 128.0 |

## Structured Generation Speed

| Model | Engine | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | Stdev | N | Mean Output Tokens |
|-------|--------|----------|----------|----------|-----------|-------|---|--------------------|
| DeepSeek-R1-Distill-Llama-8B | sglang | 36.36 | 36.78 | 36.82 | 36.38 | 0.32 | 200 | 150.0 |
| DeepSeek-R1-Distill-Llama-8B | vllm | 36.55 | 36.59 | 36.59 | 36.50 | 0.28 | 200 | 150.0 |
| DeepSeek-R1-Distill-Qwen-7B | sglang | 33.59 | 33.94 | 33.98 | 33.62 | 0.16 | 200 | 150.0 |
| DeepSeek-R1-Distill-Qwen-7B | vllm | 33.82 | 33.86 | 34.04 | 33.79 | 0.15 | 200 | 150.0 |
| Llama-3.1-8B-Instruct | sglang | 36.39 | 36.74 | 36.77 | 36.38 | 0.31 | 200 | 150.0 |
| Llama-3.1-8B-Instruct | vllm | 36.55 | 36.59 | 36.59 | 36.50 | 0.28 | 200 | 150.0 |
| Llama-3.2-3B-Instruct | sglang | 16.80 | 17.44 | 17.65 | 16.79 | 0.43 | 200 | 146.6 |
| Llama-3.2-3B-Instruct | vllm | 16.41 | 16.52 | 16.54 | 16.41 | 0.08 | 200 | 146.5 |
| Mistral-7B-Instruct-v0.3 | sglang | 37.72 | 38.78 | 38.91 | 37.38 | 1.13 | 200 | 133.1 |
| Mistral-7B-Instruct-v0.3 | vllm | 35.29 | 35.33 | 35.35 | 35.26 | 0.15 | 200 | 132.8 |
| Phi-3-mini-4k-instruct | sglang | 19.98 | 20.22 | 20.26 | 19.96 | 0.20 | 200 | 150.0 |
| Phi-3-mini-4k-instruct | vllm | 20.02 | 20.05 | 20.06 | 19.97 | 0.21 | 200 | 150.0 |
| Phi-4-mini-instruct | sglang | 22.70 | 23.37 | 23.98 | 22.42 | 0.83 | 200 | 74.3 |
| Phi-4-mini-instruct | vllm | 20.22 | 20.90 | 20.93 | 20.37 | 0.35 | 200 | 111.1 |
| Qwen2.5-7B-Instruct | sglang | 37.46 | 39.59 | 39.73 | 37.30 | 1.30 | 200 | 54.2 |
| Qwen2.5-7B-Instruct | vllm | 34.91 | 35.24 | 35.57 | 34.95 | 0.22 | 200 | 45.9 |
| Qwen3-8B | sglang | 38.07 | 38.30 | 38.31 | 38.04 | 0.34 | 200 | 149.1 |
| Qwen3-8B | vllm | 38.10 | 38.20 | 38.52 | 38.06 | 0.28 | 200 | 148.3 |
| SmolLM3-3B | sglang | 19.41 | 19.85 | 19.98 | 19.34 | 0.47 | 200 | 148.2 |
| SmolLM3-3B | vllm | 17.30 | 17.42 | 17.44 | 17.25 | 0.32 | 200 | 149.6 |
| gemma-2-2b-it | sglang | 15.95 | 16.90 | 17.41 | 15.80 | 0.86 | 200 | 86.8 |
| gemma-2-2b-it | vllm | 13.74 | 13.92 | 13.94 | 13.76 | 0.11 | 200 | 61.6 |
| gemma-2-9b-it | sglang | 50.61 | 52.81 | 53.58 | 50.41 | 1.84 | 200 | 52.8 |
| gemma-2-9b-it | vllm | 48.58 | 48.79 | 48.90 | 48.04 | 0.85 | 200 | 67.0 |
| gemma-3-4b-it | sglang | 24.90 | 26.54 | 26.90 | 24.90 | 1.14 | 198 | 144.7 |
| gemma-3-4b-it | vllm | 44.05 | 44.57 | 44.65 | 43.99 | 0.41 | 200 | 75.3 |
| granite-3.3-8b-instruct | sglang | 41.13 | 41.67 | 41.83 | 41.11 | 0.40 | 200 | 145.1 |
| granite-3.3-8b-instruct | vllm | 40.80 | 41.58 | 41.75 | 40.93 | 0.39 | 200 | 114.9 |

## Engine Comparison (All Scenarios Aggregated)

Aggregates TPOT samples from all scenarios for a high-level engine comparison.

| Model | Engine | Weighted Mean TPOT (ms) | Total Requests |
|-------|--------|------------------------|----------------|
| DeepSeek-R1-Distill-Llama-8B | sglang | 35.80 | 1070 |
| DeepSeek-R1-Distill-Llama-8B | vllm | 35.80 | 1070 |
| DeepSeek-R1-Distill-Qwen-7B | sglang | 33.49 | 1070 |
| DeepSeek-R1-Distill-Qwen-7B | vllm | 33.41 | 1070 |
| Llama-3.1-8B-Instruct | sglang | 50.47 | 2570 |
| Llama-3.1-8B-Instruct | vllm | 40.53 | 2570 |
| Llama-3.2-3B-Instruct | sglang | 16.50 | 1070 |
| Llama-3.2-3B-Instruct | vllm | 16.46 | 1070 |
| Mistral-7B-Instruct-v0.3 | sglang | 34.88 | 1069 |
| Mistral-7B-Instruct-v0.3 | vllm | 34.35 | 1070 |
| Phi-3-mini-4k-instruct | sglang | 20.25 | 1070 |
| Phi-3-mini-4k-instruct | vllm | 20.26 | 1070 |
| Phi-4-mini-instruct | sglang | 22.53 | 1070 |
| Phi-4-mini-instruct | vllm | 20.31 | 1070 |
| Qwen2.5-7B-Instruct | sglang | 34.20 | 1070 |
| Qwen2.5-7B-Instruct | vllm | 33.95 | 1070 |
| Qwen3-8B | sglang | 60.68 | 1820 |
| Qwen3-8B | vllm | 38.92 | 1820 |
| SmolLM3-3B | sglang | 21.71 | 1070 |
| SmolLM3-3B | vllm | 16.65 | 1070 |
| gemma-2-2b-it | sglang | 14.80 | 1070 |
| gemma-2-2b-it | vllm | 13.79 | 1070 |
| gemma-2-9b-it | sglang | 46.90 | 1068 |
| gemma-2-9b-it | vllm | 47.00 | 1070 |
| gemma-3-4b-it | sglang | 29.05 | 1067 |
| gemma-3-4b-it | vllm | 42.64 | 1070 |
| granite-3.3-8b-instruct | sglang | 40.05 | 1070 |
| granite-3.3-8b-instruct | vllm | 39.84 | 1070 |
