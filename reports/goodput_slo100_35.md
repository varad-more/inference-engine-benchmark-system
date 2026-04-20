# Goodput Analysis

**SLO thresholds:** TTFT ≤ 100 ms, TPOT ≤ 35.0 ms

**Goodput** = qualifying requests (meeting both SLOs) ÷ total wall-clock time. Higher is better.

## Aggregate Goodput per (Model, Engine)

| Model | Engine | Goodput (req/s) | SLO Pass Rate | Qualifying | Total Successful | Wall Time (s) |
|-------|--------|-----------------|---------------|-----------|-----------------|---------------|
| DeepSeek-R1-Distill-Llama-8B | sglang | 0.2387 | 47.8% | 511 | 1070 | 2140.6 |
| DeepSeek-R1-Distill-Llama-8B | vllm | 0.1739 | 34.9% | 373 | 1070 | 2144.5 |
| DeepSeek-R1-Distill-Qwen-7B | sglang | 0.3567 | 68.6% | 734 | 1070 | 2058.0 |
| DeepSeek-R1-Distill-Qwen-7B | vllm | 0.3830 | 73.0% | 781 | 1070 | 2039.0 |
| Llama-3.1-8B-Instruct | sglang | 0.2084 | 35.5% | 912 | 2570 | 4377.0 |
| Llama-3.1-8B-Instruct | vllm | 0.0660 | 14.5% | 372 | 2570 | 5637.3 |
| Llama-3.2-3B-Instruct | sglang | 0.9452 | 85.3% | 913 | 1070 | 966.0 |
| Llama-3.2-3B-Instruct | vllm | 0.9990 | 91.0% | 974 | 1070 | 975.0 |
| Mistral-7B-Instruct-v0.3 | sglang | 0.2529 | 47.9% | 512 | 1069 | 2024.1 |
| Mistral-7B-Instruct-v0.3 | vllm | 0.2843 | 53.7% | 575 | 1070 | 2022.5 |
| Phi-3-mini-4k-instruct | sglang | 0.7610 | 83.0% | 888 | 1070 | 1166.9 |
| Phi-3-mini-4k-instruct | vllm | 0.8275 | 88.3% | 945 | 1070 | 1142.0 |
| Phi-4-mini-instruct | sglang | 0.8686 | 85.6% | 916 | 1070 | 1054.5 |
| Phi-4-mini-instruct | vllm | 0.8426 | 88.4% | 946 | 1070 | 1122.8 |
| Qwen2.5-7B-Instruct | sglang | 0.3309 | 61.7% | 660 | 1070 | 1994.3 |
| Qwen2.5-7B-Instruct | vllm | 0.3802 | 70.7% | 757 | 1070 | 1991.2 |
| Qwen3-8B | sglang | 0.0591 | 8.2% | 149 | 1820 | 2522.3 |
| Qwen3-8B | vllm | 0.0351 | 8.2% | 149 | 1820 | 4243.8 |
| SmolLM3-3B | sglang | 0.6010 | 59.6% | 638 | 1070 | 1061.5 |
| SmolLM3-3B | vllm | 1.0566 | 93.2% | 997 | 1070 | 943.6 |
| gemma-2-2b-it | sglang | 1.2262 | 90.0% | 963 | 1070 | 785.4 |
| gemma-2-2b-it | vllm | 1.3708 | 93.1% | 996 | 1070 | 726.6 |
| gemma-2-9b-it | sglang | 0.0000 | 0.0% | 0 | 1068 | 2674.4 |
| gemma-2-9b-it | vllm | 0.0000 | 0.0% | 0 | 1070 | 2470.2 |
| gemma-3-4b-it | sglang | 0.3113 | 42.0% | 448 | 1067 | 1439.0 |
| gemma-3-4b-it | vllm | 0.0000 | 0.0% | 0 | 1070 | 2535.2 |
| granite-3.3-8b-instruct | sglang | 0.0000 | 0.0% | 0 | 1070 | 2353.2 |
| granite-3.3-8b-instruct | vllm | 0.0000 | 0.0% | 0 | 1070 | 2338.0 |

## Per-Scenario Breakdown

### Single Request Latency

| Model | Engine | Goodput (req/s) | SLO Pass Rate | Qualifying | Total | Wall Time (s) |
|-------|--------|-----------------|---------------|-----------|-------|---------------|
| DeepSeek-R1-Distill-Llama-8B | sglang | 0.2367 | 100.0% | 50 | 50 | 211.3 |
| DeepSeek-R1-Distill-Llama-8B | vllm | 0.2322 | 98.0% | 49 | 50 | 211.1 |
| DeepSeek-R1-Distill-Qwen-7B | sglang | 0.2412 | 100.0% | 50 | 50 | 207.3 |
| DeepSeek-R1-Distill-Qwen-7B | vllm | 0.2793 | 100.0% | 50 | 50 | 179.0 |
| Llama-3.1-8B-Instruct | sglang | 0.2170 | 66.7% | 100 | 150 | 460.9 |
| Llama-3.1-8B-Instruct | vllm | 0.0843 | 33.3% | 50 | 150 | 593.3 |
| Llama-3.2-3B-Instruct | sglang | 0.5291 | 100.0% | 50 | 50 | 94.5 |
| Llama-3.2-3B-Instruct | vllm | 0.5177 | 100.0% | 50 | 50 | 96.6 |
| Mistral-7B-Instruct-v0.3 | sglang | 0.2565 | 100.0% | 50 | 50 | 194.9 |
| Mistral-7B-Instruct-v0.3 | vllm | 0.2562 | 100.0% | 50 | 50 | 195.1 |
| Phi-3-mini-4k-instruct | sglang | 0.4351 | 100.0% | 50 | 50 | 114.9 |
| Phi-3-mini-4k-instruct | vllm | 0.4518 | 100.0% | 50 | 50 | 110.7 |
| Phi-4-mini-instruct | sglang | 0.5153 | 98.0% | 49 | 50 | 95.1 |
| Phi-4-mini-instruct | vllm | 0.5524 | 100.0% | 50 | 50 | 90.5 |
| Qwen2.5-7B-Instruct | sglang | 0.2679 | 100.0% | 50 | 50 | 186.6 |
| Qwen2.5-7B-Instruct | vllm | 0.2890 | 100.0% | 50 | 50 | 173.0 |
| Qwen3-8B | sglang | 0.1927 | 49.0% | 49 | 100 | 254.3 |
| Qwen3-8B | vllm | 0.1195 | 49.0% | 49 | 100 | 410.1 |
| SmolLM3-3B | sglang | 0.4855 | 98.0% | 49 | 50 | 100.9 |
| SmolLM3-3B | vllm | 0.5297 | 98.0% | 49 | 50 | 92.5 |
| gemma-2-2b-it | sglang | 0.7524 | 100.0% | 50 | 50 | 66.5 |
| gemma-2-2b-it | vllm | 0.7530 | 100.0% | 50 | 50 | 66.4 |
| gemma-2-9b-it | sglang | 0.0000 | 0.0% | 0 | 50 | 240.5 |
| gemma-2-9b-it | vllm | 0.0000 | 0.0% | 0 | 50 | 241.5 |
| gemma-3-4b-it | sglang | 0.3998 | 100.0% | 50 | 50 | 125.1 |
| gemma-3-4b-it | vllm | 0.0000 | 0.0% | 0 | 50 | 243.3 |
| granite-3.3-8b-instruct | sglang | 0.0000 | 0.0% | 0 | 50 | 232.0 |
| granite-3.3-8b-instruct | vllm | 0.0000 | 0.0% | 0 | 50 | 231.2 |

### Throughput Ramp

| Model | Engine | Goodput (req/s) | SLO Pass Rate | Qualifying | Total | Wall Time (s) |
|-------|--------|-----------------|---------------|-----------|-------|---------------|
| DeepSeek-R1-Distill-Llama-8B | sglang | 0.2003 | 50.3% | 352 | 700 | 1757.2 |
| DeepSeek-R1-Distill-Llama-8B | vllm | 0.1704 | 42.9% | 300 | 700 | 1760.6 |
| DeepSeek-R1-Distill-Qwen-7B | sglang | 0.2658 | 64.1% | 449 | 700 | 1689.0 |
| DeepSeek-R1-Distill-Qwen-7B | vllm | 0.2587 | 62.7% | 439 | 700 | 1697.1 |
| Llama-3.1-8B-Instruct | sglang | 0.1908 | 34.0% | 714 | 2100 | 3742.2 |
| Llama-3.1-8B-Instruct | vllm | 0.0614 | 14.2% | 299 | 2100 | 4871.0 |
| Llama-3.2-3B-Instruct | sglang | 0.6874 | 77.9% | 545 | 700 | 792.9 |
| Llama-3.2-3B-Instruct | vllm | 0.7532 | 86.3% | 604 | 700 | 801.9 |
| Mistral-7B-Instruct-v0.3 | sglang | 0.2405 | 57.7% | 403 | 699 | 1675.3 |
| Mistral-7B-Instruct-v0.3 | vllm | 0.2374 | 57.0% | 399 | 700 | 1680.5 |
| Phi-3-mini-4k-instruct | sglang | 0.5506 | 75.3% | 527 | 700 | 957.2 |
| Phi-3-mini-4k-instruct | vllm | 0.6125 | 82.1% | 575 | 700 | 938.8 |
| Phi-4-mini-instruct | sglang | 0.6389 | 80.6% | 564 | 700 | 882.7 |
| Phi-4-mini-instruct | vllm | 0.6065 | 82.3% | 576 | 700 | 949.8 |
| Qwen2.5-7B-Instruct | sglang | 0.2846 | 68.6% | 480 | 700 | 1686.4 |
| Qwen2.5-7B-Instruct | vllm | 0.2803 | 68.1% | 477 | 700 | 1701.5 |
| Qwen3-8B | sglang | 0.0479 | 7.1% | 100 | 1400 | 2087.8 |
| Qwen3-8B | vllm | 0.0274 | 7.1% | 100 | 1400 | 3652.7 |
| SmolLM3-3B | sglang | 0.5012 | 62.7% | 439 | 700 | 875.8 |
| SmolLM3-3B | vllm | 0.8044 | 89.7% | 628 | 700 | 780.7 |
| gemma-2-2b-it | sglang | 0.8896 | 84.7% | 593 | 700 | 666.6 |
| gemma-2-2b-it | vllm | 1.0197 | 89.4% | 626 | 700 | 613.9 |
| gemma-2-9b-it | sglang | 0.0000 | 0.0% | 0 | 698 | 2295.8 |
| gemma-2-9b-it | vllm | 0.0000 | 0.0% | 0 | 700 | 2108.0 |
| gemma-3-4b-it | sglang | 0.2448 | 42.0% | 294 | 700 | 1200.9 |
| gemma-3-4b-it | vllm | 0.0000 | 0.0% | 0 | 700 | 2137.1 |
| granite-3.3-8b-instruct | sglang | 0.0000 | 0.0% | 0 | 700 | 1931.2 |
| granite-3.3-8b-instruct | vllm | 0.0000 | 0.0% | 0 | 700 | 1931.9 |

### Long Context Stress

| Model | Engine | Goodput (req/s) | SLO Pass Rate | Qualifying | Total | Wall Time (s) |
|-------|--------|-----------------|---------------|-----------|-------|---------------|
| DeepSeek-R1-Distill-Llama-8B | sglang | 0.3829 | 85.0% | 17 | 20 | 44.4 |
| DeepSeek-R1-Distill-Llama-8B | vllm | 0.4522 | 100.0% | 20 | 20 | 44.2 |
| DeepSeek-R1-Distill-Qwen-7B | sglang | 0.4181 | 85.0% | 17 | 20 | 40.7 |
| DeepSeek-R1-Distill-Qwen-7B | vllm | 0.4736 | 100.0% | 20 | 20 | 42.2 |
| Llama-3.1-8B-Instruct | sglang | 0.4515 | 100.0% | 20 | 20 | 44.3 |
| Llama-3.1-8B-Instruct | vllm | 0.4293 | 95.0% | 19 | 20 | 44.3 |
| Llama-3.2-3B-Instruct | sglang | 0.9905 | 100.0% | 20 | 20 | 20.2 |
| Llama-3.2-3B-Instruct | vllm | 0.9941 | 100.0% | 20 | 20 | 20.1 |
| Mistral-7B-Instruct-v0.3 | sglang | 0.4943 | 100.0% | 20 | 20 | 40.5 |
| Mistral-7B-Instruct-v0.3 | vllm | 0.4973 | 100.0% | 20 | 20 | 40.2 |
| Phi-3-mini-4k-instruct | sglang | 0.8267 | 100.0% | 20 | 20 | 24.2 |
| Phi-3-mini-4k-instruct | vllm | 0.9036 | 100.0% | 20 | 20 | 22.1 |
| Phi-4-mini-instruct | sglang | 0.8275 | 100.0% | 20 | 20 | 24.2 |
| Phi-4-mini-instruct | vllm | 0.8282 | 100.0% | 20 | 20 | 24.1 |
| Qwen2.5-7B-Instruct | sglang | 0.4920 | 100.0% | 20 | 20 | 40.6 |
| Qwen2.5-7B-Instruct | vllm | 0.4733 | 100.0% | 20 | 20 | 42.3 |
| Qwen3-8B | sglang | 0.0000 | 0.0% | 0 | 20 | 44.3 |
| Qwen3-8B | vllm | 0.0000 | 0.0% | 0 | 20 | 46.3 |
| SmolLM3-3B | sglang | 0.9921 | 100.0% | 20 | 20 | 20.2 |
| SmolLM3-3B | vllm | 1.1056 | 100.0% | 20 | 20 | 18.1 |
| gemma-2-2b-it | sglang | 1.2417 | 100.0% | 20 | 20 | 16.1 |
| gemma-2-2b-it | vllm | 1.2426 | 100.0% | 20 | 20 | 16.1 |
| gemma-2-9b-it | sglang | 0.0000 | 0.0% | 0 | 20 | 50.7 |
| gemma-2-9b-it | vllm | 0.0000 | 0.0% | 0 | 20 | 26.2 |
| gemma-3-4b-it | sglang | 0.5667 | 80.0% | 16 | 20 | 28.2 |
| gemma-3-4b-it | vllm | 0.0000 | 0.0% | 0 | 20 | 52.3 |
| granite-3.3-8b-instruct | sglang | 0.0000 | 0.0% | 0 | 20 | 48.7 |
| granite-3.3-8b-instruct | vllm | 0.0000 | 0.0% | 0 | 20 | 48.2 |

### Prefix Sharing Benefit

| Model | Engine | Goodput (req/s) | SLO Pass Rate | Qualifying | Total | Wall Time (s) |
|-------|--------|-----------------|---------------|-----------|-------|---------------|
| DeepSeek-R1-Distill-Llama-8B | sglang | 1.6227 | 92.0% | 92 | 100 | 56.7 |
| DeepSeek-R1-Distill-Llama-8B | vllm | 0.0686 | 4.0% | 4 | 100 | 58.3 |
| DeepSeek-R1-Distill-Qwen-7B | sglang | 1.5610 | 85.0% | 85 | 100 | 54.5 |
| DeepSeek-R1-Distill-Qwen-7B | vllm | 1.8048 | 98.0% | 98 | 100 | 54.3 |
| Llama-3.1-8B-Instruct | sglang | 1.2081 | 71.0% | 71 | 100 | 58.8 |
| Llama-3.1-8B-Instruct | vllm | 0.0686 | 4.0% | 4 | 100 | 58.3 |
| Llama-3.2-3B-Instruct | sglang | 3.8228 | 100.0% | 100 | 100 | 26.2 |
| Llama-3.2-3B-Instruct | vllm | 3.8213 | 100.0% | 100 | 100 | 26.2 |
| Mistral-7B-Instruct-v0.3 | sglang | 0.7188 | 35.0% | 35 | 100 | 48.7 |
| Mistral-7B-Instruct-v0.3 | vllm | 2.1170 | 98.0% | 98 | 100 | 46.3 |
| Phi-3-mini-4k-instruct | sglang | 3.0985 | 100.0% | 100 | 100 | 32.3 |
| Phi-3-mini-4k-instruct | vllm | 3.1062 | 100.0% | 100 | 100 | 32.2 |
| Phi-4-mini-instruct | sglang | 3.0636 | 93.0% | 93 | 100 | 30.4 |
| Phi-4-mini-instruct | vllm | 3.5504 | 100.0% | 100 | 100 | 28.2 |
| Qwen2.5-7B-Instruct | sglang | 1.8699 | 98.0% | 98 | 100 | 52.4 |
| Qwen2.5-7B-Instruct | vllm | 1.7852 | 97.0% | 97 | 100 | 54.3 |
| Qwen3-8B | sglang | 0.0000 | 0.0% | 0 | 100 | 60.8 |
| Qwen3-8B | vllm | 0.0000 | 0.0% | 0 | 100 | 60.3 |
| SmolLM3-3B | sglang | 2.9693 | 78.0% | 78 | 100 | 26.3 |
| SmolLM3-3B | vllm | 4.9713 | 100.0% | 100 | 100 | 20.1 |
| gemma-2-2b-it | sglang | 5.5206 | 100.0% | 100 | 100 | 18.1 |
| gemma-2-2b-it | vllm | 4.9699 | 100.0% | 100 | 100 | 20.1 |
| gemma-2-9b-it | sglang | 0.0000 | 0.0% | 0 | 100 | 50.9 |
| gemma-2-9b-it | vllm | 0.0000 | 0.0% | 0 | 100 | 52.4 |
| gemma-3-4b-it | sglang | 0.7804 | 30.3% | 30 | 99 | 38.4 |
| gemma-3-4b-it | vllm | 0.0000 | 0.0% | 0 | 100 | 58.3 |
| granite-3.3-8b-instruct | sglang | 0.0000 | 0.0% | 0 | 100 | 64.6 |
| granite-3.3-8b-instruct | vllm | 0.0000 | 0.0% | 0 | 100 | 64.3 |

### Structured Generation Speed

| Model | Engine | Goodput (req/s) | SLO Pass Rate | Qualifying | Total | Wall Time (s) |
|-------|--------|-----------------|---------------|-----------|-------|---------------|
| DeepSeek-R1-Distill-Llama-8B | sglang | 0.0000 | 0.0% | 0 | 200 | 71.0 |
| DeepSeek-R1-Distill-Llama-8B | vllm | 0.0000 | 0.0% | 0 | 200 | 70.4 |
| DeepSeek-R1-Distill-Qwen-7B | sglang | 1.9981 | 66.5% | 133 | 200 | 66.6 |
| DeepSeek-R1-Distill-Qwen-7B | vllm | 2.6207 | 87.0% | 174 | 200 | 66.4 |
| Llama-3.1-8B-Instruct | sglang | 0.0987 | 3.5% | 7 | 200 | 70.9 |
| Llama-3.1-8B-Instruct | vllm | 0.0000 | 0.0% | 0 | 200 | 70.4 |
| Llama-3.2-3B-Instruct | sglang | 6.1368 | 99.0% | 198 | 200 | 32.3 |
| Llama-3.2-3B-Instruct | vllm | 6.6220 | 100.0% | 200 | 200 | 30.2 |
| Mistral-7B-Instruct-v0.3 | sglang | 0.0618 | 2.0% | 4 | 200 | 64.7 |
| Mistral-7B-Instruct-v0.3 | vllm | 0.1325 | 4.0% | 8 | 200 | 60.4 |
| Phi-3-mini-4k-instruct | sglang | 4.9844 | 95.5% | 191 | 200 | 38.3 |
| Phi-3-mini-4k-instruct | vllm | 5.2323 | 100.0% | 200 | 200 | 38.2 |
| Phi-4-mini-instruct | sglang | 8.5605 | 95.0% | 190 | 200 | 22.2 |
| Phi-4-mini-instruct | vllm | 6.6253 | 100.0% | 200 | 200 | 30.2 |
| Qwen2.5-7B-Instruct | sglang | 0.4253 | 6.0% | 12 | 200 | 28.2 |
| Qwen2.5-7B-Instruct | vllm | 5.6140 | 56.5% | 113 | 200 | 20.1 |
| Qwen3-8B | sglang | 0.0000 | 0.0% | 0 | 200 | 75.0 |
| Qwen3-8B | vllm | 0.0000 | 0.0% | 0 | 200 | 74.4 |
| SmolLM3-3B | sglang | 1.3575 | 26.0% | 52 | 200 | 38.3 |
| SmolLM3-3B | vllm | 6.2156 | 100.0% | 200 | 200 | 32.2 |
| gemma-2-2b-it | sglang | 11.0247 | 100.0% | 200 | 200 | 18.1 |
| gemma-2-2b-it | vllm | 19.8707 | 100.0% | 200 | 200 | 10.1 |
| gemma-2-9b-it | sglang | 0.0000 | 0.0% | 0 | 200 | 36.4 |
| gemma-2-9b-it | vllm | 0.0000 | 0.0% | 0 | 200 | 42.3 |
| gemma-3-4b-it | sglang | 1.2495 | 29.3% | 58 | 198 | 46.4 |
| gemma-3-4b-it | vllm | 0.0000 | 0.0% | 0 | 200 | 44.3 |
| granite-3.3-8b-instruct | sglang | 0.0000 | 0.0% | 0 | 200 | 76.7 |
| granite-3.3-8b-instruct | vllm | 0.0000 | 0.0% | 0 | 200 | 62.3 |
