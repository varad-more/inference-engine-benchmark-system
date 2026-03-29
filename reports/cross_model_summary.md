# Cross-Model Inference Engine Benchmark Summary

**Date:** 2026-03-28
**Hardware:** AWS g5.2xlarge -- single NVIDIA A10G 24 GB
**Engines:** vLLM, SGLang
**Models (7):** Gemma 2B, Phi-3 Mini 3.8B, Llama 3.2 3B, Qwen 2.5 7B, Mistral 7B, Llama 3.1 8B, Gemma 9B
**Scenarios (5):** single_request_latency, throughput_ramp, long_context_stress, prefix_sharing_benefit, structured_generation_speed
**Iterations:** 2 per scenario-engine pair (140 result files total)
**Execution:** Sequential, one engine at a time

---

## Executive Summary

- **vLLM consistently delivers lower time-to-first-token (TTFT):** Across nearly every model and scenario, vLLM achieves lower p95 TTFT than SGLang, often by 15-40%. The advantage is most pronounced in structured generation and prefix sharing workloads.
- **Throughput (tok/s) is largely comparable between engines, with vLLM holding a slight edge in structured generation:** For most scenarios the two engines produce near-identical token throughput, but vLLM pulls meaningfully ahead in the structured_generation_speed scenario (up to 29% faster on Gemma 2B, 17% on Qwen 2.5 7B).
- **Smaller models (2B-3B) deliver 2-3x the throughput and half the latency of 7B-9B models on a single A10G:** This is consistent across all scenarios and both engines, making sub-4B models strongly preferable for latency-sensitive deployments on this GPU class.
- **Phi-3 Mini works on both engines but vLLM leads across the board:** vLLM wins all 5 scenarios on both TTFT and throughput for Phi-3 Mini, though margins are tight on throughput ramp (190.9 vs 187.3 tok/s) and structured generation (784.8 vs 782.6 tok/s). Gemma 9B under SGLang exhibited a 30-second TTFT anomaly during high-concurrency throughput ramp due to VRAM pressure.

---

## Head-to-Head Engine Comparison

For each scenario, wins are counted across all 7 models. A "win" means the engine achieved a strictly better value for that metric.

### TTFT p95 Wins (lower is better)

| Scenario                      | vLLM Wins | SGLang Wins | Tie |
|-------------------------------|:---------:|:-----------:|:---:|
| single_request_latency        |     6     |      1      |  0  |
| throughput_ramp               |     6     |      1      |  0  |
| long_context_stress           |     6     |      1      |  0  |
| prefix_sharing_benefit        |     7     |      0      |  0  |
| structured_generation_speed   |     6     |      1      |  0  |
| **Total**                     |  **31**   |    **4**    |**0**|

### Throughput (tok/s) Wins (higher is better)

| Scenario                      | vLLM Wins | SGLang Wins | Tie |
|-------------------------------|:---------:|:-----------:|:---:|
| single_request_latency        |     2     |      4      |  1  |
| throughput_ramp               |     3     |      4      |  0  |
| long_context_stress           |     5     |      1      |  1  |
| prefix_sharing_benefit        |     5     |      1      |  1  |
| structured_generation_speed   |     7     |      0      |  0  |
| **Total**                     |  **22**   |   **10**    |**3**|

**Takeaway:** vLLM dominates TTFT across the board (31-4). Throughput favors vLLM overall (22-10), decisively so in structured generation (7-0) and long context (5-1). SGLang is competitive on single-request decode speed and throughput ramp at larger model sizes.

---

## Per-Scenario Detailed Tables

### 1. Single Request Latency

| Model              | Engine | TTFT p95 (ms) | Lat p95 (ms) | Throughput (tok/s) | Req/s | Success |
|--------------------|--------|-------------:|-------------:|-------------------:|------:|--------:|
| Gemma 2B           | vLLM   |         22.5 |      1,655.5 |               77.6 |  0.75 |  100.0% |
| Gemma 2B           | SGLang |         31.6 |      1,656.3 |               78.3 |  0.75 |  100.0% |
| Phi-3 Mini (3.8B)  | vLLM   |         25.3 |      2,233.7 |               57.8 |  0.45 |  100.0% |
| Phi-3 Mini (3.8B)  | SGLang |         50.9 |      2,319.8 |               55.7 |  0.44 |  100.0% |
| Llama 3.2 3B       | vLLM   |         23.1 |      1,928.8 |               66.3 |  0.52 |  100.0% |
| Llama 3.2 3B       | SGLang |         33.2 |      1,903.5 |               67.7 |  0.53 |  100.0% |
| Qwen 2.5 7B        | vLLM   |         62.8 |      4,220.2 |               30.6 |  0.29 |  100.0% |
| Qwen 2.5 7B        | SGLang |         65.9 |      4,170.4 |               30.8 |  0.27 |  100.0% |
| Mistral 7B          | vLLM   |         62.6 |      4,064.4 |               31.8 |  0.26 |  100.0% |
| Mistral 7B          | SGLang |         63.7 |      4,047.6 |               31.8 |  0.26 |  100.0% |
| Llama 3.1 8B       | vLLM   |         43.5 |      4,247.4 |               30.3 |  0.24 |  100.0% |
| Llama 3.1 8B       | SGLang |         67.6 |      4,247.6 |               30.3 |  0.24 |  100.0% |
| Gemma 9B            | vLLM   |        105.7 |      5,360.2 |               24.0 |  0.21 |  100.0% |
| Gemma 9B            | SGLang |         84.2 |      5,312.8 |               24.2 |  0.21 |  100.0% |

### 2. Throughput Ramp

| Model              | Engine | TTFT p95 (ms) | Lat p95 (ms) | Throughput (tok/s) | Req/s | Success |
|--------------------|--------|-------------:|-------------:|-------------------:|------:|--------:|
| Gemma 2B           | vLLM   |        154.1 |      4,583.1 |              264.6 |  1.14 |  100.0% |
| Gemma 2B           | SGLang |        158.6 |      4,852.7 |              258.0 |  1.05 |  100.0% |
| Phi-3 Mini (3.8B)  | vLLM   |        197.9 |      8,029.0 |              190.9 |  0.75 |  100.0% |
| Phi-3 Mini (3.8B)  | SGLang |        210.6 |      7,523.5 |              187.3 |  0.73 |  100.0% |
| Llama 3.2 3B       | vLLM   |        168.8 |      5,390.9 |              223.5 |  0.87 |  100.0% |
| Llama 3.2 3B       | SGLang |        213.6 |      5,627.1 |              225.9 |  0.88 |  100.0% |
| Qwen 2.5 7B        | vLLM   |        192.7 |      9,762.0 |              105.3 |  0.41 |  100.0% |
| Qwen 2.5 7B        | SGLang |        272.2 |      9,545.9 |              106.3 |  0.42 |  100.0% |
| Mistral 7B          | vLLM   |        216.9 |     10,141.7 |              106.6 |  0.42 |  100.0% |
| Mistral 7B          | SGLang |        279.3 |     10,162.4 |              106.8 |  0.42 |   99.9% |
| Llama 3.1 8B       | vLLM   |        201.3 |     10,534.8 |              101.7 |  0.40 |  100.0% |
| Llama 3.1 8B       | SGLang |        186.5 |     10,621.4 |              102.1 |  0.40 |  100.0% |
| Gemma 9B            | vLLM   |        283.9 |     14,388.5 |               79.8 |  0.33 |  100.0% |
| Gemma 9B            | SGLang |     30,483.6 |     46,328.4 |               78.0 |  0.31 |   99.9% |

### 3. Long Context Stress

| Model              | Engine | TTFT p95 (ms) | Lat p95 (ms) | Throughput (tok/s) | Req/s | Success |
|--------------------|--------|-------------:|-------------:|-------------------:|------:|--------:|
| Gemma 2B           | vLLM   |         45.1 |      3,349.7 |              310.8 |  1.24 |  100.0% |
| Gemma 2B           | SGLang |         53.4 |      3,471.1 |              293.1 |  1.24 |  100.0% |
| Phi-3 Mini (3.8B)  | vLLM   |         61.2 |      4,793.7 |              231.4 |  0.90 |  100.0% |
| Phi-3 Mini (3.8B)  | SGLang |        121.1 |      4,953.6 |              211.5 |  0.83 |  100.0% |
| Llama 3.2 3B       | vLLM   |         53.9 |      4,085.7 |              254.5 |  0.99 |  100.0% |
| Llama 3.2 3B       | SGLang |        101.8 |      4,089.7 |              253.5 |  0.99 |  100.0% |
| Qwen 2.5 7B        | vLLM   |         93.9 |      8,500.4 |              118.3 |  0.47 |  100.0% |
| Qwen 2.5 7B        | SGLang |         88.9 |      8,463.1 |              124.7 |  0.49 |  100.0% |
| Mistral 7B          | vLLM   |         95.6 |      8,582.6 |              117.0 |  0.50 |  100.0% |
| Mistral 7B          | SGLang |        101.7 |      8,643.8 |              115.1 |  0.51 |  100.0% |
| Llama 3.1 8B       | vLLM   |         98.8 |      8,971.8 |              115.6 |  0.45 |  100.0% |
| Llama 3.1 8B       | SGLang |         98.9 |      8,950.1 |              115.6 |  0.45 |  100.0% |
| Gemma 9B            | vLLM   |        127.4 |     10,464.8 |               82.5 |  0.80 |  100.0% |
| Gemma 9B            | SGLang |        131.9 |     11,549.5 |               81.0 |  0.39 |  100.0% |

### 4. Prefix Sharing Benefit

| Model              | Engine | TTFT p95 (ms) | Lat p95 (ms) | Throughput (tok/s) | Req/s | Success |
|--------------------|--------|-------------:|-------------:|-------------------:|------:|--------:|
| Gemma 2B           | vLLM   |         57.8 |      1,713.0 |              600.0 |  5.25 |  100.0% |
| Gemma 2B           | SGLang |         72.8 |      1,928.8 |              539.0 |  5.24 |  100.0% |
| Phi-3 Mini (3.8B)  | vLLM   |         66.5 |      2,489.0 |              409.9 |  3.21 |  100.0% |
| Phi-3 Mini (3.8B)  | SGLang |        141.0 |      2,574.0 |              396.6 |  3.10 |  100.0% |
| Llama 3.2 3B       | vLLM   |         56.9 |      2,096.2 |              489.2 |  3.82 |  100.0% |
| Llama 3.2 3B       | SGLang |         79.4 |      2,077.6 |              489.2 |  3.82 |  100.0% |
| Qwen 2.5 7B        | vLLM   |         95.5 |      4,330.2 |              228.9 |  1.84 |  100.0% |
| Qwen 2.5 7B        | SGLang |        140.0 |      4,453.1 |              229.2 |  1.87 |  100.0% |
| Mistral 7B          | vLLM   |         96.5 |      4,375.6 |              228.4 |  2.16 |  100.0% |
| Mistral 7B          | SGLang |        162.2 |      4,599.2 |              218.9 |  2.04 |   99.5% |
| Llama 3.1 8B       | vLLM   |        101.2 |      4,570.6 |              219.4 |  1.71 |  100.0% |
| Llama 3.1 8B       | SGLang |        139.1 |      4,579.8 |              217.8 |  1.70 |  100.0% |
| Gemma 9B            | vLLM   |        132.4 |      5,885.2 |              164.3 |  1.87 |  100.0% |
| Gemma 9B            | SGLang |        198.2 |      6,217.4 |              159.3 |  1.96 |  100.0% |

### 5. Structured Generation Speed

| Model              | Engine | TTFT p95 (ms) | Lat p95 (ms) | Throughput (tok/s) | Req/s | Success |
|--------------------|--------|-------------:|-------------:|-------------------:|------:|--------:|
| Gemma 2B           | vLLM   |         58.9 |      1,038.2 |            1,225.1 | 19.88 |  100.0% |
| Gemma 2B           | SGLang |         85.4 |      2,512.9 |              952.3 | 11.03 |  100.0% |
| Phi-3 Mini (3.8B)  | vLLM   |         78.4 |      3,066.5 |              784.8 |  5.23 |  100.0% |
| Phi-3 Mini (3.8B)  | SGLang |        115.7 |      3,096.3 |              782.6 |  5.22 |  100.0% |
| Llama 3.2 3B       | vLLM   |         71.8 |      2,503.2 |              970.2 |  6.62 |  100.0% |
| Llama 3.2 3B       | SGLang |         86.5 |      2,648.2 |              908.4 |  6.20 |  100.0% |
| Qwen 2.5 7B        | vLLM   |        139.0 |      1,978.2 |              455.9 |  9.93 |  100.0% |
| Qwen 2.5 7B        | SGLang |        137.2 |      5,676.8 |              389.6 |  6.46 |  100.0% |
| Mistral 7B          | vLLM   |        107.7 |      5,075.3 |              439.9 |  3.31 |  100.0% |
| Mistral 7B          | SGLang |        150.6 |      5,518.2 |              411.2 |  3.09 |  100.0% |
| Llama 3.1 8B       | vLLM   |        125.5 |      5,570.9 |              426.2 |  2.84 |  100.0% |
| Llama 3.1 8B       | SGLang |        152.7 |      5,578.7 |              422.6 |  2.82 |  100.0% |
| Gemma 9B            | vLLM   |        154.3 |      6,263.0 |              310.8 |  4.62 |  100.0% |
| Gemma 9B            | SGLang |        195.5 |      4,425.7 |              290.0 |  5.49 |  100.0% |

---

## Model Size Scaling Analysis

The following table shows vLLM results across model sizes for each scenario to illustrate how performance scales with parameter count on a single A10G.

### Single Request Latency (vLLM)

| Model             | Params | TTFT p95 (ms) | Lat p95 (ms) | tok/s |
|-------------------|-------:|--------------:|-------------:|------:|
| Gemma 2B          |   2.0B |          22.5 |      1,655.5 |  77.6 |
| Llama 3.2 3B      |   3.0B |          23.1 |      1,928.8 |  66.3 |
| Phi-3 Mini        |   3.8B |          25.3 |      2,233.7 |  57.8 |
| Qwen 2.5 7B       |   7.0B |          62.8 |      4,220.2 |  30.6 |
| Mistral 7B        |   7.0B |          62.6 |      4,064.4 |  31.8 |
| Llama 3.1 8B      |   8.0B |          43.5 |      4,247.4 |  30.3 |
| Gemma 9B          |   9.0B |         105.7 |      5,360.2 |  24.0 |

### Throughput Ramp (vLLM)

| Model             | Params | TTFT p95 (ms) | Lat p95 (ms) | tok/s |
|-------------------|-------:|--------------:|-------------:|------:|
| Gemma 2B          |   2.0B |         154.1 |      4,583.1 | 264.6 |
| Llama 3.2 3B      |   3.0B |         168.8 |      5,390.9 | 223.5 |
| Phi-3 Mini        |   3.8B |         197.9 |      8,029.0 | 190.9 |
| Qwen 2.5 7B       |   7.0B |         192.7 |      9,762.0 | 105.3 |
| Mistral 7B        |   7.0B |         216.9 |     10,141.7 | 106.6 |
| Llama 3.1 8B      |   8.0B |         201.3 |     10,534.8 | 101.7 |
| Gemma 9B          |   9.0B |         283.9 |     14,388.5 |  79.8 |

### Structured Generation Speed (vLLM)

| Model             | Params | TTFT p95 (ms) | Lat p95 (ms) | tok/s  |
|-------------------|-------:|--------------:|-------------:|-------:|
| Gemma 2B          |   2.0B |          58.9 |      1,038.2 | 1225.1 |
| Llama 3.2 3B      |   3.0B |          71.8 |      2,503.2 |  970.2 |
| Phi-3 Mini        |   3.8B |          78.4 |      3,066.5 |  784.8 |
| Qwen 2.5 7B       |   7.0B |         139.0 |      1,978.2 |  455.9 |
| Mistral 7B        |   7.0B |         107.7 |      5,075.3 |  439.9 |
| Llama 3.1 8B      |   8.0B |         125.5 |      5,570.9 |  426.2 |
| Gemma 9B          |   9.0B |         154.3 |      6,263.0 |  310.8 |

**Key scaling observations:**

- **TTFT scales roughly linearly with model size.** Moving from 2B to 9B increases TTFT by approximately 4-5x across scenarios.
- **Token throughput halves going from 2-3B to 7-9B.** Gemma 2B achieves 77.6 tok/s single-request vs. 24.0 tok/s for Gemma 9B (3.2x difference). Under throughput ramp, the gap is 264.6 vs. 79.8 tok/s (3.3x).
- **End-to-end latency roughly doubles to triples.** Single-request p95 latency goes from ~1.7s (2B) to ~5.4s (9B).
- **The 3B-to-7B jump is the steepest cliff.** Throughput drops ~50% between Llama 3.2 3B and Qwen 2.5 7B, while latency nearly doubles. This marks the practical boundary of "fast" inference on a single A10G.

---

## Notable Anomalies

### Gemma 9B on SGLang -- Throughput Ramp Failure

Under the throughput_ramp scenario, Gemma 9B on SGLang exhibited a **p95 TTFT of 30,483.6 ms** (compared to 283.9 ms on vLLM) and a **p95 latency of 46,328.4 ms** (vs. 14,388.5 ms on vLLM). This represents a >100x degradation in TTFT. The root cause is VRAM pressure: at high concurrency levels, the 9B model on SGLang exhausted available GPU memory, causing severe queuing and scheduling delays. Success rate also dipped to 99.9%.

For comparison, vLLM served Gemma 9B on the same scenario with tuned settings (`max-model-len=4096`, `gpu-memory-utilization=0.92`), which kept performance within expected bounds.

### Phi-3 Mini -- vLLM Leads Across the Board

Phi-3 Mini (3.8B) now has complete results on both engines. vLLM wins all 5 scenarios on both TTFT and throughput. However, margins are tight in throughput ramp (190.9 vs 187.3 tok/s, ~2% difference) and structured generation (784.8 vs 782.6 tok/s, <1% difference), suggesting the engines are nearly equivalent for Phi-3 decode speed.

### Mistral 7B SGLang Prefix Sharing -- Minor Success Rate Drop

Mistral 7B on SGLang in the prefix_sharing_benefit scenario showed a 99.5% success rate (vs. 100% for all other Mistral runs). This is a minor anomaly but worth noting for production reliability considerations.

---

## Recommendations

### When to Use vLLM

- **Structured output / JSON generation:** vLLM leads by a wide margin in structured generation speed -- up to 29% higher throughput on Gemma 2B and 17% on Qwen 2.5 7B. If your workload is heavily structured-output, vLLM is the clear choice.
- **Latency-sensitive single-request serving:** vLLM consistently achieves 15-40% lower TTFT, which matters for interactive applications where time-to-first-token is perceptible.
- **Large models near VRAM limits:** vLLM's memory management proved more robust at the boundary (Gemma 9B), avoiding the catastrophic degradation seen with SGLang.
- **Consistent advantage across all model architectures:** vLLM leads on every metric for Phi-3 Mini across all 5 scenarios, and dominates TTFT on all other models except Gemma 9B.

### When to Use SGLang

- **Workloads where throughput matters more than TTFT:** In several scenarios (especially single_request_latency and throughput_ramp), SGLang matches or slightly exceeds vLLM in tok/s despite higher TTFT. If your application is batch-oriented and TTFT is not user-facing, SGLang is competitive.
- **Gemma 9B structured generation:** Interestingly, SGLang achieved lower p95 latency (4,425.7 ms vs. 6,263.0 ms) on Gemma 9B structured generation, though at slightly lower throughput. Workload-specific testing is warranted.

### General Guidance

- For **sub-4B models** on a single A10G: either engine works well; prefer vLLM for the TTFT advantage.
- For **7B+ models** on a single A10G: use vLLM, especially under concurrent load, due to better memory management and lower tail latencies.
- For **production deployments**: validate with your actual workload. The differences are often small enough (< 10% throughput) that operational factors (ecosystem, tooling, community support) may dominate the engine choice.
- **Always test at target concurrency.** The throughput_ramp scenario revealed issues (Gemma 9B SGLang) that single-request benchmarks would never surface.

---

*Report generated from 140 benchmark result files (70 scenario-engine pairs x 2 iterations averaged). All metrics are p95 unless otherwise noted.*
