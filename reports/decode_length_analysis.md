# Decode-Length Sweep Analysis

Fixed ~512-token prompts, `max_output_tokens` swept across [64, 256, 1024, 4096]. Separates prefill-bound from decode-bound behaviour. Values show mean ± 95% CI across iterations.

## Llama-3.1-8B-Instruct

### Throughput (tok/s)

| max_tokens | vLLM tok/s | SGLang tok/s | Winner |
|-----------|-----------|-------------|--------|
| 64 | 189.2 ± 0.5 | 191.9 ± 0.3 | Tie |
| 256 | 189.4 ± 0.1 | 190.3 ± 0.2 | Tie |
| 1024 | 185.1 ± 0.6 | 186.5 ± 1.0 | Tie |
| 4096 | 158.5 ± 1.3 | 157.6 ± 6.1 | Tie |

### TTFT P50 (ms)

| max_tokens | vLLM TTFT | SGLang TTFT |
|-----------|-----------|-------------|
| 64 | 96.2 ± 6.7 ms | 69.1 ± 0.8 ms |
| 256 | 93.3 ± 0.9 ms | 69.7 ± 1.3 ms |
| 1024 | 96.3 ± 2.3 ms | 69.1 ± 0.7 ms |
| 4096 | 113.6 ± 10.5 ms | 106.4 ± 1.2 ms |

### TPOT P50 (ms/token)

| max_tokens | vLLM TPOT | SGLang TPOT |
|-----------|-----------|-------------|
| 64 | 35.0 ± 0.0 ms | 34.8 ± 0.1 ms |
| 256 | 35.2 ± 0.0 ms | 35.0 ± 0.1 ms |
| 1024 | 36.1 ± 0.0 ms | 35.9 ± 0.1 ms |
| 4096 | 39.3 ± 0.2 ms | 38.9 ± 0.0 ms |

### Findings

- **Throughput crossover:** sglang → vllm at max_tokens=4096
- **TTFT crossover:** no crossover — vllm leads throughout (gap 6.3% at max_tokens=4096)
- **TTFT at max_tokens=4096:** SGLang still leads by 6.3% — TTFT advantage is preserved at long decode.

## Phi-4-mini-instruct

### Throughput (tok/s)

| max_tokens | vLLM tok/s | SGLang tok/s | Winner |
|-----------|-----------|-------------|--------|
| 64 | 354.4 ± 0.2 | 340.1 ± 1.9 | vLLM +4% |
| 256 | 346.2 ± 0.9 | 333.4 ± 0.1 | vLLM +4% |
| 1024 | 304.7 ± 9.7 | 322.6 ± 1.9 | SGLang +6% |
| 4096 | 287.2 ± 34.4 | 293.5 ± 35.1 | SGLang +2% |

### TTFT P50 (ms)

| max_tokens | vLLM TTFT | SGLang TTFT |
|-----------|-----------|-------------|
| 64 | 55.1 ± 1.0 ms | 49.2 ± 2.2 ms |
| 256 | 56.0 ± 1.7 ms | 46.8 ± 0.7 ms |
| 1024 | 56.4 ± 1.0 ms | 47.6 ± 3.7 ms |
| 4096 | 56.8 ± 2.0 ms | 48.3 ± 23.7 ms |

### TPOT P50 (ms/token)

| max_tokens | vLLM TPOT | SGLang TPOT |
|-----------|-----------|-------------|
| 64 | 19.0 ± 0.0 ms | 19.9 ± 0.0 ms |
| 256 | 19.4 ± 0.0 ms | 20.1 ± 0.0 ms |
| 1024 | 20.7 ± 0.2 ms | 20.9 ± 0.0 ms |
| 4096 | 20.8 ± 0.5 ms | 23.0 ± 4.8 ms |

### Findings

- **Throughput crossover:** vllm → sglang at max_tokens=1024
- **TTFT crossover:** no crossover — vllm leads throughout (gap 15.1% at max_tokens=4096)
- **TTFT at max_tokens=4096:** SGLang still leads by 15.1% — TTFT advantage is preserved at long decode.

## gemma-2-2b-it

### Throughput (tok/s)

| max_tokens | vLLM tok/s | SGLang tok/s | Winner |
|-----------|-----------|-------------|--------|
| 64 | 523.0 ± 34.6 | 519.1 ± 14.0 | Tie |
| 256 | 493.8 ± 17.5 | 484.3 ± 1.8 | Tie |
| 1024 | 458.0 ± 11.5 | 469.7 ± 20.3 | SGLang +2% |
| 4096 | 459.2 ± 26.2 | 467.0 ± 9.3 | Tie |

### TTFT P50 (ms)

| max_tokens | vLLM TTFT | SGLang TTFT |
|-----------|-----------|-------------|
| 64 | 42.1 ± 2.1 ms | 39.4 ± 3.0 ms |
| 256 | 36.5 ± 5.2 ms | 41.9 ± 6.2 ms |
| 1024 | 37.3 ± 0.2 ms | 37.9 ± 0.2 ms |
| 4096 | 37.5 ± 0.3 ms | 37.9 ± 0.5 ms |

### TPOT P50 (ms/token)

| max_tokens | vLLM TPOT | SGLang TPOT |
|-----------|-----------|-------------|
| 64 | 12.9 ± 0.0 ms | 13.0 ± 0.0 ms |
| 256 | 13.2 ± 0.0 ms | 13.4 ± 0.2 ms |
| 1024 | 13.8 ± 0.0 ms | 13.7 ± 0.0 ms |
| 4096 | 13.9 ± 0.1 ms | 13.7 ± 0.0 ms |

### Findings

- **Throughput crossover:** vllm → sglang at max_tokens=1024
- **TTFT crossover:** vllm → sglang at max_tokens=256
- **TTFT at max_tokens=4096:** converged (gap 1.0%) — TTFT advantage does not persist at long decode.

## gemma-3-4b-it

### Throughput (tok/s)

| max_tokens | vLLM tok/s | SGLang tok/s | Winner |
|-----------|-----------|-------------|--------|
| 64 | 146.3 ± 23.1 | 280.8 ± 2.4 | SGLang +48% |
| 256 | 156.7 ± 5.2 | 289.0 ± 2.3 | SGLang +46% |
| 1024 | 152.7 ± 5.0 | 274.9 ± 3.3 | SGLang +44% |
| 4096 | 149.4 ± 9.5 | 269.3 ± 16.5 | SGLang +45% |

### TTFT P50 (ms)

| max_tokens | vLLM TTFT | SGLang TTFT |
|-----------|-----------|-------------|
| 64 | 128.2 ± 1.6 ms | 128.8 ± 11.6 ms |
| 256 | 126.8 ± 1.8 ms | 126.3 ± 0.6 ms |
| 1024 | 122.6 ± 0.2 ms | 100.1 ± 2.6 ms |
| 4096 | 123.5 ± 2.9 ms | 100.2 ± 1.2 ms |

### TPOT P50 (ms/token)

| max_tokens | vLLM TPOT | SGLang TPOT |
|-----------|-----------|-------------|
| 64 | 42.1 ± 0.2 ms | 22.4 ± 0.0 ms |
| 256 | 42.6 ± 1.0 ms | 22.7 ± 0.1 ms |
| 1024 | 42.5 ± 0.3 ms | 23.4 ± 0.1 ms |
| 4096 | 42.8 ± 1.6 ms | 23.6 ± 0.1 ms |

### Findings

- **Throughput crossover:** no crossover — sglang leads throughout (gap 44.5% at max_tokens=4096)
- **TTFT crossover:** sglang → vllm at max_tokens=256
- **TTFT at max_tokens=4096:** SGLang still leads by 18.9% — TTFT advantage is preserved at long decode.

## Summary: TTFT Advantage at Long Decode (max_tokens=4096)

Does vLLM's TTFT advantage (seen at concurrency=1) survive high output-token budgets? TTFT is determined at the prefill stage, so it should be independent of max_tokens — any divergence here indicates system-level effects (KV memory pressure, scheduler backpressure).

| Model | vLLM TTFT @4096 | SGLang TTFT @4096 | Preserved? |
|-------|----------------|------------------|------------|
| Llama-3.1-8B-Instruct | 113.6 ± 10.5 ms | 106.4 ± 1.2 ms | Yes |
| Phi-4-mini-instruct | 56.8 ± 2.0 ms | 48.3 ± 23.7 ms | Yes |
| gemma-2-2b-it | 37.5 ± 0.3 ms | 37.9 ± 0.5 ms | No (converged) |
| gemma-3-4b-it | 123.5 ± 2.9 ms | 100.2 ± 1.2 ms | Yes |
