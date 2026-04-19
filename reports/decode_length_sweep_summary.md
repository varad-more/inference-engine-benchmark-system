# Phase 3: Decode-Length Sweep Summary

Regenerated 2026-04-19 — all cells now at **n=3 iterations** after top-up runs. Values are mean across iterations. Prompt ≈ 512 tokens, concurrency levels {4,8,16}, 60 req/level (180 total).

| Model | Decode tokens | Engine | n | Tokens/s | TTFT p50 (ms) | TTFT p99 (ms) | Latency p50 (ms) | Latency p99 (ms) | Err rate |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| gemma-2-2b-it | 64 | sglang | 3 | 519.1 | 39.4 | 67.9 | 859 | 918 | 0.009 |
| gemma-2-2b-it | 64 | vllm | 3 | 523.0 | 42.1 | 188.7 | 856 | 1108 | 0.000 |
| gemma-2-2b-it | 256 | sglang | 3 | 484.3 | 41.9 | 70.5 | 3458 | 3577 | 0.004 |
| gemma-2-2b-it | 256 | vllm | 3 | 493.8 | 36.5 | 60.4 | 3358 | 3587 | 0.000 |
| gemma-2-2b-it | 1024 | sglang | 3 | 469.7 | 37.9 | 56.3 | 8285 | 12742 | 0.000 |
| gemma-2-2b-it | 1024 | vllm | 3 | 458.0 | 37.3 | 57.4 | 8059 | 12864 | 0.000 |
| gemma-2-2b-it | 4096 | sglang | 3 | 467.0 | 37.9 | 56.7 | 8104 | 11044 | 0.000 |
| gemma-2-2b-it | 4096 | vllm | 3 | 459.2 | 37.5 | 53.7 | 7977 | 12779 | 0.000 |
| gemma-3-4b-it | 64 | sglang | 3 | 280.8 | 128.8 | 155.3 | 1540 | 1598 | 0.006 |
| gemma-3-4b-it | 64 | vllm | 3 | 146.3 | 128.2 | 2827.0 | 2775 | 5758 | 0.000 |
| gemma-3-4b-it | 256 | sglang | 3 | 289.0 | 126.3 | 153.4 | 5920 | 6101 | 0.004 |
| gemma-3-4b-it | 256 | vllm | 3 | 156.7 | 126.8 | 149.5 | 10983 | 11259 | 0.000 |
| gemma-3-4b-it | 1024 | sglang | 3 | 274.9 | 100.1 | 162.9 | 23626 | 25977 | 0.000 |
| gemma-3-4b-it | 1024 | vllm | 3 | 152.7 | 122.6 | 150.2 | 43325 | 45465 | 0.000 |
| gemma-3-4b-it | 4096 | sglang | 3 | 269.3 | 100.2 | 153.5 | 26795 | 36119 | 0.000 |
| gemma-3-4b-it | 4096 | vllm | 3 | 149.4 | 123.5 | 1886.5 | 52738 | 65409 | 0.000 |
| llama-3-1-8b-instruct | 64 | sglang | 3 | 191.9 | 69.1 | 108.9 | 2256 | 2394 | 0.000 |
| llama-3-1-8b-instruct | 64 | vllm | 3 | 189.2 | 96.2 | 128.8 | 2299 | 2417 | 0.000 |
| llama-3-1-8b-instruct | 256 | sglang | 3 | 190.3 | 69.7 | 111.7 | 8997 | 9452 | 0.000 |
| llama-3-1-8b-instruct | 256 | vllm | 3 | 189.4 | 93.3 | 126.2 | 9076 | 9489 | 0.000 |
| llama-3-1-8b-instruct | 1024 | sglang | 3 | 186.5 | 69.1 | 103.6 | 36750 | 39165 | 0.000 |
| llama-3-1-8b-instruct | 1024 | vllm | 3 | 185.1 | 96.3 | 128.7 | 37004 | 39359 | 0.000 |
| llama-3-1-8b-instruct | 4096 | sglang | 3 | 157.6 | 106.4 | 99231.8 | 159669 | 301590 | 0.030 |
| llama-3-1-8b-instruct | 4096 | vllm | 3 | 158.5 | 113.6 | 36139.6 | 160926 | 283530 | 0.000 |
| phi-4-mini-instruct | 64 | sglang | 3 | 340.1 | 49.2 | 105.2 | 1305 | 1378 | 0.000 |
| phi-4-mini-instruct | 64 | vllm | 3 | 354.4 | 55.1 | 82.7 | 1255 | 1321 | 0.000 |
| phi-4-mini-instruct | 256 | sglang | 3 | 333.4 | 46.8 | 76.2 | 5177 | 5350 | 0.000 |
| phi-4-mini-instruct | 256 | vllm | 3 | 346.2 | 56.0 | 70.5 | 4997 | 5269 | 0.000 |
| phi-4-mini-instruct | 1024 | sglang | 3 | 322.6 | 47.6 | 73.9 | 21430 | 22881 | 0.000 |
| phi-4-mini-instruct | 1024 | vllm | 3 | 304.7 | 56.4 | 80.5 | 12857 | 23149 | 0.000 |
| phi-4-mini-instruct | 4096 | sglang | 3 | 293.5 | 48.3 | 99.9 | 70562 | 87423 | 0.000 |
| phi-4-mini-instruct | 4096 | vllm | 3 | 287.2 | 56.8 | 74.5 | 13473 | 79221 | 0.000 |
