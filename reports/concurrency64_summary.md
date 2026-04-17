# Phase 2: Concurrency-64 Extended Ramp — Summary

Generated 2026-04-17. Single iteration per (model, engine). Concurrency levels {1, 4, 8, 16, 32, 64}, 150 requests/level (900 total). Prompt 128 tokens, output 256 tokens.

| Model | Engine | n | Succ | Tokens/s | TTFT p50 (ms) | TTFT p99 (ms) | Latency p99 (ms) | Err rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B                          | vllm   | 1 | 900/900 | 113.7 | 103.2 | 232.2 | 11683 | 0.000 |
| Qwen3-8B                          | sglang | 1 | 900/900 | 113.9 |  73.5 | 403.0 | 11663 | 0.000 |
| Mistral-7B-Instruct-v0.3          | vllm   | 1 | 900/900 | 123.5 |  93.0 | 283.9 | 10136 | 0.000 |
| Mistral-7B-Instruct-v0.3          | sglang | — | — | pending | | | | |
| google/gemma-2-9b-it              | vllm   | — | — | pending | | | | |
| google/gemma-2-9b-it              | sglang | — | — | pending | | | | |
| meta-llama/Llama-3.1-8B-Instruct  | vllm   | — | — | pending | | | | |
| meta-llama/Llama-3.1-8B-Instruct  | sglang | — | — | pending | | | | |

**Notes:**
- All completed runs hit 0% error rate at concurrency=64 — no OOMs on A10G 24GB for 7–8B class models with 128/256 prompt/output sizes.
- Qwen3-8B aggregate throughput is identical across engines (~114 tok/s); vLLM has tighter tail TTFT (p99 232ms vs SGLang 403ms), SGLang has lower median TTFT (73.5ms vs 103ms).
- Mistral-7B (vLLM) wins absolute throughput at 123.5 tok/s — slightly faster than Qwen3-8B as expected (smaller model).
- Pending cells will be filled when phase2 resume runs (idempotent — won't re-run completed cells).
