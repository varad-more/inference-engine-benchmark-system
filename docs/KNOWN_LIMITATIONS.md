# Known Limitations

## Validated scope today

The strongest validated path in this repo is the completed A10G benchmark workflow documented in:

- `reports/final_benchmark_report_2026-03-22.md`
- `reports/benchmark_snapshot_2026-03-22.json`

## Current limitations

### 1. Single-GPU fairness requires sequential execution

On a single A10G, vLLM and SGLang should be benchmarked sequentially rather than concurrently.

### 2. Model and engine compatibility can vary

The repo includes real examples where one engine path works and another does not.

Example:
- `microsoft/Phi-3-mini-4k-instruct` was benchmarked only on vLLM on this setup because SGLang hit `unsupported head_dim=96`.

### 3. Larger models may require engine-specific tuning

Example:
- `google/gemma-2-9b-it` needed tuned vLLM settings on a single A10G:
  - `context=4096`
  - `gpu_memory_utilization=0.92`

### 4. Public benchmark numbers are environment-specific

Published results are meaningful, but they are not universal truths. Engine behavior depends on:

- GPU class,
- driver / CUDA stack,
- model family,
- context length,
- workload shape,
- concurrency,
- and prompt distribution.

### 5. This repo is a benchmark harness, not a generic managed inference platform

It is designed for reproducible benchmarking, reporting, and experimentation. It is not positioned as a fully managed production serving control plane.
