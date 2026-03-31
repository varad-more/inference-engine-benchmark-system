# Final Benchmark Summary

Source directory: `/home/ubuntu/repos/inference-engine-benchmark-system-speculative/results_specdec/20260331_llama31_rerun/llama-3.1-8b-instruct`
Result files considered: **2**

Model filter: `meta-llama/Llama-3.1-8B-Instruct`
Selection mode: `explicit-model`

## meta-llama/Llama-3.1-8B-Instruct

| Scenario | Engine | Runs | Prompt pack(s) | TTFT p50 | TTFT p95 | Latency p95 | Tok/s | Req/s | Success |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| single_request_latency | vllm | 1 | short_chat | 42.7 ms | 43.2 ms | 4248.5 ms | 30.3 | 0.24 | 100.0% |
| throughput_ramp | vllm | 1 | long_generation | 97.0 ms | 500.7 ms | 10502.0 ms | 101.7 | 0.40 | 100.0% |
