# Final Benchmark Summary

Source directory: `/home/ubuntu/repos/inference-engine-benchmark-system-speculative/results_afk/20260330_speculative/qwen3-8b`
Result files considered: **10**

Model filter: `Qwen/Qwen3-8B`
Selection mode: `explicit-model`

## Qwen/Qwen3-8B

| Scenario | Engine | Runs | Prompt pack(s) | TTFT p50 | TTFT p95 | Latency p95 | Tok/s | Req/s | Success |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| long_context_stress | sglang | 1 | long_context | 69.6 ms | 117.1 ms | 9296.3 ms | 115.5 | 0.45 | 100.0% |
| long_context_stress | vllm | 1 | long_context | 102.2 ms | 104.4 ms | 9298.6 ms | 110.7 | 0.43 | 100.0% |
| prefix_sharing_benefit | sglang | 1 | shared_prefix | 59.3 ms | 206.8 ms | 4806.8 ms | 210.4 | 1.64 | 100.0% |
| prefix_sharing_benefit | vllm | 1 | shared_prefix | 95.1 ms | 105.3 ms | 4739.6 ms | 212.2 | 1.66 | 100.0% |
| single_request_latency | sglang | 1 | short_chat | 69.1 ms | 71.4 ms | 4382.9 ms | 29.2 | 0.23 | 100.0% |
| single_request_latency | vllm | 1 | short_chat | 44.0 ms | 44.6 ms | 4377.4 ms | 29.5 | 0.23 | 100.0% |
| structured_generation_speed | sglang | 1 | structured_json | 111.5 ms | 198.1 ms | 5829.2 ms | 397.8 | 2.67 | 100.0% |
| structured_generation_speed | vllm | 1 | structured_json | 122.4 ms | 147.3 ms | 5816.2 ms | 398.5 | 2.69 | 100.0% |
| throughput_ramp | sglang | 1 | long_generation | 71.4 ms | 199.3 ms | 11749.9 ms | 98.6 | 0.39 | 100.0% |
| throughput_ramp | vllm | 1 | long_generation | 101.6 ms | 203.7 ms | 11529.1 ms | 98.4 | 0.38 | 100.0% |
