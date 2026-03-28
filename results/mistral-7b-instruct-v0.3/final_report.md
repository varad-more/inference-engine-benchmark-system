# Final Benchmark Summary

Source directory: `results/mistral-7b-instruct-v0.3`
Result files considered: **20**

Models detected: `mistralai/Mistral-7B-Instruct-v0.3`

## mistralai/Mistral-7B-Instruct-v0.3

| Scenario | Engine | Runs | Prompt pack(s) | TTFT p50 | TTFT p95 | Latency p95 | Tok/s | Req/s | Success |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| long_context_stress | SGLangClient | 2 | long_context | 93.4 ms | 101.7 ms | 8643.8 ms | 115.1 | 0.51 | 100.0% |
| long_context_stress | VLLMClient | 2 | long_context | 93.8 ms | 95.6 ms | 8582.6 ms | 117.0 | 0.50 | 100.0% |
| prefix_sharing_benefit | SGLangClient | 2 | shared_prefix | 92.6 ms | 162.2 ms | 4599.2 ms | 218.9 | 2.04 | 99.5% |
| prefix_sharing_benefit | VLLMClient | 2 | shared_prefix | 93.3 ms | 96.5 ms | 4375.6 ms | 228.4 | 2.16 | 100.0% |
| single_request_latency | SGLangClient | 2 | short_chat | 62.5 ms | 63.7 ms | 4047.6 ms | 31.8 | 0.26 | 100.0% |
| single_request_latency | VLLMClient | 2 | short_chat | 41.0 ms | 62.6 ms | 4064.4 ms | 31.8 | 0.26 | 100.0% |
| structured_generation_speed | SGLangClient | 2 | structured_json | 93.9 ms | 150.6 ms | 5518.2 ms | 411.2 | 3.09 | 100.0% |
| structured_generation_speed | VLLMClient | 2 | structured_json | 96.3 ms | 107.7 ms | 5075.3 ms | 439.9 | 3.31 | 100.0% |
| throughput_ramp | SGLangClient | 2 | long_generation | 72.6 ms | 279.3 ms | 10162.4 ms | 106.8 | 0.42 | 99.9% |
| throughput_ramp | VLLMClient | 2 | long_generation | 93.2 ms | 216.9 ms | 10141.7 ms | 106.6 | 0.42 | 100.0% |
