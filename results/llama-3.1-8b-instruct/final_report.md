# Final Benchmark Summary

Source directory: `results/llama-3.1-8b-instruct`
Result files considered: **20**

Models detected: `meta-llama/Llama-3.1-8B-Instruct`

## meta-llama/Llama-3.1-8B-Instruct

| Scenario | Engine | Runs | Prompt pack(s) | TTFT p50 | TTFT p95 | Latency p95 | Tok/s | Req/s | Success |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| long_context_stress | SGLangClient | 2 | long_context | 63.0 ms | 98.9 ms | 8950.1 ms | 115.6 | 0.45 | 100.0% |
| long_context_stress | VLLMClient | 2 | long_context | 90.1 ms | 98.8 ms | 8971.8 ms | 115.6 | 0.45 | 100.0% |
| prefix_sharing_benefit | SGLangClient | 2 | shared_prefix | 67.2 ms | 139.1 ms | 4579.8 ms | 217.8 | 1.70 | 100.0% |
| prefix_sharing_benefit | VLLMClient | 2 | shared_prefix | 91.6 ms | 101.2 ms | 4570.6 ms | 219.4 | 1.71 | 100.0% |
| single_request_latency | SGLangClient | 2 | short_chat | 66.7 ms | 67.6 ms | 4247.6 ms | 30.3 | 0.24 | 100.0% |
| single_request_latency | VLLMClient | 2 | short_chat | 42.7 ms | 43.5 ms | 4247.4 ms | 30.3 | 0.24 | 100.0% |
| structured_generation_speed | SGLangClient | 2 | structured_json | 99.0 ms | 152.7 ms | 5578.7 ms | 422.6 | 2.82 | 100.0% |
| structured_generation_speed | VLLMClient | 2 | structured_json | 96.2 ms | 125.5 ms | 5570.9 ms | 426.2 | 2.84 | 100.0% |
| throughput_ramp | SGLangClient | 2 | long_generation | 69.0 ms | 186.5 ms | 10621.4 ms | 102.1 | 0.40 | 100.0% |
| throughput_ramp | VLLMClient | 2 | long_generation | 97.3 ms | 201.3 ms | 10534.8 ms | 101.7 | 0.40 | 100.0% |
