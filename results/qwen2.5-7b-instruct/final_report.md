# Final Benchmark Summary

Source directory: `results/qwen2.5-7b-instruct`
Result files considered: **20**

Models detected: `Qwen/Qwen2.5-7B-Instruct`

## Qwen/Qwen2.5-7B-Instruct

| Scenario | Engine | Runs | Prompt pack(s) | TTFT p50 | TTFT p95 | Latency p95 | Tok/s | Req/s | Success |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| long_context_stress | SGLangClient | 2 | long_context | 76.4 ms | 88.9 ms | 8463.1 ms | 124.7 | 0.49 | 100.0% |
| long_context_stress | VLLMClient | 2 | long_context | 91.3 ms | 93.9 ms | 8500.4 ms | 118.3 | 0.47 | 100.0% |
| prefix_sharing_benefit | SGLangClient | 2 | shared_prefix | 92.1 ms | 140.0 ms | 4453.1 ms | 229.2 | 1.87 | 100.0% |
| prefix_sharing_benefit | VLLMClient | 2 | shared_prefix | 90.3 ms | 95.5 ms | 4330.2 ms | 228.9 | 1.84 | 100.0% |
| single_request_latency | SGLangClient | 2 | short_chat | 65.6 ms | 65.9 ms | 4170.4 ms | 30.8 | 0.27 | 100.0% |
| single_request_latency | VLLMClient | 2 | short_chat | 40.8 ms | 62.8 ms | 4220.2 ms | 30.6 | 0.29 | 100.0% |
| structured_generation_speed | SGLangClient | 2 | structured_json | 91.9 ms | 137.2 ms | 5676.8 ms | 389.6 | 6.46 | 100.0% |
| structured_generation_speed | VLLMClient | 2 | structured_json | 94.0 ms | 139.0 ms | 1978.2 ms | 455.9 | 9.93 | 100.0% |
| throughput_ramp | SGLangClient | 2 | long_generation | 65.7 ms | 272.2 ms | 9545.9 ms | 106.3 | 0.42 | 100.0% |
| throughput_ramp | VLLMClient | 2 | long_generation | 92.7 ms | 192.7 ms | 9762.0 ms | 105.3 | 0.41 | 100.0% |
