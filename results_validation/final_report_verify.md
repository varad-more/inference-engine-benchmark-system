# Final Benchmark Summary

Source directory: `results_validation`
Result files considered: **10**

Models detected: `Qwen/Qwen2.5-1.5B-Instruct`

## Qwen/Qwen2.5-1.5B-Instruct

| Scenario | Engine | Runs | Prompt pack(s) | TTFT p50 | TTFT p95 | Latency p95 | Tok/s | Req/s | Success |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| long_context_stress | SGLangClient | 1 | long_context | 43.1 ms | 152.7 ms | 2205.4 ms | 509.3 | 1.99 | 100.0% |
| long_context_stress | VLLMClient | 1 | long_context | 28.6 ms | 128.8 ms | 2246.6 ms | 508.9 | 1.99 | 100.0% |
| prefix_sharing_benefit | SGLangClient | 1 | shared_prefix | 37.5 ms | 95.4 ms | 1197.7 ms | 884.6 | 7.09 | 100.0% |
| prefix_sharing_benefit | VLLMClient | 1 | shared_prefix | 30.3 ms | 40.4 ms | 1130.3 ms | 905.8 | 7.09 | 100.0% |
| single_request_latency | SGLangClient | 1 | short_chat | 23.8 ms | 25.1 ms | 1000.1 ms | 129.7 | 1.18 | 100.0% |
| single_request_latency | VLLMClient | 1 | short_chat | 15.8 ms | 16.4 ms | 1046.5 ms | 122.8 | 1.18 | 100.0% |
| structured_generation_speed | SGLangClient | 1 | structured_json | 38.6 ms | 68.6 ms | 1708.1 ms | 1541.9 | 19.89 | 100.0% |
| structured_generation_speed | VLLMClient | 1 | structured_json | 31.2 ms | 50.8 ms | 1373.8 ms | 1621.0 | 19.86 | 100.0% |
| throughput_ramp | SGLangClient | 1 | long_generation | 44.5 ms | 184.0 ms | 4492.2 ms | 423.7 | 1.66 | 100.0% |
| throughput_ramp | VLLMClient | 1 | long_generation | 30.7 ms | 136.4 ms | 3079.5 ms | 417.1 | 1.63 | 100.0% |
