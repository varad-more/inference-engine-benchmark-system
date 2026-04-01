# Final Benchmark Summary

Source directory: `results/llama-3.2-3b-instruct`
Result files considered: **20**

Models detected: `meta-llama/Llama-3.2-3B-Instruct`

## meta-llama/Llama-3.2-3B-Instruct

| Scenario | Engine | Runs | Prompt pack(s) | TTFT p50 | TTFT p95 | Latency p95 | Tok/s | Req/s | Success |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| long_context_stress | SGLangClient | 2 | long_context | 44.2 ms | 101.8 ms | 4089.7 ms | 253.5 | 0.99 | 100.0% |
| long_context_stress | VLLMClient | 2 | long_context | 40.4 ms | 53.9 ms | 4085.7 ms | 254.5 | 0.99 | 100.0% |
| prefix_sharing_benefit | SGLangClient | 2 | shared_prefix | 40.7 ms | 79.4 ms | 2077.6 ms | 489.2 | 3.82 | 100.0% |
| prefix_sharing_benefit | VLLMClient | 2 | shared_prefix | 50.1 ms | 56.9 ms | 2096.2 ms | 489.2 | 3.82 | 100.0% |
| single_request_latency | SGLangClient | 2 | short_chat | 32.3 ms | 33.2 ms | 1903.5 ms | 67.7 | 0.53 | 100.0% |
| single_request_latency | VLLMClient | 2 | short_chat | 22.6 ms | 23.1 ms | 1928.8 ms | 66.3 | 0.52 | 100.0% |
| structured_generation_speed | SGLangClient | 2 | structured_json | 52.0 ms | 86.5 ms | 2648.2 ms | 908.4 | 6.20 | 100.0% |
| structured_generation_speed | VLLMClient | 2 | structured_json | 49.5 ms | 71.8 ms | 2503.2 ms | 970.2 | 6.62 | 100.0% |
| throughput_ramp | SGLangClient | 2 | long_generation | 46.3 ms | 213.6 ms | 5627.1 ms | 225.9 | 0.88 | 100.0% |
| throughput_ramp | VLLMClient | 2 | long_generation | 50.0 ms | 168.8 ms | 5390.9 ms | 223.5 | 0.87 | 100.0% |
