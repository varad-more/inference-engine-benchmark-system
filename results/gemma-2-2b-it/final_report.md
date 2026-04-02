# Final Benchmark Summary

Source directory: `results/gemma-2-2b-it`
Result files considered: **20**

Models detected: `google/gemma-2-2b-it`

## google/gemma-2-2b-it

| Scenario | Engine | Runs | Prompt pack(s) | TTFT p50 | TTFT p95 | Latency p95 | Tok/s | Req/s | Success |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| long_context_stress | SGLangClient | 2 | long_context | 43.6 ms | 53.4 ms | 3471.1 ms | 293.1 | 1.24 | 100.0% |
| long_context_stress | VLLMClient | 2 | long_context | 33.4 ms | 45.1 ms | 3349.7 ms | 310.8 | 1.24 | 100.0% |
| prefix_sharing_benefit | SGLangClient | 2 | shared_prefix | 39.9 ms | 72.8 ms | 1928.8 ms | 539.0 | 5.24 | 100.0% |
| prefix_sharing_benefit | VLLMClient | 2 | shared_prefix | 43.8 ms | 57.8 ms | 1713.0 ms | 600.0 | 5.25 | 100.0% |
| single_request_latency | SGLangClient | 2 | short_chat | 30.4 ms | 31.6 ms | 1656.3 ms | 78.3 | 0.75 | 100.0% |
| single_request_latency | VLLMClient | 2 | short_chat | 19.7 ms | 22.5 ms | 1655.5 ms | 77.6 | 0.75 | 100.0% |
| structured_generation_speed | SGLangClient | 2 | structured_json | 47.0 ms | 85.4 ms | 2512.9 ms | 952.3 | 11.03 | 100.0% |
| structured_generation_speed | VLLMClient | 2 | structured_json | 45.8 ms | 58.9 ms | 1038.2 ms | 1225.1 | 19.88 | 100.0% |
| throughput_ramp | SGLangClient | 2 | long_generation | 46.0 ms | 158.6 ms | 4852.7 ms | 258.0 | 1.05 | 100.0% |
| throughput_ramp | VLLMClient | 2 | long_generation | 40.2 ms | 154.1 ms | 4583.1 ms | 264.6 | 1.14 | 100.0% |
