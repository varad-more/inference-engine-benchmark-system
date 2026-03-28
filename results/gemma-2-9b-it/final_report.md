# Final Benchmark Summary

Source directory: `results/gemma-2-9b-it`
Result files considered: **20**

Model filter: `google/gemma-2-9b-it`
Selection mode: `explicit-model`

## google/gemma-2-9b-it

| Scenario | Engine | Runs | Prompt pack(s) | TTFT p50 | TTFT p95 | Latency p95 | Tok/s | Req/s | Success |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| long_context_stress | SGLangClient | 2 | long_context | 125.8 ms | 131.9 ms | 11549.5 ms | 81.0 | 0.39 | 100.0% |
| long_context_stress | VLLMClient | 2 | long_context | 125.4 ms | 127.4 ms | 10464.8 ms | 82.5 | 0.80 | 100.0% |
| prefix_sharing_benefit | SGLangClient | 2 | shared_prefix | 125.2 ms | 198.2 ms | 6217.4 ms | 159.3 | 1.96 | 100.0% |
| prefix_sharing_benefit | VLLMClient | 2 | shared_prefix | 128.3 ms | 132.4 ms | 5885.2 ms | 164.3 | 1.87 | 100.0% |
| single_request_latency | SGLangClient | 2 | short_chat | 83.0 ms | 84.2 ms | 5312.8 ms | 24.2 | 0.21 | 100.0% |
| single_request_latency | VLLMClient | 2 | short_chat | 74.1 ms | 105.7 ms | 5360.2 ms | 24.0 | 0.21 | 100.0% |
| structured_generation_speed | SGLangClient | 2 | structured_json | 126.1 ms | 195.5 ms | 4425.7 ms | 290.0 | 5.49 | 100.0% |
| structured_generation_speed | VLLMClient | 2 | structured_json | 128.8 ms | 154.3 ms | 6263.0 ms | 310.8 | 4.62 | 100.0% |
| throughput_ramp | SGLangClient | 2 | long_generation | 125.1 ms | 30483.6 ms | 46328.4 ms | 78.0 | 0.31 | 99.9% |
| throughput_ramp | VLLMClient | 2 | long_generation | 127.0 ms | 283.9 ms | 14388.5 ms | 79.8 | 0.33 | 100.0% |
