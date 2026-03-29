# Final Benchmark Summary

Source directory: `results/phi-3-mini-4k-instruct`
Result files considered: **20**

Model filter: `microsoft/Phi-3-mini-4k-instruct`
Selection mode: `explicit-model`

## microsoft/Phi-3-mini-4k-instruct

| Scenario | Engine | Runs | Prompt pack(s) | TTFT p50 | TTFT p95 | Latency p95 | Tok/s | Req/s | Success |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| long_context_stress | SGLangClient | 2 | long_context | 54.0 ms | 121.1 ms | 4953.6 ms | 211.5 | 0.83 | 100.0% |
| long_context_stress | VLLMClient | 2 | long_context | 56.1 ms | 61.2 ms | 4793.7 ms | 231.4 | 0.90 | 100.0% |
| prefix_sharing_benefit | SGLangClient | 2 | shared_prefix | 56.5 ms | 141.0 ms | 2574.0 ms | 396.6 | 3.10 | 100.0% |
| prefix_sharing_benefit | VLLMClient | 2 | shared_prefix | 54.7 ms | 66.5 ms | 2489.0 ms | 409.9 | 3.21 | 100.0% |
| single_request_latency | SGLangClient | 2 | short_chat | 45.6 ms | 50.9 ms | 2319.8 ms | 55.7 | 0.44 | 100.0% |
| single_request_latency | VLLMClient | 2 | short_chat | 25.0 ms | 25.3 ms | 2233.7 ms | 57.8 | 0.45 | 100.0% |
| structured_generation_speed | SGLangClient | 2 | structured_json | 70.9 ms | 115.7 ms | 3096.3 ms | 782.6 | 5.22 | 100.0% |
| structured_generation_speed | VLLMClient | 2 | structured_json | 59.0 ms | 78.4 ms | 3066.5 ms | 784.8 | 5.23 | 100.0% |
| throughput_ramp | SGLangClient | 2 | long_generation | 57.1 ms | 210.6 ms | 7523.5 ms | 187.3 | 0.73 | 100.0% |
| throughput_ramp | VLLMClient | 2 | long_generation | 54.0 ms | 197.9 ms | 8029.0 ms | 190.9 | 0.75 | 100.0% |
