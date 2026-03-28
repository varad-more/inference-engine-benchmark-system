# Final Benchmark Summary

Source directory: `results/gemma-2-9b-it`
Result files considered: **10**

Models detected: `google/gemma-2-9b-it`

## google/gemma-2-9b-it

| Scenario | Engine | Runs | Prompt pack(s) | TTFT p50 | TTFT p95 | Latency p95 | Tok/s | Req/s | Success |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| long_context_stress | SGLangClient | 2 | long_context | 125.8 ms | 131.9 ms | 11549.5 ms | 81.0 | 0.39 | 100.0% |
| prefix_sharing_benefit | SGLangClient | 2 | shared_prefix | 125.2 ms | 198.2 ms | 6217.4 ms | 159.3 | 1.96 | 100.0% |
| single_request_latency | SGLangClient | 2 | short_chat | 83.0 ms | 84.2 ms | 5312.8 ms | 24.2 | 0.21 | 100.0% |
| structured_generation_speed | SGLangClient | 2 | structured_json | 126.1 ms | 195.5 ms | 4425.7 ms | 290.0 | 5.49 | 100.0% |
| throughput_ramp | SGLangClient | 2 | long_generation | 125.1 ms | 30483.6 ms | 46328.4 ms | 78.0 | 0.31 | 99.9% |
