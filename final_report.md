# Final Benchmark Summary

Source directory: `results_validation`
Result files considered: **4**

Models detected: `Qwen/Qwen2.5-1.5B-Instruct`

## Qwen/Qwen2.5-1.5B-Instruct

| Scenario | Engine | Runs | Prompt pack(s) | TTFT p50 | TTFT p95 | Latency p95 | Tok/s | Req/s | Success |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| single_request_latency | SGLangClient | 1 | short_chat | 23.8 ms | 25.1 ms | 1000.1 ms | 129.7 | 1.18 | 100.0% |
| single_request_latency | VLLMClient | 1 | short_chat | 15.8 ms | 16.4 ms | 1046.5 ms | 122.8 | 1.18 | 100.0% |
| throughput_ramp | SGLangClient | 1 | long_generation | 44.5 ms | 184.0 ms | 4492.2 ms | 423.7 | 1.66 | 100.0% |
| throughput_ramp | VLLMClient | 1 | long_generation | 30.7 ms | 136.4 ms | 3079.5 ms | 417.1 | 1.63 | 100.0% |
