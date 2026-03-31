# Final Benchmark Summary

Source directory: `/home/ubuntu/repos/inference-engine-benchmark-system-speculative/results_afk/20260330_speculative/deepseek-r1-distill-qwen-7b`
Result files considered: **10**

Model filter: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
Selection mode: `explicit-model`

## deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

| Scenario | Engine | Runs | Prompt pack(s) | TTFT p50 | TTFT p95 | Latency p95 | Tok/s | Req/s | Success |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| long_context_stress | sglang | 1 | long_context | 58.4 ms | 105.3 ms | 8465.7 ms | 125.9 | 0.49 | 100.0% |
| long_context_stress | vllm | 1 | long_context | 87.4 ms | 93.1 ms | 8470.5 ms | 121.2 | 0.47 | 100.0% |
| prefix_sharing_benefit | sglang | 1 | shared_prefix | 77.5 ms | 281.1 ms | 4393.6 ms | 235.1 | 1.84 | 100.0% |
| prefix_sharing_benefit | vllm | 1 | shared_prefix | 86.8 ms | 95.9 ms | 4302.2 ms | 235.7 | 1.84 | 100.0% |
| single_request_latency | sglang | 1 | short_chat | 65.7 ms | 66.3 ms | 4177.2 ms | 30.9 | 0.24 | 100.0% |
| single_request_latency | vllm | 1 | short_chat | 40.1 ms | 62.8 ms | 4215.8 ms | 30.5 | 0.28 | 100.0% |
| structured_generation_speed | sglang | 1 | structured_json | 94.9 ms | 184.1 ms | 5186.4 ms | 450.7 | 3.00 | 100.0% |
| structured_generation_speed | vllm | 1 | structured_json | 85.8 ms | 115.0 ms | 5153.5 ms | 451.7 | 3.01 | 100.0% |
| throughput_ramp | sglang | 1 | long_generation | 69.7 ms | 313.3 ms | 9603.0 ms | 106.1 | 0.41 | 100.0% |
| throughput_ramp | vllm | 1 | long_generation | 92.6 ms | 184.7 ms | 9322.1 ms | 105.6 | 0.41 | 100.0% |
