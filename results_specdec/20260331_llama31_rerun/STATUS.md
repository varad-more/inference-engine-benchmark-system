# Llama 3.1 8B speculative decoding rerun status

Snapshot synced from the GPU host on 2026-03-30 MST.

## Completed
- Baseline `vllm` run completed for:
  - `single_request_latency`
  - `throughput_ramp`
- Generated artifacts:
  - `llama-3.1-8b-instruct/matrix_manifest_1774926564.json`
  - `llama-3.1-8b-instruct/single_request_latency_vllm_1774926564.json`
  - `llama-3.1-8b-instruct/throughput_ramp_vllm_1774926897.json`
  - `llama-3.1-8b-instruct/final_report.md`
  - `llama-3.1-8b-instruct/report.html`

## Attempted but not completed
- `vllm-eagle3`
- `vllm-ngram`
- `sglang`
- `sglang-ngram`
- `sglang-eagle3`

## Known blockers seen in logs
- Leftover baseline server originally held port `8000` and GPU memory, causing follow-on runs to fail.
- `vllm-eagle3` compose entrypoint used `python` instead of `python3` in the vLLM image.
- Compose startup defaulted to the fallback model when `MODEL` was not exported before interpolation.
- `vllm-eagle3` also hit KV-cache allocation issues on A10G.
- `sglang` / `sglang-ngram` hit CUDA OOM when memory was already occupied.
- `sglang-eagle3` hit a draft/target context-length mismatch (`8192` vs derived `2048`).

## Logs included
- Original overnight run log
- Recovery/fix attempt logs captured during troubleshooting

This commit is a checkpoint of current progress, not a completed speculative-decoding result set.
