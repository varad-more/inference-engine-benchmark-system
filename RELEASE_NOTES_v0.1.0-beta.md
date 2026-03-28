# v0.1.0-beta — Public Beta

## Highlights

This beta ships a production-style benchmark harness for comparing **vLLM** and **SGLang** under realistic inference workloads, with Dockerized setup, CLI runner, dashboard, and AWS deployment options.

### What’s included

- Comparative benchmark framework for vLLM vs SGLang
- Scenario runner with latency/throughput/cache-oriented workloads
- FastAPI dashboard for viewing and running benchmarks
- Docker Compose setup for local/remote benchmarking
- AWS deployment paths:
  - `deploy/ec2_deploy.sh` (quick end-to-end script)
  - `deploy/terraform/` (repeatable infra workflow)
- Result artifact persistence in `results/*.json`
- README benchmark report from validated A10G run

## Validated benchmark snapshot (2026-03-22)

Environment:
- AWS `g5.2xlarge` (NVIDIA A10G 24GB)
- Models: Gemma 2B, Phi-3 mini, Qwen 7B, Mistral 7B, Gemma 9B
- Execution mode: sequential single-GPU runs
- Source of truth: `reports/final_benchmark_report_2026-03-22.md`

### Headline results

- **Fastest single-request TTFT p95:** Gemma 2B on **vLLM** at **20.3 ms**
- **Gemma 2B:** vLLM led both latency and throughput in this setup
- **Phi-3 mini:** benchmarked on **vLLM only** because SGLang hit `unsupported head_dim=96`
- **Qwen 7B** and **Mistral 7B:** vLLM led TTFT, while throughput was split or effectively tied by metric
- **Gemma 9B:** SGLang led single-request TTFT, while tuned vLLM led throughput and ramp latency p95

### Important operational findings

- Single-GPU comparisons must run **one engine at a time** to avoid VRAM contention and misleading results.
- `google/gemma-2-9b-it` required tuned vLLM launch settings on A10G:
  - `context=4096`
  - `gpu_memory_utilization=0.92`
- The expanded benchmark matrix and generated artifacts live under `reports/` and `results/`.

## Release-readiness improvements in this beta

- Added CI workflow (Python 3.11/3.12) that validates:
  - `ruff check . --select E9,F63,F7,F82`
  - `pytest -q`
  - `python -m build`
  - `python -m twine check dist/*`
- Added release-facing benchmark report sections and replication guidance to README
- Added final multi-model report, benchmark snapshot, and chart assets under `reports/`
- Cleaned Docker Compose config by removing obsolete `version` field
- Fixed SGLang startup arg mismatch in Compose configuration
- Added prompt-pack, matrix-runner, and final-report aggregation support for reproducible benchmark execution

## Known limitations

- Public results currently reflect one model (`Qwen 1.5B`) and one GPU class (A10G)
- For stronger production claims, run a wider matrix:
  - multiple model families/sizes (e.g., Llama 8B, Qwen 7B/14B)
  - multiple reruns for variance bands
  - longer-context and structured-generation heavy workloads
- Single-GPU environments should run engines sequentially to avoid VRAM contention

## Upgrade notes

- If you pull latest changes, re-run:

```bash
pip install -e ".[dev]"
```

- CI is now enforced through GitHub Actions on push/PR.

## Tag

`v0.1.0-beta`
