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
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Execution mode: sequential single-GPU runs

### single_request_latency (50 requests)

- **vLLM**: TTFT p50 15.9 ms, p95 16.4 ms, total latency p95 1047.4 ms, 4982.1 tok/s
- **SGLang**: TTFT p50 27.5 ms, p95 28.1 ms, total latency p95 1008.2 ms, 6253.8 tok/s

### throughput_ramp (700 requests)

- **vLLM**: TTFT p50 31.5 ms, p95 178.1 ms, total latency p95 3202.4 ms, 55287.6 tok/s
- **SGLang**: TTFT p50 49.4 ms, p95 155.7 ms, total latency p95 4480.8 ms, 39840.1 tok/s

All above runs completed at 100% success rate.

## Release-readiness improvements in this beta

- Fixed packaging metadata for Hatch build targets
- Added CI workflow (Python 3.11/3.12):
  - `ruff check .`
  - `pytest -q`
- Added release-readiness + benchmark report sections to README
- Cleaned Docker Compose config by removing obsolete `version` field
- Fixed SGLang startup arg mismatch in Compose configuration

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
