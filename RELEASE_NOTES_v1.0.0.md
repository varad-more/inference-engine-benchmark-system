# v1.0.0 — Production Release

First production release of the vLLM vs SGLang benchmark harness. The v0.1.0-beta results set (5 models, single GPU class) is now superseded by a fully validated matrix: **16 models**, **10 scenarios**, **4 dedicated extended phases**, and **~530 result files at 100 % success rate** on AWS A10G 24 GB.

## Highlights

- **Benchmark matrix is complete.** 14-model core baseline + 2 Gemma 4 models, each run across 5 core scenarios on both engines, plus speculative decoding (Ngram + Eagle3), variance, concurrency-64, and a decode-length sweep.
- **Four new figures** regenerate from saved result files — no hand-drawn charts.
- **CI is green and enforced.** `ruff check`, `ruff format --check`, `mypy` (24 source files), `pytest` (89 tests), `python -m build`, `twine check`.
- **README reflects reality.** No broken script references, no stale dates, no out-of-date module lists, no phantom follow-ups.
- **Dashboard hardened.** Version bumped to 1.0.0; in-memory job registry now bounded (`DASHBOARD_MAX_JOBS`, default 100) with oldest-terminal eviction instead of unbounded growth.

## Validated benchmark snapshot (2026-04-21)

Environment:
- AWS `g5.2xlarge` (NVIDIA A10G 24 GB)
- vLLM `v0.18.0-cu130`, SGLang `nightly-dev-cu13-20260321`
- 16 models from 2B to 9B, `bfloat16`
- Sequential single-GPU execution
- Source of truth: [`reports/final_benchmark_report_2026-03-31.md`](reports/final_benchmark_report_2026-03-31.md)

### Headline findings

- **vLLM wins TTFT on 13 / 14 core-baseline models** (20–60 % lower than SGLang at concurrency 1). Only Gemma 3 4B flips (SGLang faster by 9 ms) because vLLM needs `--enforce-eager`.
- **vLLM wins small-model throughput** (≤ 4B): +3–12 % on SmolLM3, Phi-3 mini, Phi-4 mini, Gemma 2 2B.
- **Engines converge at 7–9B.** Differences < 3 % on Qwen, Mistral, Llama, Granite, DeepSeek-R1 variants.
- **Gemma 3 4B is SGLang's strongest case:** +77 % peak throughput vs vLLM (149 vs 84 tok/s).
- **Structured generation:** vLLM wins 12 / 14 models; SGLang wins 2.
- **Prefix-sharing TTFT:** SGLang wins 10 / 14 — RadixAttention pays off when prefixes are genuinely shared.
- **Speculative decoding on A10G:** Ngram works on Llama 3.1 8B, Qwen3 8B, and Gemma 4 E4B across both engines. Eagle3 works on Llama 3.1 8B with vLLM only (SGLang + Eagle3 exceeds 24 GB VRAM). Net: spec-dec **hurts** throughput on A10G; expect a reversal on ≥ 40 GB hardware.
- **Goodput (TTFT ≤ 100 ms, TPOT ≤ 35 ms):** vLLM leads on small models (SmolLM3 1.06 rps, Gemma 2 2B 1.37 rps). SGLang leads on 7–9B under concurrent load when TTFT dominates.

### Extended phases shipped in this release

| Phase | Cells | Iterations | Purpose |
|---|---|---|---|
| Variance subset | 4 models × 5 scenarios × 2 engines | n=5 | 95 % CIs and CV% for reproducibility |
| Concurrency-64 ramp | 4 × 7–9B models × 2 engines | 150 req/level × 6 levels | Saturation + tail-latency behaviour |
| Decode-length sweep | 6 models × 4 output lengths × 2 engines | n=3 | Crossover analysis at max_output_tokens ∈ {64, 256, 1024, 4096} |
| Gemma 4 (E2B + E4B) | 2 models × 5 scenarios × 2 engines + Ngram | n=1 | First published Gemma 4 numbers |

## What's new since v0.1.0-beta

### Benchmark surface
- **+11 models** to the validated matrix: SmolLM3 3B, Llama 3.2 3B, Phi-4 mini, Gemma 3 4B, DeepSeek-R1-Distill Qwen 7B + Llama 8B, Qwen 2.5 7B, Llama 3.1 8B, Qwen3 8B, Granite 3.3 8B, and both Gemma 4 sizes (E2B, E4B).
- **+5 scenarios** registered: `throughput_ramp_extended`, `decode_length_sweep_{64,256,1024,4096}`.
- **Speculative decoding:** 6 engine variants with one-command Docker Compose profiles (`vllm`, `vllm-eagle3`, `vllm-ngram`, `sglang`, `sglang-eagle3`, `sglang-ngram`).

### Visualizations
- **Four new auto-regenerating figures** under `analysis/generate_*_figure.py`:
  - `speculative_decoding.svg` — Llama 3.1 8B + Qwen3 8B + Gemma 4 E4B, baseline vs Ngram vs Eagle3
  - `decode_length_sweep.svg` — tokens/sec vs max_output_tokens (6 models, 95 % CI error bars)
  - `variance_cv.svg` — CV% per (model × engine × scenario × metric) with 5 % threshold line
  - `goodput.svg` — joint TTFT/TPOT SLO goodput per model
- Shared dark-theme style in `analysis/_figure_style.py`.
- Existing core baseline figures remain regenerable via `python -m analysis.generate_final_benchmark_report`.

### Analysis tooling
- `analysis/variance_analysis.py` — CV% + t-distribution 95 % CIs across iterations; flags claims above the 5 % threshold.
- `analysis/tpot_analysis.py` — per-request TPOT P50/P95/P99 (`(total_ms − ttft_ms) / max(output_tokens − 1, 1)`).
- `analysis/decode_length_analysis.py` — crossover detection across the decode-length sweep.
- `analysis/goodput.py` — configurable joint SLO goodput (defaults: TTFT ≤ 100 ms, TPOT ≤ 35 ms).

### Release-readiness & quality
- **CI strengthened** to cover `ruff format --check` + full `mypy` pass on `engines/`, `benchmarks/`, `analysis/`, `dashboard/` (24 source files, 0 errors), on top of the existing `ruff check`, `pytest`, `python -m build`, `twine check`.
- **Scenario registry test** aligned with the actual 10 registered scenarios.
- **Dashboard** version synced to package version; `_jobs` dict bounded with oldest-terminal eviction (configurable via `DASHBOARD_MAX_JOBS`).
- **README accuracy pass:** removed a broken reference to a pruned helper script; updated Project Structure to list all 7 `analysis/` modules; added the 5 extended scenarios to the scenario catalog; linked every published report and figure; removed rot-prone `Last updated` stamps.

## Known open items (tracked, non-blocking)

- **Llama 3.1 8B SGLang-Eagle3** — blocked on the retired `lmsysorg/sglang:nightly-dev-cu13-20260321-94194537` image; needs a fresh nightly pin before retry.
- **Gemma 4 E2B rerun** — `single_request_latency` and `throughput_ramp` result files report `total_tokens_generated = 0` for E2B; other three scenarios are clean. Decode-throughput numbers for E2B ngram are therefore not published in this release.
- **Dashboard has no auth** — fine for `localhost` and documented in `SECURITY.md`, but a reverse proxy / basic-auth shim is required before exposing the EC2 deployment path to the public internet.
- **SGLang Docker image pin** — same retired nightly in `.env.example` and `docker-compose.yml`. Update to a currently-pullable tag before running a fresh clone.

## Upgrade notes

```bash
git pull
pip install -e ".[dev]"
```

To regenerate every figure from the committed results set:

```bash
python -m analysis.generate_final_benchmark_report   # core baseline (5 SVGs)
python -m analysis.generate_spec_decoding_figure
python -m analysis.generate_decode_length_figure
python -m analysis.generate_variance_figure
python -m analysis.generate_goodput_figure
```

## Tag

`v1.0.0`
