# Benchmark Run Status

Snapshot of completed vs. remaining runs across all phases.

- **Snapshot time:** 2026-04-20 ~04:00 UTC
- **Units:** each number is an actual result JSON file on disk.
- **Source of truth:** `results/`, `results_variance/`, `results_concurrency64/`, `results_decode_sweep/`.

## Variance subset
Target: **25 runs per engine per model** (5 scenarios × 5 iterations).

| Model | vLLM | SGLang | Remaining |
|---|---|---|---|
| gemma-2-2b-it | 25/25 ✅ | 25/25 ✅ | 0 |
| phi-4-mini-instruct | 25/25 ✅ | 25/25 ✅ | 0 |
| llama-3-1-8b-instruct | 25/25 ✅ | 26/25 ✅ | 0 |
| gemma-3-4b-it | 25/25 ✅ | 25/25 ✅ | 0 |
| **Subtotal** | | | **0** ✅ |

Output dir: `results_variance/`.

## Concurrency-64 ramp
Target: **1 run per engine per model** (single `throughput_ramp_extended` iteration).

| Model | vLLM | SGLang | Remaining |
|---|---|---|---|
| llama-3-1-8b-instruct | 1/1 ✅ | 1/1 ✅ | 0 |
| qwen3-8b | 1/1 ✅ | 1/1 ✅ | 0 |
| mistral-7b-instruct-v0-3 | 1/1 ✅ | 1/1 ✅ | 0 |
| gemma-2-9b-it | 1/1 ✅ | 1/1 ✅ | 0 |
| **Subtotal** | | | **0** ✅ |

Output dir: `results_concurrency64/`.

## Decode-length sweep
Target: **12 runs per engine per model** (4 lengths × 3 iterations).

| Model | vLLM | SGLang | Remaining |
|---|---|---|---|
| gemma-2-2b-it | 12/12 ✅ | 12/12 ✅ | 0 |
| phi-4-mini-instruct | 12/12 ✅ | 12/12 ✅ | 0 |
| llama-3-1-8b-instruct | 12/12 ✅ | 12/12 ✅ | 0 |
| gemma-3-4b-it | 12/12 ✅ | 12/12 ✅ | 0 |
| gemma-4-e2b-it | 12/12 ✅ | 12/12 ✅ | 0 |
| gemma-4-e4b-it | 12/12 ✅ | 12/12 ✅ | 0 |
| **Subtotal** | | | **0** ✅ |

Output dir: `results_decode_sweep/`.

## Gemma 4 baseline + ngram spec-dec
Target: **14 runs per model** (10 baseline + 4 ngram).
- Baseline: 5 scenarios × 2 engines × 1 iter = 10
- Ngram spec-dec: 2 scenarios × 2 engines × 1 iter = 4

| Model | vLLM | SGLang | vllm-ngram | sglang-ngram | Remaining |
|---|---|---|---|---|---|
| gemma-4-e2b-it | 5/5 ✅ | 5/5 ✅ | 2/2 ✅ | 2/2 ✅ | 0 |
| gemma-4-e4b-it | 5/5 ✅ | 5/5 ✅ | 2/2 ✅ | 2/2 ✅ | 0 |
| **Subtotal** | | | | | **0** ✅ |

Output dir: `results/`.

## Grand totals

| Category | Runs |
|---|---|
| Variance subset | 201 / 200 ✅ |
| Concurrency-64 | 8 / 8 ✅ |
| Decode-length sweep | 144 / 144 ✅ |
| Gemma 4 | 28 / 28 ✅ |
| **Completed** | **381** |
| **Remaining** | **0** ✅ |

All four extended phases are complete as of 2026-04-20. The underlying 14-model baseline in `results/` (152 files, 100% success) has been in place since the March run.

## Notes on the Gemma 4 SGLang unblock

- **Symptom (`sglang:latest`, Apr-09 tag):** weight load fails with
  `ValueError: No module or parameter named 'model.language_model.layers.15.self_attn.k_norm' in TransformersMultiModalForCausalLM`.
  Gemma 3/4 introduced QK-norm (`k_norm` / `q_norm`); SGLang's generic
  Transformers wrapper does not map those params. Upgrading
  `transformers` inside the container does not help — the gap is on the
  SGLang side.
- **Fix:** pin `GEMMA4_SGLANG_IMAGE="lmsysorg/sglang:dev-cu13"` (Apr-16
  snapshot off `main`). That image ships the native Gemma 4 model class
  and loads the weights cleanly. All 38 previously-deferred Gemma 4
  SGLang cells were run against `dev-cu13`.

## Blocked / out of scope

- **Llama-3.1-8B SGLang Eagle3** (Part 0, 2 runs): pinned to
  `lmsysorg/sglang:nightly-dev-cu13-20260321-94194537`, which is no
  longer available on Docker Hub. Needs a new SGLang nightly pin before
  retrying. Not handled by `run_new_benchmarks.sh`.

## Commands

```bash
# Re-audit at any time (just tallies result files on disk)
for d in results results_variance results_concurrency64 results_decode_sweep; do
  echo "$d: $(find "$d" -name '*.json' -not -name '*manifest*' | wc -l) files"
done

# Every phase is idempotent — this is a no-op once complete
bash scripts/run_new_benchmarks.sh --all

# Analysis — regenerate summary reports from the full result set
conda run -n base python -m analysis.variance_analysis       --results-dir results_variance
conda run -n base python -m analysis.tpot_analysis           --results-dir results_variance
conda run -n base python -m analysis.decode_length_analysis  --results-dir results_decode_sweep
conda run -n base python -m analysis.generate_final_benchmark_report
```
