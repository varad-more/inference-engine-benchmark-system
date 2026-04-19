# Benchmark Run Status

Snapshot of completed vs. remaining runs across all phases.

- **Snapshot time:** 2026-04-19 ~17:35 UTC
- **Units:** each number is an actual result JSON file on disk.
- **Source of truth:** `results/`, `results_variance/`, `results_concurrency64/`, `results_decode_sweep/`.

## Phase 1 — Variance
Target: **25 runs per engine per model** (5 scenarios × 5 iterations).

| Model | vLLM | SGLang | Remaining |
|---|---|---|---|
| gemma-2-2b-it | 25/25 ✅ | 25/25 ✅ | 0 |
| phi-4-mini-instruct | 25/25 ✅ | 25/25 ✅ | 0 |
| llama-3-1-8b-instruct | 25/25 ✅ | 26/25 ✅ | 0 |
| gemma-3-4b-it | 25/25 ✅ | 25/25 ✅ | 0 |
| **Subtotal** | | | **0 runs** ✅ |

**Phase 1 complete as of 2026-04-19 ~11:30 UTC.**
Output dir: `results_variance/`.

## Phase 2 — Concurrency-64
Target: **1 run per engine per model** (single `throughput_ramp_extended` iteration).

| Model | vLLM | SGLang | Remaining |
|---|---|---|---|
| llama-3-1-8b-instruct | 1/1 ✅ | 1/1 ✅ | 0 |
| qwen3-8b | 1/1 ✅ | 1/1 ✅ | 0 |
| mistral-7b-instruct-v0-3 | 1/1 ✅ | 1/1 ✅ | 0 |
| gemma-2-9b-it | 1/1 ✅ | 1/1 ✅ | 0 |
| **Subtotal** | | | **0 runs** ✅ |

Output dir: `results_concurrency64/`.

## Phase 3 — Decode-length sweep
Target: **12 runs per engine per model** (4 lengths × 3 iterations).

| Model | vLLM | SGLang | Remaining |
|---|---|---|---|
| gemma-2-2b-it | 12/12 ✅ | 12/12 ✅ | 0 |
| phi-4-mini-instruct | 12/12 ✅ | 12/12 ✅ | 0 |
| llama-3-1-8b-instruct | 12/12 ✅ | 12/12 ✅ | 0 |
| gemma-3-4b-it | 12/12 ✅ | 11/12 🟢 finishing | 1 |
| gemma-4-e2b-it *(deferred — disk)* | 0/12 | 0/12 | 24 |
| gemma-4-e4b-it *(deferred — disk)* | 0/12 | 0/12 | 24 |
| **Subtotal** | | | **49 runs** (1 in-flight + 48 deferred) |

Output dir: `results_decode_sweep/`.

## Phase 4 — Gemma 4 baseline + ngram spec-dec
Target: **14 runs per model** (10 baseline + 4 ngram).
- Baseline: 5 scenarios × 2 engines × 1 iter = 10
- Ngram spec-dec: 2 scenarios × 2 engines × 1 iter = 4

| Model | vLLM | SGLang | vllm-ngram | sglang-ngram | Remaining |
|---|---|---|---|---|---|
| gemma-4-e2b-it | 0/5 | 0/5 | 0/2 | 0/2 | 14 |
| gemma-4-e4b-it | 0/5 | 0/5 | 0/2 | 0/2 | 14 |
| **Subtotal** | | | | | **28 runs** |

Output dir: `results/`.

## Grand totals

| Category | Runs |
|---|---|
| Completed across Phases 1–3 | 385 |
| Remaining Phase 1 | 0 ✅ |
| Remaining Phase 2 | 0 ✅ |
| Remaining Phase 3 | 49 (1 in-flight + 48 deferred Gemma 4) |
| Remaining Phase 4 | 28 (deferred Gemma 4) |
| **Total remaining** | **77** |

### Status of the 77 remaining

- 🟢 **In-flight (tmux `bench`)**: 1 run — gemma-3-4b SGLang decode_length_sweep_4096 iter 2/2 (~5 min ETA). After this, Phase 3 non-Gemma 4 is complete.
- 🔴 **Deferred — disk pressure**: 76 Gemma 4 runs (no space to pull `lmsysorg/sglang:latest`, ~50 GB; `vllm/vllm-openai:latest` already local at 32 GB).
  - Phase 3 Gemma 4 sweep: 48 (24 vLLM + 24 SGLang)
  - Phase 4 Gemma 4 baseline + ngram: 28 (14 vLLM + 14 SGLang)

### Resume commands for Gemma 4 (after freeing disk)

```bash
# If only vLLM Gemma 4 is desired (≈7.5 GB disk headroom for pip install cache):
SKIP_GEMMA4_SGLANG=1 bash scripts/run_new_benchmarks.sh --phase4
SKIP_GEMMA4_SGLANG=1 bash scripts/run_new_benchmarks.sh --phase3

# Full Gemma 4 (needs ~55 GB free to pull sglang:latest):
bash scripts/run_new_benchmarks.sh --phase4
bash scripts/run_new_benchmarks.sh --phase3
```

### Blocked / out of scope

- **Llama-3.1-8B SGLang Eagle3** (Part 0, 2 runs): pinned to `lmsysorg/sglang:nightly-dev-cu13-20260321-94194537`, which is no longer available on Docker Hub. Needs a new SGLang nightly pin before retrying. Not handled by `run_new_benchmarks.sh`.
- **Disk**: `/` was at 95% (8.7 GB free) when Phase 4 Gemma 4 tried to pull `sglang:latest`. All Gemma 4 work deferred until disk is reclaimed. The script now honors two env vars to skip Gemma 4 cleanly:
  - `SKIP_GEMMA4=1` — skip all Gemma 4 entries in Phase 3
  - `SKIP_GEMMA4_SGLANG=1` — run Gemma 4 vLLM only in Phase 3 and Phase 4

## Commands

```bash
# Re-audit at any time
bash scripts/pending.sh --audit-only

# Resume currently running logic (Phase 1 + Phase 3 top-ups, idempotent)
bash scripts/run_new_benchmarks.sh

# Trigger Gemma 4 Phase 3 decode sweep (48 runs)
bash scripts/run_new_benchmarks.sh --phase3

# Trigger Gemma 4 Phase 4 baseline + ngram (28 runs)
bash scripts/run_new_benchmarks.sh --phase4

# Everything (Phase 1–4, resume-safe)
bash scripts/run_new_benchmarks.sh --all
```
