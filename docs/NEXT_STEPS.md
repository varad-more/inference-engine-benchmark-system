# Next Steps — Benchmark Execution Plan

_Last updated: 2026-04-09_

---

## Current State

| Item | Status |
|---|---|
| Baseline runs (14 models × 5 scenarios × 2 engines) | **Complete** — 152 result files in `results/` |
| Speculative decoding — Llama 3.1 8B Ngram + Eagle3 | **Complete** — in `results/llama-3-1-8b-instruct/` |
| Speculative decoding — Qwen3 8B Ngram | **Complete** — in `results/qwen3-8b/` |
| Phase 1 — Variance subset (4 models × 5 scenarios × 2 engines × 5 iter) | **Complete** — 201 files in `results_variance/` |
| Phase 2 — Concurrency-64 | **Partial** — Llama 3.1 8B done (6 files); 3 models pending (after Phase 3) |
| Phase 3 — Decode-length sweep | **Not started** — run next |
| Gemma 4 benchmarks | **Not started** — run after Phases 2 & 3 |
| Variance/TPOT/goodput analysis on Phase 1 results | **Not run** — Phase 1 data ready |
| Decode-length analysis | **Not run** — waiting on Phase 3 |

---

## Step 1 — Run Phase 3 (Decode-Length Sweep)

Phase 3 sweeps `max_output_tokens` ∈ {64, 256, 1024, 4096} at fixed ~512-token prompts to separate prefill-bound vs decode-bound behaviour.

```bash
nohup bash scripts/run_new_benchmarks.sh --phase3 \
    2>&1 | tee logs/phase3_$(date +%Y%m%dT%H%M%S).log &
```

What it runs:
- Models: gemma-2-2b-it, Phi-4-mini, Llama-3.1-8B, gemma-3-4b-it
- Engines: vLLM + SGLang (Docker-managed)
- Scenarios: `decode_length_sweep_{64,256,1024,4096}` — 3 iterations each
- Output: `results_decode_sweep/`
- Expected: 96 files (4 models × 4 lengths × 2 engines × 3 iter)
- Cooldown: 30s between scenarios, 600s Docker health-check timeout

Monitor progress:
```bash
tail -f logs/phase3_*.log
```

---

## Step 2 — Resume Phase 2 (Concurrency-64)

After Phase 3, resume Phase 2 for the 3 remaining models. Llama 3.1 8B is already done — the script skips it automatically.

```bash
nohup bash scripts/run_new_benchmarks.sh --phase2 \
    2>&1 | tee logs/phase2_resume_$(date +%Y%m%dT%H%M%S).log &
```

What it runs:
- Models: Qwen3-8B, Mistral-7B-Instruct-v0.3, gemma-2-9b-it (Llama 3.1 8B already done)
- Engines: vLLM + SGLang (Docker-managed)
- Scenario: `throughput_ramp_extended` — concurrency [1, 4, 8, 16, 32, 64], 150 requests/level
- Output: `results_concurrency64/`
- Expected after resume: 24 total (6 existing + 18 new)

---

## Step 3 — Run Analysis

These can be run as soon as Phase 1 data is available (Phase 1 is complete now).

### Phase 1 variance analysis (ready now):
```bash
conda run -n base python -m analysis.variance_analysis \
    --results-dir results_variance

conda run -n base python -m analysis.tpot_analysis \
    --results-dir results_variance

conda run -n base python -m analysis.goodput \
    --results-dir results_variance
```

### After Phase 3 (decode sweep):
```bash
conda run -n base python -m analysis.decode_length_analysis \
    --results-dir results_decode_sweep
```

### Final consolidated report (after all phases):
```bash
conda run -n base python run_experiment.py final-report \
    --output reports/extended_benchmark_report.md
```

---

## Step 4 — Run Gemma 4 Benchmarks

Run after Phases 2 & 3 complete (GPU needs to be free).

```bash
nohup bash scripts/run_gemma4_benchmarks.sh \
    2>&1 | tee logs/gemma4_$(date +%Y%m%dT%H%M%S).log &
```

What this runs:
- Models: `google/gemma-4-E2B-it` (~4 GB VRAM), `google/gemma-4-E4B-it` (~8 GB VRAM)
- Part 1: Baseline — both engines, all 5 scenarios
- Part 2: Ngram speculative decoding — both engines, latency + throughput scenarios
- Part 3: Eagle3 — skipped (no Gemma 4 draft model published yet)
- Output: `results/gemma-4-e2b-it/`, `results/gemma-4-e4b-it/`

Requires HF token in `.env`. Runtime estimate: ~4–6 hours.

---

## Step 5 — Update README

After analysis reports are generated:

1. Copy key numbers from `reports/variance_analysis.md` into the README under **Section 6 (TPOT & Goodput)**.
2. Add decode-length sweep findings as a new **Section 8**.
3. Add Gemma 4 results to the **Models Tested** table.
4. Update the **Benchmark Execution Status** table at the top.

---

## Quick reference — result file counts

```bash
for dir in results results_variance results_concurrency64 results_decode_sweep; do
    count=$(find "$dir" -name "*.json" -not -name "*manifest*" 2>/dev/null | wc -l)
    printf "  %-30s %4s files\n" "$dir/" "$count"
done
```

Expected final counts:
- `results/` — 152 (complete)
- `results_variance/` — 201 (complete)
- `results_concurrency64/` — 24 (6 done; 18 pending Phase 2 resume)
- `results_decode_sweep/` — 96 (pending Phase 3)

---

## If a run breaks mid-way

The script is idempotent per model/engine block — it does not skip existing results by
default. If it stops partway through, restart for that specific phase:

```bash
# Restart Phase 3 only
nohup bash scripts/run_new_benchmarks.sh --phase3 \
    2>&1 | tee logs/phase3_restart_$(date +%Y%m%dT%H%M%S).log &
```

Results already written are preserved — the run appends, not overwrites.
