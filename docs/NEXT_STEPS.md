# Next Steps — Benchmark Execution Plan

_Last updated: 2026-04-08_

---

## Current State

| Item | Status |
|---|---|
| Baseline runs (14 models × 5 scenarios × 2 engines) | **Complete** — 152 result files in `results/` |
| Speculative decoding — Llama 3.1 8B Ngram + Eagle3 | **Complete** — in `results/llama-3-1-8b-instruct/` |
| Speculative decoding — Qwen3 8B Ngram | **Complete** — in `results/qwen3-8b/` |
| HuggingFace token | **Added** — `.env` populated |
| Phase 1 — Variance subset (vLLM block) | **In progress** — Phi-4-mini running, PID 21523 |
| Phase 1 — Variance subset (SGLang block) | Queued — starts automatically after vLLM |
| Phase 2 — Concurrency-64 | Queued — starts automatically after Phase 1 |
| Phase 3 — Decode-length sweep | Queued — starts automatically after Phase 2 |
| Gemma 4 benchmarks | **Not started** — run after current suite finishes |
| Analysis on new results | **Not started** — run after each phase completes |

---

## Step 1 — Wait for current run to finish (automatic)

`scripts/run_new_benchmarks.sh` is running in the background and handles all three phases
sequentially without intervention. Check progress at any time:

```bash
tail -f logs/new_benchmarks_*.log
```

Expected runtime: **~20–28 hours** total (300 s cooldowns between every iteration).

What it runs (now with HF token — full model coverage):

| Phase | Models | Engines | Scenarios | Iters | Output |
|---|---|---|---|---|---|
| 1 — Variance | gemma-2-2b-it, Phi-4-mini, Llama-3.1-8B, gemma-3-4b-it | vllm + sglang | 5 baseline | 5× | `results_variance/` |
| 2 — Concurrency-64 | Llama-3.1-8B, Qwen3-8B, Mistral-7B, gemma-2-9b-it | vllm + sglang | throughput_ramp_extended | 3× | `results_concurrency64/` |
| 3 — Decode sweep | gemma-2-2b-it, Phi-4-mini, Llama-3.1-8B | vllm + sglang | decode_length_sweep_{64,256,1024,4096} | 3× | `results_decode_sweep/` |

---

## Step 2 — Run analysis after each phase

These can be run as soon as their respective results directory has data. They do not need
the full phase to be complete — partial results produce partial reports.

### After Phase 1 (variance):
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

## Step 3 — Run Gemma 4 benchmarks

Run after the current suite finishes (GPU needs to be free).

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

Requires HF token (already set in `.env`). Runtime estimate: ~4–6 hours.

---

## Step 4 — Run speculative decoding on remaining models (optional)

Qwen3 8B Eagle3 is still blocked — no draft model published. Check:
```bash
# If RedHatAI/Qwen3-8B-speculator.eagle3 appears on HuggingFace:
bash scripts/run_phase_a_pending.sh
```

---

## Step 5 — Update README with new results

After analysis reports are generated:

1. Copy key numbers from `reports/variance_analysis.md` and `reports/tpot_analysis.md`
   into the README under **Section 6 (TPOT & Goodput)**.
2. Add Gemma 4 results to the **Models Tested** table and create a new
   **Section 8 (Gemma 4 Results)** in the README.
3. Update the **Summary** table at the top with any headline changes.

---

## Quick reference — result file counts

```bash
# Check progress at any time
for dir in results results_variance results_concurrency64 results_decode_sweep; do
    count=$(find "$dir" -name "*.json" -not -name "*manifest*" 2>/dev/null | wc -l)
    printf "  %-30s %4s files\n" "$dir/" "$count"
done
```

Expected final counts:
- `results/` — 152 (complete)
- `results_variance/` — 200 (4 models × 5 scenarios × 2 engines × 5 iters)
- `results_concurrency64/` — 24 (4 models × 1 scenario × 2 engines × 3 iters)
- `results_decode_sweep/` — 72 (3 models × 4 lengths × 2 engines × 3 iters)

---

## If the run breaks mid-way

The script is idempotent per model/engine block — it does not skip existing results by
default. If it stops partway through, just restart for the remaining phase:

```bash
# Restart a specific phase only
nohup bash scripts/run_new_benchmarks.sh --phase2 \
    2>&1 | tee logs/phase2_restart_$(date +%Y%m%dT%H%M%S).log &
```

Results already written are preserved — the run appends, not overwrites.
