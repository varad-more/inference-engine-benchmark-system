# Benchmark Execution Guide

Two entry-point scripts:

- **`scripts/run_all_benchmarks.sh`** — the 14-model baseline suite (5 scenarios
  × 2 engines, sequential, one engine at a time). This is the set that
  produces the 152-file headline result table. Auto-skips any model with
  ≥10 existing result files; use `--force` to re-run.
- **`scripts/run_new_benchmarks.sh`** — the extended phases (variance subset,
  concurrency-64 ramp, decode-length sweep, Gemma 4 baseline + ngram
  spec-dec). Every phase block is idempotent — a re-launch skips cells
  whose result file already exists.

Both manage Docker containers, wait for engine health, run the benchmark
matrix, and clean up after themselves.

## Prerequisites

1. **GPU:** NVIDIA A10G 24 GB (or equivalent ≥16 GB) — `nvidia-smi` to confirm.
2. **Docker with GPU access:** `docker info | grep -i gpu`.
3. **HuggingFace token** in `.env` (required for Llama, Gemma, Mistral):
   ```bash
   echo "HUGGING_FACE_HUB_TOKEN=hf_YOUR_TOKEN" >> .env
   ```
4. **Model cache directory:** `mkdir -p model-cache`.
5. **Disk:** ~120 GB free for all images + weights. `sglang:dev-cu13`
   (used for Gemma 4) is ~55 GB on its own.

## Running everything

```bash
chmod +x scripts/run_new_benchmarks.sh
tmux new -s bench
bash scripts/run_new_benchmarks.sh --all 2>&1 | tee logs/run_$(date +%Y%m%dT%H%M%S).log
```

Detach: `Ctrl+B D` — reattach: `tmux attach -t bench`.

## Running a single phase

```bash
bash scripts/run_new_benchmarks.sh --variance   # variance subset (4 models × 5 iter)
bash scripts/run_new_benchmarks.sh --concurrency   # concurrency-64 ramp on 7–9B
bash scripts/run_new_benchmarks.sh --decode-sweep   # decode-length sweep (incl. Gemma 4)
bash scripts/run_new_benchmarks.sh --gemma4   # Gemma 4 baseline + ngram spec-dec
```

Each phase writes to its own directory (`results_variance/`,
`results_concurrency64/`, `results_decode_sweep/`, `results/`) so output is
never overwritten.

## Env-var escape hatches

| Variable | Effect |
|---|---|
| `SKIP_GEMMA4=1` | Skip all Gemma 4 entries in the decode-sweep block. |
| `SKIP_GEMMA4_SGLANG=1` | Run Gemma 4 only on vLLM in phases 3 and 4 (handy when disk is tight — saves the ~55 GB `sglang:dev-cu13` pull). |

## Speculative decoding

Covered by the gemma4 block. Ngram works on Llama 3.1 8B, Qwen3 8B, Gemma 4 E2B, and
Gemma 4 E4B across both engines. Eagle3 works on Llama 3.1 8B with vLLM;
SGLang OOMs on A10G 24 GB, and the Qwen3 8B draft model has not been
published yet.

## Troubleshooting

| Symptom | Fix |
|---|---|
| OOM on engine startup | Reduce `--gpu-memory-utilization` or `--max-model-len` on the affected model's block. |
| SGLang crashes | `docker logs bench-sglang` — try `--disable-flashinfer` or `--disable-cuda-graph`. |
| Model download stuck | Check HF token in `.env` and `docker logs bench-vllm`. |
| Gemma 3 4B vLLM OOM | Needs `--enforce-eager --disable-frontend-multiprocessing`. |
| Gemma 4 SGLang `k_norm` ValueError | Use `lmsysorg/sglang:dev-cu13` (not `:latest`) — already pinned in the script. |
| Eagle3 OOM | Needs `--gpu-memory-utilization 0.95 --enforce-eager --max-model-len 2048`. |

## After completion

```bash
# Verify result counts
for d in results/*/; do
    echo "$(basename "$d"): $(find "$d" -name '*.json' -not -name '*manifest*' | wc -l) files"
done

# Regenerate the summary reports
conda run -n base python -m analysis.variance_analysis       --results-dir results_variance
conda run -n base python -m analysis.tpot_analysis           --results-dir results_variance
conda run -n base python -m analysis.decode_length_analysis  --results-dir results_decode_sweep
conda run -n base python -m analysis.generate_final_benchmark_report
```
