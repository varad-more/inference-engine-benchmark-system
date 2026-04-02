# Benchmark Execution Guide

## Prerequisites

1. **GPU:** NVIDIA A10G 24 GB (or equivalent ≥16 GB)
   ```bash
   nvidia-smi
   ```
2. **Docker with GPU access:**
   ```bash
   docker info | grep -i gpu
   ```
3. **HuggingFace token** in `.env` (required for Llama, Gemma, Mistral):
   ```bash
   echo "HUGGING_FACE_HUB_TOKEN=hf_YOUR_TOKEN" >> .env
   ```
4. **Model cache directory:**
   ```bash
   mkdir -p model-cache
   ```

---

## Running the Full Suite

```bash
chmod +x scripts/run_all_benchmarks.sh
tmux new -s benchmark
./scripts/run_all_benchmarks.sh 2>&1 | tee logs/run_$(date +%Y%m%dT%H%M%S).log
```

Detach: `Ctrl+B` then `D` — Reattach: `tmux attach -t benchmark`

**Force re-run** (ignore existing results):
```bash
./scripts/run_all_benchmarks.sh --force
```

The script runs 14 models × 5 scenarios × 2 engines sequentially — one engine at a time to avoid VRAM contention. It **automatically skips** any model that already has 10+ result files, so it's safe to resume after interruption.

---

## Speculative Decoding

Tested on Llama 3.1 8B and Qwen3 8B after all baseline runs complete.

| What ran | Status |
|---|---|
| Ngram — vLLM + SGLang on Llama 3.1 8B | Done |
| Ngram — vLLM + SGLang on Qwen3 8B | Done |
| Eagle3 — vLLM on Llama 3.1 8B | Done |
| Eagle3 — SGLang on Llama 3.1 8B | OOM on A10G 24 GB |
| Eagle3 — Qwen3 8B (both engines) | Draft model not yet published |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| OOM on startup | Reduce `--gpu-memory-utilization` or `--max-model-len` in the script |
| SGLang crashes | Check `docker logs sglang-server` — try adding `--disable-flashinfer` |
| Model download stuck | Verify HF token in `.env` and check `docker logs vllm-server` |
| Gemma 3 4B vLLM OOM | Requires `--enforce-eager --disable-frontend-multiprocessing` |
| Eagle3 OOM | Needs `--gpu-memory-utilization 0.95 --enforce-eager --max-model-len 2048` |

---

## After Completion

```bash
# Verify result counts
for d in results/*/; do
    echo "$(basename $d): $(find $d -name '*.json' -not -name '*manifest*' | wc -l) results"
done

# Generate reports
conda run -n base python analysis/generate_final_benchmark_report.py
```
