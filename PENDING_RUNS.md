# Pending Benchmark Runs

## What's missing

**Main benchmark matrix:** complete.

Gemma 9B on vLLM was the last missing leg in the original validated matrix, and it has now completed.

**Optional follow-up:** Phi-3 mini on SGLang is now potentially recoverable with a custom workaround config (`docker-compose.phi3mini-sglang-a10g.yml`), but it has not been rerun into the published benchmark set yet.

## Automation scripts

Use these repo-root scripts to run and verify the remaining work:

```bash
./pending_run_gemma9b_vllm.sh
./pending_run_gemma9b_vllm_verify.sh
```

Known-good compose overrides for this machine:

```bash
docker compose -f docker-compose.yml -f docker-compose.gemma9b-vllm-a10g.yml --profile vllm up -d vllm
docker compose -f docker-compose.yml -f docker-compose.phi3mini-sglang-a10g.yml --profile sglang up -d sglang
```

Optional Phi-3 mini SGLang rerun helpers:

```bash
./pending_run_phi3mini_sglang.sh
./pending_run_phi3mini_sglang_verify.sh
```

## Debugging the 404 on /v1/completions

The benchmark client uses `/v1/completions` (legacy completions endpoint). If vLLM returns 404, check:

```bash
# 1. Check which endpoints are available
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# 2. Test the completions endpoint directly
curl -s http://localhost:8000/v1/completions \
  -X POST -H "Content-Type: application/json" \
  -d '{"model":"google/gemma-2-9b-it","prompt":"hello","max_tokens":5}'

# 3. Test the chat completions endpoint (newer vLLM default)
curl -s http://localhost:8000/v1/chat/completions \
  -X POST -H "Content-Type: application/json" \
  -d '{"model":"google/gemma-2-9b-it","messages":[{"role":"user","content":"hello"}],"max_tokens":5}'

# 4. Check vLLM logs
docker compose logs vllm --tail 50
```

If `/v1/completions` returns 404 but `/v1/chat/completions` works, the vLLM version only exposes the chat endpoint by default. Fix by adding `--disable-frontend-multiprocessing` to the vLLM command in docker-compose.yml, or check if the vLLM image version supports the legacy completions endpoint.

## Run commands

### Step 1: Start vLLM with Gemma 9B (tuned for A10G 24 GB)

```bash
cd ~/inference-engine-benchmark-system

# Gemma 9B is tight on 24 GB — reduce context and increase memory utilization
MODEL=google/gemma-2-9b-it \
MAX_MODEL_LEN=4096 \
docker compose --profile vllm up -d vllm

# Wait for model to load (can take 2-3 minutes)
docker compose logs vllm -f
# Wait until you see "Uvicorn running on http://0.0.0.0:8000"
# Then Ctrl+C out of logs
```

### Step 2: Verify health and endpoint

```bash
curl http://localhost:8000/health
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# Quick test to confirm completions endpoint works
curl -s http://localhost:8000/v1/completions \
  -X POST -H "Content-Type: application/json" \
  -d '{"model":"google/gemma-2-9b-it","prompt":"The capital of France is","max_tokens":10}'
```

If the completions test returns a 404, try:

```bash
# Option A: Add --disable-frontend-multiprocessing
docker compose --profile vllm down
# Edit docker-compose.yml, add to vllm command section:
#   - "--disable-frontend-multiprocessing"
MODEL=google/gemma-2-9b-it MAX_MODEL_LEN=4096 docker compose --profile vllm up -d vllm

# Option B: If still fails, try lowering memory further
MODEL=google/gemma-2-9b-it MAX_MODEL_LEN=2048 docker compose --profile vllm up -d vllm
```

### Step 3: Run the benchmark

```bash
python run_experiment.py matrix \
  --scenarios single_request_latency,throughput_ramp,long_context_stress,prefix_sharing_benefit,structured_generation_speed \
  --engines vllm \
  --model google/gemma-2-9b-it \
  --output-dir results/gemma-2-9b-it \
  --iterations 2 \
  --cooldown-seconds 120
```

### Step 4: Verify output

```bash
ls results/gemma-2-9b-it/*VLLMClient*.json | wc -l
# Should show 10 (5 scenarios x 2 iterations)
```

### Step 5: Stop engine and push

```bash
docker compose --profile vllm down

git add results/gemma-2-9b-it/
git commit -m "data: add Gemma 9B vLLM benchmark results"
git push origin main
```

## If vLLM OOMs on Gemma 9B

Gemma 9B (9.2B params, ~18 GB in fp16) leaves very little headroom on 24 GB A10G.

Fallback options in order:
1. `MAX_MODEL_LEN=2048` with `gpu-memory-utilization=0.95`
2. Run with `--dtype float16 --enforce-eager` (disables CUDA graphs, saves memory)
3. Skip Gemma 9B vLLM entirely — the SGLang data is already collected, and the report will note vLLM was excluded due to memory constraints (same as Phi-3 mini on SGLang)

## After pushing results

I'll regenerate:
- `results/gemma-2-9b-it/report.html` (per-model HTML with charts)
- `results/gemma-2-9b-it/final_report.md` (per-model markdown)
- Cross-model summary report using `generate_final_benchmark_report.py`
