#!/usr/bin/env bash
# =============================================================================
# Phase A — Remaining Benchmark Runs
#
# COMPLETED:
#   - ibm-granite/granite-3.3-8b-instruct  (both engines done)
#   - deepseek-ai/DeepSeek-R1-Distill-Llama-8B (both engines done)
#   - google/gemma-3-4b-it (both engines done)
#
# SKIPPED — too large for A10G 24GB:
#   - Qwen/Qwen3-30B-A3B    (30B params = ~60GB at bf16)
#   - google/gemma-3-12b-it  (12B params = ~24GB weights alone, no room for KV cache)
#
# SPECULATIVE DECODING — pending:
#   - meta-llama/Llama-3.1-8B-Instruct  vllm-eagle3 + sglang-eagle3
#   (Qwen3-8B Eagle3 blocked — no draft model published yet)
#
# NOTE: Uses 'docker run' directly instead of docker-compose to avoid
#       persistent Docker network issues on this host.
#
# Usage:
#   chmod +x scripts/run_phase_a_pending.sh
#   tmux new -s benchmark
#   ./scripts/run_phase_a_pending.sh 2>&1 | tee logs/phase_a_$(date +%Y%m%dT%H%M%S).log
#
# Prerequisites:
#   - HUGGING_FACE_HUB_TOKEN set in .env (Llama 3.1 is gated)
#   - Docker running with GPU access
#   - model-cache/ directory exists
# =============================================================================

set +e

PYTHON="conda run --no-capture-output -n base python"
SPECDEC_SCENARIOS="single_request_latency,throughput_ramp"
COOLDOWN=10
RESULTS_DIR="results"
ERRORS=()

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.18.0-cu130}"
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:nightly-dev-cu13-20260321-94194537}"

log() {
    echo ""
    echo "========================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "========================================================================"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
    ERRORS+=("$1")
}

# ── Load HF token from .env ──────────────────────────────────────────────────
HF_TOKEN=""
if [ -f .env ]; then
    HF_TOKEN=$(grep "^HUGGING_FACE_HUB_TOKEN=" .env | cut -d= -f2)
fi

# ── Preflight checks ─────────────────────────────────────────────────────────
log "PREFLIGHT CHECKS"

echo -n "  Python: "
$PYTHON --version 2>&1
if [ $? -ne 0 ]; then
    error "conda python not found — check PYTHON variable"
    exit 1
fi

echo -n "  Docker: "
docker version --format '{{.Server.Version}}' 2>&1
if [ $? -ne 0 ]; then
    error "Docker not running"
    exit 1
fi

echo -n "  GPU: "
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 \
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 \
    || echo "  (GPU check failed — continuing anyway)"

echo -n "  HF token: "
if [ -n "$HF_TOKEN" ]; then echo "found"; else echo "NOT FOUND — Llama 3.1 will fail"; fi

echo -n "  model-cache/: "
if [ -d "model-cache" ]; then echo "exists"; else mkdir -p model-cache && echo "created"; fi

# Kill any leftover containers from previous runs
echo -n "  Cleaning up old containers: "
docker rm -f vllm-eagle3-server sglang-eagle3-server 2>/dev/null || true
echo "done"

log "PREFLIGHT COMPLETE"

mkdir -p logs

# ── Helper: cleanup any running benchmark container ──────────────────────────
cleanup_container() {
    local name="$1"
    docker rm -f "$name" 2>/dev/null
    sleep 2
}

# ── vLLM Eagle3 via docker run ───────────────────────────────────────────────
run_vllm_eagle3() {
    local model="$1"
    local draft_model="$2"
    local sleep_time="$3"
    local container="vllm-eagle3-server"

    log "  Starting vLLM Eagle3 for ${model}"
    cleanup_container "$container"

    docker run -d \
        --name "$container" \
        --gpus '"device=0"' \
        -p 8000:8000 \
        --shm-size 10g \
        -v "$(pwd)/model-cache:/root/.cache/huggingface" \
        -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
        -e "CUDA_VISIBLE_DEVICES=0" \
        "$VLLM_IMAGE" \
        --model "$model" \
        --host 0.0.0.0 \
        --port 8000 \
        --max-model-len 2048 \
        --gpu-memory-utilization 0.95 \
        --enforce-eager \
        --served-model-name "$model" \
        --speculative-config "{\"method\":\"eagle3\",\"model\":\"${draft_model}\",\"num_speculative_tokens\":3}"

    if [ $? -ne 0 ]; then
        error "vllm-eagle3 docker run failed for ${model}"
        return 1
    fi

    log "  Waiting ${sleep_time}s for models to load..."
    sleep "$sleep_time"

    if ! docker ps --filter "name=${container}" --format '{{.Status}}' | grep -q "Up"; then
        error "vllm-eagle3 container died during startup for ${model}"
        echo "  Container logs (last 50 lines):"
        docker logs "$container" --tail 50 2>&1
        docker rm -f "$container" 2>/dev/null
        return 1
    fi

    $PYTHON run_experiment.py matrix \
        --scenarios "$SPECDEC_SCENARIOS" \
        --engines "vllm-eagle3" \
        -m "$model" \
        --cooldown-seconds "$COOLDOWN" \
        --output-dir "$RESULTS_DIR"
    local rc=$?

    if [ $rc -ne 0 ]; then
        error "vllm-eagle3 benchmark failed for ${model} (exit code ${rc})"
        docker logs "$container" --tail 30 2>&1
    fi

    docker rm -f "$container" 2>/dev/null
    sleep 15
    return $rc
}

# ── SGLang Eagle3 via docker run ─────────────────────────────────────────────
run_sglang_eagle3() {
    local model="$1"
    local draft_model="$2"
    local sleep_time="$3"
    local container="sglang-eagle3-server"

    log "  Starting SGLang Eagle3 for ${model}"
    cleanup_container "$container"

    docker run -d \
        --name "$container" \
        --gpus '"device=0"' \
        -p 8001:8001 \
        --shm-size 10g \
        -v "$(pwd)/model-cache:/root/.cache/huggingface" \
        -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
        -e "CUDA_VISIBLE_DEVICES=0" \
        -e "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1" \
        "$SGLANG_IMAGE" \
        python -m sglang.launch_server \
        --model-path "$model" \
        --host 0.0.0.0 \
        --port 8001 \
        --mem-fraction-static 0.65 \
        --context-length 2048 \
        --speculative-algorithm EAGLE3 \
        --speculative-draft-model-path "$draft_model" \
        --speculative-num-steps 3 \
        --speculative-eagle-topk 4 \
        --speculative-num-draft-tokens 16

    if [ $? -ne 0 ]; then
        error "sglang-eagle3 docker run failed for ${model}"
        return 1
    fi

    log "  Waiting ${sleep_time}s for models to load..."
    sleep "$sleep_time"

    if ! docker ps --filter "name=${container}" --format '{{.Status}}' | grep -q "Up"; then
        error "sglang-eagle3 container died during startup for ${model}"
        echo "  Container logs (last 50 lines):"
        docker logs "$container" --tail 50 2>&1
        docker rm -f "$container" 2>/dev/null
        return 1
    fi

    $PYTHON run_experiment.py matrix \
        --scenarios "$SPECDEC_SCENARIOS" \
        --engines "sglang-eagle3" \
        -m "$model" \
        --cooldown-seconds "$COOLDOWN" \
        --output-dir "$RESULTS_DIR"
    local rc=$?

    if [ $rc -ne 0 ]; then
        error "sglang-eagle3 benchmark failed for ${model} (exit code ${rc})"
        docker logs "$container" --tail 30 2>&1
    fi

    docker rm -f "$container" 2>/dev/null
    sleep 15
    return $rc
}

# =============================================================================
#  PART 1 — BASELINE RUNS (all complete)
# =============================================================================
log "PART 1: BASELINE RUNS — all complete, skipping"
log "SKIP: Qwen3-30B-A3B — 30B params, too large for A10G 24GB"
log "SKIP: Gemma 3 12B — 12B params, weights alone fill 24GB"

# =============================================================================
#  PART 2 — SPECULATIVE DECODING (docker run, no compose)
# =============================================================================
log "PART 2: SPECULATIVE DECODING RUNS"

# ---------------------------------------------------------------------------
# Qwen3-8B spec-dec — all ngram/baseline variants already done
# Eagle3 BLOCKED: RedHatAI/Qwen3-8B-speculator.eagle3 not yet published
# ---------------------------------------------------------------------------
log "SKIP SPEC-DEC: Qwen3-8B Eagle3 — no draft model available yet"

# ---------------------------------------------------------------------------
# Llama 3.1 8B Eagle3
# Draft models:
#   vLLM  : RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3
#   SGLang: jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B
# MAX_MODEL_LEN=4096 keeps both main + draft models in 24GB VRAM
# 240s wait — two models load on startup
# ---------------------------------------------------------------------------
log "START SPEC-DEC: Llama 3.1 8B Eagle3"
LLAMA="meta-llama/Llama-3.1-8B-Instruct"

run_vllm_eagle3  "$LLAMA" "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3"          240
run_sglang_eagle3 "$LLAMA" "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B" 240

log "DONE SPEC-DEC: Llama 3.1 8B Eagle3"

# =============================================================================
#  FINAL SUMMARY
# =============================================================================
log "ALL RUNS COMPLETE"

echo ""
echo "Results per model:"
for d in ${RESULTS_DIR}/*/; do
    name=$(basename "$d")
    count=$(find "$d" -name "*.json" -not -name "*manifest*" 2>/dev/null | wc -l | tr -d ' ')
    printf "  %-40s %3s files\n" "$name" "$count"
done

echo ""
if [ ${#ERRORS[@]} -eq 0 ]; then
    echo "STATUS: ALL PASSED — no errors"
else
    echo "STATUS: ${#ERRORS[@]} ERROR(S) ENCOUNTERED:"
    for i in "${!ERRORS[@]}"; do
        echo "  $((i+1)). ${ERRORS[$i]}"
    done
fi

echo ""
echo "Next steps:"
echo "  1. $PYTHON analysis/generate_final_benchmark_report.py"
echo "  2. $PYTHON run_experiment.py final-report --output reports/phase_a_report.md"
