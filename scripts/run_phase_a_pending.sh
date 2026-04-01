#!/usr/bin/env bash
# =============================================================================
# Phase A — Remaining Benchmark Runs
#
# Exact pending state (as of 2026-04-01):
#
# BASELINE — vLLM missing only (SGLang already done):
#   - ibm-granite/granite-3.3-8b-instruct
#   - deepseek-ai/DeepSeek-R1-Distill-Llama-8B
#
# BASELINE — both engines missing:
#   - Qwen/Qwen3-30B-A3B
#   - google/gemma-3-4b-it        (needs --enforce-eager for vLLM)
#   - google/gemma-3-12b-it       (needs --enforce-eager + 0.95 util + 2048 ctx)
#
# SPECULATIVE DECODING — pending:
#   - meta-llama/Llama-3.1-8B-Instruct  vllm-eagle3 + sglang-eagle3
#   (Qwen3-8B Eagle3 blocked — no draft model published yet)
#
# Usage:
#   chmod +x scripts/run_phase_a_pending.sh
#   tmux new -s benchmark
#   ./scripts/run_phase_a_pending.sh 2>&1 | tee logs/phase_a_$(date +%Y%m%dT%H%M%S).log
#
# Prerequisites:
#   - HUGGING_FACE_HUB_TOKEN set in .env (Gemma 3, Llama 3.1 are gated)
#   - Docker running with GPU access
#   - model-cache/ directory exists
# =============================================================================

set +e

PYTHON="conda run --no-capture-output -n base python"
SCENARIOS="single_request_latency,throughput_ramp,long_context_stress,prefix_sharing_benefit,structured_generation_speed"
SPECDEC_SCENARIOS="single_request_latency,throughput_ramp"
COOLDOWN=10
RESULTS_DIR="results"
ERRORS=()

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

echo -n "  .env HF token: "
if grep -q "HUGGING_FACE_HUB_TOKEN=hf_" .env 2>/dev/null; then
    echo "found"
else
    echo "NOT FOUND — gated models (Gemma 3, Llama 3.1) will fail to download"
fi

echo -n "  model-cache/: "
if [ -d "model-cache" ]; then echo "exists"; else mkdir -p model-cache && echo "created"; fi

echo -n "  results/ model folders: "
ls -d ${RESULTS_DIR}/*/ 2>/dev/null | wc -l | tr -d ' '

echo -n "  Cleaning up Docker state: "
docker compose down --remove-orphans 2>/dev/null
docker network prune -f 2>&1 | tail -1

log "PREFLIGHT COMPLETE"

mkdir -p logs

# ── Helper: start one engine, run scenarios, tear down ───────────────────────
# Args: model  profile  scenarios  sleep_time
run_engine() {
    local model="$1"
    local profile="$2"
    local scenarios="$3"
    local sleep_time="$4"

    log "  Starting ${profile} for ${model}"
    export MODEL="$model"

    # Clean up any leftover containers/networks before starting
    docker compose down --remove-orphans 2>/dev/null
    sleep 3

    docker compose --profile "$profile" up -d
    if [ $? -ne 0 ]; then
        error "${profile} docker up failed for ${model}"
        return 1
    fi

    log "  Waiting ${sleep_time}s for model to load..."
    sleep "$sleep_time"

    if ! docker compose --profile "$profile" ps | grep -q "Up\|running"; then
        error "${profile} container died during startup for ${model}"
        echo "  Container logs (last 50 lines):"
        docker compose --profile "$profile" logs --tail 50 2>&1
        docker compose down --remove-orphans 2>/dev/null
        return 1
    fi

    $PYTHON run_experiment.py matrix \
        --scenarios "$scenarios" \
        --engines "$profile" \
        -m "$model" \
        --cooldown-seconds "$COOLDOWN" \
        --output-dir "$RESULTS_DIR"
    local rc=$?

    if [ $rc -ne 0 ]; then
        error "${profile} benchmark failed for ${model} (exit code ${rc})"
        echo "  Last 30 lines of container logs:"
        docker compose --profile "$profile" logs --tail 30 2>&1
    fi

    docker compose down --remove-orphans 2>/dev/null
    sleep 15
    return $rc
}

# ── Helper: start vLLM via docker run (for models needing extra flags) ───────
# Args: model  container_name  sleep_time  [extra vllm args...]
run_vllm_direct() {
    local model="$1"
    local container_name="$2"
    local sleep_time="$3"
    shift 3
    local extra_args=("$@")

    local max_model_len="${MAX_MODEL_LEN:-8192}"
    local gpu_mem_util="${GPU_MEM_UTIL:-0.85}"

    log "  Starting vllm (direct) for ${model}"

    # Load HF token from .env
    local hf_token=""
    if [ -f .env ]; then
        hf_token=$(grep "^HUGGING_FACE_HUB_TOKEN=" .env | cut -d= -f2)
    fi

    # Clean up any leftover containers/networks
    docker compose down --remove-orphans 2>/dev/null
    docker rm -f "$container_name" 2>/dev/null
    sleep 3

    docker run -d \
        --name "$container_name" \
        --gpus '"device=0"' \
        -p 8000:8000 \
        --shm-size 10g \
        -v "$(pwd)/model-cache:/root/.cache/huggingface" \
        -e "HUGGING_FACE_HUB_TOKEN=${hf_token}" \
        -e "CUDA_VISIBLE_DEVICES=0" \
        "${VLLM_IMAGE:-vllm/vllm-openai:v0.18.0-cu130}" \
        --model "$model" \
        --host 0.0.0.0 \
        --port 8000 \
        --enable-prefix-caching \
        --max-model-len "$max_model_len" \
        --gpu-memory-utilization "$gpu_mem_util" \
        --served-model-name "$model" \
        "${extra_args[@]}"

    if [ $? -ne 0 ]; then
        error "vllm docker run failed for ${model}"
        return 1
    fi

    log "  Waiting ${sleep_time}s for model to load..."
    sleep "$sleep_time"

    if ! docker ps --filter "name=${container_name}" --format '{{.Status}}' | grep -q "Up"; then
        error "vllm container died during startup for ${model}"
        echo "  Container logs (last 50 lines):"
        docker logs "$container_name" --tail 50 2>&1
        docker rm -f "$container_name" 2>/dev/null
        return 1
    fi

    $PYTHON run_experiment.py matrix \
        --scenarios "$SCENARIOS" \
        --engines "vllm" \
        -m "$model" \
        --cooldown-seconds "$COOLDOWN" \
        --output-dir "$RESULTS_DIR"
    local rc=$?

    if [ $rc -ne 0 ]; then
        error "vllm benchmark failed for ${model} (exit code ${rc})"
        echo "  Last 30 lines of container logs:"
        docker logs "$container_name" --tail 30 2>&1
    fi

    docker rm -f "$container_name" 2>/dev/null
    sleep 15
    return $rc
}

# ── Helper: run both engines (baseline full run) ──────────────────────────────
# Skips automatically if model already has ≥10 result files.
# Args: model  sleep_time
run_model() {
    local model="$1"
    local sleep_time="${2:-120}"
    local model_slug
    model_slug=$(echo "$model" | awk -F/ '{print tolower($NF)}' | tr '.' '-')

    local existing=0
    if [ -d "${RESULTS_DIR}/${model_slug}" ]; then
        existing=$(find "${RESULTS_DIR}/${model_slug}" -name "*.json" \
            -not -name "*manifest*" 2>/dev/null | wc -l | tr -d ' ')
    fi
    if [ "$existing" -ge 10 ]; then
        log "SKIP: ${model} — already has ${existing} result files"
        return 0
    fi

    log "START BASELINE: ${model} (${model_slug}, ${existing}/10 results)"
    run_engine "$model" "vllm"   "$SCENARIOS" "$sleep_time"
    run_engine "$model" "sglang" "$SCENARIOS" "$sleep_time"

    local after=0
    if [ -d "${RESULTS_DIR}/${model_slug}" ]; then
        after=$(find "${RESULTS_DIR}/${model_slug}" -name "*.json" \
            -not -name "*manifest*" 2>/dev/null | wc -l | tr -d ' ')
    fi
    log "DONE BASELINE: ${model} — ${after} result files"
}

# ── Helper: run vLLM only (for models where SGLang is already complete) ───────
# Args: model  sleep_time
run_vllm_only() {
    local model="$1"
    local sleep_time="${2:-120}"
    local model_slug
    model_slug=$(echo "$model" | awk -F/ '{print tolower($NF)}' | tr '.' '-')

    log "START vLLM-ONLY: ${model}"
    run_engine "$model" "vllm" "$SCENARIOS" "$sleep_time"

    local after=0
    if [ -d "${RESULTS_DIR}/${model_slug}" ]; then
        after=$(find "${RESULTS_DIR}/${model_slug}" -name "*.json" \
            -not -name "*manifest*" 2>/dev/null | wc -l | tr -d ' ')
    fi
    log "DONE vLLM-ONLY: ${model} — ${after} result files total"
}

# ── Helper: spec-dec run (2 scenarios only) ───────────────────────────────────
# Args: model  profile  sleep_time
run_specdec() {
    local model="$1"
    local profile="$2"
    local sleep_time="${3:-120}"
    run_engine "$model" "$profile" "$SPECDEC_SCENARIOS" "$sleep_time"
}

# =============================================================================
#  PART 1 — BASELINE RUNS
# =============================================================================
log "PART 1: BASELINE RUNS"

# ---------------------------------------------------------------------------
# Qwen3-30B-A3B — 0/10 — both engines needed
# MoE model; 180s wait for large weight load
# ---------------------------------------------------------------------------
run_model "Qwen/Qwen3-30B-A3B" 180

# ---------------------------------------------------------------------------
# Granite 3.3 8B — SGLang done (5/5), vLLM missing
# ---------------------------------------------------------------------------
run_vllm_only "ibm-granite/granite-3.3-8b-instruct" 120

# ---------------------------------------------------------------------------
# DeepSeek-R1-Distill-Llama-8B — SGLang done (5/5), vLLM missing
# ---------------------------------------------------------------------------
run_vllm_only "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" 120

# ---------------------------------------------------------------------------
# Gemma 3 4B — 0/10 — both engines needed
# vLLM: --enforce-eager required (hybrid sliding-window + full attention)
#       --disable-frontend-multiprocessing avoids tokenizer subprocess issues
# SGLang: no special flags needed; MAX_MODEL_LEN=4096 keeps it in VRAM
# ---------------------------------------------------------------------------
log "START BASELINE: google/gemma-3-4b-it"

export MAX_MODEL_LEN=4096
run_vllm_direct "google/gemma-3-4b-it" "vllm-server" 150 \
    --enforce-eager --disable-frontend-multiprocessing

run_engine "google/gemma-3-4b-it" "sglang" "$SCENARIOS" 150
unset MAX_MODEL_LEN

log "DONE BASELINE: google/gemma-3-4b-it"

# ---------------------------------------------------------------------------
# Gemma 3 12B — 0/10 — both engines needed
# Tight fit on A10G: max_model_len=2048, gpu_memory_utilization=0.95
# vLLM: --enforce-eager required; higher utilization to fit in 24GB
# ---------------------------------------------------------------------------
log "START BASELINE: google/gemma-3-12b-it"

export MAX_MODEL_LEN=2048
export GPU_MEM_UTIL=0.95
run_vllm_direct "google/gemma-3-12b-it" "vllm-server" 210 \
    --enforce-eager --disable-frontend-multiprocessing

unset GPU_MEM_UTIL
run_engine "google/gemma-3-12b-it" "sglang" "$SCENARIOS" 210
unset MAX_MODEL_LEN

log "DONE BASELINE: google/gemma-3-12b-it"

# =============================================================================
#  PART 2 — SPECULATIVE DECODING
# =============================================================================
log "PART 2: SPECULATIVE DECODING RUNS"

# ---------------------------------------------------------------------------
# Qwen3-8B spec-dec — all ngram/baseline variants already done
# Eagle3 BLOCKED: RedHatAI/Qwen3-8B-speculator.eagle3 not yet published
# ---------------------------------------------------------------------------
log "SKIP SPEC-DEC: Qwen3-8B Eagle3 — no draft model available yet"

# ---------------------------------------------------------------------------
# Llama 3.1 8B Eagle3 — vllm-eagle3 and sglang-eagle3 still pending
# Draft models:
#   vLLM  : RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3
#   SGLang: jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B
# MAX_MODEL_LEN=4096 keeps both main + draft models in 24GB VRAM
# 240s wait — two models load on startup
# ---------------------------------------------------------------------------
log "START SPEC-DEC: Llama 3.1 8B Eagle3"
LLAMA="meta-llama/Llama-3.1-8B-Instruct"

export MAX_MODEL_LEN=4096

run_specdec "$LLAMA" "vllm-eagle3"   240
run_specdec "$LLAMA" "sglang-eagle3" 240

unset MAX_MODEL_LEN
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
