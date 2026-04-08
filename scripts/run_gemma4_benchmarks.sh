#!/usr/bin/env bash
# =============================================================================
# Gemma 4 Benchmark Suite — vLLM vs SGLang on A10G (24GB)
#
# Runs Gemma 4 model(s) through the full 5-scenario baseline matrix on both
# engines, plus Ngram speculative decoding for single_request_latency and
# throughput_ramp.
#
# NOTE: Eagle3 is skipped — no Gemma 4 Eagle3 draft model is published yet.
#       Update EAGLE3_VLLM_DRAFT / EAGLE3_SGLANG_DRAFT and un-comment Part 3
#       once a draft model becomes available.
#
# Models (A10G 24GB):
#   google/gemma-4-E2B-it      — ~4GB VRAM
#   google/gemma-4-E4B-it      — ~8GB VRAM (default)
#   google/gemma-4-26B-A4B-it  — MoE, 26B total weights, likely too large for A10G
#   google/gemma-4-31B-it      — 31B dense, requires A100 80GB
#
# vLLM flags for Gemma 4:
#   --enforce-eager                  — required for hybrid sliding-window attention
#   --disable-frontend-multiprocessing — avoids a fork/spawn issue with Gemma arch
#   max-model-len 4096               — conservative; increase if VRAM allows
#
# Gemma models are gated — set HUGGING_FACE_HUB_TOKEN in .env.
#
# Usage:
#   chmod +x scripts/run_gemma4_benchmarks.sh
#   tmux new -s gemma4
#   ./scripts/run_gemma4_benchmarks.sh 2>&1 | tee logs/gemma4_$(date +%Y%m%dT%H%M%S).log
#
# Options:
#   --force          Re-run even if results already exist
#   --skip-existing  Skip models that already have ≥10 result files (default)
# =============================================================================

set +e

# ── Model IDs — VERIFY THESE on HuggingFace before running ─────────────────
# Gemma 4 naming follows the Gemma 3 pattern (google/gemma-4-<size>-it).
# Update the list below to match the sizes you want to benchmark.
GEMMA4_MODELS=(
    # Gemma 4 instruction-tuned variants that fit on A10G 24GB:
    #   google/gemma-4-E2B-it  — ~4GB VRAM, very fast
    #   google/gemma-4-E4B-it  — ~8GB VRAM, best quality/size tradeoff on A10G
    #
    # Larger variants (likely too big for A10G 24GB without quantization):
    #   google/gemma-4-26B-A4B-it  — MoE: 26B total weights (~52GB bf16), 4B active
    #   google/gemma-4-31B-it      — 31B dense (~62GB bf16)
    "google/gemma-4-E2B-it"
    "google/gemma-4-E4B-it"
)

# ── Config ──────────────────────────────────────────────────────────────────
PYTHON="conda run --no-capture-output -n base python"
SCENARIOS="single_request_latency,throughput_ramp,long_context_stress,prefix_sharing_benefit,structured_generation_speed"
SPECDEC_SCENARIOS="single_request_latency,throughput_ramp"
COOLDOWN=10
RESULTS_DIR="results"
SKIP_EXISTING=true
ERRORS=()
COMPLETED=0
SKIPPED=0

# Gemma 4 requires a smaller max-model-len due to KV cache pressure
GEMMA4_MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# Gemma 4 requires Transformers >= v5.5.0. The old pinned images used for
# other models (vllm v0.18.0, sglang nightly-20260321) are too old.
# Override at runtime: VLLM_IMAGE=... SGLANG_IMAGE=... ./run_gemma4_benchmarks.sh
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:latest}"

# ── Parse args ───────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --force) SKIP_EXISTING=false ;;
        --skip-existing) SKIP_EXISTING=true ;;
    esac
done

# ── Logging helpers ──────────────────────────────────────────────────────────
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

# ── Load HF token from .env ─────────────────────────────────────────────────
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
if [ -n "$HF_TOKEN" ]; then
    echo "found (required for Gemma 4 — gated model)"
else
    echo "NOT FOUND — Gemma 4 is a gated model and will fail without a token"
    echo "  Set HUGGING_FACE_HUB_TOKEN in .env and re-run"
    exit 1
fi

echo -n "  model-cache/: "
if [ -d "model-cache" ]; then echo "exists"; else mkdir -p model-cache && echo "created"; fi

echo -n "  Skip existing: "
echo "$SKIP_EXISTING"

echo "  Models to benchmark:"
for m in "${GEMMA4_MODELS[@]}"; do echo "    - $m"; done

# Kill any leftover containers
echo -n "  Cleaning up stale containers: "
docker rm -f vllm-server sglang-server vllm-ngram-server sglang-ngram-server 2>/dev/null || true
echo "done"

log "PREFLIGHT COMPLETE"

mkdir -p logs

# ── Utility: model slug ──────────────────────────────────────────────────────
model_slug() {
    echo "$1" | awk -F/ '{print tolower($NF)}' | tr '.' '-'
}

# ── Utility: count result files ──────────────────────────────────────────────
count_results() {
    local slug="$1"
    if [ -d "${RESULTS_DIR}/${slug}" ]; then
        find "${RESULTS_DIR}/${slug}" -name "*.json" -not -name "*manifest*" 2>/dev/null | wc -l | tr -d ' '
    else
        echo "0"
    fi
}

# ── Utility: should skip? ────────────────────────────────────────────────────
should_skip() {
    local slug="$1"
    local min_files="${2:-10}"
    if [ "$SKIP_EXISTING" = "true" ]; then
        local existing
        existing=$(count_results "$slug")
        if [ "$existing" -ge "$min_files" ]; then
            return 0  # yes, skip
        fi
    fi
    return 1  # no, run
}

# ── Cleanup helper ───────────────────────────────────────────────────────────
cleanup_container() {
    local name="$1"
    docker rm -f "$name" 2>/dev/null
    sleep 2
}

# ── vLLM via docker run ──────────────────────────────────────────────────────
# Args: model  container_name  port  sleep_time  scenarios  engine_label  [extra args...]
#
# Gemma 4 requires Transformers >= v5.5.0 which is not yet bundled in any
# released vLLM image. We override the entrypoint to bash and install
# transformers from the HuggingFace git main branch before starting the server.
# printf '%q ' safely quotes every arg including JSON speculative-config strings.
run_vllm() {
    local model="$1"
    local container="$2"
    local port="$3"
    local sleep_time="$4"
    local scenarios="$5"
    local engine_label="$6"
    shift 6
    local extra_args=("$@")

    local gpu_mem_util="${GPU_MEM_UTIL:-0.85}"

    log "  Starting ${engine_label} for ${model}"
    cleanup_container "$container"

    local server_args=(
        --model "$model"
        --host 0.0.0.0
        --port "$port"
        --enable-prefix-caching
        --max-model-len "$GEMMA4_MAX_MODEL_LEN"
        --gpu-memory-utilization "$gpu_mem_util"
        --served-model-name "$model"
        "${extra_args[@]}"
    )
    local quoted_args
    quoted_args=$(printf '%q ' "${server_args[@]}")
    local cmd="pip install -q --upgrade git+https://github.com/huggingface/transformers.git \
        && exec python -m vllm.entrypoints.openai.api_server ${quoted_args}"

    docker run -d \
        --name "$container" \
        --gpus '"device=0"' \
        -p "${port}:${port}" \
        --shm-size 10g \
        -v "$(pwd)/model-cache:/root/.cache/huggingface" \
        -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
        -e "CUDA_VISIBLE_DEVICES=0" \
        --entrypoint bash \
        "$VLLM_IMAGE" \
        -c "$cmd"

    if [ $? -ne 0 ]; then
        error "${engine_label} docker run failed for ${model}"
        return 1
    fi

    log "  Waiting ${sleep_time}s for model to load..."
    sleep "$sleep_time"

    if ! docker ps --filter "name=${container}" --format '{{.Status}}' | grep -q "Up"; then
        error "${engine_label} container died during startup for ${model}"
        echo "  Container logs (last 80 lines):"
        docker logs "$container" --tail 80 2>&1
        docker rm -f "$container" 2>/dev/null
        return 1
    fi

    $PYTHON run_experiment.py matrix \
        --scenarios "$scenarios" \
        --engines "$engine_label" \
        -m "$model" \
        --cooldown-seconds "$COOLDOWN" \
        --output-dir "$RESULTS_DIR"
    local rc=$?

    if [ $rc -ne 0 ]; then
        error "${engine_label} benchmark failed for ${model} (exit code ${rc})"
        docker logs "$container" --tail 30 2>&1
    fi

    docker rm -f "$container" 2>/dev/null
    sleep 10
    return $rc
}

# ── SGLang via docker run ────────────────────────────────────────────────────
# Args: model  container_name  port  sleep_time  scenarios  engine_label  [extra args...]
run_sglang() {
    local model="$1"
    local container="$2"
    local port="$3"
    local sleep_time="$4"
    local scenarios="$5"
    local engine_label="$6"
    shift 6
    local extra_args=("$@")

    log "  Starting ${engine_label} for ${model}"
    cleanup_container "$container"

    local server_args=(
        --model-path "$model"
        --host 0.0.0.0
        --port "$port"
        --mem-fraction-static 0.85
        --context-length "$GEMMA4_MAX_MODEL_LEN"
        --disable-cuda-graph   # Gemma 4: CUDA graph capture fails (batch size mismatch in kvcache kernel)
        "${extra_args[@]}"
    )
    local quoted_args
    quoted_args=$(printf '%q ' "${server_args[@]}")
    local cmd="pip install -q --upgrade git+https://github.com/huggingface/transformers.git \
        && exec python -m sglang.launch_server ${quoted_args}"

    docker run -d \
        --name "$container" \
        --gpus '"device=0"' \
        -p "${port}:${port}" \
        --shm-size 10g \
        -v "$(pwd)/model-cache:/root/.cache/huggingface" \
        -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
        -e "CUDA_VISIBLE_DEVICES=0" \
        -e "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1" \
        "$SGLANG_IMAGE" \
        bash -c "$cmd"

    if [ $? -ne 0 ]; then
        error "${engine_label} docker run failed for ${model}"
        return 1
    fi

    log "  Waiting ${sleep_time}s for model to load..."
    sleep "$sleep_time"

    if ! docker ps --filter "name=${container}" --format '{{.Status}}' | grep -q "Up"; then
        error "${engine_label} container died during startup for ${model}"
        echo "  Container logs (last 80 lines):"
        docker logs "$container" --tail 80 2>&1
        docker rm -f "$container" 2>/dev/null
        return 1
    fi

    $PYTHON run_experiment.py matrix \
        --scenarios "$scenarios" \
        --engines "$engine_label" \
        -m "$model" \
        --cooldown-seconds "$COOLDOWN" \
        --output-dir "$RESULTS_DIR"
    local rc=$?

    if [ $rc -ne 0 ]; then
        error "${engine_label} benchmark failed for ${model} (exit code ${rc})"
        docker logs "$container" --tail 30 2>&1
    fi

    docker rm -f "$container" 2>/dev/null
    sleep 10
    return $rc
}

# =============================================================================
#  PART 1 — BASELINE (vLLM + SGLang, all 5 scenarios)
# =============================================================================
log "PART 1: BASELINE RUNS"

for MODEL in "${GEMMA4_MODELS[@]}"; do
    SLUG=$(model_slug "$MODEL")

    if should_skip "$SLUG" 10; then
        log "SKIP BASELINE: ${MODEL} — already has $(count_results $SLUG) result files"
        ((SKIPPED++))
        continue
    fi

    log "START BASELINE: ${MODEL}"

    # vLLM: Gemma architecture needs --enforce-eager and --disable-frontend-multiprocessing
    # sleep 240 = ~2min pip install + ~2.5min model load
    run_vllm "$MODEL" "vllm-server" 8000 240 "$SCENARIOS" "vllm" \
        --enforce-eager \
        --disable-frontend-multiprocessing

    # SGLang: no extra flags needed for Gemma
    run_sglang "$MODEL" "sglang-server" 8001 240 "$SCENARIOS" "sglang"

    log "DONE BASELINE: ${MODEL} — $(count_results $SLUG) result files"
    ((COMPLETED++))
done

# =============================================================================
#  PART 2 — NGRAM SPECULATIVE DECODING
# =============================================================================
log "PART 2: SPECULATIVE DECODING — NGRAM"

for MODEL in "${GEMMA4_MODELS[@]}"; do
    SLUG=$(model_slug "$MODEL")

    if should_skip "$SLUG" 14; then
        log "SKIP NGRAM: ${MODEL} — already has $(count_results $SLUG) result files"
        continue
    fi

    log "START NGRAM: ${MODEL}"

    run_vllm "$MODEL" "vllm-ngram-server" 8000 240 "$SPECDEC_SCENARIOS" "vllm-ngram" \
        --enforce-eager \
        --disable-frontend-multiprocessing \
        --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}'

    run_sglang "$MODEL" "sglang-ngram-server" 8001 240 "$SPECDEC_SCENARIOS" "sglang-ngram" \
        --speculative-algorithm NGRAM \
        --speculative-num-draft-tokens 16

    log "DONE NGRAM: ${MODEL}"
done

# =============================================================================
#  PART 3 — EAGLE3 (skipped — no Gemma 4 draft model published yet)
# =============================================================================
log "SKIP EAGLE3: Gemma 4 — no Eagle3 draft model published yet"
log "  Update EAGLE3_VLLM_DRAFT / EAGLE3_SGLANG_DRAFT and un-comment Part 3 when available"

# =============================================================================
#  FINAL SUMMARY
# =============================================================================
log "ALL RUNS COMPLETE"

echo ""
echo "Results per model:"
for MODEL in "${GEMMA4_MODELS[@]}"; do
    SLUG=$(model_slug "$MODEL")
    COUNT=$(count_results "$SLUG")
    printf "  %-45s %3s files\n" "$SLUG" "$COUNT"
done

echo ""
echo "Summary: ${COMPLETED} models ran, ${SKIPPED} skipped, ${#ERRORS[@]} errors"

if [ ${#ERRORS[@]} -gt 0 ]; then
    echo ""
    echo "ERRORS:"
    for i in "${!ERRORS[@]}"; do
        echo "  $((i+1)). ${ERRORS[$i]}"
    done
fi

echo ""
echo "Next steps:"
echo "  1. $PYTHON run_experiment.py final-report --output gemma4_report.md"
echo "  2. Review gemma4_report.md"
