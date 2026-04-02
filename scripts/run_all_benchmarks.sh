#!/usr/bin/env bash
# =============================================================================
# Full Benchmark Suite — vLLM vs SGLang on A10G (24GB)
#
# Runs all models that fit on a single NVIDIA A10G GPU.
# Uses 'docker run' directly (no docker-compose) to avoid network issues.
#
# Models (14 total, ordered small → large):
#   1.  gemma-2-2b-it             (2B)
#   2.  HuggingFaceTB/SmolLM3-3B  (3B)
#   3.  meta-llama/Llama-3.2-3B-Instruct  (3B)
#   4.  microsoft/Phi-3-mini-4k-instruct   (3.8B)
#   5.  google/gemma-3-4b-it      (4B)  — vLLM needs --enforce-eager
#   6.  microsoft/Phi-4-mini-instruct      (4B)
#   7.  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B  (7B)
#   8.  Qwen/Qwen2.5-7B-Instruct  (7B)
#   9.  mistralai/Mistral-7B-Instruct-v0.3  (7B)
#  10.  meta-llama/Llama-3.1-8B-Instruct   (8B)
#  11.  Qwen/Qwen3-8B             (8B)
#  12.  ibm-granite/granite-3.3-8b-instruct (8B)
#  13.  deepseek-ai/DeepSeek-R1-Distill-Llama-8B  (8B)
#  14.  google/gemma-2-9b-it      (9B)
#
# Speculative decoding (Eagle3 + Ngram):
#  - Llama 3.1 8B: vllm-ngram, sglang-ngram, vllm-eagle3, sglang-eagle3
#  - Qwen3 8B:     vllm-ngram, sglang-ngram (Eagle3 blocked — no draft model)
#
# Skipped (too large for A10G 24GB):
#  - Qwen/Qwen3-30B-A3B    (30B total params = ~60GB at bf16)
#  - google/gemma-3-12b-it  (12B = ~24GB weights, no room for KV cache)
#
# Usage:
#   chmod +x scripts/run_all_benchmarks.sh
#   tmux new -s benchmark
#   ./scripts/run_all_benchmarks.sh 2>&1 | tee logs/full_run_$(date +%Y%m%dT%H%M%S).log
#
# Options:
#   --skip-existing   Skip models that already have ≥10 result files (default)
#   --force           Re-run all models even if results exist
#
# Prerequisites:
#   - HUGGING_FACE_HUB_TOKEN set in .env (Gemma, Llama are gated)
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
SKIP_EXISTING=true
COMPLETED=0
SKIPPED=0

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.18.0-cu130}"
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:nightly-dev-cu13-20260321-94194537}"

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

# ── Preflight checks ────────────────────────────────────────────────────────
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
if [ -n "$HF_TOKEN" ]; then echo "found"; else echo "NOT FOUND — gated models will fail"; fi

echo -n "  model-cache/: "
if [ -d "model-cache" ]; then echo "exists"; else mkdir -p model-cache && echo "created"; fi

echo -n "  Skip existing: "
echo "$SKIP_EXISTING"

# Kill any leftover containers
echo -n "  Cleaning up containers: "
docker rm -f vllm-server sglang-server vllm-eagle3-server sglang-eagle3-server \
    vllm-ngram-server sglang-ngram-server 2>/dev/null || true
echo "done"

log "PREFLIGHT COMPLETE"

mkdir -p logs

# ── Utility: model slug ─────────────────────────────────────────────────────
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

# ── Utility: should skip? ───────────────────────────────────────────────────
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
    return 1  # no, don't skip
}

# ── Cleanup helper ───────────────────────────────────────────────────────────
cleanup_container() {
    local name="$1"
    docker rm -f "$name" 2>/dev/null
    sleep 2
}

# ── vLLM via docker run ─────────────────────────────────────────────────────
# Args: model  container_name  port  sleep_time  scenarios  engine_label  [extra args...]
run_vllm() {
    local model="$1"
    local container="$2"
    local port="$3"
    local sleep_time="$4"
    local scenarios="$5"
    local engine_label="$6"
    shift 6
    local extra_args=("$@")

    local max_model_len="${MAX_MODEL_LEN:-8192}"
    local gpu_mem_util="${GPU_MEM_UTIL:-0.85}"

    log "  Starting ${engine_label} for ${model}"
    cleanup_container "$container"

    docker run -d \
        --name "$container" \
        --gpus '"device=0"' \
        -p "${port}:${port}" \
        --shm-size 10g \
        -v "$(pwd)/model-cache:/root/.cache/huggingface" \
        -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
        -e "CUDA_VISIBLE_DEVICES=0" \
        "$VLLM_IMAGE" \
        --model "$model" \
        --host 0.0.0.0 \
        --port "$port" \
        --enable-prefix-caching \
        --max-model-len "$max_model_len" \
        --gpu-memory-utilization "$gpu_mem_util" \
        --served-model-name "$model" \
        "${extra_args[@]}"

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

    local max_model_len="${MAX_MODEL_LEN:-8192}"

    log "  Starting ${engine_label} for ${model}"
    cleanup_container "$container"

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
        python -m sglang.launch_server \
        --model-path "$model" \
        --host 0.0.0.0 \
        --port "$port" \
        --mem-fraction-static 0.85 \
        --context-length "$max_model_len" \
        "${extra_args[@]}"

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

# ── Baseline run: vLLM + SGLang (5 scenarios each) ──────────────────────────
# Args: model  sleep_time  [extra_vllm_args...]
run_baseline() {
    local model="$1"
    local sleep_time="${2:-120}"
    shift 2
    local extra_vllm_args=("$@")
    local slug
    slug=$(model_slug "$model")

    if should_skip "$slug" 10; then
        local existing
        existing=$(count_results "$slug")
        log "SKIP BASELINE: ${model} — already has ${existing} result files"
        ((SKIPPED++))
        return 0
    fi

    log "START BASELINE: ${model}"

    run_vllm "$model" "vllm-server" 8000 "$sleep_time" "$SCENARIOS" "vllm" "${extra_vllm_args[@]}"
    run_sglang "$model" "sglang-server" 8001 "$sleep_time" "$SCENARIOS" "sglang"

    local after
    after=$(count_results "$slug")
    log "DONE BASELINE: ${model} — ${after} result files"
    ((COMPLETED++))
}

# =============================================================================
#  PART 1 — BASELINE RUNS (small → large)
# =============================================================================
log "PART 1: BASELINE RUNS"

# ── 2B models ────────────────────────────────────────────────────────────────
run_baseline "google/gemma-2-2b-it" 90

# ── 3B models ────────────────────────────────────────────────────────────────
run_baseline "HuggingFaceTB/SmolLM3-3B" 90
run_baseline "meta-llama/Llama-3.2-3B-Instruct" 90

# ── 4B models ────────────────────────────────────────────────────────────────
run_baseline "microsoft/Phi-3-mini-4k-instruct" 100

# Gemma 3 4B — vLLM needs --enforce-eager (hybrid sliding-window attention)
log "START BASELINE: google/gemma-3-4b-it"
SLUG=$(model_slug "google/gemma-3-4b-it")
if should_skip "$SLUG" 10; then
    log "SKIP BASELINE: google/gemma-3-4b-it — already has $(count_results $SLUG) result files"
    ((SKIPPED++))
else
    export MAX_MODEL_LEN=4096
    run_vllm "google/gemma-3-4b-it" "vllm-server" 8000 150 "$SCENARIOS" "vllm" \
        --enforce-eager --disable-frontend-multiprocessing
    run_sglang "google/gemma-3-4b-it" "sglang-server" 8001 150 "$SCENARIOS" "sglang"
    unset MAX_MODEL_LEN
    log "DONE BASELINE: google/gemma-3-4b-it — $(count_results $SLUG) result files"
    ((COMPLETED++))
fi

run_baseline "microsoft/Phi-4-mini-instruct" 100

# ── 7B models ────────────────────────────────────────────────────────────────
run_baseline "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 120
run_baseline "Qwen/Qwen2.5-7B-Instruct" 120
run_baseline "mistralai/Mistral-7B-Instruct-v0.3" 120

# ── 8B models ────────────────────────────────────────────────────────────────
run_baseline "meta-llama/Llama-3.1-8B-Instruct" 120
run_baseline "Qwen/Qwen3-8B" 120
run_baseline "ibm-granite/granite-3.3-8b-instruct" 120
run_baseline "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" 120

# ── 9B models ────────────────────────────────────────────────────────────────
run_baseline "google/gemma-2-9b-it" 150

# =============================================================================
#  PART 2 — SPECULATIVE DECODING: NGRAM (no draft model needed)
# =============================================================================
log "PART 2: SPECULATIVE DECODING — NGRAM"

# ── Llama 3.1 8B Ngram ──────────────────────────────────────────────────────
LLAMA="meta-llama/Llama-3.1-8B-Instruct"
LLAMA_SLUG=$(model_slug "$LLAMA")

if should_skip "$LLAMA_SLUG" 14; then
    log "SKIP NGRAM: ${LLAMA} — already has $(count_results $LLAMA_SLUG) result files"
else
    log "START NGRAM: ${LLAMA}"
    run_vllm "$LLAMA" "vllm-ngram-server" 8000 120 "$SPECDEC_SCENARIOS" "vllm-ngram" \
        --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}'
    run_sglang "$LLAMA" "sglang-ngram-server" 8001 120 "$SPECDEC_SCENARIOS" "sglang-ngram" \
        --speculative-algorithm NGRAM --speculative-num-draft-tokens 16
    log "DONE NGRAM: ${LLAMA}"
fi

# ── Qwen3 8B Ngram ──────────────────────────────────────────────────────────
QWEN="Qwen/Qwen3-8B"
QWEN_SLUG=$(model_slug "$QWEN")

if should_skip "$QWEN_SLUG" 14; then
    log "SKIP NGRAM: ${QWEN} — already has $(count_results $QWEN_SLUG) result files"
else
    log "START NGRAM: ${QWEN}"
    run_vllm "$QWEN" "vllm-ngram-server" 8000 120 "$SPECDEC_SCENARIOS" "vllm-ngram" \
        --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}'
    run_sglang "$QWEN" "sglang-ngram-server" 8001 120 "$SPECDEC_SCENARIOS" "sglang-ngram" \
        --speculative-algorithm NGRAM --speculative-num-draft-tokens 16
    log "DONE NGRAM: ${QWEN}"
fi

# =============================================================================
#  PART 3 — SPECULATIVE DECODING: EAGLE3 (needs draft model)
# =============================================================================
log "PART 3: SPECULATIVE DECODING — EAGLE3"

# ── Llama 3.1 8B Eagle3 ─────────────────────────────────────────────────────
# Tight fit on A10G: model+draft = ~16.8 GiB
LLAMA_VLLM_DRAFT="RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3"
LLAMA_SGLANG_DRAFT="jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"

log "START EAGLE3: ${LLAMA}"
export MAX_MODEL_LEN=2048
export GPU_MEM_UTIL=0.95

run_vllm "$LLAMA" "vllm-eagle3-server" 8000 240 "$SPECDEC_SCENARIOS" "vllm-eagle3" \
    --enforce-eager \
    --speculative-config "{\"method\":\"eagle3\",\"model\":\"${LLAMA_VLLM_DRAFT}\",\"num_speculative_tokens\":3}"

unset GPU_MEM_UTIL
run_sglang "$LLAMA" "sglang-eagle3-server" 8001 240 "$SPECDEC_SCENARIOS" "sglang-eagle3" \
    --mem-fraction-static 0.65 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$LLAMA_SGLANG_DRAFT" \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16

unset MAX_MODEL_LEN
log "DONE EAGLE3: ${LLAMA}"

# ── Qwen3-8B Eagle3 — BLOCKED ───────────────────────────────────────────────
log "SKIP EAGLE3: Qwen3-8B — RedHatAI/Qwen3-8B-speculator.eagle3 not yet published"

# =============================================================================
#  FINAL SUMMARY
# =============================================================================
log "ALL RUNS COMPLETE"

echo ""
echo "Results per model:"
for d in $(ls -d ${RESULTS_DIR}/*/ 2>/dev/null | sort); do
    name=$(basename "$d")
    count=$(find "$d" -name "*.json" -not -name "*manifest*" 2>/dev/null | wc -l | tr -d ' ')
    printf "  %-45s %3s files\n" "$name" "$count"
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
echo "  1. $PYTHON analysis/generate_final_benchmark_report.py"
echo "  2. Review reports/final_benchmark_report_*.md"
