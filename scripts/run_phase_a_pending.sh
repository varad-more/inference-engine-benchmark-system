#!/usr/bin/env bash
# =============================================================================
# Phase A — All Pending Benchmark Runs (Baseline + Speculative Decoding)
#
# Part 1: Baseline runs for 7 pending models (5 scenarios x 2 engines)
# Part 2: Speculative decoding runs for Llama 3.1 8B + Qwen3-8B
#         (Eagle3 + Ngram variants on vLLM and SGLang)
#
# Usage:
#   chmod +x scripts/run_phase_a_pending.sh
#   tmux new -s benchmark
#   ./scripts/run_phase_a_pending.sh 2>&1 | tee logs/phase_a_$(date +%Y%m%dT%H%M%S).log
#
# Prerequisites:
#   - HUGGING_FACE_HUB_TOKEN set in .env (needed for Gemma 3 4B/12B, Llama 3.1)
#   - Docker running with GPU access
#   - model-cache/ directory exists
# =============================================================================

# Do NOT use set -e — we handle errors ourselves so the script keeps going
set +e

# Use conda python — adjust if your env name differs
PYTHON="conda run --no-capture-output -n base python"

SCENARIOS="single_request_latency,throughput_ramp,long_context_stress,prefix_sharing_benefit,structured_generation_speed"
SPECDEC_SCENARIOS="single_request_latency,throughput_ramp"
COOLDOWN=10
RESULTS_DIR="results"

# Track errors
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
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || echo "  (GPU check failed — continuing anyway)"

echo -n "  .env HF token: "
if grep -q "HUGGING_FACE_HUB_TOKEN=hf_" .env 2>/dev/null; then
    echo "found"
else
    echo "NOT FOUND — gated models (Gemma, Llama) will fail"
fi

echo -n "  model-cache/: "
if [ -d "model-cache" ]; then
    echo "exists"
else
    mkdir -p model-cache
    echo "created"
fi

echo -n "  results/: "
ls -d ${RESULTS_DIR}/*/ 2>/dev/null | wc -l | tr -d ' '
echo " model folders"

log "PREFLIGHT COMPLETE"

# ── Helper functions ─────────────────────────────────────────────────────────

run_engine() {
    local model="$1"
    local profile="$2"
    local scenarios="$3"
    local sleep_time="$4"

    log "  ${profile}: ${model}"
    export MODEL="$model"

    # Start container
    docker compose --profile "$profile" up -d "$profile"
    if [ $? -ne 0 ]; then
        error "${profile} docker up failed for ${model}"
        return 1
    fi

    log "  Waiting ${sleep_time}s for model to load..."
    sleep "$sleep_time"

    # Check container is still running
    if ! docker compose --profile "$profile" ps --status running | grep -q "$profile"; then
        error "${profile} container died during startup for ${model}"
        docker compose --profile "$profile" logs --tail 50 2>&1
        docker compose --profile "$profile" down 2>/dev/null
        return 1
    fi

    # Run benchmark
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

    # Tear down
    docker compose --profile "$profile" down 2>/dev/null
    sleep 15
    return $rc
}

run_model() {
    local model="$1"
    local sleep_time="${2:-120}"

    local model_slug
    model_slug=$(echo "$model" | awk -F/ '{print tolower($NF)}')

    # Skip if already has results (10 = 5 scenarios x 2 engines)
    local existing=0
    if [ -d "${RESULTS_DIR}/${model_slug}" ]; then
        existing=$(find "${RESULTS_DIR}/${model_slug}" -name "*.json" -not -name "*manifest*" 2>/dev/null | wc -l | tr -d ' ')
    fi
    if [ "$existing" -ge 10 ]; then
        log "SKIP: ${model} — already has ${existing} result files"
        return 0
    fi

    log "START BASELINE: ${model} (${model_slug}, ${existing}/10 results exist)"

    run_engine "$model" "vllm" "$SCENARIOS" "$sleep_time"
    run_engine "$model" "sglang" "$SCENARIOS" "$sleep_time"

    # Count results after run
    local after=0
    if [ -d "${RESULTS_DIR}/${model_slug}" ]; then
        after=$(find "${RESULTS_DIR}/${model_slug}" -name "*.json" -not -name "*manifest*" 2>/dev/null | wc -l | tr -d ' ')
    fi
    log "DONE BASELINE: ${model} — ${after} result files in ${RESULTS_DIR}/${model_slug}/"
}

run_specdec() {
    local model="$1"
    local profile="$2"
    local sleep_time="${3:-120}"

    run_engine "$model" "$profile" "$SPECDEC_SCENARIOS" "$sleep_time"
}

# =============================================================================
mkdir -p logs

# =============================================================================
#  PART 1: BASELINE RUNS (7 pending models)
# =============================================================================
log "PART 1: BASELINE RUNS (7 models)"

run_model "HuggingFaceTB/SmolLM3-3B" 120

run_model "microsoft/Phi-4-mini-instruct" 120

run_model "Qwen/Qwen3-30B-A3B" 180

run_model "ibm-granite/granite-3.3-8b-instruct" 120

run_model "google/gemma-3-4b-it" 120

export MAX_MODEL_LEN=4096
run_model "google/gemma-3-12b-it" 180
unset MAX_MODEL_LEN

run_model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" 120

# =============================================================================
#  PART 2: SPECULATIVE DECODING (Llama 3.1 8B + Qwen3-8B)
# =============================================================================
log "PART 2: SPECULATIVE DECODING RUNS"

# --- Llama 3.1 8B ---
log "START SPEC-DEC: Llama 3.1 8B (baseline + Eagle3 + Ngram, both engines)"
LLAMA="meta-llama/Llama-3.1-8B-Instruct"

run_specdec "$LLAMA" "vllm" 120
run_specdec "$LLAMA" "sglang" 120
run_specdec "$LLAMA" "vllm-eagle3" 180
run_specdec "$LLAMA" "sglang-eagle3" 180
run_specdec "$LLAMA" "vllm-ngram" 120
run_specdec "$LLAMA" "sglang-ngram" 120

log "DONE SPEC-DEC: Llama 3.1 8B"

# --- Qwen3-8B ---
log "START SPEC-DEC: Qwen3-8B (baseline + Eagle3 vLLM + Ngram both engines)"
QWEN="Qwen/Qwen3-8B"

run_specdec "$QWEN" "vllm" 120
run_specdec "$QWEN" "sglang" 120
run_specdec "$QWEN" "vllm-eagle3" 180
run_specdec "$QWEN" "vllm-ngram" 120
run_specdec "$QWEN" "sglang-ngram" 120

log "DONE SPEC-DEC: Qwen3-8B"

# =============================================================================
#  FINAL SUMMARY
# =============================================================================
log "ALL RUNS COMPLETE"

echo ""
echo "Results per model:"
for d in ${RESULTS_DIR}/*/; do
    name=$(basename "$d")
    count=$(find "$d" -name "*.json" -not -name "*manifest*" 2>/dev/null | wc -l | tr -d ' ')
    printf "  %-35s %3s files\n" "$name" "$count"
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
echo "  1. $PYTHON run_experiment.py final-report --output reports/phase_a_report.md"
echo "  2. $PYTHON run_experiment.py report --output reports/phase_a_report.html"
