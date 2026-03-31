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

set -euo pipefail

SCENARIOS="single_request_latency,throughput_ramp,long_context_stress,prefix_sharing_benefit,structured_generation_speed"
SPECDEC_SCENARIOS="single_request_latency,throughput_ramp"
COOLDOWN=10
RESULTS_DIR="results"

log() {
    echo ""
    echo "========================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "========================================================================"
}

run_model() {
    local model="$1"
    local sleep_time="${2:-120}"

    local model_slug
    model_slug=$(echo "$model" | awk -F/ '{print tolower($NF)}')

    # Skip if already has results (10 = 5 scenarios x 2 engines)
    local existing
    existing=$(find "${RESULTS_DIR}/${model_slug}" -name "*.json" -not -name "*manifest*" 2>/dev/null | wc -l)
    if [ "$existing" -ge 10 ]; then
        log "SKIP: ${model} — already has ${existing} result files"
        return 0
    fi

    log "START BASELINE: ${model}"

    # vLLM
    log "  vLLM: ${model}"
    export MODEL="$model"
    docker compose --profile vllm up -d vllm
    sleep "$sleep_time"
    python run_experiment.py matrix \
        --scenarios "$SCENARIOS" \
        --engines vllm \
        -m "$model" \
        --cooldown-seconds "$COOLDOWN" \
        --output-dir "$RESULTS_DIR" || echo "  [WARN] vLLM run failed for ${model}"
    docker compose --profile vllm down
    sleep 30

    # SGLang
    log "  SGLang: ${model}"
    docker compose --profile sglang up -d sglang
    sleep "$sleep_time"
    python run_experiment.py matrix \
        --scenarios "$SCENARIOS" \
        --engines sglang \
        -m "$model" \
        --cooldown-seconds "$COOLDOWN" \
        --output-dir "$RESULTS_DIR" || echo "  [WARN] SGLang run failed for ${model}"
    docker compose --profile sglang down
    sleep 30

    log "DONE BASELINE: ${model}"
}

run_specdec() {
    local model="$1"
    local profile="$2"
    local sleep_time="${3:-120}"

    local model_slug
    model_slug=$(echo "$model" | awk -F/ '{print tolower($NF)}')

    log "  SPEC-DEC ${profile}: ${model}"
    export MODEL="$model"
    docker compose --profile "$profile" up -d "$profile"
    sleep "$sleep_time"
    python run_experiment.py matrix \
        --scenarios "$SPECDEC_SCENARIOS" \
        --engines "$profile" \
        -m "$model" \
        --cooldown-seconds "$COOLDOWN" \
        --output-dir "$RESULTS_DIR" || echo "  [WARN] ${profile} run failed for ${model}"
    docker compose --profile "$profile" down
    sleep 30
}

# =============================================================================
mkdir -p logs

# =============================================================================
#  PART 1: BASELINE RUNS (7 pending models)
# =============================================================================
log "PART 1: BASELINE RUNS"

# Model 3: SmolLM3 3B (~6GB VRAM — fastest)
run_model "HuggingFaceTB/SmolLM3-3B" 120

# Model 4: Phi-4-mini (~7GB VRAM)
run_model "microsoft/Phi-4-mini-instruct" 120

# Model 5: Qwen3-30B-A3B MoE (~17GB VRAM — longer load)
run_model "Qwen/Qwen3-30B-A3B" 180

# Model 6: Granite 3.3 8B (~16GB VRAM)
run_model "ibm-granite/granite-3.3-8b-instruct" 120

# Model 7: Gemma 3 4B (~8GB, gated — needs HF token)
run_model "google/gemma-3-4b-it" 120

# Model 8: Gemma 3 12B (~24GB — tight fit)
export MAX_MODEL_LEN=4096
run_model "google/gemma-3-12b-it" 180
unset MAX_MODEL_LEN

# Model 9: DeepSeek-R1 Distill 8B Llama-based (~16GB)
run_model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" 120

# =============================================================================
#  PART 2: SPECULATIVE DECODING (Llama 3.1 8B + Qwen3-8B)
# =============================================================================
log "PART 2: SPECULATIVE DECODING RUNS"

# --- Llama 3.1 8B: Eagle3 + Ngram on both engines ---
log "START SPEC-DEC: Llama 3.1 8B (Eagle3 + Ngram, both engines)"
LLAMA="meta-llama/Llama-3.1-8B-Instruct"

# Baseline (re-run for fair comparison in same session)
run_specdec "$LLAMA" "vllm" 120
run_specdec "$LLAMA" "sglang" 120

# Eagle3
run_specdec "$LLAMA" "vllm-eagle3" 180
run_specdec "$LLAMA" "sglang-eagle3" 180

# Ngram
run_specdec "$LLAMA" "vllm-ngram" 120
run_specdec "$LLAMA" "sglang-ngram" 120

log "DONE SPEC-DEC: Llama 3.1 8B"

# --- Qwen3-8B: Eagle3 (vLLM only) + Ngram on both engines ---
log "START SPEC-DEC: Qwen3-8B (Eagle3 vLLM + Ngram both engines)"
QWEN="Qwen/Qwen3-8B"

# Baseline
run_specdec "$QWEN" "vllm" 120
run_specdec "$QWEN" "sglang" 120

# Eagle3 — vLLM only (SGLang Eagle3 draft not yet available for Qwen3)
run_specdec "$QWEN" "vllm-eagle3" 180

# Ngram
run_specdec "$QWEN" "vllm-ngram" 120
run_specdec "$QWEN" "sglang-ngram" 120

log "DONE SPEC-DEC: Qwen3-8B"

# =============================================================================
#  SUMMARY
# =============================================================================
log "ALL RUNS COMPLETE"
echo ""
echo "Results: ${RESULTS_DIR}/"
echo ""
echo "Next steps:"
echo "  1. Review:  ls -la ${RESULTS_DIR}/*/"
echo "  2. Report:  python run_experiment.py final-report --output reports/phase_a_report.md"
echo "  3. HTML:    python run_experiment.py report --output reports/phase_a_report.html"
