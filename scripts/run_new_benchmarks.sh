#!/usr/bin/env bash
# =============================================================================
# New Benchmark Suite — Phase 1 (Variance), Phase 2 (Concurrency-64),
#                       Phase 3 (Decode-Length Sweep)
#
# Manages Docker containers per model/engine.
# Runs only models that are cached locally or not gated (no HF token required).
#
# All models included — full HuggingFace access available.
#
# vLLM binds to 172.31.73.137:8000 (avoids conflict with selfhosted-chat-api
# which occupies 127.0.0.1:8000). SGLang binds to 0.0.0.0:8001.
#
# Usage:
#   chmod +x scripts/run_new_benchmarks.sh
#   nohup bash scripts/run_new_benchmarks.sh 2>&1 | tee logs/new_benchmarks_$(date +%Y%m%dT%H%M%S).log &
#
# Options:
#   --phase1   Run only Phase 1 (variance subset)
#   --phase2   Run only Phase 2 (concurrency-64)
#   --phase3   Run only Phase 3 (decode-length sweep)
#   (default: all three phases)
# =============================================================================

set +e

PYTHON="conda run --no-capture-output -n base python"
VLLM_IMAGE="vllm/vllm-openai:v0.18.0-cu130"
SGLANG_IMAGE="lmsysorg/sglang:nightly-dev-cu13-20260321-94194537"

# vLLM binds to the external NIC to avoid colliding with selfhosted-chat-api
VLLM_BIND_IP="172.31.73.137"
VLLM_HOST="172.31.73.137"
SGLANG_HOST="localhost"

ERRORS=()
COMPLETED=0

# ── Parse args ───────────────────────────────────────────────────────────────
RUN_PHASE1=true
RUN_PHASE2=true
RUN_PHASE3=true
for arg in "$@"; do
    case "$arg" in
        --phase1) RUN_PHASE2=false; RUN_PHASE3=false ;;
        --phase2) RUN_PHASE1=false; RUN_PHASE3=false ;;
        --phase3) RUN_PHASE1=false; RUN_PHASE2=false ;;
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

# ── Load HF token from .env if present ──────────────────────────────────────
HF_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
if [ -z "$HF_TOKEN" ] && [ -f .env ]; then
    HF_TOKEN=$(grep "^HUGGING_FACE_HUB_TOKEN=" .env | cut -d= -f2)
fi

# ── Preflight ────────────────────────────────────────────────────────────────
log "PREFLIGHT"
echo -n "  Python: "; $PYTHON --version 2>&1
echo -n "  Docker: "; docker version --format '{{.Server.Version}}' 2>&1
echo -n "  GPU:    "; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1
echo    "  HF token: $([ -n "$HF_TOKEN" ] && echo 'found' || echo 'NOT FOUND — Gemma/Llama downloads will fail (cached Llama is OK)')"
echo    "  vLLM bind: ${VLLM_BIND_IP}:8000"
echo    "  SGLang:    localhost:8001"
mkdir -p logs results_variance results_concurrency64 results_decode_sweep reports
docker rm -f bench-vllm bench-sglang 2>/dev/null || true
log "PREFLIGHT COMPLETE"

# ── Utility: cleanup a container ─────────────────────────────────────────────
cleanup() { docker rm -f "$1" 2>/dev/null; sleep 3; }

# ── Utility: wait for vLLM to be healthy ────────────────────────────────────
wait_vllm() {
    local max_wait="${1:-300}"
    local elapsed=0
    echo -n "  Waiting for vLLM health"
    while ! curl -sf "http://${VLLM_HOST}:8000/health" >/dev/null 2>&1; do
        sleep 5; elapsed=$((elapsed+5))
        echo -n "."
        if [ "$elapsed" -ge "$max_wait" ]; then
            echo " TIMEOUT"
            return 1
        fi
    done
    echo " ready (${elapsed}s)"
    return 0
}

# ── Utility: wait for SGLang to be healthy ──────────────────────────────────
wait_sglang() {
    local max_wait="${1:-300}"
    local elapsed=0
    echo -n "  Waiting for SGLang health"
    while ! curl -sf "http://${SGLANG_HOST}:8001/health" >/dev/null 2>&1; do
        sleep 5; elapsed=$((elapsed+5))
        echo -n "."
        if [ "$elapsed" -ge "$max_wait" ]; then
            echo " TIMEOUT"
            return 1
        fi
    done
    echo " ready (${elapsed}s)"
    return 0
}

# ── Run vLLM benchmark block ─────────────────────────────────────────────────
# Args: model  scenarios  iterations  cooldown  output_dir  [extra docker args...]
run_vllm_model() {
    local model="$1"
    local scenarios="$2"
    local iterations="$3"
    local cooldown="$4"
    local output_dir="$5"
    shift 5
    local extra_docker_args=("$@")

    log "vLLM — ${model}"
    cleanup bench-vllm

    # Default vLLM server args — extra_docker_args can override (e.g. --max-model-len 4096
    # for Gemma 3 4B, or --enforce-eager for models with CUDA graph incompatibilities).
    local vllm_server_args=(
        --model "$model"
        --host 0.0.0.0
        --port 8000
        --max-model-len 8192
        --gpu-memory-utilization 0.85
        --enable-prefix-caching
        --served-model-name "$model"
        "${extra_docker_args[@]}"
    )

    docker run -d \
        --name bench-vllm \
        --gpus '"device=0"' \
        -p "${VLLM_BIND_IP}:8000:8000" \
        --shm-size 10g \
        -v "$(pwd)/model-cache:/root/.cache/huggingface" \
        -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
        -e "CUDA_VISIBLE_DEVICES=0" \
        "$VLLM_IMAGE" \
        "${vllm_server_args[@]}"

    if [ $? -ne 0 ]; then
        error "vLLM docker run failed for ${model}"; return 1
    fi

    if ! wait_vllm 600; then
        error "vLLM failed to start for ${model}"
        docker logs bench-vllm --tail 40 2>&1
        cleanup bench-vllm; return 1
    fi

    $PYTHON run_experiment.py matrix \
        --model "$model" \
        --scenarios "$scenarios" \
        --engines vllm \
        --iterations "$iterations" \
        --cooldown-seconds "$cooldown" \
        --vllm-host "$VLLM_HOST" \
        --output-dir "$output_dir"
    local rc=$?
    [ $rc -ne 0 ] && error "vLLM benchmark failed for ${model} (exit ${rc})"

    cleanup bench-vllm
    return $rc
}

# ── Run SGLang benchmark block ───────────────────────────────────────────────
run_sglang_model() {
    local model="$1"
    local scenarios="$2"
    local iterations="$3"
    local cooldown="$4"
    local output_dir="$5"
    shift 5
    local extra_docker_args=("$@")

    log "SGLang — ${model}"
    cleanup bench-sglang

    docker run -d \
        --name bench-sglang \
        --gpus '"device=0"' \
        -p "8001:8001" \
        --shm-size 10g \
        -v "$(pwd)/model-cache:/root/.cache/huggingface" \
        -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
        -e "CUDA_VISIBLE_DEVICES=0" \
        -e "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1" \
        "${extra_docker_args[@]}" \
        "$SGLANG_IMAGE" \
        python -m sglang.launch_server \
        --model-path "$model" \
        --host 0.0.0.0 \
        --port 8001 \
        --mem-fraction-static 0.85 \
        --context-length 8192

    if [ $? -ne 0 ]; then
        error "SGLang docker run failed for ${model}"; return 1
    fi

    if ! wait_sglang 600; then
        error "SGLang failed to start for ${model}"
        docker logs bench-sglang --tail 40 2>&1
        cleanup bench-sglang; return 1
    fi

    $PYTHON run_experiment.py matrix \
        --model "$model" \
        --scenarios "$scenarios" \
        --engines sglang \
        --iterations "$iterations" \
        --cooldown-seconds "$cooldown" \
        --sglang-host "$SGLANG_HOST" \
        --output-dir "$output_dir"
    local rc=$?
    [ $rc -ne 0 ] && error "SGLang benchmark failed for ${model} (exit ${rc})"

    cleanup bench-sglang
    return $rc
}

# =============================================================================
#  PHASE 1 — Variance Subset
#  Credibility backbone: 5 iterations × 5 scenarios × 2 engines × 4 models
# =============================================================================
if [ "$RUN_PHASE1" = "true" ]; then
    log "PHASE 1 — VARIANCE SUBSET"
    echo "  Output dir : results_variance/"
    echo "  Iterations : 5"
    echo "  Cooldown   : 300s"
    echo "  Models     : gemma-2-2b-it, Phi-4-mini, Llama-3.1-8B, gemma-3-4b-it"
    echo "  Scenarios  : 5 baseline scenarios"

    VARIANCE_SCENARIOS="single_request_latency,throughput_ramp,long_context_stress,prefix_sharing_benefit,structured_generation_speed"
    VARIANCE_MODELS=(
        "google/gemma-2-2b-it"
        "microsoft/Phi-4-mini-instruct"
        "meta-llama/Llama-3.1-8B-Instruct"
        "google/gemma-3-4b-it"
    )

    for model in "${VARIANCE_MODELS[@]}"; do
        if [ "$model" = "google/gemma-3-4b-it" ]; then
            # Gemma 3 4B: vLLM requires --enforce-eager (hybrid sliding-window attention
            # is incompatible with CUDA graph capture) and a reduced max-model-len.
            run_vllm_model "$model" "$VARIANCE_SCENARIOS" 5 300 "results_variance" \
                --max-model-len 4096 --enforce-eager --disable-frontend-multiprocessing
        else
            run_vllm_model "$model" "$VARIANCE_SCENARIOS" 5 300 "results_variance"
        fi
        run_sglang_model "$model" "$VARIANCE_SCENARIOS" 5 300 "results_variance"
        ((COMPLETED++))
    done

    log "PHASE 1 COMPLETE"
fi

# =============================================================================
#  PHASE 2 — Concurrency-64 Extended Ramp
#  Adds concurrency=64 to find saturation / OOM ceiling on 7-9B models
# =============================================================================
if [ "$RUN_PHASE2" = "true" ]; then
    log "PHASE 2 — CONCURRENCY-64 EXTENDED RAMP"
    echo "  Output dir : results_concurrency64/"
    echo "  Iterations : 3"
    echo "  Cooldown   : 300s"
    echo "  Models     : Llama-3.1-8B, Qwen3-8B, Mistral-7B, gemma-2-9b-it"
    echo "  Scenario   : throughput_ramp_extended"

    CONC64_SCENARIO="throughput_ramp_extended"
    CONC64_MODELS=(
        "meta-llama/Llama-3.1-8B-Instruct"
        "Qwen/Qwen3-8B"
        "mistralai/Mistral-7B-Instruct-v0.3"
        "google/gemma-2-9b-it"
    )

    for model in "${CONC64_MODELS[@]}"; do
        run_vllm_model   "$model" "$CONC64_SCENARIO" 3 300 "results_concurrency64"
        run_sglang_model "$model" "$CONC64_SCENARIO" 3 300 "results_concurrency64"
        ((COMPLETED++))
    done

    log "PHASE 2 COMPLETE"
fi

# =============================================================================
#  PHASE 3 — Decode-Length Sweep
#  Fixed ~512-token prompts, max_output_tokens ∈ {64, 256, 1024, 4096}
# =============================================================================
if [ "$RUN_PHASE3" = "true" ]; then
    log "PHASE 3 — DECODE-LENGTH SWEEP"
    echo "  Output dir : results_decode_sweep/"
    echo "  Iterations : 3"
    echo "  Cooldown   : 300s"
    echo "  Models     : gemma-2-2b-it, Phi-4-mini, Llama-3.1-8B, gemma-3-4b-it"
    echo "  Scenarios  : decode_length_sweep_{64,256,1024,4096}"

    DECODE_SCENARIOS="decode_length_sweep_64,decode_length_sweep_256,decode_length_sweep_1024,decode_length_sweep_4096"
    DECODE_MODELS=(
        "google/gemma-2-2b-it"
        "microsoft/Phi-4-mini-instruct"
        "meta-llama/Llama-3.1-8B-Instruct"
        "google/gemma-3-4b-it"
    )

    for model in "${DECODE_MODELS[@]}"; do
        if [ "$model" = "google/gemma-3-4b-it" ]; then
            run_vllm_model "$model" "$DECODE_SCENARIOS" 3 300 "results_decode_sweep" \
                --max-model-len 4096 --enforce-eager --disable-frontend-multiprocessing
        else
            run_vllm_model "$model" "$DECODE_SCENARIOS" 3 300 "results_decode_sweep"
        fi
        run_sglang_model "$model" "$DECODE_SCENARIOS" 3 300 "results_decode_sweep"
        ((COMPLETED++))
    done

    log "PHASE 3 COMPLETE"
fi

# =============================================================================
#  SUMMARY
# =============================================================================
log "ALL PHASES COMPLETE"

echo ""
echo "Results:"
for dir in results_variance results_concurrency64 results_decode_sweep; do
    count=$(find "$dir" -name "*.json" -not -name "*manifest*" 2>/dev/null | wc -l | tr -d ' ')
    printf "  %-30s %3s result files\n" "$dir/" "$count"
done

echo ""
if [ ${#ERRORS[@]} -eq 0 ]; then
    echo "STATUS: ALL PASSED — no errors"
else
    echo "STATUS: ${#ERRORS[@]} ERROR(S):"
    for i in "${!ERRORS[@]}"; do
        echo "  $((i+1)). ${ERRORS[$i]}"
    done
fi

echo ""
echo "Next steps:"
echo "  python -m analysis.variance_analysis --results-dir results_variance"
echo "  python -m analysis.tpot_analysis --results-dir results_variance"
echo "  python -m analysis.decode_length_analysis --results-dir results_decode_sweep"
