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
# vLLM binds to the primary private IP (auto-detected) to avoid conflict with
# selfhosted-chat-api on 127.0.0.1:8000. SGLang binds to the same IP on :8001.
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
SGLANG_IMAGE="lmsysorg/sglang:v0.5.10.post1-cu130"

# Gemma 4 (E2B/E4B) needs Transformers >= 5.5.0 → only shipped in :latest images.
GEMMA4_VLLM_IMAGE="vllm/vllm-openai:latest"
GEMMA4_SGLANG_IMAGE="lmsysorg/sglang:latest"

# vLLM binds to the external NIC to avoid colliding with selfhosted-chat-api
VLLM_BIND_IP="$(hostname -I | awk '{print $1}')"
VLLM_HOST="$VLLM_BIND_IP"
SGLANG_HOST="$VLLM_BIND_IP"

ERRORS=()
COMPLETED=0

# ── Parse args ───────────────────────────────────────────────────────────────
RUN_PHASE1=true
RUN_PHASE2=true
RUN_PHASE3=true
RUN_PHASE4=false     # Part 4 (Gemma 4) — off by default; opt-in via --phase4 or --all
RUN_PHASE3_REDO=false
for arg in "$@"; do
    case "$arg" in
        --phase1) RUN_PHASE2=false; RUN_PHASE3=false; RUN_PHASE4=false ;;
        --phase2) RUN_PHASE1=false; RUN_PHASE3=false; RUN_PHASE4=false ;;
        --phase3) RUN_PHASE1=false; RUN_PHASE2=false; RUN_PHASE4=false ;;
        --phase4) RUN_PHASE1=false; RUN_PHASE2=false; RUN_PHASE3=false; RUN_PHASE4=true ;;
        --all)    RUN_PHASE1=true;  RUN_PHASE2=true;  RUN_PHASE3=true;  RUN_PHASE4=true ;;
        --phase3-redo)
            RUN_PHASE1=false; RUN_PHASE2=false; RUN_PHASE3=true; RUN_PHASE4=false
            RUN_PHASE3_REDO=true
            ;;
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

# ── Ensure docker runs with sudo if needed ───────────────────────────────────
DOCKER="docker"
if ! docker info >/dev/null 2>&1; then
    if sudo docker info >/dev/null 2>&1; then
        DOCKER="sudo docker"
    else
        echo "ERROR: Docker is not accessible. Install Docker or add user to docker group." >&2
        exit 1
    fi
fi

# ── Pull Docker images if not present locally ────────────────────────────────
pull_image_if_missing() {
    local image="$1"
    if $DOCKER image inspect "$image" >/dev/null 2>&1; then
        echo "  [pull] $image — already present, skipping pull"
    else
        echo "  [pull] $image — not found locally, pulling..."
        if ! $DOCKER pull "$image"; then
            echo "ERROR: Failed to pull Docker image: $image" >&2
            exit 1
        fi
        echo "  [pull] $image — done"
    fi
}

# ── Preflight ────────────────────────────────────────────────────────────────
log "PREFLIGHT"
echo -n "  Python: "; $PYTHON --version 2>&1
echo -n "  Docker: "; $DOCKER version --format '{{.Server.Version}}' 2>&1
echo -n "  GPU:    "; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1
echo    "  HF token: $([ -n "$HF_TOKEN" ] && echo 'found' || echo 'NOT FOUND — Gemma/Llama downloads will fail (cached Llama is OK)')"
echo    "  vLLM bind: ${VLLM_BIND_IP}:8000"
echo    "  SGLang:    ${SGLANG_HOST}:8001"
mkdir -p logs results_variance results_concurrency64 results_decode_sweep reports

log "PULLING DOCKER IMAGES (if not cached)"
pull_image_if_missing "$VLLM_IMAGE"
pull_image_if_missing "$SGLANG_IMAGE"

$DOCKER rm -f bench-vllm bench-sglang 2>/dev/null || true
log "PREFLIGHT COMPLETE"

# ── Utility: cleanup a container ─────────────────────────────────────────────
cleanup() { $DOCKER rm -f "$1" 2>/dev/null; sleep 3; }

# ── Utility: wait for vLLM to be healthy ────────────────────────────────────
wait_vllm() {
    local max_wait="${1:-300}"
    local elapsed=0
    echo -n "  Waiting for vLLM health"
    while ! curl -sf "http://${VLLM_HOST}:8000/health" >/dev/null 2>&1; do
        # Bail out early if the container has already exited (e.g. auth error, OOM)
        if ! $DOCKER ps --format '{{.Names}}' | grep -q '^bench-vllm$'; then
            echo " CONTAINER EXITED"
            echo "  Container logs (last 20 lines):"
            $DOCKER logs bench-vllm --tail 20 2>&1 | sed 's/^/    /'
            return 1
        fi
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
        # Bail out early if the container has already exited (e.g. auth error, OOM)
        if ! $DOCKER ps --format '{{.Names}}' | grep -q '^bench-sglang$'; then
            echo " CONTAINER EXITED"
            echo "  Container logs (last 20 lines):"
            $DOCKER logs bench-sglang --tail 20 2>&1 | sed 's/^/    /'
            return 1
        fi
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

    $DOCKER run -d \
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
        $DOCKER logs bench-vllm --tail 40 2>&1
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

    $DOCKER run -d \
        --name bench-sglang \
        --gpus '"device=0"' \
        -p "${SGLANG_HOST}:8001:8001" \
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
        $DOCKER logs bench-sglang --tail 40 2>&1
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
#
#  STATUS (as of 2026-04-18): PARTIAL — 131 / 200 files in results_variance/
#    google/gemma-2-2b-it             — ✅ COMPLETE (50/50)
#    microsoft/Phi-4-mini-instruct    — ✅ COMPLETE (50/50)
#    meta-llama/Llama-3.1-8B-Instruct — ⚠️ PARTIAL  (31/50)
#         vLLM   : all 5 scenarios × 5 iter  ✓
#         SGLang : only single_request_latency complete; long_context_stress,
#                  prefix_sharing_benefit, structured_generation_speed, and
#                  throughput_ramp are missing or incomplete.
#    google/gemma-3-4b-it             — ❌ MISSING  (0/50, entire model absent)
#
#  The resume block below runs only the missing cells. run_experiment.py does
#  not check for existing files per-iteration, so re-runs append new iterations
#  with fresh timestamps. Safe to re-launch — the completed models are gated
#  behind compgen checks and will skip cleanly.
# =============================================================================
if [ "$RUN_PHASE1" = "true" ]; then
    log "PHASE 1 — VARIANCE SUBSET (resume, missing cells only)"

    VARIANCE_SCENARIOS_ALL="single_request_latency,throughput_ramp,long_context_stress,prefix_sharing_benefit,structured_generation_speed"

    # Helper: returns 0 if a model has a "complete" variance set. We treat a
    # (model, engine) pair as complete if all 5 scenarios each have ≥5 files.
    phase1_engine_complete() {
        local slug="$1"; local engine="$2"
        local scen
        for scen in single_request_latency throughput_ramp long_context_stress prefix_sharing_benefit structured_generation_speed; do
            local n
            n=$(compgen -G "results_variance/${slug}/${scen}_${engine}_*.json" 2>/dev/null | wc -l)
            if [ "$n" -lt 5 ]; then return 1; fi
        done
        return 0
    }

    # ── gemma-2-2b-it — COMPLETE (50/50). Skip unless results_variance wiped.
    if phase1_engine_complete "gemma-2-2b-it" "vllm" && phase1_engine_complete "gemma-2-2b-it" "sglang"; then
        echo "  [SKIP] gemma-2-2b-it — both engines complete"
    else
        run_vllm_model   "google/gemma-2-2b-it" "$VARIANCE_SCENARIOS_ALL" 5 10 "results_variance"
        run_sglang_model "google/gemma-2-2b-it" "$VARIANCE_SCENARIOS_ALL" 5 10 "results_variance"
    fi

    # ── phi-4-mini — COMPLETE (50/50). Skip.
    if phase1_engine_complete "phi-4-mini-instruct" "vllm" && phase1_engine_complete "phi-4-mini-instruct" "sglang"; then
        echo "  [SKIP] phi-4-mini-instruct — both engines complete"
    else
        run_vllm_model   "microsoft/Phi-4-mini-instruct" "$VARIANCE_SCENARIOS_ALL" 5 10 "results_variance"
        run_sglang_model "microsoft/Phi-4-mini-instruct" "$VARIANCE_SCENARIOS_ALL" 5 10 "results_variance"
    fi

    # ── Llama-3.1-8B — vLLM complete, SGLang incomplete.
    if phase1_engine_complete "llama-3-1-8b-instruct" "vllm"; then
        echo "  [SKIP] vllm llama-3-1-8b-instruct — complete"
    else
        run_vllm_model   "meta-llama/Llama-3.1-8B-Instruct" "$VARIANCE_SCENARIOS_ALL" 5 10 "results_variance"
    fi
    if phase1_engine_complete "llama-3-1-8b-instruct" "sglang"; then
        echo "  [SKIP] sglang llama-3-1-8b-instruct — complete"
    else
        # Only 4 scenarios are missing (single_request_latency already has 5 iters).
        # Re-running all 5 scenarios would duplicate the complete one; run only
        # what's missing. Cheaper and avoids polluting existing stats.
        PHASE1_LLAMA_SGLANG_MISSING="long_context_stress,prefix_sharing_benefit,structured_generation_speed,throughput_ramp"
        run_sglang_model "meta-llama/Llama-3.1-8B-Instruct" "$PHASE1_LLAMA_SGLANG_MISSING" 5 10 "results_variance"
    fi

    # ── gemma-3-4b-it — entirely missing. Needs --max-model-len 4096 --enforce-eager
    #    --disable-frontend-multiprocessing on vLLM (Gemma 3 CUDA graph quirks on A10G).
    if phase1_engine_complete "gemma-3-4b-it" "vllm"; then
        echo "  [SKIP] vllm gemma-3-4b-it — complete"
    else
        run_vllm_model "google/gemma-3-4b-it" "$VARIANCE_SCENARIOS_ALL" 5 10 "results_variance" \
            --max-model-len 4096 --enforce-eager --disable-frontend-multiprocessing
    fi
    if phase1_engine_complete "gemma-3-4b-it" "sglang"; then
        echo "  [SKIP] sglang gemma-3-4b-it — complete"
    else
        run_sglang_model "google/gemma-3-4b-it" "$VARIANCE_SCENARIOS_ALL" 5 10 "results_variance"
    fi

    log "PHASE 1 COMPLETE"
fi

# =============================================================================
#  PHASE 2 — Concurrency-64 Extended Ramp
#  Adds concurrency=64 to find saturation / OOM ceiling on 7-9B models
#
#  STATUS (as of 2026-04-18): COMPLETE — all 8 cells done, 0% error rate
#    Qwen/Qwen3-8B                     — vLLM ✓  SGLang ✓
#    mistralai/Mistral-7B-Instruct-v0.3 — vLLM ✓  SGLang ✓
#    meta-llama/Llama-3.1-8B-Instruct  — vLLM ✓  SGLang ✓
#    google/gemma-2-9b-it              — vLLM ✓  SGLang ✓  (vLLM needs --max-model-len 2048)
#
#  Cost-conscious plan: single iteration per (model × engine), and SKIP cells
#  whose result file already exists. Safe to re-run idempotently — won't re-do
#  completed cells, won't double-spend GPU.
#
#  Run with:
#    nohup bash scripts/run_new_benchmarks.sh --phase2 2>&1 \
#      | tee logs/phase2_resume_$(date +%Y%m%dT%H%M%S).log &
# =============================================================================

# Helper: returns 0 if a result file already exists for (model_dir_slug, engine).
# run_experiment.py writes to results_concurrency64/<model_slug>/throughput_ramp_extended_<engine>_*.json
phase2_has_result() {
    local slug="$1"; local engine="$2"
    compgen -G "results_concurrency64/${slug}/throughput_ramp_extended_${engine}_*.json" >/dev/null
}

# Map HF model id → on-disk directory slug used by run_experiment.py
phase2_slug_for() {
    case "$1" in
        "meta-llama/Llama-3.1-8B-Instruct")     echo "llama-3-1-8b-instruct" ;;
        "Qwen/Qwen3-8B")                        echo "qwen3-8b" ;;
        "mistralai/Mistral-7B-Instruct-v0.3")   echo "mistral-7b-instruct-v0-3" ;;
        "google/gemma-2-9b-it")                 echo "gemma-2-9b-it" ;;
        *)                                       echo "" ;;
    esac
}

if [ "$RUN_PHASE2" = "true" ]; then
    log "PHASE 2 — CONCURRENCY-64 EXTENDED RAMP (resume, missing-only, 1 iter)"
    echo "  Output dir : results_concurrency64/"
    echo "  Iterations : 1   (single run per engine to conserve GPU cost)"
    echo "  Cooldown   : 10s between scenarios"
    echo "  Behavior   : skip any (model,engine) that already has a result file"

    CONC64_SCENARIO="throughput_ramp_extended"
    CONC64_ITERATIONS=1
    CONC64_COOLDOWN=10

    CONC64_MODELS=(
        "meta-llama/Llama-3.1-8B-Instruct"
        "Qwen/Qwen3-8B"
        "mistralai/Mistral-7B-Instruct-v0.3"
        "google/gemma-2-9b-it"
    )

    for model in "${CONC64_MODELS[@]}"; do
        slug=$(phase2_slug_for "$model")
        if [ -z "$slug" ]; then
            error "phase2: no slug mapping for ${model}"; continue
        fi

        # Per-model vLLM overrides. gemma-2-9b-it on A10G 24GB cannot hold an
        # 8192-token KV cache at 9B params — need a smaller window. 384 tok is
        # enough for this scenario (128 prompt + 256 output); use 2048 for
        # headroom. --enforce-eager avoids CUDA graph issues on gemma-2.
        vllm_extra=()
        case "$model" in
            "google/gemma-2-9b-it")
                vllm_extra=(--max-model-len 2048 --enforce-eager --gpu-memory-utilization 0.90)
                ;;
        esac

        # vLLM
        if phase2_has_result "$slug" "vllm"; then
            echo "  [SKIP] vllm ${model} — result file already exists"
        else
            run_vllm_model "$model" "$CONC64_SCENARIO" "$CONC64_ITERATIONS" "$CONC64_COOLDOWN" "results_concurrency64" "${vllm_extra[@]}"
        fi

        # SGLang
        if phase2_has_result "$slug" "sglang"; then
            echo "  [SKIP] sglang ${model} — result file already exists"
        else
            run_sglang_model "$model" "$CONC64_SCENARIO" "$CONC64_ITERATIONS" "$CONC64_COOLDOWN" "results_concurrency64"
        fi

        # Notes per model:
        #   Qwen/Qwen3-8B                       — no special flags needed
        #   mistralai/Mistral-7B-Instruct-v0.3  — gated, requires HF_TOKEN
        #   google/gemma-2-9b-it                — largest of the set; if vLLM OOMs at conc=64
        #                                         on A10G 24GB, retry with
        #                                         --gpu-memory-utilization 0.90 or --max-model-len 4096
        ((COMPLETED++))
    done

    log "PHASE 2 COMPLETE"
fi

# =============================================================================
#  PHASE 3 — Decode-Length Sweep
#  Fixed ~512-token prompts, max_output_tokens ∈ {64, 256, 1024, 4096}
#
#  STATUS (as of 2026-04-18): 72 files, all cells valid but iteration counts
#  are uneven. Target: ≥3 iterations per (model, length, engine) so CIs/mean
#  are reportable.
#    google/gemma-2-2b-it             — ✅ 3 iter per cell (24 files)
#    microsoft/Phi-4-mini-instruct    — ✅ 3 iter per cell (24 files)
#    meta-llama/Llama-3.1-8B-Instruct — ⚠️  2 iter per cell (16 files, needs +1)
#    google/gemma-3-4b-it             — ⚠️  1 iter per cell (8 files, needs +2)
#
#  The top-up block below computes each model's minimum iteration count across
#  its 4 decode lengths and runs only the deficit needed to reach TARGET_ITERS.
#  Over-runs nothing: complete models skip, partial models run exactly the
#  missing iterations. Cost-bounded and idempotent.
# =============================================================================

# Return the minimum iteration count (across 4 decode lengths) for (slug, engine).
phase3_min_iters() {
    local slug="$1"; local engine="$2"
    local m=999 length n
    for length in 64 256 1024 4096; do
        n=$(compgen -G "results_decode_sweep/${slug}/decode_length_sweep_${length}_${engine}_*.json" 2>/dev/null | wc -l)
        [ "$n" -lt "$m" ] && m="$n"
    done
    echo "$m"
}

if [ "$RUN_PHASE3" = "true" ] && [ "$RUN_PHASE3_REDO" = "false" ]; then
    log "PHASE 3 — DECODE-LENGTH SWEEP (top-up to ≥3 iter per cell)"

    DECODE_SCENARIOS_ALL="decode_length_sweep_64,decode_length_sweep_256,decode_length_sweep_1024,decode_length_sweep_4096"
    PHASE3_TARGET_ITERS=3

    # Per-model top-up. Deficit = TARGET - min(iters across decode lengths).
    # Iterations passed to matrix are appended with fresh timestamps.
    # Entry format: "hf_id|slug|vllm_flags|image_tag"
    #   image_tag = ""       → use pinned VLLM_IMAGE / SGLANG_IMAGE
    #   image_tag = "gemma4" → swap to GEMMA4_*_IMAGE (Transformers >= 5.5.0)
    # Gemma 4 entries are placed FIRST so any Transformers-from-git install
    # or arch issue surfaces early, before the safe llama/gemma-3-4b top-ups.
    for entry in \
        "google/gemma-4-E2B-it|gemma-4-e2b-it||gemma4" \
        "google/gemma-4-E4B-it|gemma-4-e4b-it||gemma4" \
        "google/gemma-2-2b-it|gemma-2-2b-it||" \
        "microsoft/Phi-4-mini-instruct|phi-4-mini-instruct||" \
        "meta-llama/Llama-3.1-8B-Instruct|llama-3-1-8b-instruct||" \
        "google/gemma-3-4b-it|gemma-3-4b-it|--max-model-len 5632 --enforce-eager --disable-frontend-multiprocessing|" ; do
        IFS='|' read -r model slug vllm_flags image_tag <<< "$entry"

        # SKIP_GEMMA4=1 bypasses Gemma 4 entries entirely (use while disk is tight
        # or if the :latest image/transformers-from-git install misbehaves).
        if [ "$image_tag" = "gemma4" ] && [ "${SKIP_GEMMA4:-0}" = "1" ]; then
            echo "  [SKIP] ${model} — SKIP_GEMMA4=1"
            continue
        fi

        # Swap in the newer :latest images for Gemma 4 only — scoped to this
        # model's run, then restored so other entries keep the pinned images.
        # SKIP_GEMMA4_SGLANG=1 avoids pulling/running the sglang:latest image
        # (~50 GB) when disk is tight.
        if [ "$image_tag" = "gemma4" ]; then
            saved_vllm="$VLLM_IMAGE"; saved_sglang="$SGLANG_IMAGE"
            VLLM_IMAGE="$GEMMA4_VLLM_IMAGE"; SGLANG_IMAGE="$GEMMA4_SGLANG_IMAGE"
            pull_image_if_missing "$VLLM_IMAGE"
            [ "${SKIP_GEMMA4_SGLANG:-0}" = "1" ] || pull_image_if_missing "$SGLANG_IMAGE"
        fi

        gemma4_engines=(vllm sglang)
        [ "$image_tag" = "gemma4" ] && [ "${SKIP_GEMMA4_SGLANG:-0}" = "1" ] && gemma4_engines=(vllm)
        for engine in "${gemma4_engines[@]}"; do
            existing=$(phase3_min_iters "$slug" "$engine")
            deficit=$(( PHASE3_TARGET_ITERS - existing ))
            if [ "$deficit" -le 0 ]; then
                echo "  [SKIP] ${engine} ${model} — already ${existing} iter (≥${PHASE3_TARGET_ITERS})"
                continue
            fi
            echo "  [TOPUP] ${engine} ${model} — have ${existing}, adding ${deficit} iter"
            # Gemma 4 entries need the dedicated launcher (apt git + pip
            # transformers-from-git). Other models use the pinned-image path.
            if [ "$image_tag" = "gemma4" ]; then
                if [ "$engine" = "vllm" ]; then
                    MAX_MODEL_LEN=5632 run_vllm_gemma4 "$model" "$DECODE_SCENARIOS_ALL" "$deficit" 15 "results_decode_sweep" "vllm"
                else
                    MAX_MODEL_LEN=5632 run_sglang_gemma4 "$model" "$DECODE_SCENARIOS_ALL" "$deficit" 15 "results_decode_sweep" "sglang"
                fi
            else
                if [ "$engine" = "vllm" ]; then
                    # shellcheck disable=SC2086 # intentional word-split of vllm_flags
                    run_vllm_model "$model" "$DECODE_SCENARIOS_ALL" "$deficit" 15 "results_decode_sweep" $vllm_flags
                else
                    run_sglang_model "$model" "$DECODE_SCENARIOS_ALL" "$deficit" 15 "results_decode_sweep"
                fi
            fi
        done

        if [ "$image_tag" = "gemma4" ]; then
            VLLM_IMAGE="$saved_vllm"; SGLANG_IMAGE="$saved_sglang"
        fi
        ((COMPLETED++))
    done

    log "PHASE 3 COMPLETE"
fi

# =============================================================================
#  PHASE 3 REDO — Targeted reruns for the 2 missing/partial cells
#    1. gemma-3-4b-it / decode_length_sweep_4096 / vLLM  — DONE 2026-04-17
#    2. llama-3-1-8b-instruct / decode_length_sweep_4096 / vLLM  — DONE 2026-04-17
#
#  STATUS: both cells now have valid result files. Block is idempotent — it
#  skips cells that already have a 4096_vllm_*.json; re-running is safe but
#  will be a no-op unless those files are removed.
# =============================================================================
if [ "$RUN_PHASE3_REDO" = "true" ]; then
    log "PHASE 3 REDO — TARGETED RERUNS (idempotent)"
    echo "  Output dir : results_decode_sweep/"

    $DOCKER rm -f bench-vllm bench-sglang 2>/dev/null || true
    sleep 5

    REDO_SCENARIO="decode_length_sweep_4096"

    # (1) gemma-3-4b-it — needs --max-model-len >= 4608 (prompt ~512 + output 4096).
    if compgen -G "results_decode_sweep/gemma-3-4b-it/decode_length_sweep_4096_vllm_*.json" >/dev/null; then
        echo "  [SKIP] vllm gemma-3-4b-it 4096 — result file already exists"
    else
        run_vllm_model "google/gemma-3-4b-it" "$REDO_SCENARIO" 1 10 "results_decode_sweep" \
            --max-model-len 5632 --enforce-eager --disable-frontend-multiprocessing
    fi

    # (2) llama-3-1-8b-instruct — default --max-model-len 8192 is sufficient.
    if compgen -G "results_decode_sweep/llama-3-1-8b-instruct/decode_length_sweep_4096_vllm_*.json" >/dev/null; then
        echo "  [SKIP] vllm llama-3-1-8b 4096 — result file already exists"
    else
        run_vllm_model "meta-llama/Llama-3.1-8B-Instruct" "$REDO_SCENARIO" 1 10 "results_decode_sweep"
    fi

    ((COMPLETED+=2))
    log "PHASE 3 REDO COMPLETE"
fi

# =============================================================================
#  PHASE 4 — Gemma 4 baseline + ngram spec-dec  (results/)
#
#  Models    : google/gemma-4-E2B-it, google/gemma-4-E4B-it
#  Baseline  : 5 scenarios × {vllm, sglang}       → 20 cells
#  Ngram     : 2 scenarios × {vllm-ngram, sglang-ngram} → 8 cells
#  Iters     : 1 per cell (matches Part 0 baseline convention)
#
#  Gemma 4 requires Transformers >= 5.5.0 — uses GEMMA4_*_IMAGE and pip-
#  installs transformers-from-git inside the container on startup (the
#  :latest images lag the release). Skip any cell already in results/.
#
#  Idempotent resume: cells whose file already exists under
#    results/<slug>/<scenario>_<engine>_*.json
#  are skipped. Run with: bash scripts/run_new_benchmarks.sh --phase4
# =============================================================================

# Run Gemma 4 vLLM (baseline or ngram). Installs transformers-from-git on boot.
# Args: model scenarios iterations cooldown output_dir engine_label [extra server args...]
run_vllm_gemma4() {
    local model="$1" scenarios="$2" iterations="$3" cooldown="$4"
    local output_dir="$5" engine="$6"; shift 6
    local extra=("$@")
    local mml="${MAX_MODEL_LEN:-4096}" gmu="${GPU_MEM_UTIL:-0.85}"

    log "vLLM-Gemma4 (${engine}) — ${model}"
    cleanup bench-vllm

    local server_args=(
        --model "$model" --host 0.0.0.0 --port 8000
        --enable-prefix-caching --max-model-len "$mml"
        --gpu-memory-utilization "$gmu" --served-model-name "$model"
        "${extra[@]}"
    )
    local quoted
    quoted=$(printf '%q ' "${server_args[@]}")
    # vllm/vllm-openai:latest lacks `python` alias and `git` → install both.
    # Use python3 (canonical on the upstream image) to invoke vLLM.
    local cmd="(command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git)) && pip install -q --upgrade git+https://github.com/huggingface/transformers.git && exec python3 -m vllm.entrypoints.openai.api_server ${quoted}"

    $DOCKER run -d --name bench-vllm --gpus '"device=0"' \
        -p "${VLLM_BIND_IP}:8000:8000" --shm-size 10g \
        -v "$(pwd)/model-cache:/root/.cache/huggingface" \
        -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" -e "CUDA_VISIBLE_DEVICES=0" \
        --entrypoint bash "$GEMMA4_VLLM_IMAGE" -c "$cmd"
    [ $? -ne 0 ] && { error "vLLM-Gemma4 docker run failed for ${model}"; return 1; }

    if ! wait_vllm 900; then
        error "vLLM-Gemma4 failed to start for ${model}"
        $DOCKER logs bench-vllm --tail 40 2>&1
        cleanup bench-vllm; return 1
    fi

    $PYTHON run_experiment.py matrix \
        --model "$model" --scenarios "$scenarios" --engines "$engine" \
        --iterations "$iterations" --cooldown-seconds "$cooldown" \
        --vllm-host "$VLLM_HOST" --output-dir "$output_dir"
    local rc=$?
    [ $rc -ne 0 ] && error "vLLM-Gemma4 (${engine}) benchmark failed for ${model} (exit ${rc})"

    cleanup bench-vllm
    return $rc
}

# Run Gemma 4 SGLang (baseline or ngram).
run_sglang_gemma4() {
    local model="$1" scenarios="$2" iterations="$3" cooldown="$4"
    local output_dir="$5" engine="$6"; shift 6
    local extra=("$@")
    local mml="${MAX_MODEL_LEN:-4096}" mfs="${MEM_FRAC_STATIC:-0.85}"

    log "SGLang-Gemma4 (${engine}) — ${model}"
    cleanup bench-sglang

    local server_args=(
        --model-path "$model" --host 0.0.0.0 --port 8001
        --mem-fraction-static "$mfs" --context-length "$mml"
        --disable-cuda-graph
        "${extra[@]}"
    )
    local quoted
    quoted=$(printf '%q ' "${server_args[@]}")
    local cmd="(command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git)) && pip install -q --upgrade git+https://github.com/huggingface/transformers.git && exec python3 -m sglang.launch_server ${quoted}"

    $DOCKER run -d --name bench-sglang --gpus '"device=0"' \
        -p "${SGLANG_HOST}:8001:8001" --shm-size 10g \
        -v "$(pwd)/model-cache:/root/.cache/huggingface" \
        -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" -e "CUDA_VISIBLE_DEVICES=0" \
        -e "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1" \
        --entrypoint bash "$GEMMA4_SGLANG_IMAGE" -c "$cmd"
    [ $? -ne 0 ] && { error "SGLang-Gemma4 docker run failed for ${model}"; return 1; }

    if ! wait_sglang 900; then
        error "SGLang-Gemma4 failed to start for ${model}"
        $DOCKER logs bench-sglang --tail 40 2>&1
        cleanup bench-sglang; return 1
    fi

    $PYTHON run_experiment.py matrix \
        --model "$model" --scenarios "$scenarios" --engines "$engine" \
        --iterations "$iterations" --cooldown-seconds "$cooldown" \
        --sglang-host "$SGLANG_HOST" --output-dir "$output_dir"
    local rc=$?
    [ $rc -ne 0 ] && error "SGLang-Gemma4 (${engine}) benchmark failed for ${model} (exit ${rc})"

    cleanup bench-sglang
    return $rc
}

# Gather pending scenarios (comma-joined) for a given (model, engine) under results/.
phase4_pending_scenarios() {
    local slug="$1" engine="$2"; shift 2
    local scen out=""
    for scen in "$@"; do
        if ! compgen -G "results/${slug}/${scen}_${engine}_*.json" >/dev/null; then
            out="${out:+${out},}${scen}"
        fi
    done
    echo "$out"
}

if [ "$RUN_PHASE4" = "true" ]; then
    log "PHASE 4 — GEMMA 4 (baseline + ngram spec-dec, resume)"

    # Ensure the :latest images exist locally (may not have been pulled earlier).
    # SKIP_GEMMA4_SGLANG=1 skips pulling the sglang:latest image (~50 GB).
    pull_image_if_missing "$GEMMA4_VLLM_IMAGE"
    [ "${SKIP_GEMMA4_SGLANG:-0}" = "1" ] || pull_image_if_missing "$GEMMA4_SGLANG_IMAGE"

    PHASE4_BASELINE_SCENARIOS=(single_request_latency throughput_ramp long_context_stress prefix_sharing_benefit structured_generation_speed)
    PHASE4_NGRAM_SCENARIOS=(single_request_latency throughput_ramp)

    for entry in "google/gemma-4-E2B-it|gemma-4-e2b-it" "google/gemma-4-E4B-it|gemma-4-e4b-it"; do
        IFS='|' read -r model slug <<<"$entry"

        # ── Baseline vLLM ──
        pending=$(phase4_pending_scenarios "$slug" "vllm" "${PHASE4_BASELINE_SCENARIOS[@]}")
        if [ -z "$pending" ]; then
            echo "  [SKIP] vllm ${model} baseline — all 5 scenarios present"
        else
            echo "  [RUN]  vllm ${model} baseline — ${pending}"
            # --disable-frontend-multiprocessing removed in recent vLLM (on by
            # default). Keep --enforce-eager to dodge CUDA graph issues.
            MAX_MODEL_LEN=4096 run_vllm_gemma4 "$model" "$pending" 1 10 "results" "vllm" \
                --enforce-eager
        fi

        # ── Baseline SGLang ──
        if [ "${SKIP_GEMMA4_SGLANG:-0}" = "1" ]; then
            echo "  [SKIP] sglang ${model} baseline — SKIP_GEMMA4_SGLANG=1 (disk)"
        else
            pending=$(phase4_pending_scenarios "$slug" "sglang" "${PHASE4_BASELINE_SCENARIOS[@]}")
            if [ -z "$pending" ]; then
                echo "  [SKIP] sglang ${model} baseline — all 5 scenarios present"
            else
                echo "  [RUN]  sglang ${model} baseline — ${pending}"
                MAX_MODEL_LEN=4096 run_sglang_gemma4 "$model" "$pending" 1 10 "results" "sglang"
            fi
        fi

        # ── Ngram vLLM ──
        pending=$(phase4_pending_scenarios "$slug" "vllm-ngram" "${PHASE4_NGRAM_SCENARIOS[@]}")
        if [ -z "$pending" ]; then
            echo "  [SKIP] vllm-ngram ${model} — all 2 scenarios present"
        else
            echo "  [RUN]  vllm-ngram ${model} — ${pending}"
            MAX_MODEL_LEN=4096 run_vllm_gemma4 "$model" "$pending" 1 10 "results" "vllm-ngram" \
                --enforce-eager \
                --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}'
        fi

        # ── Ngram SGLang ──
        if [ "${SKIP_GEMMA4_SGLANG:-0}" = "1" ]; then
            echo "  [SKIP] sglang-ngram ${model} — SKIP_GEMMA4_SGLANG=1 (disk)"
        else
            pending=$(phase4_pending_scenarios "$slug" "sglang-ngram" "${PHASE4_NGRAM_SCENARIOS[@]}")
            if [ -z "$pending" ]; then
                echo "  [SKIP] sglang-ngram ${model} — all 2 scenarios present"
            else
                echo "  [RUN]  sglang-ngram ${model} — ${pending}"
                MAX_MODEL_LEN=4096 run_sglang_gemma4 "$model" "$pending" 1 10 "results" "sglang-ngram" \
                    --speculative-algorithm NGRAM --speculative-num-draft-tokens 16
            fi
        fi

        ((COMPLETED++))
    done

    log "PHASE 4 COMPLETE"
fi

# =============================================================================
#  SUMMARY
# =============================================================================
log "ALL PHASES COMPLETE"

echo ""
echo "Results:"
for dir in results results_variance results_concurrency64 results_decode_sweep; do
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
echo "  Resume Phase 2 (missing-only, idempotent, 1 iter/cell):"
echo "    nohup bash scripts/run_new_benchmarks.sh --phase2 2>&1 | tee logs/phase2_resume_\$(date +%Y%m%dT%H%M%S).log &"
echo ""
echo "  Then run analysis:"
echo "    python -m analysis.variance_analysis --results-dir results_variance"
echo "    python -m analysis.tpot_analysis --results-dir results_variance"
echo "    python -m analysis.decode_length_analysis --results-dir results_decode_sweep"
