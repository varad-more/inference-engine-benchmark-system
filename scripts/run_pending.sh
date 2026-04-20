#!/usr/bin/env bash
# =============================================================================
# Run Pending Benchmarks
#
# Audits results/ for missing (model × engine × scenario) combinations and
# runs only those. Pulls Docker images and pre-fetches HF model weights if
# not already present locally.
#
# Audit logic — a result is "present" when a JSON file matching
#   results/<model-slug>/<scenario>_<engine>_*.json
# exists. Older runs used capitalised client names (VLLMClient/SGLangClient);
# those are accepted as equivalent for the baseline vllm/sglang engines.
#
# Expected matrix:
#   Baseline    : 14 models × 5 scenarios × {vllm, sglang}                = 140
#   Spec-dec    : Llama 3.1 8B × 2 scenarios × {vllm,sglang}-{ngram,eagle3} = 8
#                 Qwen3 8B    × 2 scenarios × {vllm,sglang}-ngram          = 4
#                 (Qwen3 8B Eagle3 blocked — no draft model published)
#
# Usage:
#   chmod +x scripts/run_pending.sh
#   ./scripts/run_pending.sh                 # audit + run missing
#   ./scripts/run_pending.sh --audit-only    # print missing matrix and exit
#   ./scripts/run_pending.sh --no-prefetch   # skip HF snapshot prefetch
# =============================================================================

set +e

PYTHON="${PYTHON:-conda run --no-capture-output -n base python}"
RESULTS_DIR="results"
COOLDOWN=10
ERRORS=()
COMPLETED=0
SKIPPED=0
AUDIT_ONLY=false
PREFETCH=true

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.18.0-cu130}"
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:nightly-dev-cu13-20260321-94194537}"

BASELINE_SCENARIOS=(single_request_latency throughput_ramp long_context_stress prefix_sharing_benefit structured_generation_speed)
SPECDEC_SCENARIOS=(single_request_latency throughput_ramp)

# ── Parse args ───────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --audit-only)  AUDIT_ONLY=true ;;
        --no-prefetch) PREFETCH=false ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
    esac
done

log()   { echo ""; echo "=========================================================================="; echo "[$(date '+%F %T')] $*"; echo "=========================================================================="; }
info()  { echo "[$(date '+%F %T')] $*"; }
error() { echo "[$(date '+%F %T')] ERROR: $*" >&2; ERRORS+=("$*"); }

# ── Load HF token ────────────────────────────────────────────────────────────
HF_TOKEN=""
if [ -f .env ]; then
    HF_TOKEN=$(grep '^HUGGING_FACE_HUB_TOKEN=' .env | cut -d= -f2-)
fi

# ── Model slug (matches existing layout) ─────────────────────────────────────
model_slug() {
    echo "$1" | awk -F/ '{print tolower($NF)}' | tr '.' '-'
}

# ── Result presence check ────────────────────────────────────────────────────
# args: model_slug scenario engine_label
has_result() {
    local slug="$1" scenario="$2" engine="$3"
    local dir="${RESULTS_DIR}/${slug}"
    [ -d "$dir" ] || return 1
    # Engine label match, plus legacy class-name match for baseline engines.
    local pat
    case "$engine" in
        vllm)   pat="(_${engine}_|_VLLMClient_)" ;;
        sglang) pat="(_${engine}_|_SGLangClient_)" ;;
        *)      pat="_${engine}_" ;;
    esac
    find "$dir" -maxdepth 1 -name "${scenario}_*.json" 2>/dev/null \
        | grep -Eq "$pat"
}

# =============================================================================
#  AUDIT — build PENDING list
# =============================================================================
declare -a PENDING_BASELINE   # "model|scenario|engine"
declare -a PENDING_SPECDEC    # "model|scenario|engine"

BASELINE_MODELS=(
    "google/gemma-2-2b-it"
    "HuggingFaceTB/SmolLM3-3B"
    "meta-llama/Llama-3.2-3B-Instruct"
    "microsoft/Phi-3-mini-4k-instruct"
    "google/gemma-3-4b-it"
    "microsoft/Phi-4-mini-instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "Qwen/Qwen2.5-7B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-8B"
    "ibm-granite/granite-3.3-8b-instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "google/gemma-2-9b-it"
)

# Spec-dec coverage: model | scenario | engine
SPECDEC_MATRIX=(
    "meta-llama/Llama-3.1-8B-Instruct|vllm-ngram"
    "meta-llama/Llama-3.1-8B-Instruct|sglang-ngram"
    "meta-llama/Llama-3.1-8B-Instruct|vllm-eagle3"
    "meta-llama/Llama-3.1-8B-Instruct|sglang-eagle3"
    "Qwen/Qwen3-8B|vllm-ngram"
    "Qwen/Qwen3-8B|sglang-ngram"
    # Eagle3 on Qwen3-8B intentionally absent — RedHatAI draft not published
)

log "AUDIT — checking results/ for missing runs"

for model in "${BASELINE_MODELS[@]}"; do
    slug=$(model_slug "$model")
    for scenario in "${BASELINE_SCENARIOS[@]}"; do
        for engine in vllm sglang; do
            if ! has_result "$slug" "$scenario" "$engine"; then
                PENDING_BASELINE+=("${model}|${scenario}|${engine}")
            fi
        done
    done
done

for entry in "${SPECDEC_MATRIX[@]}"; do
    model="${entry%|*}"
    engine="${entry##*|}"
    slug=$(model_slug "$model")
    for scenario in "${SPECDEC_SCENARIOS[@]}"; do
        if ! has_result "$slug" "$scenario" "$engine"; then
            PENDING_SPECDEC+=("${model}|${scenario}|${engine}")
        fi
    done
done

echo ""
echo "Pending baseline runs : ${#PENDING_BASELINE[@]}"
for p in "${PENDING_BASELINE[@]}"; do echo "  - $p"; done
echo ""
echo "Pending spec-dec runs : ${#PENDING_SPECDEC[@]}"
for p in "${PENDING_SPECDEC[@]}"; do echo "  - $p"; done
echo ""

if [ "$AUDIT_ONLY" = true ]; then
    log "AUDIT-ONLY MODE — exiting"
    exit 0
fi

if [ ${#PENDING_BASELINE[@]} -eq 0 ] && [ ${#PENDING_SPECDEC[@]} -eq 0 ]; then
    log "Nothing to do — all expected runs are present"
    exit 0
fi

# =============================================================================
#  PREFLIGHT — Docker, GPU, HF token, model-cache
# =============================================================================
log "PREFLIGHT"

echo -n "  Python   : "; $PYTHON --version 2>&1 | head -1 \
    || { error "conda python not found"; exit 1; }

echo -n "  Docker   : "
if ! docker version --format '{{.Server.Version}}' 2>/dev/null; then
    error "Docker daemon not reachable"; exit 1
fi

echo -n "  GPU      : "
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 \
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "(GPU check failed — continuing)"

echo -n "  HF token : "; [ -n "$HF_TOKEN" ] && echo "found" || echo "MISSING (gated models will fail)"
echo -n "  cache    : "; [ -d model-cache ] && echo "exists" || { mkdir -p model-cache && echo "created"; }

# Always start clean
docker rm -f vllm-server sglang-server vllm-eagle3-server sglang-eagle3-server \
    vllm-ngram-server sglang-ngram-server 2>/dev/null
mkdir -p logs

# ── Docker image pull (only if missing) ──────────────────────────────────────
ensure_image() {
    local img="$1"
    if docker image inspect "$img" >/dev/null 2>&1; then
        info "  image present: $img"
    else
        info "  pulling image: $img"
        docker pull "$img" || { error "docker pull failed for $img"; return 1; }
    fi
}

log "DOCKER IMAGES"
NEED_VLLM=false; NEED_SGLANG=false
for entry in "${PENDING_BASELINE[@]}" "${PENDING_SPECDEC[@]}"; do
    case "${entry##*|}" in
        vllm|vllm-*)   NEED_VLLM=true ;;
        sglang|sglang-*) NEED_SGLANG=true ;;
    esac
done
$NEED_VLLM   && ensure_image "$VLLM_IMAGE"
$NEED_SGLANG && ensure_image "$SGLANG_IMAGE"

# ── HF model prefetch via a throwaway container that uses model-cache ────────
# Avoids paying the download time twice (once for vLLM, once for SGLang) and
# lets us fail fast on auth issues before spinning up GPU servers.
prefetch_model() {
    local repo="$1"
    info "  prefetch: $repo"
    docker run --rm \
        -v "$(pwd)/model-cache:/root/.cache/huggingface" \
        -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
        --entrypoint python \
        "$VLLM_IMAGE" -c "
from huggingface_hub import snapshot_download
snapshot_download('${repo}', allow_patterns=['*.json','*.safetensors','*.txt','*.model','tokenizer*'])
" 2>&1 | tail -5
}

if $PREFETCH; then
    log "HF MODEL PREFETCH"
    declare -A SEEN
    for entry in "${PENDING_BASELINE[@]}" "${PENDING_SPECDEC[@]}"; do
        m="${entry%%|*}"
        [ -n "${SEEN[$m]}" ] && continue
        SEEN[$m]=1
        prefetch_model "$m" || error "prefetch failed for $m (will retry on engine startup)"
    done
    # Eagle3 draft for Llama 3.1 8B (only spec-dec model that needs it here)
    for entry in "${PENDING_SPECDEC[@]}"; do
        case "$entry" in
            "meta-llama/Llama-3.1-8B-Instruct|"*"|vllm-eagle3")
                prefetch_model "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3" || true ;;
            "meta-llama/Llama-3.1-8B-Instruct|"*"|sglang-eagle3")
                prefetch_model "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B" || true ;;
        esac
    done
fi

# =============================================================================
#  ENGINE LAUNCH HELPERS
# =============================================================================
cleanup_container() {
    docker rm -f "$1" 2>/dev/null
    sleep 2
}

# Wait for /health to return 200, up to N seconds. Returns 0 on ready.
wait_health() {
    local port="$1" deadline="$(( $(date +%s) + ${2:-300} ))"
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    return 1
}

# vLLM launcher. Args: model container port engine_label scenario [extra_args...]
run_vllm() {
    local model="$1" container="$2" port="$3" engine="$4" scenario="$5"; shift 5
    local extra=("$@")
    local mml="${MAX_MODEL_LEN:-8192}" gmu="${GPU_MEM_UTIL:-0.85}"

    info "  launch vLLM ($engine) for $model"
    cleanup_container "$container"
    docker run -d --name "$container" --gpus '"device=0"' \
        -p "${port}:${port}" --shm-size 10g \
        -v "$(pwd)/model-cache:/root/.cache/huggingface" \
        -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" -e "CUDA_VISIBLE_DEVICES=0" \
        "$VLLM_IMAGE" \
        --model "$model" --host 0.0.0.0 --port "$port" \
        --enable-prefix-caching --max-model-len "$mml" \
        --gpu-memory-utilization "$gmu" --served-model-name "$model" \
        "${extra[@]}" >/dev/null \
        || { error "$engine: docker run failed for $model"; return 1; }

    if ! wait_health "$port" 420; then
        error "$engine: /health never came up for $model"
        docker logs "$container" --tail 80 2>&1
        docker rm -f "$container" 2>/dev/null
        return 1
    fi

    $PYTHON run_experiment.py matrix \
        --scenarios "$scenario" --engines "$engine" \
        -m "$model" --cooldown-seconds "$COOLDOWN" --output-dir "$RESULTS_DIR"
    local rc=$?
    [ $rc -ne 0 ] && { error "$engine: matrix run failed for $model/$scenario (rc=$rc)"; docker logs "$container" --tail 30 2>&1; }

    docker rm -f "$container" 2>/dev/null
    sleep 10
    return $rc
}

# SGLang launcher. Args: model container port engine_label scenario [extra_args...]
run_sglang() {
    local model="$1" container="$2" port="$3" engine="$4" scenario="$5"; shift 5
    local extra=("$@")
    local mml="${MAX_MODEL_LEN:-8192}" mfs="${MEM_FRAC_STATIC:-0.85}"

    info "  launch SGLang ($engine) for $model"
    cleanup_container "$container"
    docker run -d --name "$container" --gpus '"device=0"' \
        -p "${port}:${port}" --shm-size 10g \
        -v "$(pwd)/model-cache:/root/.cache/huggingface" \
        -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" -e "CUDA_VISIBLE_DEVICES=0" \
        -e "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1" \
        "$SGLANG_IMAGE" \
        python -m sglang.launch_server \
        --model-path "$model" --host 0.0.0.0 --port "$port" \
        --mem-fraction-static "$mfs" --context-length "$mml" \
        "${extra[@]}" >/dev/null \
        || { error "$engine: docker run failed for $model"; return 1; }

    if ! wait_health "$port" 420; then
        error "$engine: /health never came up for $model"
        docker logs "$container" --tail 80 2>&1
        docker rm -f "$container" 2>/dev/null
        return 1
    fi

    $PYTHON run_experiment.py matrix \
        --scenarios "$scenario" --engines "$engine" \
        -m "$model" --cooldown-seconds "$COOLDOWN" --output-dir "$RESULTS_DIR"
    local rc=$?
    [ $rc -ne 0 ] && { error "$engine: matrix run failed for $model/$scenario (rc=$rc)"; docker logs "$container" --tail 30 2>&1; }

    docker rm -f "$container" 2>/dev/null
    sleep 10
    return $rc
}

# Dispatch one pending entry. Sets per-engine flags (Gemma 3, Eagle3, Ngram).
dispatch() {
    local model="$1" scenario="$2" engine="$3"

    # Per-model overrides
    local mml=8192 gmu=0.85 mfs=0.85
    local -a extra=()

    if [ "$model" = "google/gemma-3-4b-it" ]; then
        mml=4096
        if [ "$engine" = "vllm" ]; then
            extra=(--enforce-eager --disable-frontend-multiprocessing)
        fi
    fi

    case "$engine" in
        vllm)
            MAX_MODEL_LEN=$mml GPU_MEM_UTIL=$gmu \
                run_vllm "$model" vllm-server 8000 vllm "$scenario" "${extra[@]}"
            ;;
        sglang)
            MAX_MODEL_LEN=$mml MEM_FRAC_STATIC=$mfs \
                run_sglang "$model" sglang-server 8001 sglang "$scenario"
            ;;
        vllm-ngram)
            MAX_MODEL_LEN=$mml GPU_MEM_UTIL=$gmu \
                run_vllm "$model" vllm-ngram-server 8000 vllm-ngram "$scenario" \
                --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}'
            ;;
        sglang-ngram)
            MAX_MODEL_LEN=$mml MEM_FRAC_STATIC=$mfs \
                run_sglang "$model" sglang-ngram-server 8001 sglang-ngram "$scenario" \
                --speculative-algorithm NGRAM --speculative-num-draft-tokens 16
            ;;
        vllm-eagle3)
            MAX_MODEL_LEN=2048 GPU_MEM_UTIL=0.95 \
                run_vllm "$model" vllm-eagle3-server 8000 vllm-eagle3 "$scenario" \
                --enforce-eager \
                --speculative-config "{\"method\":\"eagle3\",\"model\":\"RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3\",\"num_speculative_tokens\":3}"
            ;;
        sglang-eagle3)
            # Tight on A10G — reduce mem-fraction to leave room for draft
            MAX_MODEL_LEN=2048 MEM_FRAC_STATIC=0.65 \
                run_sglang "$model" sglang-eagle3-server 8001 sglang-eagle3 "$scenario" \
                --speculative-algorithm EAGLE3 \
                --speculative-draft-model-path jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B \
                --speculative-num-steps 3 \
                --speculative-eagle-topk 4 \
                --speculative-num-draft-tokens 16
            ;;
        *) error "unknown engine: $engine"; return 1 ;;
    esac
}

# =============================================================================
#  RUN
# =============================================================================
log "RUNNING ${#PENDING_BASELINE[@]} BASELINE + ${#PENDING_SPECDEC[@]} SPEC-DEC"

for entry in "${PENDING_BASELINE[@]}" "${PENDING_SPECDEC[@]}"; do
    IFS='|' read -r model scenario engine <<<"$entry"
    log "RUN: $model | $scenario | $engine"
    dispatch "$model" "$scenario" "$engine" \
        && COMPLETED=$((COMPLETED+1)) \
        || SKIPPED=$((SKIPPED+1))
done

# =============================================================================
#  SUMMARY
# =============================================================================
log "SUMMARY"
echo "  completed runs : $COMPLETED"
echo "  failed runs    : $SKIPPED"
echo "  errors         : ${#ERRORS[@]}"
if [ ${#ERRORS[@]} -gt 0 ]; then
    echo ""
    echo "Errors:"
    for i in "${!ERRORS[@]}"; do echo "  $((i+1)). ${ERRORS[$i]}"; done
fi

echo ""
echo "Re-run with --audit-only to see remaining gaps."
