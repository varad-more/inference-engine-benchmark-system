#!/usr/bin/env bash
# =============================================================================
# Run ALL Pending Benchmarks
#
# One script to find and run every gap across the entire benchmark suite.
# Audits each results directory for missing (model × engine × scenario ×
# iteration) cells and runs only what's absent.  Pulls Docker images and
# pre-fetches HF model weights if not already present locally.
#
# ── Scope ────────────────────────────────────────────────────────────────────
# PART 0 — Baseline + Spec-dec  (results/)
#   14 models × 5 scenarios × {vllm, sglang}                          = 140
#   Llama 3.1 8B × 2 scenarios × {vllm,sglang}-{ngram,eagle3}         =   8
#   Qwen3 8B     × 2 scenarios × {vllm,sglang}-ngram                  =   4
#   (Eagle3 on Qwen3 8B blocked — no draft model published)
#
# PART 1 — Variance  (results_variance/)
#   4 models × 5 scenarios × 2 engines × 5 iterations                 = 200
#   Missing: gemma-3-4b-it entirely (0/50), Llama SGLang partial (6/25)
#
# PART 2 — Concurrency-64  (results_concurrency64/)  — ALL COMPLETE (8/8)
#
# PART 3 — Decode-length sweep  (results_decode_sweep/)
#   4 models × 4 lengths × 2 engines × 3 iterations                   =  96
#   Missing: llama ×1 iter/cell (+8), gemma-3-4b ×2 iter/cell (+16)
#
# PART 4 — Gemma 4  (results/)  — not yet run
#
# Usage:
#   chmod +x scripts/run_pending.sh
#   ./scripts/run_pending.sh                 # audit + run all missing
#   ./scripts/run_pending.sh --audit-only    # print missing matrix and exit
#   ./scripts/run_pending.sh --no-prefetch   # skip HF snapshot prefetch
#   ./scripts/run_pending.sh --part 0        # run only Part 0 (baseline)
#   ./scripts/run_pending.sh --part 1        # run only Part 1 (variance)
#   ./scripts/run_pending.sh --part 3        # run only Part 3 (decode sweep)
#   ./scripts/run_pending.sh --part 4        # run only Part 4 (Gemma 4)
# =============================================================================

set +e

PYTHON="${PYTHON:-conda run --no-capture-output -n base python}"
COOLDOWN=10
ERRORS=()
COMPLETED=0
SKIPPED=0
AUDIT_ONLY=false
PREFETCH=true
PART_FILTER=""  # empty = all parts

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.18.0-cu130}"
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:nightly-dev-cu13-20260321-94194537}"

# Gemma 4 needs newer images with Transformers >= v5.5.0
GEMMA4_VLLM_IMAGE="${GEMMA4_VLLM_IMAGE:-vllm/vllm-openai:latest}"
GEMMA4_SGLANG_IMAGE="${GEMMA4_SGLANG_IMAGE:-lmsysorg/sglang:latest}"

BASELINE_SCENARIOS=(single_request_latency throughput_ramp long_context_stress prefix_sharing_benefit structured_generation_speed)
SPECDEC_SCENARIOS=(single_request_latency throughput_ramp)
DECODE_LENGTHS=(64 256 1024 4096)

# ── Parse args ───────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --audit-only)  AUDIT_ONLY=true ;;
        --no-prefetch) PREFETCH=false ;;
        --part)        ;; # value captured below
        0|1|2|3|4)
            # Capture --part N (previous arg was --part)
            PART_FILTER="$arg" ;;
        -h|--help)
            sed -n '2,38p' "$0"; exit 0 ;;
    esac
done
# Also handle --part=N syntax
for arg in "$@"; do
    case "$arg" in
        --part=*) PART_FILTER="${arg#--part=}" ;;
    esac
done

should_run_part() {
    [ -z "$PART_FILTER" ] || [ "$PART_FILTER" = "$1" ]
}

log()   { echo ""; echo "=========================================================================="; echo "[$(date '+%F %T')] $*"; echo "=========================================================================="; }
info()  { echo "[$(date '+%F %T')] $*"; }
error() { echo "[$(date '+%F %T')] ERROR: $*" >&2; ERRORS+=("$*"); }

# ── Load HF token ────────────────────────────────────────────────────────────
HF_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
if [ -z "$HF_TOKEN" ] && [ -f .env ]; then
    HF_TOKEN=$(grep '^HUGGING_FACE_HUB_TOKEN=' .env | cut -d= -f2-)
fi

# ── Model slug (matches existing results directory layout) ───────────────────
model_slug() {
    echo "$1" | awk -F/ '{print tolower($NF)}' | tr '.' '-'
}

# ── Result presence check ────────────────────────────────────────────────────
# args: results_dir model_slug scenario engine
has_result() {
    local dir="$1/$2" scenario="$3" engine="$4"
    [ -d "$dir" ] || return 1
    # Handle legacy class-name results (VLLMClient / SGLangClient)
    local pat
    case "$engine" in
        vllm)   pat="(_${engine}_|_VLLMClient_)" ;;
        sglang) pat="(_${engine}_|_SGLangClient_)" ;;
        *)      pat="_${engine}_" ;;
    esac
    find "$dir" -maxdepth 1 -name "${scenario}_*.json" 2>/dev/null \
        | grep -Eq "$pat"
}

# Count result files matching: results_dir/slug/scenario_engine_*.json
count_result_files() {
    local dir="$1/$2" scenario="$3" engine="$4"
    [ -d "$dir" ] || { echo 0; return; }
    find "$dir" -maxdepth 1 -name "${scenario}_${engine}_*.json" 2>/dev/null | wc -l | tr -d ' '
}

# =============================================================================
#  AUDIT — build PENDING lists for all parts
# =============================================================================
log "AUDIT — checking all results directories for missing runs"

# ── Part 0: Baseline + Spec-dec (results/) ───────────────────────────────────
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

SPECDEC_MATRIX=(
    "meta-llama/Llama-3.1-8B-Instruct|vllm-ngram"
    "meta-llama/Llama-3.1-8B-Instruct|sglang-ngram"
    "meta-llama/Llama-3.1-8B-Instruct|vllm-eagle3"
    "meta-llama/Llama-3.1-8B-Instruct|sglang-eagle3"
    "Qwen/Qwen3-8B|vllm-ngram"
    "Qwen/Qwen3-8B|sglang-ngram"
)

declare -a PENDING_P0=()  # "model|scenario|engine"

for model in "${BASELINE_MODELS[@]}"; do
    slug=$(model_slug "$model")
    for scenario in "${BASELINE_SCENARIOS[@]}"; do
        for engine in vllm sglang; do
            has_result "results" "$slug" "$scenario" "$engine" || \
                PENDING_P0+=("${model}|${scenario}|${engine}")
        done
    done
done

for entry in "${SPECDEC_MATRIX[@]}"; do
    model="${entry%|*}"; engine="${entry##*|}"
    slug=$(model_slug "$model")
    for scenario in "${SPECDEC_SCENARIOS[@]}"; do
        has_result "results" "$slug" "$scenario" "$engine" || \
            PENDING_P0+=("${model}|${scenario}|${engine}")
    done
done

echo ""
echo "Part 0 — Baseline + Spec-dec (results/): ${#PENDING_P0[@]} pending"
for p in "${PENDING_P0[@]}"; do echo "  - $p"; done

# ── Part 1: Variance (results_variance/) ─────────────────────────────────────
VARIANCE_MODELS=(
    "google/gemma-2-2b-it"
    "microsoft/Phi-4-mini-instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-3-4b-it"
)
VARIANCE_TARGET=5  # iterations per (scenario, engine)

declare -a PENDING_P1=()  # "model|scenario|engine|deficit"

for model in "${VARIANCE_MODELS[@]}"; do
    slug=$(model_slug "$model")
    for scenario in "${BASELINE_SCENARIOS[@]}"; do
        for engine in vllm sglang; do
            existing=$(count_result_files "results_variance" "$slug" "$scenario" "$engine")
            deficit=$(( VARIANCE_TARGET - existing ))
            if [ "$deficit" -gt 0 ]; then
                PENDING_P1+=("${model}|${scenario}|${engine}|${deficit}")
            fi
        done
    done
done

echo ""
echo "Part 1 — Variance (results_variance/): ${#PENDING_P1[@]} scenario/engine cells with deficits"
for p in "${PENDING_P1[@]}"; do echo "  - $p"; done

# ── Part 2: Concurrency-64 (results_concurrency64/) ─────────────────────────
declare -a PENDING_P2=()
CONC64_MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-8B"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "google/gemma-2-9b-it"
)
for model in "${CONC64_MODELS[@]}"; do
    slug=$(model_slug "$model")
    for engine in vllm sglang; do
        has_result "results_concurrency64" "$slug" "throughput_ramp_extended" "$engine" || \
            PENDING_P2+=("${model}|throughput_ramp_extended|${engine}")
    done
done

echo ""
echo "Part 2 — Concurrency-64 (results_concurrency64/): ${#PENDING_P2[@]} pending"
for p in "${PENDING_P2[@]}"; do echo "  - $p"; done

# ── Part 3: Decode sweep (results_decode_sweep/) ────────────────────────────
DECODE_MODELS=(
    "google/gemma-2-2b-it"
    "microsoft/Phi-4-mini-instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-3-4b-it"
)
DECODE_TARGET=3  # iterations per (length, engine)

declare -a PENDING_P3=()  # "model|length|engine|deficit"

for model in "${DECODE_MODELS[@]}"; do
    slug=$(model_slug "$model")
    for length in "${DECODE_LENGTHS[@]}"; do
        scenario="decode_length_sweep_${length}"
        for engine in vllm sglang; do
            existing=$(count_result_files "results_decode_sweep" "$slug" "$scenario" "$engine")
            deficit=$(( DECODE_TARGET - existing ))
            if [ "$deficit" -gt 0 ]; then
                PENDING_P3+=("${model}|${scenario}|${engine}|${deficit}")
            fi
        done
    done
done

echo ""
echo "Part 3 — Decode sweep (results_decode_sweep/): ${#PENDING_P3[@]} cells with deficits"
for p in "${PENDING_P3[@]}"; do echo "  - $p"; done

# ── Part 4: Gemma 4 (results/) ──────────────────────────────────────────────
GEMMA4_MODELS=(
    "google/gemma-4-E2B-it"
    "google/gemma-4-E4B-it"
)

declare -a PENDING_P4_BASE=()
declare -a PENDING_P4_NGRAM=()

for model in "${GEMMA4_MODELS[@]}"; do
    slug=$(model_slug "$model")
    for scenario in "${BASELINE_SCENARIOS[@]}"; do
        for engine in vllm sglang; do
            has_result "results" "$slug" "$scenario" "$engine" || \
                PENDING_P4_BASE+=("${model}|${scenario}|${engine}")
        done
    done
    for scenario in "${SPECDEC_SCENARIOS[@]}"; do
        for engine in vllm-ngram sglang-ngram; do
            has_result "results" "$slug" "$scenario" "$engine" || \
                PENDING_P4_NGRAM+=("${model}|${scenario}|${engine}")
        done
    done
done

echo ""
echo "Part 4 — Gemma 4 baseline (results/): ${#PENDING_P4_BASE[@]} pending"
for p in "${PENDING_P4_BASE[@]}"; do echo "  - $p"; done
echo "Part 4 — Gemma 4 ngram (results/): ${#PENDING_P4_NGRAM[@]} pending"
for p in "${PENDING_P4_NGRAM[@]}"; do echo "  - $p"; done

# ── Grand total ──────────────────────────────────────────────────────────────
TOTAL_PENDING=$(( ${#PENDING_P0[@]} + ${#PENDING_P1[@]} + ${#PENDING_P2[@]} + ${#PENDING_P3[@]} + ${#PENDING_P4_BASE[@]} + ${#PENDING_P4_NGRAM[@]} ))
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  TOTAL PENDING CELLS: $TOTAL_PENDING"
echo "═══════════════════════════════════════════════════════════════"

if [ "$AUDIT_ONLY" = true ]; then
    log "AUDIT-ONLY MODE — exiting"
    exit 0
fi

if [ "$TOTAL_PENDING" -eq 0 ]; then
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

# Clean up stale containers
docker rm -f vllm-server sglang-server vllm-eagle3-server sglang-eagle3-server \
    vllm-ngram-server sglang-ngram-server bench-vllm bench-sglang 2>/dev/null
mkdir -p logs results results_variance results_concurrency64 results_decode_sweep

# ── Docker image pull ────────────────────────────────────────────────────────
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
# Determine which images we need
NEED_VLLM=false; NEED_SGLANG=false; NEED_GEMMA4=false
if should_run_part 0 && [ ${#PENDING_P0[@]} -gt 0 ]; then NEED_VLLM=true; NEED_SGLANG=true; fi
if should_run_part 1 && [ ${#PENDING_P1[@]} -gt 0 ]; then NEED_VLLM=true; NEED_SGLANG=true; fi
if should_run_part 2 && [ ${#PENDING_P2[@]} -gt 0 ]; then NEED_VLLM=true; NEED_SGLANG=true; fi
if should_run_part 3 && [ ${#PENDING_P3[@]} -gt 0 ]; then NEED_VLLM=true; NEED_SGLANG=true; fi
if should_run_part 4 && ([ ${#PENDING_P4_BASE[@]} -gt 0 ] || [ ${#PENDING_P4_NGRAM[@]} -gt 0 ]); then NEED_GEMMA4=true; fi

$NEED_VLLM   && ensure_image "$VLLM_IMAGE"
$NEED_SGLANG && ensure_image "$SGLANG_IMAGE"
$NEED_GEMMA4 && { ensure_image "$GEMMA4_VLLM_IMAGE"; ensure_image "$GEMMA4_SGLANG_IMAGE"; }

# ── HF model prefetch ───────────────────────────────────────────────────────
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
    # Gather all models across all pending lists
    for entry in "${PENDING_P0[@]}" "${PENDING_P1[@]}" "${PENDING_P2[@]}" "${PENDING_P3[@]}"; do
        m="${entry%%|*}"
        [ -n "${SEEN[$m]:-}" ] && continue
        SEEN[$m]=1
        prefetch_model "$m" || error "prefetch failed for $m (will retry on engine startup)"
    done
    # Eagle3 draft models
    for entry in "${PENDING_P0[@]}"; do
        case "$entry" in
            *"|vllm-eagle3") prefetch_model "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3" || true ;;
            *"|sglang-eagle3") prefetch_model "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B" || true ;;
        esac
    done
    # Gemma 4 models (use Gemma4 image for prefetch since older image may lack arch support)
    if should_run_part 4 && ([ ${#PENDING_P4_BASE[@]} -gt 0 ] || [ ${#PENDING_P4_NGRAM[@]} -gt 0 ]); then
        for model in "${GEMMA4_MODELS[@]}"; do
            [ -n "${SEEN[$model]:-}" ] && continue
            SEEN[$model]=1
            info "  prefetch (gemma4): $model"
            docker run --rm \
                -v "$(pwd)/model-cache:/root/.cache/huggingface" \
                -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
                --entrypoint python \
                "$GEMMA4_VLLM_IMAGE" -c "
from huggingface_hub import snapshot_download
snapshot_download('${model}', allow_patterns=['*.json','*.safetensors','*.txt','*.model','tokenizer*'])
" 2>&1 | tail -5 || error "prefetch failed for $model"
        done
    fi
fi

# =============================================================================
#  ENGINE LAUNCH HELPERS
# =============================================================================
cleanup_container() {
    docker rm -f "$1" 2>/dev/null
    sleep 2
}

# Wait for /health to return 200, up to N seconds.
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

# ── vLLM launcher ───────────────────────────────────────────────────────────
# Args: model container port engine_label scenarios cooldown iterations output_dir image [extra_args...]
run_vllm() {
    local model="$1" container="$2" port="$3" engine="$4" scenarios="$5"
    local cooldown="$6" iterations="$7" output_dir="$8" image="$9"; shift 9
    local extra=("$@")
    local mml="${MAX_MODEL_LEN:-8192}" gmu="${GPU_MEM_UTIL:-0.85}"

    info "  launch vLLM ($engine) for $model [iter=$iterations, cooldown=$cooldown]"
    cleanup_container "$container"
    docker run -d --name "$container" --gpus '"device=0"' \
        -p "${port}:${port}" --shm-size 10g \
        -v "$(pwd)/model-cache:/root/.cache/huggingface" \
        -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" -e "CUDA_VISIBLE_DEVICES=0" \
        "$image" \
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
        --scenarios "$scenarios" --engines "$engine" \
        -m "$model" --cooldown-seconds "$cooldown" \
        --iterations "$iterations" --output-dir "$output_dir"
    local rc=$?
    [ $rc -ne 0 ] && { error "$engine: matrix run failed for $model (rc=$rc)"; docker logs "$container" --tail 30 2>&1; }

    docker rm -f "$container" 2>/dev/null
    sleep 10
    return $rc
}

# ── SGLang launcher ──────────────────────────────────────────────────────────
# Args: model container port engine_label scenarios cooldown iterations output_dir image [extra_args...]
run_sglang() {
    local model="$1" container="$2" port="$3" engine="$4" scenarios="$5"
    local cooldown="$6" iterations="$7" output_dir="$8" image="$9"; shift 9
    local extra=("$@")
    local mml="${MAX_MODEL_LEN:-8192}" mfs="${MEM_FRAC_STATIC:-0.85}"

    info "  launch SGLang ($engine) for $model [iter=$iterations, cooldown=$cooldown]"
    cleanup_container "$container"
    docker run -d --name "$container" --gpus '"device=0"' \
        -p "${port}:${port}" --shm-size 10g \
        -v "$(pwd)/model-cache:/root/.cache/huggingface" \
        -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" -e "CUDA_VISIBLE_DEVICES=0" \
        -e "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1" \
        "$image" \
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
        --scenarios "$scenarios" --engines "$engine" \
        -m "$model" --cooldown-seconds "$cooldown" \
        --iterations "$iterations" --output-dir "$output_dir"
    local rc=$?
    [ $rc -ne 0 ] && { error "$engine: matrix run failed for $model (rc=$rc)"; docker logs "$container" --tail 30 2>&1; }

    docker rm -f "$container" 2>/dev/null
    sleep 10
    return $rc
}

# ── Gemma 4 vLLM launcher (needs pip install transformers from git) ──────────
run_vllm_gemma4() {
    local model="$1" container="$2" port="$3" engine="$4" scenarios="$5"
    local cooldown="$6" iterations="$7" output_dir="$8"; shift 8
    local extra=("$@")
    local mml="${MAX_MODEL_LEN:-4096}" gmu="${GPU_MEM_UTIL:-0.85}"

    info "  launch vLLM-Gemma4 ($engine) for $model [iter=$iterations]"
    cleanup_container "$container"

    local server_args=(
        --model "$model" --host 0.0.0.0 --port "$port"
        --enable-prefix-caching --max-model-len "$mml"
        --gpu-memory-utilization "$gmu" --served-model-name "$model"
        "${extra[@]}"
    )
    local quoted_args
    quoted_args=$(printf '%q ' "${server_args[@]}")
    local cmd="pip install -q --upgrade git+https://github.com/huggingface/transformers.git \
        && exec python -m vllm.entrypoints.openai.api_server ${quoted_args}"

    docker run -d --name "$container" --gpus '"device=0"' \
        -p "${port}:${port}" --shm-size 10g \
        -v "$(pwd)/model-cache:/root/.cache/huggingface" \
        -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" -e "CUDA_VISIBLE_DEVICES=0" \
        --entrypoint bash "$GEMMA4_VLLM_IMAGE" -c "$cmd" >/dev/null \
        || { error "$engine: docker run failed for $model"; return 1; }

    if ! wait_health "$port" 600; then
        error "$engine: /health never came up for $model"
        docker logs "$container" --tail 80 2>&1
        docker rm -f "$container" 2>/dev/null
        return 1
    fi

    $PYTHON run_experiment.py matrix \
        --scenarios "$scenarios" --engines "$engine" \
        -m "$model" --cooldown-seconds "$cooldown" \
        --iterations "$iterations" --output-dir "$output_dir"
    local rc=$?
    [ $rc -ne 0 ] && { error "$engine: matrix run failed for $model (rc=$rc)"; docker logs "$container" --tail 30 2>&1; }

    docker rm -f "$container" 2>/dev/null
    sleep 10
    return $rc
}

# ── Gemma 4 SGLang launcher ─────────────────────────────────────────────────
run_sglang_gemma4() {
    local model="$1" container="$2" port="$3" engine="$4" scenarios="$5"
    local cooldown="$6" iterations="$7" output_dir="$8"; shift 8
    local extra=("$@")
    local mml="${MAX_MODEL_LEN:-4096}" mfs="${MEM_FRAC_STATIC:-0.85}"

    info "  launch SGLang-Gemma4 ($engine) for $model [iter=$iterations]"
    cleanup_container "$container"

    local server_args=(
        --model-path "$model" --host 0.0.0.0 --port "$port"
        --mem-fraction-static "$mfs" --context-length "$mml"
        --disable-cuda-graph
        "${extra[@]}"
    )
    local quoted_args
    quoted_args=$(printf '%q ' "${server_args[@]}")
    local cmd="pip install -q --upgrade git+https://github.com/huggingface/transformers.git \
        && exec python -m sglang.launch_server ${quoted_args}"

    docker run -d --name "$container" --gpus '"device=0"' \
        -p "${port}:${port}" --shm-size 10g \
        -v "$(pwd)/model-cache:/root/.cache/huggingface" \
        -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" -e "CUDA_VISIBLE_DEVICES=0" \
        -e "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1" \
        "$GEMMA4_SGLANG_IMAGE" bash -c "$cmd" >/dev/null \
        || { error "$engine: docker run failed for $model"; return 1; }

    if ! wait_health "$port" 600; then
        error "$engine: /health never came up for $model"
        docker logs "$container" --tail 80 2>&1
        docker rm -f "$container" 2>/dev/null
        return 1
    fi

    $PYTHON run_experiment.py matrix \
        --scenarios "$scenarios" --engines "$engine" \
        -m "$model" --cooldown-seconds "$cooldown" \
        --iterations "$iterations" --output-dir "$output_dir"
    local rc=$?
    [ $rc -ne 0 ] && { error "$engine: matrix run failed for $model (rc=$rc)"; docker logs "$container" --tail 30 2>&1; }

    docker rm -f "$container" 2>/dev/null
    sleep 10
    return $rc
}

# ── Part 0 dispatch (one pending entry at a time) ────────────────────────────
dispatch_p0() {
    local model="$1" scenario="$2" engine="$3"
    local mml=8192 gmu=0.85 mfs=0.85
    local -a extra=()

    # Per-model overrides
    if [ "$model" = "google/gemma-3-4b-it" ]; then
        mml=4096
        if [[ "$engine" == vllm* ]]; then
            extra=(--enforce-eager --disable-frontend-multiprocessing)
        fi
    fi

    case "$engine" in
        vllm)
            MAX_MODEL_LEN=$mml GPU_MEM_UTIL=$gmu \
                run_vllm "$model" vllm-server 8000 vllm "$scenario" "$COOLDOWN" 1 results "$VLLM_IMAGE" "${extra[@]}"
            ;;
        sglang)
            MAX_MODEL_LEN=$mml MEM_FRAC_STATIC=$mfs \
                run_sglang "$model" sglang-server 8001 sglang "$scenario" "$COOLDOWN" 1 results "$SGLANG_IMAGE"
            ;;
        vllm-ngram)
            MAX_MODEL_LEN=$mml GPU_MEM_UTIL=$gmu \
                run_vllm "$model" vllm-ngram-server 8000 vllm-ngram "$scenario" "$COOLDOWN" 1 results "$VLLM_IMAGE" \
                --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}'
            ;;
        sglang-ngram)
            MAX_MODEL_LEN=$mml MEM_FRAC_STATIC=$mfs \
                run_sglang "$model" sglang-ngram-server 8001 sglang-ngram "$scenario" "$COOLDOWN" 1 results "$SGLANG_IMAGE" \
                --speculative-algorithm NGRAM --speculative-num-draft-tokens 16
            ;;
        vllm-eagle3)
            MAX_MODEL_LEN=2048 GPU_MEM_UTIL=0.95 \
                run_vllm "$model" vllm-eagle3-server 8000 vllm-eagle3 "$scenario" "$COOLDOWN" 1 results "$VLLM_IMAGE" \
                --enforce-eager \
                --speculative-config '{"method":"eagle3","model":"RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3","num_speculative_tokens":3}'
            ;;
        sglang-eagle3)
            MAX_MODEL_LEN=2048 MEM_FRAC_STATIC=0.65 \
                run_sglang "$model" sglang-eagle3-server 8001 sglang-eagle3 "$scenario" "$COOLDOWN" 1 results "$SGLANG_IMAGE" \
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
#  PART 0 — Baseline + Spec-dec
# =============================================================================
if should_run_part 0 && [ ${#PENDING_P0[@]} -gt 0 ]; then
    log "PART 0: RUNNING ${#PENDING_P0[@]} BASELINE + SPEC-DEC GAPS"

    for entry in "${PENDING_P0[@]}"; do
        IFS='|' read -r model scenario engine <<<"$entry"
        log "P0 RUN: $model | $scenario | $engine"
        dispatch_p0 "$model" "$scenario" "$engine" \
            && COMPLETED=$((COMPLETED+1)) \
            || SKIPPED=$((SKIPPED+1))
    done
fi

# =============================================================================
#  PART 1 — Variance top-up
#
#  Strategy: group pending cells by (model, engine) so we can launch the
#  container once and run all missing scenarios in a single matrix call.
#  For cells that need < 5 iterations, we pass the deficit as --iterations.
# =============================================================================
if should_run_part 1 && [ ${#PENDING_P1[@]} -gt 0 ]; then
    log "PART 1: VARIANCE TOP-UP — ${#PENDING_P1[@]} cells with deficits"

    # Build unique (model, engine) pairs and their max deficit
    declare -A P1_GROUPS  # key=model|engine, value=scenario1,scenario2,...
    declare -A P1_ITERS   # key=model|engine, value=max deficit across scenarios

    for entry in "${PENDING_P1[@]}"; do
        IFS='|' read -r model scenario engine deficit <<<"$entry"
        key="${model}|${engine}"
        if [ -z "${P1_GROUPS[$key]:-}" ]; then
            P1_GROUPS[$key]="$scenario"
            P1_ITERS[$key]="$deficit"
        else
            P1_GROUPS[$key]="${P1_GROUPS[$key]},${scenario}"
            # Take max deficit (safe: launches extra-iterations for cells that
            # already have some, but they just add more data points)
            [ "$deficit" -gt "${P1_ITERS[$key]}" ] && P1_ITERS[$key]="$deficit"
        fi
    done

    for key in "${!P1_GROUPS[@]}"; do
        IFS='|' read -r model engine <<<"$key"
        scenarios="${P1_GROUPS[$key]}"
        iterations="${P1_ITERS[$key]}"

        log "P1 RUN: $model | $engine | scenarios=$scenarios | iterations=$iterations"

        local_extra=()
        local_mml=8192

        if [ "$model" = "google/gemma-3-4b-it" ]; then
            local_mml=4096
            if [ "$engine" = "vllm" ]; then
                local_extra=(--enforce-eager --disable-frontend-multiprocessing)
            fi
        fi

        if [ "$engine" = "vllm" ]; then
            MAX_MODEL_LEN=$local_mml \
                run_vllm "$model" vllm-server 8000 vllm "$scenarios" "$COOLDOWN" "$iterations" results_variance "$VLLM_IMAGE" "${local_extra[@]}" \
                && COMPLETED=$((COMPLETED+1)) \
                || SKIPPED=$((SKIPPED+1))
        else
            MAX_MODEL_LEN=$local_mml \
                run_sglang "$model" sglang-server 8001 sglang "$scenarios" "$COOLDOWN" "$iterations" results_variance "$SGLANG_IMAGE" \
                && COMPLETED=$((COMPLETED+1)) \
                || SKIPPED=$((SKIPPED+1))
        fi
    done
fi

# =============================================================================
#  PART 2 — Concurrency-64 (expected complete, but handle gaps)
# =============================================================================
if should_run_part 2 && [ ${#PENDING_P2[@]} -gt 0 ]; then
    log "PART 2: CONCURRENCY-64 — ${#PENDING_P2[@]} GAPS"

    for entry in "${PENDING_P2[@]}"; do
        IFS='|' read -r model scenario engine <<<"$entry"
        log "P2 RUN: $model | $scenario | $engine"

        local_extra=()
        local_mml=8192
        if [ "$model" = "google/gemma-2-9b-it" ] && [ "$engine" = "vllm" ]; then
            local_mml=2048
            local_extra=(--enforce-eager --gpu-memory-utilization 0.90)
        fi

        if [ "$engine" = "vllm" ]; then
            MAX_MODEL_LEN=$local_mml \
                run_vllm "$model" vllm-server 8000 vllm "$scenario" "$COOLDOWN" 1 results_concurrency64 "$VLLM_IMAGE" "${local_extra[@]}" \
                && COMPLETED=$((COMPLETED+1)) \
                || SKIPPED=$((SKIPPED+1))
        else
            MAX_MODEL_LEN=$local_mml \
                run_sglang "$model" sglang-server 8001 sglang "$scenario" "$COOLDOWN" 1 results_concurrency64 "$SGLANG_IMAGE" \
                && COMPLETED=$((COMPLETED+1)) \
                || SKIPPED=$((SKIPPED+1))
        fi
    done
fi

# =============================================================================
#  PART 3 — Decode sweep top-up
#
#  Strategy: group by (model, engine) and run all 4 decode-length scenarios
#  together. Use max deficit as iteration count.
# =============================================================================
if should_run_part 3 && [ ${#PENDING_P3[@]} -gt 0 ]; then
    log "PART 3: DECODE SWEEP TOP-UP — ${#PENDING_P3[@]} cells with deficits"

    declare -A P3_GROUPS
    declare -A P3_ITERS

    for entry in "${PENDING_P3[@]}"; do
        IFS='|' read -r model scenario engine deficit <<<"$entry"
        key="${model}|${engine}"
        if [ -z "${P3_GROUPS[$key]:-}" ]; then
            P3_GROUPS[$key]="$scenario"
            P3_ITERS[$key]="$deficit"
        else
            P3_GROUPS[$key]="${P3_GROUPS[$key]},${scenario}"
            [ "$deficit" -gt "${P3_ITERS[$key]}" ] && P3_ITERS[$key]="$deficit"
        fi
    done

    for key in "${!P3_GROUPS[@]}"; do
        IFS='|' read -r model engine <<<"$key"
        scenarios="${P3_GROUPS[$key]}"
        iterations="${P3_ITERS[$key]}"

        log "P3 RUN: $model | $engine | scenarios=$scenarios | iterations=$iterations"

        local_extra=()
        local_mml=8192

        if [ "$model" = "google/gemma-3-4b-it" ]; then
            # Needs >= 4608 for 512-token prompt + 4096-token output
            local_mml=5632
            if [ "$engine" = "vllm" ]; then
                local_extra=(--enforce-eager --disable-frontend-multiprocessing)
            fi
        fi

        if [ "$engine" = "vllm" ]; then
            MAX_MODEL_LEN=$local_mml \
                run_vllm "$model" vllm-server 8000 vllm "$scenarios" "$COOLDOWN" "$iterations" results_decode_sweep "$VLLM_IMAGE" "${local_extra[@]}" \
                && COMPLETED=$((COMPLETED+1)) \
                || SKIPPED=$((SKIPPED+1))
        else
            MAX_MODEL_LEN=$local_mml \
                run_sglang "$model" sglang-server 8001 sglang "$scenarios" "$COOLDOWN" "$iterations" results_decode_sweep "$SGLANG_IMAGE" \
                && COMPLETED=$((COMPLETED+1)) \
                || SKIPPED=$((SKIPPED+1))
        fi
    done
fi

# =============================================================================
#  PART 4 — Gemma 4 (baseline + ngram)
# =============================================================================
if should_run_part 4 && ([ ${#PENDING_P4_BASE[@]} -gt 0 ] || [ ${#PENDING_P4_NGRAM[@]} -gt 0 ]); then
    log "PART 4: GEMMA 4 — ${#PENDING_P4_BASE[@]} baseline + ${#PENDING_P4_NGRAM[@]} ngram"

    for model in "${GEMMA4_MODELS[@]}"; do
        slug=$(model_slug "$model")

        # ── Baseline: vLLM ──
        pending_baseline_vllm=""
        for entry in "${PENDING_P4_BASE[@]}"; do
            IFS='|' read -r m s e <<<"$entry"
            if [ "$m" = "$model" ] && [ "$e" = "vllm" ]; then
                if [ -z "$pending_baseline_vllm" ]; then
                    pending_baseline_vllm="$s"
                else
                    pending_baseline_vllm="${pending_baseline_vllm},${s}"
                fi
            fi
        done
        if [ -n "$pending_baseline_vllm" ]; then
            log "P4 BASELINE VLLM: $model | $pending_baseline_vllm"
            MAX_MODEL_LEN=4096 \
                run_vllm_gemma4 "$model" vllm-server 8000 vllm "$pending_baseline_vllm" "$COOLDOWN" 1 results \
                --enforce-eager --disable-frontend-multiprocessing \
                && COMPLETED=$((COMPLETED+1)) \
                || SKIPPED=$((SKIPPED+1))
        fi

        # ── Baseline: SGLang ──
        pending_baseline_sglang=""
        for entry in "${PENDING_P4_BASE[@]}"; do
            IFS='|' read -r m s e <<<"$entry"
            if [ "$m" = "$model" ] && [ "$e" = "sglang" ]; then
                if [ -z "$pending_baseline_sglang" ]; then
                    pending_baseline_sglang="$s"
                else
                    pending_baseline_sglang="${pending_baseline_sglang},${s}"
                fi
            fi
        done
        if [ -n "$pending_baseline_sglang" ]; then
            log "P4 BASELINE SGLANG: $model | $pending_baseline_sglang"
            MAX_MODEL_LEN=4096 \
                run_sglang_gemma4 "$model" sglang-server 8001 sglang "$pending_baseline_sglang" "$COOLDOWN" 1 results \
                && COMPLETED=$((COMPLETED+1)) \
                || SKIPPED=$((SKIPPED+1))
        fi

        # ── Ngram: vLLM ──
        pending_ngram_vllm=""
        for entry in "${PENDING_P4_NGRAM[@]}"; do
            IFS='|' read -r m s e <<<"$entry"
            if [ "$m" = "$model" ] && [ "$e" = "vllm-ngram" ]; then
                if [ -z "$pending_ngram_vllm" ]; then
                    pending_ngram_vllm="$s"
                else
                    pending_ngram_vllm="${pending_ngram_vllm},${s}"
                fi
            fi
        done
        if [ -n "$pending_ngram_vllm" ]; then
            log "P4 NGRAM VLLM: $model | $pending_ngram_vllm"
            MAX_MODEL_LEN=4096 \
                run_vllm_gemma4 "$model" vllm-ngram-server 8000 vllm-ngram "$pending_ngram_vllm" "$COOLDOWN" 1 results \
                --enforce-eager --disable-frontend-multiprocessing \
                --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}' \
                && COMPLETED=$((COMPLETED+1)) \
                || SKIPPED=$((SKIPPED+1))
        fi

        # ── Ngram: SGLang ──
        pending_ngram_sglang=""
        for entry in "${PENDING_P4_NGRAM[@]}"; do
            IFS='|' read -r m s e <<<"$entry"
            if [ "$m" = "$model" ] && [ "$e" = "sglang-ngram" ]; then
                if [ -z "$pending_ngram_sglang" ]; then
                    pending_ngram_sglang="$s"
                else
                    pending_ngram_sglang="${pending_ngram_sglang},${s}"
                fi
            fi
        done
        if [ -n "$pending_ngram_sglang" ]; then
            log "P4 NGRAM SGLANG: $model | $pending_ngram_sglang"
            MAX_MODEL_LEN=4096 \
                run_sglang_gemma4 "$model" sglang-server 8001 sglang-ngram "$pending_ngram_sglang" "$COOLDOWN" 1 results \
                --speculative-algorithm NGRAM --speculative-num-draft-tokens 16 \
                && COMPLETED=$((COMPLETED+1)) \
                || SKIPPED=$((SKIPPED+1))
        fi
    done
fi

# =============================================================================
#  SUMMARY
# =============================================================================
log "SUMMARY"
echo "  completed groups : $COMPLETED"
echo "  failed groups    : $SKIPPED"
echo "  errors           : ${#ERRORS[@]}"

echo ""
echo "Results per directory:"
for dir in results results_variance results_concurrency64 results_decode_sweep; do
    count=$(find "$dir" -name "*.json" -not -name "*manifest*" 2>/dev/null | wc -l | tr -d ' ')
    printf "  %-30s %4s result files\n" "$dir/" "$count"
done

if [ ${#ERRORS[@]} -gt 0 ]; then
    echo ""
    echo "Errors:"
    for i in "${!ERRORS[@]}"; do echo "  $((i+1)). ${ERRORS[$i]}"; done
fi

echo ""
echo "Re-run with --audit-only to see remaining gaps."
echo ""
echo "Next steps (analysis):"
echo "  $PYTHON -m analysis.variance_analysis --results-dir results_variance"
echo "  $PYTHON -m analysis.tpot_analysis --results-dir results_variance"
echo "  $PYTHON -m analysis.decode_length_analysis --results-dir results_decode_sweep"
