#!/usr/bin/env bash
# Phase 2 — Concurrency-64 extended ramp on 7–9B models
#
# New scenario: throughput_ramp_extended
#   concurrency levels: [1, 4, 8, 16, 32, 64]
#   150 requests/level
#
# Models: Llama 3.1 8B, Qwen3 8B, Mistral 7B v0.3, Gemma 2 9B
# 4 models × 2 engines × 1 scenario × 3 iterations = 24 runs
# Runtime estimate: ~3 hours (~$4)
#
# OOM handling: if a run fails (exit non-zero), the model/engine/concurrency tuple
# is appended to reports/oom_ceiling.md and the loop continues.
#
# Usage:
#   bash scripts/run_concurrency_64.sh [vllm|sglang|both]

set -uo pipefail   # note: NOT -e so we can trap per-run failures

ENGINE_TARGET="${1:-both}"
RESULTS_DIR="results_concurrency64"
SCENARIO="throughput_ramp_extended"
COOLDOWN=300
ITERATIONS=3
OOM_LOG="reports/oom_ceiling.md"

MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen3-8B"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "google/gemma-2-9b-it"
)

mkdir -p "$RESULTS_DIR" reports

# Initialise OOM log if it doesn't exist
if [[ ! -f "$OOM_LOG" ]]; then
  cat > "$OOM_LOG" <<'EOF'
# OOM / Failure Ceiling Log

Records (model, engine, scenario) tuples where a run failed — likely GPU OOM
at high concurrency. Treat these as memory-pressure ceilings, not harness bugs.

| Timestamp | Model | Engine | Scenario | Exit code | Notes |
|-----------|-------|--------|----------|-----------|-------|
EOF
fi

echo "========================================"
echo " Phase 2 — Concurrency-64 Extended Ramp"
echo " Target engines : $ENGINE_TARGET"
echo " Scenario       : $SCENARIO"
echo " Iterations     : $ITERATIONS"
echo " Cooldown (s)   : $COOLDOWN"
echo " Output dir     : $RESULTS_DIR"
echo " OOM log        : $OOM_LOG"
echo "========================================"
echo ""

run_model_engine() {
  local model="$1"
  local engine="$2"
  local model_slug
  model_slug="$(echo "$model" | tr '/' '-' | tr '[:upper:]' '[:lower:]')"

  echo "--- Model: $model  Engine: $engine ---"
  if python run_experiment.py matrix \
      --model "$model" \
      --scenarios "$SCENARIO" \
      --engines "$engine" \
      --iterations "$ITERATIONS" \
      --cooldown-seconds "$COOLDOWN" \
      --output-dir "$RESULTS_DIR"; then
    echo "    OK"
  else
    local exit_code=$?
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "    FAILED (exit $exit_code) — logging to $OOM_LOG"
    echo "| $ts | $model | $engine | $SCENARIO | $exit_code | Likely OOM at high concurrency |" >> "$OOM_LOG"
  fi
  echo ""
}

run_engine_block() {
  local engine="$1"
  echo ">>> Starting $engine block — $(date)"
  echo "    Ensure the $engine server is running."
  echo ""
  for model in "${MODELS[@]}"; do
    run_model_engine "$model" "$engine"
  done
  echo ">>> $engine block complete — $(date)"
  echo ""
}

if [[ "$ENGINE_TARGET" == "vllm" || "$ENGINE_TARGET" == "both" ]]; then
  run_engine_block "vllm"
fi

if [[ "$ENGINE_TARGET" == "sglang" || "$ENGINE_TARGET" == "both" ]]; then
  if [[ "$ENGINE_TARGET" == "both" ]]; then
    echo ">>> Switching to SGLang. Stop vLLM, start SGLang, then press Enter."
    read -r
  fi
  run_engine_block "sglang"
fi

echo "========================================"
echo " Phase 2 complete."
echo " OOM events (if any): $OOM_LOG"
echo " Analyse saturation with:"
echo "   python run_experiment.py final-report --output reports/saturation_report.md"
echo "========================================"
