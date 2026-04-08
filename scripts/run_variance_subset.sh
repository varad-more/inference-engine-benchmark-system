#!/usr/bin/env bash
# Phase 1 — Variance subset (credibility backbone)
#
# 4 models × 5 scenarios × 2 engines × 5 iterations = 200 runs
# Results land in results_variance/ (separate from main results/) so the
# variance analysis script can find them without false-positives from prior runs.
#
# Runtime estimate on g5.2xlarge: ~11–12 hours (~$15 at $1.21/hr)
# One engine at a time — start the engine, run its block, stop it, switch.
#
# Usage:
#   bash scripts/run_variance_subset.sh [vllm|sglang|both]
#
# Default (no arg) runs both engines sequentially.
# Pass "vllm" or "sglang" to run a single engine block (useful for resuming).

set -euo pipefail

ENGINE_TARGET="${1:-both}"
RESULTS_DIR="results_variance"
SCENARIOS="single_request_latency,throughput_ramp,long_context_stress,prefix_sharing_benefit,structured_generation_speed"
COOLDOWN=300
ITERATIONS=5

MODELS=(
  "google/gemma-2-2b-it"
  "microsoft/Phi-4-mini-instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "google/gemma-3-4b-it"
)

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo " Phase 1 — Variance Subset"
echo " Target engines : $ENGINE_TARGET"
echo " Iterations     : $ITERATIONS"
echo " Cooldown (s)   : $COOLDOWN"
echo " Output dir     : $RESULTS_DIR"
echo "========================================"
echo ""

run_engine_block() {
  local engine="$1"  # "vllm" or "sglang"

  echo ">>> Starting $engine block — $(date)"
  echo "    Make sure the $engine server is running before continuing."
  echo ""

  for model in "${MODELS[@]}"; do
    echo "--- Model: $model  Engine: $engine ---"
    python run_experiment.py matrix \
      --model "$model" \
      --scenarios "$SCENARIOS" \
      --engines "$engine" \
      --iterations "$ITERATIONS" \
      --cooldown-seconds "$COOLDOWN" \
      --output-dir "$RESULTS_DIR"
    echo ""
  done

  echo ">>> $engine block complete — $(date)"
  echo ""
}

if [[ "$ENGINE_TARGET" == "vllm" || "$ENGINE_TARGET" == "both" ]]; then
  run_engine_block "vllm"
fi

if [[ "$ENGINE_TARGET" == "sglang" || "$ENGINE_TARGET" == "both" ]]; then
  if [[ "$ENGINE_TARGET" == "both" ]]; then
    echo ">>> Switching to SGLang. Stop vLLM, start SGLang, then press Enter to continue."
    read -r
  fi
  run_engine_block "sglang"
fi

echo "========================================"
echo " All variance runs complete."
echo " Analyse with:"
echo "   python -m analysis.variance_analysis --results-dir $RESULTS_DIR"
echo "========================================"
