#!/usr/bin/env bash
# Phase 3 — Decode-length sweep
#
# Fixed ~512-token prompts, max_output_tokens ∈ {64, 256, 1024, 4096}
# Isolates prefill-bound vs decode-bound behaviour.
#
# Models: Gemma 2 2B, Phi-4 mini, Llama 3.1 8B
# 3 models × 2 engines × 4 lengths × 3 iterations = 72 runs
# Runtime estimate: ~4 hours (~$5)
#
# Usage:
#   bash scripts/run_decode_sweep.sh [vllm|sglang|both]

set -euo pipefail

ENGINE_TARGET="${1:-both}"
RESULTS_DIR="results_decode_sweep"
COOLDOWN=300
ITERATIONS=3

# All four decode-length configs run in a single matrix call per model/engine
SCENARIOS="decode_length_sweep_64,decode_length_sweep_256,decode_length_sweep_1024,decode_length_sweep_4096"

MODELS=(
  "google/gemma-2-2b-it"
  "microsoft/Phi-4-mini-instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
)

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo " Phase 3 — Decode-Length Sweep"
echo " Target engines : $ENGINE_TARGET"
echo " Scenarios      : $SCENARIOS"
echo " Iterations     : $ITERATIONS"
echo " Cooldown (s)   : $COOLDOWN"
echo " Output dir     : $RESULTS_DIR"
echo "========================================"
echo ""

run_engine_block() {
  local engine="$1"
  echo ">>> Starting $engine block — $(date)"
  echo "    Ensure the $engine server is running."
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
    echo ">>> Switching to SGLang. Stop vLLM, start SGLang, then press Enter."
    read -r
  fi
  run_engine_block "sglang"
fi

echo "========================================"
echo " Phase 3 complete."
echo " Analyse with:"
echo "   python -m analysis.decode_length_analysis --results-dir $RESULTS_DIR"
echo "========================================"
