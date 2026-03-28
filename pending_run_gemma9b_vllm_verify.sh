#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

OUTDIR="results/gemma-2-9b-it"
MODEL="google/gemma-2-9b-it"
VENV_PY="./.venv/bin/python"

if [ ! -d "$OUTDIR" ]; then
  echo "Missing results dir: $OUTDIR"
  exit 1
fi

count=$(find "$OUTDIR" -maxdepth 1 -name '*VLLMClient*.json' | wc -l | tr -d ' ')
echo "VLLM JSON count for Gemma 9B: $count"
find "$OUTDIR" -maxdepth 1 -name '*VLLMClient*.json' | sort

if [ "$count" -lt 10 ]; then
  echo "Expected at least 10 VLLM JSON files (5 scenarios x 2 iterations)."
  exit 1
fi

echo
echo "Regenerating per-model reports..."
"$VENV_PY" run_experiment.py report --results-dir "$OUTDIR" --model "$MODEL" --output "$OUTDIR/report.html"
"$VENV_PY" run_experiment.py final-report --results-dir "$OUTDIR" --model "$MODEL" --output "$OUTDIR/final_report.md"

echo
echo "Gemma 9B vLLM results look complete."
echo "Report: $OUTDIR/report.html"
echo "Final report: $OUTDIR/final_report.md"
