#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/ubuntu/repos/inference-engine-benchmark-system
mkdir -p results results/_stale_runs

LOG="results/phase3_resume_from_phi_$(date +%Y%m%d_%H%M%S).log"
STATUS_FILE="results/phase3_status.txt"
exec > >(tee -a "$LOG") 2>&1

FAILED_ITEMS=()
SKIPPED_ITEMS=()

update_status() {
  printf '%s\n' "$1" > "$STATUS_FILE"
}

note_failure() {
  local model="$1"
  local engine="$2"
  local step="$3"
  FAILED_ITEMS+=("$model|$engine|$step")
  update_status "FAILED model=$model engine=$engine step=$step"
}

note_skip() {
  local model="$1"
  local engine="$2"
  local reason="$3"
  SKIPPED_ITEMS+=("$model|$engine|$reason")
  echo "SKIP model=$model engine=$engine reason=$reason"
  update_status "SKIPPED model=$model engine=$engine reason=$reason"
}

model_max_len() {
  local model="$1"
  case "$model" in
    microsoft/Phi-3-mini-4k-instruct) echo 4096 ;;
    *) echo 8192 ;;
  esac
}

wait_healthy() {
  local engine="$1"
  local max_attempts="${2:-80}"
  local healthy=0
  for i in $(seq 1 "$max_attempts"); do
    echo "[$engine] health attempt $i"
    if ./.venv/bin/python run_experiment.py health --engines "$engine" | tee "/tmp/${engine}_health.txt" && grep -q "healthy" "/tmp/${engine}_health.txt"; then
      healthy=1
      break
    fi
    sleep 15
  done
  if [ "$healthy" -ne 1 ]; then
    echo "[$engine] failed to become healthy"
    docker logs --tail 200 "${engine}-server" || true
    return 1
  fi
}

verify_engine_model() {
  local engine="$1"
  local expected="$2"
  local actual=""
  if [ "$engine" = "vllm" ]; then
    actual=$(curl -fsS http://localhost:8000/v1/models | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
  else
    actual=$(docker inspect sglang-server --format '{{range .Config.Cmd}}{{printf "%s " .}}{{end}}' | sed -n 's/.*--model-path \([^ ]*\).*/\1/p')
  fi
  echo "[$engine] expected model: $expected"
  echo "[$engine] actual model  : $actual"
  test "$actual" = "$expected"
}

smoke_completion() {
  local engine="$1"
  local expected="$2"
  local port=8000
  [ "$engine" = "sglang" ] && port=8001
  python3 - "$port" "$expected" <<'PY'
import json, sys, urllib.request
port, expected = sys.argv[1], sys.argv[2]
payload = json.dumps({"model": expected, "prompt": "Hello", "max_tokens": 4}).encode()
req = urllib.request.Request(
    f"http://localhost:{port}/v1/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=30) as resp:
    body = json.loads(resp.read().decode())
    print("smoke_status", resp.status)
    print("smoke_model", body.get("model", ""))
PY
}

should_skip_engine_model() {
  local engine="$1"
  local model="$2"
  if [ "$model" = "microsoft/Phi-3-mini-4k-instruct" ] && [ "$engine" = "sglang" ]; then
    return 0
  fi
  return 1
}

run_engine_matrix() {
  local engine="$1"
  local model="$2"
  local outdir="$3"
  local max_len
  local rc
  max_len="$(model_max_len "$model")"

  update_status "RUNNING model=$model engine=$engine outdir=$outdir"
  echo
  echo "== $engine :: $model =="
  echo "max context :: $max_len"
  date -u
  df -h / | tail -1

  if ! MODEL="$model" MAX_MODEL_LEN="$max_len" docker compose --profile "$engine" up -d "$engine"; then
    note_failure "$model" "$engine" "compose-up"
    docker compose stop "$engine" || true
    return 1
  fi

  if ! wait_healthy "$engine" 80; then
    note_failure "$model" "$engine" "health"
    docker compose stop "$engine" || true
    return 1
  fi

  if ! verify_engine_model "$engine" "$model"; then
    note_failure "$model" "$engine" "verify-model"
    docker compose stop "$engine" || true
    return 1
  fi

  if ! smoke_completion "$engine" "$model"; then
    note_failure "$model" "$engine" "smoke"
    docker compose stop "$engine" || true
    return 1
  fi

  set +e
  ./.venv/bin/python run_experiment.py matrix \
    --scenarios "single_request_latency,throughput_ramp,long_context_stress,prefix_sharing_benefit,structured_generation_speed" \
    --engines "$engine" \
    --model "$model" \
    --output-dir "$outdir" \
    --iterations 2 \
    --cooldown-seconds 60
  rc=$?
  set -e

  docker compose stop "$engine" || true

  if [ "$rc" -ne 0 ]; then
    note_failure "$model" "$engine" "matrix"
    return 1
  fi

  update_status "DONE model=$model engine=$engine outdir=$outdir"
  return 0
}

generate_reports() {
  local model="$1"
  local outdir="$2"
  if ! find "$outdir" -maxdepth 1 -name '*.json' -print -quit | grep -q .; then
    echo "No JSON results for $model; skipping report generation"
    note_skip "$model" "report" "no-json-results"
    return 0
  fi

  echo "== Generating per-model reports for $model =="
  if ! ./.venv/bin/python run_experiment.py report \
    --results-dir "$outdir" \
    --output "$outdir/report.html"; then
    note_failure "$model" "report" "html-report"
  fi
  if ! ./.venv/bin/python run_experiment.py final-report \
    --results-dir "$outdir" \
    --output "$outdir/final_report.md"; then
    note_failure "$model" "report" "markdown-report"
  fi
}

# Reuse existing images/tags to avoid unnecessary repulls.
docker image inspect vllm/vllm-openai:latest >/dev/null 2>&1 && docker tag vllm/vllm-openai:latest vllm/vllm-openai:v0.18.0-cu130 || true
docker image inspect lmsysorg/sglang:latest >/dev/null 2>&1 && docker tag lmsysorg/sglang:latest lmsysorg/sglang:nightly-dev-cu13-20260321-94194537 || true

MODELS=(
  'microsoft/Phi-3-mini-4k-instruct|phi-3-mini-4k-instruct'
  'Qwen/Qwen2.5-7B-Instruct|qwen2.5-7b-instruct'
  'mistralai/Mistral-7B-Instruct-v0.3|mistral-7b-instruct-v0.3'
  'google/gemma-2-9b-it|gemma-2-9b-it'
  'meta-llama/Llama-3.2-3B-Instruct|llama-3.2-3b-instruct'
  'meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8b-instruct'
)

update_status "STARTED phase3-resume-from-phi log=$LOG"
echo "== Phase 3 resume from Phi-3 + Llama =="
date -u
printf 'Log file: %s\n' "$LOG"

for entry in "${MODELS[@]}"; do
  IFS='|' read -r model slug <<<"$entry"
  outdir="results/$slug"

  if [ -d "$outdir" ] && [ "$(find "$outdir" -type f | wc -l)" -gt 0 ]; then
    mv "$outdir" "results/_stale_runs/${slug}_$(date +%Y%m%d_%H%M%S)"
  fi
  mkdir -p "$outdir"

  update_status "STARTING model=$model outdir=$outdir"
  echo
  echo "############################################################"
  echo "## MODEL: $model"
  echo "## OUTDIR: $outdir"
  echo "############################################################"

  successful_engines=0
  for engine in vllm sglang; do
    if should_skip_engine_model "$engine" "$model"; then
      note_skip "$model" "$engine" "known-incompatible"
      continue
    fi
    if run_engine_matrix "$engine" "$model" "$outdir"; then
      successful_engines=$((successful_engines + 1))
    else
      echo "Continuing after failure: model=$model engine=$engine"
    fi
  done

  generate_reports "$model" "$outdir"
  update_status "COMPLETED model=$model outdir=$outdir successful_engines=$successful_engines"
done

echo
if [ "${#FAILED_ITEMS[@]}" -gt 0 ]; then
  echo "== Failures =="
  printf ' - %s\n' "${FAILED_ITEMS[@]}"
fi
if [ "${#SKIPPED_ITEMS[@]}" -gt 0 ]; then
  echo "== Skipped =="
  printf ' - %s\n' "${SKIPPED_ITEMS[@]}"
fi

update_status "COMPLETED phase3 remaining models with_failures=${#FAILED_ITEMS[@]} skipped=${#SKIPPED_ITEMS[@]}"
echo "== Phase 3 remaining models complete =="
date -u
