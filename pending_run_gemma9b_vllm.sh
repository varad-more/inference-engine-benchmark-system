#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

MODEL="google/gemma-2-9b-it"
OUTDIR="results/gemma-2-9b-it"
STATUS_FILE="results/pending_gemma9b_vllm_status.txt"
LOG="results/pending_gemma9b_vllm_$(date +%Y%m%d_%H%M%S).log"
VENV_PY="./.venv/bin/python"
SCENARIOS="single_request_latency,throughput_ramp,long_context_stress,prefix_sharing_benefit,structured_generation_speed"
KNOWN_GOOD_CONFIG="docker-compose.gemma9b-vllm-a10g.yml"

mkdir -p results results/_stale_runs "$OUTDIR"
exec > >(tee -a "$LOG") 2>&1

update_status() {
  printf '%s\n' "$1" > "$STATUS_FILE"
}

cleanup() {
  docker compose stop vllm >/dev/null 2>&1 || true
  rm -f .tmp.gemma9b-vllm.override.yml
}
trap cleanup EXIT

wait_healthy() {
  local attempts="${1:-60}"
  for i in $(seq 1 "$attempts"); do
    echo "[health] attempt $i"

    if curl -fsS http://localhost:8000/health >/tmp/gemma9b_vllm_health.json 2>/dev/null; then
      cat /tmp/gemma9b_vllm_health.json || true
      return 0
    fi

    local state=""
    state=$(docker inspect -f '{{.State.Status}}' vllm-server 2>/dev/null || true)
    if [ -n "$state" ] && [ "$state" != "running" ] && [ "$state" != "created" ] && [ "$state" != "restarting" ]; then
      echo "[health] container state is $state; aborting early"
      docker compose logs vllm --tail 200 || true
      return 1
    fi

    sleep 10
  done

  echo "[health] timed out waiting for /health"
  docker compose logs vllm --tail 200 || true
  return 1
}

verify_model() {
  local actual
  actual=$(curl -fsS http://localhost:8000/v1/models | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
  echo "[verify] expected: $MODEL"
  echo "[verify] actual  : $actual"
  test "$actual" = "$MODEL"
}

probe_completions() {
  python3 - <<'PY'
import json
import sys
import urllib.error
import urllib.request

payload = json.dumps({
    "model": "google/gemma-2-9b-it",
    "prompt": "hello",
    "max_tokens": 5,
}).encode()
req = urllib.request.Request(
    "http://localhost:8000/v1/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode()
        print(body)
        sys.exit(0)
except urllib.error.HTTPError as exc:
    print(exc.read().decode())
    sys.exit(exc.code)
PY
}

probe_chat_completions() {
  python3 - <<'PY'
import json
import sys
import urllib.error
import urllib.request

payload = json.dumps({
    "model": "google/gemma-2-9b-it",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 5,
}).encode()
req = urllib.request.Request(
    "http://localhost:8000/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode()
        print(body)
        sys.exit(0)
except urllib.error.HTTPError as exc:
    print(exc.read().decode())
    sys.exit(exc.code)
PY
}

write_override_file() {
  local max_len="$1"
  local gpu_mem="$2"
  local disable_frontend="$3"
  local enforce_eager="$4"

  cat > .tmp.gemma9b-vllm.override.yml <<EOF
services:
  vllm:
    command:
      - "--model"
      - "$MODEL"
      - "--host"
      - "0.0.0.0"
      - "--port"
      - "8000"
      - "--enable-prefix-caching"
      - "--max-model-len"
      - "$max_len"
      - "--gpu-memory-utilization"
      - "$gpu_mem"
      - "--served-model-name"
      - "$MODEL"
EOF

  if [ "$disable_frontend" = "1" ]; then
    cat >> .tmp.gemma9b-vllm.override.yml <<'EOF'
      - "--disable-frontend-multiprocessing"
EOF
  fi

  if [ "$enforce_eager" = "1" ]; then
    cat >> .tmp.gemma9b-vllm.override.yml <<'EOF'
      - "--enforce-eager"
EOF
  fi
}

finish_start_attempt() {
  local label="$1"

  if ! wait_healthy 60; then
    echo "[attempt:$label] vLLM did not become healthy"
    docker compose stop vllm || true
    return 1
  fi

  if ! verify_model; then
    echo "[attempt:$label] model verification failed"
    docker compose logs vllm --tail 200 || true
    docker compose stop vllm || true
    return 1
  fi

  set +e
  probe_completions >/tmp/gemma9b_vllm_probe.log 2>&1
  local probe_rc=$?
  set -e

  echo "[attempt:$label] /v1/completions rc=$probe_rc"
  cat /tmp/gemma9b_vllm_probe.log || true

  if [ "$probe_rc" -eq 0 ]; then
    update_status "READY attempt=$label model=$MODEL"
    return 0
  fi

  if [ "$probe_rc" -eq 404 ]; then
    echo "[attempt:$label] legacy completions missing; checking chat endpoint"
    set +e
    probe_chat_completions >/tmp/gemma9b_vllm_chat_probe.log 2>&1
    local chat_rc=$?
    set -e
    echo "[attempt:$label] /v1/chat/completions rc=$chat_rc"
    cat /tmp/gemma9b_vllm_chat_probe.log || true
  fi

  docker compose logs vllm --tail 200 || true
  docker compose stop vllm || true
  return 1
}

start_attempt() {
  local label="$1"
  local max_len="$2"
  local gpu_mem="$3"
  local disable_frontend="$4"
  local enforce_eager="$5"

  echo
  echo "============================================================"
  echo "Attempt: $label"
  echo "MODEL=$MODEL"
  echo "MAX_MODEL_LEN=$max_len"
  echo "VLLM_GPU_MEMORY_UTILIZATION=$gpu_mem"
  echo "disable_frontend_multiprocessing=$disable_frontend"
  echo "enforce_eager=$enforce_eager"
  echo "============================================================"

  update_status "STARTING attempt=$label model=$MODEL"

  docker compose stop sglang vllm >/dev/null 2>&1 || true
  docker compose rm -f sglang vllm >/dev/null 2>&1 || true

  write_override_file "$max_len" "$gpu_mem" "$disable_frontend" "$enforce_eager"

  docker compose -f docker-compose.yml -f .tmp.gemma9b-vllm.override.yml --profile vllm up -d vllm
  finish_start_attempt "$label"
}

start_known_good_attempt() {
  local label="known-good-a10g"

  echo
  echo "============================================================"
  echo "Attempt: $label"
  echo "MODEL=$MODEL"
  echo "config=$KNOWN_GOOD_CONFIG"
  echo "============================================================"

  update_status "STARTING attempt=$label model=$MODEL config=$KNOWN_GOOD_CONFIG"

  docker compose stop sglang vllm >/dev/null 2>&1 || true
  docker compose rm -f sglang vllm >/dev/null 2>&1 || true

  docker compose -f docker-compose.yml -f "$KNOWN_GOOD_CONFIG" --profile vllm up -d vllm
  finish_start_attempt "$label"
}

archive_existing_vllm_outputs() {
  local stale_dir
  stale_dir="results/_stale_runs/gemma-2-9b-it_vllm_$(date +%Y%m%d_%H%M%S)"
  if find "$OUTDIR" -maxdepth 1 -name '*VLLMClient*.json' -print -quit | grep -q .; then
    mkdir -p "$stale_dir"
    find "$OUTDIR" -maxdepth 1 -name '*VLLMClient*.json' -exec mv {} "$stale_dir"/ \;
    find "$OUTDIR" -maxdepth 1 -name 'matrix_manifest_*.json' -exec mv {} "$stale_dir"/ \; || true
    echo "Archived existing vLLM outputs to $stale_dir"
  fi
}

run_matrix() {
  update_status "RUNNING matrix model=$MODEL outdir=$OUTDIR"
  archive_existing_vllm_outputs
  "$VENV_PY" run_experiment.py matrix \
    --scenarios "$SCENARIOS" \
    --engines vllm \
    --model "$MODEL" \
    --output-dir "$OUTDIR" \
    --iterations 2 \
    --cooldown-seconds 120
}

generate_reports() {
  update_status "GENERATING_REPORTS model=$MODEL outdir=$OUTDIR"
  "$VENV_PY" run_experiment.py report --results-dir "$OUTDIR" --model "$MODEL" --output "$OUTDIR/report.html"
  "$VENV_PY" run_experiment.py final-report --results-dir "$OUTDIR" --model "$MODEL" --output "$OUTDIR/final_report.md"
}

verify_output_count() {
  local count
  count=$(find "$OUTDIR" -maxdepth 1 -name '*VLLMClient*.json' | wc -l | tr -d ' ')
  echo "[verify] VLLM JSON count: $count"
  if [ "$count" -lt 10 ]; then
    echo "Expected at least 10 VLLM JSON files (5 scenarios x 2 iterations)."
    return 1
  fi
}

update_status "STARTED pending-gemma9b-vllm log=$LOG"
echo "== Pending run: Gemma 9B on vLLM =="
date -u
printf 'Log file: %s\n' "$LOG"

if ! start_known_good_attempt; then
  if ! start_attempt "standard" 4096 0.85 0 0; then
    if ! start_attempt "disable-frontend-multiprocessing" 4096 0.85 1 0; then
      if ! start_attempt "low-memory" 2048 0.95 1 0; then
        start_attempt "low-memory-eager" 2048 0.95 1 1
      fi
    fi
  fi
fi

run_matrix
verify_output_count
generate_reports

docker compose stop vllm || true
update_status "COMPLETED pending-gemma9b-vllm log=$LOG outdir=$OUTDIR"

echo
echo "== Pending Gemma 9B vLLM run complete =="
echo "Results dir: $OUTDIR"
echo "Status file: $STATUS_FILE"
date -u
