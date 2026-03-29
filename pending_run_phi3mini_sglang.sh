#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

MODEL="microsoft/Phi-3-mini-4k-instruct"
OUTDIR="results/phi-3-mini-4k-instruct"
STATUS_FILE="results/pending_phi3mini_sglang_status.txt"
LOG="results/pending_phi3mini_sglang_$(date +%Y%m%d_%H%M%S).log"
VENV_PY="./.venv/bin/python"
SCENARIOS="single_request_latency,throughput_ramp,long_context_stress,prefix_sharing_benefit,structured_generation_speed"
KNOWN_GOOD_CONFIG="docker-compose.phi3mini-sglang-a10g.yml"
CLEAN_ARCHIVE_ROOT="$REPO_DIR/../tmp/phi3mini_sglang_cleanup_$(date +%Y%m%d_%H%M%S)"
DOCKER_CMD=(docker)

mkdir -p results "$OUTDIR"
exec > >(tee -a "$LOG") 2>&1

update_status() {
  printf '%s\n' "$1" > "$STATUS_FILE"
}

setup_docker_cmd() {
  if docker ps >/dev/null 2>&1; then
    DOCKER_CMD=(docker)
    return 0
  fi

  if sudo -n docker ps >/dev/null 2>&1; then
    DOCKER_CMD=(sudo -n docker)
    return 0
  fi

  echo "Unable to access Docker via docker or sudo -n docker" >&2
  return 1
}

cleanup() {
  "${DOCKER_CMD[@]}" compose stop sglang >/dev/null 2>&1 || true
}
trap cleanup EXIT

wait_healthy() {
  local attempts="${1:-30}"
  for i in $(seq 1 "$attempts"); do
    echo "[health] attempt $i"

    if curl -fsS http://localhost:8001/health >/tmp/phi3mini_sglang_health.json 2>/dev/null; then
      cat /tmp/phi3mini_sglang_health.json || true
      return 0
    fi

    local state=""
    state=$("${DOCKER_CMD[@]}" inspect -f '{{.State.Status}}' sglang-server 2>/dev/null || true)
    if [ -n "$state" ] && [ "$state" != "running" ] && [ "$state" != "created" ] && [ "$state" != "restarting" ]; then
      echo "[health] container state is $state; aborting early"
      "${DOCKER_CMD[@]}" compose logs sglang --tail 200 || true
      return 1
    fi

    sleep 10
  done

  echo "[health] timed out waiting for /health"
  "${DOCKER_CMD[@]}" compose logs sglang --tail 200 || true
  return 1
}

verify_model() {
  local actual
  actual=$(curl -fsS http://localhost:8001/v1/models | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
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
    "model": "microsoft/Phi-3-mini-4k-instruct",
    "prompt": "hello",
    "max_tokens": 8,
}).encode()
req = urllib.request.Request(
    "http://localhost:8001/v1/completions",
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

clean_existing_phi3_sglang_outputs() {
  local cleaned=0
  local model_archive="$CLEAN_ARCHIVE_ROOT/model_dir"
  local results_archive="$CLEAN_ARCHIVE_ROOT/results_root"
  mkdir -p "$model_archive" "$results_archive"

  while IFS= read -r -d '' f; do
    mv "$f" "$model_archive/"
    echo "Archived stale model artifact to $model_archive: $(basename "$f")"
    cleaned=1
  done < <(find "$OUTDIR" -maxdepth 1 -type f \( -name '*SGLangClient*.json' -o -name 'matrix_manifest_*.json' -o -name 'report.html' -o -name 'final_report.md' \) -print0)

  while IFS= read -r -d '' f; do
    if [ "$f" = "$LOG" ] || [ "$f" = "$STATUS_FILE" ]; then
      continue
    fi
    mv "$f" "$results_archive/"
    echo "Archived stale run artifact to $results_archive: $(basename "$f")"
    cleaned=1
  done < <(find results -maxdepth 1 -type f \( -name 'pending_phi3mini_sglang_*.log' -o -name 'pending_phi3mini_sglang_status.txt' \) -print0)

  if [ "$cleaned" -eq 1 ]; then
    echo "Archived stale Phi-3/SGLang artifacts under $CLEAN_ARCHIVE_ROOT"
  fi
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

  "${DOCKER_CMD[@]}" compose stop vllm sglang >/dev/null 2>&1 || true
  "${DOCKER_CMD[@]}" compose rm -f vllm sglang >/dev/null 2>&1 || true

  "${DOCKER_CMD[@]}" compose -f docker-compose.yml -f "$KNOWN_GOOD_CONFIG" --profile sglang up -d sglang

  wait_healthy 30
  verify_model
  probe_completions >/tmp/phi3mini_sglang_probe.log 2>&1
  local probe_rc=$?
  echo "[attempt:$label] /v1/completions rc=$probe_rc"
  cat /tmp/phi3mini_sglang_probe.log || true
  if [ "$probe_rc" -ne 0 ]; then
    "${DOCKER_CMD[@]}" compose logs sglang --tail 200 || true
    return 1
  fi

  update_status "READY attempt=$label model=$MODEL"
}

run_matrix() {
  update_status "RUNNING matrix model=$MODEL outdir=$OUTDIR"
  clean_existing_phi3_sglang_outputs
  "$VENV_PY" run_experiment.py matrix \
    --scenarios "$SCENARIOS" \
    --engines sglang \
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
  count=$(find "$OUTDIR" -maxdepth 1 -name '*SGLangClient*.json' | wc -l | tr -d ' ')
  echo "[verify] SGLang JSON count: $count"
  if [ "$count" -lt 10 ]; then
    echo "Expected at least 10 SGLang JSON files (5 scenarios x 2 iterations)."
    return 1
  fi
}

setup_docker_cmd

update_status "STARTED pending-phi3mini-sglang log=$LOG docker=${DOCKER_CMD[*]}"
echo "== Pending run: Phi-3 mini on SGLang =="
date -u
printf 'Log file: %s\n' "$LOG"
printf 'Docker command: %s\n' "${DOCKER_CMD[*]}"

start_known_good_attempt
run_matrix
verify_output_count
generate_reports

"${DOCKER_CMD[@]}" compose stop sglang || true
update_status "COMPLETED pending-phi3mini-sglang log=$LOG outdir=$OUTDIR"

echo
echo "== Pending Phi-3 mini SGLang run complete =="
echo "Results dir: $OUTDIR"
echo "Status file: $STATUS_FILE"
date -u
