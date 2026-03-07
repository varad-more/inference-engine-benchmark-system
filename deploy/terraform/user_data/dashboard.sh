#!/usr/bin/env bash
# =============================================================================
# dashboard.sh — Bootstrap for the CPU-only dashboard/CLI instance (multi mode)
#
# Injected variables (Terraform templatefile):
#   HF_TOKEN    — HuggingFace token (passed through to .env, not used here)
#   MODEL_ID    — Model ID (for CLI commands)
#   GIT_REPO    — Git repo URL
#   PROJECT     — Project tag
#   VLLM_HOST   — Private IP of the vLLM engine instance
#   SGLANG_HOST — Private IP of the SGLang engine instance
#
# This instance does NOT need a GPU — it only runs:
#   - FastAPI dashboard (port 3000)
#   - Benchmark CLI (python run_experiment.py)
#   - HTML report generation
# =============================================================================

set -euo pipefail
exec > >(tee /var/log/benchmark-dashboard-setup.log | logger -t benchmark-dashboard) 2>&1

HF_TOKEN="${HF_TOKEN}"
MODEL_ID="${MODEL_ID}"
GIT_REPO="${GIT_REPO}"
PROJECT="${PROJECT}"
VLLM_HOST="${VLLM_HOST}"
SGLANG_HOST="${SGLANG_HOST}"
HOME_DIR="/home/ubuntu"
APP_DIR="$HOME_DIR/benchmark"

echo "=========================================="
echo " Dashboard bootstrap starting"
echo " VLLM_HOST=$VLLM_HOST  SGLANG_HOST=$SGLANG_HOST"
echo " $(date)"
echo "=========================================="

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
apt-get update -y
apt-get install -y --no-install-recommends \
    git curl wget jq python3 python3-pip python3-venv \
    ca-certificates build-essential

# ---------------------------------------------------------------------------
# 2. Clone repository
# ---------------------------------------------------------------------------
echo "Cloning $GIT_REPO → $APP_DIR"
if [ -d "$APP_DIR" ]; then
    git -C "$APP_DIR" pull
else
    git clone "$GIT_REPO" "$APP_DIR"
fi
chown -R ubuntu:ubuntu "$APP_DIR"

# ---------------------------------------------------------------------------
# 3. Python virtual environment + dependencies
# ---------------------------------------------------------------------------
python3 -m venv "$APP_DIR/.venv"
source "$APP_DIR/.venv/bin/activate"

pip install --upgrade pip --quiet
pip install --no-cache-dir --quiet \
    httpx \
    pydantic \
    structlog \
    fastapi \
    "uvicorn[standard]" \
    websockets \
    typer \
    rich \
    matplotlib \
    jinja2 \
    aiofiles \
    numpy

deactivate
chown -R ubuntu:ubuntu "$APP_DIR/.venv"

# ---------------------------------------------------------------------------
# 4. Write .env and config
# ---------------------------------------------------------------------------
mkdir -p "$APP_DIR/results"
chown -R ubuntu:ubuntu "$APP_DIR/results"

cat > "$APP_DIR/.env" <<ENV
HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
VLLM_HOST=$VLLM_HOST
VLLM_PORT=8000
SGLANG_HOST=$SGLANG_HOST
SGLANG_PORT=8001
RESULTS_DIR=$APP_DIR/results
ENV
chmod 600 "$APP_DIR/.env"
chown ubuntu:ubuntu "$APP_DIR/.env"

# ---------------------------------------------------------------------------
# 5. Systemd service — FastAPI dashboard
# ---------------------------------------------------------------------------
cat > /etc/systemd/system/benchmark-dashboard.service <<SERVICE
[Unit]
Description=LLM Benchmark FastAPI Dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/.venv/bin/python -m uvicorn dashboard.app:app --host 0.0.0.0 --port 3000
Restart=on-failure
RestartSec=10
User=ubuntu
Group=ubuntu
EnvironmentFile=$APP_DIR/.env
StandardOutput=journal
StandardError=journal
SyslogIdentifier=benchmark-dashboard

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable benchmark-dashboard.service
systemctl start benchmark-dashboard.service

# ---------------------------------------------------------------------------
# 6. Write convenience wrapper scripts
# ---------------------------------------------------------------------------
cat > "$HOME_DIR/run_benchmark.sh" <<'SCRIPT'
#!/usr/bin/env bash
# Quick-start: run all benchmark scenarios and generate report
set -euo pipefail
cd /home/ubuntu/benchmark
source .venv/bin/activate
source .env

echo "=== Health check ==="
python run_experiment.py health \
    --vllm-host "$VLLM_HOST" --sglang-host "$SGLANG_HOST"

for SCENARIO in single_request_latency throughput_ramp long_context_stress \
                prefix_sharing_benefit structured_generation_speed; do
    echo ""
    echo "=== Running: $SCENARIO ==="
    python run_experiment.py compare \
        --scenario "$SCENARIO" \
        --vllm-host "$VLLM_HOST" \
        --sglang-host "$SGLANG_HOST"
done

echo ""
echo "=== Generating HTML report ==="
python run_experiment.py report --output ~/report.html
echo "Report: ~/report.html"
echo "Copy it: scp ubuntu@\$(curl -s ifconfig.me):~/report.html ./report.html"
SCRIPT
chmod +x "$HOME_DIR/run_benchmark.sh"
chown ubuntu:ubuntu "$HOME_DIR/run_benchmark.sh"

# ---------------------------------------------------------------------------
# 7. Write instance info
# ---------------------------------------------------------------------------
cat > "$HOME_DIR/INSTANCE_INFO.txt" <<INFO
===================================================
 LLM Benchmark Dashboard — $PROJECT
===================================================
Role      : Dashboard / CLI (CPU only)
vLLM host : $VLLM_HOST:8000
SGLang    : $SGLANG_HOST:8001
Dashboard : http://localhost:3000  (or your public IP)
App dir   : $APP_DIR
Logs      : journalctl -u benchmark-dashboard -f

Quick commands:
  ~/run_benchmark.sh                     # run all scenarios + report
  systemctl status benchmark-dashboard   # dashboard service status
  source $APP_DIR/.venv/bin/activate && source $APP_DIR/.env
  python run_experiment.py health
  python run_experiment.py run --scenario throughput_ramp --engines vllm,sglang \\
      --vllm-host $VLLM_HOST --sglang-host $SGLANG_HOST
INFO
chown ubuntu:ubuntu "$HOME_DIR/INSTANCE_INFO.txt"

echo "=========================================="
echo " Dashboard bootstrap complete — $(date)"
echo "=========================================="
