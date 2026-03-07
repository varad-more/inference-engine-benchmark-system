#!/usr/bin/env bash
# =============================================================================
# gpu_instance.sh — Bootstrap script for GPU inference engine instances
#
# Injected variables (filled by Terraform templatefile()):
#   HF_TOKEN    — HuggingFace hub token (may be empty)
#   MODEL_ID    — e.g. "Qwen/Qwen2.5-1.5B-Instruct"
#   GIT_REPO    — Git repo URL to clone
#   PROJECT     — project name tag
#   MODE        — "single" | "vllm_only" | "sglang_only"
#   VLLM_HOST   — host for vLLM (used in single mode)
#   SGLANG_HOST — host for SGLang (used in single mode)
#
# Runs as root during EC2 first-boot. All output is captured to
# /var/log/benchmark-setup.log for debugging.
# =============================================================================

set -euo pipefail
exec > >(tee /var/log/benchmark-setup.log | logger -t benchmark-setup) 2>&1

HF_TOKEN="${HF_TOKEN}"
MODEL_ID="${MODEL_ID}"
GIT_REPO="${GIT_REPO}"
PROJECT="${PROJECT}"
MODE="${MODE}"
HOME_DIR="/home/ubuntu"
APP_DIR="$HOME_DIR/benchmark"

echo "=========================================="
echo " Benchmark bootstrap starting"
echo " MODE=$MODE  MODEL=$MODEL_ID"
echo " $(date)"
echo "=========================================="

# ---------------------------------------------------------------------------
# 1. System update
# ---------------------------------------------------------------------------
apt-get update -y
apt-get upgrade -y --no-install-recommends
apt-get install -y --no-install-recommends \
    git curl wget jq unzip awscli ca-certificates gnupg lsb-release python3-pip

# ---------------------------------------------------------------------------
# 2. Docker (the DL AMI usually has Docker, but ensure compose plugin is present)
# ---------------------------------------------------------------------------
if ! command -v docker &>/dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] \
        https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
        > /etc/apt/sources.list.d/docker.list
    apt-get update -y
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
fi

# Ensure docker-compose CLI plugin is available as "docker compose"
if ! docker compose version &>/dev/null; then
    apt-get install -y docker-compose-plugin
fi

# Add ubuntu user to docker group
usermod -aG docker ubuntu
systemctl enable docker
systemctl start docker

# ---------------------------------------------------------------------------
# 3. NVIDIA Container Toolkit
# ---------------------------------------------------------------------------
echo "Installing NVIDIA Container Toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update -y
apt-get install -y nvidia-container-toolkit

# Configure Docker daemon to use the NVIDIA runtime by default
nvidia-ctk runtime configure --runtime=docker
cat > /etc/docker/daemon.json <<'DAEMON_JSON'
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "50m",
    "max-file": "3"
  }
}
DAEMON_JSON

systemctl restart docker
sleep 5

# Verify GPU is accessible from Docker
echo "Verifying GPU in Docker..."
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi \
    && echo "GPU OK" || echo "WARNING: GPU not accessible in Docker — check driver"

# ---------------------------------------------------------------------------
# 4. Clone the benchmark repository
# ---------------------------------------------------------------------------
echo "Cloning $GIT_REPO → $APP_DIR"
if [ -d "$APP_DIR" ]; then
    git -C "$APP_DIR" pull
else
    git clone "$GIT_REPO" "$APP_DIR"
fi
chown -R ubuntu:ubuntu "$APP_DIR"

# ---------------------------------------------------------------------------
# 5. Write .env file
# ---------------------------------------------------------------------------
cat > "$APP_DIR/.env" <<ENV
HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
RESULTS_DIR=/app/results
ENV
chmod 600 "$APP_DIR/.env"
chown ubuntu:ubuntu "$APP_DIR/.env"

# Create model cache and results directories on the host
mkdir -p "$APP_DIR/model-cache" "$APP_DIR/results"
chown -R ubuntu:ubuntu "$APP_DIR/model-cache" "$APP_DIR/results"

# ---------------------------------------------------------------------------
# 6. Select docker-compose profile based on MODE
# ---------------------------------------------------------------------------
# In "single" mode: start vllm + sglang + dashboard (standard compose)
# In "vllm_only":   start only the vllm service
# In "sglang_only": start only the sglang service

cd "$APP_DIR"

case "$MODE" in
    "single")
        echo "Starting all services (single mode)..."
        sudo -u ubuntu docker compose up -d
        ;;
    "vllm_only")
        echo "Starting vLLM only..."
        sudo -u ubuntu docker compose up -d vllm
        ;;
    "sglang_only")
        echo "Starting SGLang only..."
        sudo -u ubuntu docker compose up -d sglang
        ;;
    *)
        echo "Unknown MODE=$MODE — starting all services"
        sudo -u ubuntu docker compose up -d
        ;;
esac

# ---------------------------------------------------------------------------
# 7. Install Python benchmark CLI dependencies (for running from this host)
# ---------------------------------------------------------------------------
pip3 install --no-cache-dir \
    httpx pydantic structlog fastapi "uvicorn[standard]" \
    websockets typer rich matplotlib jinja2 aiofiles numpy

# ---------------------------------------------------------------------------
# 8. Systemd service for auto-restart on reboot
# ---------------------------------------------------------------------------
cat > /etc/systemd/system/benchmark-compose.service <<SERVICE
[Unit]
Description=LLM Benchmark Docker Compose
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$APP_DIR
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=300
User=ubuntu
Group=ubuntu
EnvironmentFile=$APP_DIR/.env

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable benchmark-compose.service

# ---------------------------------------------------------------------------
# 9. Wait for engines to become healthy (up to 10 min for model download)
# ---------------------------------------------------------------------------
echo "Waiting for engines to pass health checks..."

WAIT_SECS=600
ELAPSED=0
INTERVAL=15

check_health() {
    local port=$1
    curl -sf --max-time 5 "http://localhost:$port/health" > /dev/null 2>&1
}

case "$MODE" in
    "single")
        PORTS=(8000 8001)
        ;;
    "vllm_only")
        PORTS=(8000)
        ;;
    "sglang_only")
        PORTS=(8001)
        ;;
    *)
        PORTS=(8000 8001)
        ;;
esac

for port in "$${PORTS[@]}"; do
    echo "Waiting for port $port..."
    until check_health "$port" || [ "$ELAPSED" -ge "$WAIT_SECS" ]; do
        echo "  Port $port not ready yet ($ELAPSED/$WAIT_SECS s)..."
        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))
    done
    if check_health "$port"; then
        echo "Port $port is healthy."
    else
        echo "WARNING: Port $port did not become healthy in time. Check docker logs."
    fi
done

# ---------------------------------------------------------------------------
# 10. Write instance info file for quick reference
# ---------------------------------------------------------------------------
cat > "$HOME_DIR/INSTANCE_INFO.txt" <<INFO
===================================================
 LLM Benchmark Instance — $PROJECT
===================================================
Mode    : $MODE
Model   : $MODEL_ID
App dir : $APP_DIR
Logs    : /var/log/benchmark-setup.log

Useful commands:
  docker compose ps                  # service status
  docker compose logs -f vllm        # vLLM logs
  docker compose logs -f sglang      # SGLang logs
  curl localhost:8000/health         # vLLM health
  curl localhost:8001/health         # SGLang health
  cd $APP_DIR && python3 run_experiment.py health
INFO
chown ubuntu:ubuntu "$HOME_DIR/INSTANCE_INFO.txt"

echo "=========================================="
echo " Bootstrap complete — $(date)"
echo "=========================================="
