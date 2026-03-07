#!/usr/bin/env bash
# =============================================================================
# ec2_deploy.sh — Deploy the vLLM vs SGLang Benchmark System on AWS EC2
#
# No Terraform required — uses only the AWS CLI.
#
# USAGE
#   ./deploy/ec2_deploy.sh [OPTIONS]
#   ./deploy/ec2_deploy.sh --destroy          # tear down all created resources
#
# OPTIONS
#   --mode         single|multi         Deployment topology (default: single)
#   --region       AWS region           (default: us-east-1)
#   --instance     EC2 instance type    (default: g5.2xlarge)
#   --key          EC2 key pair name    (required)
#   --hf-token     HuggingFace token    (default: empty)
#   --model        HF model ID          (default: Qwen/Qwen2.5-1.5B-Instruct)
#   --volume-gb    Root EBS size (GB)   (default: 100)
#   --project      Resource name prefix (default: llm-benchmark)
#   --state-file   Path to state JSON   (default: .ec2_state.json)
#   --destroy      Tear down all resources recorded in the state file
#   --yes          Non-interactive: auto-confirm all prompts
#   -h, --help     Show this help
#
# PREREQUISITES
#   aws   — AWS CLI v2 (configured with credentials + region)
#   jq    — JSON processor
#   ssh   — OpenSSH client
#   scp   — for uploading project files
#
# The script uploads the local project directory to the instance instead of
# cloning from git, so no public repo is required.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
header()  { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════${RESET}"; \
             echo -e "${BOLD}${CYAN}  $*${RESET}"; \
             echo -e "${BOLD}${CYAN}══════════════════════════════════════${RESET}"; }
step()    { echo -e "\n${BOLD}▶ $*${RESET}"; }

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODE="single"
REGION="us-east-1"
INSTANCE_TYPE="g5.2xlarge"
KEY_NAME=""
HF_TOKEN=""
MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
VOLUME_GB=100
PROJECT="llm-benchmark"
STATE_FILE=".ec2_state.json"
DESTROY=false
AUTO_YES=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
  grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \{0,2\}//' | head -40
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)        MODE="$2";          shift 2 ;;
    --region)      REGION="$2";        shift 2 ;;
    --instance)    INSTANCE_TYPE="$2"; shift 2 ;;
    --key)         KEY_NAME="$2";      shift 2 ;;
    --hf-token)    HF_TOKEN="$2";      shift 2 ;;
    --model)       MODEL_ID="$2";      shift 2 ;;
    --volume-gb)   VOLUME_GB="$2";     shift 2 ;;
    --project)     PROJECT="$2";       shift 2 ;;
    --state-file)  STATE_FILE="$2";    shift 2 ;;
    --destroy)     DESTROY=true;       shift   ;;
    --yes|-y)      AUTO_YES=true;      shift   ;;
    -h|--help)     usage ;;
    *) error "Unknown option: $1"; usage ;;
  esac
done

# ---------------------------------------------------------------------------
# Prereq checks
# ---------------------------------------------------------------------------
check_prereqs() {
  step "Checking prerequisites"
  local missing=()
  for cmd in aws jq ssh scp curl; do
    if ! command -v "$cmd" &>/dev/null; then
      missing+=("$cmd")
    fi
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    error "Missing required tools: ${missing[*]}"
    echo "  aws  → https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html"
    echo "  jq   → brew install jq  |  apt install jq"
    exit 1
  fi

  # Verify AWS credentials
  if ! aws sts get-caller-identity --region "$REGION" &>/dev/null; then
    error "AWS credentials not configured or invalid."
    echo "Run: aws configure"
    exit 1
  fi
  local ACCOUNT
  ACCOUNT=$(aws sts get-caller-identity --region "$REGION" --query Account --output text)
  success "AWS account: $ACCOUNT  region: $REGION"
}

# ---------------------------------------------------------------------------
# Confirmation prompt
# ---------------------------------------------------------------------------
confirm() {
  local msg="$1"
  if [[ "$AUTO_YES" == true ]]; then
    info "$msg [auto-yes]"
    return 0
  fi
  echo -e "${YELLOW}$msg${RESET} [y/N] "
  read -r answer
  [[ "$answer" =~ ^[Yy]$ ]]
}

# ---------------------------------------------------------------------------
# State file helpers
# ---------------------------------------------------------------------------
STATE_DATA="{}"

load_state() {
  if [[ -f "$STATE_FILE" ]]; then
    STATE_DATA=$(cat "$STATE_FILE")
    info "Loaded state from $STATE_FILE"
  fi
}

save_state() {
  echo "$STATE_DATA" | jq '.' > "$STATE_FILE"
  info "State saved → $STATE_FILE"
}

state_set() {
  STATE_DATA=$(echo "$STATE_DATA" | jq --arg k "$1" --arg v "$2" '. + {($k): $v}')
}

state_get() {
  echo "$STATE_DATA" | jq -r --arg k "$1" '.[$k] // empty'
}

# ---------------------------------------------------------------------------
# Interactive config (fill in anything not provided via flags)
# ---------------------------------------------------------------------------
interactive_config() {
  step "Configuration"

  # Deployment mode
  if [[ -z "${MODE:-}" ]] || ! [[ "$MODE" =~ ^(single|multi)$ ]]; then
    echo "Deployment mode:"
    echo "  single — one GPU instance (both engines + dashboard), cheapest (~\$1.21/hr)"
    echo "  multi  — dedicated GPU per engine + CPU dashboard (~\$2.46/hr)"
    read -rp "Mode [single]: " input
    MODE="${input:-single}"
  fi

  # Key pair name
  if [[ -z "$KEY_NAME" ]]; then
    echo ""
    info "Available key pairs in $REGION:"
    aws ec2 describe-key-pairs --region "$REGION" \
        --query 'KeyPairs[].KeyName' --output table 2>/dev/null || true
    read -rp "EC2 key pair name: " KEY_NAME
    [[ -z "$KEY_NAME" ]] && { error "Key pair name is required."; exit 1; }
  fi

  # HF token
  if [[ -z "$HF_TOKEN" ]]; then
    read -rp "HuggingFace token (leave blank for public models): " HF_TOKEN
  fi

  echo ""
  echo -e "${BOLD}Deployment summary:${RESET}"
  echo "  Mode          : $MODE"
  echo "  Region        : $REGION"
  echo "  Instance type : $INSTANCE_TYPE"
  echo "  Key pair      : $KEY_NAME"
  echo "  Model         : $MODEL_ID"
  echo "  Volume        : ${VOLUME_GB} GB"
  echo "  Project tag   : $PROJECT"
  echo "  State file    : $STATE_FILE"
  echo ""

  local hourly
  if [[ "$MODE" == "single" ]]; then
    hourly="~\$1.21/hr (1× $INSTANCE_TYPE)"
  else
    hourly="~\$2.46/hr (2× $INSTANCE_TYPE + 1× t3.medium)"
  fi
  warn "Estimated cost: $hourly — remember to destroy when done!"
  echo ""

  confirm "Proceed with deployment?" || { info "Aborted."; exit 0; }
}

# ---------------------------------------------------------------------------
# Find the latest Deep Learning Base GPU AMI (Ubuntu 22.04)
# ---------------------------------------------------------------------------
find_ami() {
  step "Finding Deep Learning GPU AMI (Ubuntu 22.04)"
  local ami
  ami=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters \
      "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
      "Name=state,Values=available" \
      "Name=architecture,Values=x86_64" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text 2>/dev/null)

  if [[ -z "$ami" || "$ami" == "None" ]]; then
    # Fallback: Deep Learning AMI (Ubuntu)
    ami=$(aws ec2 describe-images \
      --region "$REGION" \
      --owners amazon \
      --filters \
        "Name=name,Values=Deep Learning AMI GPU PyTorch*Ubuntu 22.04*" \
        "Name=state,Values=available" \
      --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
      --output text 2>/dev/null)
  fi

  if [[ -z "$ami" || "$ami" == "None" ]]; then
    error "Could not find a Deep Learning GPU AMI in $REGION."
    echo "Try: aws ec2 describe-images --region $REGION --owners amazon \\"
    echo "       --filters 'Name=name,Values=Deep Learning*' --query 'Images[].Name'"
    exit 1
  fi

  local name
  name=$(aws ec2 describe-images --region "$REGION" --image-ids "$ami" \
    --query 'Images[0].Name' --output text 2>/dev/null)
  success "AMI: $ami ($name)"
  AMI_ID="$ami"
}

# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------
MY_CIDR=""

get_my_ip() {
  local ip
  ip=$(curl -sf --max-time 5 https://checkip.amazonaws.com 2>/dev/null \
    || curl -sf --max-time 5 https://api.ipify.org 2>/dev/null \
    || echo "")
  if [[ -z "$ip" ]]; then
    warn "Could not auto-detect public IP. Defaulting to 0.0.0.0/0 (open to all)."
    warn "Update the security group afterward to restrict SSH/dashboard access."
    MY_CIDR="0.0.0.0/0"
  else
    MY_CIDR="${ip}/32"
    info "Restricting SSH/dashboard to your IP: $MY_CIDR"
  fi
}

create_vpc() {
  step "Creating VPC"

  # Reuse existing VPC if recorded in state
  local existing
  existing=$(state_get "vpc_id")
  if [[ -n "$existing" ]]; then
    info "Reusing existing VPC: $existing"
    VPC_ID="$existing"
    return
  fi

  VPC_ID=$(aws ec2 create-vpc \
    --region "$REGION" \
    --cidr-block "10.42.0.0/16" \
    --tag-specifications "ResourceType=vpc,Tags=[{Key=Name,Value=${PROJECT}-vpc},{Key=Project,Value=$PROJECT}]" \
    --query 'Vpc.VpcId' --output text)

  aws ec2 modify-vpc-attribute --region "$REGION" --vpc-id "$VPC_ID" \
    --enable-dns-hostnames
  aws ec2 modify-vpc-attribute --region "$REGION" --vpc-id "$VPC_ID" \
    --enable-dns-support

  success "VPC: $VPC_ID"
  state_set "vpc_id" "$VPC_ID"
  save_state
}

create_subnet() {
  step "Creating public subnet"

  local existing
  existing=$(state_get "subnet_id")
  if [[ -n "$existing" ]]; then
    info "Reusing existing subnet: $existing"
    SUBNET_ID="$existing"
    return
  fi

  local AZ
  AZ=$(aws ec2 describe-availability-zones --region "$REGION" \
    --query 'AvailabilityZones[0].ZoneName' --output text)

  SUBNET_ID=$(aws ec2 create-subnet \
    --region "$REGION" \
    --vpc-id "$VPC_ID" \
    --cidr-block "10.42.1.0/24" \
    --availability-zone "$AZ" \
    --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=${PROJECT}-subnet},{Key=Project,Value=$PROJECT}]" \
    --query 'Subnet.SubnetId' --output text)

  aws ec2 modify-subnet-attribute --region "$REGION" --subnet-id "$SUBNET_ID" \
    --map-public-ip-on-launch

  success "Subnet: $SUBNET_ID ($AZ)"
  state_set "subnet_id" "$SUBNET_ID"
  save_state
}

create_igw() {
  step "Creating Internet Gateway"

  local existing
  existing=$(state_get "igw_id")
  if [[ -n "$existing" ]]; then
    info "Reusing existing IGW: $existing"
    IGW_ID="$existing"
    return
  fi

  IGW_ID=$(aws ec2 create-internet-gateway \
    --region "$REGION" \
    --tag-specifications "ResourceType=internet-gateway,Tags=[{Key=Name,Value=${PROJECT}-igw},{Key=Project,Value=$PROJECT}]" \
    --query 'InternetGateway.InternetGatewayId' --output text)

  aws ec2 attach-internet-gateway \
    --region "$REGION" \
    --internet-gateway-id "$IGW_ID" \
    --vpc-id "$VPC_ID"

  # Route table
  local RTB_ID
  RTB_ID=$(aws ec2 describe-route-tables --region "$REGION" \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query 'RouteTables[0].RouteTableId' --output text)

  aws ec2 create-route --region "$REGION" \
    --route-table-id "$RTB_ID" \
    --destination-cidr-block "0.0.0.0/0" \
    --gateway-id "$IGW_ID" > /dev/null

  aws ec2 associate-route-table --region "$REGION" \
    --route-table-id "$RTB_ID" \
    --subnet-id "$SUBNET_ID" > /dev/null

  success "IGW: $IGW_ID  Route table: $RTB_ID"
  state_set "igw_id" "$IGW_ID"
  state_set "rtb_id" "$RTB_ID"
  save_state
}

create_security_groups() {
  step "Creating security groups"

  local existing_gpu existing_dash
  existing_gpu=$(state_get "sg_gpu_id")
  existing_dash=$(state_get "sg_dashboard_id")

  if [[ -n "$existing_gpu" && -n "$existing_dash" ]]; then
    info "Reusing existing SGs: gpu=$existing_gpu  dashboard=$existing_dash"
    SG_GPU_ID="$existing_gpu"
    SG_DASH_ID="$existing_dash"
    return
  fi

  # GPU SG — SSH from your IP; engine ports internal-only
  SG_GPU_ID=$(aws ec2 create-security-group \
    --region "$REGION" \
    --group-name "${PROJECT}-gpu-sg" \
    --description "GPU inference engine instances — ${PROJECT}" \
    --vpc-id "$VPC_ID" \
    --query 'GroupId' --output text)

  aws ec2 create-tags --region "$REGION" --resources "$SG_GPU_ID" \
    --tags "Key=Name,Value=${PROJECT}-gpu-sg" "Key=Project,Value=$PROJECT"

  # SSH from your IP
  aws ec2 authorize-security-group-ingress --region "$REGION" \
    --group-id "$SG_GPU_ID" \
    --protocol tcp --port 22 --cidr "$MY_CIDR" > /dev/null

  # Engine ports: self-referencing (only from within the SG)
  aws ec2 authorize-security-group-ingress --region "$REGION" \
    --group-id "$SG_GPU_ID" \
    --protocol tcp --port 8000 \
    --source-group "$SG_GPU_ID" > /dev/null

  aws ec2 authorize-security-group-ingress --region "$REGION" \
    --group-id "$SG_GPU_ID" \
    --protocol tcp --port 8001 \
    --source-group "$SG_GPU_ID" > /dev/null

  # Outbound: all
  aws ec2 authorize-security-group-egress --region "$REGION" \
    --group-id "$SG_GPU_ID" \
    --protocol -1 --port -1 --cidr 0.0.0.0/0 > /dev/null 2>&1 || true

  success "GPU security group: $SG_GPU_ID"
  state_set "sg_gpu_id" "$SG_GPU_ID"

  # Dashboard SG — SSH + port 3000 from your IP
  SG_DASH_ID=$(aws ec2 create-security-group \
    --region "$REGION" \
    --group-name "${PROJECT}-dashboard-sg" \
    --description "Dashboard / CLI instance — ${PROJECT}" \
    --vpc-id "$VPC_ID" \
    --query 'GroupId' --output text)

  aws ec2 create-tags --region "$REGION" --resources "$SG_DASH_ID" \
    --tags "Key=Name,Value=${PROJECT}-dashboard-sg" "Key=Project,Value=$PROJECT"

  aws ec2 authorize-security-group-ingress --region "$REGION" \
    --group-id "$SG_DASH_ID" \
    --protocol tcp --port 22 --cidr "$MY_CIDR" > /dev/null

  aws ec2 authorize-security-group-ingress --region "$REGION" \
    --group-id "$SG_DASH_ID" \
    --protocol tcp --port 3000 --cidr "$MY_CIDR" > /dev/null

  aws ec2 authorize-security-group-egress --region "$REGION" \
    --group-id "$SG_DASH_ID" \
    --protocol -1 --port -1 --cidr 0.0.0.0/0 > /dev/null 2>&1 || true

  success "Dashboard security group: $SG_DASH_ID"
  state_set "sg_dashboard_id" "$SG_DASH_ID"

  # Cross-SG: allow dashboard → engine ports
  aws ec2 authorize-security-group-ingress --region "$REGION" \
    --group-id "$SG_GPU_ID" \
    --protocol tcp --port 8000 \
    --source-group "$SG_DASH_ID" > /dev/null 2>&1 || true

  aws ec2 authorize-security-group-ingress --region "$REGION" \
    --group-id "$SG_GPU_ID" \
    --protocol tcp --port 8001 \
    --source-group "$SG_DASH_ID" > /dev/null 2>&1 || true

  save_state
}

# ---------------------------------------------------------------------------
# Generate user-data script (inline — no Terraform dependency)
# ---------------------------------------------------------------------------
generate_user_data() {
  local mode="$1"
  local hf_token="$2"
  local model_id="$3"
  local vllm_host="${4:-localhost}"
  local sglang_host="${5:-localhost}"
  local project="$6"

  cat <<USERDATA
#!/usr/bin/env bash
set -euo pipefail
exec > >(tee /var/log/benchmark-setup.log | logger -t benchmark-setup) 2>&1

MODE="${mode}"
HF_TOKEN="${hf_token}"
MODEL_ID="${model_id}"
VLLM_HOST="${vllm_host}"
SGLANG_HOST="${sglang_host}"
PROJECT="${project}"
HOME_DIR="/home/ubuntu"
APP_DIR="\$HOME_DIR/benchmark"

echo "Bootstrap starting: MODE=\$MODE  MODEL=\$MODEL_ID  \$(date)"

# System packages
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
    git curl wget jq unzip ca-certificates gnupg lsb-release python3-pip

# Docker (DL AMI already has it; ensure compose plugin exists)
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sh
fi
if ! docker compose version &>/dev/null 2>&1; then
    apt-get install -y docker-compose-plugin
fi
usermod -aG docker ubuntu
systemctl enable --now docker

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -y
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker

# Docker daemon config
cat > /etc/docker/daemon.json <<'DAEMON'
{
  "default-runtime": "nvidia",
  "runtimes": { "nvidia": { "path": "nvidia-container-runtime", "runtimeArgs": [] } },
  "log-driver": "json-file",
  "log-opts": { "max-size": "50m", "max-file": "3" }
}
DAEMON
systemctl restart docker
sleep 5

# Python deps (for CLI on this host)
pip3 install --no-cache-dir \
    httpx pydantic structlog fastapi "uvicorn[standard]" \
    websockets typer rich matplotlib jinja2 aiofiles numpy

# Placeholder app dir — files will be uploaded via scp by ec2_deploy.sh
mkdir -p "\$APP_DIR/model-cache" "\$APP_DIR/results"

# Write .env
cat > "\$APP_DIR/.env" <<ENV
HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN
VLLM_HOST=\$VLLM_HOST
VLLM_PORT=8000
SGLANG_HOST=\$SGLANG_HOST
SGLANG_PORT=8001
RESULTS_DIR=/app/results
ENV
chmod 600 "\$APP_DIR/.env"
chown -R ubuntu:ubuntu "\$APP_DIR"

echo "Bootstrap pre-stage done at \$(date)"
USERDATA
}

# ---------------------------------------------------------------------------
# Launch a single EC2 instance
# ---------------------------------------------------------------------------
launch_instance() {
  local name="$1"
  local instance_type="$2"
  local sg_ids="$3"
  local userdata="$4"
  local state_key="$5"

  step "Launching instance: $name ($instance_type)"

  local existing
  existing=$(state_get "$state_key")
  if [[ -n "$existing" ]]; then
    local state
    state=$(aws ec2 describe-instances --region "$REGION" \
      --instance-ids "$existing" \
      --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "not-found")
    if [[ "$state" != "terminated" && "$state" != "not-found" ]]; then
      info "Reusing existing instance $existing (state: $state)"
      echo "$existing"
      return
    fi
  fi

  local tmpfile
  tmpfile=$(mktemp /tmp/userdata.XXXXXX.sh)
  echo "$userdata" > "$tmpfile"

  local id
  id=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$instance_type" \
    --key-name "$KEY_NAME" \
    --subnet-id "$SUBNET_ID" \
    --security-group-ids $sg_ids \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${VOLUME_GB},\"VolumeType\":\"gp3\",\"Encrypted\":true,\"DeleteOnTermination\":true}}]" \
    --user-data "file://${tmpfile}" \
    --metadata-options "HttpTokens=required,HttpEndpoint=enabled" \
    --tag-specifications \
      "ResourceType=instance,Tags=[{Key=Name,Value=${name}},{Key=Project,Value=${PROJECT}}]" \
      "ResourceType=volume,Tags=[{Key=Name,Value=${name}-vol},{Key=Project,Value=${PROJECT}}]" \
    --query 'Instances[0].InstanceId' --output text)

  rm -f "$tmpfile"
  success "Instance launched: $id"
  state_set "$state_key" "$id"
  save_state
  echo "$id"
}

# ---------------------------------------------------------------------------
# Wait for instance to reach 'running' state
# ---------------------------------------------------------------------------
wait_running() {
  local instance_id="$1"
  local name="$2"
  info "Waiting for $name ($instance_id) to reach 'running'..."
  aws ec2 wait instance-running \
    --region "$REGION" \
    --instance-ids "$instance_id"
  success "$name is running."
}

# ---------------------------------------------------------------------------
# Allocate + associate an Elastic IP
# ---------------------------------------------------------------------------
allocate_eip() {
  local instance_id="$1"
  local state_key="$2"

  local existing_alloc existing_ip
  existing_alloc=$(state_get "${state_key}_alloc")
  existing_ip=$(state_get "${state_key}_ip")

  if [[ -n "$existing_ip" ]]; then
    info "Reusing existing EIP: $existing_ip"
    echo "$existing_ip"
    return
  fi

  local alloc_id public_ip
  alloc_id=$(aws ec2 allocate-address \
    --region "$REGION" \
    --domain vpc \
    --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=Project,Value=${PROJECT}}]" \
    --query 'AllocationId' --output text)

  aws ec2 associate-address \
    --region "$REGION" \
    --instance-id "$instance_id" \
    --allocation-id "$alloc_id" > /dev/null

  public_ip=$(aws ec2 describe-addresses \
    --region "$REGION" \
    --allocation-ids "$alloc_id" \
    --query 'Addresses[0].PublicIp' --output text)

  success "EIP: $public_ip (allocation: $alloc_id)"
  state_set "${state_key}_alloc" "$alloc_id"
  state_set "${state_key}_ip" "$public_ip"
  save_state
  echo "$public_ip"
}

# ---------------------------------------------------------------------------
# Wait for SSH to become available on an instance
# ---------------------------------------------------------------------------
wait_ssh() {
  local ip="$1"
  local key_file="$2"
  info "Waiting for SSH on $ip (may take 2-5 min while instance initialises)..."
  local attempts=0
  local max=60
  while ! ssh -o StrictHostKeyChecking=no \
              -o ConnectTimeout=5 \
              -o BatchMode=yes \
              -i "$key_file" \
              "ubuntu@${ip}" "echo ready" &>/dev/null; do
    attempts=$((attempts + 1))
    if [[ $attempts -ge $max ]]; then
      error "SSH did not become available after $((max * 10))s. Check instance logs."
      exit 1
    fi
    echo -n "."
    sleep 10
  done
  echo ""
  success "SSH ready on $ip"
}

# ---------------------------------------------------------------------------
# Upload project files to instance (rsync preferred, scp fallback)
# ---------------------------------------------------------------------------
upload_project() {
  local ip="$1"
  local key_file="$2"
  local remote_dir="/home/ubuntu/benchmark"

  step "Uploading project files to $ip:$remote_dir"

  # Determine project root (script is in deploy/)
  local SCRIPT_DIR PROJECT_ROOT
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

  # Create remote directory
  ssh -o StrictHostKeyChecking=no -i "$key_file" "ubuntu@${ip}" \
    "mkdir -p $remote_dir"

  if command -v rsync &>/dev/null; then
    rsync -avz --progress \
      -e "ssh -o StrictHostKeyChecking=no -i $key_file" \
      --exclude='.git' \
      --exclude='__pycache__' \
      --exclude='*.pyc' \
      --exclude='.venv' \
      --exclude='model-cache' \
      --exclude='results' \
      --exclude='.ec2_state.json' \
      --exclude='deploy/terraform/.terraform' \
      "${PROJECT_ROOT}/" \
      "ubuntu@${ip}:${remote_dir}/"
  else
    warn "rsync not found; using scp (slower)"
    scp -o StrictHostKeyChecking=no -i "$key_file" -r \
      "${PROJECT_ROOT}/." \
      "ubuntu@${ip}:${remote_dir}/"
  fi

  success "Project files uploaded."
}

# ---------------------------------------------------------------------------
# Start Docker Compose services on the instance
# ---------------------------------------------------------------------------
start_services() {
  local ip="$1"
  local key_file="$2"
  local mode="$3"     # single | vllm_only | sglang_only

  step "Starting Docker Compose services on $ip (mode: $mode)"

  local compose_cmd
  case "$mode" in
    "single")     compose_cmd="docker compose up -d" ;;
    "vllm_only")  compose_cmd="docker compose up -d vllm" ;;
    "sglang_only") compose_cmd="docker compose up -d sglang" ;;
    "dashboard")  compose_cmd="docker compose up -d dashboard" ;;
    *)            compose_cmd="docker compose up -d" ;;
  esac

  ssh -o StrictHostKeyChecking=no -i "$key_file" "ubuntu@${ip}" \
    "cd /home/ubuntu/benchmark && $compose_cmd"

  success "Services started."
}

# ---------------------------------------------------------------------------
# Poll health endpoint until healthy or timeout
# ---------------------------------------------------------------------------
wait_healthy() {
  local ip="$1"
  local port="$2"
  local service_name="$3"
  local timeout_sec="${4:-600}"
  local interval=15
  local elapsed=0

  info "Waiting for $service_name to be healthy (http://${ip}:${port}/health, timeout ${timeout_sec}s)..."
  info "Note: first boot downloads Docker images + model weights (~4 GB). This takes ~5-10 min."

  while ! curl -sf --max-time 5 "http://${ip}:${port}/health" &>/dev/null; do
    if [[ $elapsed -ge $timeout_sec ]]; then
      warn "$service_name did not become healthy within ${timeout_sec}s."
      warn "Check logs: ssh -i <key> ubuntu@${ip} 'docker compose logs -f'"
      return 1
    fi
    printf "  [%3ds] %s not ready yet...\r" "$elapsed" "$service_name"
    sleep $interval
    elapsed=$((elapsed + interval))
  done
  echo ""
  success "$service_name is healthy at http://${ip}:${port}"
}

# ---------------------------------------------------------------------------
# Resolve key file path from key pair name
# ---------------------------------------------------------------------------
resolve_key_file() {
  local key_name="$1"
  local candidates=(
    "${HOME}/.ssh/${key_name}.pem"
    "${HOME}/.ssh/${key_name}"
    "./${key_name}.pem"
    "./${key_name}"
  )
  for f in "${candidates[@]}"; do
    if [[ -f "$f" ]]; then
      echo "$f"
      return 0
    fi
  done
  # Ask user
  warn "Could not find key file for '$key_name'."
  read -rp "Enter full path to the .pem file: " key_path
  if [[ -f "$key_path" ]]; then
    echo "$key_path"
  else
    error "Key file not found: $key_path"
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# Print final connection summary
# ---------------------------------------------------------------------------
print_summary() {
  local key_file="$1"
  shift
  # Remaining args: name=ip pairs
  echo ""
  header "Deployment Complete"
  echo ""
  for pair in "$@"; do
    local name="${pair%%=*}"
    local ip="${pair##*=}"
    printf "  %-16s %s\n" "$name" "$ip"
  done
  echo ""

  local dashboard_ip
  if [[ "$MODE" == "single" ]]; then
    dashboard_ip=$(state_get "single_ip")
  else
    dashboard_ip=$(state_get "dashboard_ip")
  fi

  echo -e "${BOLD}Dashboard:${RESET} http://${dashboard_ip}:3000"
  echo ""
  echo -e "${BOLD}SSH commands:${RESET}"
  for pair in "$@"; do
    local name="${pair%%=*}"
    local ip="${pair##*=}"
    printf "  %-16s ssh -i %s ubuntu@%s\n" "$name" "$key_file" "$ip"
  done
  echo ""
  echo -e "${BOLD}Run benchmarks (from the instance):${RESET}"
  echo "  ~/run_benchmark.sh                    # all scenarios + HTML report"
  echo "  python run_experiment.py health"
  echo "  python run_experiment.py run --scenario throughput_ramp --engines vllm,sglang"
  echo ""
  echo -e "${BOLD}Copy HTML report to laptop:${RESET}"
  echo "  scp -i $key_file ubuntu@${dashboard_ip}:~/report.html ./report.html"
  echo ""
  echo -e "${BOLD}Teardown (stops billing):${RESET}"
  echo "  ./deploy/ec2_deploy.sh --destroy --state-file $STATE_FILE"
  echo ""
  warn "Instances cost money while running. Stop them when done!"
}

# ---------------------------------------------------------------------------
# DESTROY MODE — tear down all resources in state file
# ---------------------------------------------------------------------------
destroy() {
  header "Destroying AWS resources"
  load_state

  if [[ "$STATE_DATA" == "{}" ]]; then
    warn "State file is empty or missing: $STATE_FILE"
    warn "Nothing to destroy. Delete resources manually in the AWS Console."
    exit 0
  fi

  confirm "This will TERMINATE instances and DELETE all resources. Continue?" || {
    info "Aborted."
    exit 0
  }

  # Terminate instances
  for key in "single_instance_id" "vllm_instance_id" "sglang_instance_id" "dashboard_instance_id"; do
    local id
    id=$(state_get "$key")
    if [[ -n "$id" ]]; then
      info "Terminating instance: $id ($key)"
      aws ec2 terminate-instances --region "$REGION" \
        --instance-ids "$id" > /dev/null 2>&1 || warn "Could not terminate $id"
    fi
  done

  # Wait for termination before releasing EIPs
  info "Waiting for instances to terminate..."
  sleep 30

  # Release Elastic IPs
  for key in "single_alloc" "vllm_alloc" "sglang_alloc" "dashboard_alloc"; do
    local alloc
    alloc=$(state_get "$key")
    if [[ -n "$alloc" ]]; then
      info "Releasing EIP: $alloc"
      aws ec2 release-address --region "$REGION" \
        --allocation-id "$alloc" 2>/dev/null || warn "Could not release EIP $alloc"
    fi
  done

  # Delete security groups (retry — may fail if instances still terminating)
  sleep 15
  for key in "sg_gpu_id" "sg_dashboard_id"; do
    local sg
    sg=$(state_get "$key")
    if [[ -n "$sg" ]]; then
      info "Deleting security group: $sg"
      aws ec2 delete-security-group --region "$REGION" \
        --group-id "$sg" 2>/dev/null || warn "Could not delete SG $sg (may still have dependencies)"
    fi
  done

  # Detach + delete IGW
  local igw vpc
  igw=$(state_get "igw_id")
  vpc=$(state_get "vpc_id")
  if [[ -n "$igw" && -n "$vpc" ]]; then
    info "Detaching IGW: $igw"
    aws ec2 detach-internet-gateway --region "$REGION" \
      --internet-gateway-id "$igw" --vpc-id "$vpc" 2>/dev/null || true
    aws ec2 delete-internet-gateway --region "$REGION" \
      --internet-gateway-id "$igw" 2>/dev/null || warn "Could not delete IGW $igw"
  fi

  # Delete subnet
  local subnet
  subnet=$(state_get "subnet_id")
  if [[ -n "$subnet" ]]; then
    info "Deleting subnet: $subnet"
    aws ec2 delete-subnet --region "$REGION" \
      --subnet-id "$subnet" 2>/dev/null || warn "Could not delete subnet $subnet"
  fi

  # Delete VPC
  if [[ -n "$vpc" ]]; then
    info "Deleting VPC: $vpc"
    aws ec2 delete-vpc --region "$REGION" \
      --vpc-id "$vpc" 2>/dev/null || warn "Could not delete VPC $vpc"
  fi

  # Clear state file
  echo "{}" > "$STATE_FILE"
  success "Teardown complete. State file reset: $STATE_FILE"
}

# ---------------------------------------------------------------------------
# SINGLE MODE — one g5.2xlarge runs all services
# ---------------------------------------------------------------------------
deploy_single() {
  header "Deploying: Single Instance Mode"
  info "One $INSTANCE_TYPE hosts vLLM (8000) + SGLang (8001) + Dashboard (3000)"

  create_vpc
  create_subnet
  create_igw
  create_security_groups

  local userdata
  userdata=$(generate_user_data "single" "$HF_TOKEN" "$MODEL_ID" "localhost" "localhost" "$PROJECT")

  local instance_id
  instance_id=$(launch_instance \
    "${PROJECT}-single" "$INSTANCE_TYPE" \
    "$SG_GPU_ID $SG_DASH_ID" \
    "$userdata" "single_instance_id")

  wait_running "$instance_id" "${PROJECT}-single"

  local public_ip
  public_ip=$(allocate_eip "$instance_id" "single")
  state_set "single_ip" "$public_ip"
  save_state

  local KEY_FILE
  KEY_FILE=$(resolve_key_file "$KEY_NAME")

  wait_ssh "$public_ip" "$KEY_FILE"
  upload_project "$public_ip" "$KEY_FILE"
  start_services "$public_ip" "$KEY_FILE" "single"

  # Dashboard run_benchmark.sh convenience script
  ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" "ubuntu@${public_ip}" \
    "cat > ~/run_benchmark.sh" <<'RUNSCRIPT'
#!/usr/bin/env bash
set -euo pipefail
cd ~/benchmark
source .env 2>/dev/null || true
echo "=== Health check ===" && python3 run_experiment.py health
for S in single_request_latency throughput_ramp long_context_stress \
          prefix_sharing_benefit structured_generation_speed; do
  echo "" && echo "=== $S ===" && python3 run_experiment.py compare --scenario "$S"
done
echo "" && echo "=== Generating report ===" && python3 run_experiment.py report --output ~/report.html
echo "Report: ~/report.html"
RUNSCRIPT

  ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" "ubuntu@${public_ip}" \
    "chmod +x ~/run_benchmark.sh"

  # Health checks
  wait_healthy "$public_ip" 8000 "vLLM"    600
  wait_healthy "$public_ip" 8001 "SGLang"  600
  wait_healthy "$public_ip" 3000 "Dashboard" 120

  print_summary "$KEY_FILE" "single=${public_ip}"
}

# ---------------------------------------------------------------------------
# MULTI MODE — dedicated GPU per engine + CPU dashboard
# ---------------------------------------------------------------------------
deploy_multi() {
  header "Deploying: Multi-Instance Mode"
  info "vLLM and SGLang each get a dedicated $INSTANCE_TYPE"
  info "Dashboard runs on a t3.medium"

  create_vpc
  create_subnet
  create_igw
  create_security_groups

  # Launch vLLM instance
  local vllm_ud
  vllm_ud=$(generate_user_data "vllm_only" "$HF_TOKEN" "$MODEL_ID" "localhost" "" "$PROJECT")
  local vllm_id
  vllm_id=$(launch_instance "${PROJECT}-vllm" "$INSTANCE_TYPE" \
    "$SG_GPU_ID" "$vllm_ud" "vllm_instance_id")

  # Launch SGLang instance
  local sg_ud
  sg_ud=$(generate_user_data "sglang_only" "$HF_TOKEN" "$MODEL_ID" "" "localhost" "$PROJECT")
  local sglang_id
  sglang_id=$(launch_instance "${PROJECT}-sglang" "$INSTANCE_TYPE" \
    "$SG_GPU_ID" "$sg_ud" "sglang_instance_id")

  # Wait for both GPU instances to run before launching dashboard
  # (we need their private IPs for the dashboard config)
  wait_running "$vllm_id"   "${PROJECT}-vllm"
  wait_running "$sglang_id" "${PROJECT}-sglang"

  local vllm_private sglang_private
  vllm_private=$(aws ec2 describe-instances --region "$REGION" \
    --instance-ids "$vllm_id" \
    --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text)
  sglang_private=$(aws ec2 describe-instances --region "$REGION" \
    --instance-ids "$sglang_id" \
    --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text)

  info "vLLM private IP: $vllm_private"
  info "SGLang private IP: $sglang_private"
  state_set "vllm_private_ip" "$vllm_private"
  state_set "sglang_private_ip" "$sglang_private"
  save_state

  # Launch dashboard on t3.medium
  local dash_ud
  dash_ud=$(generate_user_data "dashboard" "$HF_TOKEN" "$MODEL_ID" \
    "$vllm_private" "$sglang_private" "$PROJECT")
  local dash_id
  dash_id=$(launch_instance "${PROJECT}-dashboard" "t3.medium" \
    "$SG_GPU_ID $SG_DASH_ID" "$dash_ud" "dashboard_instance_id")

  wait_running "$dash_id" "${PROJECT}-dashboard"

  # Allocate EIPs
  local vllm_ip sglang_ip dash_ip
  vllm_ip=$(allocate_eip "$vllm_id" "vllm")
  sglang_ip=$(allocate_eip "$sglang_id" "sglang")
  dash_ip=$(allocate_eip "$dash_id" "dashboard")
  state_set "vllm_ip"      "$vllm_ip"
  state_set "sglang_ip"    "$sglang_ip"
  state_set "dashboard_ip" "$dash_ip"
  save_state

  local KEY_FILE
  KEY_FILE=$(resolve_key_file "$KEY_NAME")

  # Upload + start services on each node in parallel
  for pair in "vllm_only:$vllm_ip" "sglang_only:$sglang_ip" "dashboard:$dash_ip"; do
    local mode="${pair%%:*}"
    local ip="${pair##*:}"
    wait_ssh "$ip" "$KEY_FILE"
    upload_project "$ip" "$KEY_FILE"
    start_services "$ip" "$KEY_FILE" "$mode"
  done

  # Drop run_benchmark.sh on dashboard
  ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" "ubuntu@${dash_ip}" \
    "cat > ~/run_benchmark.sh" <<RUNSCRIPT
#!/usr/bin/env bash
set -euo pipefail
cd ~/benchmark
source .env 2>/dev/null || true
VLLM_HOST="$vllm_private"
SGLANG_HOST="$sglang_private"
echo "=== Health check ===" && python3 run_experiment.py health \
  --vllm-host "\$VLLM_HOST" --sglang-host "\$SGLANG_HOST"
for S in single_request_latency throughput_ramp long_context_stress \
          prefix_sharing_benefit structured_generation_speed; do
  echo "" && echo "=== \$S ===" && python3 run_experiment.py compare \
    --scenario "\$S" --vllm-host "\$VLLM_HOST" --sglang-host "\$SGLANG_HOST"
done
echo "" && python3 run_experiment.py report --output ~/report.html
echo "Report: ~/report.html"
RUNSCRIPT

  ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" "ubuntu@${dash_ip}" \
    "chmod +x ~/run_benchmark.sh"

  # Health checks
  wait_healthy "$vllm_ip"   8000 "vLLM"     600
  wait_healthy "$sglang_ip" 8001 "SGLang"   600
  wait_healthy "$dash_ip"   3000 "Dashboard" 120

  print_summary "$KEY_FILE" \
    "vllm=${vllm_ip}" "sglang=${sglang_ip}" "dashboard=${dash_ip}"
}

# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
main() {
  header "vLLM vs SGLang — EC2 Deployment Script"

  if [[ "$DESTROY" == true ]]; then
    load_state
    destroy
    exit 0
  fi

  check_prereqs
  load_state
  interactive_config
  get_my_ip
  find_ami

  case "$MODE" in
    single) deploy_single ;;
    multi)  deploy_multi  ;;
    *)
      error "Unknown mode: $MODE. Use 'single' or 'multi'."
      exit 1 ;;
  esac
}

main "$@"
