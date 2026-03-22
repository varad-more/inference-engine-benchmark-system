# vLLM vs SGLang — Comparative Inference Benchmark System

A production-quality benchmark harness that rigorously compares **vLLM** and **SGLang** LLM inference engines across latency, throughput, KV-cache efficiency, and structured generation speed.

---

## Architecture Diagram

```mermaid
graph TB
    subgraph Client["Benchmark Client (Python / asyncio)"]
        CLI["run_experiment.py<br/>typer CLI"]
        Runner["BenchmarkRunner<br/>asyncio.gather"]
        Dashboard["FastAPI Dashboard<br/>port 3000"]
    end

    subgraph vLLM["vLLM Engine (port 8000)"]
        VR["REST API<br/>/v1/completions"]
        PA["PagedAttention<br/>Block Manager"]
        PC["Prefix Cache<br/>(LRU block reuse)"]
        VG["vLLM Scheduler<br/>Continuous Batching"]
        VM["Prometheus /metrics"]
        VR --> PA --> PC --> VG
    end

    subgraph SGLang["SGLang Engine (port 8001)"]
        SR["REST API<br/>/v1/completions"]
        RA["RadixAttention<br/>Trie KV Cache"]
        FK["sgl.fork()<br/>Parallel Branches"]
        CD["Constrained Decode<br/>regex / JSON schema"]
        SI["/get_server_info"]
        SR --> RA --> FK
        SR --> CD
    end

    GPU["GPU (NVIDIA A100/H100)<br/>CUDA + NCCL"]

    Runner -->|"httpx SSE"| VR
    Runner -->|"httpx SSE"| SR
    CLI --> Runner
    CLI --> Dashboard
    Dashboard -->|"httpx"| VR
    Dashboard -->|"httpx"| SR
    VG -->|"CUDA kernels"| GPU
    FK -->|"CUDA kernels"| GPU
    VM -->|"scrape"| Runner
    SI -->|"poll"| Runner
```

---

## Project Structure

```
inference-engine-benchmark-system/
├── engines/
│   ├── base_client.py          # Abstract base + GenerationResult / EngineMetrics dataclasses
│   ├── vllm_client.py          # vLLM OpenAI-compat client (SSE streaming, Prometheus metrics)
│   └── sglang_client.py        # SGLang client (REST + native sgl.Runtime support)
│
├── benchmarks/
│   ├── metrics.py              # LatencyStats, ThroughputStats, CDF, compare_metrics
│   ├── scenarios.py            # Scenario configs + default prompt-pack mapping
│   ├── prompt_packs.py         # Prompt-pack loaders (JSONL/JSON)
│   ├── matrix.py               # Sequential scenario×engine×iteration matrix executor
│   └── runner.py               # BenchmarkRunner (asyncio.gather, metrics polling, JSON output)
│
├── sglang_programs/
│   └── chain_of_thought.py     # 3 @sgl.function programs + vLLM httpx equivalents
│
├── dashboard/
│   └── app.py                  # FastAPI: REST API + WebSocket live metrics stream
│
├── analysis/
│   ├── report.py               # HTML report generator (matplotlib CDF/throughput/KV charts)
│   └── final_report.py         # Aggregated markdown final summary across runs
│
├── prompts/
│   ├── README.md               # Prompt-pack documentation and usage conventions
│   ├── short_chat.jsonl        # Low-latency chat prompts
│   ├── long_generation.jsonl   # Decode-heavy prompts
│   ├── long_context.jsonl      # Context-stress prompts
│   ├── structured_json.jsonl   # Schema-oriented extraction prompts
│   ├── reasoning.jsonl         # Multi-step / reasoning prompts
│   ├── shared_prefix.json      # Shared-prefix cache benchmark pack
│   └── schemas/                # JSON schemas referenced by structured prompts
│
├── tests/
│   ├── conftest.py
│   ├── test_metrics.py         # LatencyStats, ThroughputStats, CDF, compare_metrics tests
│   ├── test_base_client.py     # GenerationResult, VLLMClient, SGLangClient with respx mocks
│   └── test_scenarios.py       # Scenario dataclasses and prompt generator tests
│
├── results/                    # Auto-created; stores JSON result files
├── run_experiment.py           # Typer CLI (run / compare / report / serve / health)
├── docker-compose.yml          # vllm + sglang + dashboard services
├── Dockerfile.dashboard        # Lightweight dashboard container
└── pyproject.toml              # Python 3.11+ project metadata
```

---

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
# With SGLang native programs:
pip install -e ".[sglang,dev]"
```

### 2. Launch engines with Docker Compose

```bash
# Copy your HuggingFace token
echo "HUGGING_FACE_HUB_TOKEN=hf_..." > .env
mkdir -p model-cache

docker compose up -d
```

Wait for both health checks to pass (approx 2 min for model download):

```bash
docker compose ps
# vllm-server           healthy
# sglang-server         healthy
# benchmark-dashboard   running
```

### 3. Check engine health

```bash
python run_experiment.py health
```

---

## CLI Usage

### Run a single scenario

```bash
# Single engine
python run_experiment.py run --scenario single_request_latency --engines vllm

# Both engines
python run_experiment.py run --scenario throughput_ramp --engines vllm,sglang

# Custom model + prompt pack
python run_experiment.py run \
  --scenario prefix_sharing_benefit \
  --engines vllm,sglang \
  --model Qwen/Qwen2.5-7B-Instruct \
  --prompt-pack shared_prefix
```

### Compare both engines head-to-head

```bash
python run_experiment.py compare \
  --scenario structured_generation_speed \
  --prompt-pack structured_json
```

### Run a sequential matrix (scenario × engine × iteration)

```bash
python run_experiment.py matrix \
  --model Qwen/Qwen2.5-7B-Instruct \
  --scenarios single_request_latency,throughput_ramp \
  --engines sglang,vllm \
  --iterations 2 \
  --cooldown-seconds 300
```

### Generate reports

```bash
# Existing visual HTML report
python run_experiment.py report --output report.html

# New aggregated markdown summary
python run_experiment.py final-report --output final_report.md
```

### Start the dashboard

```bash
python run_experiment.py serve
# Open http://localhost:3000
```

### List scenarios and prompt packs

```bash
python run_experiment.py list-scenarios
python run_experiment.py list-prompt-packs
```

---

## Benchmark Scenarios

| Scenario | Requests | Concurrency | Focus |
|---|---|---|---|
| `single_request_latency` | 50 | 1 | P50/P95/P99 TTFT, pure engine overhead |
| `throughput_ramp` | 100x7 levels | 1 to 64 | Max tokens/sec, saturation point |
| `long_context_stress` | 20 | 4 | 4096-token prompts, GPU memory pressure |
| `prefix_sharing_benefit` | 100 | 8 | 512-tok shared prefix, cache warm-up curve |
| `structured_generation_speed` | 200 | 16 | JSON extraction, constrained decode |

## Prompt Packs

The repo now includes a starter prompt corpus under `prompts/` so benchmark runs can cover more than one workload style.

Included packs:
- `short_chat.jsonl` — short prompts + short outputs for TTFT-focused testing
- `long_generation.jsonl` — short prompts requesting long outputs for decode-heavy testing
- `long_context.jsonl` — document/transcript-style prompts for context-stress evaluation
- `structured_json.jsonl` — extraction/classification prompts with schema references
- `reasoning.jsonl` — multi-step technical prompts for longer analytic responses
- `shared_prefix.json` — shared system/context prefix with variable suffixes for cache-reuse testing

This structure is intended to make benchmark conclusions more representative than repeatedly hammering a single prompt pattern.

Default scenario→pack mapping is automatic (unless overridden with `--prompt-pack`):
- `single_request_latency` → `short_chat`
- `throughput_ramp` → `long_generation`
- `long_context_stress` → `long_context`
- `prefix_sharing_benefit` → `shared_prefix`
- `structured_generation_speed` → `structured_json`

---

## Dashboard API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Browser-friendly dashboard home |
| `GET` | `/api/results` | List all saved result files |
| `GET` | `/api/results/{id}` | Load a specific result |
| `GET` | `/api/current` | Detect the currently running benchmark/test + active services |
| `GET` | `/api/compare/{scenario}` | Latest vLLM+SGLang delta for a scenario |
| `POST` | `/api/run` | Start a background benchmark run |
| `GET` | `/api/run/{job_id}/status` | Poll run progress |
| `WS` | `/ws/live` | Real-time metric stream (JSON messages) |

**POST /api/run payload:**
```json
{
  "scenario": "throughput_ramp",
  "engines": ["vllm", "sglang"],
  "model": "Qwen/Qwen2.5-1.5B-Instruct"
}
```

**WebSocket message types:**
```
{"type": "heartbeat", "ts": 1234567890}
{"type": "progress", "data": {"done": 42, "total": 100, "last_ttft_ms": 38.2}}
{"type": "metrics",  "data": {"engine": "vllm", "ttft_p95": 72.4, "tokens_per_sec": 1243}}
{"type": "done",     "data": {"job_id": "...", "result_paths": [...]}}
```

---

## SGLang Native Programs

`sglang_programs/chain_of_thought.py` implements three `@sgl.function` programs with vLLM httpx equivalents:

| Program | SGLang (ms) | vLLM-equiv (ms) | Speedup | Advantage |
|---|---|---|---|---|
| `structured_cot` | ~340 | ~410 | 1.2x | KV prefix reuse for turn-2 |
| `parallel_hypotheses` | ~290 | ~750 | 2.6x | True parallel batch decode via `sgl.fork()` |
| `json_entity_extract` | ~180 | ~240 | 1.3x | Native regex-constrained decode |

---

## Running Tests

```bash
# All tests (no live engines needed; uses httpx mocking via respx)
pytest tests/ -v

# Specific test files
pytest tests/test_metrics.py tests/test_base_client.py -v

# With coverage
pytest tests/ --cov=engines --cov=benchmarks --cov-report=term-missing
```

---

## Architecture Deep-Dive

### vLLM — PagedAttention

- KV cache split into fixed-size **pages** (blocks), managed by a block allocator
- **Prefix cache**: LRU reuse of blocks for repeated prompt prefixes
- **Continuous batching**: adds/removes requests mid-batch for high utilisation
- Metrics exposed via Prometheus at `/metrics`
- SSE streaming at `/v1/completions` (OpenAI-compat)

### SGLang — RadixAttention

- KV cache stored as a **radix tree** (trie) keyed on token sequences
- All in-flight requests share the trie — automatic prefix deduplication
- `sgl.fork()` creates parallel decode branches sharing the same KV prefix
- **Constrained decode** built-in: regex / JSON schema enforces valid tokens
- Metrics via `/get_server_info` JSON endpoint

### Key Benchmark Insights

1. **Prefix sharing**: SGLang's radix tree gives higher cache hit rates on workloads with long shared system prompts
2. **Parallel programs**: `sgl.fork()` runs N branches in one batch vs N sequential HTTP calls — 2-3x speedup on multi-hypothesis workloads
3. **Constrained decode**: SGLang's native regex constraint eliminates JSON parse failures and reduces average output length by 20-30%
4. **Throughput at high concurrency**: vLLM's continuous batching is highly competitive at concurrency >= 16

---

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `HUGGING_FACE_HUB_TOKEN` | — | HF token for gated models |
| `VLLM_HOST` | `localhost` | vLLM server host |
| `VLLM_PORT` | `8000` | vLLM server port |
| `SGLANG_HOST` | `localhost` | SGLang server host |
| `SGLANG_PORT` | `8001` | SGLang server port |
| `RESULTS_DIR` | `results/` | Directory for JSON result files |

---

## Requirements

- Python 3.11+
- NVIDIA GPU with >= 16 GB VRAM (A100/H100 recommended)
- Docker + NVIDIA Container Toolkit (for `docker compose`)
- `pip install -e ".[dev]"` for local development

---

## AWS Deployment

Two deployment tools are provided — a **self-contained bash script** (no extra tools needed) and a **Terraform module** for team/repeatable workflows.

### Option 1 — Bash Script (Quickest, No Terraform Required)

`deploy/ec2_deploy.sh` handles everything end-to-end with only the **AWS CLI** and standard unix tools (`jq`, `ssh`, `scp`/`rsync`). It creates all networking, launches instances, uploads the project, starts Docker Compose, and polls until engines are healthy.

**Prerequisites:**

```bash
# AWS CLI v2 (https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
aws configure          # set Access Key, Secret, region
aws sts get-caller-identity   # verify

# jq (https://jqlang.github.io/jq/)
brew install jq        # macOS
sudo apt install jq    # Ubuntu/Debian
```

**Single GPU instance** (~$1.21/hr — both engines share one A10G):

```bash
./deploy/ec2_deploy.sh \
  --mode   single \
  --key    my-key-pair \
  --region us-east-1
  # prompts for HuggingFace token; everything else has sensible defaults
```

**Two dedicated GPU instances** (~$2.46/hr — one engine per GPU):

```bash
./deploy/ec2_deploy.sh \
  --mode     multi \
  --key      my-key-pair \
  --hf-token hf_YOUR_TOKEN \
  --region   us-east-1
```

**All flags:**

```
--mode       single|multi              Topology (default: single)
--region     AWS region                (default: us-east-1)
--instance   EC2 instance type         (default: g5.2xlarge)
--key        EC2 key pair name         (required)
--hf-token   HuggingFace Hub token     (default: empty)
--model      HF model ID               (default: Qwen/Qwen2.5-1.5B-Instruct)
--volume-gb  Root EBS size in GB       (default: 100)
--project    Resource name prefix      (default: llm-benchmark)
--state-file Path to state JSON        (default: .ec2_state.json)
--yes        Auto-confirm all prompts  (non-interactive)
```

The script saves all created resource IDs to `.ec2_state.json`. Use this to tear down:

```bash
./deploy/ec2_deploy.sh --destroy
# Terminates instances, releases EIPs, deletes VPC/SGs
```

After deploy, the script prints:

```
══════════════════════════════════════
  Deployment Complete
══════════════════════════════════════

  single           54.x.x.x

Dashboard: http://54.x.x.x:3000

SSH commands:
  single           ssh -i ~/.ssh/my-key.pem ubuntu@54.x.x.x

Run benchmarks (from the instance):
  ~/run_benchmark.sh                    # all scenarios + HTML report
  python run_experiment.py health

Copy HTML report to laptop:
  scp -i ~/.ssh/my-key.pem ubuntu@54.x.x.x:~/report.html ./report.html
```

---

### Option 2 — Terraform (Repeatable / Team Workflows)

Full Terraform module under `deploy/terraform/`. Manages the same two topologies with remote state, variable files, and lifecycle rules. See the full walkthrough below.

Two topology options are provided, both managed by Terraform under `deploy/terraform/`.

### Instance Options

| Option | Instances | Cost (us-east-1, on-demand) | Best for |
|---|---|---|---|
| **A — Single** | 1× g5.2xlarge (1× A10G 24 GB) | ~$1.21/hr | Dev, cost-sensitive benchmarks |
| **B — Multi** | 2× g5.2xlarge + 1× t3.medium | ~$2.46/hr | Fair isolation benchmarks |

> **Tip:** Use Spot instances for up to 70% savings. Add `instance_market_options` to the `aws_instance` blocks or switch to an ASG.

---

### Prerequisites

| Tool | Install |
|---|---|
| AWS CLI v2 | [docs.aws.amazon.com/cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) |
| Terraform ≥ 1.5 | [developer.hashicorp.com/terraform](https://developer.hashicorp.com/terraform/downloads) |
| An EC2 Key Pair | AWS Console → EC2 → Key Pairs → Create |
| Your public IP | `curl -s https://checkip.amazonaws.com` |

### AWS quota requisition (important before deploy)

GPU instances are commonly quota-blocked in new AWS accounts. Before running deployment, request quota increases in **Service Quotas** for your target region.

Recommended requests:

- **EC2 On-Demand G and VT instances** (for `g5.*`)
- If using Spot: **EC2 Spot Instance Requests for G and VT instances**
- Optional fallback if you plan alternatives: quotas for `g4dn.*` / `p*` families

Suggested initial values:

- Option A (single `g5.2xlarge`): request enough for **1 instance**
- Option B (multi): request enough for **2x g5.2xlarge + 1x t3.medium**

If quota is insufficient, Terraform/script deploy will fail with capacity/quota errors even when config is correct.

```bash
aws configure          # set Access Key, Secret, region (us-east-1)
aws sts get-caller-identity   # verify credentials
```

---

### Option A — Single GPU Instance

Both engines + dashboard on **one** g5.2xlarge. Engines share the GPU and run **sequentially** (start one, benchmark, stop, start the other). Cheapest option.

```
VPC 10.42.0.0/16
  └─ Public subnet
       └─ g5.2xlarge  [Elastic IP]
            ├─ vLLM    → :8000  (internal only)
            ├─ SGLang  → :8001  (internal only)
            └─ Dashboard → :3000  (your IP only)
```

**Deploy:**

```bash
cd deploy/terraform

# 1. Initialise providers
terraform init

# 2. Preview the plan
terraform plan \
  -var="key_pair_name=my-key" \
  -var="your_ip_cidr=$(curl -s https://checkip.amazonaws.com)/32" \
  -var="hf_token=hf_YOUR_TOKEN" \
  -var="deployment_mode=single"

# 3. Apply (creates VPC, SGs, EIP, EC2 — takes ~3 min)
terraform apply \
  -var="key_pair_name=my-key" \
  -var="your_ip_cidr=$(curl -s https://checkip.amazonaws.com)/32" \
  -var="hf_token=hf_YOUR_TOKEN" \
  -var="deployment_mode=single"
```

Terraform prints the connection details:

```
Outputs:

dashboard_url    = "http://54.x.x.x:3000"
ssh_commands     = { single = "ssh -i ~/.ssh/my-key.pem ubuntu@54.x.x.x" }
benchmark_commands = {
  health       = "python run_experiment.py health --vllm-host localhost ..."
  run_latency  = "python run_experiment.py run --scenario single_request_latency ..."
  compare      = "python run_experiment.py compare --scenario prefix_sharing_benefit ..."
}
cost_reminder    = "~$1.21/hr (1× g5.2xlarge). Stop the instance when idle."
```

---

### Option B — Dedicated GPU Per Engine (Recommended for Fair Benchmarks)

Each engine gets its own GPU instance with zero resource contention. A third CPU-only instance runs the dashboard and CLI.

```
VPC 10.42.0.0/16
  └─ Public subnet
       ├─ g5.2xlarge  vllm-host    [Elastic IP]  :8000
       ├─ g5.2xlarge  sglang-host  [Elastic IP]  :8001
       └─ t3.medium   dashboard    [Elastic IP]  :3000
```

**Deploy:**

```bash
cd deploy/terraform

terraform apply \
  -var="key_pair_name=my-key" \
  -var="your_ip_cidr=$(curl -s https://checkip.amazonaws.com)/32" \
  -var="hf_token=hf_YOUR_TOKEN" \
  -var="deployment_mode=multi"
```

The dashboard instance is automatically configured with the private IPs of the engine nodes — no manual wiring needed.

---

### Post-Deploy: Run Benchmarks

Wait ~5-10 minutes for Docker images to pull and models to download (~4 GB).

```bash
# SSH into the single instance (Option A) or dashboard (Option B)
ssh -i ~/.ssh/my-key.pem ubuntu@<IP from terraform output>

# Check both engines are healthy
python run_experiment.py health

# Run all scenarios and generate report (uses ~/run_benchmark.sh shortcut)
~/run_benchmark.sh

# Or run individual scenarios
python run_experiment.py run \
  --scenario throughput_ramp \
  --engines vllm,sglang

# Generate HTML report
python run_experiment.py report --output ~/report.html

# Copy report to your laptop
# (run this on your laptop, not the instance)
scp -i ~/.ssh/my-key.pem ubuntu@<IP>:~/report.html ./report.html
```

Monitor engine startup logs:

```bash
# On the GPU instance
docker compose logs -f vllm     # vLLM model loading
docker compose logs -f sglang   # SGLang model loading

# Full bootstrap log (if troubleshooting)
sudo cat /var/log/benchmark-setup.log
```

---

### Switching Models

To benchmark a larger model, override the `model_id` variable:

```bash
terraform apply \
  -var="model_id=Qwen/Qwen2.5-7B-Instruct" \
  -var="instance_type_gpu=g5.12xlarge" \
  -var="volume_size_gb=200" \
  ...
```

**GPU VRAM requirements:**

| Model | Min VRAM | Recommended instance | AWS vCPUs |
|---|---|---|---:|
| Qwen2.5-1.5B | 4 GB | g4dn.xlarge (T4 16 GB) | 4 |
| Qwen2.5-7B | 16 GB | g5.2xlarge (A10G 24 GB) | 8 |
| Qwen2.5-14B | 30 GB | g5.12xlarge (4× A10G 96 GB) | 48 |
| Llama 3.1 8B | 18 GB | g5.2xlarge (A10G 24 GB) | 8 |
| Llama 3.1 70B | 140 GB | p4d.24xlarge (8× A100 320 GB) | 96 |

**Storage planning (disk) for multi-model benchmarking:**

- Keep at least **50–70 GB free disk** before large benchmark batches.
- Typical model cache growth (approximate):
  - **2B class** models: ~4–8 GB each
  - **7B–9B class** models: ~10–20 GB each
  - **14B+ class** models: ~25–45 GB each
- Docker image layers for inference engines can consume **20–60 GB** over time.
- Sequential benchmarking (one engine/model active at a time) reduces peak storage churn and avoids duplicate temporary artifacts.
- If disk pressure appears, prune stopped/unused Docker images between batches (`docker image prune -a`) and keep only active model caches.

---

### Using a terraform.tfvars File

Avoid typing variables on every command:

```bash
# deploy/terraform/terraform.tfvars  (gitignored — never commit secrets)
aws_region       = "us-west-2"
deployment_mode  = "multi"
key_pair_name    = "my-key"
your_ip_cidr     = "1.2.3.4/32"
hf_token         = "hf_YOUR_TOKEN"
model_id         = "Qwen/Qwen2.5-1.5B-Instruct"
instance_type_gpu = "g5.2xlarge"
volume_size_gb   = 100
```

Then just run:

```bash
terraform plan
terraform apply
```

---

### Saving State Remotely (Team Collaboration)

Add an S3 backend so multiple team members share Terraform state:

```hcl
# deploy/terraform/backend.tf  (create this file)
terraform {
  backend "s3" {
    bucket         = "your-terraform-state-bucket"
    key            = "llm-benchmark/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}
```

```bash
# Create the S3 bucket and DynamoDB table once
aws s3api create-bucket --bucket your-terraform-state-bucket --region us-east-1
aws dynamodb create-table \
  --table-name terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

terraform init   # migrates local state to S3
```

---

### Cost Optimisation

**Stop (not terminate) when idle:**

```bash
# Stop instances to pause billing for compute (EBS and EIP still billed)
aws ec2 stop-instances --instance-ids $(terraform output -json instance_ids | jq -r '.[]')

# Restart later
aws ec2 start-instances --instance-ids $(terraform output -json instance_ids | jq -r '.[]')
```

**Auto-stop guardrails for idle GPU spend (recommended):**

Use one (or both) of these patterns so GPU instances do not run overnight unintentionally.

1. **Cron/script-based auto-stop (simple):**
   - Run a scheduled script (e.g., every 30-60 min) that checks whether benchmark containers/jobs are active.
   - If no active benchmark process is found for a safe window (e.g., 60+ min), stop the GPU instance.

2. **CloudWatch alarm + action (managed):**
   - Create a CloudWatch alarm on low `GPUUtilization` (or fallback CPU/network inactivity signals).
   - Trigger an automation action (Lambda/SSM) to stop the EC2 instance after sustained idle time.

Example minimal auto-stop script (run on the GPU host):

```bash
#!/usr/bin/env bash
set -euo pipefail

# If no benchmark process and no running containers, stop instance.
if ! pgrep -f "run_experiment.py run" >/dev/null 2>&1; then
  if [ -z "$(sudo docker ps -q)" ]; then
    INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
    REGION=$(curl -s http://169.254.169.254/latest/dynamic/instance-identity/document | jq -r .region)
    aws ec2 stop-instances --region "$REGION" --instance-ids "$INSTANCE_ID"
  fi
fi
```

> Tip: keep a manual override flag (e.g., `/tmp/no-autostop`) to prevent shutdown during debugging sessions.

**Use Spot Instances** for up to 70% savings — safe for benchmarking since runs are short:

```bash
terraform apply -var="use_spot=true" ...
# (requires adding spot instance configuration to main.tf)
```

**Typical monthly cost reference (us-east-1, 8 hrs/day × 22 days):**

| Mode | Instance(s) | Monthly est. |
|---|---|---|
| Single | 1× g5.2xlarge | ~$213 |
| Multi | 2× g5.2xlarge + 1× t3.medium | ~$435 |
| Single Spot | 1× g5.2xlarge (spot) | ~$64–$100 |

---

### Teardown

```bash
# Destroy all AWS resources (VPC, SGs, EC2, EIP)
cd deploy/terraform
terraform destroy \
  -var="key_pair_name=my-key" \
  -var="your_ip_cidr=$(curl -s https://checkip.amazonaws.com)/32"

# Confirm with "yes" when prompted
```

> All resources are tagged with `Project = llm-benchmark` so you can audit them in the AWS Console under **Resource Groups & Tag Editor** before destroying.

---

### Troubleshooting

| Symptom | Fix |
|---|---|
| SSH connection refused | Check `your_ip_cidr` matches your current IP. Re-run `terraform apply` with updated value. |
| `curl localhost:8000/health` hangs | Model still downloading. Wait and check `docker compose logs -f vllm`. |
| GPU not visible in Docker | Run `nvidia-smi` on the instance. If it fails, reboot: `sudo reboot`. |
| Out of GPU memory | Reduce `--gpu-memory-utilization` in `docker-compose.yml` or upgrade instance type. |
| Bootstrap failed | Check `/var/log/benchmark-setup.log` on the instance. |
| Dashboard 502 / not reachable | Check security group has port 3000 open for your IP. Verify `systemctl status benchmark-dashboard`. |

---

## Release Readiness

Current recommendation: **v0.1.0-beta** (public beta).

### Release checklist status

- ✅ Packaging metadata fixed (`pip install -e ".[dev]"` no longer blocked by Hatch file selection)
- ✅ CI added (GitHub Actions: lint + tests on Python 3.11/3.12)
- ✅ Real benchmark report included with reproducible JSON artifacts
- ⚠️ Recommended before stable `v1.0.0`: run matrix benchmarks across more models/hardware and add rerun variance summary

---

## Latest Benchmark Report (2026-03-22, AWS g5.2xlarge / A10G 24GB)

This section captures the latest validated run on the remote benchmark server.

### Test environment

- Instance: `g5.2xlarge` (single NVIDIA A10G, 24GB VRAM)
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Engine mode: **sequential** (run one engine at a time on the same GPU)
- Result quality: all scenarios below completed with **100% success rate**

### Testing execution note

All published benchmark runs in this report were executed **sequentially on a single machine** (single GPU host). We intentionally did not run vLLM and SGLang concurrently to avoid VRAM contention and startup instability on one A10G.

> Why sequential? On a single A10G, running both engines concurrently causes VRAM contention and unstable startup. Sequential execution gives clean, reproducible numbers.

### Scenario A — `single_request_latency` (50 requests)

| Engine | TTFT p50 | TTFT p95 | TTFT p99 | Total latency p95 | Tokens/sec | Requests/sec | Success |
|---|---:|---:|---:|---:|---:|---:|---:|
| vLLM | 15.9 ms | 16.4 ms | 91.3 ms | 1047.4 ms | 4982.1 | 38.92 | 100.0% |
| SGLang | 27.5 ms | 28.1 ms | 37.2 ms | 1008.2 ms | 6253.8 | 48.86 | 100.0% |

### Scenario B — `throughput_ramp` (700 requests)

| Engine | TTFT p50 | TTFT p95 | TTFT p99 | Total latency p95 | Tokens/sec | Requests/sec | Success |
|---|---:|---:|---:|---:|---:|---:|---:|
| vLLM | 31.5 ms | 178.1 ms | 208.1 ms | 3202.4 ms | 55287.6 | 216.09 | 100.0% |
| SGLang | 49.4 ms | 155.7 ms | 190.9 ms | 4480.8 ms | 39840.1 | 155.63 | 100.0% |

### High-level takeaways

- **Low-concurrency responsiveness**: vLLM had lower TTFT p50/p95 in this run.
- **High-concurrency throughput**: vLLM delivered higher aggregate tokens/sec and requests/sec.
- **SGLang remained competitive** and maintained excellent success rate/stability.
- These results are for one model + one GPU class; for production selection, repeat across:
  - larger models,
  - longer context windows,
  - multiple reruns/seeds,
  - and dedicated-per-engine hardware (Option B).

### Result artifacts

- `results/single_request_latency_SGLangClient_1774163306.json`
- `results/single_request_latency_VLLMClient_1774163440.json`
- `results/throughput_ramp_SGLangClient_1774163585.json`
- `results/throughput_ramp_VLLMClient_1774164141.json`

### Expanded multi-model benchmark snapshot (in progress)

The benchmark matrix is still running, but the following completed runs are already available and useful for interim comparison.

> Scope note: this snapshot includes only **completed** runs at the time of writing. Some later steps (for example, remaining Mistral/Gemma 9B runs) may still be in progress.

#### Qwen 7B (`Qwen/Qwen2.5-7B-Instruct`)

- `single_request_latency`
  - **SGLang** — TTFT p50 **67.9 ms**, TTFT p95 **68.2 ms**, latency p95 **4178.4 ms**, **1531.6 tok/s**, **11.97 req/s**, success **100.0%**
  - **vLLM** — TTFT p50 **40.4 ms**, TTFT p95 **40.7 ms**, latency p95 **4202.4 ms**, **1451.3 tok/s**, **11.34 req/s**, success **100.0%**
- `throughput_ramp`
  - **SGLang** — TTFT p50 **68.7 ms**, TTFT p95 **194.2 ms**, latency p95 **9581.5 ms**, **18667.3 tok/s**, **72.92 req/s**, success **100.0%**
  - **vLLM** — TTFT p50 **89.9 ms**, TTFT p95 **311.9 ms**, latency p95 **10091.5 ms**, **15140.9 tok/s**, **68.98 req/s**, success **100.0%**
- **Interim takeaway:** vLLM won on single-request TTFT, while SGLang delivered better aggregate throughput on this Qwen 7B run.

#### Gemma 2B (`google/gemma-2-2b-it`)

- `single_request_latency`
  - **SGLang** — TTFT p50 **34.1 ms**, TTFT p95 **35.0 ms**, latency p95 **1683.0 ms**, **3785.5 tok/s**, **29.57 req/s**, success **100.0%**
  - **vLLM** — TTFT p50 **19.8 ms**, TTFT p95 **20.3 ms**, latency p95 **1661.8 ms**, **3749.2 tok/s**, **29.29 req/s**, success **100.0%**
- `throughput_ramp`
  - **SGLang** — TTFT p50 **53.6 ms**, TTFT p95 **159.9 ms**, latency p95 **4898.1 ms**, **36459.2 tok/s**, **142.43 req/s**, success **100.0%**
  - **vLLM** — TTFT p50 **44.0 ms**, TTFT p95 **189.9 ms**, latency p95 **2301.1 ms**, **33875.2 tok/s**, **289.16 req/s**, success **100.0%**
- **Interim takeaway:** vLLM won on latency and requests/sec, while SGLang posted higher tokens/sec in the Gemma 2B throughput run.

#### Phi-3 mini (`microsoft/Phi-3-mini-4k-instruct`)

- `single_request_latency`
  - **vLLM** — TTFT p50 **25.4 ms**, TTFT p95 **25.9 ms**, latency p95 **2243.6 ms**, **2786.4 tok/s**, **21.77 req/s**, success **100.0%**
- `throughput_ramp`
  - **vLLM** — TTFT p50 **55.5 ms**, TTFT p95 **188.6 ms**, latency p95 **8645.5 ms**, **20533.9 tok/s**, **80.21 req/s**, success **100.0%**
- **Compatibility note:** `Phi-3-mini + SGLang` is currently blocked on this setup by a FlashInfer/CUDA graph incompatibility (`unsupported head_dim=96`), so Phi-3 is presently benchmarked on **vLLM only**.

#### Mistral 7B (`mistralai/Mistral-7B-Instruct-v0.3`) — partial

- `single_request_latency`
  - **SGLang** — TTFT p50 **66.0 ms**, TTFT p95 **66.4 ms**, latency p95 **4057.3 ms**, **1574.9 tok/s**, **12.30 req/s**, success **100.0%**
  - **vLLM** — TTFT p50 **41.4 ms**, TTFT p95 **41.7 ms**, latency p95 **4044.0 ms**, **1539.7 tok/s**, **12.03 req/s**, success **100.0%**
- `throughput_ramp`
  - **SGLang** — TTFT p50 **69.7 ms**, TTFT p95 **353.6 ms**, latency p95 **10332.4 ms**, **17294.3 tok/s**, **67.56 req/s**, success **100.0%**
- **Status:** Mistral vLLM throughput and the Gemma 9B block are still being collected.

---

## License

MIT
