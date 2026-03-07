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
│   ├── scenarios.py            # 5 BenchmarkScenario dataclasses + prompt generators
│   └── runner.py               # BenchmarkRunner (asyncio.gather, metrics polling, JSON output)
│
├── sglang_programs/
│   └── chain_of_thought.py     # 3 @sgl.function programs + vLLM httpx equivalents
│
├── dashboard/
│   └── app.py                  # FastAPI: REST API + WebSocket live metrics stream
│
├── analysis/
│   └── report.py               # HTML report generator (matplotlib CDF/throughput/KV charts)
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

# Custom model
python run_experiment.py run \
  --scenario prefix_sharing_benefit \
  --engines vllm,sglang \
  --model Qwen/Qwen2.5-7B-Instruct
```

### Compare both engines head-to-head

```bash
python run_experiment.py compare --scenario structured_generation_speed
```

### Generate HTML report

```bash
python run_experiment.py report --output report.html
```

### Start the dashboard

```bash
python run_experiment.py serve
# Open http://localhost:3000
```

### List available scenarios

```bash
python run_experiment.py list-scenarios
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

---

## Dashboard API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/results` | List all saved result files |
| `GET` | `/api/results/{id}` | Load a specific result |
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

| Model | Min VRAM | Recommended instance |
|---|---|---|
| Qwen2.5-1.5B | 4 GB | g4dn.xlarge (T4 16 GB) |
| Qwen2.5-7B | 16 GB | g5.2xlarge (A10G 24 GB) |
| Qwen2.5-14B | 30 GB | g5.12xlarge (4× A10G 96 GB) |
| Llama 3.1 8B | 18 GB | g5.2xlarge (A10G 24 GB) |
| Llama 3.1 70B | 140 GB | p4d.24xlarge (8× A100 320 GB) |

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

## License

MIT
