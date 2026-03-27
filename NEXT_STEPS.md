# Next Steps to Go Live

## Goal

Validate the end-to-end pipeline: CLI produces JSON results, report generators consume them, dashboard displays them. Full multi-model benchmarking comes after the pipeline is proven.

## Setup

- **Infrastructure:** Single GPU machine, one engine at a time
- **Validation model:** `Qwen/Qwen2.5-1.5B-Instruct` (CLI default, small and fast)
- **Validation scenarios:** `single_request_latency` and `throughput_ramp`
- **Output folder:** Fresh `results_validation/` to keep it separate from any future production runs

## Existing Data

The `reports/` directory has 18 data points from a prior manual run on AWS g5.2xlarge (A10G GPU) covering 5 models across `single_request_latency` and `throughput_ramp`. This data is valid for reference but was not produced by the CLI pipeline.

---

## Phase 1: Pipeline Validation

### 1. Provision and activate

```bash
# Provision (pick one)
./deploy/ec2_deploy.sh --mode single --key my-key-pair --region us-east-1
# or any machine with an NVIDIA GPU (16+ GB VRAM)

# Activate project environment
conda activate base
pip install -e ".[dev]"
```

### 2. Validate vLLM end-to-end

```bash
docker compose up -d vllm
python run_experiment.py health

python run_experiment.py run \
  --scenario single_request_latency \
  --engines vllm \
  --output-dir results_validation \
  --strict

python run_experiment.py run \
  --scenario throughput_ramp \
  --engines vllm \
  --output-dir results_validation \
  --strict

docker compose stop vllm
```

### 3. Validate SGLang end-to-end

```bash
docker compose up -d sglang
python run_experiment.py health

python run_experiment.py run \
  --scenario single_request_latency \
  --engines sglang \
  --output-dir results_validation \
  --strict

python run_experiment.py run \
  --scenario throughput_ramp \
  --engines sglang \
  --output-dir results_validation \
  --strict

docker compose stop sglang
```

### 4. Verify pipeline output

```bash
ls results_validation/*.json
# Expect 4 JSON files (2 scenarios x 2 engines)
```

### 5. Generate reports from the validation dataset

```bash
python run_experiment.py report \
  --results-dir results_validation \
  --output results_validation/report.html

python run_experiment.py final-report \
  --results-dir results_validation \
  --output results_validation/final_report.md
```

Open `results_validation/report.html` and confirm charts render with real data.

### 6. Validate dashboard against the same dataset

```bash
RESULTS_DIR=results_validation python run_experiment.py serve
# Open http://localhost:3000
# Confirm: /api/results lists the 4 JSON files, result detail loads
```

At this point the pipeline is proven.

---

## Phase 2: Broader Coverage

Run the remaining scenarios that were never tested:

```bash
docker compose up -d vllm
python run_experiment.py matrix \
  --scenarios long_context_stress,prefix_sharing_benefit,structured_generation_speed \
  --engines vllm \
  --output-dir results_validation \
  --iterations 1 \
  --cooldown-seconds 60
docker compose stop vllm

docker compose up -d sglang
python run_experiment.py matrix \
  --scenarios long_context_stress,prefix_sharing_benefit,structured_generation_speed \
  --engines sglang \
  --output-dir results_validation \
  --iterations 1 \
  --cooldown-seconds 60
docker compose stop sglang
```

Regenerate reports to include all 5 scenarios:

```bash
python run_experiment.py report \
  --results-dir results_validation \
  --output results_validation/report.html

python run_experiment.py final-report \
  --results-dir results_validation \
  --output results_validation/final_report.md
```

---

## Phase 3: Full Benchmark Set

Once the pipeline is proven with Qwen 1.5B, repeat with the models from the prior run:

- `google/gemma-2-2b-it`
- `microsoft/Phi-3-mini-4k-instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `google/gemma-2-9b-it`

Use `matrix` with `--iterations 2` and `--cooldown-seconds 300` for stable results. Output to `results/` (the production directory).

The `compare` command requires both engines accessible at the same time. Save it for multi-host setups or machines with enough VRAM for both.

---

## Live Ready Checklist

- [ ] `results_validation/` contains JSON output from 2 scenarios x 2 engines
- [ ] `report` command produces a valid HTML report from that data
- [ ] `final-report` command produces a valid markdown summary from that data
- [ ] Dashboard loads and displays results from the same directory
- [ ] All unit tests pass (`pytest tests/ -v`)
