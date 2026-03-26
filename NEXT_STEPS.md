# Next Steps to Go Live

## Current State

The benchmarking code is tested (62 unit tests passing) and simplified. The `reports/` directory has real data from a prior run on AWS g5.2xlarge (A10G GPU) covering 5 models and 2 scenarios (18 data points).

**What is missing:** The `results/` directory is empty. The prior benchmark data was collected and aggregated manually outside the CLI pipeline. The report generation commands (`report`, `final-report`) read from `results/` and have never been validated against real pipeline output.

## Existing Data

The snapshot in `reports/benchmark_snapshot_2026-03-22.json` covers:

| Model | Size | Scenarios | Engines |
|---|---|---|---|
| Gemma 2B | 2B | latency, throughput | vLLM, SGLang |
| Phi-3 mini | 3B | latency, throughput | vLLM only |
| Qwen 7B | 7B | latency, throughput | vLLM, SGLang |
| Mistral 7B | 7B | latency, throughput | vLLM, SGLang |
| Gemma 9B | 9B | latency, throughput | vLLM, SGLang |

Key findings:
- vLLM consistently won low-latency single-request TTFT tests
- SGLang matched or beat throughput on mid-sized models (Qwen 7B, Mistral 7B)
- SGLang could not run Phi-3 mini (FlashInfer `head_dim=96` incompatibility)

This data is valid for comparative analysis but was not produced by the CLI pipeline.

## Step 1: Provision GPU infrastructure

```bash
# Option A: AWS (quickest)
./deploy/ec2_deploy.sh --mode single --key my-key-pair --region us-east-1

# Option B: Any machine with an NVIDIA GPU (16+ GB VRAM)
pip install -e ".[dev]"
docker compose up -d vllm
```

## Step 2: Run the core benchmark matrix

Run one engine at a time on a single GPU to avoid VRAM contention:

```bash
# Start vLLM, run all scenarios
docker compose up -d vllm
python run_experiment.py health
python run_experiment.py matrix \
  --scenarios single_request_latency,throughput_ramp,long_context_stress,prefix_sharing_benefit,structured_generation_speed \
  --engines vllm \
  --iterations 2 \
  --cooldown-seconds 120

# Stop vLLM, start SGLang, repeat
docker compose stop vllm
docker compose up -d sglang
python run_experiment.py matrix \
  --scenarios single_request_latency,throughput_ramp,long_context_stress,prefix_sharing_benefit,structured_generation_speed \
  --engines sglang \
  --iterations 2 \
  --cooldown-seconds 120
```

This populates `results/` with per-run JSON files.

## Step 3: Generate reports from real pipeline data

```bash
python run_experiment.py report --output report.html
python run_experiment.py final-report --output final_report.md
```

## Step 4: Validate the dashboard

```bash
python run_experiment.py serve
# Open http://localhost:3000
# Confirm: result list loads, comparison view works, WebSocket metrics stream
```

## Step 5: Run head-to-head comparison (optional, needs both engines accessible)

```bash
python run_experiment.py compare --scenario single_request_latency
```

## Coverage Gaps to Fill

Three scenarios have never been run:
- `long_context_stress` — 4096-token prompts, GPU memory pressure
- `prefix_sharing_benefit` — shared prefix cache warmup measurement
- `structured_generation_speed` — JSON extraction with constrained decode

These are where SGLang's architecture (RadixAttention, constrained decode) should show the biggest advantages over vLLM.

## Live Ready Checklist

- [ ] `results/` contains JSON output from at least 2 scenarios x 2 engines
- [ ] `python run_experiment.py report` produces a valid HTML report from those results
- [ ] `python run_experiment.py final-report` produces a valid markdown summary
- [ ] Dashboard loads and displays results at `/api/results`
- [ ] All unit tests pass (`pytest tests/ -v`)
