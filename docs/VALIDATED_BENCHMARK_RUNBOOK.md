# Validated Benchmark Runbook

This is the shortest path to reproduce the validated single-A10G benchmark flow.

## Environment

- Host: AWS `g5.2xlarge`
- GPU: NVIDIA A10G 24 GB
- Execution mode: sequential, one engine at a time
- Core scenarios:
  - `single_request_latency`
  - `throughput_ramp`

## Default prep

```bash
cp .env.example .env
mkdir -p model-cache
```

## Run vLLM

```bash
docker compose up -d dashboard vllm
curl http://localhost:8000/health
python run_experiment.py run --scenario single_request_latency --engines vllm --model <MODEL>
python run_experiment.py run --scenario throughput_ramp --engines vllm --model <MODEL>
```

## Switch engines

```bash
docker compose stop vllm sglang
sleep 300
```

## Run SGLang

```bash
docker compose up -d dashboard sglang
curl http://localhost:8001/health
python run_experiment.py run --scenario single_request_latency --engines sglang --model <MODEL>
python run_experiment.py run --scenario throughput_ramp --engines sglang --model <MODEL>
```

## Review results

```bash
ls results/*.json
python run_experiment.py final-report --output final_report.md
```

## Known model-specific exceptions

- `microsoft/Phi-3-mini-4k-instruct`: skip SGLang on this setup due to `unsupported head_dim=96`
- `google/gemma-2-9b-it`: use tuned vLLM settings for A10G fit
