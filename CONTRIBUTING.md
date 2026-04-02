# Contributing

Thank you for your interest in contributing to the vLLM vs SGLang Inference Engine Benchmark System.

## Ways to Contribute

- **New models** — add benchmark results for models not yet covered
- **New scenarios** — implement additional benchmark scenarios in `benchmarks/scenarios/`
- **New engines** — add client adapters in `engines/` for other inference engines
- **Hardware results** — reproduce runs on different GPUs (A100, H100, RTX 4090, etc.)
- **Bug fixes** — fix issues in the runner, analysis, or dashboard
- **Documentation** — improve setup guides, scenario descriptions, or result explanations

## Getting Started

```bash
# Fork and clone
git clone https://github.com/<your-username>/inference-engine-benchmark-system.git
cd inference-engine-benchmark-system

# Set up environment
conda create -n benchmark python=3.11
conda activate benchmark
pip install -r requirements.txt

# Copy env template
cp .env.example .env
# Add your HuggingFace token for gated models
```

## Adding a New Model

1. Add the model to the matrix in `scripts/run_all_benchmarks.sh`
2. Note any engine-specific flags needed (e.g. `--enforce-eager` for Gemma 3)
3. Run the full benchmark suite and commit the result JSONs under `results/<model-slug>/`
4. Update the tables in `README.md` with the new numbers

## Adding a New Scenario

Scenarios live in `benchmarks/scenarios/`. Each scenario is a Python class that:

1. Inherits from `BenchmarkScenario`
2. Implements `generate_requests()` → list of prompt/config dicts
3. Implements `compute_metrics()` → dict with standardized metric keys

See `benchmarks/scenarios/single_request_latency.py` for a reference implementation.

## Pull Request Guidelines

- One logical change per PR
- Include raw result JSON files when submitting new benchmark runs
- Update `README.md` tables if your PR adds new numbers
- Keep commit messages concise and descriptive

## Reporting Issues

Open a GitHub Issue with:
- GPU model and VRAM
- Docker image versions (vLLM and SGLang)
- The exact command that failed
- Relevant log output (last 50 lines of container logs)

## Code Style

- Python: `ruff` for linting, `black` for formatting
- Shell scripts: `shellcheck` clean
- No new dependencies without discussion

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
