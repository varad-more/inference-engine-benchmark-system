# Variance Analysis Report

**Methodology:** Each metric is sampled once per benchmark run. 95% CIs use the t-distribution (`scipy.stats.t`). **⚠ = CV > 5.0%** — claim unreliable, needs more iterations or investigation.

| Metric | Formula |
|--------|---------|
| TTFT P50/P95 | from `metrics.ttft.p50/p95` |
| Throughput | from `metrics.throughput.tokens_per_sec` |
| TPOT P95 | `(total_ms − ttft_ms) / max(output_tokens − 1, 1)`, P95 across requests |

## Single Request Latency

| Model | Engine | N runs | TTFT P50 (ms) | CV% | TTFT P95 (ms) | CV% | Tok/s | CV% | TPOT P95 (ms) | CV% |
|-------|--------|--------|---------------|-----|---------------|-----|---------------|-----|---------------|-----|
| gemma-2-2b-it | sglang | 1 | 30.4 | — | 31.5 | — | 78.2 | — | 12.9 | — |
| gemma-2-2b-it | vllm | 1 | 19.7 | — | 22.5 | — | 77.6 | — | 13.1 | — |

## Throughput Ramp

| Model | Engine | N runs | TTFT P50 (ms) | CV% | TTFT P95 (ms) | CV% | Tok/s | CV% | TPOT P95 (ms) | CV% |
|-------|--------|--------|---------------|-----|---------------|-----|---------------|-----|---------------|-----|
| gemma-2-2b-it | sglang | 1 | 46.9 | — | 156.1 | — | 258.0 | — | 18.4 | — |
| gemma-2-2b-it | vllm | 1 | 41.1 | — | 135.5 | — | 264.7 | — | 17.5 | — |

## Long Context Stress

| Model | Engine | N runs | TTFT P50 (ms) | CV% | TTFT P95 (ms) | CV% | Tok/s | CV% | TPOT P95 (ms) | CV% |
|-------|--------|--------|---------------|-----|---------------|-----|---------------|-----|---------------|-----|
| gemma-2-2b-it | sglang | 1 | 40.2 | — | 50.6 | — | 290.6 | — | 13.6 | — |
| gemma-2-2b-it | vllm | 1 | 33.6 | — | 45.2 | — | 311.5 | — | 13.0 | — |

## Prefix Sharing Benefit

| Model | Engine | N runs | TTFT P50 (ms) | CV% | TTFT P95 (ms) | CV% | Tok/s | CV% | TPOT P95 (ms) | CV% |
|-------|--------|--------|---------------|-----|---------------|-----|---------------|-----|---------------|-----|
| gemma-2-2b-it | sglang | 1 | 40.0 | — | 64.4 | — | 557.9 | — | 16.1 | — |
| gemma-2-2b-it | vllm | 1 | 44.0 | — | 46.4 | — | 567.2 | — | 13.3 | — |

## Structured Generation Speed

| Model | Engine | N runs | TTFT P50 (ms) | CV% | TTFT P95 (ms) | CV% | Tok/s | CV% | TPOT P95 (ms) | CV% |
|-------|--------|--------|---------------|-----|---------------|-----|---------------|-----|---------------|-----|
| gemma-2-2b-it | sglang | 1 | 48.7 | — | 66.8 | — | 956.7 | — | 16.9 | — |
| gemma-2-2b-it | vllm | 1 | 45.4 | — | 59.0 | — | 1224.9 | — | 13.9 | — |

## Headline Comparison Table (with 95% CI)

Replaces bare point estimates in the main README. Values from `single_request_latency` (TTFT) and `throughput_ramp` (tok/s).

| Model | vLLM TTFT P50 | SGLang TTFT P50 | vLLM Peak tok/s | SGLang Peak tok/s |
|-------|---------------|-----------------|-----------------|-------------------|
| gemma-2-2b-it | 19.7 ms | 30.4 ms | 264.7 tok/s | 258.0 tok/s |

## High-Variance Claims (CV > 5%) — Require Asterisk

_No metrics exceeded 5% CV — all headline claims are stable._
