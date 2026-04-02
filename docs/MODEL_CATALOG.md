# Model Catalog — 2025/2026 Open Source LLMs

Reference catalog of open source models to benchmark, organized by GPU tier. All VRAM estimates are for bf16/fp16 unless noted.

---

## Tier 1 — A10G 24GB (single GPU, fits today)

These run on the current benchmark hardware without quantization.

| Model | HF ID | Params | VRAM | Token | Spec-dec | Why benchmark |
|---|---|---|---|---|---|---|
| **Qwen3-8B** | `Qwen/Qwen3-8B` | 8B | ~16GB | No | Eagle3+Ngram | Default model; hybrid thinking mode |
| **Llama 3.1 8B** | `meta-llama/Llama-3.1-8B-Instruct` | 8B | ~16GB | Yes | Eagle3+Ngram (both engines) | Best Eagle3 draft support |
| **Gemma 3 4B** | `google/gemma-3-4b-it` | 4B | ~8GB | Yes | Ngram | Multimodal, vision-capable |
| **Gemma 3 12B** | `google/gemma-3-12b-it` | 12B | ~24GB | Yes | Ngram | Tight fit on A10G; multimodal |
| **DeepSeek-R1 Distill 7B** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 7B | ~14GB | No | Ngram | Reasoning model; different latency profile |
| **DeepSeek-R1 Distill 8B** | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 8B | ~16GB | No | Ngram | Llama-based distill; compare vs Qwen-based |
| **Phi-4-mini** | `microsoft/Phi-4-mini-instruct` | 3.8B | ~7GB | No | Ngram | Successor to Phi-3 mini; much better quality |
| **Llama 3.2 3B** | `meta-llama/Llama-3.2-3B-Instruct` | 3B | ~6GB | Yes | Ngram | Throughput ceiling / concurrency baseline |
| **Granite 3.3 8B** | `ibm-granite/granite-3.3-8b-instruct` | 8B | ~16GB | No | Ngram | Apache 2.0; enterprise / coding focus; 128K ctx |
| **SmolLM3 3B** | `HuggingFaceTB/SmolLM3-3B` | 3B | ~6GB | No | Ngram | Beats Llama 3.2 3B at same size; reasoning mode |

---

## Tier 2 — A10G 24GB with AWQ/INT4 quantization

These need quantization to fit. AWQ int4 typically halves VRAM with <5% quality loss.

| Model | HF ID | Params | VRAM (int4) | Token | Notes |
|---|---|---|---|---|---|
| **Phi-4** | `microsoft/phi-4` | 14B | ~8GB | No | Strong reasoning and coding; quantized variants available |
| **DeepSeek-R1 Distill 14B** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | 14B | ~8GB | No | Reasoning; 2× the depth of 7B distill |
| **Gemma 3 27B** | `google/gemma-3-27b-it` | 27B | ~14GB | Yes | Best Gemma quality; multimodal; needs Q4 |
| **Mistral Small 24B** | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | 24B | ~12GB | No | Strong multilingual; needs Q4 |
| **Qwen3-14B** | `Qwen/Qwen3-14B` | 14B | ~8GB | No | Thinking mode; between 8B and 32B quality |

---

## Tier 3 — MoE Models (special VRAM profile)

MoE models load all expert weights into VRAM but only activate a fraction per token. VRAM usage is determined by **total weights**, but compute and throughput behave like the **active parameters**.

| Model | HF ID | Total | Active | VRAM (bf16) | GPU req | Notes |
|---|---|---|---|---|---|---|
| **Qwen3-30B-A3B** | `Qwen/Qwen3-30B-A3B` | 30B | 3B | ~17GB | 1× A10G | Fits on A10G; compute like a 3B but quality of 30B |
| **Llama 4 Scout** | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | ~109B | 17B | ~55GB Q4 | 2-4× A10G or 1× A100 80GB | Native multimodal; 10M ctx; early fusion |
| **Qwen3-235B-A22B** | `Qwen/Qwen3-235B-A22B` | 235B | 22B | ~112GB | 2× A100 80GB | Flagship Qwen3; top open-source quality |
| **DeepSeek-V3** | `deepseek-ai/DeepSeek-V3` | 671B | 37B | Needs 8× H100 | H100 cluster | Highest open-source throughput; not A10G feasible |

> **Key insight on MoE benchmarking:** Qwen3-30B-A3B is particularly interesting — it fits on a single A10G but delivers 30B-class quality. TTFT will be similar to a 3B model (small active compute), but VRAM footprint matches a 17B dense model. This makes it a genuine outlier in the latency/quality curve.

---

## Tier 4 — A100/H100 Multi-GPU (dense models)

| Model | HF ID | Params | VRAM (bf16) | GPU config | TP size | Notes |
|---|---|---|---|---|---|---|
| **Gemma 3 27B** | `google/gemma-3-27b-it` | 27B | ~54GB | 1× A100 80GB | 1 | Multimodal; fits unquantized on A100 80GB |
| **Mistral Small 24B** | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | 24B | ~48GB | 1× A100 80GB | 1 | Strong multilingual; fits unquantized |
| **Qwen3-32B** | `Qwen/Qwen3-32B` | 32B | ~64GB | 1× A100 80GB | 1 | Dense flagship; near-frontier quality |
| **DeepSeek-R1 Distill 32B** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | 32B | ~64GB | 1× A100 80GB | 1 | Best reasoning below 70B |
| **Llama 3.3 70B** | `meta-llama/Llama-3.3-70B-Instruct` | 70B | ~140GB | 2× A100 80GB | 2 | Eagle3 draft available for both engines |
| **Llama 4 Scout** | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | ~109B MoE | 17B active | ~55GB bf16 | 1× A100 80GB | Multimodal MoE; fits on single A100 |
| **Llama 4 Maverick** | `meta-llama/Llama-4-Maverick-17B-128E-Instruct` | ~400B MoE | 17B active | ~200GB | 4× A100 80GB | Most capable Llama 4; 1M ctx |

---

## Reasoning Model Notes

Reasoning models (DeepSeek-R1 distills, Qwen3 thinking mode) behave fundamentally differently from chat models:

| Metric | Chat model | Reasoning model |
|---|---|---|
| TTFT | Low (50–200ms) | Similar — first token of `<think>` chain |
| End-to-end latency | 1–5s | 10–60s (generates 500–3000 thinking tokens) |
| Output tokens | 50–300 | 500–3000+ |
| Throughput (tok/s) | High | Similar — decode is the bottleneck |
| Throughput (req/s) | High | **Very low** — long sequences per request |

**Qwen3-8B thinking mode toggle** — pass via system prompt:
```python
# Enable thinking (default for complex tasks)
{"role": "system", "content": "/think"}

# Disable thinking (fast, chat-style)
{"role": "system", "content": "/no_think"}
```
Benchmarking both modes on the same model reveals the pure reasoning cost with zero model weight change.

---

## Multimodal Notes

Most 2025/2026 models support image input. The benchmark harness currently sends text-only prompts. To benchmark vision capabilities, the `structured_generation_speed` and `single_request_latency` scenarios would need image-bearing prompt packs — a future extension (see `ROADMAP.md` Phase F).

Models with vision that run on A10G today: `gemma-3-4b-it`, `gemma-3-12b-it`, `Phi-4-mini-instruct`, `Llama-4-Scout` (with quantization).

---

## Token Access Reference

| Requires HF token | Models |
|---|---|
| **Yes** (accept license on HF) | All Llama models, Gemma 3 series |
| **No** (fully public) | Qwen3, DeepSeek-R1 distills, Phi-4, Granite, SmolLM3, Mistral |

Token setup:
```bash
# .env
HUGGING_FACE_HUB_TOKEN=hf_YOUR_TOKEN
```

---

## Recommended First-Run Order (A10G 24GB)

1. `Qwen/Qwen3-8B` — baseline, no token, Eagle3 support
2. `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` — reasoning profile, no token
3. `HuggingFaceTB/SmolLM3-3B` — throughput ceiling, no token
4. `microsoft/Phi-4-mini-instruct` — best small model quality, no token
5. `Qwen/Qwen3-30B-A3B` *(MoE)* — fits on A10G, 30B quality at 3B active compute
6. `google/gemma-3-4b-it` — needs token, multimodal
7. `ibm-granite/granite-3.3-8b-instruct` — enterprise/coding, 128K context
