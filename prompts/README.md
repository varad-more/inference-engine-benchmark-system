# Prompt Packs

This directory contains reusable prompt corpora for benchmarking inference engines across multiple workload types.

## Files

- `short_chat.jsonl` — short prompts with short outputs; best for TTFT / low-latency testing
- `long_generation.jsonl` — short prompts that request longer outputs; best for decode-heavy throughput testing
- `long_context.jsonl` — long reference-context style prompts; best for KV/cache and context-ingestion stress
- `structured_json.jsonl` — extraction/classification prompts with explicit schema targets
- `reasoning.jsonl` — analysis / multi-step prompts that produce longer thoughtful outputs
- `shared_prefix.json` — one long shared prefix plus many suffix variants for cache-reuse testing

## JSONL format

Each line is a JSON object, for example:

```json
{"id":"short_001","category":"short_chat","prompt":"Explain what a load balancer does.","target_tokens":64}
```

Recommended fields:

- `id` — stable unique identifier
- `category` — workload family
- `prompt` — benchmark prompt text
- `target_tokens` — approximate desired output budget
- `tags` — optional tags like `cloud`, `systems`, `json`, `summarization`

## Shared prefix format

`shared_prefix.json` uses a different structure:

- `id`
- `category`
- `shared_prefix`
- `suffixes` — list of per-request user suffixes
- `target_tokens`

## Usage guidance

A strong benchmark suite should mix:

- low-latency prompts
- longer generation prompts
- long-context prompts
- structured-output prompts
- shared-prefix prompts

This keeps comparisons more representative than reusing only one prompt style.
