# Single-GPU Operation Guide

This repo supports both vLLM and SGLang, but on a single GPU host (for example AWS `g5.2xlarge` with one A10G) the recommended operating mode is:

1. start **one engine only**,
2. wait for health,
3. run one or more scenarios,
4. stop that engine,
5. cool down if switching engines,
6. start the other engine.

## Why this matters

Running both engines at the same time on a single A10G can cause:

- VRAM contention,
- misleading latency / throughput results,
- unstable startup behavior,
- cache allocation failures,
- and unfair comparisons.

## Recommended compose commands

### vLLM path

```bash
docker compose up -d dashboard vllm
curl http://localhost:8000/health
```

### SGLang path

```bash
docker compose up -d dashboard sglang
curl http://localhost:8001/health
```

### Switch engines cleanly

```bash
docker compose stop vllm sglang
sleep 300   # optional cooldown between engines for long benchmark sessions
```

## Dashboard behavior

The dashboard is intentionally decoupled from engine startup so it can run while:

- only vLLM is running,
- only SGLang is running,
- or both are absent.

That makes it safer for sequential benchmark workflows.
