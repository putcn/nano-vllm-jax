# nano-vllm-jax

A JAX/Flax rewrite of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), a minimal vLLM-style LLM inference engine.

> See [MIGRATION_PLAN.md](./MIGRATION_PLAN.md) for the full porting plan and progress tracker.

## Goals
- Drop-in API compatible with `nano-vllm`
- Pure JAX/Flax backend (no PyTorch at inference time)
- JIT-compiled prefill and decode paths
- Paged KV cache via functional `jax.lax` ops
- Multi-device support via `jax.sharding`

## Setup

```bash
# Requires Python 3.11+, CUDA 12
pip install -e ".[dev]"
```

## Run Tests

```bash
pytest tests/unit/          # fast, no model weights needed
pytest tests/e2e/           # requires HuggingFace model access
```

## Quick Example

```python
from nanovllm_jax import LLM, SamplingParams

llm = LLM("meta-llama/Llama-3.2-1B")
outputs = llm.generate(["Hello, my name is"], SamplingParams(max_tokens=50))
print(outputs[0].text)
```

## Status

See [MIGRATION_PLAN.md](./MIGRATION_PLAN.md) for detailed per-module progress.
