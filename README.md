# nano-vllm-jax

A JAX/Flax NNX port of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) —
a minimal, readable implementation of continuous-batching LLM inference.

## Status

| Component | Status |
|---|---|
| RMSNorm | ✅ Done |
| RoPE | ✅ Done |
| Linear layers (TP) | ✅ Done |
| Paged KV Cache | ✅ Done |
| Attention (prefill + decode) | ✅ Done |
| LM Head / Embeddings | ✅ Done |
| Sampler | ✅ Done |
| LlamaForCausalLM (Qwen3) | ✅ Done |
| Weight loader (safetensors) | ✅ Done |
| Block manager | ✅ Done |
| Scheduler | ✅ Done |
| Model runner | ✅ Fixed |
| LLM engine | ✅ Done |
| GPU demo (Qwen3-0.6B) | ✅ Fixed |

## Quick Start

### GPU (NVIDIA CUDA 12.x)

```bash
./demo.gpu.sh
```

### Apple Silicon

```bash
./demo.apple.sh
```

## Fixed Bugs

### Bug #1: KV Cache Not Persisted Across JIT Boundary (Critical)

**Symptom:** All output tokens are garbled / random-looking text regardless
of the prompt.

**Root cause:** `PagedKVCache.write()` mutates `self.cache` (an `nnx.Variable`)
inside a `jax.jit`-compiled forward function. JAX JIT compiles **pure
functions** — Python-level object mutations inside a JIT call are **not**
propagated back to the outer scope after the call returns.

Consequence: every decode step reads an all-zero KV cache. Attention over
zero-valued value vectors produces near-zero output regardless of the query,
so every token after the first is effectively random.

**Fix:** Replace `jax.jit` with `nnx.jit` in `model_runner.py`.
`nnx.jit` is NNX-aware: it extracts `nnx.Variable` state before the call,
passes it through XLA as mutable arrays, and writes updates back into the
Python objects after each call.

See [MIGRATION.md](MIGRATION.md#1-jax-functional-purity-and-nnxjit-critical)
for a detailed explanation.

### Bug #2: `last_indices` Not Passed to `lm_head` During Prefill

**Symptom:** In multi-sequence prefill batches, the wrong logit rows are
selected, producing wrong next tokens.

**Fix:** Pass `last_indices` from `_run_prefill` through `_jit_prefill` into
`LlamaForCausalLM.__call__`. `ParallelLMHead` now receives the correct slice
indices and returns shape `[num_seqs, vocab_size]`.

### Bug #3: `seq_lens` Padding with `val=1` in Decode

**Symptom:** Padding sequences in a padded decode batch could attend to
position 0 of the KV cache, potentially contaminating real sequences.

**Fix:** Pad `seq_lens` with `val=0` so padding sequences have no valid
context and all their attention logits become `-inf`.

## Architecture

```
nanovllm_jax/
  config.py              Engine / model configuration
  llm.py                 High-level LLM API
  sampling_params.py     SamplingParams dataclass
  engine/
    llm_engine.py        Ties scheduler + block manager + model runner
    model_runner.py      Builds JAX inputs, calls nnx.jit forward
    scheduler.py         FCFS continuous-batching scheduler
    block_manager.py     Paged KV cache block allocator
    sequence.py          Sequence / SequenceGroup dataclasses
  layers/
    attention.py         PagedKVCache + Attention (prefill & decode)
    rotary_embedding.py  RoPE (GPT-NeoX style)
    layernorm.py         RMSNorm with fused residual
    linear.py            ColumnParallel / RowParallel / QKV linear
    embed_head.py        VocabParallelEmbedding + ParallelLMHead
    activation.py        SiluAndMul
    sampler.py           Greedy / temperature / top-k / top-p
  models/
    llama.py             LlamaForCausalLM (also Qwen2/Qwen3)
  loader/
    weight_loader.py     HuggingFace safetensors weight loader
    model_registry.py    Architecture → class mapping
tests/
  unit/
    layers/              Per-layer unit tests
    engine/              Engine component tests (incl. KV cache regression)
  e2e/                   End-to-end tests
```

## Design Differences from PyTorch nano-vllm

See [MIGRATION.md](MIGRATION.md) for a full list. Key points:

1. **`nnx.jit` instead of `jax.jit`** — required for KV cache mutations to
   persist across call boundaries.
2. **Functional tensor ops** — `tensor.at[i].set(val)` instead of
   `tensor[i] = val`.
3. **No `torch.no_grad()`** — JAX is functional; gradients are opt-in via
   `jax.grad`.
4. **bfloat16 via torch bridge** — `safetensors.numpy` returns bf16 as
   `uint16` bytes; we load via `safetensors.torch` and reinterpret.

## Requirements

- Python 3.10+
- JAX with CUDA support: `pip install jax[cuda12]`
- Flax: `pip install flax`
- safetensors, transformers, huggingface-hub

## Running Tests

```bash
pip install -e .[dev]
pytest tests/unit/ -v
```
