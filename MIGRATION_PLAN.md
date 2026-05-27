# nano-vllm-jax Migration Plan
## PyTorch → JAX / Flax Rewrite

**Source**: [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)  
**Target**: [putcn/nano-vllm-jax](https://github.com/putcn/nano-vllm-jax)  
**Started**: 2026-05-27  
**Last Updated**: 2026-05-27

---

## Overview

This project rewrites the `nano-vllm` LLM inference engine — originally built on PyTorch — using [JAX](https://github.com/google/jax) + [Flax](https://github.com/google/flax) (NNX) as the deep learning backend. The goal is to leverage JAX's functional programming model, `jit` compilation, `vmap`/`pmap` for multi-device parallelism, and XLA-based kernel fusion for high-performance LLM inference.

### Key Principles
- **Functional purity**: Replace stateful `nn.Module` (PyTorch) with `flax.nnx` or pure JAX functions.
- **Explicit PRNG**: All randomness via `jax.random.PRNGKey`.
- **JIT-first**: Every forward pass must be `jax.jit`-compilable.
- **Test-driven**: Each module must have passing unit tests AND end-to-end tests before merging.
- **Numerical equivalence**: JAX outputs must match PyTorch reference outputs within tolerance (atol=1e-3, rtol=1e-3) for deterministic ops.

---

## Architecture Map: PyTorch → JAX

| Component | PyTorch Source File | JAX Target File | Status |
|-----------|-------------------|-----------------|--------|
| Config | `nanovllm/config.py` | `nanovllm_jax/config.py` | ✅ Done |
| Sampling Params | `nanovllm/sampling_params.py` | `nanovllm_jax/sampling_params.py` | ✅ Done |
| Activation (SiLU/GeGLU) | `nanovllm/layers/activation.py` | `nanovllm_jax/layers/activation.py` | ✅ Done |
| RMSNorm / LayerNorm | `nanovllm/layers/layernorm.py` | `nanovllm_jax/layers/layernorm.py` | ✅ Done |
| Rotary Embedding (RoPE) | `nanovllm/layers/rotary_embedding.py` | `nanovllm_jax/layers/rotary_embedding.py` | ✅ Done |
| Linear (column/row parallel) | `nanovllm/layers/linear.py` | `nanovllm_jax/layers/linear.py` | ⬜ Not Started |
| Embed + LM Head | `nanovllm/layers/embed_head.py` | `nanovllm_jax/layers/embed_head.py` | ⬜ Not Started |
| Attention (PagedKV) | `nanovllm/layers/attention.py` | `nanovllm_jax/layers/attention.py` | ⬜ Not Started |
| Sampler | `nanovllm/layers/sampler.py` | `nanovllm_jax/layers/sampler.py` | ⬜ Not Started |
| Sequence | `nanovllm/engine/sequence.py` | `nanovllm_jax/engine/sequence.py` | ⬜ Not Started |
| Block Manager (PagedAttn) | `nanovllm/engine/block_manager.py` | `nanovllm_jax/engine/block_manager.py` | ⬜ Not Started |
| Scheduler | `nanovllm/engine/scheduler.py` | `nanovllm_jax/engine/scheduler.py` | ⬜ Not Started |
| Model Runner | `nanovllm/engine/model_runner.py` | `nanovllm_jax/engine/model_runner.py` | ⬜ Not Started |
| LLM Engine | `nanovllm/engine/llm_engine.py` | `nanovllm_jax/engine/llm_engine.py` | ⬜ Not Started |
| LLaMA Model | `nanovllm/models/llama.py` | `nanovllm_jax/models/llama.py` | ⬜ Not Started |
| LLM Entry Point | `nanovllm/llm.py` | `nanovllm_jax/llm.py` | ⬜ Not Started |

**Status Legend**: ⬜ Not Started | 🔄 In Progress | 🧪 Testing | ✅ Done | ❌ Blocked

---

## Phase Plan

### Phase 0 — Project Setup ✅
**Goal**: Repository structure, CI, dependencies, tooling.

- [x] `pyproject.toml` with JAX, Flax, optax, pytest, chex dependencies
- [x] `nanovllm_jax/` package skeleton with `__init__.py` files
- [x] `tests/` directory structure mirroring source
- [x] GitHub Actions CI workflow (pytest on push/PR)
- [x] `README.md` with setup instructions
- [x] `.gitignore`

**Tests**: `tests/unit/test_smoke.py` — import + config + sampling_params validation. ✅

---

### Phase 1 — Stateless Utility Layers ✅
**Goal**: Port all pure-math layers that have no state or minimal learned params.

#### 1.1 Config & SamplingParams ✅
- Port `config.py` and `sampling_params.py` as plain Python dataclasses.
- **Unit tests**: Instantiation, field validation. ✅

#### 1.2 Activation Functions ✅
- `silu_and_mul` pure function + `SiluAndMul` NNX wrapper.
- **Tests**: 4 shapes, 3 seeds numerical match vs PyTorch, zero/large edge cases, bfloat16, jit. ✅
- **Tolerance**: atol=1e-5 (float32), atol=1e-2 (bfloat16).

#### 1.3 RMSNorm ✅
- `RMSNorm(nnx.Module)` with fused `add_rms` path.
- **Tests**: 4 shapes, 3 seeds numerical match, residual value correctness, zero input, unit-variance, jit. ✅
- **Tolerance**: atol=1e-5 (float32), atol=1e-2 (bfloat16).

#### 1.4 Rotary Embedding (RoPE) ✅
- `apply_rotary_emb` pure function + `RotaryEmbedding(nnx.Module)` + `get_rope` cache.
- **Tests**: identity (sin=0), long seq 4096, multi head_dim, lru_cache, numerical vs PyTorch, jit. ✅
- **Tolerance**: atol=1e-4 (trig accumulation).

---

### Phase 2 — Linear Layers & Parallel Wrappers
**Goal**: Port column-parallel and row-parallel linear projections.

#### 2.1 Basic Linear
- `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear`, `MergedColumnParallelLinear`
- Use `flax.nnx.Linear` or raw `jnp.dot` with explicit params.
- **Unit tests**: Forward pass shape, weight loading from HuggingFace checkpoint.
- **Numerical tests**: Match PyTorch matmul within atol=1e-3.

#### 2.2 Embed + LM Head
- `VocabParallelEmbedding`, `ParallelLMHead`
- Use `jnp.take` / embedding lookup.
- Weight tying between embed and LM head must be preserved.
- **Unit tests**: Token lookup correctness, tied-weight sharing.

---

### Phase 3 — Attention with Paged KV Cache
**Goal**: This is the most complex layer — paged KV cache requires dynamic index manipulation.

#### 3.1 KV Cache Design for JAX
- PyTorch uses in-place `kv_cache[block_table]` scatter. JAX requires `.at[].set()`.
- Design `KVCacheState` as a pytree-compatible frozen structure.
- Use `jax.lax.dynamic_update_slice` for block writes.

#### 3.2 Attention Forward
- Prefill (full causal) and decode (single-step KV cache read) paths.
- Use `jax.nn.dot_product_attention` (JAX >= 0.4.25) for flash-attention.
- **Unit tests**: Prefill output vs PyTorch sdpa, decode step correctness.

#### 3.3 Sampler
- Temperature / top-p / top-k via `jax.random` and `jnp.sort`.
- **Unit tests**: Greedy matches argmax, top-k distribution shape.

---

### Phase 4 — Full Model (LLaMA)
**Goal**: Assemble all layers into a full LLaMA Transformer.

#### 4.1 LLaMA Model Architecture
- `LlamaDecoderLayer`: RMSNorm → Attention → RMSNorm → MLP (SwiGLU)
- `LlamaModel`: Embedding → N×DecoderLayer → RMSNorm → LM Head
- Weight loading from HuggingFace `safetensors`.
- **E2E test**: Load `meta-llama/Llama-3.2-1B`, generate tokens, verify perplexity.

#### 4.2 JIT Compilation
- `jax.jit` the full prefill and decode functions with static argument shapes.
- **Benchmark**: Time-to-first-token and tokens/sec throughput.

---

### Phase 5 — Engine (Scheduler, Block Manager, Model Runner)
**Goal**: Port the orchestration layer.

#### 5.1 Sequence & Block Manager
- Pure Python dataclasses, port as-is with numpy/JAX equivalents.
- **Unit tests**: Block alloc/free lifecycle, preemption, swap logic.

#### 5.2 Scheduler
- Pure Python priority queue logic — minimal JAX changes.
- **Unit tests**: Batch construction, chunked prefill scheduling.

#### 5.3 Model Runner
- Replace `torch.cuda.synchronize()` with `jax.effects_barrier()`.
- Replace `torch.tensor()` with `jnp.array()`, manage `jax.device_put`.
- **Integration tests**: Multi-request batching.

---

### Phase 6 — LLM Engine & Public API
**Goal**: Full end-to-end pipeline matching the original `nano-vllm` public API.

- Public `LLM` class: `llm.generate(prompts, sampling_params)`
- **E2E tests**: 100-token generation, throughput within 20% of PyTorch baseline.

---

### Phase 7 — Multi-Device & Performance
**Goal**: Leverage JAX's native multi-device support.

- [ ] Tensor parallelism using `jax.sharding.NamedSharding`
- [ ] XLA profiling with `jax.profiler`
- [ ] Benchmark vs original PyTorch nano-vllm

---

## Testing Strategy

### Per-Module Test Requirements
Every module must have **both** before merging:

1. **Unit Tests** (`tests/unit/test_<module>.py`) — isolated, numerical equivalence vs PyTorch.
2. **Integration/E2E Tests** (`tests/e2e/test_<feature>.py`) — real model forward pass.

### Test Directory Structure
```
tests/
├── unit/
│   ├── test_smoke.py              # Phase 0: import + config + sampling_params
│   ├── layers/
│   │   ├── test_activation.py     # Phase 1 ✅
│   │   ├── test_layernorm.py      # Phase 1 ✅
│   │   ├── test_rotary_embedding.py # Phase 1 ✅
│   │   ├── test_linear.py         # Phase 2
│   │   ├── test_embed_head.py     # Phase 2
│   │   ├── test_attention.py      # Phase 3
│   │   └── test_sampler.py        # Phase 3
│   ├── engine/
│   │   ├── test_sequence.py       # Phase 5
│   │   ├── test_block_manager.py  # Phase 5
│   │   ├── test_scheduler.py      # Phase 5
│   │   └── test_model_runner.py   # Phase 5
│   └── models/
│       └── test_llama.py          # Phase 4
└── e2e/
    ├── test_generation.py         # Phase 6
    └── test_throughput.py         # Phase 7
```

### Numerical Equivalence Tolerances
- **Deterministic ops** (layernorm, linear, rope): atol=1e-3, rtol=1e-3
- **Attention softmax**: atol=1e-2
- **Sampling** (stochastic): compare distributions, not exact samples

---

## Key JAX Migration Patterns

### 1. In-place ops → functional updates
```python
# PyTorch
kv_cache[block_idx, :, :] = new_kv
# JAX
kv_cache = kv_cache.at[block_idx].set(new_kv)
```

### 2. `nn.Module` → `flax.nnx.Module`
```python
# PyTorch
class RMSNorm(nn.Module):
    def __init__(self, dim): self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x): ...
# JAX
class RMSNorm(nnx.Module):
    def __init__(self, dim, rngs): self.weight = nnx.Param(jnp.ones(dim))
    def __call__(self, x): ...
```

### 3. Dynamic shapes → static shapes with padding
```python
# Use padding + masking, or jax.lax.dynamic_slice for variable-length sequences
```

### 4. CUDA sync → JAX async dispatch
```python
torch.cuda.synchronize()   # PyTorch
jax.effects_barrier()      # JAX
```

### 5. Random sampling
```python
torch.multinomial(probs, 1)        # PyTorch
jax.random.categorical(key, logits) # JAX
```

---

## Dependencies

```toml
[project]
dependencies = [
    "jax[cuda12]>=0.4.25",
    "flax>=0.9.0",
    "optax>=0.2.0",
    "transformers>=4.40.0",
    "safetensors>=0.4.0",
    "chex>=0.1.86",
    "numpy>=1.26",
]
```

---

## Progress Tracker

| Date | Phase | Module | Status | Notes |
|------|-------|--------|--------|-------|
| 2026-05-27 | 0 | Project scaffold | ✅ Done | pyproject, CI, package skeleton, test stubs |
| 2026-05-27 | 1.1 | Config & SamplingParams | ✅ Done | Pure Python dataclasses, smoke tests pass |
| 2026-05-27 | 1.2 | Activation (SiluAndMul) | ✅ Done | Numerical match vs PyTorch atol=1e-5 |
| 2026-05-27 | 1.3 | RMSNorm | ✅ Done | Fused add+norm, numerical match atol=1e-5 |
| 2026-05-27 | 1.4 | Rotary Embedding (RoPE) | ✅ Done | Full cache, identity test, long-seq 4096, jit |

---

## References
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax NNX Guide](https://flax.readthedocs.io/en/latest/nnx_basics.html)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [Flash Attention in JAX](https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.dot_product_attention.html)
- [Original nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [chex testing library](https://github.com/google-deepmind/chex)
