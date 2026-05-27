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
| Config | `nanovllm/config.py` | `nanovllm_jax/config.py` | ⬜ Not Started |
| Sampling Params | `nanovllm/sampling_params.py` | `nanovllm_jax/sampling_params.py` | ⬜ Not Started |
| Activation (SiLU/GeGLU) | `nanovllm/layers/activation.py` | `nanovllm_jax/layers/activation.py` | ⬜ Not Started |
| RMSNorm / LayerNorm | `nanovllm/layers/layernorm.py` | `nanovllm_jax/layers/layernorm.py` | ⬜ Not Started |
| Rotary Embedding (RoPE) | `nanovllm/layers/rotary_embedding.py` | `nanovllm_jax/layers/rotary_embedding.py` | ⬜ Not Started |
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

### Phase 0 — Project Setup
**Goal**: Repository structure, CI, dependencies, tooling.

- [ ] `pyproject.toml` with JAX, Flax, optax, pytest, chex dependencies
- [ ] `nanovllm_jax/` package skeleton with `__init__.py` files
- [ ] `tests/` directory structure mirroring source
- [ ] GitHub Actions CI workflow (pytest on push/PR)
- [ ] `README.md` with setup instructions
- [ ] Pre-commit hooks (ruff, mypy)

**Tests**: CI pipeline smoke test (import package successfully).

---

### Phase 1 — Stateless Utility Layers
**Goal**: Port all pure-math layers that have no state or minimal learned params. These are easiest to verify numerically against PyTorch.

#### 1.1 Config & SamplingParams
- Port `config.py` and `sampling_params.py` as plain Python dataclasses (no framework changes needed, but verify compatibility with JAX arrays).
- **Unit tests**: Instantiation, field validation, serialization.

#### 1.2 Activation Functions
- `SiluAndMul` (SwiGLU): `jax.nn.silu(x[..., :half]) * x[..., half:]`
- **Unit tests**: Shape correctness, numerical match vs PyTorch `F.silu`.
- **Numerical tolerance**: atol=1e-5.

#### 1.3 RMSNorm
- Replace `torch.nn.RMSNorm` with JAX functional equivalent.
- Learnable `weight` param via `flax.nnx.Param`.
- **Unit tests**: Output shape, numerical match vs PyTorch reference.
- **Edge cases**: zero input, unit variance input.

#### 1.4 Rotary Embedding (RoPE)
- Port `RotaryEmbedding` using `jnp.cos`/`jnp.sin` and `jnp.einsum`.
- Must be JIT-compilable with static shapes.
- **Unit tests**: cos/sin cache correctness, apply_rotary numerical match.
- **Edge tests**: long sequence (>4096 tokens), different head dims.

---

### Phase 2 — Linear Layers & Parallel Wrappers
**Goal**: Port column-parallel and row-parallel linear projections. In nano-vllm these wrap weight loading for tensor parallelism.

#### 2.1 Basic Linear
- `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear`, `MergedColumnParallelLinear`
- Use `flax.nnx.Linear` or raw `jnp.dot` with explicit params.
- For single-device: skip actual NCCL sharding, add shard stubs with `jax.sharding` annotations.
- **Unit tests**: Forward pass shape, weight loading from HuggingFace checkpoint.
- **Numerical tests**: Match PyTorch matmul within atol=1e-3.

#### 2.2 Embed + LM Head
- `VocabParallelEmbedding`, `ParallelLMHead`
- Use `jnp.take` / embedding lookup.
- Weight tying between embed and LM head must be preserved.
- **Unit tests**: Token lookup correctness, tied-weight sharing.

---

### Phase 3 — Attention with Paged KV Cache
**Goal**: This is the most complex layer — paged KV cache requires dynamic index manipulation that JAX handles differently (no in-place ops).

#### 3.1 KV Cache Design for JAX
- PyTorch uses in-place `kv_cache[block_table]` scatter. JAX requires `jax.lax.scatter` or `.at[].set()`.
- Design `KVCacheState` as a pytree-compatible frozen structure.
- Use `jax.lax.dynamic_update_slice` for block writes.
- **Architecture decision**: Document the paged-attention approach (static buffer with dynamic indexing).

#### 3.2 Attention Forward
- Implement prefill (full causal attention) and decode (single-step with KV cache read) paths.
- Use `jax.lax.dot_general` for batched QK matmul.
- Optionally integrate flash-attention via `jax.nn.dot_product_attention` (JAX >= 0.4.25).
- **Unit tests**: Prefill output vs PyTorch sdpa, decode step correctness.
- **Performance test**: Measure tokens/sec vs PyTorch baseline.

#### 3.3 Sampler
- Port temperature / top-p / top-k sampling using `jax.random` and `jnp.sort`.
- **Unit tests**: Greedy (temp=0) matches argmax, top-k distribution shape.

---

### Phase 4 — Full Model (LLaMA)
**Goal**: Assemble all layers into a full LLaMA Transformer.

#### 4.1 LLaMA Model Architecture
- `LlamaDecoderLayer`: RMSNorm → Attention → RMSNorm → MLP (SwiGLU)
- `LlamaModel`: Embedding → N×DecoderLayer → RMSNorm → LM Head
- Implement weight loading from HuggingFace `safetensors` / `torch` checkpoint.
- **Unit tests**: Forward pass shape, single-token generation sanity check.
- **End-to-end test**: Load `meta-llama/Llama-3.2-1B` weights, generate "Hello" continuation, verify perplexity is reasonable.

#### 4.2 JIT Compilation
- `jax.jit` the full prefill and decode functions with static argument shapes.
- Verify compilation completes without `TracerBoolConversionError`.
- **Benchmark**: Time-to-first-token and tokens/sec throughput.

---

### Phase 5 — Engine (Scheduler, Block Manager, Model Runner)
**Goal**: Port the orchestration layer. Most of this code is Python-level logic with no deep learning; JAX impact is mainly in `model_runner.py`.

#### 5.1 Sequence & Block Manager
- `Sequence`, `SequenceGroup`: Pure Python dataclasses, port as-is.
- `BlockManager`: KV cache block allocator. Replace PyTorch tensor ops with numpy/JAX equivalents.
- **Unit tests**: Block alloc/free lifecycle, preemption, swap logic.

#### 5.2 Scheduler
- Pure Python priority queue logic — minimal JAX changes.
- **Unit tests**: Batch construction, chunked prefill scheduling.

#### 5.3 Model Runner
- This is the bridge between engine and model.
- Replace `torch.cuda.synchronize()` with `jax.effects_barrier()`.
- Replace `torch.tensor()` input construction with `jnp.array()`.
- Manage JAX device placement (`jax.device_put`).
- **Unit tests**: Input preparation correctness, output token extraction.
- **Integration tests**: Multi-request batching.

---

### Phase 6 — LLM Engine & Public API
**Goal**: Full end-to-end pipeline matching the original `nano-vllm` public API.

#### 6.1 LLM Engine
- Wire together scheduler + model runner.
- Replace Python multiprocessing (torch.mp) with JAX-native approach or keep Python subprocess with JAX workers.
- **Integration tests**: `engine.generate()` with multiple concurrent requests.

#### 6.2 LLM Entry Point
- Public `LLM` class matching original API: `llm.generate(prompts, sampling_params)`
- **End-to-end tests**:
  - Generate 100 tokens from a real LLaMA model.
  - Verify output matches PyTorch version within sampling tolerance.
  - Measure throughput within 20% of PyTorch baseline.

---

### Phase 7 — Multi-Device & Performance
**Goal**: Leverage JAX's native multi-device support.

- [ ] Tensor parallelism using `jax.sharding.NamedSharding`
- [ ] Pipeline via `jax.lax.ppermute`
- [ ] XLA profiling with `jax.profiler`
- [ ] Benchmark vs original PyTorch nano-vllm on same hardware

---

## Testing Strategy

### Per-Module Test Requirements
Every module must have **both** before merging:

1. **Unit Tests** (`tests/unit/test_<module>.py`):
   - Test each function/class in isolation
   - Cover edge cases (empty input, max length, batch size 1 and >1)
   - Numerical equivalence vs PyTorch where applicable

2. **Integration/E2E Tests** (`tests/e2e/test_<feature>.py`):
   - Test the module within the full pipeline context
   - At least one real model forward pass

### Test Directory Structure
```
tests/
├── unit/
│   ├── layers/
│   │   ├── test_activation.py
│   │   ├── test_layernorm.py
│   │   ├── test_rotary_embedding.py
│   │   ├── test_linear.py
│   │   ├── test_embed_head.py
│   │   ├── test_attention.py
│   │   └── test_sampler.py
│   ├── engine/
│   │   ├── test_sequence.py
│   │   ├── test_block_manager.py
│   │   ├── test_scheduler.py
│   │   └── test_model_runner.py
│   └── models/
│       └── test_llama.py
└── e2e/
    ├── test_generation.py
    └── test_throughput.py
```

### Numerical Equivalence Tolerances
Use `chex.assert_trees_all_close` with:
- **Deterministic ops** (layernorm, linear, rope): atol=1e-3, rtol=1e-3
- **Attention softmax**: atol=1e-2 (accumulation differences expected)
- **Sampling** (stochastic): compare distributions, not exact samples

---

## Key JAX Migration Patterns

### 1. In-place ops → functional updates
```python
# PyTorch (in-place)
kv_cache[block_idx, :, :] = new_kv

# JAX (functional)
kv_cache = kv_cache.at[block_idx].set(new_kv)
```

### 2. `nn.Module` → `flax.nnx.Module`
```python
# PyTorch
class RMSNorm(nn.Module):
    def __init__(self, dim): self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x): ...

# JAX / Flax NNX
class RMSNorm(nnx.Module):
    def __init__(self, dim, rngs): self.weight = nnx.Param(jnp.ones(dim))
    def __call__(self, x): ...
```

### 3. Dynamic shapes → static shapes with padding
```python
# JAX jit requires static shapes; use padding + masking
# or jax.lax.dynamic_slice for variable-length sequences
```

### 4. CUDA synchronization → JAX async dispatch
```python
# PyTorch
torch.cuda.synchronize()

# JAX
jax.effects_barrier()
# or: x.block_until_ready()
```

### 5. Random sampling
```python
# PyTorch
torch.multinomial(probs, num_samples=1)

# JAX
jax.random.categorical(key, logits)
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
    "pytest>=8.0",
    "pytest-xdist",
]
```

---

## Progress Tracker

> This section is updated as modules are completed. Each row added after a PR is merged.

| Date | Phase | Module | Status | PR / Notes |
|------|-------|--------|--------|------------|
| 2026-05-27 | 0 | Migration Plan | ✅ Done | Initial document created |

---

## References
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax NNX Guide](https://flax.readthedocs.io/en/latest/nnx_basics.html)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [Flash Attention in JAX](https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.dot_product_attention.html)
- [Original nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [chex testing library](https://github.com/google-deepmind/chex)
