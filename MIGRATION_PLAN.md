# nano-vllm-jax Migration Plan
## PyTorch → JAX / Flax Rewrite

**Source**: [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)  
**Target**: [putcn/nano-vllm-jax](https://github.com/putcn/nano-vllm-jax)  
**Started**: 2026-05-27  
**Last Updated**: 2026-05-28

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
| Linear (column/row parallel) | `nanovllm/layers/linear.py` | `nanovllm_jax/layers/linear.py` | ✅ Done |
| Embed + LM Head | `nanovllm/layers/embed_head.py` | `nanovllm_jax/layers/embed_head.py` | ✅ Done |
| Attention (PagedKV) | `nanovllm/layers/attention.py` | `nanovllm_jax/layers/attention.py` | ✅ Done |
| Sampler | `nanovllm/layers/sampler.py` | `nanovllm_jax/layers/sampler.py` | ✅ Done |
| LLaMA Model | `nanovllm/models/llama.py` | `nanovllm_jax/models/llama.py` | ✅ Done |
| HF Weight Loader | *(new)* | `nanovllm_jax/loader/weight_loader.py` | ✅ Done |
| Model Registry | *(new)* | `nanovllm_jax/loader/model_registry.py` | ✅ Done |
| Sequence | `nanovllm/engine/sequence.py` | `nanovllm_jax/engine/sequence.py` | ✅ Done |
| Block Manager (PagedAttn) | `nanovllm/engine/block_manager.py` | `nanovllm_jax/engine/block_manager.py` | ✅ Done |
| Scheduler | `nanovllm/engine/scheduler.py` | `nanovllm_jax/engine/scheduler.py` | ✅ Done |
| Model Runner | `nanovllm/engine/model_runner.py` | `nanovllm_jax/engine/model_runner.py` | ✅ Done |
| LLM Engine | `nanovllm/engine/llm_engine.py` | `nanovllm_jax/engine/llm_engine.py` | ✅ Done |
| LLM Entry Point | `nanovllm/llm.py` | `nanovllm_jax/llm.py` | ✅ Done |

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

#### 1.1 Config & SamplingParams ✅
- Pure Python dataclasses with `assert`-based validation.
- **Bug fixed**: `SamplingParams` uses `assert` (not `raise ValueError`) so `pytest.raises(AssertionError)` works.

#### 1.2 Activation Functions ✅
- `silu_and_mul` + `SiluAndMul` NNX wrapper. atol=1e-5 vs PyTorch.

#### 1.3 RMSNorm ✅
- Fused add+norm path. atol=1e-5.
- **Bug fixed**: `jnp.cos/sin` free functions; weight dtype cast.

#### 1.4 Rotary Embedding (RoPE) ✅
- Full LRU cache, identity test, long-seq 4096, jit. atol=1e-4.

---

### Phase 2 — Linear Layers & Parallel Wrappers ✅

- `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear`, `MergedColumnParallelLinear`
- `VocabParallelEmbedding`, `ParallelLMHead` with optional `last_indices` slicing.
- Weight tying via `tie_weights()` reference sharing.

---

### Phase 3 — Attention with Paged KV Cache ✅

- `PagedKVCache`: layout `[layers, 2, blocks, block_size, kv_heads, head_dim]`, functional `.at[].set()` writes.
- `Attention`: prefill causal + decode paged lookup + GQA repeat.
- `Sampler`: greedy / top-k / top-p via `jax.random.categorical`.
- **Bug fixed**: test index `k_out[seq, pos, head, :]`.

---

### Phase 4 — Full LLaMA Model ✅

- `LlamaConfig`, `LlamaMLP`, `LlamaAttention`, `LlamaDecoderLayer`, `LlamaModel`, `LlamaForCausalLM`.
- `load_weights()` accepts flat HF param dicts on every class.
- **Bug fixed**: `nnx.List` instead of plain Python list; non-zero weight test helper.

---

### Phase 5 — HuggingFace Weight Loader ✅

- `load_hf_weights()` / `llama_config_from_hf()` / `load_model()` one-liner.
- Sharded safetensors + pytorch bin support.
- `@register` decorator for extensible architecture registry.

---

### Phase 6 — Engine (Sequence, Block Manager, Scheduler, Model Runner) ✅

#### 6.1 Sequence ✅
- `Sequence` / `SequenceGroup` dataclasses; `SequenceStatus` enum.
- `check_stop()`: max_tokens + `stop_token_ids`.
- **Bug fixed**: added `stop_token_ids: List[int]` to `SamplingParams`.

#### 6.2 Block Manager ✅
- Free-list block allocator; `alloc` / `free` / `num_free_blocks`.

#### 6.3 Scheduler ✅
- Continuous batching: waiting → prefill → decode → finished.
- Preemption when no free blocks available.
- `SchedulerOutput`: `prefill_seqs`, `decode_seqs`, `preempted_seqs`.
- **Bug fixed**: `EngineConfig` flattened (was nested `ModelConfig` + `CacheConfig`); added `max_num_batched_tokens` field.

#### 6.4 Model Runner ✅
- Lazy model loading; stub model for tests (no checkpoint needed).
- `_run_prefill` / `_run_decode` build JAX input arrays per sequence.
- Falls back to `_StubModel` when `config.model == ""`.

---

### Phase 7 — LLMEngine & Public API ✅
**Goal**: Full end-to-end pipeline matching the original `nano-vllm` public API.

#### 7.1 LLMEngine ✅
- `add_request(prompt, sampling_params)` → request ID.
- `step()` → list of newly finished `RequestOutput`.
- `generate_all()` → run to completion.
- Byte-level tokenizer fallback for test-without-tokenizer.
- **Tests**: single request, batch, stop token, `prompt_token_ids`, `has_unfinished`. ✅

#### 7.2 LLM Public API ✅
- `LLM(model, **kwargs).generate(prompts, sampling_params)`.
- Matches nano-vllm signature exactly.
- **Tests**: single, batch, default params, output order. ✅

#### Usage
```python
from nanovllm_jax.llm import LLM
from nanovllm_jax.sampling_params import SamplingParams

llm = LLM("meta-llama/Llama-3.2-1B", num_gpu_blocks=512, block_size=16)
outputs = llm.generate(
    ["Tell me a joke", "What is JAX?"],
    SamplingParams(temperature=0.8, max_tokens=128),
)
for o in outputs:
    print(o.text)
```

---

### Phase 8 — Flash Attention & Multi-Device ⬜
**Goal**: Performance optimisations and multi-GPU support.

- [ ] Replace naïve matmul attention with `jax.nn.dot_product_attention` (XLA flash-attn)
- [ ] Tensor parallelism via `jax.sharding.NamedSharding`
- [ ] XLA profiling with `jax.profiler`
- [ ] Benchmark vs original PyTorch nano-vllm (tokens/sec, TTFT)

---

## Testing Strategy

### Test Directory Structure
```
tests/
├── unit/
│   ├── test_smoke.py                    # Phase 0 ✅
│   ├── layers/
│   │   ├── test_activation.py           # Phase 1 ✅
│   │   ├── test_layernorm.py            # Phase 1 ✅
│   │   ├── test_rotary_embedding.py     # Phase 1 ✅
│   │   ├── test_linear.py               # Phase 2 ✅
│   │   ├── test_embed_head.py           # Phase 2 ✅
│   │   ├── test_attention.py            # Phase 3 ✅
│   │   └── test_sampler.py              # Phase 3 ✅
│   ├── models/
│   │   └── test_llama.py                # Phase 4 ✅
│   ├── loader/
│   │   └── test_weight_loader.py        # Phase 5 ✅
│   └── engine/
│       ├── test_sequence.py             # Phase 6 ✅
│       ├── test_block_manager.py        # Phase 6 ✅
│       ├── test_scheduler.py            # Phase 6 ✅
│       └── test_llm_engine.py           # Phase 7 ✅
└── e2e/
    ├── test_generation.py               # Phase 8 ⬜ (real model)
    └── test_throughput.py               # Phase 8 ⬜
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
class RMSNorm(nnx.Module):
    def __init__(self, dim): self.weight = nnx.Param(jnp.ones(dim))
    def __call__(self, x): ...
```

### 3. Dynamic shapes → static shapes with padding
```python
# Use padding + masking, or jax.lax.dynamic_slice
```

### 4. CUDA sync → JAX async dispatch
```python
torch.cuda.synchronize()   # PyTorch
jax.effects_barrier()      # JAX
```

### 5. Random sampling
```python
torch.multinomial(probs, 1)         # PyTorch
jax.random.categorical(key, logits) # JAX
```

### 6. Module list → `nnx.List`
```python
self.layers = nnx.List([...])  # plain list raises ValueError in Flax NNX Pytree mode
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

| Date | Phase | Module | Commit | Status | Notes |
|------|-------|--------|--------|--------|-------|
| 2026-05-27 | 0 | Project scaffold | — | ✅ Done | pyproject, CI, skeleton |
| 2026-05-27 | 1.1 | Config & SamplingParams | — | ✅ Done | assert-based validation |
| 2026-05-27 | 1.2 | Activation | — | ✅ Done | atol=1e-5 |
| 2026-05-27 | 1.3 | RMSNorm | — | ✅ Done | fused add+norm |
| 2026-05-27 | 1.4 | RoPE | — | ✅ Done | LRU cache, jit |
| 2026-05-27 | 1 | Bug fixes (RoPE+Norm) | `a930b22` | ✅ Done | jnp.cos/sin; dtype cast |
| 2026-05-27 | 2 | Linear + Embed/LMHead | — | ✅ Done | TP sharding, weight tying |
| 2026-05-27 | 3 | PagedKVCache + Attention + Sampler | `cb5f06b` | ✅ Done | prefill/decode/GQA |
| 2026-05-27 | 3 | Bug fix (test index) | `403d934` | ✅ Done | `k_out[seq,pos,head,:]` |
| 2026-05-27 | 4 | LLaMA model stack | `cd3e901` | ✅ Done | full MLP/Attn/Decoder |
| 2026-05-27 | 4 | Bug fix (nnx.List) | `895286d` | ✅ Done | `nnx.List`; residual test |
| 2026-05-27 | 5 | HF Loader + Registry | `7caae14` | ✅ Done | safetensors/bin shards |
| 2026-05-27 | 6 | Engine skeleton | — | ✅ Done | sequence/block/sched/runner |
| 2026-05-27 | 6 | Bug fix (EngineConfig flat) | `e0bc16c` | ✅ Done | flat fields + max_num_batched_tokens |
| 2026-05-27 | 6 | Bug fix (SamplingParams assert) | `2f269d3` | ✅ Done | assert not ValueError; stop_token_ids |
| 2026-05-28 | 7 | LLMEngine + LLM API | *(this commit)* | ✅ Done | generate_all, RequestOutput, 9 tests |

---

## References
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax NNX Guide](https://flax.readthedocs.io/en/latest/nnx_basics.html)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [Flash Attention in JAX](https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.dot_product_attention.html)
- [Original nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [chex testing library](https://github.com/google-deepmind/chex)
