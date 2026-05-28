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
| Linear (column/row parallel) | `nanovllm/layers/linear.py` | `nanovllm_jax/layers/linear.py` | ✅ Done |
| Embed + LM Head | `nanovllm/layers/embed_head.py` | `nanovllm_jax/layers/embed_head.py` | ✅ Done |
| Attention (PagedKV) | `nanovllm/layers/attention.py` | `nanovllm_jax/layers/attention.py` | ✅ Done |
| Sampler | `nanovllm/layers/sampler.py` | `nanovllm_jax/layers/sampler.py` | ✅ Done |
| LLaMA Model | `nanovllm/models/llama.py` | `nanovllm_jax/models/llama.py` | ✅ Done |
| HF Weight Loader | *(new)* | `nanovllm_jax/loader/weight_loader.py` | ✅ Done |
| Model Registry | *(new)* | `nanovllm_jax/loader/model_registry.py` | ✅ Done |
| Sequence | `nanovllm/engine/sequence.py` | `nanovllm_jax/engine/sequence.py` | ⬜ Not Started |
| Block Manager (PagedAttn) | `nanovllm/engine/block_manager.py` | `nanovllm_jax/engine/block_manager.py` | ⬜ Not Started |
| Scheduler | `nanovllm/engine/scheduler.py` | `nanovllm_jax/engine/scheduler.py` | ⬜ Not Started |
| Model Runner | `nanovllm/engine/model_runner.py` | `nanovllm_jax/engine/model_runner.py` | ⬜ Not Started |
| LLM Engine | `nanovllm/engine/llm_engine.py` | `nanovllm_jax/engine/llm_engine.py` | ⬜ Not Started |
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
**Commits**: Initial scaffold

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
- **Bug fixed**: `jnp.cos(emb)` / `jnp.sin(emb)` instead of instance methods; weight dtype cast before multiply.

#### 1.4 Rotary Embedding (RoPE) ✅
- `apply_rotary_emb` pure function + `RotaryEmbedding(nnx.Module)` + `get_rope` cache.
- **Tests**: identity (sin=0), long seq 4096, multi head_dim, lru_cache, numerical vs PyTorch, jit. ✅
- **Tolerance**: atol=1e-4 (trig accumulation).

---

### Phase 2 — Linear Layers & Parallel Wrappers ✅
**Goal**: Port column-parallel and row-parallel linear projections.

#### 2.1 Basic Linear ✅
- `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear`, `MergedColumnParallelLinear`
- Raw `jnp.dot` with explicit weight params; TP shard slicing.
- **Unit tests**: Forward shape, weight loading, TP shard correctness, numerical match PyTorch. ✅

#### 2.2 Embed + LM Head ✅
- `VocabParallelEmbedding` (`jnp.take`), `ParallelLMHead` with optional `last_indices` slicing.
- Weight tying via `tie_weights()` reference sharing.
- **Unit tests**: Token lookup, tied-weight sharing, `last_indices` shape. ✅

---

### Phase 3 — Attention with Paged KV Cache ✅
**Goal**: Paged KV cache + attention (prefill & decode paths).

#### 3.1 PagedKVCache ✅
- Layout: `[num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]`
- Writes via `jax.lax.fori_loop` + `.at[].set()` (functional, jit-safe).
- Reads via fancy indexing `cache[block_table]`.
- **Tests**: Init shape, write/read roundtrip, multi-layer isolation. ✅
- **Bug fixed**: Test index `k_out[seq, pos, head, :]` not `k_out[seq, pos, :]`.

#### 3.2 Attention Forward ✅
- Prefill: full causal mask + `jnp.matmul` (flash-attn in Phase 7).
- Decode: paged KV lookup + padding mask per sequence.
- GQA: `jnp.repeat` KV heads to match Q heads.
- **Tests**: prefill shape, causal property, GQA, decode shape, custom scale. ✅

#### 3.3 Sampler ✅
- `greedy_sample` (argmax), `temperature_sample` (temp / top-k / top-p).
- `jax.random.categorical` for stochastic sampling.
- **Tests**: greedy=argmax, top_k=1 forces greedy, top_p≈0 forces greedy, reproducibility. ✅

---

### Phase 4 — Full Model (LLaMA) ✅
**Goal**: Assemble all layers into a full LLaMA Transformer.

#### 4.1 LLaMA Components ✅
- `LlamaConfig` dataclass with `__post_init__` head_dim inference.
- `LlamaMLP`: SwiGLU via `MergedColumnParallelLinear` + `SiluAndMul` + `RowParallelLinear`.
- `LlamaAttention`: QKV proj + RoPE + `Attention` + output proj.
- `LlamaDecoderLayer`: fused RMSNorm residual → attn → fused RMSNorm residual → MLP.
- `LlamaModel`: embed + `nnx.List` of N decoder layers + final norm.
- `LlamaForCausalLM`: model + LM head + sampler; optional weight tying.
- `load_weights()` on every class accepts HuggingFace flat param dicts.
- **Tests**: MLP shape, attention prefill/decode shape, decoder residual, model determinism, causal LM logits, last_indices, weight tying, greedy sample. ✅
- **Bug fixed**: `self.layers = nnx.List([...])` — plain Python `list` raises `ValueError` in Flax NNX Pytree mode.
- **Bug fixed**: `test_decoder_layer_residual` now loads non-zero weights via `load_random_weights()` helper.

---

### Phase 5 — HuggingFace Weight Loader ✅
**Goal**: Load real model weights from HF checkpoints; auto-detect architecture.

#### 5.1 Weight Loader ✅
- Supports `.safetensors` (preferred) and `pytorch_model.bin` (fallback).
- Handles sharded checkpoints via `*.index.json` files.
- `load_hf_weights(model, dir, dtype)` — loads into any model with `load_weights()`.
- `llama_config_from_hf(dir)` — builds `LlamaConfig` from `config.json`.
- **Tests**: config read, missing file error, shard file detection, safetensors roundtrip forward pass. ✅

#### 5.2 Model Registry ✅
- `@register(hf_arch)` decorator for extensible architecture mapping.
- `LlamaForCausalLM` + `MistralForCausalLM` registered out-of-the-box.
- `load_model(dir)` — one-liner: auto-detect arch → build config → build model → load weights.
- **Tests**: registry lookup, Mistral→Llama mapping, unknown arch raises, end-to-end load+forward. ✅

#### Real model usage (after Phase 5)
```python
from nanovllm_jax.loader.model_registry import load_model
model = load_model("~/.cache/huggingface/Llama-3.1-8B", dtype="bfloat16")
```

---

### Phase 6 — Engine (Sequence, Block Manager, Scheduler, Model Runner) ⬜
**Goal**: Port the orchestration layer for continuous batching.

#### 6.1 Sequence & Block Manager
- Pure Python dataclasses, port as-is.
- `BlockManager`: paged block alloc / free / preemption / swap.
- **Unit tests**: Block alloc/free lifecycle, preemption, swap logic.

#### 6.2 Scheduler
- Priority queue, chunked prefill, continuous batching logic.
- **Unit tests**: Batch construction, chunked prefill scheduling.

#### 6.3 Model Runner
- Assembles per-step `input_ids`, `positions`, `block_indices`, `block_offsets` arrays.
- Replace `torch.cuda.synchronize()` → `jax.effects_barrier()`.
- Replace `torch.tensor()` → `jnp.array()` + `jax.device_put`.
- **Integration tests**: Multi-request batching.

---

### Phase 7 — LLM Engine & Public API ⬜
**Goal**: Full end-to-end pipeline matching the original `nano-vllm` public API.

- `LLMEngine`: ties together scheduler + model runner + sampler.
- Public `LLM` class: `llm.generate(prompts, sampling_params)`.
- **E2E tests**: 100-token generation, throughput within 20% of PyTorch baseline.

---

### Phase 8 — Flash Attention & Multi-Device ⬜
**Goal**: Leverage JAX's native multi-device support and optimized attention.

- [ ] Replace naive matmul attention with `jax.nn.dot_product_attention` (XLA flash-attn)
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
│   └── engine/                          # Phase 6 ⬜
│       ├── test_sequence.py
│       ├── test_block_manager.py
│       ├── test_scheduler.py
│       └── test_model_runner.py
└── e2e/
    ├── test_generation.py               # Phase 7 ⬜
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
# PyTorch
class RMSNorm(nn.Module):
    def __init__(self, dim): self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x): ...
# JAX
class RMSNorm(nnx.Module):
    def __init__(self, dim): self.weight = nnx.Param(jnp.ones(dim))
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
torch.multinomial(probs, 1)         # PyTorch
jax.random.categorical(key, logits) # JAX
```

### 6. Module list → `nnx.List`
```python
# PyTorch
self.layers = nn.ModuleList([...])
# JAX / Flax NNX — plain list raises ValueError in Pytree mode
self.layers = nnx.List([...])
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
| 2026-05-27 | 0 | Project scaffold | — | ✅ Done | pyproject, CI, package skeleton |
| 2026-05-27 | 1.1 | Config & SamplingParams | — | ✅ Done | Pure Python dataclasses |
| 2026-05-27 | 1.2 | Activation (SiluAndMul) | — | ✅ Done | atol=1e-5 vs PyTorch |
| 2026-05-27 | 1.3 | RMSNorm | — | ✅ Done | Fused add+norm, atol=1e-5 |
| 2026-05-27 | 1.4 | Rotary Embedding (RoPE) | — | ✅ Done | Full cache, jit, long-seq 4096 |
| 2026-05-27 | 1 | Bug fixes (RoPE + LayerNorm) | `a930b22` | ✅ Done | jnp.cos/sin; weight dtype cast |
| 2026-05-27 | 2 | Linear + Embed/LMHead | — | ✅ Done | TP sharding, weight tying |
| 2026-05-27 | 3 | PagedKVCache + Attention + Sampler | `cb5f06b` | ✅ Done | prefill/decode/GQA, top-k/top-p |
| 2026-05-27 | 3 | Bug fix (test index shape) | `403d934` | ✅ Done | `k_out[seq,pos,head,:]` |
| 2026-05-27 | 4 | LLaMA model stack | `cd3e901` | ✅ Done | Full MLP/Attn/Decoder/Model/CausalLM |
| 2026-05-27 | 4 | Bug fixes (nnx.List + residual test) | `895286d` | ✅ Done | `nnx.List`; non-zero weight test |
| 2026-05-27 | 5 | HF Weight Loader + Model Registry | `7caae14` | ✅ Done | safetensors/bin shards, load_model() |

---

## References
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax NNX Guide](https://flax.readthedocs.io/en/latest/nnx_basics.html)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [Flash Attention in JAX](https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.dot_product_attention.html)
- [Original nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [chex testing library](https://github.com/google-deepmind/chex)
