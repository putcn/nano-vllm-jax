# nano-vllm-jax Migration Plan
## PyTorch → JAX / Flax Rewrite

**Source**: [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)  
**Target**: [putcn/nano-vllm-jax](https://github.com/putcn/nano-vllm-jax)  
**Started**: 2026-05-27  
**Last Updated**: 2026-05-28

---

## Overview

This project rewrites the `nano-vllm` LLM inference engine — originally built on PyTorch — using [JAX](https://github.com/google/jax) + [Flax](https://github.com/google/flax) (NNX) as the deep learning backend. The goal is to leverage JAX’s functional programming model, `jit` compilation, `vmap`/`pmap` for multi-device parallelism, and XLA-based kernel fusion for high-performance LLM inference.

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
| RMSNorm / PerHeadRMSNorm | `nanovllm/layers/layernorm.py` | `nanovllm_jax/layers/layernorm.py` | ✅ Done |
| Rotary Embedding (RoPE) | `nanovllm/layers/rotary_embedding.py` | `nanovllm_jax/layers/rotary_embedding.py` | ✅ Done |
| Linear (column/row parallel) | `nanovllm/layers/linear.py` | `nanovllm_jax/layers/linear.py` | ✅ Done |
| Embed + LM Head | `nanovllm/layers/embed_head.py` | `nanovllm_jax/layers/embed_head.py` | ✅ Done |
| Attention (PagedKV) | `nanovllm/layers/attention.py` | `nanovllm_jax/layers/attention.py` | ✅ Done |
| Sampler | `nanovllm/layers/sampler.py` | `nanovllm_jax/layers/sampler.py` | ✅ Done |
| LLaMA / Qwen3 Model | `nanovllm/models/llama.py` | `nanovllm_jax/models/llama.py` | ✅ Done |
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
- `RMSNorm`: fused add+norm path, float32 internal compute, output dtype matches input. atol=1e-5.
- `PerHeadRMSNorm`: per-head RMSNorm for Qwen3 QK-Norm. Input `(T, num_heads, head_dim)`,
  normalises along `head_dim`, weight shape `(head_dim,)` shared across heads.
- **Bug fixed**: `jnp.cos/sin` free functions; weight dtype cast.

#### 1.4 Rotary Embedding (RoPE) ✅
- Full LRU cache, identity test, long-seq 4096, jit. atol=1e-4.
- Matches Qwen3 style: `emb = concat([freqs, freqs])`, rotation `[-x2, x1]`.

---

### Phase 2 — Linear Layers & Parallel Wrappers ✅

- `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear`, `MergedColumnParallelLinear`
- `VocabParallelEmbedding`, `ParallelLMHead` with optional `last_indices` slicing.
- Weight tying via `tie_weights()` reference sharing.

---

### Phase 3 — Attention with Paged KV Cache ✅

- `PagedKVCache`: layout `[layers, 2, blocks, block_size, kv_heads, head_dim]`, functional `.at[].set()` writes.
- `Attention`: prefill causal + decode paged lookup + GQA repeat.
- `_make_batched_causal_mask`: block-diagonal mask preventing cross-sequence attention in packed prefill.
- `Sampler`: greedy / top-k / top-p via `jax.random.categorical`.
- **Bug fixed**: test index `k_out[seq, pos, head, :]`.

---

### Phase 4 — Full LLaMA / Qwen3 Model ✅

- `LlamaConfig`, `LlamaMLP`, `LlamaAttention`, `LlamaDecoderLayer`, `LlamaModel`, `LlamaForCausalLM`.
- `load_weights()` accepts flat HF param dicts on every class.
- **Qwen3 QK-Norm**: `LlamaAttention` applies `PerHeadRMSNorm` to Q and K *before* RoPE,
  using weights `self_attn.q_norm.weight` / `self_attn.k_norm.weight` (shape `head_dim`).
  Falls back to identity (weight=ones) for plain Llama models that lack these weights.
- **Bug fixed**: `nnx.List` instead of plain Python list.
- **Bug fixed**: residual connections — explicit `x = attn(x) + residual` pattern (pre-norm).
- **Bug fixed**: `tie_word_embeddings` — copy embed weight into lm_head at load time.

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
- `nnx.jit` for both `_jit_prefill` and `_jit_decode` — KV cache writes propagate back correctly.
- Padding to next-power-of-2 token count — O(log N) distinct XLA compilation shapes.
- `last_indices` passed into `LlamaForCausalLM` so lm_head returns `[num_seqs, vocab]`.
- `seq_lens` padding uses `val=0` (not `val=1`) to prevent stale KV cache reads.
- **Bug fixed**: `jax.jit` → `nnx.jit` for KV cache persistence across steps.
- **Bug fixed**: `last_indices` wiring through prefill path.
- **Bug fixed**: `seq_lens` padding value `0` prevents phantom context in decode.

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

#### 7.3 GPU End-to-End Demo ✅
- `demo.gpu.sh`: Docker-based single-command demo on NVIDIA GPU.
- Verified correct output on **Qwen3-0.6B** (geography, math, coding, reasoning, translation).
- Prints prefill latency (ms) and decode throughput (tok/s) per question.

#### Usage
```python
from nanovllm_jax.llm import LLM
from nanovllm_jax.sampling_params import SamplingParams

llm = LLM("Qwen/Qwen3-0.6B", num_gpu_blocks=512, block_size=16)
outputs = llm.generate(
    ["What is the capital of France?", "Write a Python palindrome checker."],
    SamplingParams(temperature=0.0, max_tokens=128),
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
- [ ] e2e tests: `tests/e2e/test_generation.py`, `tests/e2e/test_throughput.py`

---

## Implementation Notes & Bug Log

This section documents design decisions and non-obvious bugs encountered during
the PyTorch → JAX port. Each entry corresponds to a real bug that was debugged
and fixed in production.

### 1. JAX Functional Purity and `nnx.jit` (Critical)

PyTorch operations are **eager** and **stateful**. In-place mutations like
`cache[layer, 0, bi, bo] = keys` are immediately visible everywhere.

JAX’s `jax.jit` compiles a **pure function**. Python-level object mutations
inside a `jax.jit`-decorated function are **not propagated back** after the
call returns. For `PagedKVCache.write()` this means:

```python
new_cache = jax.lax.fori_loop(0, num_real, body, cache)
self.cache = nnx.Variable(new_cache)   # Python-level mutation — lost under jax.jit
```

After the JIT call returns, the outer `kv_cache.cache` is still the original
all-zeros array. Every decode step reads a blank cache — garbage output.

**Fix**: Use `nnx.jit`. It uses `nnx.split` / `nnx.update` to thread all
`nnx.Variable` state through XLA as mutable arguments, writing updates back
to Python objects after each call.

```python
# ❌ Wrong — KV cache writes are silently discarded
@jax.jit
def forward(model, kv_cache, ...): ...

# ✅ Correct — KV cache writes propagate back
@nnx.jit
def forward(model, kv_cache, ...): ...
```

---

### 2. `last_indices` in Prefill

During prefill, `T_total` tokens are processed but only the **last token per
sequence** should produce a logit. `ParallelLMHead` accepts `last_indices`:

```python
def __call__(self, x, last_indices=None):
    if last_indices is not None:
        x = x[last_indices]   # [num_seqs, hidden]
    return x @ self.effective_weight.T
```

**Previous bug**: `_jit_prefill` did not pass `last_indices`, returning logits
for all `T_pad` tokens. Incorrect for multi-sequence batches.

**Fix**: `last_indices` is now threaded from `_run_prefill` through
`_jit_prefill` into `LlamaForCausalLM.__call__`.

---

### 3. `seq_lens` Padding Value in Decode

**Previous bug**: padding rows used `val=1`, making them appear to have
sequence length 1 and potentially attending to position 0 of the KV cache.

**Fix**: `val=0` — padding sequences have no valid context positions, all
attention logits become `-inf`, softmax output is zero.

---

### 4. Residual Connection Order (Pre-Norm)

**Previous bug**: `attn_out` was passed directly into `post_attention_layernorm`
via the fused-residual API, bypassing the attention residual add entirely.
Every layer’s hidden state was just the raw attention output, not `attn + x`.

**Fix**: Explicit pre-norm pattern everywhere:
```python
residual = x
x = self.input_layernorm(x)
x = self.self_attn(x, ...)
x = x + residual          # ← residual add after attention

residual = x
x = self.post_attention_layernorm(x)
x = self.mlp(x)
x = x + residual          # ← residual add after MLP
```

---

### 5. Qwen3 QK-Norm (Architecture-Specific)

Qwen3 applies a **per-head RMSNorm** to Q and K before RoPE:
```python
q = self.q_norm(q)   # (T, num_heads, head_dim)
k = self.k_norm(k)   # (T, num_kv_heads, head_dim)
q, k = self.rope(q, k, positions)
```
Weights: `self_attn.q_norm.weight` and `self_attn.k_norm.weight`, shape `(head_dim,)`.

Without QK-Norm, Q/K magnitudes are uncontrolled from layer 0. Attention scores
diverge, hidden states explode by layer 2, and all output logits are garbage.
Symptom: `act_out max` jumps from ~3 to ~95 at layer 2, then collapses to ~0.02
at layer 3 as RMSNorm tries to compensate.

**Fix**: `PerHeadRMSNorm` class in `layernorm.py`; applied in `LlamaAttention`
before RoPE. Graceful fallback to identity for plain Llama models (weight stays
ones when `q_norm.weight` / `k_norm.weight` are absent from checkpoint).

---

### 6. PyTorch → JAX API Reference

| PyTorch | JAX |
|---------|-----|
| `tensor.fill_(0)` | `jnp.zeros_like(tensor)` |
| `tensor[i] = val` (in-place) | `tensor.at[i].set(val)` |
| `torch.jit.script` | `jax.jit` / `nnx.jit` |
| `nn.Module` | `nnx.Module` |
| `tensor.cuda()` | JAX auto-selects device via `jax.devices()` |
| `torch.no_grad()` | Not needed (JAX is functional; grad opt-in via `jax.grad`) |
| `model.parameters()` | `nnx.variables(model, nnx.Param)` |
| `nn.ModuleList` | `nnx.List` (plain `list` raises in Flax NNX pytree mode) |
| `torch.multinomial` | `jax.random.categorical` |
| `tensor.cos()` / `.sin()` | `jnp.cos(x)` / `jnp.sin(x)` (no instance methods) |
| dynamic shape ops | `jax.lax.dynamic_slice` or padding + masking |
| `torch.cuda.synchronize()` | `jax.effects_barrier()` |

---

### 7. Known Limitations

- **No FlashAttention**: Phase 8 will add `jax.nn.dot_product_attention`.
- **TP > 1 not tested**: Tensor parallelism compiles for `tp_size=1`; multi-GPU
  TP requires `jax.lax.psum` allreduce in linear layers (scaffolding exists).
- **O(log N) recompilations**: Each distinct padded shape triggers XLA recompile.
  Padding to next-power-of-2 bounds this; persistent compilation cache planned.
- **No CUDA graphs**: Each call goes through full XLA dispatch. `jax.jit` caches
  compiled kernels but does not currently use CUDA graph capture.

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
    ├── test_generation.py               # Phase 8 ⬜
    └── test_throughput.py               # Phase 8 ⬜
```

### Numerical Equivalence Tolerances
- **Deterministic ops** (layernorm, linear, rope): atol=1e-3, rtol=1e-3
- **Attention softmax**: atol=1e-2
- **Sampling** (stochastic): compare distributions, not exact samples

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

| Date | Phase | Module | Commit | Notes |
|------|-------|--------|--------|-------|
| 2026-05-27 | 0 | Project scaffold | — | pyproject, CI, skeleton |
| 2026-05-27 | 1.1 | Config & SamplingParams | — | assert-based validation |
| 2026-05-27 | 1.2 | Activation | — | atol=1e-5 |
| 2026-05-27 | 1.3 | RMSNorm | — | fused add+norm |
| 2026-05-27 | 1.4 | RoPE | — | LRU cache, jit |
| 2026-05-27 | 1 | Bug fix: RoPE+Norm | `a930b22` | jnp.cos/sin; dtype cast |
| 2026-05-27 | 2 | Linear + Embed/LMHead | — | TP sharding, weight tying |
| 2026-05-27 | 3 | PagedKVCache + Attention + Sampler | `cb5f06b` | prefill/decode/GQA |
| 2026-05-27 | 3 | Bug fix: test index | `403d934` | `k_out[seq,pos,head,:]` |
| 2026-05-27 | 4 | LLaMA model stack | `cd3e901` | full MLP/Attn/Decoder |
| 2026-05-27 | 4 | Bug fix: nnx.List | `895286d` | `nnx.List`; residual test |
| 2026-05-27 | 5 | HF Loader + Registry | `7caae14` | safetensors/bin shards |
| 2026-05-27 | 6 | Engine skeleton | — | sequence/block/sched/runner |
| 2026-05-27 | 6 | Bug fix: EngineConfig flat | `e0bc16c` | flat fields + max_num_batched_tokens |
| 2026-05-27 | 6 | Bug fix: SamplingParams assert | `2f269d3` | assert not ValueError; stop_token_ids |
| 2026-05-28 | 7 | LLMEngine + LLM API | — | generate_all, RequestOutput, 9 tests |
| 2026-05-28 | 7 | Bug fix: residual connections | — | explicit pre-norm residual pattern |
| 2026-05-28 | 7 | Bug fix: nnx.jit KV cache | — | jax.jit → nnx.jit in model_runner |
| 2026-05-28 | 7 | Bug fix: last_indices prefill | — | lm_head returns [num_seqs, vocab] |
| 2026-05-28 | 7 | Bug fix: seq_lens pad val=0 | — | prevent phantom context in decode |
| 2026-05-28 | 4 | Bug fix: Qwen3 QK-Norm | `7378e74` | PerHeadRMSNorm; q/k norm before RoPE |
| 2026-05-28 | 7 | GPU demo: multi-QA showcase | `221ff38` | 10 Q&A, timing, Qwen3-0.6B verified ✅ |

---

## References
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax NNX Guide](https://flax.readthedocs.io/en/latest/nnx_basics.html)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [Flash Attention in JAX](https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.dot_product_attention.html)
- [Original nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [chex testing library](https://github.com/google-deepmind/chex)
- [Qwen3 HuggingFace Model](https://huggingface.co/Qwen/Qwen3-0.6B)
