# Migration Guide: nano-vllm PyTorch → nano-vllm-jax

This document describes design decisions, API changes, and gotchas when
porting nano-vllm from PyTorch to JAX/Flax NNX.

---

## 1. JAX Functional Purity and `nnx.jit` (Critical)

### The Problem

PyTorch operations are **eager** and **stateful** by default. In-place tensor
mutations (e.g. `cache[layer, 0, bi, bo] = keys`) are immediately visible to
all subsequent Python code.

JAX's `jax.jit` compiles a **pure function**. Any Python-level object mutation
that occurs inside a `jax.jit`-decorated function — including assigning a new
value to an attribute of a Python object — is **NOT propagated back** to the
outer scope after the call returns.

### Consequence for KV Cache

`PagedKVCache.write()` contains:

```python
new_cache = jax.lax.fori_loop(0, num_real, body, cache)
self.cache = nnx.Variable(new_cache)   # Python-level object mutation
```

When called inside a `jax.jit`-compiled forward function:

- The `fori_loop` is correctly compiled into XLA and produces the right
  updated cache array **inside the JIT trace**.
- But the assignment `self.cache = nnx.Variable(new_cache)` is a Python
  object mutation. JIT does not carry Python side-effects back to the caller.
- After the JIT call returns, the outer `kv_cache.cache` is **still the
  original all-zeros array**.
- Every decode step reads an all-zero KV cache, so attention output is
  garbage (all queries attend uniformly over zero-valued value vectors),
  producing completely incoherent token sequences.

### Fix: Use `nnx.jit`

Flax NNX provides `nnx.jit` which is NNX-aware:

1. Before calling the function, `nnx.jit` uses `nnx.split` to extract all
   `nnx.Variable` state from every NNX module in the call arguments into a
   pure pytree of JAX arrays.
2. The extracted state is passed through XLA as **mutable function arguments**.
3. After the call, `nnx.jit` uses `nnx.update` to write the updated state
   back into the Python objects.

This means `self.cache = nnx.Variable(new_cache)` produces the correct
behaviour: the write is visible in the outer Python scope after the call.

```python
# ❌ Wrong — KV cache writes are silently discarded
@jax.jit
def forward(model, kv_cache, ...):
    return model(kv_cache=kv_cache, ...)

# ✅ Correct — KV cache writes are propagated back
@nnx.jit
def forward(model, kv_cache, ...):
    return model(kv_cache=kv_cache, ...)
```

All forward functions in `model_runner.py` (`_jit_prefill`, `_jit_decode`)
now use `nnx.jit`.

---

## 2. `last_indices` in Prefill

### Background

During prefill, multiple sequences are packed into a single flat token array
of length `T_total`. The model runs over all `T_total` hidden states, but we
only want the **last token per sequence** to compute the next-token logit.

`ParallelLMHead.__call__` accepts an optional `last_indices` argument:

```python
def __call__(self, x, last_indices=None):
    if last_indices is not None:
        x = x[last_indices]   # [num_seqs, hidden]
    return x @ self.effective_weight.T  # [num_seqs, vocab]
```

### Previous Bug

`_jit_prefill` did not pass `last_indices`, so `lm_head` returned logits for
all `T_pad` tokens. The model runner then indexed `logits[last_idx[i]]` which
happens to work for a **single sequence** but produces wrong results for
multi-sequence batches when shapes don't align as expected.

### Fix

`last_indices` is now passed from `_run_prefill` through `_jit_prefill` into
`LlamaForCausalLM.__call__`. The model now returns shape `[num_seqs, vocab]`
and the runner indexes with `logits[i]` (plain enumerate).

---

## 3. `seq_lens` Padding in Decode

### Previous Bug

`_run_decode` padded `seq_lens` with `val=1`:

```python
seq_lens = self._pad1d(seq_lens, B_pad, val=1)  # ❌
```

Padding rows appeared to have sequence length 1, allowing them to attend to
position 0 of the KV cache (which may hold a real token from a different
sequence). Although `Attention.__call__` only calls `_decode` for `B_real`
sequences, the padded seq_lens with `val=1` could interfere in edge cases.

### Fix

```python
seq_lens = self._pad1d(seq_lens, B_pad, val=0)  # ✅
```

Padding sequences with `seq_len=0` have no valid context positions. All
attention logits become `-inf`, softmax output is `0`, and the padding rows
contribute nothing to the real sequences' outputs.

---

## 4. PyTorch → JAX Tensor Operations Reference

| PyTorch | JAX |
|---------|-----|
| `tensor.fill_(0)` | `jnp.zeros_like(tensor)` |
| `tensor[i] = val` (in-place) | `tensor.at[i].set(val)` |
| `torch.jit.script` | `jax.jit` / `nnx.jit` |
| `nn.Module` | `nnx.Module` |
| `tensor.cuda()` | JAX auto-selects device via `jax.devices()` |
| `torch.no_grad()` | Not needed (JAX is functional; grad opt-in via `jax.grad`) |
| `model.parameters()` | `nnx.variables(model, nnx.Param)` |

---

## 5. RoPE Implementation

The original PyTorch code uses interleaved RoPE (GPT-NeoX style):
- Frequency table: `emb = concat([freqs, freqs], axis=-1)` where `freqs` has
  shape `(seq, head_dim/2)`.
- Rotation: `[-x2, x1]` where `x1 = x[..., :half]`, `x2 = x[..., half:]`.

This style is matched exactly in `rotary_embedding.py`. Qwen3 uses the same
style (confirmed against the HuggingFace Qwen3 implementation).

---

## 6. Known Limitations

- **No FlashAttention**: Phase 6 will add `jax.nn.dot_product_attention`.
- **TP > 1 not tested**: Tensor parallelism works for `tp_size=1`; multi-GPU
  TP requires `jax.lax.psum` allreduce in linear layers.
- **No CUDA graph / persistent compilation**: Each distinct (shape, num_real)
  combination triggers a recompilation. Padding to next-power-of-2 reduces
  this to O(log N) shapes.
