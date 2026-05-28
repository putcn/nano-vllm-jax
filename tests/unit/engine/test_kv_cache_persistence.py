"""Regression tests: KV cache writes must persist across nnx.jit boundaries.

Bug #1 (fixed): jax.jit does not propagate Python-level object mutations
(e.g. self.cache = nnx.Variable(...)) back to the outer scope after a call.
This caused every decode step to read an all-zero KV cache, producing garbage
attention outputs and garbled tokens.

Fix: use nnx.jit instead of jax.jit. nnx.jit extracts nnx.Variable state,
threads it through XLA as mutable arrays, and writes updates back after
each call.

Tests in this file verify:
  1. A cache write inside nnx.jit is visible after the call returns.
  2. Decode reads the KV data that was written during prefill.
  3. Padding tokens (seq_lens=0) do NOT corrupt real sequences.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from flax import nnx
import pytest

from nanovllm_jax.layers.attention import PagedKVCache, Attention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cache(num_layers=2, num_kv_heads=2, head_dim=8,
               num_blocks=16, block_size=4) -> PagedKVCache:
    return PagedKVCache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=jnp.float32,
    )


# ---------------------------------------------------------------------------
# Test 1: cache write inside nnx.jit persists outside the call
# ---------------------------------------------------------------------------

@nnx.jit
def _write_cache(cache: PagedKVCache, layer: int,
                 bi: jax.Array, bo: jax.Array,
                 k: jax.Array, v: jax.Array) -> None:
    cache.write(layer, bi, bo, k, v)


def test_nnx_jit_cache_write_persists():
    """KV cache write inside nnx.jit must be visible after the call."""
    cache = make_cache()
    layer = 0
    bi = jnp.array([0], dtype=jnp.int32)
    bo = jnp.array([0], dtype=jnp.int32)
    k_val = jnp.ones((1, 2, 8), dtype=jnp.float32) * 3.14
    v_val = jnp.ones((1, 2, 8), dtype=jnp.float32) * 2.71

    # Before write: cache should be all zeros
    assert jnp.allclose(cache.cache.value[layer, 0, 0, 0], jnp.zeros((2, 8)))

    _write_cache(cache, layer, bi, bo, k_val, v_val)

    # After write: cache must reflect the written values
    k_stored = cache.cache.value[layer, 0, 0, 0]  # [num_kv_heads, head_dim]
    v_stored = cache.cache.value[layer, 1, 0, 0]
    assert jnp.allclose(k_stored, k_val[0]), (
        f"KV cache write not persisted! Expected {k_val[0]}, got {k_stored}"
    )
    assert jnp.allclose(v_stored, v_val[0])


# ---------------------------------------------------------------------------
# Test 2: jax.jit does NOT persist the write (documents the original bug)
# ---------------------------------------------------------------------------

@jax.jit
def _write_cache_jit(cache: PagedKVCache, layer: int,
                     bi: jax.Array, bo: jax.Array,
                     k: jax.Array, v: jax.Array) -> None:
    cache.write(layer, bi, bo, k, v)


def test_jax_jit_cache_write_does_not_persist():
    """Documents that jax.jit does NOT persist nnx.Variable mutations.

    This is the original bug: jax.jit traces functions purely-functionally.
    Any Python-level object mutation (self.cache = nnx.Variable(new_cache))
    inside a jitted function is NOT propagated back to the outer scope.
    """
    cache = make_cache()
    layer = 0
    bi = jnp.array([0], dtype=jnp.int32)
    bo = jnp.array([0], dtype=jnp.int32)
    k_val = jnp.ones((1, 2, 8), dtype=jnp.float32) * 9.99
    v_val = jnp.ones((1, 2, 8), dtype=jnp.float32) * 8.88

    _write_cache_jit(cache, layer, bi, bo, k_val, v_val)

    k_stored = cache.cache.value[layer, 0, 0, 0]
    # Under jax.jit the write is NOT persisted — cache should still be zeros.
    assert not jnp.allclose(k_stored, k_val[0]), (
        "Unexpected: jax.jit DID persist the cache write. "
        "JAX behaviour may have changed — review model_runner.py."
    )


# ---------------------------------------------------------------------------
# Test 3: decode reads the KV data written during prefill
# ---------------------------------------------------------------------------

@nnx.jit
def _prefill_and_decode(
    cache: PagedKVCache,
    attn: Attention,
    q_pre: jax.Array,
    k_pre: jax.Array,
    v_pre: jax.Array,
    q_dec: jax.Array,
    k_dec: jax.Array,
    v_dec: jax.Array,
    block_table: jax.Array,
    seq_lens_pre: jax.Array,
    seq_lens_dec: jax.Array,
    block_indices_pre: jax.Array,
    block_offsets_pre: jax.Array,
    block_indices_dec: jax.Array,
    block_offsets_dec: jax.Array,
):
    out_pre = attn(
        q_pre, k_pre, v_pre, cache,
        block_indices_pre, block_offsets_pre, seq_lens_pre,
        is_prefill=True,
        num_real_tokens=q_pre.shape[0],
    )
    out_dec = attn(
        q_dec, k_dec, v_dec, cache,
        block_indices_dec, block_offsets_dec, seq_lens_dec,
        is_prefill=False,
        block_table=block_table,
        num_real_seqs=1,
    )
    return out_pre, out_dec


def test_decode_reads_prefill_kv_cache():
    """Decode attention output must be non-zero because it reads the prefill cache."""
    NH, HD, BS = 2, 8, 4
    cache = make_cache(num_kv_heads=NH, head_dim=HD, block_size=BS)
    attn = Attention(num_heads=NH, head_dim=HD, num_kv_heads=NH, layer_idx=0)

    T = 3  # 3 prefill tokens
    key = jax.random.PRNGKey(0)
    q_pre = jax.random.normal(key, (T, NH, HD))
    k_pre = jax.random.normal(jax.random.PRNGKey(1), (T, NH, HD))
    v_pre = jax.random.normal(jax.random.PRNGKey(2), (T, NH, HD))

    q_dec = jax.random.normal(jax.random.PRNGKey(3), (1, NH, HD))
    k_dec = jax.random.normal(jax.random.PRNGKey(4), (1, NH, HD))
    v_dec = jax.random.normal(jax.random.PRNGKey(5), (1, NH, HD))

    # Block layout: tokens 0,1,2 fit in block 0 (positions 0,1,2)
    # decode token goes to block 0 position 3
    bi_pre = jnp.array([0, 0, 0], dtype=jnp.int32)
    bo_pre = jnp.array([0, 1, 2], dtype=jnp.int32)
    bi_dec = jnp.array([0], dtype=jnp.int32)
    bo_dec = jnp.array([3], dtype=jnp.int32)
    seq_lens_pre = jnp.array([T], dtype=jnp.int32)
    seq_lens_dec = jnp.array([T + 1], dtype=jnp.int32)
    block_table = jnp.array([[0]], dtype=jnp.int32)

    out_pre, out_dec = _prefill_and_decode(
        cache, attn,
        q_pre, k_pre, v_pre,
        q_dec, k_dec, v_dec,
        block_table,
        seq_lens_pre, seq_lens_dec,
        bi_pre, bo_pre, bi_dec, bo_dec,
    )

    # Decode output must be non-zero: it attended over the prefill KV cache.
    # If the cache was all-zeros (jax.jit bug), the softmax would be uniform
    # over zero-valued v vectors, producing ~zero output.
    assert not jnp.allclose(out_dec, jnp.zeros_like(out_dec), atol=1e-4), (
        "Decode output is all-zeros. KV cache from prefill was not persisted. "
        "Ensure model_runner uses nnx.jit."
    )


# ---------------------------------------------------------------------------
# Test 4: seq_lens=0 padding does not corrupt real sequences
# ---------------------------------------------------------------------------

def test_zero_seq_lens_padding_does_not_corrupt_decode():
    """Padding sequences (seq_lens=0) should produce zero attention output
    and must NOT affect the real sequences.
    """
    NH, HD, BS = 2, 8, 4
    cache = make_cache(num_kv_heads=NH, head_dim=HD, block_size=BS)
    attn = Attention(num_heads=NH, head_dim=HD, num_kv_heads=NH, layer_idx=0)

    # One real sequence (seq_len=4), one padding sequence (seq_len=0)
    # We call _decode directly with B_real=1 so padding is ignored.
    key = jax.random.PRNGKey(42)
    q = jax.random.normal(key, (1, NH, HD))
    # Build a non-trivial k/v cache for the real sequence
    k_ctx = jax.random.normal(jax.random.PRNGKey(10), (1, 4, NH, HD))
    v_ctx = jax.random.normal(jax.random.PRNGKey(11), (1, 4, NH, HD))
    seq_lens = jnp.array([4], dtype=jnp.int32)

    out_real = attn._decode(q, k_ctx, v_ctx, seq_lens)

    # Compute again with seq_lens=0 — output should be near-zero (no valid KV)
    seq_lens_zero = jnp.array([0], dtype=jnp.int32)
    out_zero = attn._decode(q, k_ctx, v_ctx, seq_lens_zero)

    # Real output should be meaningful (non-zero)
    assert not jnp.allclose(out_real, jnp.zeros_like(out_real), atol=1e-4)
    # Zero-seq-len output should be essentially zero (all logits = -inf → softmax = 0)
    # Numerically softmax([-inf,...,-inf]) in float32 can produce NaN or 0;
    # either way it must not equal out_real.
    assert not jnp.allclose(out_zero, out_real, atol=1e-4), (
        "seq_lens=0 padding produced same output as seq_lens=4. "
        "Padding may be corrupting real sequence results."
    )
