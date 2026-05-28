"""Unit tests for PagedKVCache and Attention (Phase 3)."""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from nanovllm_jax.layers.attention import PagedKVCache, Attention


def rand(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# PagedKVCache
# ---------------------------------------------------------------------------

def test_cache_init_shape():
    cache = PagedKVCache(
        num_layers=4, num_kv_heads=2, head_dim=16,
        num_blocks=8, block_size=4,
    )
    assert cache.cache.get_value().shape == (4, 2, 8, 4, 2, 16)


def test_cache_write_read_roundtrip():
    """Write tokens to layer 0, read back and check values."""
    cache = PagedKVCache(
        num_layers=2, num_kv_heads=2, head_dim=8,
        num_blocks=4, block_size=4, dtype=jnp.float32,
    )
    num_tokens = 3
    k = jnp.array(rand((num_tokens, 2, 8), seed=1))
    v = jnp.array(rand((num_tokens, 2, 8), seed=2))
    block_indices = jnp.array([0, 0, 1], dtype=jnp.int32)
    block_offsets = jnp.array([0, 1, 0], dtype=jnp.int32)

    cache.write(0, block_indices, block_offsets, k, v)

    # Read back via block_table
    block_table = jnp.array([[0, 1]], dtype=jnp.int32)  # 1 seq, 2 blocks
    seq_lens = jnp.array([3], dtype=jnp.int32)
    k_out, v_out = cache.read(0, block_table, seq_lens)

    # Shape: (1, 8, 2, 8) — 1 seq, 2*block_size=8 positions
    assert k_out.shape == (1, 8, 2, 8)
    np.testing.assert_allclose(np.array(k_out[0, 0]), np.array(k[0]), atol=1e-5)
    np.testing.assert_allclose(np.array(k_out[0, 1]), np.array(k[1]), atol=1e-5)
    np.testing.assert_allclose(np.array(k_out[0, 4]), np.array(k[2]), atol=1e-5)


def test_cache_multi_layer_isolation():
    """Writes to different layers must not interfere."""
    cache = PagedKVCache(
        num_layers=3, num_kv_heads=1, head_dim=4,
        num_blocks=4, block_size=2, dtype=jnp.float32,
    )
    k0 = jnp.ones((1, 1, 4)) * 1.0
    k1 = jnp.ones((1, 1, 4)) * 2.0
    v = jnp.zeros((1, 1, 4))
    bi = jnp.array([0], dtype=jnp.int32)
    bo = jnp.array([0], dtype=jnp.int32)

    cache.write(0, bi, bo, k0, v)
    cache.write(1, bi, bo, k1, v)

    bt = jnp.array([[0, 1]], dtype=jnp.int32)
    sl = jnp.array([1], dtype=jnp.int32)
    k_out0, _ = cache.read(0, bt, sl)
    k_out1, _ = cache.read(1, bt, sl)

    np.testing.assert_allclose(np.array(k_out0[0, 0]), np.ones(4) * 1.0, atol=1e-5)
    np.testing.assert_allclose(np.array(k_out1[0, 0]), np.ones(4) * 2.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Attention — prefill
# ---------------------------------------------------------------------------

def test_attention_prefill_shape():
    attn = Attention(num_heads=4, head_dim=16, num_kv_heads=2)
    cache = PagedKVCache(
        num_layers=1, num_kv_heads=2, head_dim=16,
        num_blocks=16, block_size=8,
    )
    T = 6
    q = jnp.array(rand((T, 4, 16)))
    k = jnp.array(rand((T, 2, 16)))
    v = jnp.array(rand((T, 2, 16)))
    bi = jnp.arange(T, dtype=jnp.int32) // 8
    bo = jnp.arange(T, dtype=jnp.int32) % 8
    sl = jnp.array([T], dtype=jnp.int32)

    out = attn(q, k, v, cache, bi, bo, sl, is_prefill=True)
    assert out.shape == (T, 4, 16)


def test_attention_prefill_causal():
    """Output for token 0 must not depend on tokens 1+."""
    attn = Attention(num_heads=2, head_dim=8, num_kv_heads=2)
    cache_a = PagedKVCache(num_layers=1, num_kv_heads=2, head_dim=8, num_blocks=8, block_size=8)
    cache_b = PagedKVCache(num_layers=1, num_kv_heads=2, head_dim=8, num_blocks=8, block_size=8)

    q = jnp.array(rand((4, 2, 8), seed=0))
    k = jnp.array(rand((4, 2, 8), seed=1))
    v = jnp.array(rand((4, 2, 8), seed=2))
    bi = jnp.zeros(4, dtype=jnp.int32)
    bo = jnp.arange(4, dtype=jnp.int32)
    sl = jnp.array([4], dtype=jnp.int32)
    sl1 = jnp.array([1], dtype=jnp.int32)

    out_full = attn(q, k, v, cache_a, bi, bo, sl, is_prefill=True)
    out_one = attn(q[:1], k[:1], v[:1], cache_b, bi[:1], bo[:1], sl1, is_prefill=True)

    np.testing.assert_allclose(
        np.array(out_full[0]), np.array(out_one[0]), atol=1e-4,
    )


def test_attention_prefill_gqa():
    """GQA: num_heads=8, num_kv_heads=2."""
    attn = Attention(num_heads=8, head_dim=16, num_kv_heads=2)
    cache = PagedKVCache(num_layers=1, num_kv_heads=2, head_dim=16, num_blocks=8, block_size=8)
    T = 5
    q = jnp.array(rand((T, 8, 16)))
    k = jnp.array(rand((T, 2, 16)))
    v = jnp.array(rand((T, 2, 16)))
    bi = jnp.zeros(T, dtype=jnp.int32)
    bo = jnp.arange(T, dtype=jnp.int32)
    sl = jnp.array([T], dtype=jnp.int32)
    out = attn(q, k, v, cache, bi, bo, sl, is_prefill=True)
    assert out.shape == (T, 8, 16)


# ---------------------------------------------------------------------------
# Attention — decode
# ---------------------------------------------------------------------------

def test_attention_decode_shape():
    attn = Attention(num_heads=4, head_dim=16, num_kv_heads=2)
    cache = PagedKVCache(
        num_layers=1, num_kv_heads=2, head_dim=16,
        num_blocks=16, block_size=8,
    )
    # Prefill 4 tokens first
    T = 4
    q_p = jnp.array(rand((T, 4, 16)))
    k_p = jnp.array(rand((T, 2, 16)))
    v_p = jnp.array(rand((T, 2, 16)))
    bi_p = jnp.zeros(T, dtype=jnp.int32)
    bo_p = jnp.arange(T, dtype=jnp.int32)
    sl = jnp.array([T], dtype=jnp.int32)
    attn(q_p, k_p, v_p, cache, bi_p, bo_p, sl, is_prefill=True)

    # Decode step: 1 new token
    q_d = jnp.array(rand((1, 4, 16)))
    k_d = jnp.array(rand((1, 2, 16)))
    v_d = jnp.array(rand((1, 2, 16)))
    bi_d = jnp.array([0], dtype=jnp.int32)
    bo_d = jnp.array([T], dtype=jnp.int32)
    sl_d = jnp.array([T + 1], dtype=jnp.int32)
    bt = jnp.array([[0, 1]], dtype=jnp.int32)

    out = attn(q_d, k_d, v_d, cache, bi_d, bo_d, sl_d,
               is_prefill=False, block_table=bt)
    assert out.shape == (1, 4, 16)


# ---------------------------------------------------------------------------
# Attention — scale
# ---------------------------------------------------------------------------

def test_attention_custom_scale():
    attn = Attention(num_heads=2, head_dim=8, scale=0.5)
    cache = PagedKVCache(num_layers=1, num_kv_heads=2, head_dim=8, num_blocks=4, block_size=8)
    q = jnp.array(rand((2, 2, 8)))
    k = jnp.array(rand((2, 2, 8)))
    v = jnp.array(rand((2, 2, 8)))
    bi = jnp.zeros(2, dtype=jnp.int32)
    bo = jnp.arange(2, dtype=jnp.int32)
    sl = jnp.array([2], dtype=jnp.int32)
    out = attn(q, k, v, cache, bi, bo, sl, is_prefill=True)
    assert out.shape == (2, 2, 8)
