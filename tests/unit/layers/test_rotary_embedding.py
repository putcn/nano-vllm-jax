"""Unit tests for Rotary Positional Embedding (Phase 1.4)."""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from nanovllm_jax.layers.rotary_embedding import apply_rotary_emb, get_rope, RotaryEmbedding


def rand(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def torch_rope(q_np, k_np, positions_np, head_dim, base=10000.0):
    try:
        import torch
        theta = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        pos = positions_np.astype(np.float32)
        freqs = np.outer(pos, theta)
        emb = np.concatenate([freqs, freqs], axis=-1)
        cos = np.cos(emb)
        sin = np.sin(emb)

        def rotate(x, c, s):
            half = x.shape[-1] // 2
            x1, x2 = x[..., :half], x[..., half:]
            rot = np.concatenate([-x2, x1], axis=-1)
            return x * c[:, None, :] + rot * s[:, None, :]

        return rotate(q_np, cos, sin), rotate(k_np, cos, sin)
    except Exception:
        return q_np, k_np


def test_identity_when_sin_zero():
    """When sin=0 and cos=1, rotary emb is identity."""
    seq, nh, hd = 4, 2, 8
    q = jnp.array(rand((seq, nh, hd)))
    k = jnp.array(rand((seq, 1, hd)))
    cos = jnp.ones((seq, hd))
    sin = jnp.zeros((seq, hd))
    q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
    np.testing.assert_allclose(np.array(q_rot), np.array(q), atol=1e-6)
    np.testing.assert_allclose(np.array(k_rot), np.array(k), atol=1e-6)


@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_get_rope_shape(head_dim):
    cos, sin = get_rope(head_dim, 512)
    assert cos.shape == (512, head_dim)
    assert sin.shape == (512, head_dim)


def test_get_rope_lru_cache():
    a = get_rope(64, 512)
    b = get_rope(64, 512)
    assert a[0] is b[0]


def test_long_sequence():
    cos, sin = get_rope(64, 4096)
    assert cos.shape[0] == 4096


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_numerical_vs_reference(seed):
    seq, nh, hd = 8, 4, 64
    q_np = rand((seq, nh, hd), seed)
    k_np = rand((seq, 2, hd), seed + 1)
    positions = np.arange(seq)
    cos, sin = get_rope(hd, seq)
    q_rot, k_rot = apply_rotary_emb(
        jnp.array(q_np), jnp.array(k_np),
        cos[positions], sin[positions]
    )
    ref_q, ref_k = torch_rope(q_np, k_np, positions, hd)
    np.testing.assert_allclose(np.array(q_rot), ref_q, atol=1e-4)
    np.testing.assert_allclose(np.array(k_rot), ref_k, atol=1e-4)


def test_rotary_embedding_module():
    rope = RotaryEmbedding(head_dim=64, max_seq_len=512)
    seq, nh, hd = 8, 4, 64
    q = jnp.array(rand((seq, nh, hd)))
    k = jnp.array(rand((seq, 2, hd)))
    pos = jnp.arange(seq)
    q_rot, k_rot = rope(q, k, pos)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_jit():
    rope = RotaryEmbedding(head_dim=32, max_seq_len=128)
    jitted = nnx.jit(rope)
    q = jnp.array(rand((4, 2, 32)))
    k = jnp.array(rand((4, 1, 32)))
    pos = jnp.arange(4)
    q_rot, k_rot = jitted(q, k, pos)
    assert q_rot.shape == q.shape
