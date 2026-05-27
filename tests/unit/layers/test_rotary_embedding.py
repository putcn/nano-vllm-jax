"""Unit tests for Rotary Position Embedding (Phase 1.4).

Numerical equivalence verified against PyTorch reference.
Tolerances: atol=1e-4 for float32 (trig accumulation differences).
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp

from nanovllm_jax.layers.rotary_embedding import apply_rotary_emb, RotaryEmbedding, get_rope


def make_rope(head_size=64, max_pos=512, base=10000.0):
    return RotaryEmbedding(head_size, head_size, max_pos, base)


def torch_apply_rotary(x_np, cos_np, sin_np):
    try:
        import torch
        x = torch.tensor(x_np, dtype=torch.float32)
        cos = torch.tensor(cos_np, dtype=torch.float32)
        sin = torch.tensor(sin_np, dtype=torch.float32)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1*cos - x2*sin, x2*cos + x1*sin], dim=-1).numpy()
    except ImportError:
        half = x_np.shape[-1] // 2
        x1, x2 = x_np[..., :half], x_np[..., half:]
        return np.concatenate([x1*cos_np - x2*sin_np, x2*cos_np + x1*sin_np], axis=-1)


def test_apply_rotary_emb_shape():
    out = apply_rotary_emb(jnp.ones((8, 4, 64)), jnp.ones((8, 1, 32)), jnp.zeros((8, 1, 32)))
    assert out.shape == (8, 4, 64)


@pytest.mark.parametrize("seed", [0, 1, 5])
def test_apply_rotary_emb_numerical(seed):
    rng = np.random.default_rng(seed)
    head_size = 64
    x_np = rng.standard_normal((4, 2, head_size)).astype(np.float32)
    cos_np = rng.standard_normal((4, 1, head_size // 2)).astype(np.float32)
    sin_np = rng.standard_normal((4, 1, head_size // 2)).astype(np.float32)
    ref = torch_apply_rotary(x_np, cos_np, sin_np)
    out = np.array(apply_rotary_emb(jnp.array(x_np), jnp.array(cos_np), jnp.array(sin_np)))
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_apply_rotary_preserves_dtype_bfloat16():
    x = jnp.ones((4, 2, 32), dtype=jnp.bfloat16)
    out = apply_rotary_emb(x, jnp.ones((4, 1, 16), dtype=jnp.bfloat16), jnp.zeros((4, 1, 16), dtype=jnp.bfloat16))
    assert out.dtype == jnp.bfloat16


def test_identity_with_zero_sin():
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((4, 2, 64)).astype(np.float32)
    out = np.array(apply_rotary_emb(jnp.array(x_np), jnp.ones((4, 1, 32)), jnp.zeros((4, 1, 32))))
    np.testing.assert_allclose(out, x_np, atol=1e-6)


def test_rope_cache_shape():
    assert make_rope(head_size=64, max_pos=512).cos_sin_cache.shape == (512, 1, 64)


def test_rope_forward_shape():
    rope = make_rope(head_size=64, max_pos=128)
    q_rot, k_rot = rope(jnp.arange(10), jnp.ones((10, 4, 64)), jnp.ones((10, 2, 64)))
    assert q_rot.shape == (10, 4, 64)
    assert k_rot.shape == (10, 2, 64)


@pytest.mark.parametrize("seed", [0, 3])
def test_rope_numerical_match(seed):
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available for numerical reference")

    head_size, max_pos, base, seq_len = 64, 256, 10000.0, 16
    rope_jax = make_rope(head_size=head_size, max_pos=max_pos, base=base)

    inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2, dtype=torch.float) / head_size))
    t = torch.arange(max_pos, dtype=torch.float)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos_pt = freqs.cos().unsqueeze(1)
    sin_pt = freqs.sin().unsqueeze(1)

    rng = np.random.default_rng(seed)
    q_np = rng.standard_normal((seq_len, 2, head_size)).astype(np.float32)
    k_np = rng.standard_normal((seq_len, 1, head_size)).astype(np.float32)
    positions_np = np.arange(seq_len)

    cos_sel = cos_pt[positions_np]
    sin_sel = sin_pt[positions_np]

    def pt_apply(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos_sel - x2 * sin_sel, x2 * cos_sel + x1 * sin_sel], dim=-1)

    q_ref = pt_apply(torch.tensor(q_np)).numpy()
    k_ref = pt_apply(torch.tensor(k_np)).numpy()

    q_jax, k_jax = rope_jax(jnp.array(positions_np), jnp.array(q_np), jnp.array(k_np))
    np.testing.assert_allclose(np.array(q_jax), q_ref, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.array(k_jax), k_ref, atol=1e-4, rtol=1e-4)


def test_long_sequence():
    rope = make_rope(head_size=128, max_pos=8192)
    q_rot, _ = rope(jnp.arange(4096), jnp.ones((4096, 4, 128)), jnp.ones((4096, 4, 128)))
    assert q_rot.shape == (4096, 4, 128)


@pytest.mark.parametrize("head_size", [32, 64, 128])
def test_different_head_dims(head_size):
    rope = RotaryEmbedding(head_size, head_size, 512, 10000.0)
    q_r, k_r = rope(jnp.arange(8), jnp.ones((8, 2, head_size)), jnp.ones((8, 2, head_size)))
    assert q_r.shape == (8, 2, head_size)


def test_get_rope_cache():
    assert get_rope(64, 64, 512, 10000.0) is get_rope(64, 64, 512, 10000.0)


def test_rope_jit_compilable():
    rope = make_rope(head_size=64, max_pos=128)

    @jax.jit
    def fwd(positions, q, k):
        return rope(positions, q, k)

    q_r, k_r = fwd(jnp.arange(8), jnp.ones((8, 4, 64)), jnp.ones((8, 2, 64)))
    assert q_r.shape == (8, 4, 64)
