"""Rotary positional embedding (JAX port of nanovllm/layers/rotary_embedding.py).

Status: ✅ Done (Phase 1.4)
"""
from __future__ import annotations
from functools import lru_cache
import jax
import jax.numpy as jnp
from flax import nnx


def apply_rotary_emb(
    q: jax.Array,
    k: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Apply rotary embeddings to query and key tensors.

    Args:
        q:   (seq_len, num_heads, head_dim)
        k:   (seq_len, num_kv_heads, head_dim)
        cos: (seq_len, head_dim)
        sin: (seq_len, head_dim)
    Returns:
        (q_rot, k_rot) with same shapes as inputs
    """
    def rotate_half(x: jax.Array) -> jax.Array:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return jnp.concatenate([-x2, x1], axis=-1)

    cos_ = cos[:, None, :]  # (seq, 1, head_dim)
    sin_ = sin[:, None, :]
    q_rot = q * cos_ + rotate_half(q) * sin_
    k_rot = k * cos_ + rotate_half(k) * sin_
    return q_rot, k_rot


@lru_cache(maxsize=8)
def get_rope(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[jax.Array, jax.Array]:
    """Build and cache (cos, sin) tables for RoPE.

    Args:
        head_dim:    per-head feature dimension (must be even)
        max_seq_len: maximum sequence length to pre-compute
        base:        RoPE base frequency
        dtype:       output dtype
    Returns:
        (cos, sin) each of shape (max_seq_len, head_dim)
    """
    assert head_dim % 2 == 0
    theta = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(positions, theta)          # (seq, head_dim/2)
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # (seq, head_dim)
    return emb.cos().astype(dtype), emb.sin().astype(dtype)


class RotaryEmbedding(nnx.Module):
    """Stateful NNX wrapper that caches and applies RoPE.

    Args:
        head_dim:    per-head dimension
        max_seq_len: maximum sequence length
        base:        RoPE base frequency
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
    ) -> None:
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        cos, sin = get_rope(head_dim, max_seq_len, base)
        # Store as non-trainable buffers
        self.cos_cached = nnx.Variable(cos)
        self.sin_cached = nnx.Variable(sin)

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        positions: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Apply RoPE to q and k at the given positions.

        Args:
            q:         (seq_len, num_heads, head_dim)
            k:         (seq_len, num_kv_heads, head_dim)
            positions: (seq_len,) int32 position indices
        Returns:
            (q_rot, k_rot)
        """
        cos = self.cos_cached.get_value()[positions]
        sin = self.sin_cached.get_value()[positions]
        return apply_rotary_emb(q, k, cos, sin)
