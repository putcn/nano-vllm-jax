"""Rotary Position Embedding (JAX port of nanovllm/layers/rotary_embedding.py).

Status: ✅ Done (Phase 1.4)

Original: pre-computes cos/sin cache as a buffer, applies per position.
JAX: cos_sin_cache stored as plain jnp array (not a param - not learned).
apply_rotary_emb is a pure function, jit-compilable.
get_rope is cached via functools.lru_cache.
"""
from functools import lru_cache
import jax
import jax.numpy as jnp
from flax import nnx


def apply_rotary_emb(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
) -> jax.Array:
    """Apply rotary embeddings to query or key tensor.

    Args:
        x:   (..., head_size) in any dtype
        cos: (..., head_size//2) slice of cos cache
        sin: (..., head_size//2) slice of sin cache
    Returns:
        rotated x, same shape and dtype as input.
    """
    orig_dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    half = x_f32.shape[-1] // 2
    x1, x2 = x_f32[..., :half], x_f32[..., half:]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return jnp.concatenate([y1, y2], axis=-1).astype(orig_dtype)


class RotaryEmbedding(nnx.Module):
    """Precomputed RoPE cache with positional lookup.

    Args:
        head_size: size of each attention head.
        rotary_dim: must equal head_size (full RoPE).
        max_position_embeddings: maximum sequence length.
        base: frequency base (default 10000).
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        assert rotary_dim == head_size, "Only full RoPE (rotary_dim == head_size) supported."
        self.head_size = head_size

        inv_freq = 1.0 / (
            base ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim)
        )
        t = jnp.arange(max_position_embeddings, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", t, inv_freq)  # (max_pos, rotary_dim//2)
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)
        # shape: (max_pos, 1, rotary_dim)  [1 = num_heads broadcast dim]
        self.cos_sin_cache = jnp.concatenate([cos, sin], axis=-1)[:, None, :]

    def __call__(
        self,
        positions: jax.Array,   # (seq_len,) int32
        query: jax.Array,       # (seq_len, num_heads, head_size)
        key: jax.Array,         # (seq_len, num_kv_heads, head_size)
    ) -> tuple[jax.Array, jax.Array]:
        """Apply RoPE to query and key."""
        cos_sin = self.cos_sin_cache[positions]          # (seq_len, 1, rotary_dim)
        half = cos_sin.shape[-1] // 2
        cos = cos_sin[..., :half]
        sin = cos_sin[..., half:]
        query_rot = apply_rotary_emb(query, cos, sin)
        key_rot = apply_rotary_emb(key, cos, sin)
        return query_rot, key_rot


@lru_cache(maxsize=8)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
) -> RotaryEmbedding:
    """Cached factory for RotaryEmbedding instances."""
    return RotaryEmbedding(head_size, rotary_dim, max_position, base)
