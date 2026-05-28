"""Activation functions (JAX port of nanovllm/layers/activation.py).

Status: ✅ Done (Phase 1.2)
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from flax import nnx


def silu_and_mul(x: jax.Array) -> jax.Array:
    """SwiGLU gating: silu(x[..., :half]) * x[..., half:].

    Args:
        x: (..., 2 * d) array
    Returns:
        (..., d) array
    """
    half = x.shape[-1] // 2
    return jax.nn.silu(x[..., :half]) * x[..., half:]


class SiluAndMul(nnx.Module):
    """NNX wrapper around silu_and_mul for use inside model graphs."""

    def __call__(self, x: jax.Array) -> jax.Array:
        return silu_and_mul(x)
