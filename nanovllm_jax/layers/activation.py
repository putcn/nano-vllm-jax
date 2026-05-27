"""Activation functions (JAX port of nanovllm/layers/activation.py).

Status: ✅ Done (Phase 1.2)

Original: SiluAndMul wraps torch.compile + F.silu * gate.
JAX: pure function, jit-compilable, no state.
"""
import jax
import jax.numpy as jnp
from flax import nnx


def silu_and_mul(x: jax.Array) -> jax.Array:
    """SwiGLU activation: silu(x[:half]) * x[half:].

    Args:
        x: (..., 2*d) array
    Returns:
        (..., d) array
    """
    half = x.shape[-1] // 2
    gate, val = x[..., :half], x[..., half:]
    return jax.nn.silu(gate) * val


class SiluAndMul(nnx.Module):
    """Stateless Flax NNX wrapper around silu_and_mul for use in model graphs."""

    def __call__(self, x: jax.Array) -> jax.Array:
        return silu_and_mul(x)
