"""RMSNorm (JAX port of nanovllm/layers/layernorm.py).

Status: ✅ Done (Phase 1.3)
"""
from __future__ import annotations
from typing import Optional
import jax
import jax.numpy as jnp
from flax import nnx


class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalisation with optional fused residual add.

    Args:
        hidden_size: feature dimension
        eps:         numerical stability epsilon
        rngs:        Flax NNX RNG streams (unused here, accepted for API compat)
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        rngs: Optional[nnx.Rngs] = None,
    ) -> None:
        self.eps = eps
        self.hidden_size = hidden_size
        self.weight = nnx.Param(jnp.ones(hidden_size))

    def _norm(self, x: jax.Array) -> jax.Array:
        return x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)

    def __call__(
        self,
        x: jax.Array,
        residual: Optional[jax.Array] = None,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Apply RMSNorm.

        Args:
            x:        input tensor (..., hidden_size)
            residual: if provided, adds residual to x first (fused path),
                      and returns (normed, x_after_add) so the caller can
                      use x_after_add as the next residual.
        Returns:
            normed output, or (normed, residual) when residual is not None.
        """
        if residual is not None:
            x = x + residual
            residual = x
        out = self._norm(x.astype(jnp.float32)).astype(x.dtype) * self.weight[...]
        if residual is not None:
            return out, residual
        return out
