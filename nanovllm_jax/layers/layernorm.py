"""RMSNorm (JAX port of nanovllm/layers/layernorm.py).

Status: ✅ Done (Phase 1.3)
"""
from __future__ import annotations
from typing import Optional
import jax
import jax.numpy as jnp
from flax import nnx


class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalisation.

    Dtype contract: output dtype always matches input x.dtype.
    The norm computation is promoted to float32 internally for numerical
    stability, then cast back before multiplying the learned weight.

    Args:
        hidden_size: feature dimension
        eps:         numerical stability epsilon
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
                      and returns (normed, x_after_add).
        Returns:
            normed output, or (normed, residual) when residual is not None.
        """
        if residual is not None:
            x = x + residual
            residual = x

        normed = self._norm(x.astype(jnp.float32)).astype(x.dtype)
        weight = self.weight[...].astype(x.dtype)
        out = normed * weight

        if residual is not None:
            return out, residual
        return out


class PerHeadRMSNorm(nnx.Module):
    """Per-head RMSNorm used by Qwen3 for Q and K tensors.

    Applies RMSNorm independently to each attention head.
    Input shape:  (seq_len, num_heads, head_dim)
    Output shape: (seq_len, num_heads, head_dim)  -- same

    The learned weight has shape (head_dim,) and is shared across heads
    and sequence positions (exactly as in HuggingFace Qwen3Attention).

    Args:
        head_dim: per-head feature dimension
        eps:      numerical stability epsilon
    """

    def __init__(self, head_dim: int, eps: float = 1e-6) -> None:
        self.head_dim = head_dim
        self.eps = eps
        self.weight = nnx.Param(jnp.ones(head_dim))

    def __call__(self, x: jax.Array) -> jax.Array:
        """x: (seq_len, num_heads, head_dim) -> same shape."""
        orig_dtype = x.dtype
        x_f = x.astype(jnp.float32)
        normed = x_f * jax.lax.rsqrt(
            jnp.mean(x_f ** 2, axis=-1, keepdims=True) + self.eps
        )
        w = self.weight[...].astype(orig_dtype)
        return normed.astype(orig_dtype) * w
