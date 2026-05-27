"""RMSNorm layer (JAX port of nanovllm/layers/layernorm.py).

Status: ✅ Done (Phase 1.3)

Original: PyTorch RMSNorm with two forward paths:
  - rms_forward(x)
  - add_rms_forward(x, residual)  -> fused add+norm
JAX: Flax NNX module, functional, jit-compilable.
All compute promoted to float32 internally, cast back to input dtype.
"""
import jax
import jax.numpy as jnp
from flax import nnx


class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization.

    Args:
        hidden_size: dimensionality of the last axis.
        eps: small constant for numerical stability.
        rngs: Flax NNX RNG streams (needed for nnx.Module init).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, *, rngs: nnx.Rngs) -> None:
        self.eps = eps
        self.weight = nnx.Param(jnp.ones(hidden_size))

    def _rms_norm(self, x: jax.Array) -> jax.Array:
        """Core RMSNorm computation (float32 internally)."""
        orig_dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        x_normed = x_f32 * jax.lax.rsqrt(var + self.eps)
        return (x_normed.astype(orig_dtype)) * self.weight.value.astype(orig_dtype)

    def __call__(
        self,
        x: jax.Array,
        residual: jax.Array | None = None,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Forward pass.

        Args:
            x: input tensor (..., hidden_size)
            residual: optional residual to fuse add before norm
        Returns:
            normed x, or (normed x, updated residual) if residual given.
        """
        if residual is None:
            return self._rms_norm(x)
        else:
            orig_dtype = x.dtype
            combined = x.astype(jnp.float32) + residual.astype(jnp.float32)
            new_residual = combined.astype(orig_dtype)
            return self._rms_norm(new_residual), new_residual
