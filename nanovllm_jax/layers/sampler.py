"""Token sampler (JAX port of nanovllm/layers/sampler.py).

Status: ✅ Done (Phase 3)

Design notes vs PyTorch original:
- Uses jax.random.PRNGKey instead of torch random state.
- temperature / top_k / top_p are all jit-compilable.
- Greedy (temperature=0) returns argmax directly.
"""
from __future__ import annotations
from typing import Optional
import jax
import jax.numpy as jnp
from flax import nnx


def greedy_sample(logits: jax.Array) -> jax.Array:
    """Greedy decoding: argmax over vocab dimension.

    Args:
        logits: (..., vocab_size)
    Returns:
        token ids (...,)
    """
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)


def temperature_sample(
    logits: jax.Array,
    key: jax.Array,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> jax.Array:
    """Sample next tokens with temperature / top-k / top-p filtering.

    Args:
        logits:      (batch, vocab_size) float32
        key:         JAX PRNGKey
        temperature: softmax temperature (>0). temperature=0 falls back to greedy.
        top_k:       keep only top-k logits (0 = disabled)
        top_p:       nucleus sampling threshold (1.0 = disabled)
    Returns:
        sampled token ids (batch,) int32
    """
    if temperature == 0.0:
        return greedy_sample(logits)

    logits = logits.astype(jnp.float32) / temperature

    # Top-k
    if top_k > 0:
        # Zero out all but top-k
        top_k_val = jnp.sort(logits, axis=-1)[..., -top_k]  # threshold value
        logits = jnp.where(logits >= top_k_val[..., None], logits, jnp.finfo(jnp.float32).min)

    # Top-p (nucleus)
    if top_p < 1.0:
        sorted_idx = jnp.argsort(logits, axis=-1, descending=True)
        sorted_logits = jnp.take_along_axis(logits, sorted_idx, axis=-1)
        cumprobs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
        # Remove tokens whose cumulative prob exceeds top_p
        # (shift by 1 so we always keep at least 1 token)
        remove = jnp.concatenate(
            [jnp.zeros_like(cumprobs[..., :1]), (cumprobs[..., :-1] >= top_p)],
            axis=-1,
        )
        sorted_logits = jnp.where(remove, jnp.finfo(jnp.float32).min, sorted_logits)
        # Scatter back to original order
        logits = jnp.zeros_like(logits).at[
            jnp.arange(logits.shape[0])[:, None], sorted_idx
        ].set(sorted_logits)

    return jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)


class Sampler(nnx.Module):
    """NNX wrapper for token sampling."""

    def __call__(
        self,
        logits: jax.Array,
        key: Optional[jax.Array] = None,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> jax.Array:
        """Sample from logits.

        Args:
            logits:      (batch, vocab_size)
            key:         PRNGKey (required if temperature > 0)
            temperature: 0 = greedy
            top_k:       top-k filter (0 = off)
            top_p:       nucleus threshold (1.0 = off)
        Returns:
            token ids (batch,) int32
        """
        if temperature == 0.0:
            return greedy_sample(logits)
        assert key is not None, "PRNGKey required for temperature sampling"
        return temperature_sample(logits, key, temperature, top_k, top_p)
