"""Token embedding and LM head (JAX port of nanovllm/layers/embed_head.py).

Status: ✅ Done (Phase 2.2)
"""
from __future__ import annotations
from typing import Optional
import jax
import jax.numpy as jnp
from flax import nnx


def _param_array(param) -> jax.Array:
    """Extract the raw jax.Array from an nnx.Param regardless of Flax version.

    nnx.Param.__getitem__(Ellipsis) changed behaviour across Flax versions:
    - Old Flax: returns the underlying array directly.
    - New Flax: returns a wrapped Variable object, not a bare array.
    Using .raw_value (new) / .value (deprecated) / direct cast avoids this.
    """
    if hasattr(param, 'raw_value'):
        return jnp.asarray(param.raw_value)
    if hasattr(param, 'get_value'):
        return jnp.asarray(param.get_value())
    if hasattr(param, 'value'):
        return jnp.asarray(param.value)
    # Already a plain array
    return jnp.asarray(param)


class VocabParallelEmbedding(nnx.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tp_size: int = 1,
        tp_rank: int = 0,
    ) -> None:
        assert num_embeddings % tp_size == 0
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.num_embeddings_per_partition = num_embeddings // tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nnx.Param(jnp.zeros((self.num_embeddings_per_partition, embedding_dim)))

    def load_weight(self, weight: jax.Array) -> None:
        self.weight = nnx.Param(jnp.array(weight[self.vocab_start_idx:self.vocab_end_idx, :]))

    def __call__(self, x: jax.Array) -> jax.Array:
        w = _param_array(self.weight)
        if self.tp_size == 1:
            return w[x]
        mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
        local_x = jnp.where(mask, x - self.vocab_start_idx, 0)
        y = w[local_x]
        return jnp.where(mask[..., None], y, jnp.zeros_like(y))


class ParallelLMHead(nnx.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tp_size: int = 1,
        tp_rank: int = 0,
    ) -> None:
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.num_embeddings_per_partition = num_embeddings // tp_size
        self.weight = nnx.Param(jnp.zeros((self.num_embeddings_per_partition, embedding_dim)))
        self._tied_embed: nnx.data = nnx.data(None)

    def tie_weights(self, embed: VocabParallelEmbedding) -> None:
        assert embed.num_embeddings == self.num_embeddings
        assert embed.embedding_dim == self.embedding_dim
        self._tied_embed = nnx.data(embed)

    def load_weight(self, weight: jax.Array) -> None:
        start = self.tp_rank * self.num_embeddings_per_partition
        end = start + self.num_embeddings_per_partition
        self.weight = nnx.Param(jnp.array(weight[start:end, :]))

    @property
    def effective_weight(self) -> jax.Array:
        """Return the weight matrix as a plain jax.Array."""
        tied = self._tied_embed
        # unwrap nnx.data wrapper if present
        if hasattr(tied, 'value'):
            tied = tied.value
        if tied is not None:
            return _param_array(tied.weight)
        return _param_array(self.weight)

    def __call__(
        self,
        x: jax.Array,
        last_indices: Optional[jax.Array] = None,
    ) -> jax.Array:
        if last_indices is not None:
            x = x[last_indices]
        return x @ self.effective_weight.T
