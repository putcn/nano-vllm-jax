"""Token embedding and LM head (JAX port of nanovllm/layers/embed_head.py).

Status: ✅ Done (Phase 2.2)
"""
from __future__ import annotations
from typing import Optional
import jax
import jax.numpy as jnp
from flax import nnx


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
        # _weight_cache bypasses nnx.Param getter instability; always a plain jax.Array
        object.__setattr__(self, '_weight_cache', None)

    def load_weight(self, weight: jax.Array) -> None:
        arr = jnp.array(weight[self.vocab_start_idx:self.vocab_end_idx, :])
        self.weight = nnx.Param(arr)
        object.__setattr__(self, '_weight_cache', arr)

    def _get_weight(self) -> jax.Array:
        """Return the weight as a plain jax.Array, always."""
        cached = object.__getattribute__(self, '_weight_cache')
        if cached is not None:
            return cached
        # Fallback for freshly-constructed (not yet loaded) modules
        w = self.weight
        if hasattr(w, 'get_raw_value'):
            return jnp.asarray(w.get_raw_value())
        if hasattr(w, 'value'):
            return jnp.asarray(w.value)
        return jnp.asarray(w)

    def __call__(self, x: jax.Array) -> jax.Array:
        w = self._get_weight()
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
        object.__setattr__(self, '_weight_cache', None)

    def tie_weights(self, embed: VocabParallelEmbedding) -> None:
        assert embed.num_embeddings == self.num_embeddings
        assert embed.embedding_dim == self.embedding_dim
        self._tied_embed = nnx.data(embed)

    def load_weight(self, weight: jax.Array) -> None:
        start = self.tp_rank * self.num_embeddings_per_partition
        end = start + self.num_embeddings_per_partition
        arr = jnp.array(weight[start:end, :])
        self.weight = nnx.Param(arr)
        object.__setattr__(self, '_weight_cache', arr)

    def _get_weight(self) -> jax.Array:
        """Return the effective weight as a plain jax.Array, always stable."""
        # Check for tied embedding first
        tied = self._tied_embed
        if hasattr(tied, 'value'):
            tied = tied.value
        if tied is not None:
            return tied._get_weight()
        # Use local cache
        cached = object.__getattribute__(self, '_weight_cache')
        if cached is not None:
            return cached
        # Fallback
        w = self.weight
        if hasattr(w, 'get_raw_value'):
            return jnp.asarray(w.get_raw_value())
        if hasattr(w, 'value'):
            return jnp.asarray(w.value)
        return jnp.asarray(w)

    @property
    def effective_weight(self) -> jax.Array:
        """Return the weight matrix as a plain jax.Array."""
        return self._get_weight()

    def __call__(
        self,
        x: jax.Array,
        last_indices: Optional[jax.Array] = None,
    ) -> jax.Array:
        if last_indices is not None:
            x = x[last_indices]
        return x @ self._get_weight().T
