"""Token embedding and LM head (JAX port of nanovllm/layers/embed_head.py).

Status: ✅ Fixed (Phase 2.2 + weight-access consistency + tied-weight storage)
"""
from __future__ import annotations
from typing import Optional
import jax
import jax.numpy as jnp
from flax import nnx


def _param_array(param: nnx.Param) -> jax.Array:
    """Extract a plain jax.Array from an nnx.Param, compatible with all Flax versions.

    Flax >= 0.10 deprecates .raw_value in favour of .get_raw_value().
    Using .value (the traced Variable accessor) is the safest cross-version path
    and avoids creating a second copy of the data.
    """
    return jnp.asarray(param.value)


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
        arr = jnp.asarray(weight[self.vocab_start_idx:self.vocab_end_idx, :])
        self.weight = nnx.Param(arr)

    @property
    def weight_array(self) -> jax.Array:
        """Return the underlying weight as a plain jax.Array (single allocation)."""
        return _param_array(self.weight)

    def __call__(self, x: jax.Array) -> jax.Array:
        w = self.weight_array
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
        # Optional reference to a tied VocabParallelEmbedding.
        # Stored as a plain Python attribute (not nnx.Variable / nnx.data) so
        # that nnx.jit does not treat it as a trainable parameter while still
        # keeping the live reference across JIT retracing.
        self._tied_embed: Optional[VocabParallelEmbedding] = None

    def tie_weights(self, embed: VocabParallelEmbedding) -> None:
        assert embed.num_embeddings == self.num_embeddings
        assert embed.embedding_dim == self.embedding_dim
        self._tied_embed = embed

    def load_weight(self, weight: jax.Array) -> None:
        start = self.tp_rank * self.num_embeddings_per_partition
        end = start + self.num_embeddings_per_partition
        arr = jnp.asarray(weight[start:end, :])
        self.weight = nnx.Param(arr)

    @property
    def effective_weight(self) -> jax.Array:
        """Return the weight matrix as a plain jax.Array.

        Uses the tied embedding's weight when tie_word_embeddings=True,
        otherwise uses the LM head's own weight.  Both paths go through
        _param_array() to guarantee a single consistent allocation.
        """
        if self._tied_embed is not None:
            return _param_array(self._tied_embed.weight)
        return _param_array(self.weight)

    def __call__(
        self,
        x: jax.Array,
        last_indices: Optional[jax.Array] = None,
    ) -> jax.Array:
        ew = self.effective_weight
        if last_indices is not None:
            x = x[last_indices]
        return x @ ew.T
