"""Token embedding and LM head (JAX port of nanovllm/layers/embed_head.py).

Status: ✅ Fixed (weight tying: store embed as NNX submodule, not nnx.data)

Root cause of garbled output
-----------------------------
Previously ``_tied_embed`` was stored via ``nnx.data(embed)``.  Reading it
back required ``tied.value``, but ``.value`` is deprecated in current Flax
and silently returns ``None``.  This caused ``effective_weight`` to fall
through to the zero-initialised ``self.weight``, making every logit ~0 and
producing garbage predictions (repeating "Question / Instructions" loop).

Fix
---
Store ``VocabParallelEmbedding`` directly as ``self._tied_embed`` (a plain
NNX submodule reference).  Flax NNX Pytree machinery handles nested Module
references natively — no ``nnx.data()`` wrapper is needed or correct here.
``effective_weight`` now checks ``self._tied_embed is not None`` and calls
``self._tied_embed.weight_array`` directly.
"""
from __future__ import annotations
from typing import Optional
import jax
import jax.numpy as jnp
from flax import nnx


def _param_array(param: nnx.Param) -> jax.Array:
    """Extract a plain jax.Array from an nnx.Param, compatible with all Flax versions.

    Priority:
      1. get_value()   — Flax >= 0.10 canonical non-deprecated path
      2. param[...]    — nnx Variable __getitem__ (works on all recent versions)
    """
    if hasattr(param, 'get_value'):
        return jnp.asarray(param.get_value())
    return jnp.asarray(param[...])


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
        # None sentinel: no tied embedding yet.
        # Assigned to a VocabParallelEmbedding instance in tie_weights().
        # Stored directly as an NNX submodule — no nnx.data() wrapper needed.
        self._tied_embed: Optional[VocabParallelEmbedding] = None

    def tie_weights(self, embed: VocabParallelEmbedding) -> None:
        assert embed.num_embeddings == self.num_embeddings
        assert embed.embedding_dim == self.embedding_dim
        # Store the embedding module directly as an NNX submodule reference.
        # NNX Pytree machinery traverses nested Module attributes automatically,
        # so this is the correct pattern — no nnx.data() wrapper required.
        self._tied_embed = embed

    def load_weight(self, weight: jax.Array) -> None:
        start = self.tp_rank * self.num_embeddings_per_partition
        end = start + self.num_embeddings_per_partition
        arr = jnp.asarray(weight[start:end, :])
        self.weight = nnx.Param(arr)

    @property
    def effective_weight(self) -> jax.Array:
        """Return the weight matrix as a plain jax.Array.

        When tie_word_embeddings=True, reads directly from the tied
        VocabParallelEmbedding's weight_array.  Otherwise uses the LM
        head's own weight parameter.
        """
        if self._tied_embed is not None:
            return self._tied_embed.weight_array
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
