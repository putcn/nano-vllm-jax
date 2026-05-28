"""Token embedding and LM head (JAX port of nanovllm/layers/embed_head.py).

Status: ✅ Fixed (weight tying via object.__setattr__ bypass)

Weight tying design
-------------------
Flax NNX's Pytree machinery enforces strict type consistency on Module
attributes: a slot initialised as ``None`` (static) cannot later be assigned
a Module instance (data).  The ``nnx.data()`` wrapper is the official fix,
but unwrapping it reliably across Flax versions is fragile — ``.value``,
``.raw_value``, and ``.get_raw_value()`` all have deprecation or availability
issues depending on the exact Flax release.

Instead we bypass the NNX Pytree type system entirely by storing the tied
embedding reference with ``object.__setattr__``.  This is a well-known Python
pattern for injecting attributes that should be invisible to a class's
``__setattr__`` override.  NNX's Pytree traversal only sees attributes that
were set through its own ``__setattr__``, so ``_tied_embed_ref`` is simply
skipped during tracing — which is exactly what we want: the reference is a
pure Python pointer, not a JAX parameter.
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
        # Initialise the bypass slot via object.__setattr__ so NNX Pytree
        # machinery never sees it.  Must be set here (not skipped) so that
        # object.__getattribute__ always finds it even before tie_weights().
        object.__setattr__(self, '_tied_embed_ref', None)

    def tie_weights(self, embed: VocabParallelEmbedding) -> None:
        assert embed.num_embeddings == self.num_embeddings
        assert embed.embedding_dim == self.embedding_dim
        # Bypass NNX __setattr__ to store a plain Python reference.
        # NNX Pytree traversal only visits attributes set via its own
        # __setattr__, so this reference is invisible to JAX tracing.
        object.__setattr__(self, '_tied_embed_ref', embed)

    def load_weight(self, weight: jax.Array) -> None:
        start = self.tp_rank * self.num_embeddings_per_partition
        end = start + self.num_embeddings_per_partition
        arr = jnp.asarray(weight[start:end, :])
        self.weight = nnx.Param(arr)

    @property
    def effective_weight(self) -> jax.Array:
        """Return the weight matrix as a plain jax.Array.

        Reads the tied VocabParallelEmbedding's weight when tie_word_embeddings
        is True, otherwise falls back to this module's own weight parameter.
        The tied embed reference is stored via object.__setattr__ to bypass
        NNX Pytree type enforcement.
        """
        embed = object.__getattribute__(self, '_tied_embed_ref')
        if embed is not None:
            return embed.weight_array
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
