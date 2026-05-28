"""Token embedding and LM head (JAX port of nanovllm/layers/embed_head.py).

Status: ✅ Fixed (weight tying: nnx.data slot + get_raw_value() unwrap)

Weight tying rules for Flax NNX
--------------------------------
1. The slot must be initialised as ``nnx.data(None)`` in ``__init__`` so
   Flax marks it as a *data* attribute (not static).  Assigning a bare
   Module to a slot initialised as ``None`` (static) raises ValueError.
2. ``tie_weights`` wraps the embed module with ``nnx.data(embed)`` to keep
   the slot type consistent with the initial ``nnx.data(None)``.
3. ``effective_weight`` unwraps via ``get_raw_value()`` — the canonical
   non-deprecated accessor (as flagged by the DeprecationWarning for
   ``.raw_value``).  The old ``.value`` accessor silently returned ``None``
   in current Flax, causing lm_head to fall back to zero-initialised
   weights and produce garbage predictions for Qwen3.
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
        # Must be nnx.data(None) — not bare None — so Flax marks this slot as
        # a *data* attribute.  Assigning nnx.data(embed) later is only allowed
        # when the initial value is also wrapped in nnx.data().
        self._tied_embed = nnx.data(None)

    def tie_weights(self, embed: VocabParallelEmbedding) -> None:
        assert embed.num_embeddings == self.num_embeddings
        assert embed.embedding_dim == self.embedding_dim
        # Keep slot type consistent: must wrap with nnx.data() to match the
        # nnx.data(None) established in __init__.
        self._tied_embed = nnx.data(embed)

    def load_weight(self, weight: jax.Array) -> None:
        start = self.tp_rank * self.num_embeddings_per_partition
        end = start + self.num_embeddings_per_partition
        arr = jnp.asarray(weight[start:end, :])
        self.weight = nnx.Param(arr)

    @property
    def effective_weight(self) -> jax.Array:
        """Return the weight matrix as a plain jax.Array.

        Uses the tied embedding's weight when tie_word_embeddings=True.
        Unwraps nnx.data via get_raw_value() — the canonical non-deprecated
        API (replaces the broken .value accessor that silently returned None).
        """
        tied_wrapper = self._tied_embed
        # get_raw_value() is the non-deprecated replacement for .raw_value
        # and .value, as shown in the DeprecationWarning emitted at runtime.
        if hasattr(tied_wrapper, 'get_raw_value'):
            embed = tied_wrapper.get_raw_value()
        else:
            # Fallback for older Flax versions that use .raw_value
            embed = getattr(tied_wrapper, 'raw_value', None)
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
