"""Token embedding and LM head (JAX port of nanovllm/layers/embed_head.py).

Status: ✅ Done (Phase 2.2)

Design notes vs PyTorch original:
- VocabParallelEmbedding: tp sharding via vocab range masking.
- ParallelLMHead: supports weight tying via tie_weights().
  _tied_embed is stored as nnx.data() to satisfy Flax NNX pytree rules.
- All ops are jit-compilable.
"""
from __future__ import annotations
from typing import Optional
import jax
import jax.numpy as jnp
from flax import nnx


class VocabParallelEmbedding(nnx.Module):
    """Token embedding table, optionally sharded across TP ranks.

    Args:
        num_embeddings: full vocabulary size
        embedding_dim:  hidden dimension
        tp_size:        tensor-parallel world size (1 = single device)
        tp_rank:        tensor-parallel rank
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tp_size: int = 1,
        tp_rank: int = 0,
    ) -> None:
        assert num_embeddings % tp_size == 0, (
            f"num_embeddings {num_embeddings} must be divisible by tp_size {tp_size}"
        )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        self.num_embeddings_per_partition = num_embeddings // tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        self.weight = nnx.Param(
            jnp.zeros((self.num_embeddings_per_partition, embedding_dim))
        )

    def load_weight(self, weight: jax.Array) -> None:
        """Load embedding weight shard for this tp_rank.

        Args:
            weight: full embedding table (num_embeddings, embedding_dim)
        """
        start = self.vocab_start_idx
        end = self.vocab_end_idx
        self.weight = nnx.Param(jnp.array(weight[start:end, :]))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Token id lookup.

        Args:
            x: token ids (...,) int32
        Returns:
            embeddings (..., embedding_dim)
        """
        if self.tp_size == 1:
            return self.weight.value[x]
        # TP > 1: mask tokens outside this rank’s vocab range
        mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
        local_x = jnp.where(mask, x - self.vocab_start_idx, 0)
        y = self.weight.value[local_x]
        y = jnp.where(mask[..., None], y, jnp.zeros_like(y))
        # all-reduce across TP ranks (Phase 7 — single device: no-op)
        return y


class ParallelLMHead(nnx.Module):
    """Language model head: projects hidden states to vocab logits.

    Supports weight tying with VocabParallelEmbedding via tie_weights().
    _tied_embed is wrapped with nnx.data() so Flax NNX does not treat it
    as a static pytree attribute.

    Args:
        num_embeddings: full vocabulary size
        embedding_dim:  hidden/model dimension
        tp_size:        tensor-parallel world size
        tp_rank:        tensor-parallel rank
    """

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

        self.weight = nnx.Param(
            jnp.zeros((self.num_embeddings_per_partition, embedding_dim))
        )
        # Use nnx.data() so NNX treats this as a dynamic value, not a
        # static pytree field (avoids "Cannot assign Module to static attr" error)
        self._tied_embed: nnx.data = nnx.data(None)

    def tie_weights(self, embed: VocabParallelEmbedding) -> None:
        """Share weight with a VocabParallelEmbedding module.

        After calling this, __call__ uses embed.weight instead of self.weight.
        """
        assert embed.num_embeddings == self.num_embeddings
        assert embed.embedding_dim == self.embedding_dim
        self._tied_embed = nnx.data(embed)

    def load_weight(self, weight: jax.Array) -> None:
        """Load LM head weight shard for this tp_rank.

        Args:
            weight: full LM head weight (num_embeddings, embedding_dim)
        """
        start = self.tp_rank * self.num_embeddings_per_partition
        end = start + self.num_embeddings_per_partition
        self.weight = nnx.Param(jnp.array(weight[start:end, :]))

    @property
    def effective_weight(self) -> jax.Array:
        """Return the weight to use — tied embed or own weight."""
        embed = self._tied_embed.value if hasattr(self._tied_embed, 'value') else self._tied_embed
        if embed is not None:
            return embed.weight.value
        return self.weight.value

    def __call__(
        self,
        x: jax.Array,
        last_indices: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute vocab logits.

        Args:
            x:            hidden states (seq_len, hidden_dim)
            last_indices: optional 1-D int array; selects x[last_indices]
                          before projection (prefill last-token logits).
        Returns:
            logits (..., vocab_size_per_partition)
        """
        if last_indices is not None:
            x = x[last_indices]
        return x @ self.effective_weight.T
