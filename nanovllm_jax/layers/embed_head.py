"""Token embedding and LM head (JAX port of nanovllm/layers/embed_head.py).

Status: ✅ Fixed (weight tying via explicit weight_override — no cross-module refs)

Weight tying design
-------------------
Previous approaches stored a reference to VocabParallelEmbedding inside
ParallelLMHead (via nnx.data or object.__setattr__).  Both caused nnx.jit
to detect the same Param appearing in two graph nodes at different trace
levels:

  ValueError: Cannot extract graph node from different trace level

The correct solution: ParallelLMHead holds NO reference to embed_tokens.
Instead, LlamaForCausalLM.__call__ passes embed_tokens.weight_array as an
explicit ``weight_override`` argument when tie_word_embeddings=True.  From
JAX/NNX’s perspective this is just a plain Array argument — no shared nodes,
no trace level conflict.

``effective_weight`` is kept as a public property so that unit tests
(test_prefill_last_indices) can access the module’s own weight directly
for reference computations.  It always returns this module’s own weight;
the tied-embed weight is delivered exclusively via ``weight_override``.
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

    def load_weight(self, weight: jax.Array) -> None:
        start = self.tp_rank * self.num_embeddings_per_partition
        end = start + self.num_embeddings_per_partition
        arr = jnp.asarray(weight[start:end, :])
        self.weight = nnx.Param(arr)

    @property
    def effective_weight(self) -> jax.Array:
        """Return this module's own weight as a plain jax.Array.

        For tied-weight models the actual weight used at inference time is
        delivered via the ``weight_override`` argument to ``__call__`` by
        LlamaForCausalLM.  This property exposes the module-local weight so
        that unit tests can use it as a numerical reference.
        """
        return _param_array(self.weight)

    def __call__(
        self,
        x: jax.Array,
        last_indices: Optional[jax.Array] = None,
        weight_override: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute logits.

        Args:
            x: hidden states, shape [T, hidden_size]
            last_indices: if provided, select x[last_indices] before matmul.
            weight_override: if provided (tied-weight models), use this array
                instead of self.weight.  LlamaForCausalLM passes
                embed_tokens.weight_array here so JIT sees a plain Array
                argument — no shared-node trace-level conflict.
        """
        if last_indices is not None:
            x = x[last_indices]
        w = weight_override if weight_override is not None else self.effective_weight
        return x @ w.T
