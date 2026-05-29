"""Token embedding 和 LM head（JAX port of nanovllm/layers/embed_head.py）.

Status: ✅ Fixed

Weight tying design
-------------------
lm_head 始终使用自身的 self.weight (nnx.Param)。
当 tie_word_embeddings=True 时，LlamaForCausalLM.load_weights() 在加载阶段
直接把 embed_tokens.weight_array 复制到 lm_head.weight。
这样 __call__ 里永远只有一个代码路径，不存在 JIT abstract tracer 问题。

_param_array 优先级
-------------------
根据当前 Flax 版本 deprecation 警告：
  - param.value 已被废弃 → 使用 param[...] (Variable.__getitem__)
  - get_value() 保留作为 fallback
"""
from __future__ import annotations
from typing import Optional
import jax
import jax.numpy as jnp
from flax import nnx


def _param_array(param: nnx.Param) -> jax.Array:
    """Extract a plain jax.Array from an nnx.Param.

    根据 deprecation 提示：Variable[Array] 应该用 param[...]。
    """
    # param[...] 是 Variable.__getitem__ 的正确调用方式
    try:
        return jnp.asarray(param[...])
    except Exception:
        pass
    # fallback: get_value()
    if hasattr(param, 'get_value'):
        return jnp.asarray(param.get_value())
    # last resort
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
        assert float(jnp.abs(arr).max()) > 0, (
            f"VocabParallelEmbedding.load_weight: 全零权重 shape={arr.shape}"
        )

    @property
    def weight_array(self) -> jax.Array:
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
        """weight shape: (full_vocab, hidden) — 自动按 tp_rank 切片。"""
        start = self.tp_rank * self.num_embeddings_per_partition
        end   = start + self.num_embeddings_per_partition
        arr = jnp.asarray(weight[start:end, :])
        self.weight = nnx.Param(arr)
        assert float(jnp.abs(arr).max()) > 0, (
            f"ParallelLMHead.load_weight: 全零权重 shape={arr.shape}"
        )

    @property
    def effective_weight(self) -> jax.Array:
        """Return this module's own weight as a plain jax.Array."""
        return _param_array(self.weight)

    def __call__(
        self,
        x: jax.Array,
        last_indices: Optional[jax.Array] = None,
    ) -> jax.Array:
        if last_indices is not None:
            x = x[last_indices]
        return x @ self.effective_weight.T
