"""Linear layers (JAX port of nanovllm/layers/linear.py).

Status: ✅ Done (Phase 2.1)

Design notes vs PyTorch original:
- PyTorch uses torch.distributed for tensor parallelism (TP).
  JAX port: single-device by default; TP sharding stubs use jax.sharding
  annotations but do NOT require an actual multi-process group.
- Weight loaders accept plain jnp arrays instead of nn.Parameter.
- All forward passes are pure functions, jit-compilable.
- tp_size=1 by default; multi-device TP is Phase 7.
"""
from __future__ import annotations
from typing import Optional
import jax
import jax.numpy as jnp
from flax import nnx


def _divide(numerator: int, denominator: int) -> int:
    assert numerator % denominator == 0, f"{numerator} not divisible by {denominator}"
    return numerator // denominator


class LinearBase(nnx.Module):
    """Base class for all linear layers.

    Args:
        input_size:  number of input features
        output_size: number of output features (local shard size)
        bias:        whether to include a bias term
        tp_size:     tensor-parallel world size (1 = single device)
        tp_rank:     tensor-parallel rank (0 = single device)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_size: int = 1,
        tp_rank: int = 0,
    ) -> None:
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.input_size = input_size
        self.output_size = output_size
        # weight shape: (output_size, input_size) — matches PyTorch convention
        self.weight = nnx.Param(jnp.zeros((output_size, input_size)))
        self.bias_param: Optional[nnx.Param] = nnx.Param(jnp.zeros(output_size)) if bias else None

    def load_weight(self, weight: jax.Array) -> None:
        """Set weight from a pre-loaded array (e.g. from HuggingFace checkpoint)."""
        assert weight.shape == (self.output_size, self.input_size), (
            f"Expected shape {(self.output_size, self.input_size)}, got {weight.shape}"
        )
        self.weight = nnx.Param(jnp.array(weight))

    def load_bias(self, bias: jax.Array) -> None:
        assert self.bias_param is not None, "This layer has no bias"
        self.bias_param = nnx.Param(jnp.array(bias))

    def __call__(self, x: jax.Array) -> jax.Array:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """Standard (non-sharded) linear layer.

    Weight is replicated across all TP ranks.
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False) -> None:
        super().__init__(input_size, output_size, bias)

    def load_weight(self, weight: jax.Array) -> None:
        """Load full weight directly."""
        self.weight = nnx.Param(jnp.array(weight))

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (..., input_size)  weight: (output_size, input_size)
        y = x @ self.weight.value.T
        if self.bias_param is not None:
            y = y + self.bias_param.value
        return y


class ColumnParallelLinear(LinearBase):
    """Column-parallel linear: output dimension is sharded across TP ranks.

    Full output_size is passed; each rank holds output_size // tp_size rows.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_size: int = 1,
        tp_rank: int = 0,
    ) -> None:
        local_output = _divide(output_size, tp_size)
        super().__init__(input_size, local_output, bias, tp_size, tp_rank)
        self.full_output_size = output_size

    def load_weight(self, weight: jax.Array) -> None:
        """Load the shard for this tp_rank from the full weight matrix.

        Args:
            weight: full weight (full_output_size, input_size)
        """
        shard_size = self.output_size
        start = self.tp_rank * shard_size
        shard = weight[start: start + shard_size, :]
        self.weight = nnx.Param(jnp.array(shard))

    def __call__(self, x: jax.Array) -> jax.Array:
        y = x @ self.weight.value.T
        if self.bias_param is not None:
            y = y + self.bias_param.value
        return y


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Merged column-parallel for fused projections (e.g. gate+up in MLP).

    Multiple output weight matrices are concatenated along the output dimension.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        tp_size: int = 1,
        tp_rank: int = 0,
    ) -> None:
        self.output_sizes = output_sizes
        total_output = sum(output_sizes)
        super().__init__(input_size, total_output, bias, tp_size, tp_rank)

    def load_weight(self, weight: jax.Array, shard_id: int) -> None:
        """Load one sub-weight shard into the merged weight.

        Args:
            weight:   full weight for this sub-projection (output_sizes[shard_id], input_size)
            shard_id: index into self.output_sizes
        """
        shard_offset = sum(self.output_sizes[:shard_id]) // self.tp_size
        shard_size = self.output_sizes[shard_id] // self.tp_size
        start = self.tp_rank * shard_size
        w_shard = weight[start: start + shard_size, :]

        # Patch into the merged weight
        current = self.weight.value
        new_weight = current.at[shard_offset: shard_offset + shard_size, :].set(w_shard)
        self.weight = nnx.Param(new_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """QKV projection with separate Q, K, V weight loaders.

    Packed layout: [Q | K | V] along the output dimension.
    Supports GQA (grouped-query attention) where num_kv_heads < num_heads.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = False,
        tp_size: int = 1,
        tp_rank: int = 0,
    ) -> None:
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = _divide(total_num_heads, tp_size)
        self.num_kv_heads = _divide(total_num_kv_heads, tp_size)
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, bias, tp_size, tp_rank)

    def _shard_info(self, shard_id: str) -> tuple[int, int]:
        """Return (shard_offset, shard_size) in the local packed weight."""
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        if shard_id == "q":
            return 0, q_size
        elif shard_id == "k":
            return q_size, kv_size
        elif shard_id == "v":
            return q_size + kv_size, kv_size
        else:
            raise ValueError(f"shard_id must be 'q', 'k', or 'v', got '{shard_id}'")

    def load_weight(self, weight: jax.Array, shard_id: str) -> None:  # type: ignore[override]
        """Load Q, K, or V weight shard.

        Args:
            weight:   full weight for this projection
            shard_id: 'q', 'k', or 'v'
        """
        shard_offset, shard_size = self._shard_info(shard_id)
        # Take this rank's slice from the full weight
        full_shard_size = shard_size * self.tp_size
        if shard_id == "q":
            full_shard_size = self.total_num_heads * self.head_size
        else:
            full_shard_size = self.total_num_kv_heads * self.head_size
        start = self.tp_rank * shard_size
        w_shard = weight[start: start + shard_size, :]

        current = self.weight.value
        new_weight = current.at[shard_offset: shard_offset + shard_size, :].set(w_shard)
        self.weight = nnx.Param(new_weight)


class RowParallelLinear(LinearBase):
    """Row-parallel linear: input dimension is sharded across TP ranks.

    Each rank holds input_size // tp_size columns of the weight matrix.
    An all-reduce is needed after the local matmul (single-device: no-op).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_size: int = 1,
        tp_rank: int = 0,
    ) -> None:
        local_input = _divide(input_size, tp_size)
        super().__init__(local_input, output_size, bias, tp_size, tp_rank)
        self.full_input_size = input_size

    def load_weight(self, weight: jax.Array) -> None:
        """Load the shard for this tp_rank from the full weight matrix.

        Args:
            weight: full weight (output_size, full_input_size)
        """
        shard_size = self.input_size
        start = self.tp_rank * shard_size
        shard = weight[:, start: start + shard_size]
        self.weight = nnx.Param(jnp.array(shard))

    def __call__(self, x: jax.Array) -> jax.Array:
        y = x @ self.weight.value.T
        # bias only added on rank 0 (single-device: always rank 0)
        if self.bias_param is not None and self.tp_rank == 0:
            y = y + self.bias_param.value
        # Multi-device: all-reduce here (Phase 7). Single-device: no-op.
        return y
