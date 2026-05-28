"""Linear layers (JAX port of nanovllm/layers/linear.py).

Status: ✅ Done (Phase 2.1)
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
        self.weight = nnx.Param(jnp.zeros((output_size, input_size)))
        self.bias_param: Optional[nnx.Param] = nnx.Param(jnp.zeros(output_size)) if bias else None

    def load_weight(self, weight: jax.Array) -> None:
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
    def __init__(self, input_size: int, output_size: int, bias: bool = False) -> None:
        super().__init__(input_size, output_size, bias)

    def load_weight(self, weight: jax.Array) -> None:
        self.weight = nnx.Param(jnp.array(weight))

    def __call__(self, x: jax.Array) -> jax.Array:
        y = x @ self.weight[...].T
        if self.bias_param is not None:
            y = y + self.bias_param[...]
        return y


class ColumnParallelLinear(LinearBase):
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
        shard_size = self.output_size
        start = self.tp_rank * shard_size
        self.weight = nnx.Param(jnp.array(weight[start: start + shard_size, :]))

    def __call__(self, x: jax.Array) -> jax.Array:
        y = x @ self.weight[...].T
        if self.bias_param is not None:
            y = y + self.bias_param[...]
        return y


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        tp_size: int = 1,
        tp_rank: int = 0,
    ) -> None:
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias, tp_size, tp_rank)

    def load_weight(self, weight: jax.Array, shard_id: int) -> None:
        shard_offset = sum(self.output_sizes[:shard_id]) // self.tp_size
        shard_size = self.output_sizes[shard_id] // self.tp_size
        start = self.tp_rank * shard_size
        w_shard = weight[start: start + shard_size, :]
        new_weight = self.weight[...].at[shard_offset: shard_offset + shard_size, :].set(w_shard)
        self.weight = nnx.Param(new_weight)


class QKVParallelLinear(ColumnParallelLinear):
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
        shard_offset, shard_size = self._shard_info(shard_id)
        start = self.tp_rank * shard_size
        w_shard = weight[start: start + shard_size, :]
        new_weight = self.weight[...].at[shard_offset: shard_offset + shard_size, :].set(w_shard)
        self.weight = nnx.Param(new_weight)


class RowParallelLinear(LinearBase):
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
        shard_size = self.input_size
        start = self.tp_rank * shard_size
        self.weight = nnx.Param(jnp.array(weight[:, start: start + shard_size]))

    def __call__(self, x: jax.Array) -> jax.Array:
        y = x @ self.weight[...].T
        if self.bias_param is not None and self.tp_rank == 0:
            y = y + self.bias_param[...]
        return y
