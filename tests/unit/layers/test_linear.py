"""Unit tests for linear layers (Phase 2.1).

Numerical equivalence verified against PyTorch F.linear reference.

Tolerance note:
  JAX/XLA matmul uses different floating-point accumulation order vs PyTorch,
  causing systematic differences up to ~0.01 in float32. We use atol=1e-2
  which still catches wrong implementations while accepting XLA differences.
  For bfloat16 the tolerance would need to be wider (~0.1).
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from nanovllm_jax.layers.linear import (
    ReplicatedLinear,
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

# XLA vs PyTorch matmul accumulation difference in float32
ATOL = 1e-2
RTOL = 1e-2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rand(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def torch_linear(x_np, w_np, b_np=None):
    try:
        import torch
        import torch.nn.functional as F
        x = torch.tensor(x_np)
        w = torch.tensor(w_np)
        b = torch.tensor(b_np) if b_np is not None else None
        return F.linear(x, w, b).numpy()
    except ImportError:
        y = x_np @ w_np.T
        if b_np is not None:
            y = y + b_np
        return y


# ---------------------------------------------------------------------------
# ReplicatedLinear
# ---------------------------------------------------------------------------

def test_replicated_linear_shape():
    layer = ReplicatedLinear(32, 64)
    layer.load_weight(jnp.array(rand((64, 32))))
    assert layer(jnp.ones((4, 32))).shape == (4, 64)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_replicated_linear_numerical(seed):
    in_f, out_f = 32, 64
    x_np = rand((8, in_f), seed)
    w_np = rand((out_f, in_f), seed + 10)
    b_np = rand((out_f,), seed + 20)

    layer = ReplicatedLinear(in_f, out_f, bias=True)
    layer.load_weight(jnp.array(w_np))
    layer.load_bias(jnp.array(b_np))

    out = np.array(layer(jnp.array(x_np)))
    ref = torch_linear(x_np, w_np, b_np)
    np.testing.assert_allclose(out, ref, atol=ATOL, rtol=RTOL)


def test_replicated_linear_no_bias():
    layer = ReplicatedLinear(16, 32, bias=False)
    layer.load_weight(jnp.array(rand((32, 16))))
    assert layer(jnp.ones((2, 16))).shape == (2, 32)


def test_replicated_linear_jit():
    layer = ReplicatedLinear(16, 32)
    layer.load_weight(jnp.array(rand((32, 16))))
    jitted = nnx.jit(layer)
    assert jitted(jnp.ones((2, 16))).shape == (2, 32)


# ---------------------------------------------------------------------------
# ColumnParallelLinear
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1), (4, 2)])
def test_column_parallel_shape(tp_size, tp_rank):
    in_f, out_f = 32, 64
    layer = ColumnParallelLinear(in_f, out_f, tp_size=tp_size, tp_rank=tp_rank)
    layer.load_weight(jnp.array(rand((out_f, in_f))))
    assert layer(jnp.ones((4, in_f))).shape == (4, out_f // tp_size)


def test_column_parallel_weight_sharding():
    in_f, out_f = 16, 32
    w_full = rand((out_f, in_f))
    layers = [ColumnParallelLinear(in_f, out_f, tp_size=2, tp_rank=r) for r in range(2)]
    for layer in layers:
        layer.load_weight(jnp.array(w_full))
    np.testing.assert_allclose(np.array(layers[0].weight.value), w_full[:16, :], atol=1e-6)
    np.testing.assert_allclose(np.array(layers[1].weight.value), w_full[16:, :], atol=1e-6)


@pytest.mark.parametrize("seed", [0, 5])
def test_column_parallel_numerical(seed):
    in_f, out_f = 32, 64
    x_np = rand((8, in_f), seed)
    w_full = rand((out_f, in_f), seed + 1)

    layer = ColumnParallelLinear(in_f, out_f, tp_size=1, tp_rank=0)
    layer.load_weight(jnp.array(w_full))
    out = np.array(layer(jnp.array(x_np)))
    ref = torch_linear(x_np, w_full)
    np.testing.assert_allclose(out, ref, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# MergedColumnParallelLinear
# ---------------------------------------------------------------------------

def test_merged_column_shape():
    layer = MergedColumnParallelLinear(32, [64, 64])
    layer.load_weight(jnp.array(rand((64, 32), 0)), shard_id=0)
    layer.load_weight(jnp.array(rand((64, 32), 1)), shard_id=1)
    assert layer(jnp.ones((4, 32))).shape == (4, 128)


def test_merged_column_correctness():
    in_f = 32
    w0 = rand((64, in_f), 0)
    w1 = rand((64, in_f), 1)
    x_np = rand((4, in_f), 2)

    merged = MergedColumnParallelLinear(in_f, [64, 64])
    merged.load_weight(jnp.array(w0), shard_id=0)
    merged.load_weight(jnp.array(w1), shard_id=1)
    out = np.array(merged(jnp.array(x_np)))

    ref = np.concatenate([torch_linear(x_np, w0), torch_linear(x_np, w1)], axis=-1)
    np.testing.assert_allclose(out, ref, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# QKVParallelLinear
# ---------------------------------------------------------------------------

def test_qkv_parallel_shape():
    hidden, head_size, num_heads, num_kv_heads = 128, 32, 4, 2
    layer = QKVParallelLinear(hidden, head_size, num_heads, num_kv_heads)
    layer.load_weight(jnp.array(rand((num_heads * head_size, hidden))), shard_id="q")
    layer.load_weight(jnp.array(rand((num_kv_heads * head_size, hidden))), shard_id="k")
    layer.load_weight(jnp.array(rand((num_kv_heads * head_size, hidden))), shard_id="v")
    out = layer(jnp.ones((4, hidden)))
    assert out.shape == (4, (num_heads + 2 * num_kv_heads) * head_size)


def test_qkv_parallel_gqa_shape():
    hidden, head_size, num_heads, num_kv_heads = 256, 64, 8, 2
    layer = QKVParallelLinear(hidden, head_size, num_heads, num_kv_heads)
    layer.load_weight(jnp.array(rand((num_heads * head_size, hidden))), shard_id="q")
    layer.load_weight(jnp.array(rand((num_kv_heads * head_size, hidden))), shard_id="k")
    layer.load_weight(jnp.array(rand((num_kv_heads * head_size, hidden))), shard_id="v")
    out = layer(jnp.ones((4, hidden)))
    assert out.shape == (4, (num_heads + 2 * num_kv_heads) * head_size)


def test_qkv_invalid_shard_id():
    layer = QKVParallelLinear(64, 16, 2, 2)
    with pytest.raises(ValueError, match="shard_id"):
        layer.load_weight(jnp.ones((32, 64)), shard_id="x")


def test_qkv_jit():
    hidden, head_size, num_heads, num_kv_heads = 64, 16, 4, 2
    layer = QKVParallelLinear(hidden, head_size, num_heads, num_kv_heads)
    layer.load_weight(jnp.array(rand((num_heads * head_size, hidden))), shard_id="q")
    layer.load_weight(jnp.array(rand((num_kv_heads * head_size, hidden))), shard_id="k")
    layer.load_weight(jnp.array(rand((num_kv_heads * head_size, hidden))), shard_id="v")
    jitted = nnx.jit(layer)
    assert jitted(jnp.ones((2, hidden))).shape[0] == 2


# ---------------------------------------------------------------------------
# RowParallelLinear
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1)])
def test_row_parallel_shape(tp_size, tp_rank):
    in_f, out_f = 64, 32
    layer = RowParallelLinear(in_f, out_f, tp_size=tp_size, tp_rank=tp_rank)
    layer.load_weight(jnp.array(rand((out_f, in_f))))
    assert layer(jnp.ones((4, in_f // tp_size))).shape == (4, out_f)


@pytest.mark.parametrize("seed", [0, 3])
def test_row_parallel_numerical(seed):
    in_f, out_f = 64, 32
    x_np = rand((8, in_f), seed)
    w_full = rand((out_f, in_f), seed + 1)

    layer = RowParallelLinear(in_f, out_f, tp_size=1, tp_rank=0)
    layer.load_weight(jnp.array(w_full))
    out = np.array(layer(jnp.array(x_np)))
    ref = torch_linear(x_np, w_full)
    np.testing.assert_allclose(out, ref, atol=ATOL, rtol=RTOL)


def test_row_parallel_weight_sharding():
    in_f, out_f = 64, 32
    w_full = rand((out_f, in_f))
    for r in range(2):
        layer = RowParallelLinear(in_f, out_f, tp_size=2, tp_rank=r)
        layer.load_weight(jnp.array(w_full))
        expected = w_full[:, r * 32: (r + 1) * 32]
        np.testing.assert_allclose(np.array(layer.weight.value), expected, atol=1e-6)
