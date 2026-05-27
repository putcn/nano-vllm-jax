"""Unit tests for RMSNorm (Phase 1.3).

Numerical equivalence verified against PyTorch RMSNorm reference.
Tolerances: atol=1e-5 float32, atol=1e-2 bfloat16.
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from nanovllm_jax.layers.layernorm import RMSNorm


def make_rmsnorm(hidden_size: int, eps: float = 1e-6) -> RMSNorm:
    return RMSNorm(hidden_size, eps=eps, rngs=nnx.Rngs(0))


def torch_rmsnorm(x_np, weight_np, eps=1e-6):
    try:
        import torch, torch.nn as nn
        hidden = x_np.shape[-1]
        norm = nn.RMSNorm(hidden, eps=eps)
        norm.weight = nn.Parameter(torch.tensor(weight_np))
        with torch.no_grad():
            return norm(torch.tensor(x_np)).numpy()
    except ImportError:
        x_f32 = x_np.astype(np.float32)
        var = np.mean(x_f32 ** 2, axis=-1, keepdims=True)
        return (x_f32 / np.sqrt(var + eps) * weight_np).astype(x_np.dtype)


@pytest.mark.parametrize("shape", [(4, 64), (1, 128), (8, 256), (2, 5, 64)])
def test_output_shape(shape):
    norm = make_rmsnorm(shape[-1])
    assert norm(jnp.ones(shape)).shape == shape


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_numerical_match_float32(seed):
    hidden = 64
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal((8, hidden)).astype(np.float32)
    weight_np = rng.standard_normal(hidden).astype(np.float32) * 0.5 + 1.0
    norm = make_rmsnorm(hidden)
    norm.weight = nnx.Param(jnp.array(weight_np))
    out = np.array(norm(jnp.array(x_np)))
    ref = torch_rmsnorm(x_np, weight_np)
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_numerical_match_bfloat16():
    hidden = 64
    rng = np.random.default_rng(99)
    x_np = rng.standard_normal((4, hidden)).astype(np.float32)
    weight_np = np.ones(hidden, dtype=np.float32)
    norm = make_rmsnorm(hidden)
    norm.weight = nnx.Param(jnp.array(weight_np))
    out = np.array(norm(jnp.array(x_np).astype(jnp.bfloat16)).astype(jnp.float32))
    ref = torch_rmsnorm(x_np, weight_np)
    np.testing.assert_allclose(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("seed", [0, 7])
def test_add_rms_forward_shape(seed):
    rng = np.random.default_rng(seed)
    hidden = 32
    x = jnp.array(rng.standard_normal((4, hidden)).astype(np.float32))
    res = jnp.array(rng.standard_normal((4, hidden)).astype(np.float32))
    norm = make_rmsnorm(hidden)
    normed, new_res = norm(x, residual=res)
    assert normed.shape == (4, hidden)
    assert new_res.shape == (4, hidden)


def test_add_rms_residual_value():
    hidden = 16
    x_np = np.ones((2, hidden), dtype=np.float32)
    res_np = np.full((2, hidden), 2.0, dtype=np.float32)
    norm = make_rmsnorm(hidden)
    _, new_res = norm(jnp.array(x_np), residual=jnp.array(res_np))
    np.testing.assert_allclose(np.array(new_res), x_np + res_np, atol=1e-6)


def test_unit_variance_input():
    hidden = 64
    norm = make_rmsnorm(hidden)
    x_np = np.random.default_rng(0).standard_normal((4, hidden)).astype(np.float32)
    x_np /= np.sqrt(np.mean(x_np**2, axis=-1, keepdims=True))
    out = np.array(norm(jnp.array(x_np)))
    np.testing.assert_allclose(out, x_np, atol=1e-5)


def test_zero_input_output_zero():
    norm = make_rmsnorm(32)
    out = norm(jnp.zeros((3, 32)))
    assert jnp.allclose(out, jnp.zeros_like(out), atol=1e-6)


def test_jit_compilable():
    norm = make_rmsnorm(64)
    jitted = nnx.jit(norm)
    assert jitted(jnp.ones((4, 64))).shape == (4, 64)
