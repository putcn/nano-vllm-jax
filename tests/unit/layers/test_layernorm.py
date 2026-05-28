"""Unit tests for RMSNorm (Phase 1.3)."""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from nanovllm_jax.layers.layernorm import RMSNorm


def rand(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def torch_rmsnorm(x_np, w_np, eps=1e-6):
    """Reference RMSNorm via PyTorch. All inputs kept in float32 to avoid
    dtype-mismatch warnings from PyTorch's RMSNorm kernel."""
    try:
        import torch
        x = torch.tensor(x_np, dtype=torch.float32)
        w = torch.tensor(w_np, dtype=torch.float32)
        norm = torch.nn.RMSNorm(x_np.shape[-1], eps=eps, elementwise_affine=True)
        norm.weight = torch.nn.Parameter(w)
        return norm(x).detach().numpy()
    except ImportError:
        rms = np.sqrt(np.mean(x_np ** 2, axis=-1, keepdims=True) + eps)
        return x_np / rms * w_np


@pytest.mark.parametrize("shape", [(4, 64), (2, 8, 128), (1, 256), (16, 32)])
def test_output_shape(shape):
    layer = RMSNorm(shape[-1])
    assert layer(jnp.ones(shape)).shape == shape


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_numerical_match_float32(seed):
    dim = 64
    x_np = rand((8, dim), seed)
    layer = RMSNorm(dim)
    out = np.array(layer(jnp.array(x_np)))
    ref = torch_rmsnorm(x_np, np.ones(dim, dtype=np.float32))
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_learned_weight():
    dim = 32
    x_np = rand((4, dim), 0)
    w_np = (rand((dim,), 1) + 1.0).astype(np.float32)
    layer = RMSNorm(dim)
    layer.weight = nnx.Param(jnp.array(w_np))
    out = np.array(layer(jnp.array(x_np)))
    ref = torch_rmsnorm(x_np, w_np)
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_fused_residual():
    dim = 64
    x_np = rand((4, dim), 0)
    r_np = rand((4, dim), 1)
    layer = RMSNorm(dim)
    normed, residual_out = layer(jnp.array(x_np), residual=jnp.array(r_np))
    np.testing.assert_allclose(np.array(residual_out), x_np + r_np, atol=1e-6)
    ref = torch_rmsnorm(x_np + r_np, np.ones(dim, dtype=np.float32))
    np.testing.assert_allclose(np.array(normed), ref, atol=1e-5)


def test_zero_input():
    layer = RMSNorm(32)
    x = jnp.zeros((4, 32))
    out = layer(x)
    assert not jnp.any(jnp.isnan(out))


def test_unit_variance_input():
    dim = 64
    layer = RMSNorm(dim)
    x = jnp.ones((4, dim))
    out = np.array(layer(x))
    np.testing.assert_allclose(out, np.ones((4, dim)), atol=1e-5)


def test_bfloat16():
    dim = 64
    layer = RMSNorm(dim)
    x = jnp.array(rand((4, dim)), dtype=jnp.bfloat16)
    out = layer(x)
    assert out.dtype == jnp.bfloat16


def test_jit():
    dim = 32
    layer = RMSNorm(dim)
    jitted = nnx.jit(layer)
    assert jitted(jnp.ones((4, dim))).shape == (4, dim)
