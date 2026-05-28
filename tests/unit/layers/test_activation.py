"""Unit tests for activation functions (Phase 1.2)."""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from nanovllm_jax.layers.activation import silu_and_mul, SiluAndMul


def rand(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def torch_silu_and_mul(x_np):
    try:
        import torch
        import torch.nn.functional as F
        x = torch.tensor(x_np)
        half = x.shape[-1] // 2
        return (F.silu(x[..., :half]) * x[..., half:]).numpy()
    except ImportError:
        half = x_np.shape[-1] // 2
        return (1 / (1 + np.exp(-x_np[..., :half]))) * x_np[..., :half] * x_np[..., half:]


@pytest.mark.parametrize("shape", [(4, 8), (2, 3, 16), (1, 32), (8, 64)])
def test_output_shape(shape):
    x = jnp.ones(shape)
    out = silu_and_mul(x)
    assert out.shape == shape[:-1] + (shape[-1] // 2,)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_numerical_match_float32(seed):
    x_np = rand((8, 32), seed)
    out = np.array(silu_and_mul(jnp.array(x_np)))
    ref = torch_silu_and_mul(x_np)
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_zero_input():
    x = jnp.zeros((4, 8))
    out = silu_and_mul(x)
    np.testing.assert_allclose(np.array(out), np.zeros((4, 4)), atol=1e-6)


def test_large_input_no_nan():
    x = jnp.array(rand((4, 16), 0) * 100)
    out = silu_and_mul(x)
    assert not jnp.any(jnp.isnan(out))


def test_bfloat16():
    x = jnp.array(rand((4, 8), 0), dtype=jnp.bfloat16)
    out = silu_and_mul(x)
    assert out.dtype == jnp.bfloat16
    np.testing.assert_allclose(np.array(out, dtype=np.float32),
                               torch_silu_and_mul(rand((4, 8), 0)), atol=1e-2)


def test_nnx_wrapper():
    layer = SiluAndMul()
    x = jnp.array(rand((4, 16)))
    np.testing.assert_allclose(np.array(layer(x)), np.array(silu_and_mul(x)), atol=1e-7)


def test_jit():
    jitted = jax.jit(silu_and_mul)
    x = jnp.array(rand((4, 16)))
    np.testing.assert_allclose(np.array(jitted(x)), np.array(silu_and_mul(x)), atol=1e-7)
