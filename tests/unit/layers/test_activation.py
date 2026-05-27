"""Unit tests for activation functions (Phase 1.2).

Numerical equivalence verified against PyTorch F.silu reference.
Tolerances: atol=1e-5, rtol=1e-5 (float32 ops).
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", False)

from nanovllm_jax.layers.activation import silu_and_mul, SiluAndMul


def torch_silu_and_mul(x_np: np.ndarray) -> np.ndarray:
    try:
        import torch
        import torch.nn.functional as F
        x = torch.tensor(x_np)
        half = x.shape[-1] // 2
        gate, val = x[..., :half], x[..., half:]
        return (F.silu(gate) * val).numpy()
    except ImportError:
        half = x_np.shape[-1] // 2
        gate, val = x_np[..., :half], x_np[..., half:]
        silu = gate * (1.0 / (1.0 + np.exp(-gate)))
        return silu * val


@pytest.mark.parametrize("shape", [
    (4, 16), (1, 8), (8, 64), (2, 3, 32),
])
def test_output_shape(shape):
    x = jnp.ones(shape)
    out = silu_and_mul(x)
    assert out.shape == shape[:-1] + (shape[-1] // 2,)


@pytest.mark.parametrize("seed", [0, 42, 123])
def test_numerical_match_float32(seed):
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal((8, 64)).astype(np.float32)
    ref = torch_silu_and_mul(x_np)
    out = np.array(silu_and_mul(jnp.array(x_np)))
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_numerical_match_bfloat16():
    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((4, 32)).astype(np.float32)
    ref = torch_silu_and_mul(x_np)
    x_bf16 = jnp.array(x_np).astype(jnp.bfloat16)
    out = np.array(silu_and_mul(x_bf16).astype(jnp.float32))
    np.testing.assert_allclose(out, ref, atol=1e-2, rtol=1e-2)


def test_zero_input():
    x = jnp.zeros((4, 16))
    out = silu_and_mul(x)
    np.testing.assert_array_equal(np.array(out), np.zeros((4, 8)))


def test_large_positive():
    x = jnp.full((2, 8), 100.0)
    out = silu_and_mul(x)
    assert float(out.min()) > 9000


def test_jit_compilable():
    jitted = jax.jit(silu_and_mul)
    out = jitted(jnp.ones((4, 16)))
    assert out.shape == (4, 8)


def test_module_wrapper():
    from flax import nnx
    m = SiluAndMul()
    out = m(jnp.ones((2, 8)))
    assert out.shape == (2, 4)


def test_module_jit():
    from flax import nnx
    m = SiluAndMul()
    jitted = nnx.jit(m)
    out = jitted(jnp.ones((2, 8)))
    assert out.shape == (2, 4)
