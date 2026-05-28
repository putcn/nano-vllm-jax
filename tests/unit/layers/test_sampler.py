"""Unit tests for Sampler (Phase 3)."""
import pytest
import numpy as np
import jax
import jax.numpy as jnp

from nanovllm_jax.layers.sampler import greedy_sample, temperature_sample, Sampler


def rand_logits(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# greedy_sample
# ---------------------------------------------------------------------------

def test_greedy_returns_argmax():
    logits = jnp.array([[0.1, 0.9, 0.5], [0.8, 0.1, 0.2]])
    out = greedy_sample(logits)
    np.testing.assert_array_equal(np.array(out), [1, 0])


def test_greedy_dtype():
    logits = jnp.array(rand_logits((4, 128)))
    assert greedy_sample(logits).dtype == jnp.int32


def test_greedy_batch():
    logits = jnp.array(rand_logits((8, 256)))
    out = greedy_sample(logits)
    assert out.shape == (8,)


# ---------------------------------------------------------------------------
# temperature_sample
# ---------------------------------------------------------------------------

def test_temperature_zero_is_greedy():
    logits = jnp.array(rand_logits((4, 64)))
    key = jax.random.PRNGKey(0)
    sampled = temperature_sample(logits, key, temperature=0.0)
    greedy = greedy_sample(logits)
    np.testing.assert_array_equal(np.array(sampled), np.array(greedy))


def test_temperature_output_in_vocab():
    vocab = 100
    logits = jnp.array(rand_logits((4, vocab)))
    key = jax.random.PRNGKey(42)
    out = temperature_sample(logits, key, temperature=1.0)
    assert out.shape == (4,)
    assert jnp.all((out >= 0) & (out < vocab))


def test_top_k_restricts_vocab():
    """With top_k=1 the output must equal the argmax."""
    logits = jnp.array(rand_logits((4, 64)))
    key = jax.random.PRNGKey(7)
    out = temperature_sample(logits, key, temperature=1.0, top_k=1)
    greedy = greedy_sample(logits)
    np.testing.assert_array_equal(np.array(out), np.array(greedy))


def test_top_p_one_is_unconstrained():
    """top_p=1.0 should not crash and should return valid tokens."""
    vocab = 200
    logits = jnp.array(rand_logits((2, vocab)))
    key = jax.random.PRNGKey(3)
    out = temperature_sample(logits, key, temperature=0.8, top_p=1.0)
    assert jnp.all((out >= 0) & (out < vocab))


def test_top_p_near_zero_is_greedy():
    """Very small top_p should force selection of the top token."""
    logits = jnp.array(rand_logits((4, 64)))
    key = jax.random.PRNGKey(5)
    out = temperature_sample(logits, key, temperature=1.0, top_p=1e-6)
    greedy = greedy_sample(logits)
    np.testing.assert_array_equal(np.array(out), np.array(greedy))


def test_reproducibility():
    logits = jnp.array(rand_logits((4, 64)))
    key = jax.random.PRNGKey(99)
    out1 = temperature_sample(logits, key, temperature=1.0)
    out2 = temperature_sample(logits, key, temperature=1.0)
    np.testing.assert_array_equal(np.array(out1), np.array(out2))


# ---------------------------------------------------------------------------
# Sampler module
# ---------------------------------------------------------------------------

def test_sampler_greedy():
    sampler = Sampler()
    logits = jnp.array(rand_logits((3, 64)))
    out = sampler(logits, temperature=0.0)
    np.testing.assert_array_equal(np.array(out), np.array(greedy_sample(logits)))


def test_sampler_requires_key_for_temperature():
    sampler = Sampler()
    logits = jnp.array(rand_logits((2, 64)))
    with pytest.raises(AssertionError, match="PRNGKey"):
        sampler(logits, temperature=1.0)


def test_sampler_temperature_output():
    sampler = Sampler()
    logits = jnp.array(rand_logits((4, 128)))
    key = jax.random.PRNGKey(0)
    out = sampler(logits, key=key, temperature=0.8, top_k=50, top_p=0.9)
    assert out.shape == (4,)
    assert jnp.all((out >= 0) & (out < 128))
