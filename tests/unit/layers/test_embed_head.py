"""Unit tests for VocabParallelEmbedding and ParallelLMHead (Phase 2.2).

Numerical equivalence verified against numpy/PyTorch reference.
Tolerances: atol=1e-5 float32.
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from nanovllm_jax.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rand(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# VocabParallelEmbedding
# ---------------------------------------------------------------------------

def test_embedding_shape():
    vocab, dim = 256, 64
    emb = VocabParallelEmbedding(vocab, dim)
    emb.load_weight(jnp.array(rand((vocab, dim))))
    ids = jnp.array([0, 1, 2, 127, 255])
    out = emb(ids)
    assert out.shape == (5, dim)


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_embedding_numerical(seed):
    vocab, dim = 128, 32
    w_np = rand((vocab, dim), seed)
    emb = VocabParallelEmbedding(vocab, dim)
    emb.load_weight(jnp.array(w_np))

    ids_np = np.array([0, 5, 10, 63, 127])
    out = np.array(emb(jnp.array(ids_np)))
    ref = w_np[ids_np]
    np.testing.assert_allclose(out, ref, atol=1e-6)


def test_embedding_batch():
    """2-D token id input (batch, seq_len)."""
    vocab, dim = 256, 64
    emb = VocabParallelEmbedding(vocab, dim)
    emb.load_weight(jnp.array(rand((vocab, dim))))
    ids = jnp.array([[0, 1, 2], [3, 4, 5]])
    out = emb(ids)
    assert out.shape == (2, 3, dim)


def test_embedding_tp_sharding():
    """TP sharding: each rank holds vocab//tp_size rows."""
    vocab, dim, tp = 128, 32, 2
    w_full = rand((vocab, dim))
    embs = [VocabParallelEmbedding(vocab, dim, tp_size=tp, tp_rank=r) for r in range(tp)]
    for emb in embs:
        emb.load_weight(jnp.array(w_full))
    # rank 0: tokens 0..63, rank 1: tokens 64..127
    np.testing.assert_allclose(np.array(embs[0].weight.value), w_full[:64], atol=1e-6)
    np.testing.assert_allclose(np.array(embs[1].weight.value), w_full[64:], atol=1e-6)


def test_embedding_tp_forward_single_rank():
    """TP forward: for tokens in-range returns correct embed, out-of-range returns 0."""
    vocab, dim, tp = 128, 32, 2
    w_full = rand((vocab, dim))
    emb0 = VocabParallelEmbedding(vocab, dim, tp_size=tp, tp_rank=0)  # handles 0..63
    emb0.load_weight(jnp.array(w_full))

    # token 5 is in range, token 70 is out of range
    ids = jnp.array([5, 70])
    out = np.array(emb0(ids))
    np.testing.assert_allclose(out[0], w_full[5], atol=1e-6)
    np.testing.assert_allclose(out[1], np.zeros(dim), atol=1e-6)


def test_embedding_jit():
    vocab, dim = 64, 16
    emb = VocabParallelEmbedding(vocab, dim)
    emb.load_weight(jnp.array(rand((vocab, dim))))
    jitted = nnx.jit(emb)
    out = jitted(jnp.array([0, 1, 2]))
    assert out.shape == (3, dim)


# ---------------------------------------------------------------------------
# ParallelLMHead
# ---------------------------------------------------------------------------

def test_lm_head_shape():
    vocab, dim = 256, 64
    head = ParallelLMHead(vocab, dim)
    head.load_weight(jnp.array(rand((vocab, dim))))
    x = jnp.ones((8, dim))
    logits = head(x)
    assert logits.shape == (8, vocab)


@pytest.mark.parametrize("seed", [0, 3])
def test_lm_head_numerical(seed):
    vocab, dim = 128, 32
    w_np = rand((vocab, dim), seed)
    x_np = rand((4, dim), seed + 1)

    head = ParallelLMHead(vocab, dim)
    head.load_weight(jnp.array(w_np))
    out = np.array(head(jnp.array(x_np)))
    ref = x_np @ w_np.T
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_lm_head_last_indices():
    """last_indices should select rows before projection."""
    vocab, dim = 64, 16
    w_np = rand((vocab, dim))
    x_np = rand((10, dim))
    last_idx = np.array([2, 5, 9])

    head = ParallelLMHead(vocab, dim)
    head.load_weight(jnp.array(w_np))
    out = np.array(head(jnp.array(x_np), last_indices=jnp.array(last_idx)))
    ref = x_np[last_idx] @ w_np.T
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)
    assert out.shape == (3, vocab)


def test_weight_tying():
    """After tie_weights, LM head must use embedding's weight."""
    vocab, dim = 128, 32
    w_np = rand((vocab, dim))

    emb = VocabParallelEmbedding(vocab, dim)
    emb.load_weight(jnp.array(w_np))

    head = ParallelLMHead(vocab, dim)
    head.tie_weights(emb)  # share weight

    x_np = rand((4, dim))
    out = np.array(head(jnp.array(x_np)))
    ref = x_np @ w_np.T
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)

    # Mutate embed weight — LM head output must change too
    new_w = rand((vocab, dim), seed=99)
    emb.load_weight(jnp.array(new_w))
    out2 = np.array(head(jnp.array(x_np)))
    ref2 = x_np @ new_w.T
    np.testing.assert_allclose(out2, ref2, atol=1e-4, rtol=1e-4)


def test_lm_head_jit():
    vocab, dim = 64, 16
    head = ParallelLMHead(vocab, dim)
    head.load_weight(jnp.array(rand((vocab, dim))))
    jitted = nnx.jit(head)
    assert jitted(jnp.ones((4, dim))).shape == (4, vocab)
