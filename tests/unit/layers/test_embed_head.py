"""Unit tests for VocabParallelEmbedding and ParallelLMHead (Phase 2.2).

Tolerance note:
  Embedding lookup is exact (integer indexing, no accumulation).
  LM head is a matmul so uses ATOL=1e-2 for XLA vs PyTorch fp32 differences.
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from nanovllm_jax.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

ATOL_EXACT = 1e-6   # embedding lookup: no accumulation error
ATOL_MATMUL = 1e-2  # LM head matmul: XLA vs PyTorch accumulation


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
    assert emb(jnp.array([0, 1, 2, 127, 255])).shape == (5, dim)


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_embedding_numerical(seed):
    vocab, dim = 128, 32
    w_np = rand((vocab, dim), seed)
    emb = VocabParallelEmbedding(vocab, dim)
    emb.load_weight(jnp.array(w_np))

    ids_np = np.array([0, 5, 10, 63, 127])
    out = np.array(emb(jnp.array(ids_np)))
    np.testing.assert_allclose(out, w_np[ids_np], atol=ATOL_EXACT)


def test_embedding_batch():
    vocab, dim = 256, 64
    emb = VocabParallelEmbedding(vocab, dim)
    emb.load_weight(jnp.array(rand((vocab, dim))))
    out = emb(jnp.array([[0, 1, 2], [3, 4, 5]]))
    assert out.shape == (2, 3, dim)


def test_embedding_tp_sharding():
    vocab, dim, tp = 128, 32, 2
    w_full = rand((vocab, dim))
    embs = [VocabParallelEmbedding(vocab, dim, tp_size=tp, tp_rank=r) for r in range(tp)]
    for emb in embs:
        emb.load_weight(jnp.array(w_full))
    np.testing.assert_allclose(np.array(embs[0].weight.value), w_full[:64], atol=ATOL_EXACT)
    np.testing.assert_allclose(np.array(embs[1].weight.value), w_full[64:], atol=ATOL_EXACT)


def test_embedding_tp_forward_single_rank():
    vocab, dim, tp = 128, 32, 2
    w_full = rand((vocab, dim))
    emb0 = VocabParallelEmbedding(vocab, dim, tp_size=tp, tp_rank=0)
    emb0.load_weight(jnp.array(w_full))
    ids = jnp.array([5, 70])
    out = np.array(emb0(ids))
    np.testing.assert_allclose(out[0], w_full[5], atol=ATOL_EXACT)
    np.testing.assert_allclose(out[1], np.zeros(dim), atol=ATOL_EXACT)


def test_embedding_jit():
    vocab, dim = 64, 16
    emb = VocabParallelEmbedding(vocab, dim)
    emb.load_weight(jnp.array(rand((vocab, dim))))
    jitted = nnx.jit(emb)
    assert jitted(jnp.array([0, 1, 2])).shape == (3, dim)


# ---------------------------------------------------------------------------
# ParallelLMHead
# ---------------------------------------------------------------------------

def test_lm_head_shape():
    vocab, dim = 256, 64
    head = ParallelLMHead(vocab, dim)
    head.load_weight(jnp.array(rand((vocab, dim))))
    assert head(jnp.ones((8, dim))).shape == (8, vocab)


@pytest.mark.parametrize("seed", [0, 3])
def test_lm_head_numerical(seed):
    vocab, dim = 128, 32
    w_np = rand((vocab, dim), seed)
    x_np = rand((4, dim), seed + 1)

    head = ParallelLMHead(vocab, dim)
    head.load_weight(jnp.array(w_np))
    out = np.array(head(jnp.array(x_np)))
    ref = x_np @ w_np.T
    np.testing.assert_allclose(out, ref, atol=ATOL_MATMUL, rtol=ATOL_MATMUL)


def test_lm_head_last_indices():
    vocab, dim = 64, 16
    w_np = rand((vocab, dim))
    x_np = rand((10, dim))
    last_idx = np.array([2, 5, 9])

    head = ParallelLMHead(vocab, dim)
    head.load_weight(jnp.array(w_np))
    out = np.array(head(jnp.array(x_np), last_indices=jnp.array(last_idx)))
    ref = x_np[last_idx] @ w_np.T
    np.testing.assert_allclose(out, ref, atol=ATOL_MATMUL, rtol=ATOL_MATMUL)
    assert out.shape == (3, vocab)


def test_weight_tying():
    """After tie_weights, LM head must use embedding's weight."""
    vocab, dim = 128, 32
    w_np = rand((vocab, dim))

    emb = VocabParallelEmbedding(vocab, dim)
    emb.load_weight(jnp.array(w_np))

    head = ParallelLMHead(vocab, dim)
    head.tie_weights(emb)

    x_np = rand((4, dim))
    out = np.array(head(jnp.array(x_np)))
    np.testing.assert_allclose(out, x_np @ w_np.T, atol=ATOL_MATMUL)

    # Mutate embed weight — LM head output must change too
    new_w = rand((vocab, dim), seed=99)
    emb.load_weight(jnp.array(new_w))
    out2 = np.array(head(jnp.array(x_np)))
    np.testing.assert_allclose(out2, x_np @ new_w.T, atol=ATOL_MATMUL)


def test_lm_head_jit():
    vocab, dim = 64, 16
    head = ParallelLMHead(vocab, dim)
    head.load_weight(jnp.array(rand((vocab, dim))))
    jitted = nnx.jit(head)
    assert jitted(jnp.ones((4, dim))).shape == (4, vocab)
