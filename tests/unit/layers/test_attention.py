"""Unit tests for PagedKVCache and Attention.

Covers:
- write() does not corrupt cache when padding tokens are present
- read() only reads real sequence rows
- decode output shape matches padded batch size
- prefill output shape is correct
- padding tokens do not bleed into real cache slots
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from nanovllm_jax.layers.attention import Attention, PagedKVCache


NUM_LAYERS   = 2
NUM_KV_HEADS = 2
HEAD_DIM     = 8
NUM_BLOCKS   = 16
BLOCK_SIZE   = 4
NUM_HEADS    = 4  # GQA: 4Q / 2KV


@pytest.fixture
def kv_cache():
    return PagedKVCache(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        dtype=jnp.float32,
    )


@pytest.fixture
def attn():
    return Attention(
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_kv_heads=NUM_KV_HEADS,
        layer_idx=0,
    )


# ---------------------------------------------------------------------------
# PagedKVCache.write — padding isolation
# ---------------------------------------------------------------------------

class TestKVCacheWrite:
    def test_write_real_tokens_only(self, kv_cache):
        """Writing 2 real tokens must not touch blocks used by other sequences."""
        num_real = 2
        bi = jnp.array([0, 0], dtype=jnp.int32)
        bo = jnp.array([0, 1], dtype=jnp.int32)
        k  = jnp.ones((num_real, NUM_KV_HEADS, HEAD_DIM), dtype=jnp.float32)
        v  = jnp.ones((num_real, NUM_KV_HEADS, HEAD_DIM), dtype=jnp.float32) * 2.0

        kv_cache.write(0, bi, bo, k, v)
        cache = kv_cache.cache.get_value()

        # Block 0, offsets 0-1 should be written
        assert jnp.allclose(cache[0, 0, 0, 0], 1.0)
        assert jnp.allclose(cache[0, 0, 0, 1], 1.0)
        # Block 0, offsets 2-3 must still be zero
        assert jnp.allclose(cache[0, 0, 0, 2], 0.0)
        # Block 1 must still be zero (not touched)
        assert jnp.allclose(cache[0, 0, 1], 0.0)

    def test_padding_tokens_excluded(self, kv_cache):
        """Verify block 0 offset 0 is not overwritten by a subsequent write."""
        bi_real = jnp.array([0], dtype=jnp.int32)
        bo_real = jnp.array([0], dtype=jnp.int32)
        k_real  = jnp.ones((1, NUM_KV_HEADS, HEAD_DIM)) * 99.0
        v_real  = jnp.ones((1, NUM_KV_HEADS, HEAD_DIM)) * 99.0
        kv_cache.write(0, bi_real, bo_real, k_real, v_real)

        bi_pad = jnp.array([0], dtype=jnp.int32)
        bo_pad = jnp.array([1], dtype=jnp.int32)
        k_pad  = jnp.ones((1, NUM_KV_HEADS, HEAD_DIM)) * -1.0
        v_pad  = jnp.ones((1, NUM_KV_HEADS, HEAD_DIM)) * -1.0
        kv_cache.write(0, bi_pad, bo_pad, k_pad, v_pad)

        cache = kv_cache.cache.get_value()
        assert jnp.allclose(cache[0, 0, 0, 0], 99.0)
        assert jnp.allclose(cache[0, 0, 0, 1], -1.0)


# ---------------------------------------------------------------------------
# PagedKVCache.read — only real sequences
# ---------------------------------------------------------------------------

class TestKVCacheRead:
    def test_read_returns_correct_shape(self, kv_cache):
        num_seqs   = 2
        max_blocks = 2
        block_table = jnp.zeros((num_seqs, max_blocks), dtype=jnp.int32)
        seq_lens    = jnp.array([3, 2], dtype=jnp.int32)
        k_out, v_out = kv_cache.read(0, block_table, seq_lens)
        expected_ctx = max_blocks * BLOCK_SIZE
        assert k_out.shape == (num_seqs, expected_ctx, NUM_KV_HEADS, HEAD_DIM)
        assert v_out.shape == (num_seqs, expected_ctx, NUM_KV_HEADS, HEAD_DIM)

    def test_read_isolates_sequences(self, kv_cache):
        """Two sequences using different blocks must read independent KV."""
        bi0 = jnp.array([0], dtype=jnp.int32)
        bi1 = jnp.array([1], dtype=jnp.int32)
        bo  = jnp.array([0], dtype=jnp.int32)
        kv_cache.write(0, bi0, bo,
                       jnp.ones((1, NUM_KV_HEADS, HEAD_DIM)) * 1.0,
                       jnp.ones((1, NUM_KV_HEADS, HEAD_DIM)) * 1.0)
        kv_cache.write(0, bi1, bo,
                       jnp.ones((1, NUM_KV_HEADS, HEAD_DIM)) * 2.0,
                       jnp.ones((1, NUM_KV_HEADS, HEAD_DIM)) * 2.0)

        block_table = jnp.array([[0, 0], [1, 1]], dtype=jnp.int32)
        seq_lens    = jnp.array([1, 1], dtype=jnp.int32)
        k_out, _ = kv_cache.read(0, block_table, seq_lens)

        assert jnp.allclose(k_out[0, 0], 1.0), "seq 0 should read block 0"
        assert jnp.allclose(k_out[1, 0], 2.0), "seq 1 should read block 1"


# ---------------------------------------------------------------------------
# Attention.__call__ — prefill
# ---------------------------------------------------------------------------

class TestAttentionPrefill:
    def test_prefill_output_shape(self, attn, kv_cache):
        T_real = 5
        T_pad  = 8
        q  = jnp.ones((T_pad, NUM_HEADS,    HEAD_DIM))
        k  = jnp.ones((T_pad, NUM_KV_HEADS, HEAD_DIM))
        v  = jnp.ones((T_pad, NUM_KV_HEADS, HEAD_DIM))
        bi = jnp.zeros(T_pad, dtype=jnp.int32)
        bo = jnp.arange(T_pad, dtype=jnp.int32)
        seq_lens = jnp.array([T_real], dtype=jnp.int32)

        out = attn(q, k, v, kv_cache, bi, bo, seq_lens,
                   is_prefill=True, num_real_tokens=T_real)
        assert out.shape == (T_real, NUM_HEADS, HEAD_DIM)

    def test_prefill_does_not_write_padding(self, kv_cache):
        """Padding tokens must not pollute the KV cache."""
        attn_layer = Attention(NUM_HEADS, HEAD_DIM, NUM_KV_HEADS, layer_idx=0)
        T_real, T_pad = 2, 4
        q  = jnp.ones((T_pad, NUM_HEADS,    HEAD_DIM))
        k  = jnp.ones((T_pad, NUM_KV_HEADS, HEAD_DIM))
        v  = jnp.ones((T_pad, NUM_KV_HEADS, HEAD_DIM))
        bi = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        bo = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        seq_lens = jnp.array([T_real], dtype=jnp.int32)

        attn_layer(q, k, v, kv_cache, bi, bo, seq_lens,
                   is_prefill=True, num_real_tokens=T_real)

        cache = kv_cache.cache.get_value()
        assert not jnp.allclose(cache[0, 0, 0, 0], 0.0)
        assert not jnp.allclose(cache[0, 0, 0, 1], 0.0)
        assert jnp.allclose(cache[0, 0, 0, 2], 0.0), "padding wrote to cache!"
        assert jnp.allclose(cache[0, 0, 0, 3], 0.0), "padding wrote to cache!"


# ---------------------------------------------------------------------------
# Attention.__call__ — decode
# ---------------------------------------------------------------------------

class TestAttentionDecode:
    def test_decode_output_shape_matches_padded_batch(self, attn, kv_cache):
        """Decode output must be padded back to B_pad for downstream ops."""
        B_real, B_pad = 2, 4
        q  = jnp.ones((B_pad, NUM_HEADS,    HEAD_DIM))
        k  = jnp.ones((B_pad, NUM_KV_HEADS, HEAD_DIM))
        v  = jnp.ones((B_pad, NUM_KV_HEADS, HEAD_DIM))
        bi = jnp.zeros(B_pad, dtype=jnp.int32)
        bo = jnp.zeros(B_pad, dtype=jnp.int32)
        seq_lens    = jnp.ones(B_pad, dtype=jnp.int32)
        block_table = jnp.zeros((B_pad, 2), dtype=jnp.int32)

        out = attn(q, k, v, kv_cache, bi, bo, seq_lens,
                   is_prefill=False,
                   block_table=block_table,
                   num_real_seqs=B_real)
        assert out.shape == (B_pad, NUM_HEADS, HEAD_DIM), \
            f"expected ({B_pad}, {NUM_HEADS}, {HEAD_DIM}), got {out.shape}"

    def test_decode_padded_rows_are_zero(self, attn, kv_cache):
        """Rows beyond num_real_seqs must be zero-padded in the output."""
        B_real, B_pad = 1, 4
        q  = jnp.ones((B_pad, NUM_HEADS,    HEAD_DIM))
        k  = jnp.ones((B_pad, NUM_KV_HEADS, HEAD_DIM))
        v  = jnp.ones((B_pad, NUM_KV_HEADS, HEAD_DIM))
        bi = jnp.zeros(B_pad, dtype=jnp.int32)
        bo = jnp.zeros(B_pad, dtype=jnp.int32)
        seq_lens    = jnp.ones(B_pad, dtype=jnp.int32)
        block_table = jnp.zeros((B_pad, 1), dtype=jnp.int32)

        out = attn(q, k, v, kv_cache, bi, bo, seq_lens,
                   is_prefill=False,
                   block_table=block_table,
                   num_real_seqs=B_real)
        assert jnp.allclose(out[1:], 0.0), "padded rows should be zero"

    def test_decode_sequences_independent(self):
        """Two decode sequences with different block tables must get different outputs."""
        cache = PagedKVCache(
            num_layers=1, num_kv_heads=1, head_dim=4,
            num_blocks=4, block_size=2, dtype=jnp.float32,
        )
        attn_layer = Attention(num_heads=1, head_dim=4, num_kv_heads=1, layer_idx=0)

        cache.write(0,
                    jnp.array([0], dtype=jnp.int32),
                    jnp.array([0], dtype=jnp.int32),
                    jnp.ones((1, 1, 4)) * 1.0,
                    jnp.ones((1, 1, 4)) * 1.0)
        cache.write(0,
                    jnp.array([1], dtype=jnp.int32),
                    jnp.array([0], dtype=jnp.int32),
                    jnp.ones((1, 1, 4)) * 2.0,
                    jnp.ones((1, 1, 4)) * 2.0)

        B_real = 2
        q  = jnp.ones((B_real, 1, 4))
        k  = jnp.zeros((B_real, 1, 4))
        v  = jnp.zeros((B_real, 1, 4))
        bi = jnp.array([0, 1], dtype=jnp.int32)
        bo = jnp.array([1, 1], dtype=jnp.int32)
        seq_lens    = jnp.array([2, 2], dtype=jnp.int32)
        block_table = jnp.array([[0, 0], [1, 1]], dtype=jnp.int32)

        out = attn_layer(q, k, v, cache, bi, bo, seq_lens,
                         is_prefill=False,
                         block_table=block_table,
                         num_real_seqs=B_real)
        assert not jnp.allclose(out[0], out[1]), \
            "sequences should have independent outputs"
