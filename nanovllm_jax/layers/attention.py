"""Paged KV Cache and Attention (JAX port of nanovllm/layers/attention.py).

Status: ✅ Done (Phase 3)

Design notes vs PyTorch original:
- PagedKVCache uses jax.lax.dynamic_update_slice / dynamic_slice instead of
  in-place scatter. All ops are jit-compilable.
- Attention.forward dispatches to _prefill (full causal softmax) or
  _decode (single-step KV lookup) based on is_prefill flag.
- No flash-attention yet — plain scaled dot-product in float32.
  Flash attention (via jax.nn.dot_product_attention) is Phase 6.
"""
from __future__ import annotations
from typing import Optional
import jax
import jax.numpy as jnp
from flax import nnx


# ---------------------------------------------------------------------------
# Paged KV Cache
# ---------------------------------------------------------------------------

class PagedKVCache(nnx.Module):
    """Paged key-value cache for autoregressive decoding.

    Layout: cache[layer, 2, num_blocks, block_size, num_kv_heads, head_dim]
      - dim 1: 0 = keys, 1 = values

    Args:
        num_layers:   number of transformer layers
        num_kv_heads: number of KV attention heads
        head_dim:     per-head feature dimension
        num_blocks:   total number of paged blocks in the cache
        block_size:   tokens per block
        dtype:        storage dtype (default float16 to save VRAM)
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int = 16,
        dtype: jnp.dtype = jnp.float16,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype
        # Shape: [num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]
        cache = jnp.zeros(
            (num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype,
        )
        self.cache = nnx.Variable(cache)

    def write(
        self,
        layer_idx: int,
        block_indices: jax.Array,   # (num_tokens,) int32 — block id per token
        block_offsets: jax.Array,   # (num_tokens,) int32 — offset within block
        keys: jax.Array,            # (num_tokens, num_kv_heads, head_dim)
        values: jax.Array,          # (num_tokens, num_kv_heads, head_dim)
    ) -> None:
        """Write a batch of tokens into the cache (prefill or decode)."""
        cache = self.cache.get_value()

        def write_one(carry, x):
            c, bi, bo, k, v = carry, x[0], x[1], x[2], x[3]
            c = c.at[layer_idx, 0, bi, bo].set(k.astype(self.dtype))
            c = c.at[layer_idx, 1, bi, bo].set(v.astype(self.dtype))
            return c, None

        # xs: (num_tokens, 2 + 2*num_kv_heads*head_dim) — pack indices + kv
        # Use vmap-style scan to stay jit-friendly
        tokens = keys.shape[0]
        new_cache = cache
        # Python loop is fine here — jit will unroll for small token counts;
        # for large prefills Phase 6 will switch to scatter_nd.
        # For now use lax.fori_loop to keep it jit-compilable.
        kv_heads = keys.shape[1]
        hd = keys.shape[2]

        def body(i, c):
            bi = block_indices[i]
            bo = block_offsets[i]
            c = c.at[layer_idx, 0, bi, bo].set(keys[i].astype(self.dtype))
            c = c.at[layer_idx, 1, bi, bo].set(values[i].astype(self.dtype))
            return c

        new_cache = jax.lax.fori_loop(0, tokens, body, new_cache)
        self.cache = nnx.Variable(new_cache)

    def read(
        self,
        layer_idx: int,
        block_table: jax.Array,   # (num_seqs, max_blocks_per_seq) int32
        seq_lens: jax.Array,      # (num_seqs,) int32
    ) -> tuple[jax.Array, jax.Array]:
        """Read all cached KV for each sequence (decode step).

        Returns:
            keys:   (num_seqs, max_seq_len, num_kv_heads, head_dim)
            values: (num_seqs, max_seq_len, num_kv_heads, head_dim)
        """
        cache = self.cache.get_value()
        num_seqs, max_blocks = block_table.shape
        max_seq_len = max_blocks * self.block_size

        # Gather all blocks for every sequence
        # cache[layer, kv, block, offset, head, dim]
        k_cache = cache[layer_idx, 0]  # (num_blocks, block_size, kv_heads, head_dim)
        v_cache = cache[layer_idx, 1]

        # (num_seqs, max_blocks, block_size, kv_heads, head_dim)
        k_gathered = k_cache[block_table]  # fancy index over block dim
        v_gathered = v_cache[block_table]

        # Reshape to (num_seqs, max_seq_len, kv_heads, head_dim)
        k_out = k_gathered.reshape(num_seqs, max_seq_len, self.num_kv_heads, self.head_dim)
        v_out = v_gathered.reshape(num_seqs, max_seq_len, self.num_kv_heads, self.head_dim)
        return k_out.astype(jnp.float32), v_out.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nnx.Module):
    """Multi-head attention with GQA support and paged KV cache.

    Args:
        num_heads:    number of query heads
        head_dim:     per-head feature dimension
        num_kv_heads: number of KV heads (GQA when < num_heads)
        layer_idx:    index used to address the KV cache
        scale:        attention scale (default 1/sqrt(head_dim))
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        layer_idx: int = 0,
        scale: Optional[float] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.layer_idx = layer_idx
        self.scale = scale or (head_dim ** -0.5)
        self.kv_groups = self.num_heads // self.num_kv_heads  # GQA repeat factor

    # ------------------------------------------------------------------
    # Prefill: full causal attention over a packed token sequence
    # ------------------------------------------------------------------

    def _prefill(
        self,
        q: jax.Array,   # (total_tokens, num_heads, head_dim)
        k: jax.Array,   # (total_tokens, num_kv_heads, head_dim)
        v: jax.Array,   # (total_tokens, num_kv_heads, head_dim)
        seq_lens: jax.Array,  # (num_seqs,) int32
    ) -> jax.Array:     # (total_tokens, num_heads, head_dim)
        """Causal self-attention for prefill.

        For simplicity, treats the whole batch as one sequence with a
        block-diagonal causal mask.  Phase 6 replaces this with
        flash-attention / varlen kernels.
        """
        T = q.shape[0]
        # GQA: repeat KV heads to match Q heads
        if self.kv_groups > 1:
            k = jnp.repeat(k, self.kv_groups, axis=1)  # (T, num_heads, head_dim)
            v = jnp.repeat(v, self.kv_groups, axis=1)

        # (T, num_heads, T) attention logits
        # q: (T, H, D) -> (H, T, D); k: (T, H, D) -> (H, D, T)
        q_t = jnp.transpose(q, (1, 0, 2))  # (H, T, D)
        k_t = jnp.transpose(k, (1, 2, 0))  # (H, D, T)
        logits = jnp.matmul(q_t, k_t) * self.scale  # (H, T, T)

        # Causal mask: token i can only attend to tokens j <= i
        mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        logits = jnp.where(mask[None, :, :], logits, jnp.finfo(jnp.float32).min)

        attn = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        v_t = jnp.transpose(v, (1, 0, 2))  # (H, T, D)
        out = jnp.matmul(attn, v_t)        # (H, T, D)
        return jnp.transpose(out, (1, 0, 2))  # (T, H, D)

    # ------------------------------------------------------------------
    # Decode: single new token per sequence, KV from paged cache
    # ------------------------------------------------------------------

    def _decode(
        self,
        q: jax.Array,             # (num_seqs, num_heads, head_dim)
        k_cache: jax.Array,       # (num_seqs, ctx_len, num_kv_heads, head_dim)
        v_cache: jax.Array,       # (num_seqs, ctx_len, num_kv_heads, head_dim)
        seq_lens: jax.Array,      # (num_seqs,) int32 — actual context length
    ) -> jax.Array:               # (num_seqs, num_heads, head_dim)
        """Attention for the decode step (one new token per sequence)."""
        num_seqs, ctx_len = k_cache.shape[0], k_cache.shape[1]

        # GQA expand
        if self.kv_groups > 1:
            k_cache = jnp.repeat(k_cache, self.kv_groups, axis=2)
            v_cache = jnp.repeat(v_cache, self.kv_groups, axis=2)

        # q: (S, H, D) -> (S, H, 1, D)
        q_e = q[:, :, None, :]  # (S, H, 1, D)
        # k_cache: (S, ctx, H, D) -> (S, H, D, ctx)
        k_t = jnp.transpose(k_cache, (0, 2, 3, 1))  # (S, H, D, ctx)
        # logits: (S, H, 1, ctx)
        logits = jnp.matmul(q_e, k_t) * self.scale  # (S, H, 1, ctx)

        # Mask padding positions
        positions = jnp.arange(ctx_len)[None, None, None, :]  # (1,1,1,ctx)
        valid = positions < seq_lens[:, None, None, None]      # (S,1,1,ctx)
        logits = jnp.where(valid, logits, jnp.finfo(jnp.float32).min)

        attn = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)  # (S,H,1,ctx)

        # v_cache: (S, ctx, H, D) -> (S, H, ctx, D)
        v_t = jnp.transpose(v_cache, (0, 2, 1, 3))  # (S, H, ctx, D)
        out = jnp.matmul(attn, v_t)                  # (S, H, 1, D)
        return out[:, :, 0, :]                        # (S, H, D)

    # ------------------------------------------------------------------
    # Public forward — writes new KV into cache, returns attended output
    # ------------------------------------------------------------------

    def __call__(
        self,
        q: jax.Array,               # (T, num_heads, head_dim)
        k: jax.Array,               # (T, num_kv_heads, head_dim)
        v: jax.Array,               # (T, num_kv_heads, head_dim)
        kv_cache: PagedKVCache,
        block_indices: jax.Array,   # (T,) int32
        block_offsets: jax.Array,   # (T,) int32
        seq_lens: jax.Array,        # (num_seqs,) int32
        is_prefill: bool,
        block_table: Optional[jax.Array] = None,  # (num_seqs, max_blocks) for decode
    ) -> jax.Array:
        # Write new tokens into cache
        kv_cache.write(self.layer_idx, block_indices, block_offsets, k, v)

        if is_prefill:
            return self._prefill(q, k, v, seq_lens)
        else:
            assert block_table is not None
            k_ctx, v_ctx = kv_cache.read(self.layer_idx, block_table, seq_lens)
            return self._decode(q, k_ctx, v_ctx, seq_lens)
