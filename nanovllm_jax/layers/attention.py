"""Paged KV Cache and Attention (JAX port of nanovllm/layers/attention.py).

Status: ✅ Done (Phase 3)

Design notes vs PyTorch original:
- PagedKVCache uses jax.lax.dynamic_update_slice / dynamic_slice instead of
  in-place scatter. All ops are jit-compilable.
- Attention.forward dispatches to _prefill (full causal softmax) or
  _decode (single-step KV lookup) based on is_prefill flag.
- No flash-attention yet — plain scaled dot-product in float32.
  Flash attention (via jax.nn.dot_product_attention) is Phase 6.

Shape contract
--------------
__call__ always returns the same leading dimension as the input q:
  prefill : (T_pad, num_heads, head_dim) — padding rows are zero
  decode  : (B_pad, num_heads, head_dim) — padding rows are zero

This keeps all downstream layers (residual adds, layernorm) shape-stable.
num_real_tokens / num_real_seqs only govern which slots are written to
the KV cache; they do NOT change the output tensor shape.
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
        cache = jnp.zeros(
            (num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype,
        )
        self.cache = nnx.Variable(cache)

    def write(
        self,
        layer_idx: int,
        block_indices: jax.Array,   # (num_real_tokens,) int32
        block_offsets: jax.Array,   # (num_real_tokens,) int32
        keys: jax.Array,            # (num_real_tokens, num_kv_heads, head_dim)
        values: jax.Array,          # (num_real_tokens, num_kv_heads, head_dim)
    ) -> None:
        """Write *real* tokens into the cache.  Padding tokens must NOT be included."""
        cache = self.cache.get_value()
        num_real = keys.shape[0]

        def body(i, c):
            bi = block_indices[i]
            bo = block_offsets[i]
            c = c.at[layer_idx, 0, bi, bo].set(keys[i].astype(self.dtype))
            c = c.at[layer_idx, 1, bi, bo].set(values[i].astype(self.dtype))
            return c

        new_cache = jax.lax.fori_loop(0, num_real, body, cache)
        self.cache = nnx.Variable(new_cache)

    def read(
        self,
        layer_idx: int,
        block_table: jax.Array,   # (num_real_seqs, max_blocks_per_seq) int32
        seq_lens: jax.Array,      # (num_real_seqs,) int32
    ) -> tuple[jax.Array, jax.Array]:
        """Read cached KV for *real* sequences only."""
        cache = self.cache.get_value()
        k_cache = cache[layer_idx, 0]
        v_cache = cache[layer_idx, 1]

        k_gathered = k_cache[block_table]
        v_gathered = v_cache[block_table]

        num_real_seqs = block_table.shape[0]
        max_seq_len   = block_table.shape[1] * self.block_size
        k_out = k_gathered.reshape(num_real_seqs, max_seq_len, self.num_kv_heads, self.head_dim)
        v_out = v_gathered.reshape(num_real_seqs, max_seq_len, self.num_kv_heads, self.head_dim)
        return k_out.astype(jnp.float32), v_out.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nnx.Module):
    """Multi-head attention with GQA support and paged KV cache."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        layer_idx: int = 0,
        scale: Optional[float] = None,
    ) -> None:
        self.num_heads    = num_heads
        self.head_dim     = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.layer_idx    = layer_idx
        self.scale        = scale or (head_dim ** -0.5)
        self.kv_groups    = self.num_heads // self.num_kv_heads

    # ------------------------------------------------------------------
    # Prefill: causal attention over T_real tokens
    # Returns (T_real, num_heads, head_dim) — caller pads back to T_pad
    # ------------------------------------------------------------------

    def _prefill(
        self,
        q: jax.Array,         # (T_real, num_heads, head_dim)
        k: jax.Array,         # (T_real, num_kv_heads, head_dim)
        v: jax.Array,         # (T_real, num_kv_heads, head_dim)
        seq_lens: jax.Array,  # (num_seqs,) int32
    ) -> jax.Array:           # (T_real, num_heads, head_dim)
        T = q.shape[0]
        if self.kv_groups > 1:
            k = jnp.repeat(k, self.kv_groups, axis=1)
            v = jnp.repeat(v, self.kv_groups, axis=1)

        q_t = jnp.transpose(q, (1, 0, 2))  # (H, T, D)
        k_t = jnp.transpose(k, (1, 2, 0))  # (H, D, T)
        logits = jnp.matmul(q_t, k_t) * self.scale  # (H, T, T)

        mask   = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        logits = jnp.where(mask[None, :, :], logits, jnp.finfo(jnp.float32).min)

        attn = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        v_t  = jnp.transpose(v, (1, 0, 2))   # (H, T, D)
        out  = jnp.matmul(attn, v_t)          # (H, T, D)
        return jnp.transpose(out, (1, 0, 2))  # (T, H, D)

    # ------------------------------------------------------------------
    # Decode: single token per real sequence
    # Returns (B_real, num_heads, head_dim) — caller pads back to B_pad
    # ------------------------------------------------------------------

    def _decode(
        self,
        q: jax.Array,         # (B_real, num_heads, head_dim)
        k_cache: jax.Array,   # (B_real, ctx_len, num_kv_heads, head_dim)
        v_cache: jax.Array,
        seq_lens: jax.Array,  # (B_real,) int32
    ) -> jax.Array:           # (B_real, num_heads, head_dim)
        ctx_len = k_cache.shape[1]
        if self.kv_groups > 1:
            k_cache = jnp.repeat(k_cache, self.kv_groups, axis=2)
            v_cache = jnp.repeat(v_cache, self.kv_groups, axis=2)

        q_e   = q[:, :, None, :]                        # (B, H, 1, D)
        k_t   = jnp.transpose(k_cache, (0, 2, 3, 1))   # (B, H, D, ctx)
        logits = jnp.matmul(q_e, k_t) * self.scale     # (B, H, 1, ctx)

        positions = jnp.arange(ctx_len)[None, None, None, :]
        valid     = positions < seq_lens[:, None, None, None]
        logits    = jnp.where(valid, logits, jnp.finfo(jnp.float32).min)

        attn = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        v_t  = jnp.transpose(v_cache, (0, 2, 1, 3))    # (B, H, ctx, D)
        out  = jnp.matmul(attn, v_t)                    # (B, H, 1, D)
        return out[:, :, 0, :]                           # (B, H, D)

    # ------------------------------------------------------------------
    # Public forward
    #
    # Output shape == input q shape (T_pad or B_pad as leading dim).
    # Padding rows in the output are zero.
    # ------------------------------------------------------------------

    def __call__(
        self,
        q: jax.Array,               # (T_pad, num_heads, head_dim)
        k: jax.Array,
        v: jax.Array,
        kv_cache: PagedKVCache,
        block_indices: jax.Array,
        block_offsets: jax.Array,
        seq_lens: jax.Array,
        is_prefill: bool,
        block_table: Optional[jax.Array] = None,
        num_real_tokens: Optional[int] = None,
        num_real_seqs: Optional[int] = None,
    ) -> jax.Array:                 # same leading dim as q
        if is_prefill:
            T_pad  = q.shape[0]
            T_real = num_real_tokens if num_real_tokens is not None else T_pad

            # Only write real tokens to avoid cache corruption
            kv_cache.write(
                self.layer_idx,
                block_indices[:T_real],
                block_offsets[:T_real],
                k[:T_real],
                v[:T_real],
            )
            out_real = self._prefill(q[:T_real], k[:T_real], v[:T_real], seq_lens)

            # Pad output back to T_pad so residual adds remain shape-stable
            if T_real < T_pad:
                pad = jnp.zeros(
                    (T_pad - T_real, self.num_heads, self.head_dim),
                    dtype=out_real.dtype,
                )
                return jnp.concatenate([out_real, pad], axis=0)
            return out_real

        else:
            assert block_table is not None
            B_pad  = q.shape[0]
            B_real = num_real_seqs if num_real_seqs is not None else B_pad

            # Only write the new decode token for real sequences
            kv_cache.write(
                self.layer_idx,
                block_indices[:B_real],
                block_offsets[:B_real],
                k[:B_real],
                v[:B_real],
            )
            k_ctx, v_ctx = kv_cache.read(
                self.layer_idx,
                block_table[:B_real],
                seq_lens[:B_real],
            )
            out_real = self._decode(q[:B_real], k_ctx, v_ctx, seq_lens[:B_real])

            # Pad output back to B_pad
            if B_real < B_pad:
                pad = jnp.zeros(
                    (B_pad - B_real, self.num_heads, self.head_dim),
                    dtype=out_real.dtype,
                )
                return jnp.concatenate([out_real, pad], axis=0)
            return out_real
