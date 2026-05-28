"""ModelRunner: assembles JAX input arrays and calls the model forward pass.

Status: ✅ Fixed (Bug #1 KV-cache persistence, Bug #2 last_indices, Bug #3 padding)

Two construction modes
----------------------
1. **Full engine mode** (used by LLMEngine)::

       runner = ModelRunner(config)   # EngineConfig

2. **Direct / test mode** (used by unit tests — avoids loading a real checkpoint)::

       runner = ModelRunner(
           model=None,               # None = stub
           kv_cache=cache,
           block_manager=bm,
           block_size=8,
       )

JIT strategy
------------
We use ``nnx.jit`` (not ``jax.jit``) for all model calls.

**Why nnx.jit is required:**
JAX's ``jax.jit`` compiles pure functions. Any mutation to Python objects
inside a jitted function — including ``self.cache = nnx.Variable(new_cache)``
in ``PagedKVCache.write`` — is NOT propagated back to the outer Python scope
after the call returns. This means every decode step would read an all-zeros
KV cache, producing garbage attention outputs and garbled tokens.

``nnx.jit`` is NNX-aware: it automatically extracts the ``nnx.Variable``
state from all NNX modules, passes them through XLA as mutable arrays, and
writes the updated values back into the Python objects after each call.
This makes KV cache writes performed during prefill visible to subsequent
decode steps.

num_real_tokens / num_real_seqs are passed as Python ints (static values)
so Attention can slice to real data before writing the KV cache, preventing
padding tokens from corrupting valid cache entries.

Public helpers
--------------
- ``_build_prefill_inputs(seqs)``  →  (seqs, ids, pos, bi, bo, seq_lens, last_indices)
- ``_build_decode_inputs(seqs)``   →  (seqs, ids, pos, bi, bo, seq_lens, block_table)
- ``run(prefill_seqs, decode_seqs)`` →  {seq_id: next_token_id}
"""
from __future__ import annotations
import functools
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from nanovllm_jax.config import EngineConfig
    from nanovllm_jax.engine.block_manager import BlockManager
    from nanovllm_jax.layers.attention import PagedKVCache

from nanovllm_jax.engine.sequence import Sequence

import jax
import jax.numpy as jnp
from flax import nnx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Stub model
# ---------------------------------------------------------------------------

class _StubModel(nnx.Module):
    """Minimal NNX module that always predicts eos_token_id.

    Must be an nnx.Module so nnx.jit can trace it.
    """
    def __init__(self, vocab_size: int = 256, eos_token_id: int = 2):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.config = type("C", (), {"vocab_size": vocab_size})()
        # Dummy variable so NNX has something to trace
        self._dummy = nnx.Variable(jnp.zeros(()))

    def __call__(self, *args, **kwargs):
        ids = args[0]
        batch = ids.shape[0]
        logits = jnp.zeros((batch, self.vocab_size))
        return logits.at[:, self.eos_token_id].set(1.0)

    def load_weights(self, params):
        pass


# ---------------------------------------------------------------------------
# JIT-compiled forward functions using nnx.jit
#
# IMPORTANT: nnx.jit (not jax.jit) is required here.
#
# PagedKVCache.write() mutates self.cache (an nnx.Variable) inside the
# forward pass. With plain jax.jit this mutation is invisible outside the
# JIT boundary — every decode step would see an all-zeros cache.
# nnx.jit extracts the Variable state, passes it through XLA as mutable
# arrays, then writes updated values back into the Python objects after
# each call.  This is the standard Flax NNX pattern for stateful modules.
# ---------------------------------------------------------------------------

@functools.partial(nnx.jit, static_argnames=("num_real_tokens",))
def _jit_prefill(model, kv_cache, ids, pos, bi, bo, seq_lens, last_indices, *, num_real_tokens):
    """Prefill forward pass.

    Args:
        model:           LlamaForCausalLM (nnx.Module)
        kv_cache:        PagedKVCache (nnx.Module) — mutated in-place by this call
        ids:             (T_pad,) int32 padded token ids
        pos:             (T_pad,) int32 padded positions
        bi:              (T_pad,) int32 padded block indices
        bo:              (T_pad,) int32 padded block offsets
        seq_lens:        (num_seqs,) int32 real sequence lengths
        last_indices:    (num_seqs,) int32 index of last real token per sequence
        num_real_tokens: Python int — number of real (non-padding) tokens

    Returns:
        logits: (num_seqs, vocab_size) — one logit row per sequence
    """
    return model(
        ids, pos, kv_cache, bi, bo, seq_lens,
        is_prefill=True,
        last_indices=last_indices,
        num_real_tokens=num_real_tokens,
    )


@functools.partial(nnx.jit, static_argnames=("num_real_seqs",))
def _jit_decode(model, kv_cache, ids, pos, bi, bo, seq_lens, block_table, *, num_real_seqs):
    """Decode forward pass (one token per sequence).

    Args:
        model:        LlamaForCausalLM (nnx.Module)
        kv_cache:     PagedKVCache (nnx.Module) — mutated in-place by this call
        ids:          (B_pad,) int32 padded token ids
        pos:          (B_pad,) int32 padded positions
        bi:           (B_pad,) int32 padded block indices
        bo:           (B_pad,) int32 padded block offsets
        seq_lens:     (B_pad,) int32 padded sequence lengths (0 for padding rows)
        block_table:  (B_pad, BT_pad) int32 padded block table
        num_real_seqs: Python int — number of real (non-padding) sequences

    Returns:
        logits: (B_pad, vocab_size) — only rows [:num_real_seqs] are meaningful
    """
    # For decode, last_indices is not needed: each row corresponds to exactly
    # one sequence's single new token, so lm_head(hidden) returns [B_pad, vocab]
    # and the runner reads rows by enumerate index.
    return model(
        ids, pos, kv_cache, bi, bo, seq_lens,
        is_prefill=False,
        block_table=block_table,
        num_real_seqs=num_real_seqs,
    )


# ---------------------------------------------------------------------------
# ModelRunner
# ---------------------------------------------------------------------------

class ModelRunner:
    def __init__(
        self,
        config: Optional["EngineConfig"] = None,
        *,
        model=None,
        kv_cache: Optional["PagedKVCache"] = None,
        block_manager: Optional["BlockManager"] = None,
        block_size: Optional[int] = None,
    ):
        if config is not None:
            self._config_mode = True
            self.config = config
            self.block_size = config.block_size
            self._model = None
            self._kv_cache = None
            self._block_manager = None
        else:
            self._config_mode = False
            self.config = None
            self.block_size = block_size or 16
            self._model = model
            self._kv_cache = kv_cache
            self._block_manager = block_manager

    def _ensure_model(self):
        if self._model is not None:
            return
        if self._config_mode:
            cfg = self.config
            if not cfg.model or cfg.enforce_eager:
                self._model = _StubModel(eos_token_id=cfg.eos_token_id)
                self._init_kv_cache_stub()
            else:
                self._load_real_model()
        else:
            self._model = _StubModel()

    def _init_kv_cache_stub(self):
        from nanovllm_jax.layers.attention import PagedKVCache
        cfg = self.config
        self._kv_cache = PagedKVCache(
            num_layers=2, num_kv_heads=2, head_dim=16,
            num_blocks=cfg.num_gpu_blocks or 32,
            block_size=cfg.block_size, dtype=jnp.float32,
        )

    def _load_real_model(self):
        from nanovllm_jax.loader.model_registry import load_model
        from nanovllm_jax.layers.attention import PagedKVCache
        cfg = self.config
        self._model = load_model(cfg.model, dtype=cfg.dtype)
        mc = self._model.config
        self._kv_cache = PagedKVCache(
            num_layers=mc.num_hidden_layers,
            num_kv_heads=mc.num_key_value_heads,
            head_dim=mc.head_dim,
            num_blocks=cfg.num_gpu_blocks,
            block_size=cfg.block_size,
            dtype=jnp.bfloat16 if cfg.dtype == "bfloat16" else jnp.float32,
        )

    # ---- padding helpers ------------------------------------------------

    @staticmethod
    def _pad1d(arr, target_len: int, val: int = 0):
        pad = target_len - arr.shape[0]
        return jnp.pad(arr, (0, pad), constant_values=val)

    @staticmethod
    def _pad2d(arr, target_rows: int, target_cols: int, val: int = 0):
        pr = target_rows - arr.shape[0]
        pc = target_cols - arr.shape[1]
        return jnp.pad(arr, ((0, pr), (0, pc)), constant_values=val)

    # ---- input builders -------------------------------------------------

    def _build_prefill_inputs(self, seqs: List[Sequence]) -> Tuple:
        all_ids: List[int] = []
        all_pos: List[int] = []
        all_bi:  List[int] = []
        all_bo:  List[int] = []
        seq_lens: List[int] = []
        last_indices: List[int] = []
        offset = 0
        for seq in seqs:
            tokens = seq.all_token_ids
            T = len(tokens)
            all_ids.extend(tokens)
            all_pos.extend(range(T))
            for i in range(T):
                blk_num = i // self.block_size
                blk_off = i % self.block_size
                all_bi.append(seq.block_table[blk_num] if seq.block_table else 0)
                all_bo.append(blk_off)
            seq_lens.append(T)
            last_indices.append(offset + T - 1)
            offset += T
        return (
            seqs,
            jnp.array(all_ids,      dtype=jnp.int32),
            jnp.array(all_pos,      dtype=jnp.int32),
            jnp.array(all_bi,       dtype=jnp.int32),
            jnp.array(all_bo,       dtype=jnp.int32),
            jnp.array(seq_lens,     dtype=jnp.int32),
            jnp.array(last_indices, dtype=jnp.int32),
        )

    def _build_decode_inputs(self, seqs: List[Sequence]) -> Tuple:
        ids:   List[int] = []
        pos:   List[int] = []
        bi:    List[int] = []
        bo:    List[int] = []
        slens: List[int] = []
        block_tables: List[List[int]] = []
        for seq in seqs:
            step = seq.total_len - 1
            ids.append(seq.last_token_id)
            pos.append(step)
            blk_num = step // self.block_size
            blk_off = step % self.block_size
            bi.append(seq.block_table[blk_num] if seq.block_table else 0)
            bo.append(blk_off)
            slens.append(seq.total_len)
            block_tables.append(list(seq.block_table))
        max_blocks = max(len(bt) for bt in block_tables) if block_tables else 1
        padded = [bt + [0] * (max_blocks - len(bt)) for bt in block_tables]
        return (
            seqs,
            jnp.array(ids,    dtype=jnp.int32),
            jnp.array(pos,    dtype=jnp.int32),
            jnp.array(bi,     dtype=jnp.int32),
            jnp.array(bo,     dtype=jnp.int32),
            jnp.array(slens,  dtype=jnp.int32),
            jnp.array(padded, dtype=jnp.int32),
        )

    # ---- forward pass ---------------------------------------------------

    def run(self, prefill_seqs: List[Sequence], decode_seqs: List[Sequence]) -> Dict[int, int]:
        self._ensure_model()
        results: Dict[int, int] = {}
        if prefill_seqs:
            results.update(self._run_prefill(prefill_seqs))
        if decode_seqs:
            results.update(self._run_decode(decode_seqs))
        return results

    def _run_prefill(self, seqs: List[Sequence]) -> Dict[int, int]:
        self._ensure_model()
        if isinstance(self._model, _StubModel):
            return {seq.seq_id: self._model.eos_token_id for seq in seqs}

        _, ids, pos, bi, bo, seq_lens, last_idx = self._build_prefill_inputs(seqs)

        T_real = ids.shape[0]
        T_pad  = _next_pow2(T_real)
        if T_pad != T_real:
            ids = self._pad1d(ids, T_pad)
            pos = self._pad1d(pos, T_pad)
            bi  = self._pad1d(bi,  T_pad)
            bo  = self._pad1d(bo,  T_pad)

        # Pass last_idx so lm_head returns shape [num_seqs, vocab] directly.
        # nnx.jit ensures KV cache writes are persisted after this call.
        logits = _jit_prefill(
            self._model, self._kv_cache,
            ids, pos, bi, bo, seq_lens, last_idx,
            num_real_tokens=T_real,
        )
        # logits: [num_seqs, vocab] — one row per sequence (last_indices applied inside model)
        return {seq.seq_id: int(jnp.argmax(logits[i])) for i, seq in enumerate(seqs)}

    def _run_decode(self, seqs: List[Sequence]) -> Dict[int, int]:
        self._ensure_model()
        if isinstance(self._model, _StubModel):
            return {seq.seq_id: self._model.eos_token_id for seq in seqs}

        _, ids, pos, bi, bo, seq_lens, block_table = self._build_decode_inputs(seqs)

        B_real  = ids.shape[0]
        BT_real = block_table.shape[1]
        B_pad   = _next_pow2(B_real)
        BT_pad  = _next_pow2(BT_real)
        if B_pad != B_real or BT_pad != BT_real:
            ids         = self._pad1d(ids,      B_pad)
            pos         = self._pad1d(pos,      B_pad)
            bi          = self._pad1d(bi,       B_pad)
            bo          = self._pad1d(bo,       B_pad)
            # Pad seq_lens with 0, not 1.  Using val=1 would make fake padding
            # sequences appear to have length 1, allowing them to attend to
            # position 0 of the KV cache and potentially corrupt real sequences.
            seq_lens    = self._pad1d(seq_lens, B_pad, val=0)
            block_table = self._pad2d(block_table, B_pad, BT_pad)

        # nnx.jit ensures KV cache writes (new decode token) are persisted.
        logits = _jit_decode(
            self._model, self._kv_cache,
            ids, pos, bi, bo, seq_lens, block_table,
            num_real_seqs=B_real,
        )
        # logits: [B_pad, vocab] — read only real rows
        return {seq.seq_id: int(jnp.argmax(logits[i])) for i, seq in enumerate(seqs)}
