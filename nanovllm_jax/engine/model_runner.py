"""ModelRunner: assembles JAX input arrays and calls the model forward pass.

Status: ✅ Done (Phase 6 / 7)

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
JAX traces and compiles a new XLA program whenever array *shapes* change.
To avoid recompilation every decode step we **pad all inputs to the next
power-of-two size** before calling the JIT'd forward function, so the GPU
sees only O(log N) distinct shapes in practice.

- ``_jit_prefill(model, kv_cache, ids, pos, bi, bo, seq_lens)``
  Padded to the next power-of-two token count.

- ``_jit_decode(model, kv_cache, ids, pos, bi, bo, seq_lens, block_table)``
  Padded to the next power-of-two batch size and block-table width.

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= n (minimum 1)."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Stub model
# ---------------------------------------------------------------------------

class _StubModel:
    """Returns logits that argmax to eos_token_id — deterministic, terminates fast."""
    def __init__(self, vocab_size: int = 256, eos_token_id: int = 2):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.config = type("C", (), {"vocab_size": vocab_size})()

    def __call__(self, *args, **kwargs):
        import jax.numpy as jnp
        batch = args[0].shape[0] if args else 1
        logits = jnp.zeros((batch, self.vocab_size))
        logits = logits.at[:, self.eos_token_id].set(1.0)
        return logits

    def load_weights(self, params):
        pass


# ---------------------------------------------------------------------------
# JIT-compiled forward functions
# ---------------------------------------------------------------------------
# We use module-level @functools.partial(jax.jit) functions rather than
# methods so that JAX can trace them cleanly without capturing `self`.
# The model and kv_cache are passed as regular arguments; JAX traces on
# their *structure* (pytree) and recompiles only when shapes change.

import jax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnums=())
def _jit_prefill(model, kv_cache, ids, pos, bi, bo, seq_lens):
    """JIT-compiled prefill forward pass. Returns full logit matrix."""
    return model(ids, pos, kv_cache, bi, bo, seq_lens, is_prefill=True)


@functools.partial(jax.jit, static_argnums=())
def _jit_decode(model, kv_cache, ids, pos, bi, bo, seq_lens, block_table):
    """JIT-compiled decode forward pass. Returns full logit matrix."""
    return model(
        ids, pos, kv_cache, bi, bo, seq_lens,
        is_prefill=False, block_table=block_table,
    )


# ---------------------------------------------------------------------------
# ModelRunner
# ---------------------------------------------------------------------------

class ModelRunner:
    """Wraps model loading, input array construction, and forward pass.

    Accepts **two** different calling conventions:

    *Config mode* (used by LLMEngine)::

        runner = ModelRunner(config)   # EngineConfig

    *Direct mode* (used by unit tests — avoids loading a real checkpoint)::

        runner = ModelRunner(
            model=<model_or_None>,
            kv_cache=<PagedKVCache>,
            block_manager=<BlockManager>,
            block_size=<int>,
        )
    """

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

    # ---- lazy init (config mode) ----------------------------------------

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
        """Pad a 1-D int array to target_len."""
        pad = target_len - arr.shape[0]
        return jnp.pad(arr, (0, pad), constant_values=val)

    @staticmethod
    def _pad2d(arr, target_rows: int, target_cols: int, val: int = 0):
        """Pad a 2-D int array to (target_rows, target_cols)."""
        pr = target_rows - arr.shape[0]
        pc = target_cols - arr.shape[1]
        return jnp.pad(arr, ((0, pr), (0, pc)), constant_values=val)

    # ---- input builders -------------------------------------------------

    def _build_prefill_inputs(self, seqs: List[Sequence]) -> Tuple:
        """Pack multiple prefill sequences into flat JAX arrays.

        Returns:
            (seqs, token_ids, positions, block_indices, block_offsets,
             seq_lens, last_indices)
        """
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
        """Build one-token-per-sequence decode arrays.

        Returns:
            (seqs, token_ids, positions, block_indices, block_offsets,
             seq_lens, block_table)

        block_table is 2-D int32 [B, max_blocks], padded with 0.
        """
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
        """Run one forward step; return {seq_id: next_token_id}."""
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

        # Pad token dimension to next power-of-2 to minimise JIT recompilation.
        T = ids.shape[0]
        T_pad = _next_pow2(T)
        if T_pad != T:
            ids     = self._pad1d(ids,  T_pad)
            pos     = self._pad1d(pos,  T_pad)
            bi      = self._pad1d(bi,   T_pad)
            bo      = self._pad1d(bo,   T_pad)

        logits = _jit_prefill(self._model, self._kv_cache, ids, pos, bi, bo, seq_lens)
        # logits shape: [T_pad, vocab] — read only at the real last-token positions
        return {seq.seq_id: int(jnp.argmax(logits[last_idx[i]])) for i, seq in enumerate(seqs)}

    def _run_decode(self, seqs: List[Sequence]) -> Dict[int, int]:
        self._ensure_model()
        if isinstance(self._model, _StubModel):
            return {seq.seq_id: self._model.eos_token_id for seq in seqs}

        _, ids, pos, bi, bo, seq_lens, block_table = self._build_decode_inputs(seqs)

        # Pad batch and block-table width to next power-of-2.
        B  = ids.shape[0]
        BT = block_table.shape[1]
        B_pad  = _next_pow2(B)
        BT_pad = _next_pow2(BT)
        if B_pad != B or BT_pad != BT:
            ids         = self._pad1d(ids,   B_pad)
            pos         = self._pad1d(pos,   B_pad)
            bi          = self._pad1d(bi,    B_pad)
            bo          = self._pad1d(bo,    B_pad)
            seq_lens    = self._pad1d(seq_lens, B_pad, val=1)  # avoid div-by-zero
            block_table = self._pad2d(block_table, B_pad, BT_pad)

        logits = _jit_decode(
            self._model, self._kv_cache,
            ids, pos, bi, bo, seq_lens, block_table,
        )
        # logits shape: [B_pad, vocab] — read only the first B real rows
        return {seq.seq_id: int(jnp.argmax(logits[i])) for i, seq in enumerate(seqs)}
