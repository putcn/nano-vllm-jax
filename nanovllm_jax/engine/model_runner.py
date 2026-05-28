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

Public helpers
--------------
- ``_build_prefill_inputs(seqs)``  →  (seqs, ids, pos, bi, bo, seq_lens, last_indices)
- ``_build_decode_inputs(seqs)``   →  (seqs, ids, pos, bi, bo, seq_lens, block_table)
- ``run(prefill_seqs, decode_seqs)`` →  {seq_id: next_token_id}
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from nanovllm_jax.config import EngineConfig
    from nanovllm_jax.engine.block_manager import BlockManager
    from nanovllm_jax.layers.attention import PagedKVCache

from nanovllm_jax.engine.sequence import Sequence


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
        import jax.numpy as jnp
        from nanovllm_jax.layers.attention import PagedKVCache
        cfg = self.config
        self._kv_cache = PagedKVCache(
            num_layers=2, num_kv_heads=2, head_dim=16,
            num_blocks=cfg.num_gpu_blocks or 32,
            block_size=cfg.block_size, dtype=jnp.float32,
        )

    def _load_real_model(self):
        import jax.numpy as jnp
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

    # ---- input builders -------------------------------------------------

    def _build_prefill_inputs(self, seqs: List[Sequence]) -> Tuple:
        """Pack multiple prefill sequences into flat JAX arrays.

        Returns:
            (seqs, token_ids, positions, block_indices, block_offsets,
             seq_lens, last_indices)
        """
        import jax.numpy as jnp
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
        import jax.numpy as jnp
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
        import jax.numpy as jnp
        self._ensure_model()
        if isinstance(self._model, _StubModel):
            return {seq.seq_id: self._model.eos_token_id for seq in seqs}
        _, ids, pos, bi, bo, seq_lens, last_idx = self._build_prefill_inputs(seqs)
        logits = self._model(ids, pos, self._kv_cache, bi, bo, seq_lens, is_prefill=True)
        return {seq.seq_id: int(jnp.argmax(logits[last_idx[i]])) for i, seq in enumerate(seqs)}

    def _run_decode(self, seqs: List[Sequence]) -> Dict[int, int]:
        import jax.numpy as jnp
        self._ensure_model()
        if isinstance(self._model, _StubModel):
            return {seq.seq_id: self._model.eos_token_id for seq in seqs}
        _, ids, pos, bi, bo, seq_lens, block_table = self._build_decode_inputs(seqs)
        logits = self._model(
            ids, pos, self._kv_cache, bi, bo, seq_lens,
            is_prefill=False, block_table=block_table,
        )
        return {seq.seq_id: int(jnp.argmax(logits[i])) for i, seq in enumerate(seqs)}
