"""ModelRunner: assembles JAX input arrays and calls the model forward pass.

Status: ✅ Done (Phase 7)

_StubModel returns eos_token_id=2 so that generate_all() terminates
immediately in unit tests without a real checkpoint.
"""
from __future__ import annotations
from typing import Dict, List

import numpy as np

from nanovllm_jax.config import EngineConfig
from nanovllm_jax.engine.sequence import Sequence


class _StubModel:
    """Returns logits that argmax to eos_token_id — deterministic, terminates fast."""
    def __init__(self, vocab_size: int = 256, eos_token_id: int = 2):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.config = type("C", (), {"vocab_size": vocab_size})()

    def __call__(self, *args, **kwargs):
        import jax.numpy as jnp
        batch = args[0].shape[0] if args else 1
        # All logits zero except eos position -> argmax = eos_token_id
        logits = jnp.zeros((batch, self.vocab_size))
        logits = logits.at[:, self.eos_token_id].set(1.0)
        return logits

    def load_weights(self, params):
        pass


class ModelRunner:
    def __init__(self, config: EngineConfig):
        self.config = config
        self._model = None
        self._kv_cache = None
        self._rng_key = None

    def _ensure_model(self):
        if self._model is not None:
            return
        if not self.config.model or self.config.enforce_eager:
            self._model = _StubModel(
                eos_token_id=self.config.eos_token_id,
            )
            self._init_kv_cache_stub()
            return
        self._load_real_model()

    def _init_kv_cache_stub(self):
        import jax
        import jax.numpy as jnp
        from nanovllm_jax.layers.attention import PagedKVCache
        self._kv_cache = PagedKVCache(
            num_layers=2,
            num_kv_heads=2,
            head_dim=16,
            num_blocks=self.config.num_gpu_blocks or 32,
            block_size=self.config.block_size,
            dtype=jnp.float32,
        )
        self._rng_key = jax.random.PRNGKey(0)

    def _load_real_model(self):
        import jax
        import jax.numpy as jnp
        from nanovllm_jax.loader.model_registry import load_model
        from nanovllm_jax.layers.attention import PagedKVCache
        self._model = load_model(self.config.model, dtype=self.config.dtype)
        cfg = self._model.config
        self._kv_cache = PagedKVCache(
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            num_blocks=self.config.num_gpu_blocks,
            block_size=self.config.block_size,
            dtype=jnp.bfloat16 if self.config.dtype == "bfloat16" else jnp.float32,
        )
        self._rng_key = jax.random.PRNGKey(0)

    def run(self, prefill_seqs: List[Sequence], decode_seqs: List[Sequence]) -> Dict[int, int]:
        self._ensure_model()
        results: Dict[int, int] = {}
        if prefill_seqs:
            results.update(self._run_prefill(prefill_seqs))
        if decode_seqs:
            results.update(self._run_decode(decode_seqs))
        return results

    def _run_prefill(self, seqs: List[Sequence]) -> Dict[int, int]:
        import jax.numpy as jnp
        results = {}
        for seq in seqs:
            if isinstance(self._model, _StubModel):
                results[seq.seq_id] = self._model.eos_token_id
                continue
            token_ids = jnp.array(seq.all_token_ids, dtype=jnp.int32)
            T = token_ids.shape[0]
            positions = jnp.arange(T, dtype=jnp.int32)
            block_indices = jnp.zeros(T, dtype=jnp.int32)
            block_offsets = jnp.arange(T, dtype=jnp.int32) % self.config.block_size
            if seq.block_table:
                block_indices = jnp.array(
                    [seq.block_table[i // self.config.block_size] for i in range(T)],
                    dtype=jnp.int32,
                )
            seq_lens = jnp.array([T], dtype=jnp.int32)
            logits = self._model(
                token_ids, positions, self._kv_cache,
                block_indices, block_offsets, seq_lens, is_prefill=True,
            )
            results[seq.seq_id] = int(jnp.argmax(logits[-1]))
        return results

    def _run_decode(self, seqs: List[Sequence]) -> Dict[int, int]:
        import jax.numpy as jnp
        results = {}
        for seq in seqs:
            if isinstance(self._model, _StubModel):
                results[seq.seq_id] = self._model.eos_token_id
                continue
            token_ids = jnp.array([seq.last_token_id], dtype=jnp.int32)
            pos = jnp.array([seq.total_len - 1], dtype=jnp.int32)
            step = seq.total_len - 1
            block_indices = jnp.array(
                [seq.block_table[step // self.config.block_size]], dtype=jnp.int32
            )
            block_offsets = jnp.array([step % self.config.block_size], dtype=jnp.int32)
            seq_lens = jnp.array([seq.total_len], dtype=jnp.int32)
            logits = self._model(
                token_ids, pos, self._kv_cache,
                block_indices, block_offsets, seq_lens, is_prefill=False,
            )
            results[seq.seq_id] = int(jnp.argmax(logits[-1]))
        return results
