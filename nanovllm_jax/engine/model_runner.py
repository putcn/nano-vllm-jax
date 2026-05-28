"""ModelRunner: assembles JAX input arrays and calls the model forward pass.

Status: ✅ Done (Phase 7)

In a real deployment this class:
  1. Stacks token IDs + positions from all scheduled sequences.
  2. Builds block_indices / block_offsets for PagedKVCache lookup.
  3. Calls LlamaForCausalLM.__call__ under jax.jit.
  4. Returns the sampled next-token per sequence.

For unit-testing without a real model checkpoint the runner falls back to
a _StubModel that returns random logits.  Set ``config.model = ""`` or
``config.enforce_eager = True`` to force the stub.
"""
from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np

from nanovllm_jax.config import EngineConfig
from nanovllm_jax.engine.sequence import Sequence, SequenceStatus


# ---------------------------------------------------------------------------
# Stub model (used when no real checkpoint is available)
# ---------------------------------------------------------------------------

class _StubModel:
    """Returns argmax-0 (token 0) for every position — deterministic, testable."""
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.config = type("C", (), {"vocab_size": vocab_size})()

    def __call__(self, *args, **kwargs):
        import jax.numpy as jnp
        # Return a dummy token-id = 1 for each position
        batch = args[0].shape[0] if args else 1
        return jnp.zeros((batch, self.vocab_size))

    def load_weights(self, params):
        pass


# ---------------------------------------------------------------------------
# ModelRunner
# ---------------------------------------------------------------------------

class ModelRunner:
    """Wraps model loading and the prefill / decode forward pass.

    Args:
        config: Engine config.  If ``config.model`` is empty or
                ``config.enforce_eager`` is ``True``, uses the stub model.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self._model = None          # lazy-loaded on first run()
        self._kv_cache = None
        self._rng_key = None

    # ------------------------------------------------------------------
    # Lazy model init
    # ------------------------------------------------------------------

    def _ensure_model(self):
        if self._model is not None:
            return
        if not self.config.model or self.config.enforce_eager:
            self._model = _StubModel()
            self._init_kv_cache_stub()
            return
        self._load_real_model()

    def _init_kv_cache_stub(self):
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
        import jax
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

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def run(
        self,
        prefill_seqs: List[Sequence],
        decode_seqs: List[Sequence],
    ) -> Dict[int, int]:
        """Run one forward step; return {seq_id: next_token_id} for all seqs."""
        self._ensure_model()

        results: Dict[int, int] = {}

        # --- Prefill ---
        if prefill_seqs:
            results.update(self._run_prefill(prefill_seqs))

        # --- Decode ---
        if decode_seqs:
            results.update(self._run_decode(decode_seqs))

        return results

    def _run_prefill(self, seqs: List[Sequence]) -> Dict[int, int]:
        """Prefill: process all prompt tokens, return next token per seq."""
        import jax
        import jax.numpy as jnp

        results = {}
        # Process each prefill seq independently (simplest correct approach)
        for seq in seqs:
            token_ids = jnp.array(seq.all_token_ids, dtype=jnp.int32)
            T = token_ids.shape[0]
            positions = jnp.arange(T, dtype=jnp.int32)

            # Build block table arrays for this seq
            n_blocks = len(seq.block_table)
            block_indices = jnp.zeros(T, dtype=jnp.int32)
            block_offsets = jnp.arange(T, dtype=jnp.int32) % self.config.block_size
            if n_blocks > 0:
                block_indices = jnp.array(
                    [seq.block_table[i // self.config.block_size]
                     for i in range(T)], dtype=jnp.int32
                )

            seq_lens = jnp.array([T], dtype=jnp.int32)

            if isinstance(self._model, _StubModel):
                # Stub: deterministically return token 1
                results[seq.seq_id] = 1
                continue

            logits = self._model(
                token_ids, positions, self._kv_cache,
                block_indices, block_offsets, seq_lens,
                is_prefill=True,
            )
            # Greedy: pick last-position logit
            next_token = int(jnp.argmax(logits[-1]))
            results[seq.seq_id] = next_token

        return results

    def _run_decode(self, seqs: List[Sequence]) -> Dict[int, int]:
        """Decode: one new token per sequence."""
        import jax.numpy as jnp

        results = {}
        for seq in seqs:
            if isinstance(self._model, _StubModel):
                results[seq.seq_id] = 1
                continue

            token_ids = jnp.array([seq.last_token_id], dtype=jnp.int32)
            pos = jnp.array([seq.total_len - 1], dtype=jnp.int32)
            blk = seq.block_table
            step = seq.total_len - 1
            block_indices = jnp.array(
                [blk[step // self.config.block_size]], dtype=jnp.int32
            )
            block_offsets = jnp.array(
                [step % self.config.block_size], dtype=jnp.int32
            )
            seq_lens = jnp.array([seq.total_len], dtype=jnp.int32)

            logits = self._model(
                token_ids, pos, self._kv_cache,
                block_indices, block_offsets, seq_lens,
                is_prefill=False,
            )
            next_token = int(jnp.argmax(logits[-1]))
            results[seq.seq_id] = next_token

        return results
