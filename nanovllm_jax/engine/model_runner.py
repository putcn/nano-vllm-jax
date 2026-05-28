"""Model runner: builds JAX input arrays and runs one forward step (Phase 6).

Bridges the Python-level scheduler output to the JAX model.
Replaces torch.cuda.synchronize() with jax.effects_barrier().
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp

from nanovllm_jax.engine.block_manager import BlockManager
from nanovllm_jax.engine.scheduler import SchedulerOutput
from nanovllm_jax.engine.sequence import Sequence
from nanovllm_jax.layers.attention import PagedKVCache


class ModelRunner:
    """Prepares inputs and calls model.forward for one scheduler step.

    Args:
        model:         LlamaForCausalLM (or any model with __call__ + sample).
        kv_cache:      Shared PagedKVCache instance.
        block_manager: Same BlockManager used by the Scheduler.
        block_size:    Tokens per KV cache block.
    """

    def __init__(
        self,
        model,
        kv_cache: PagedKVCache,
        block_manager: BlockManager,
        block_size: int = 16,
    ) -> None:
        self.model = model
        self.kv_cache = kv_cache
        self.block_manager = block_manager
        self.block_size = block_size

    # ----------------------------------------------------------------
    # Main entry point
    # ----------------------------------------------------------------

    def run(
        self,
        output: SchedulerOutput,
        *,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        key: Optional[jax.Array] = None,
    ) -> dict[int, int]:
        """Execute one step; return {seq_id: next_token_id} for each sequence.

        Returns an empty dict if SchedulerOutput is empty.
        """
        if output.is_empty:
            return {}

        results: dict[int, int] = {}

        # ---- Prefill ----
        if output.prefill_seqs:
            token_map, input_ids, positions, bi, bo, seq_lens, last_indices = \
                self._build_prefill_inputs(output.prefill_seqs)
            logits = self.model(
                input_ids, positions, self.kv_cache,
                bi, bo, seq_lens,
                is_prefill=True,
                last_indices=last_indices,
            )
            jax.effects_barrier()  # ensure computation completes
            tokens = self.model.sample(
                logits, key=key, temperature=temperature,
                top_k=top_k, top_p=top_p,
            )
            for i, seq in enumerate(output.prefill_seqs):
                results[seq.seq_id] = int(tokens[i])

        # ---- Decode ----
        if output.decode_seqs:
            token_map, input_ids, positions, bi, bo, seq_lens, block_table = \
                self._build_decode_inputs(output.decode_seqs)
            logits = self.model(
                input_ids, positions, self.kv_cache,
                bi, bo, seq_lens,
                is_prefill=False,
                block_table=block_table,
            )
            jax.effects_barrier()
            tokens = self.model.sample(
                logits, key=key, temperature=temperature,
                top_k=top_k, top_p=top_p,
            )
            for i, seq in enumerate(output.decode_seqs):
                results[seq.seq_id] = int(tokens[i])

        return results

    # ----------------------------------------------------------------
    # Input builders
    # ----------------------------------------------------------------

    def _build_prefill_inputs(self, seqs: list[Sequence]):
        """Build packed prefill inputs for a list of sequences."""
        all_token_ids = []
        all_positions = []
        all_bi = []
        all_bo = []
        seq_lens = []
        last_indices = []
        offset = 0

        for seq in seqs:
            tokens = seq.all_token_ids
            T = len(tokens)
            all_token_ids.extend(tokens)
            all_positions.extend(range(T))
            for pos in range(T):
                block_num, block_off = self.block_manager.get_position_for_token(seq, pos)
                all_bi.append(block_num)
                all_bo.append(block_off)
            seq_lens.append(T)
            last_indices.append(offset + T - 1)
            offset += T

        input_ids   = jnp.array(all_token_ids, dtype=jnp.int32)
        positions   = jnp.array(all_positions, dtype=jnp.int32)
        bi          = jnp.array(all_bi,        dtype=jnp.int32)
        bo          = jnp.array(all_bo,        dtype=jnp.int32)
        seq_lens_a  = jnp.array(seq_lens,      dtype=jnp.int32)
        last_idx_a  = jnp.array(last_indices,  dtype=jnp.int32)
        return None, input_ids, positions, bi, bo, seq_lens_a, last_idx_a

    def _build_decode_inputs(self, seqs: list[Sequence]):
        """Build single-token decode inputs for a list of sequences."""
        token_ids = []
        positions = []
        bi_list = []
        bo_list = []
        seq_lens = []
        max_blocks = max(len(s.block_table) for s in seqs)
        block_table_rows = []

        for seq in seqs:
            # The new decode token goes at position total_len - 1
            # (append_slot already extended the block_table if needed)
            pos = seq.total_len - 1
            block_num, block_off = self.block_manager.get_position_for_token(seq, pos)
            token_ids.append(seq.last_token_id)
            positions.append(pos)
            bi_list.append(block_num)
            bo_list.append(block_off)
            seq_lens.append(seq.total_len)
            bt = self.block_manager.get_block_table(seq, max_blocks)
            block_table_rows.append(bt)

        input_ids   = jnp.array(token_ids,        dtype=jnp.int32)
        pos_arr     = jnp.array(positions,        dtype=jnp.int32)
        bi          = jnp.array(bi_list,          dtype=jnp.int32)
        bo          = jnp.array(bo_list,          dtype=jnp.int32)
        seq_lens_a  = jnp.array(seq_lens,         dtype=jnp.int32)
        bt_arr      = jnp.array(block_table_rows, dtype=jnp.int32)
        return None, input_ids, pos_arr, bi, bo, seq_lens_a, bt_arr
