"""Unit tests for ModelRunner input builders (Phase 6).

We test _build_prefill_inputs and _build_decode_inputs directly
without running the actual model (avoids GPU requirement in CI).
"""
import pytest
import numpy as np
import jax.numpy as jnp

from nanovllm_jax.engine.block_manager import BlockManager
from nanovllm_jax.engine.model_runner import ModelRunner
from nanovllm_jax.engine.sequence import Sequence, SequenceStatus
from nanovllm_jax.layers.attention import PagedKVCache
from nanovllm_jax.sampling_params import SamplingParams


def make_seq(seq_id, prompt):
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(max_tokens=32),
    )


def setup_runner(block_size=8, num_blocks=16, num_layers=2, num_kv_heads=2, head_dim=16):
    bm = BlockManager(num_blocks=num_blocks, block_size=block_size)
    cache = PagedKVCache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=jnp.float32,
    )
    runner = ModelRunner(model=None, kv_cache=cache, block_manager=bm, block_size=block_size)
    return runner, bm


# ---------------------------------------------------------------------------
# Prefill input builder
# ---------------------------------------------------------------------------

def test_prefill_input_shapes_single_seq():
    runner, bm = setup_runner(block_size=8)
    seq = make_seq(0, prompt=[1, 2, 3, 4])
    bm.allocate(seq)
    seq.status = SequenceStatus.RUNNING
    _, ids, pos, bi, bo, sl, last = runner._build_prefill_inputs([seq])
    T = 4
    assert ids.shape == (T,)
    assert pos.shape == (T,)
    assert bi.shape == (T,)
    assert bo.shape == (T,)
    assert sl.shape == (1,)
    assert last.shape == (1,)
    assert int(last[0]) == T - 1


def test_prefill_positions_correct():
    runner, bm = setup_runner()
    seq = make_seq(0, prompt=[10, 20, 30])
    bm.allocate(seq)
    _, ids, pos, bi, bo, sl, last = runner._build_prefill_inputs([seq])
    np.testing.assert_array_equal(np.array(pos), [0, 1, 2])
    np.testing.assert_array_equal(np.array(ids), [10, 20, 30])


def test_prefill_packed_multi_seq():
    runner, bm = setup_runner()
    seq0 = make_seq(0, prompt=[1, 2])
    seq1 = make_seq(1, prompt=[3, 4, 5])
    bm.allocate(seq0)
    bm.allocate(seq1)
    _, ids, pos, bi, bo, sl, last = runner._build_prefill_inputs([seq0, seq1])
    assert ids.shape == (5,)
    assert sl.shape == (2,)
    assert int(sl[0]) == 2
    assert int(sl[1]) == 3
    # last_indices: 1 and 4
    assert int(last[0]) == 1
    assert int(last[1]) == 4


# ---------------------------------------------------------------------------
# Decode input builder
# ---------------------------------------------------------------------------

def test_decode_input_shapes():
    runner, bm = setup_runner(block_size=8)
    seq = make_seq(0, prompt=[1, 2, 3, 4])
    bm.allocate(seq)
    seq.output_token_ids.append(5)  # simulate 1 decode step already done
    seq.status = SequenceStatus.RUNNING
    bm.append_slot(seq)  # ensure slot exists
    _, ids, pos, bi, bo, sl, bt = runner._build_decode_inputs([seq])
    assert ids.shape == (1,)
    assert pos.shape == (1,)
    assert sl.shape == (1,)
    assert bt.shape[0] == 1  # 1 seq


def test_decode_position_is_total_len_minus_one():
    runner, bm = setup_runner(block_size=8)
    seq = make_seq(0, prompt=[1, 2, 3])
    bm.allocate(seq)
    seq.output_token_ids.append(42)
    bm.append_slot(seq)
    _, ids, pos, bi, bo, sl, bt = runner._build_decode_inputs([seq])
    # total_len = 3 + 1 = 4, position = 3
    assert int(pos[0]) == 3
    assert int(ids[0]) == 42  # last_token_id


def test_decode_block_table_padded():
    runner, bm = setup_runner(block_size=4)
    seq0 = make_seq(0, prompt=[1, 2, 3, 4])    # 1 block
    seq1 = make_seq(1, prompt=[1, 2, 3, 4, 5, 6, 7, 8])  # 2 blocks
    bm.allocate(seq0)
    bm.allocate(seq1)
    seq0.output_token_ids.append(10)
    seq1.output_token_ids.append(20)
    bm.append_slot(seq0)
    bm.append_slot(seq1)
    _, ids, pos, bi, bo, sl, bt = runner._build_decode_inputs([seq0, seq1])
    # max_blocks = 3 (seq1 may need 3 blocks after append)
    assert bt.shape[0] == 2
    assert bt.shape[1] >= 2  # at least 2 columns
