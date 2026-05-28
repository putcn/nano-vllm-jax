"""Unit tests for BlockAllocator and BlockManager (Phase 6)."""
import pytest
from nanovllm_jax.engine.block_manager import BlockAllocator, BlockManager
from nanovllm_jax.engine.sequence import Sequence, SequenceStatus
from nanovllm_jax.sampling_params import SamplingParams


def make_seq(prompt_len=5, seq_id=0):
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=32),
    )


# ---------------------------------------------------------------------------
# BlockAllocator
# ---------------------------------------------------------------------------

def test_allocator_initial_free():
    a = BlockAllocator(8)
    assert a.num_free == 8
    assert a.num_used == 0


def test_allocator_alloc_free():
    a = BlockAllocator(4)
    b0 = a.allocate()
    b1 = a.allocate()
    assert a.num_free == 2
    a.free(b0)
    assert a.num_free == 3


def test_allocator_exhaustion():
    a = BlockAllocator(2)
    a.allocate()
    a.allocate()
    with pytest.raises(RuntimeError, match="Out of KV cache blocks"):
        a.allocate()


# ---------------------------------------------------------------------------
# BlockManager
# ---------------------------------------------------------------------------

def test_manager_allocate_and_free():
    bm = BlockManager(num_blocks=16, block_size=4)
    seq = make_seq(prompt_len=5)  # needs ceil(5/4)=2 blocks
    assert bm.can_allocate(seq)
    bm.allocate(seq)
    assert len(seq.block_table) == 2
    assert bm.num_free_blocks == 14
    bm.free(seq)
    assert bm.num_free_blocks == 16
    assert seq.block_table == []


def test_manager_append_slot_new_block():
    bm = BlockManager(num_blocks=16, block_size=4)
    seq = make_seq(prompt_len=4)  # exactly 1 block full
    bm.allocate(seq)
    assert len(seq.block_table) == 1
    # Simulate adding one output token (total_len now = 5)
    seq.output_token_ids.append(99)
    assert bm.can_append(seq)
    bm.append_slot(seq)
    assert len(seq.block_table) == 2


def test_manager_no_free_blocks():
    bm = BlockManager(num_blocks=2, block_size=4)
    seq1 = make_seq(prompt_len=4, seq_id=0)
    seq2 = make_seq(prompt_len=4, seq_id=1)
    bm.allocate(seq1)
    bm.allocate(seq2)
    assert bm.num_free_blocks == 0
    seq3 = make_seq(prompt_len=4, seq_id=2)
    assert not bm.can_allocate(seq3)


def test_get_block_table_padding():
    bm = BlockManager(num_blocks=8, block_size=4)
    seq = make_seq(prompt_len=5)
    bm.allocate(seq)
    bt = bm.get_block_table(seq, max_blocks=4)
    assert len(bt) == 4
    assert bt[2] == 0 and bt[3] == 0  # padded


def test_get_position_for_token():
    bm = BlockManager(num_blocks=8, block_size=4)
    seq = make_seq(prompt_len=8)  # 2 blocks
    bm.allocate(seq)
    block_id_0, off_0 = bm.get_position_for_token(seq, 0)
    block_id_4, off_4 = bm.get_position_for_token(seq, 4)
    assert block_id_0 == seq.block_table[0] and off_0 == 0
    assert block_id_4 == seq.block_table[1] and off_4 == 0
    _, off_5 = bm.get_position_for_token(seq, 5)
    assert off_5 == 1
