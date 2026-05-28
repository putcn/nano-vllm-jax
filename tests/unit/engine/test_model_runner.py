"""Unit tests for ModelRunner.

Covers:
- _next_pow2 helper
- _build_prefill_inputs / _build_decode_inputs produce correct arrays
- _run_prefill and _run_decode with stub model return seq_id -> token_id dicts
- padding does not cause index errors
"""
import pytest
import jax.numpy as jnp
import numpy as np

from nanovllm_jax.engine.model_runner import ModelRunner, _next_pow2
from nanovllm_jax.engine.sequence import Sequence
from nanovllm_jax.layers.attention import PagedKVCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_seq(seq_id: int, token_ids: list, block_table: list) -> Sequence:
    seq = Sequence(seq_id=seq_id, token_ids=token_ids)
    seq.block_table = block_table
    return seq


def _make_runner_with_stub(num_blocks: int = 32, block_size: int = 4) -> ModelRunner:
    cache = PagedKVCache(
        num_layers=2, num_kv_heads=2, head_dim=8,
        num_blocks=num_blocks, block_size=block_size,
        dtype=jnp.float32,
    )
    return ModelRunner(
        model=None,   # None -> _StubModel
        kv_cache=cache,
        block_size=block_size,
    )


# ---------------------------------------------------------------------------
# _next_pow2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [
    (1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (8, 8), (9, 16), (128, 128),
])
def test_next_pow2(n, expected):
    assert _next_pow2(n) == expected


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------

class TestBuildPrefillInputs:
    def test_single_seq(self):
        runner = _make_runner_with_stub()
        seq = _make_seq(42, [10, 20, 30], block_table=[0])
        _, ids, pos, bi, bo, seq_lens, last_idx = runner._build_prefill_inputs([seq])

        assert ids.tolist() == [10, 20, 30]
        assert pos.tolist() == [0, 1, 2]
        assert seq_lens.tolist() == [3]
        assert last_idx.tolist() == [2]

    def test_multi_seq_offsets(self):
        runner = _make_runner_with_stub()
        s0 = _make_seq(0, [1, 2],    block_table=[0])
        s1 = _make_seq(1, [3, 4, 5], block_table=[1])
        _, ids, pos, bi, bo, seq_lens, last_idx = runner._build_prefill_inputs([s0, s1])

        assert ids.tolist() == [1, 2, 3, 4, 5]
        assert last_idx.tolist() == [1, 4]   # end of each seq
        assert seq_lens.tolist() == [2, 3]


class TestBuildDecodeInputs:
    def test_single_seq(self):
        runner = _make_runner_with_stub(block_size=4)
        # seq has 5 tokens -> step=4 -> block 1, offset 0
        seq = _make_seq(7, list(range(5)), block_table=[0, 1])
        seq._last_token_id = 99  # set via attribute for test
        # Patch last_token_id property
        seq.last_token_id = 99
        _, ids, pos, bi, bo, slens, block_table = runner._build_decode_inputs([seq])

        assert ids.shape == (1,)
        assert pos.tolist() == [4]      # step = total_len - 1
        assert slens.tolist() == [5]
        assert block_table.shape == (1, 2)


# ---------------------------------------------------------------------------
# run() with stub model
# ---------------------------------------------------------------------------

class TestRunWithStub:
    def test_prefill_returns_eos(self):
        runner = _make_runner_with_stub()
        seq = _make_seq(1, [10, 20], block_table=[0])
        results = runner.run([seq], [])
        assert 1 in results
        # Stub returns eos_token_id=2 for all
        assert results[1] == 2

    def test_decode_returns_eos(self):
        runner = _make_runner_with_stub()
        seq = _make_seq(2, [10, 20, 30], block_table=[0])
        seq.last_token_id = 30
        results = runner.run([], [seq])
        assert 2 in results
        assert results[2] == 2

    def test_multi_seq_all_ids_returned(self):
        runner = _make_runner_with_stub()
        seqs = [_make_seq(i, [i * 10], block_table=[i]) for i in range(3)]
        results = runner.run(seqs, [])
        assert set(results.keys()) == {0, 1, 2}
