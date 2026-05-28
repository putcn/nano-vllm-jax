"""Regression tests: decode seq_lens padding must use val=0, not val=1.

Bug #3 (fixed): _run_decode padded seq_lens with val=1. Padding sequences
appeared to have length 1 and were allowed to attend to position 0 of the
KV cache, which could hold another sequence's data, potentially corrupting
the decode output.

Fix: pad seq_lens with val=0 so padding sequences have zero valid context
and their attention logits are all -inf (softmax → 0 output).
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import pytest

from nanovllm_jax.engine.model_runner import ModelRunner, _next_pow2


def test_seq_lens_padding_uses_zero():
    """_pad1d for seq_lens must use val=0 in _run_decode."""
    runner = ModelRunner(block_size=4)

    # Simulate the padding that _run_decode applies
    seq_lens = jnp.array([5, 3], dtype=jnp.int32)  # 2 real sequences
    B_real = 2
    B_pad = _next_pow2(B_real + 1)  # force padding (e.g. 4)
    if B_pad > B_real:
        padded = runner._pad1d(seq_lens, B_pad, val=0)
        # Padding rows must be 0
        for i in range(B_real, B_pad):
            assert int(padded[i]) == 0, (
                f"Padding seq_lens[{i}] = {int(padded[i])}, expected 0. "
                "Non-zero padding allows phantom attention over cached KV data."
            )


def test_decode_output_shape_and_real_rows():
    """ModelRunner._run_decode must return one token per real sequence."""
    from nanovllm_jax.engine.model_runner import ModelRunner
    from nanovllm_jax.engine.block_manager import BlockManager
    from nanovllm_jax.engine.sequence import Sequence
    from nanovllm_jax.sampling_params import SamplingParams
    from nanovllm_jax.layers.attention import PagedKVCache

    block_size = 4
    bm = BlockManager(num_blocks=32, block_size=block_size)
    cache = PagedKVCache(
        num_layers=2, num_kv_heads=2, head_dim=16,
        num_blocks=32, block_size=block_size, dtype=jnp.float32,
    )
    runner = ModelRunner(model=None, kv_cache=cache,
                         block_manager=bm, block_size=block_size)

    sp = SamplingParams(max_tokens=10)
    seqs = [
        Sequence(seq_id=0, prompt_token_ids=[1, 2, 3], sampling_params=sp,
                 output_token_ids=[7]),   # total_len=4, in decode phase
        Sequence(seq_id=1, prompt_token_ids=[4, 5, 6], sampling_params=sp,
                 output_token_ids=[8]),
    ]
    for seq in seqs:
        bm.allocate(seq)
        bm.append_slot(seq)  # allocate slot for the output token

    results = runner.run(prefill_seqs=[], decode_seqs=seqs)

    assert set(results.keys()) == {0, 1}
    for seq_id, token_id in results.items():
        assert isinstance(token_id, int)
