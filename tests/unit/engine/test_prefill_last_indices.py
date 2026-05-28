"""Regression tests: prefill must select the correct last-token logit."""
from __future__ import annotations
import inspect
import os
import sys
import jax
import jax.numpy as jnp
from flax import nnx
import pytest

from nanovllm_jax.layers.embed_head import ParallelLMHead, VocabParallelEmbedding

_TEST_VERSION = "v8-debug"


def test_lm_head_last_indices_selects_correct_token():
    import nanovllm_jax.layers.embed_head as _eh
    print(f"\n[DEBUG] test version  : {_TEST_VERSION}")
    print(f"[DEBUG] embed_head.py : {inspect.getfile(_eh)}")

    vocab_size = 16
    hidden_size = 8
    num_seqs = 3
    T_pad = 16

    head = ParallelLMHead(vocab_size, hidden_size)
    w = jax.random.normal(jax.random.PRNGKey(0), (vocab_size, hidden_size))
    head.load_weight(w)

    hidden = jax.random.normal(jax.random.PRNGKey(1), (T_pad, hidden_size))
    last_indices = jnp.array([2, 5, 11], dtype=jnp.int32)

    ew = head.effective_weight

    # Compute all T_pad rows projected through the same weight
    all_logits = hidden @ ew.T   # (T_pad, vocab_size)

    logits_with = head(hidden, last_indices=last_indices)
    print(f"[DEBUG] logits_with.shape : {logits_with.shape}")
    assert logits_with.shape == (num_seqs, vocab_size)

    # Find which row of hidden each logits_with[i] actually came from
    for i in range(num_seqs):
        diffs = jnp.max(jnp.abs(all_logits - logits_with[i]), axis=1)  # (T_pad,)
        best_match = int(jnp.argmin(diffs))
        best_diff = float(jnp.min(diffs))
        print(f"[DEBUG] logits_with[{i}] best matches hidden row {best_match} (max_diff={best_diff:.6f}), expected {[2,5,11][i]}")

    # Now the real assertion
    for i, idx in enumerate([2, 5, 11]):
        expected = all_logits[idx]  # hidden[idx] @ ew.T, precomputed
        diff = float(jnp.max(jnp.abs(logits_with[i] - expected)))
        print(f"[DEBUG] logits_with[{i}] vs all_logits[{idx}] max_diff={diff:.8f}")
        assert jnp.allclose(logits_with[i], expected, atol=1e-4), (
            f"[{_TEST_VERSION}] row {i}: logits_with[{i}] != hidden[{idx}] @ ew.T (max_diff={diff:.6f})"
        )


def test_lm_head_without_last_indices_returns_all_rows():
    vocab_size = 16
    hidden_size = 8
    T = 7
    head = ParallelLMHead(vocab_size, hidden_size)
    w = jax.random.normal(jax.random.PRNGKey(0), (vocab_size, hidden_size))
    head.load_weight(w)
    hidden = jax.random.normal(jax.random.PRNGKey(1), (T, hidden_size))
    logits = head(hidden, last_indices=None)
    assert logits.shape == (T, vocab_size)


def test_run_prefill_returns_one_token_per_sequence():
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
    sp = SamplingParams(max_tokens=1)
    seqs = [
        Sequence(seq_id=0, prompt_token_ids=[1, 2, 3], sampling_params=sp),
        Sequence(seq_id=1, prompt_token_ids=[4, 5], sampling_params=sp),
        Sequence(seq_id=2, prompt_token_ids=[6, 7, 8, 9], sampling_params=sp),
    ]
    for seq in seqs:
        bm.allocate(seq)
    results = runner.run(prefill_seqs=seqs, decode_seqs=[])
    assert set(results.keys()) == {0, 1, 2}
    for seq_id, token_id in results.items():
        assert isinstance(token_id, int)
