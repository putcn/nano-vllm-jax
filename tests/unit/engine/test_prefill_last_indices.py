"""Regression tests: prefill must select the correct last-token logit.

Bug #2 (fixed): _jit_prefill never passed last_indices to the model, so
lm_head returned logits for all T_pad tokens. This made multi-sequence
batches read logits from the wrong token positions.

Fix: pass last_indices from _run_prefill → _jit_prefill → model.__call__
so ParallelLMHead slices hidden states before projecting, returning
shape [num_seqs, vocab_size].
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from flax import nnx
import pytest

from nanovllm_jax.layers.embed_head import ParallelLMHead, VocabParallelEmbedding


def test_lm_head_last_indices_selects_correct_token():
    """ParallelLMHead with last_indices must return logits for the correct rows."""
    vocab_size = 16
    hidden_size = 8
    num_seqs = 3
    T_pad = 16  # padded token sequence (much larger than real)

    head = ParallelLMHead(vocab_size, hidden_size)
    # Load identity-like weight so output = input projection
    w = jax.random.normal(jax.random.PRNGKey(0), (vocab_size, hidden_size))
    head.load_weight(w)

    # Hidden states: each row is distinct
    hidden = jax.random.normal(jax.random.PRNGKey(1), (T_pad, hidden_size))

    # last token of each sequence is at positions 2, 5, 11
    last_indices = jnp.array([2, 5, 11], dtype=jnp.int32)

    logits_with = head(hidden, last_indices=last_indices)
    assert logits_with.shape == (num_seqs, vocab_size), (
        f"Expected ({num_seqs}, {vocab_size}), got {logits_with.shape}"
    )

    # Each row must match manually extracted hidden[last_idx] @ w.T
    for i, idx in enumerate([2, 5, 11]):
        expected = hidden[idx] @ w.T
        assert jnp.allclose(logits_with[i], expected, atol=1e-5), (
            f"Logit row {i} does not match hidden[{idx}] @ w.T"
        )


def test_lm_head_without_last_indices_returns_all_rows():
    """ParallelLMHead without last_indices returns logits for every row."""
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
    """ModelRunner._run_prefill must return exactly one next-token per sequence."""
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

    assert set(results.keys()) == {0, 1, 2}, "Must return one token per sequence id"
    for seq_id, token_id in results.items():
        assert isinstance(token_id, int), f"Token for seq {seq_id} must be int"
