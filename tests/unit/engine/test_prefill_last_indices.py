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

# ── version sentinel — bump this to verify the file is actually reloaded ──
_TEST_VERSION = "v7-debug"


def test_lm_head_last_indices_selects_correct_token():
    # --- diagnostics: print which files are actually being used ---
    import nanovllm_jax.layers.embed_head as _eh
    print(f"\n[DEBUG] test file     : {__file__}")
    print(f"[DEBUG] test version  : {_TEST_VERSION}")
    print(f"[DEBUG] embed_head.py : {inspect.getfile(_eh)}")
    print(f"[DEBUG] Python        : {sys.executable}")
    print(f"[DEBUG] cwd           : {os.getcwd()}")

    vocab_size = 16
    hidden_size = 8
    num_seqs = 3
    T_pad = 16

    head = ParallelLMHead(vocab_size, hidden_size)
    w = jax.random.normal(jax.random.PRNGKey(0), (vocab_size, hidden_size))
    head.load_weight(w)

    hidden = jax.random.normal(jax.random.PRNGKey(1), (T_pad, hidden_size))
    last_indices = jnp.array([2, 5, 11], dtype=jnp.int32)

    logits_with = head(hidden, last_indices=last_indices)
    print(f"[DEBUG] logits_with.shape : {logits_with.shape}")
    assert logits_with.shape == (num_seqs, vocab_size), (
        f"Expected ({num_seqs}, {vocab_size}), got {logits_with.shape}"
    )

    # --- inspect what head.__call__ is actually doing ---
    print(f"[DEBUG] head.__call__ source:\n{inspect.getsource(head.__call__)}")
    print(f"[DEBUG] effective_weight source:\n{inspect.getsource(type(head).effective_weight.fget)}")

    ew = head.effective_weight
    print(f"[DEBUG] effective_weight shape : {ew.shape}")
    print(f"[DEBUG] effective_weight[0,:4] : {ew[0, :4]}")
    print(f"[DEBUG] w[0,:4]                : {w[0, :4]}")
    print(f"[DEBUG] w == ew allclose       : {jnp.allclose(w, ew, atol=1e-7)}")

    # reference via single-row call
    for i, idx in enumerate([2, 5, 11]):
        single_row = hidden[idx:idx + 1]
        expected_single = head(single_row)[0]
        manual = hidden[idx] @ ew.T
        print(f"[DEBUG] row {i} (hidden[{idx}]):")
        print(f"  logits_with[{i}][:4]    = {logits_with[i][:4]}")
        print(f"  expected_single[:4]     = {expected_single[:4]}")
        print(f"  manual (hidden@ew.T)[:4]= {manual[:4]}")
        print(f"  logits_with vs single   allclose={jnp.allclose(logits_with[i], expected_single, atol=1e-5)}")
        print(f"  logits_with vs manual   allclose={jnp.allclose(logits_with[i], manual, atol=1e-5)}")

    # actual assertion using single-row path
    for i, idx in enumerate([2, 5, 11]):
        single_row = hidden[idx:idx + 1]
        expected = head(single_row)[0]
        assert jnp.allclose(logits_with[i], expected, atol=1e-5), (
            f"[{_TEST_VERSION}] Logit row {i}: logits_with[{i}] != head(hidden[{idx}:{idx+1}])[0]"
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
