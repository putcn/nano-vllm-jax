"""Unit tests for LLMEngine and LLM public API (Phase 7).

All tests use the stub model (no real checkpoint required).
"""
import pytest
from nanovllm_jax.config import EngineConfig
from nanovllm_jax.engine.llm_engine import LLMEngine, RequestOutput
from nanovllm_jax.llm import LLM
from nanovllm_jax.sampling_params import SamplingParams


def make_engine(max_tokens_per_seq=8, num_blocks=32, block_size=8, max_num_seqs=4):
    cfg = EngineConfig(
        model="",          # empty => stub model
        num_gpu_blocks=num_blocks,
        block_size=block_size,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=512,
        max_model_len=128,
    )
    return LLMEngine(cfg)


# ---------------------------------------------------------------------------
# LLMEngine tests
# ---------------------------------------------------------------------------

def test_engine_add_and_run_to_completion():
    engine = make_engine()
    sp = SamplingParams(max_tokens=4)
    rid = engine.add_request("hello", sp)
    outputs = engine.generate_all()
    assert rid in outputs
    out = outputs[rid]
    assert isinstance(out, RequestOutput)
    assert len(out.token_ids) == 4


def test_engine_multiple_requests():
    engine = make_engine()
    sp = SamplingParams(max_tokens=3)
    ids = [engine.add_request(f"prompt {i}", sp) for i in range(3)]
    engine.generate_all()
    for rid in ids:
        out = engine.get_outputs(rid)
        assert out is not None
        assert len(out.token_ids) == 3


def test_engine_has_unfinished():
    engine = make_engine()
    assert not engine.has_unfinished
    engine.add_request("hi", SamplingParams(max_tokens=2))
    assert engine.has_unfinished
    engine.generate_all()
    assert not engine.has_unfinished


def test_engine_stop_token_id():
    """Stub always emits token 1; if stop_token_ids=[1] seq stops after 1 token."""
    engine = make_engine()
    sp = SamplingParams(max_tokens=16, stop_token_ids=[1])
    rid = engine.add_request("test", sp)
    engine.generate_all()
    out = engine.get_outputs(rid)
    # Should stop after first token (token_id=1 triggers stop)
    assert len(out.token_ids) == 1
    assert out.token_ids[0] == 1


def test_engine_with_prompt_token_ids():
    engine = make_engine()
    sp = SamplingParams(max_tokens=2)
    rid = engine.add_request("", sp, prompt_token_ids=[10, 20, 30])
    engine.generate_all()
    out = engine.get_outputs(rid)
    assert out is not None
    assert len(out.token_ids) == 2


# ---------------------------------------------------------------------------
# LLM public API tests
# ---------------------------------------------------------------------------

def test_llm_generate_single():
    llm = LLM(model="", num_gpu_blocks=32, block_size=8,
              max_num_seqs=4, max_num_batched_tokens=512, max_model_len=128)
    outputs = llm.generate("Hello", SamplingParams(max_tokens=3))
    assert len(outputs) == 1
    assert len(outputs[0].token_ids) == 3


def test_llm_generate_batch():
    llm = LLM(model="", num_gpu_blocks=32, block_size=8,
              max_num_seqs=4, max_num_batched_tokens=512, max_model_len=128)
    prompts = ["A", "B", "C"]
    outputs = llm.generate(prompts, SamplingParams(max_tokens=2))
    assert len(outputs) == 3
    for o in outputs:
        assert len(o.token_ids) == 2


def test_llm_default_sampling_params():
    """generate() without explicit SamplingParams uses greedy defaults."""
    llm = LLM(model="", num_gpu_blocks=32, block_size=8,
              max_num_seqs=4, max_num_batched_tokens=512, max_model_len=128)
    outputs = llm.generate(["test"])
    assert outputs[0] is not None


def test_llm_generate_preserves_order():
    """Output list must be in same order as input prompts."""
    llm = LLM(model="", num_gpu_blocks=32, block_size=8,
              max_num_seqs=4, max_num_batched_tokens=512, max_model_len=128)
    prompts = [f"prompt_{i}" for i in range(5)]
    outputs = llm.generate(prompts, SamplingParams(max_tokens=2))
    assert len(outputs) == 5
    for i, o in enumerate(outputs):
        assert o.request_id == str(i)
