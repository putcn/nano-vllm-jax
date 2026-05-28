"""Unit tests for LLMEngine and LLM public API (Phase 7).

All tests use the stub model (no real checkpoint required).
Stub always emits eos_token_id=2, so sequences stop after the first
generated token (or after max_tokens if max_tokens=1).
"""
import pytest
from nanovllm_jax.config import EngineConfig
from nanovllm_jax.engine.llm_engine import LLMEngine, RequestOutput
from nanovllm_jax.llm import LLM
from nanovllm_jax.sampling_params import SamplingParams

# Stub always emits token 2 = EOS; engine.eos_token_id=2 by default.
_EOS = 2


def make_engine(num_blocks=32, block_size=8, max_num_seqs=4):
    cfg = EngineConfig(
        model="",
        num_gpu_blocks=num_blocks,
        block_size=block_size,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=512,
        max_model_len=128,
        eos_token_id=_EOS,
    )
    return LLMEngine(cfg)


def make_llm(**kwargs):
    defaults = dict(model="", num_gpu_blocks=32, block_size=8,
                    max_num_seqs=4, max_num_batched_tokens=512,
                    max_model_len=128, eos_token_id=_EOS)
    defaults.update(kwargs)
    return LLM(**defaults)


# ---------------------------------------------------------------------------
# LLMEngine tests
# ---------------------------------------------------------------------------

def test_engine_eos_stops_generation():
    """Stub emits EOS immediately; sequence should finish after 1 token."""
    engine = make_engine()
    sp = SamplingParams(max_tokens=64)   # high limit; EOS should fire first
    rid = engine.add_request("hello", sp)
    engine.generate_all()
    out = engine.get_outputs(rid)
    assert out is not None
    assert out.token_ids == [_EOS]


def test_engine_max_tokens_respected():
    """ignore_eos=True prevents EOS stop; max_tokens limits output."""
    engine = make_engine()
    sp = SamplingParams(max_tokens=4, ignore_eos=True)
    rid = engine.add_request("hello", sp)
    engine.generate_all()
    out = engine.get_outputs(rid)
    assert len(out.token_ids) == 4


def test_engine_multiple_requests():
    engine = make_engine()
    sp = SamplingParams(max_tokens=8)   # EOS fires first
    ids = [engine.add_request(f"prompt {i}", sp) for i in range(3)]
    engine.generate_all()
    for rid in ids:
        out = engine.get_outputs(rid)
        assert out is not None
        assert out.token_ids == [_EOS]


def test_engine_has_unfinished():
    engine = make_engine()
    assert not engine.has_unfinished
    engine.add_request("hi", SamplingParams(max_tokens=8))
    assert engine.has_unfinished
    engine.generate_all()
    assert not engine.has_unfinished


def test_engine_stop_token_ids():
    """stop_token_ids=[2] should also stop when stub emits 2."""
    engine = make_engine()
    sp = SamplingParams(max_tokens=16, stop_token_ids=[_EOS])
    rid = engine.add_request("test", sp)
    engine.generate_all()
    out = engine.get_outputs(rid)
    assert len(out.token_ids) == 1
    assert out.token_ids[0] == _EOS


def test_engine_with_prompt_token_ids():
    engine = make_engine()
    sp = SamplingParams(max_tokens=8)
    rid = engine.add_request("", sp, prompt_token_ids=[10, 20, 30])
    engine.generate_all()
    out = engine.get_outputs(rid)
    assert out is not None
    assert out.token_ids == [_EOS]


# ---------------------------------------------------------------------------
# LLM public API tests
# ---------------------------------------------------------------------------

def test_llm_generate_single():
    llm = make_llm()
    outputs = llm.generate("Hello", SamplingParams(max_tokens=8))
    assert len(outputs) == 1
    assert outputs[0].token_ids == [_EOS]


def test_llm_generate_batch():
    llm = make_llm()
    outputs = llm.generate(["A", "B", "C"], SamplingParams(max_tokens=8))
    assert len(outputs) == 3
    for o in outputs:
        assert o.token_ids == [_EOS]


def test_llm_default_sampling_params():
    """generate() without SamplingParams uses defaults; EOS fires before max_tokens=256."""
    llm = make_llm()
    outputs = llm.generate(["test"])   # no explicit SamplingParams
    assert outputs[0] is not None
    # Stub emits EOS on first token regardless of max_tokens
    assert outputs[0].token_ids == [_EOS]


def test_llm_generate_preserves_order():
    llm = make_llm()
    prompts = [f"prompt_{i}" for i in range(5)]
    outputs = llm.generate(prompts, SamplingParams(max_tokens=8))
    assert len(outputs) == 5
    for i, o in enumerate(outputs):
        assert o.request_id == str(i)
