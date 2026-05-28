"""End-to-end generation test with Qwen3-0.6B (or any HF checkpoint).

This test requires a real model checkpoint and is **skipped automatically**
when the checkpoint is not present, so it never blocks CI.

Usage
-----
# Download model once (HuggingFace CLI)
hf_download Qwen/Qwen3-0.6B

# Run e2e test only
pytest tests/e2e/test_qwen3_generation.py -v -s

# Override model path
MODEL_PATH=/data/models/Qwen3-0.6B pytest tests/e2e/ -v -s

Environment variables
---------------------
MODEL_PATH   : path to local HF checkpoint  (default: ~/models/Qwen3-0.6B)
MAX_TOKENS   : max tokens to generate       (default: 64)
NUM_BLOCKS   : number of KV cache blocks    (default: 512)
"""
from __future__ import annotations
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = Path.home() / "models" / "Qwen3-0.6B"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", _DEFAULT_MODEL))
MAX_TOKENS  = int(os.environ.get("MAX_TOKENS", 64))
NUM_BLOCKS  = int(os.environ.get("NUM_BLOCKS", 512))
BLOCK_SIZE  = 16

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH / "config.json").exists(),
    reason=f"Model checkpoint not found at {MODEL_PATH}. "
           f"Set MODEL_PATH env var or download with: "
           f"huggingface-cli download Qwen/Qwen3-0.6B --local-dir {MODEL_PATH}",
)


# ---------------------------------------------------------------------------
# Shared LLM fixture (load once per session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def llm():
    from nanovllm_jax.llm import LLM
    print(f"\nLoading model from {MODEL_PATH} ...")
    model = LLM(
        model=str(MODEL_PATH),
        num_gpu_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        max_model_len=2048,
        eos_token_id=151645,   # Qwen3 <|im_end|> token
    )
    print("Model loaded.")
    return model


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    return tok


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=True)


def decode(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_simple_completion(llm, tokenizer):
    """Model should produce a non-empty response."""
    from nanovllm_jax.sampling_params import SamplingParams

    prompt = "The capital of France is"
    token_ids = encode(tokenizer, prompt)

    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS),
        prompt_token_ids=[token_ids],
    )
    text = decode(tokenizer, outputs[0].token_ids)
    print(f"\n[simple_completion]\n  prompt : {prompt!r}\n  output : {text!r}")

    assert len(outputs[0].token_ids) > 0
    assert isinstance(text, str)
    # Greedy should produce a deterministic, sensible continuation
    assert len(text.strip()) > 0


def test_batch_generation(llm, tokenizer):
    """Batch of prompts should return one output per prompt in order."""
    from nanovllm_jax.sampling_params import SamplingParams

    prompts = [
        "What is JAX?",
        "Write a Python function to compute fibonacci numbers.",
        "Translate 'Hello world' to Chinese.",
    ]
    token_ids_list = [encode(tokenizer, p) for p in prompts]

    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS),
        prompt_token_ids=token_ids_list,
    )

    assert len(outputs) == len(prompts)
    for i, (prompt, out) in enumerate(zip(prompts, outputs)):
        text = decode(tokenizer, out.token_ids)
        print(f"\n[batch {i}]\n  prompt : {prompt!r}\n  output : {text!r}")
        assert len(out.token_ids) > 0


def test_greedy_determinism(llm, tokenizer):
    """Same prompt with temperature=0 must produce identical outputs twice."""
    from nanovllm_jax.sampling_params import SamplingParams

    prompt = "Once upon a time"
    token_ids = encode(tokenizer, prompt)
    sp = SamplingParams(temperature=0.0, max_tokens=32)

    out1 = llm.generate([prompt], sp, prompt_token_ids=[token_ids])[0]
    out2 = llm.generate([prompt], sp, prompt_token_ids=[token_ids])[0]

    print(f"\n[determinism]\n  run1: {out1.token_ids}\n  run2: {out2.token_ids}")
    assert out1.token_ids == out2.token_ids, \
        "Greedy decoding must be deterministic"


def test_max_tokens_limit(llm, tokenizer):
    """Output must never exceed max_tokens (even without EOS)."""
    from nanovllm_jax.sampling_params import SamplingParams

    prompt = "Count from 1 to 1000:"
    token_ids = encode(tokenizer, prompt)
    max_t = 20

    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=max_t, ignore_eos=True),
        prompt_token_ids=[token_ids],
    )
    assert len(outputs[0].token_ids) == max_t, \
        f"Expected exactly {max_t} tokens, got {len(outputs[0].token_ids)}"


def test_chat_template(llm, tokenizer):
    """Apply Qwen3 chat template and verify coherent response."""
    from nanovllm_jax.sampling_params import SamplingParams

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "What is 2 + 2?"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    token_ids = encode(tokenizer, prompt)

    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS),
        prompt_token_ids=[token_ids],
    )
    text = decode(tokenizer, outputs[0].token_ids)
    print(f"\n[chat_template]\n  prompt : {prompt!r}\n  output : {text!r}")

    assert len(outputs[0].token_ids) > 0
    # Model should mention '4' when asked '2+2'
    assert "4" in text, f"Expected '4' in response, got: {text!r}"


def test_long_prompt(llm, tokenizer):
    """A prompt with 500+ tokens should not crash the block manager."""
    from nanovllm_jax.sampling_params import SamplingParams

    # Generate a ~500 token prompt by repeating text
    base = "The quick brown fox jumps over the lazy dog. "
    prompt = base * 30   # ~300 words, ~400 tokens
    token_ids = encode(tokenizer, prompt)
    print(f"\n[long_prompt] prompt length: {len(token_ids)} tokens")

    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=16),
        prompt_token_ids=[token_ids],
    )
    assert len(outputs[0].token_ids) > 0
