"""Unit tests for the HuggingFace weight loader (Phase 5).

All tests use a tiny synthetic checkpoint written to a tmp directory;
no real model download is required.
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers: build a minimal synthetic HF checkpoint
# ---------------------------------------------------------------------------

def _make_tiny_config(model_dir: Path, **overrides) -> dict:
    """Write a minimal config.json and return it as dict."""
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-5,
        "vocab_size": 256,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
    }
    cfg.update(overrides)
    (model_dir / "config.json").write_text(json.dumps(cfg))
    return cfg


def _make_weight_dict(cfg: dict, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Return a flat weight dict matching HF Llama key names for cfg."""
    H = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    L = cfg["num_hidden_layers"]
    V = cfg["vocab_size"]
    Hq = cfg["num_attention_heads"] * (H // cfg["num_attention_heads"])
    Hkv = cfg["num_key_value_heads"] * (H // cfg["num_attention_heads"])

    params: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": rng.standard_normal((V, H)).astype(np.float32),
        "model.norm.weight": rng.standard_normal(H).astype(np.float32) + 1.0,
        "lm_head.weight": rng.standard_normal((V, H)).astype(np.float32),
    }
    for i in range(L):
        p = f"model.layers.{i}."
        params[p + "input_layernorm.weight"] = rng.standard_normal(H).astype(np.float32) + 1.0
        params[p + "post_attention_layernorm.weight"] = rng.standard_normal(H).astype(np.float32) + 1.0
        params[p + "self_attn.q_proj.weight"] = rng.standard_normal((Hq, H)).astype(np.float32)
        params[p + "self_attn.k_proj.weight"] = rng.standard_normal((Hkv, H)).astype(np.float32)
        params[p + "self_attn.v_proj.weight"] = rng.standard_normal((Hkv, H)).astype(np.float32)
        params[p + "self_attn.o_proj.weight"] = rng.standard_normal((H, Hq)).astype(np.float32)
        params[p + "mlp.gate_proj.weight"] = rng.standard_normal((I, H)).astype(np.float32)
        params[p + "mlp.up_proj.weight"] = rng.standard_normal((I, H)).astype(np.float32)
        params[p + "mlp.down_proj.weight"] = rng.standard_normal((H, I)).astype(np.float32)
    return params


def _write_safetensors(path: Path, params: dict[str, np.ndarray]) -> None:
    try:
        from safetensors.numpy import save_file
    except ImportError:
        pytest.skip("safetensors not installed")
    save_file(params, str(path))


def _write_pytorch_bin(path: Path, params: dict[str, np.ndarray]) -> None:
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    state = {k: torch.tensor(v) for k, v in params.items()}
    torch.save(state, str(path))


# ---------------------------------------------------------------------------
# Tests: load_hf_config
# ---------------------------------------------------------------------------

def test_load_hf_config_reads_json():
    from nanovllm_jax.loader.weight_loader import load_hf_config
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _make_tiny_config(d)
        cfg = load_hf_config(d)
        assert cfg["hidden_size"] == 64
        assert cfg["architectures"] == ["LlamaForCausalLM"]


def test_load_hf_config_missing_raises():
    from nanovllm_jax.loader.weight_loader import load_hf_config
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError, match="config.json"):
            load_hf_config(tmp)


# ---------------------------------------------------------------------------
# Tests: llama_config_from_hf
# ---------------------------------------------------------------------------

def test_llama_config_from_hf_fields():
    from nanovllm_jax.loader.weight_loader import llama_config_from_hf
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _make_tiny_config(d)
        cfg = llama_config_from_hf(d)
        assert cfg.hidden_size == 64
        assert cfg.num_key_value_heads == 2
        assert cfg.head_dim == 16  # 64 // 4


def test_llama_config_from_hf_defaults():
    """Fields absent from config.json should use LlamaConfig defaults."""
    from nanovllm_jax.loader.weight_loader import llama_config_from_hf
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        raw = _make_tiny_config(d)
        # Remove optional fields
        raw.pop("rope_theta")
        raw.pop("rms_norm_eps")
        (d / "config.json").write_text(json.dumps(raw))
        cfg = llama_config_from_hf(d)
        assert cfg.rope_theta == 10000.0
        assert cfg.rms_norm_eps == 1e-5


# ---------------------------------------------------------------------------
# Tests: _shard_files
# ---------------------------------------------------------------------------

def test_shard_files_single_safetensors():
    from nanovllm_jax.loader.weight_loader import _shard_files
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "model.safetensors").touch()
        shards = _shard_files(d)
        assert len(shards) == 1
        assert shards[0].name == "model.safetensors"


def test_shard_files_index_safetensors():
    from nanovllm_jax.loader.weight_loader import _shard_files
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        index = {"weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.norm.weight": "model-00002-of-00002.safetensors",
        }}
        (d / "model.safetensors.index.json").write_text(json.dumps(index))
        shards = _shard_files(d)
        assert len(shards) == 2


def test_shard_files_no_weights_raises():
    from nanovllm_jax.loader.weight_loader import _shard_files
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            _shard_files(Path(tmp))


# ---------------------------------------------------------------------------
# Tests: load_hf_weights (safetensors)
# ---------------------------------------------------------------------------

def test_load_hf_weights_safetensors():
    """Load a tiny synthetic safetensors checkpoint and verify forward pass."""
    from nanovllm_jax.loader.weight_loader import load_hf_weights, llama_config_from_hf
    from nanovllm_jax.models.llama import LlamaForCausalLM
    from nanovllm_jax.layers.attention import PagedKVCache
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        cfg_raw = _make_tiny_config(d)
        params = _make_weight_dict(cfg_raw, rng)
        _write_safetensors(d / "model.safetensors", params)

        cfg = llama_config_from_hf(d)
        model = LlamaForCausalLM(cfg)
        load_hf_weights(model, d)

        cache = PagedKVCache(
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            num_blocks=8, block_size=8, dtype=jnp.float32,
        )
        T = 4
        ids = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        pos = jnp.arange(T, dtype=jnp.int32)
        bi = jnp.zeros(T, dtype=jnp.int32)
        bo = jnp.arange(T, dtype=jnp.int32)
        sl = jnp.array([T], dtype=jnp.int32)
        logits = model(ids, pos, cache, bi, bo, sl, is_prefill=True)
        assert logits.shape == (T, cfg.vocab_size)


# ---------------------------------------------------------------------------
# Tests: model_registry
# ---------------------------------------------------------------------------

def test_registry_llama():
    from nanovllm_jax.loader.model_registry import get_model_class
    from nanovllm_jax.models.llama import LlamaForCausalLM
    model_cls, _ = get_model_class("LlamaForCausalLM")
    assert model_cls is LlamaForCausalLM


def test_registry_mistral_maps_to_llama():
    from nanovllm_jax.loader.model_registry import get_model_class
    from nanovllm_jax.models.llama import LlamaForCausalLM
    model_cls, _ = get_model_class("MistralForCausalLM")
    assert model_cls is LlamaForCausalLM


def test_registry_unknown_raises():
    from nanovllm_jax.loader.model_registry import get_model_class
    with pytest.raises(ValueError, match="Unsupported architecture"):
        get_model_class("GPT2LMHeadModel")


def test_load_model_safetensors():
    """load_model() auto-detects arch and returns a working model."""
    from nanovllm_jax.loader.model_registry import load_model
    from nanovllm_jax.layers.attention import PagedKVCache
    import jax.numpy as jnp

    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        cfg_raw = _make_tiny_config(d)
        params = _make_weight_dict(cfg_raw, rng)
        _write_safetensors(d / "model.safetensors", params)

        model = load_model(d)
        cfg = model.config

        cache = PagedKVCache(
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            num_blocks=8, block_size=8, dtype=jnp.float32,
        )
        T = 3
        ids = jnp.array([5, 10, 15], dtype=jnp.int32)
        pos = jnp.arange(T, dtype=jnp.int32)
        bi = jnp.zeros(T, dtype=jnp.int32)
        bo = jnp.arange(T, dtype=jnp.int32)
        sl = jnp.array([T], dtype=jnp.int32)
        logits = model(ids, pos, cache, bi, bo, sl, is_prefill=True)
        assert logits.shape == (T, cfg.vocab_size)
