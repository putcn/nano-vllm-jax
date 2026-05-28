"""Unit tests for Llama model components (Phase 4)."""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from nanovllm_jax.models.llama import (
    LlamaConfig,
    LlamaMLP,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
)
from nanovllm_jax.layers.attention import PagedKVCache


def rand(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def tiny_config(**kwargs) -> LlamaConfig:
    defaults = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        vocab_size=256,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    defaults.update(kwargs)
    return LlamaConfig(**defaults)


def make_cache(config: LlamaConfig, num_blocks=16, block_size=8) -> PagedKVCache:
    return PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=jnp.float32,
    )


def prefill_args(T: int):
    bi = jnp.arange(T, dtype=jnp.int32) // 8
    bo = jnp.arange(T, dtype=jnp.int32) % 8
    sl = jnp.array([T], dtype=jnp.int32)
    return bi, bo, sl


def load_random_weights(layer, cfg, seed=42):
    """Load non-trivial random weights into a LlamaDecoderLayer."""
    rng = np.random.default_rng(seed)
    H = cfg.hidden_size
    I = cfg.intermediate_size
    Hq = cfg.num_attention_heads * cfg.head_dim
    Hkv = cfg.num_key_value_heads * cfg.head_dim
    params = {
        "input_layernorm.weight":           rng.standard_normal(H).astype(np.float32) + 1.0,
        "post_attention_layernorm.weight":  rng.standard_normal(H).astype(np.float32) + 1.0,
        "self_attn.q_proj.weight":          rng.standard_normal((Hq, H)).astype(np.float32),
        "self_attn.k_proj.weight":          rng.standard_normal((Hkv, H)).astype(np.float32),
        "self_attn.v_proj.weight":          rng.standard_normal((Hkv, H)).astype(np.float32),
        "self_attn.o_proj.weight":          rng.standard_normal((H, Hq)).astype(np.float32),
        "mlp.gate_proj.weight":             rng.standard_normal((I, H)).astype(np.float32),
        "mlp.up_proj.weight":               rng.standard_normal((I, H)).astype(np.float32),
        "mlp.down_proj.weight":             rng.standard_normal((H, I)).astype(np.float32),
    }
    layer.load_weights(params)


# ---------------------------------------------------------------------------
# LlamaMLP
# ---------------------------------------------------------------------------

def test_mlp_shape():
    cfg = tiny_config()
    mlp = LlamaMLP(cfg)
    x = jnp.array(rand((6, cfg.hidden_size)))
    out = mlp(x)
    assert out.shape == (6, cfg.hidden_size)


def test_mlp_load_weights():
    cfg = tiny_config()
    mlp = LlamaMLP(cfg)
    params = {
        "gate_proj.weight": rand((cfg.intermediate_size, cfg.hidden_size), 0),
        "up_proj.weight":   rand((cfg.intermediate_size, cfg.hidden_size), 1),
        "down_proj.weight": rand((cfg.hidden_size, cfg.intermediate_size), 2),
    }
    mlp.load_weights(params)
    x = jnp.array(rand((4, cfg.hidden_size)))
    assert mlp(x).shape == (4, cfg.hidden_size)


# ---------------------------------------------------------------------------
# LlamaAttention
# ---------------------------------------------------------------------------

def test_attention_layer_shape():
    cfg = tiny_config()
    attn = LlamaAttention(cfg, layer_idx=0)
    cache = make_cache(cfg)
    T = 5
    x = jnp.array(rand((T, cfg.hidden_size)))
    pos = jnp.arange(T, dtype=jnp.int32)
    bi, bo, sl = prefill_args(T)
    out = attn(x, pos, cache, bi, bo, sl, is_prefill=True)
    assert out.shape == (T, cfg.hidden_size)


def test_attention_layer_decode_shape():
    cfg = tiny_config()
    attn = LlamaAttention(cfg, layer_idx=0)
    cache = make_cache(cfg)
    T = 4
    x_p = jnp.array(rand((T, cfg.hidden_size)))
    pos_p = jnp.arange(T, dtype=jnp.int32)
    bi, bo, sl = prefill_args(T)
    attn(x_p, pos_p, cache, bi, bo, sl, is_prefill=True)
    x_d = jnp.array(rand((1, cfg.hidden_size)))
    pos_d = jnp.array([T], dtype=jnp.int32)
    bi_d = jnp.array([0], dtype=jnp.int32)
    bo_d = jnp.array([T % 8], dtype=jnp.int32)
    sl_d = jnp.array([T + 1], dtype=jnp.int32)
    bt = jnp.array([[0, 1]], dtype=jnp.int32)
    out = attn(x_d, pos_d, cache, bi_d, bo_d, sl_d, is_prefill=False, block_table=bt)
    assert out.shape == (1, cfg.hidden_size)


# ---------------------------------------------------------------------------
# LlamaDecoderLayer
# ---------------------------------------------------------------------------

def test_decoder_layer_shape():
    cfg = tiny_config()
    layer = LlamaDecoderLayer(cfg, layer_idx=0)
    cache = make_cache(cfg)
    T = 6
    x = jnp.array(rand((T, cfg.hidden_size)))
    pos = jnp.arange(T, dtype=jnp.int32)
    bi, bo, sl = prefill_args(T)
    out = layer(x, pos, cache, bi, bo, sl, is_prefill=True)
    assert out.shape == (T, cfg.hidden_size)


def test_decoder_layer_residual():
    """With non-trivial weights, output must differ from input."""
    cfg = tiny_config()
    layer = LlamaDecoderLayer(cfg, layer_idx=0)
    load_random_weights(layer, cfg)
    cache = make_cache(cfg)
    T = 4
    x = jnp.array(rand((T, cfg.hidden_size)))
    pos = jnp.arange(T, dtype=jnp.int32)
    bi, bo, sl = prefill_args(T)
    out = layer(x, pos, cache, bi, bo, sl, is_prefill=True)
    assert not jnp.allclose(out, x, atol=1e-3)


# ---------------------------------------------------------------------------
# LlamaModel
# ---------------------------------------------------------------------------

def test_llama_model_shape():
    cfg = tiny_config()
    model = LlamaModel(cfg)
    cache = make_cache(cfg)
    T = 8
    ids = jnp.array(np.random.randint(0, cfg.vocab_size, (T,)), dtype=jnp.int32)
    pos = jnp.arange(T, dtype=jnp.int32)
    bi, bo, sl = prefill_args(T)
    out = model(ids, pos, cache, bi, bo, sl, is_prefill=True)
    assert out.shape == (T, cfg.hidden_size)


def test_llama_model_deterministic():
    cfg = tiny_config()
    model = LlamaModel(cfg)
    cache_a = make_cache(cfg)
    cache_b = make_cache(cfg)
    T = 4
    ids = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    pos = jnp.arange(T, dtype=jnp.int32)
    bi, bo, sl = prefill_args(T)
    out_a = model(ids, pos, cache_a, bi, bo, sl, is_prefill=True)
    out_b = model(ids, pos, cache_b, bi, bo, sl, is_prefill=True)
    np.testing.assert_allclose(np.array(out_a), np.array(out_b), atol=1e-5)


# ---------------------------------------------------------------------------
# LlamaForCausalLM
# ---------------------------------------------------------------------------

def test_causal_lm_logits_shape():
    cfg = tiny_config()
    lm = LlamaForCausalLM(cfg)
    cache = make_cache(cfg)
    T = 6
    ids = jnp.array(np.random.randint(0, cfg.vocab_size, (T,)), dtype=jnp.int32)
    pos = jnp.arange(T, dtype=jnp.int32)
    bi, bo, sl = prefill_args(T)
    logits = lm(ids, pos, cache, bi, bo, sl, is_prefill=True)
    assert logits.shape == (T, cfg.vocab_size)


def test_causal_lm_last_indices():
    cfg = tiny_config()
    lm = LlamaForCausalLM(cfg)
    cache = make_cache(cfg)
    T = 8
    ids = jnp.array(np.random.randint(0, cfg.vocab_size, (T,)), dtype=jnp.int32)
    pos = jnp.arange(T, dtype=jnp.int32)
    bi, bo, sl = prefill_args(T)
    last = jnp.array([3, 7], dtype=jnp.int32)
    logits = lm(ids, pos, cache, bi, bo, sl, is_prefill=True, last_indices=last)
    assert logits.shape == (2, cfg.vocab_size)


def test_causal_lm_weight_tying():
    cfg = tiny_config(tie_word_embeddings=True)
    lm = LlamaForCausalLM(cfg)
    cache = make_cache(cfg)
    T = 4
    ids = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    pos = jnp.arange(T, dtype=jnp.int32)
    bi, bo, sl = prefill_args(T)
    logits = lm(ids, pos, cache, bi, bo, sl, is_prefill=True)
    assert logits.shape == (T, cfg.vocab_size)


def test_causal_lm_greedy_sample():
    cfg = tiny_config()
    lm = LlamaForCausalLM(cfg)
    cache = make_cache(cfg)
    T = 4
    ids = jnp.array([10, 20, 30, 40], dtype=jnp.int32)
    pos = jnp.arange(T, dtype=jnp.int32)
    bi, bo, sl = prefill_args(T)
    last = jnp.array([T - 1], dtype=jnp.int32)
    logits = lm(ids, pos, cache, bi, bo, sl, is_prefill=True, last_indices=last)
    tokens = lm.sample(logits, temperature=0.0)
    assert tokens.shape == (1,)
    assert 0 <= int(tokens[0]) < cfg.vocab_size
