"""Llama model (JAX port of nanovllm/models/llama.py).

Status: ✅ Done (Phase 4)

Components built bottom-up:
  LlamaMLP          — SwiGLU feed-forward
  LlamaAttention    — QKV proj + RoPE + Attention + output proj
  LlamaDecoderLayer — RMSNorm + Attention + MLP with residual
  LlamaModel        — embedding + N decoder layers + final norm
  LlamaForCausalLM  — LlamaModel + LM head (weight-tied)

Weight loading follows the original nanovllm weight_loader pattern:
  each layer exposes load_weights(params: dict[str, jax.Array]).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import jax
import jax.numpy as jnp
from flax import nnx

from nanovllm_jax.layers.activation import SiluAndMul
from nanovllm_jax.layers.attention import Attention, PagedKVCache
from nanovllm_jax.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm_jax.layers.layernorm import RMSNorm
from nanovllm_jax.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nanovllm_jax.layers.rotary_embedding import RotaryEmbedding
from nanovllm_jax.layers.sampler import Sampler


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LlamaConfig:
    """Subset of HuggingFace LlamaConfig fields used by this implementation."""
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: Optional[int] = None          # inferred if None
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    vocab_size: int = 32000
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    # TP (single-device default)
    tp_size: int = 1
    tp_rank: int = 0

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class LlamaMLP(nnx.Module):
    """SwiGLU MLP: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: LlamaConfig) -> None:
        tp = config.tp_size
        tr = config.tp_rank
        # gate and up are merged into one ColumnParallel for efficiency
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            tp_size=tp, tp_rank=tr,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size,
            tp_size=tp, tp_rank=tr,
        )
        self.act = SiluAndMul()

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(self.act(self.gate_up_proj(x)))

    def load_weights(self, params: dict) -> None:
        self.gate_up_proj.load_weight(jnp.array(params["gate_proj.weight"]), shard_id=0)
        self.gate_up_proj.load_weight(jnp.array(params["up_proj.weight"]), shard_id=1)
        self.down_proj.load_weight(jnp.array(params["down_proj.weight"]))


# ---------------------------------------------------------------------------
# Attention layer
# ---------------------------------------------------------------------------

class LlamaAttention(nnx.Module):
    """Multi-head / GQA attention with RoPE."""

    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        tp = config.tp_size
        tr = config.tp_rank
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size, config.head_dim,
            config.num_attention_heads, config.num_key_value_heads,
            tp_size=tp, tp_rank=tr,
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            tp_size=tp, tp_rank=tr,
        )
        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.attn = Attention(
            num_heads=config.num_attention_heads // tp,
            head_dim=config.head_dim,
            num_kv_heads=config.num_key_value_heads // tp,
            layer_idx=layer_idx,
        )

    def __call__(
        self,
        x: jax.Array,             # (T, hidden_size)
        positions: jax.Array,     # (T,) int32
        kv_cache: PagedKVCache,
        block_indices: jax.Array,
        block_offsets: jax.Array,
        seq_lens: jax.Array,
        is_prefill: bool,
        block_table: Optional[jax.Array] = None,
    ) -> jax.Array:
        T = x.shape[0]
        qkv = self.qkv_proj(x)   # (T, (H + 2*Hkv) * head_dim)

        # Split QKV
        q_size = (self.num_heads // self.attn.num_heads * self.attn.num_heads) * self.head_dim
        # Use the layer's local head counts (post-TP)
        local_q = self.attn.num_heads
        local_kv = self.attn.num_kv_heads
        hd = self.head_dim
        q = qkv[:, :local_q * hd].reshape(T, local_q, hd)
        k = qkv[:, local_q * hd: (local_q + local_kv) * hd].reshape(T, local_kv, hd)
        v = qkv[:, (local_q + local_kv) * hd:].reshape(T, local_kv, hd)

        # Apply RoPE
        q, k = self.rope(q, k, positions)

        # Attention
        out = self.attn(
            q, k, v, kv_cache,
            block_indices, block_offsets, seq_lens,
            is_prefill, block_table,
        )  # (T, local_q, hd) or (num_seqs, local_q, hd)

        # Merge heads and project
        out = out.reshape(out.shape[0], -1)  # (T, local_q * hd)
        return self.o_proj(out)

    def load_weights(self, params: dict) -> None:
        self.qkv_proj.load_weight(jnp.array(params["q_proj.weight"]), shard_id="q")
        self.qkv_proj.load_weight(jnp.array(params["k_proj.weight"]), shard_id="k")
        self.qkv_proj.load_weight(jnp.array(params["v_proj.weight"]), shard_id="v")
        self.o_proj.load_weight(jnp.array(params["o_proj.weight"]))


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class LlamaDecoderLayer(nnx.Module):
    """Single transformer block: norm → attn → residual → norm → mlp → residual."""

    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)

    def __call__(
        self,
        x: jax.Array,
        positions: jax.Array,
        kv_cache: PagedKVCache,
        block_indices: jax.Array,
        block_offsets: jax.Array,
        seq_lens: jax.Array,
        is_prefill: bool,
        block_table: Optional[jax.Array] = None,
    ) -> jax.Array:
        # Fused residual norm: returns (normed, residual)
        normed, residual = self.input_layernorm(x, residual=jnp.zeros_like(x))
        attn_out = self.self_attn(
            normed, positions, kv_cache,
            block_indices, block_offsets, seq_lens,
            is_prefill, block_table,
        )
        # Second fused residual norm
        normed2, residual2 = self.post_attention_layernorm(attn_out, residual=residual)
        mlp_out = self.mlp(normed2)
        return mlp_out + residual2

    def load_weights(self, params: dict) -> None:
        self.input_layernorm.weight = nnx.Param(
            jnp.array(params["input_layernorm.weight"])
        )
        self.post_attention_layernorm.weight = nnx.Param(
            jnp.array(params["post_attention_layernorm.weight"])
        )
        self.self_attn.load_weights({
            k.removeprefix("self_attn."): v
            for k, v in params.items() if k.startswith("self_attn.")
        })
        self.mlp.load_weights({
            k.removeprefix("mlp."): v
            for k, v in params.items() if k.startswith("mlp.")
        })


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class LlamaModel(nnx.Module):
    """Llama transformer: embedding + N decoder layers + final norm."""

    def __init__(self, config: LlamaConfig) -> None:
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size,
            tp_size=config.tp_size, tp_rank=config.tp_rank,
        )
        self.layers = [
            LlamaDecoderLayer(config, i)
            for i in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: jax.Array,       # (T,) int32
        positions: jax.Array,       # (T,) int32
        kv_cache: PagedKVCache,
        block_indices: jax.Array,
        block_offsets: jax.Array,
        seq_lens: jax.Array,
        is_prefill: bool,
        block_table: Optional[jax.Array] = None,
    ) -> jax.Array:
        x = self.embed_tokens(input_ids)  # (T, hidden_size)
        for layer in self.layers:
            x = layer(
                x, positions, kv_cache,
                block_indices, block_offsets, seq_lens,
                is_prefill, block_table,
            )
        return self.norm(x)

    def load_weights(self, params: dict) -> None:
        self.embed_tokens.load_weight(jnp.array(params["embed_tokens.weight"]))
        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}."
            layer_params = {
                k.removeprefix(prefix): v
                for k, v in params.items() if k.startswith(prefix)
            }
            layer.load_weights(layer_params)
        self.norm.weight = nnx.Param(jnp.array(params["norm.weight"]))


class LlamaForCausalLM(nnx.Module):
    """Llama causal LM: model + LM head, optional weight tying."""

    def __init__(self, config: LlamaConfig) -> None:
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size,
            tp_size=config.tp_size, tp_rank=config.tp_rank,
        )
        if config.tie_word_embeddings:
            self.lm_head.tie_weights(self.model.embed_tokens)
        self.sampler = Sampler()

    def __call__(
        self,
        input_ids: jax.Array,
        positions: jax.Array,
        kv_cache: PagedKVCache,
        block_indices: jax.Array,
        block_offsets: jax.Array,
        seq_lens: jax.Array,
        is_prefill: bool,
        block_table: Optional[jax.Array] = None,
        last_indices: Optional[jax.Array] = None,
    ) -> jax.Array:
        hidden = self.model(
            input_ids, positions, kv_cache,
            block_indices, block_offsets, seq_lens,
            is_prefill, block_table,
        )
        logits = self.lm_head(hidden, last_indices=last_indices)
        return logits

    def sample(
        self,
        logits: jax.Array,
        key: Optional[jax.Array] = None,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> jax.Array:
        return self.sampler(logits, key=key, temperature=temperature,
                            top_k=top_k, top_p=top_p)

    def load_weights(self, params: dict) -> None:
        """Load weights from a flat dict keyed by HuggingFace param names.

        Expects keys like:
          model.embed_tokens.weight
          model.layers.0.self_attn.q_proj.weight
          ...
          lm_head.weight
        """
        model_params = {
            k.removeprefix("model."): v
            for k, v in params.items() if k.startswith("model.")
        }
        self.model.load_weights(model_params)
        if not self.config.tie_word_embeddings and "lm_head.weight" in params:
            self.lm_head.load_weight(jnp.array(params["lm_head.weight"]))
