"""Llama model (JAX port of nanovllm/models/llama.py).

Status: ✅ Fixed (residual connections, tie_word_embeddings)

Residual connection design (Pre-Norm)
--------------------------------------
Standard Pre-Norm transformer block:

    residual = x
    x = rms_norm1(x)          # normalise only
    x = attn(x) + residual    # add residual after attention
    residual = x
    x = rms_norm2(x)          # normalise only
    x = mlp(x) + residual     # add residual after MLP

Previous bug: attn_out was passed directly into post_attention_layernorm
via the fused-residual API, bypassing the attention residual add entirely.
This broke the residual stream across every layer, producing garbage logits.

Fix: use RMSNorm without the fused-residual path in the decoder layer;
manage residuals explicitly so the flow is always clear and correct.

Weight tying design
-------------------
When tie_word_embeddings=True, load_weights() copies embed_tokens weights
into lm_head.weight at load time. lm_head.__call__ always reads its own
nnx.Param — no JIT-boundary / abstract-tracer issues.
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
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: Optional[int] = None
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    vocab_size: int = 32000
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    tp_size: int = 1
    tp_rank: int = 0

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class LlamaMLP(nnx.Module):
    def __init__(self, config: LlamaConfig) -> None:
        tp = config.tp_size
        tr = config.tp_rank
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
        x: jax.Array,
        positions: jax.Array,
        kv_cache: PagedKVCache,
        block_indices: jax.Array,
        block_offsets: jax.Array,
        seq_lens: jax.Array,
        is_prefill: bool,
        block_table: Optional[jax.Array] = None,
        num_real_tokens: Optional[int] = None,
        num_real_seqs: Optional[int] = None,
    ) -> jax.Array:
        T = x.shape[0]
        qkv = self.qkv_proj(x)
        local_q  = self.attn.num_heads
        local_kv = self.attn.num_kv_heads
        hd = self.head_dim
        q = qkv[:, :local_q * hd].reshape(T, local_q, hd)
        k = qkv[:, local_q * hd: (local_q + local_kv) * hd].reshape(T, local_kv, hd)
        v = qkv[:, (local_q + local_kv) * hd:].reshape(T, local_kv, hd)
        q, k = self.rope(q, k, positions)
        out = self.attn(
            q, k, v, kv_cache,
            block_indices, block_offsets, seq_lens,
            is_prefill, block_table,
            num_real_tokens=num_real_tokens,
            num_real_seqs=num_real_seqs,
        )
        out = out.reshape(out.shape[0], -1)
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
        num_real_tokens: Optional[int] = None,
        num_real_seqs: Optional[int] = None,
    ) -> jax.Array:
        # --- Pre-Norm attention block ---
        # residual add happens AFTER attention, not inside the norm call
        residual = x
        x = self.input_layernorm(x)           # normalise only (no residual arg)
        x = self.self_attn(
            x, positions, kv_cache,
            block_indices, block_offsets, seq_lens,
            is_prefill, block_table,
            num_real_tokens=num_real_tokens,
            num_real_seqs=num_real_seqs,
        )
        x = x + residual                       # residual add after attention

        # --- Pre-Norm MLP block ---
        residual = x
        x = self.post_attention_layernorm(x)  # normalise only
        x = self.mlp(x)
        x = x + residual                       # residual add after MLP
        return x

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
    def __init__(self, config: LlamaConfig) -> None:
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size,
            tp_size=config.tp_size, tp_rank=config.tp_rank,
        )
        self.layers = nnx.List([
            LlamaDecoderLayer(config, i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        num_real_tokens: Optional[int] = None,
        num_real_seqs: Optional[int] = None,
    ) -> jax.Array:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(
                x, positions, kv_cache,
                block_indices, block_offsets, seq_lens,
                is_prefill, block_table,
                num_real_tokens=num_real_tokens,
                num_real_seqs=num_real_seqs,
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
    def __init__(self, config: LlamaConfig) -> None:
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size,
            tp_size=config.tp_size, tp_rank=config.tp_rank,
        )
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
        num_real_tokens: Optional[int] = None,
        num_real_seqs: Optional[int] = None,
    ) -> jax.Array:
        hidden = self.model(
            input_ids, positions, kv_cache,
            block_indices, block_offsets, seq_lens,
            is_prefill, block_table,
            num_real_tokens=num_real_tokens,
            num_real_seqs=num_real_seqs,
        )
        return self.lm_head(hidden, last_indices=last_indices)

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
        model_params = {
            k.removeprefix("model."): v
            for k, v in params.items() if k.startswith("model.")
        }
        self.model.load_weights(model_params)

        if self.config.tie_word_embeddings:
            embed_w = self.model.embed_tokens.weight_array
            self.lm_head.load_weight(embed_w)
            print(f"[WEIGHT] tie_word_embeddings: copied embed_tokens -> lm_head "
                  f"(norm={float(jnp.linalg.norm(embed_w)):.4f})")
        elif "lm_head.weight" in params:
            self.lm_head.load_weight(jnp.array(params["lm_head.weight"]))
