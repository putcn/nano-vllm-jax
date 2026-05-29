#!/usr/bin/env bash
# demo.gpu.sh — NVIDIA GPU demo (CUDA 12, requires nvidia-container-toolkit)
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MODEL_PATH="${MODEL_PATH:-}"
MAX_TOKENS="${MAX_TOKENS:-64}"
NUM_BLOCKS="${NUM_BLOCKS:-512}"
GPU_ID="${GPU_ID:-all}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

echo "==> nano-vllm-jax GPU demo"
echo "    model      : ${MODEL_PATH:-$MODEL}"
echo "    max_tokens : $MAX_TOKENS"
echo "    num_blocks : $NUM_BLOCKS"
echo "    GPU        : $GPU_ID"
echo "    HF cache   : $HF_CACHE"
echo ""

docker build --progress=plain -f docker/Dockerfile.gpu -t nano-vllm-jax:gpu .

if [ "$GPU_ID" = "all" ]; then
  GPU_FLAG="--gpus all"
else
  GPU_FLAG="--gpus device=$GPU_ID"
fi

docker run --rm -i $GPU_FLAG \
  -e HF_HOME=/root/.cache/huggingface \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e MODEL="$MODEL" \
  -e MODEL_PATH="$MODEL_PATH" \
  -e MAX_TOKENS="$MAX_TOKENS" \
  -e NUM_BLOCKS="$NUM_BLOCKS" \
  -v "$(pwd)":/workspace \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -w /workspace \
  nano-vllm-jax:gpu \
  python - <<'PYEOF'
import os, pathlib, math
import jax
import jax.numpy as jnp
import numpy as np

print(f"JAX devices: {jax.devices()}")

model_id   = os.environ["MODEL"]
model_path = os.environ.get("MODEL_PATH", "").strip()
max_tokens = int(os.environ.get("MAX_TOKENS", 64))
num_blocks = int(os.environ.get("NUM_BLOCKS", 512))
block_size = 16

from huggingface_hub import snapshot_download
if model_path and pathlib.Path(model_path).exists():
    resolved = model_path
elif pathlib.Path(model_id).exists():
    resolved = model_id
else:
    print(f"Downloading {model_id} ...")
    resolved = snapshot_download(model_id)
    print(f"Saved to: {resolved}")

print(f"\nLoading model from {resolved} ...")

from nanovllm_jax.loader.model_registry import load_model
from nanovllm_jax.layers.attention import PagedKVCache, _make_batched_causal_mask
from nanovllm_jax.engine.model_runner import _jit_prefill, _jit_decode, _next_pow2
from nanovllm_jax.engine.sequence import Sequence
from nanovllm_jax.engine.block_manager import BlockManager
from nanovllm_jax.sampling_params import SamplingParams
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(resolved)
model = load_model(resolved, dtype="bfloat16")
mc = model.config

print(f"\n[CONFIG] vocab={mc.vocab_size}  hidden={mc.hidden_size}  "
      f"layers={mc.num_hidden_layers}  heads={mc.num_attention_heads}  "
      f"kv_heads={mc.num_key_value_heads}  head_dim={mc.head_dim}  "
      f"rope_theta={mc.rope_theta}  tie_embed={mc.tie_word_embeddings}")

# ============================================================
# SECTION 1: HuggingFace reference logits
# ============================================================
print("\n" + "="*60)
print("SECTION 1: HuggingFace reference forward pass")
print("="*60)

raw_question = "What is the capital of France? Answer in one word."
messages = [{"role": "user", "content": raw_question}]
try:
    prompt = tok.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
except TypeError:
    prompt = tok.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True,
    )

input_ids = tok.encode(prompt, add_special_tokens=False)
T = len(input_ids)
print(f"  Prompt ({T} tokens): {input_ids}")
print(f"  Prompt text: {prompt!r}")

try:
    import torch
    from transformers import AutoModelForCausalLM
    print("  Loading HF model for reference...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        resolved, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    hf_model.eval()
    with torch.no_grad():
        ids_t = torch.tensor([input_ids])
        hf_out = hf_model(ids_t)
        hf_logits = hf_out.logits[0, -1].float().numpy()  # (vocab,)
    top5_hf = np.argsort(hf_logits)[::-1][:5]
    print(f"  HF top-5: {[tok.decode([i]) for i in top5_hf]}")
    print(f"  HF logits: {hf_logits[top5_hf].tolist()}")
    print(f"  HF spread: {hf_logits.max()-hf_logits.min():.4f}")
    del hf_model
except Exception as e:
    print(f"  [SKIP] HF reference failed: {e}")

# ============================================================
# SECTION 2: Layer-by-layer eager forward with sub-module breakdown
# ============================================================
print("\n" + "="*60)
print("SECTION 2: Eager per-layer forward (sub-module breakdown)")
print("="*60)

kv_diag = PagedKVCache(
    num_layers=mc.num_hidden_layers,
    num_kv_heads=mc.num_key_value_heads,
    head_dim=mc.head_dim,
    num_blocks=64,
    block_size=block_size,
    dtype=jnp.bfloat16,
)
bm_diag = BlockManager(num_blocks=64, block_size=block_size)
seq_diag = Sequence(seq_id=99, prompt_token_ids=input_ids,
                    sampling_params=SamplingParams(temperature=0.0, max_tokens=8))
blocks_for_diag = math.ceil((T + 8) / block_size) + 1
while len(seq_diag.block_table) < blocks_for_diag:
    seq_diag.block_table.append(bm_diag._allocator.allocate())

ids_e   = jnp.array(input_ids, dtype=jnp.int32)
pos_e   = jnp.arange(T, dtype=jnp.int32)
bi_e    = jnp.array([seq_diag.block_table[i // block_size] for i in range(T)], dtype=jnp.int32)
bo_e    = jnp.array([i % block_size for i in range(T)], dtype=jnp.int32)
slens_e = jnp.array([T], dtype=jnp.int32)

def stat(name, arr):
    a = np.array(arr, dtype=np.float32).ravel()
    print(f"    {name:30s} norm={np.linalg.norm(a):9.4f}  "
          f"mean={a.mean():8.4f}  std={a.std():8.4f}  "
          f"max={a.max():9.4f}  min={a.min():9.4f}")

x = model.model.embed_tokens(ids_e)
stat("embed output [last tok]", np.array(x, dtype=np.float32)[T-1])

for li in range(mc.num_hidden_layers):
    layer = model.model.layers[li]
    residual = x

    # --- attention sub-block ---
    x_norm1 = layer.input_layernorm(x)
    qkv = layer.self_attn.qkv_proj(x_norm1)
    lq  = layer.self_attn.attn.num_heads
    lkv = layer.self_attn.attn.num_kv_heads
    hd  = layer.self_attn.head_dim
    q = qkv[:, :lq*hd].reshape(T, lq, hd)
    k = qkv[:, lq*hd:(lq+lkv)*hd].reshape(T, lkv, hd)
    v = qkv[:, (lq+lkv)*hd:].reshape(T, lkv, hd)
    q_r, k_r = layer.self_attn.rope(q, k, pos_e)
    kv_diag.write(li, bi_e, bo_e, k_r, v)

    kv_groups = layer.self_attn.attn.kv_groups
    k_rep = jnp.repeat(k_r, kv_groups, axis=1) if kv_groups > 1 else k_r
    v_rep = jnp.repeat(v,   kv_groups, axis=1) if kv_groups > 1 else v
    scale = layer.self_attn.attn.scale

    # cast to float32 for numerical stability (matches reference)
    q_f = q_r.astype(jnp.float32)
    k_f = k_rep.astype(jnp.float32)
    v_f = v_rep.astype(jnp.float32)
    q_t = jnp.transpose(q_f, (1, 0, 2))   # (H, T, D)
    k_t = jnp.transpose(k_f, (1, 2, 0))   # (H, D, T)
    attn_logits = jnp.matmul(q_t, k_t) * scale
    mask = _make_batched_causal_mask(slens_e, T)
    attn_logits = jnp.where(mask[None], attn_logits, jnp.finfo(jnp.float32).min)
    attn_w = jax.nn.softmax(attn_logits, axis=-1).astype(q_r.dtype)
    v_t   = jnp.transpose(v_rep.astype(q_r.dtype), (1, 0, 2))
    attn_out_heads = jnp.transpose(jnp.matmul(attn_w, v_t), (1, 0, 2))  # (T,H,D)
    attn_proj = layer.self_attn.o_proj(attn_out_heads.reshape(T, -1))
    x_after_attn = attn_proj + residual

    # --- MLP sub-block ---
    residual2 = x_after_attn
    x_norm2   = layer.post_attention_layernorm(x_after_attn)
    gate_up   = layer.mlp.gate_up_proj(x_norm2)
    act_out   = layer.mlp.act(gate_up)
    mlp_out   = layer.mlp.down_proj(act_out)
    x = mlp_out + residual2

    # Print breakdown for first 4 layers and last layer
    if li < 4 or li == mc.num_hidden_layers - 1:
        last = lambda a: np.array(a, dtype=np.float32)[T-1]
        print(f"  [layer {li:2d}]")
        stat("  x_norm1 last",       last(x_norm1))
        stat("  q last",             last(q))
        stat("  k last",             last(k))
        stat("  q_rope last",        last(q_r))
        stat("  attn_proj last",     last(attn_proj))
        stat("  x_after_attn last",  last(x_after_attn))
        stat("  x_norm2 last",       last(x_norm2))
        stat("  gate_up last",       np.array(gate_up, dtype=np.float32)[T-1])
        stat("  act_out last",       np.array(act_out, dtype=np.float32)[T-1])
        stat("  mlp_out last",       last(mlp_out))
        stat("  x (after residual)", last(x))
        print()

x_final = model.model.norm(x)
lmh_w = np.array(model.lm_head.effective_weight, dtype=np.float32)
logits_eager = np.array(x_final[T-1:T].astype(jnp.float32), dtype=np.float32) @ lmh_w.T
logits_eager = logits_eager[0]
top5_e = np.argsort(logits_eager)[::-1][:5]
print(f"\n[EAGER] lm_head spread={logits_eager.max()-logits_eager.min():.4f}")
print(f"  top-5: {[tok.decode([i]) for i in top5_e]}")
print(f"  logits: {logits_eager[top5_e].tolist()}")

# ============================================================
# SECTION 3: JIT prefill + decode
# ============================================================
print("\n" + "="*60)
print("SECTION 3: JIT prefill + decode")
print("="*60)

total_needed = T + max_tokens
blocks_needed = math.ceil(total_needed / block_size)

kv_cache = PagedKVCache(
    num_layers=mc.num_hidden_layers,
    num_kv_heads=mc.num_key_value_heads,
    head_dim=mc.head_dim,
    num_blocks=num_blocks,
    block_size=block_size,
    dtype=jnp.bfloat16,
)
block_manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
seq = Sequence(seq_id=0, prompt_token_ids=input_ids,
               sampling_params=SamplingParams(temperature=0.0, max_tokens=max_tokens))
while len(seq.block_table) < blocks_needed:
    seq.block_table.append(block_manager._allocator.allocate())

all_ids = jnp.array(input_ids, dtype=jnp.int32)
all_pos = jnp.arange(T, dtype=jnp.int32)
all_bi  = jnp.array([seq.block_table[i // block_size] for i in range(T)], dtype=jnp.int32)
all_bo  = jnp.array([i % block_size for i in range(T)], dtype=jnp.int32)
seq_lens     = jnp.array([T], dtype=jnp.int32)
last_indices = jnp.array([T - 1], dtype=jnp.int32)

T_pad = _next_pow2(T)
def pad1d(a, n, v=0):
    return jnp.pad(a, (0, n - a.shape[0]), constant_values=v)
if T_pad != T:
    all_ids = pad1d(all_ids, T_pad)
    all_pos = pad1d(all_pos, T_pad)
    all_bi  = pad1d(all_bi,  T_pad)
    all_bo  = pad1d(all_bo,  T_pad)

print(f"  T_real={T}  T_pad={T_pad}")
logits = _jit_prefill(
    model, kv_cache,
    all_ids, all_pos, all_bi, all_bo, seq_lens, last_indices,
    num_real_tokens=T,
)
logits_np = np.array(logits, dtype=np.float32)
top5_j = np.argsort(logits_np[0])[::-1][:5]
print(f"  JIT prefill spread={logits_np[0].max()-logits_np[0].min():.4f}")
print(f"  top-5: {[tok.decode([i]) for i in top5_j]}")
print(f"  logits: {logits_np[0][top5_j].tolist()}")

argmax_token = int(np.argmax(logits_np[0]))
print(f"  first token: {tok.decode([argmax_token])!r}  (id={argmax_token})")

seq.append_token(argmax_token)
generated = [argmax_token]

for step_i in range(max_tokens - 1):
    step    = seq.total_len - 1
    blk_num = step // block_size
    blk_off = step % block_size
    block_manager.append_slot(seq)
    if blk_num >= len(seq.block_table):
        break
    dec_ids  = jnp.array([seq.last_token_id], dtype=jnp.int32)
    dec_pos  = jnp.array([step], dtype=jnp.int32)
    dec_bi   = jnp.array([seq.block_table[blk_num]], dtype=jnp.int32)
    dec_bo   = jnp.array([blk_off], dtype=jnp.int32)
    dec_lens = jnp.array([seq.total_len], dtype=jnp.int32)
    dec_bt   = jnp.array([list(seq.block_table)], dtype=jnp.int32)
    dec_logits = _jit_decode(
        model, kv_cache, dec_ids, dec_pos, dec_bi, dec_bo,
        dec_lens, dec_bt, num_real_seqs=1,
    )
    next_tok = int(np.argmax(np.array(dec_logits, dtype=np.float32)[0]))
    seq.append_token(next_tok)
    generated.append(next_tok)
    if next_tok == tok.eos_token_id:
        break

full_response = tok.decode(generated, skip_special_tokens=True)
print(f"\n{'='*60}")
print(f"Q: {raw_question}")
print(f"A: {full_response}")
print(f"{'='*60}")
PYEOF
