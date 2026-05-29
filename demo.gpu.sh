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
from nanovllm_jax.layers.attention import PagedKVCache
from nanovllm_jax.engine.model_runner import _jit_prefill, _jit_decode, _next_pow2
from nanovllm_jax.engine.sequence import Sequence
from nanovllm_jax.engine.block_manager import BlockManager
from nanovllm_jax.sampling_params import SamplingParams
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(resolved)
model = load_model(resolved, dtype="bfloat16")
mc = model.config

print(f"\n[DEBUG] 模型配置:")
print(f"  vocab_size         = {mc.vocab_size}")
print(f"  tie_word_embeddings= {mc.tie_word_embeddings}")
print(f"  hidden_size        = {mc.hidden_size}")
print(f"  num_hidden_layers  = {mc.num_hidden_layers}")
print(f"  num_kv_heads       = {mc.num_key_value_heads}")
print(f"  head_dim           = {mc.head_dim}")

# 权重绑定诊断
print("\n[DIAG] === 权重绑定检查 ===")
embed_w = np.array(model.model.embed_tokens.weight_array, dtype=np.float32)
print(f"  embed_tokens  shape={embed_w.shape}  norm={np.linalg.norm(embed_w):.4f}")
lmh_w = np.array(model.lm_head.effective_weight, dtype=np.float32)
print(f"  lm_head       shape={lmh_w.shape}  norm={np.linalg.norm(lmh_w):.4f}")
if mc.tie_word_embeddings:
    ok = np.allclose(embed_w, lmh_w, atol=1e-4)
    print(f"  tied weights match: {ok}" + ("" if ok else f"  !! max_diff={np.abs(embed_w-lmh_w).max():.6f}"))

kv_cache = PagedKVCache(
    num_layers=mc.num_hidden_layers,
    num_kv_heads=mc.num_key_value_heads,
    head_dim=mc.head_dim,
    num_blocks=num_blocks,
    block_size=block_size,
    dtype=jnp.bfloat16,
)
block_manager = BlockManager(num_blocks=num_blocks, block_size=block_size)

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
print(f"\n[DEBUG] Question : {raw_question!r}")
print(f"[DEBUG] Prompt tokens ({T}): {input_ids}")
print(f"[DEBUG] Prompt text : {prompt!r}")

# Pre-allocate enough blocks for prompt + max_tokens
total_tokens_needed = T + max_tokens
blocks_needed = math.ceil(total_tokens_needed / block_size)
print(f"\n[DEBUG] Pre-allocating {blocks_needed} blocks for {total_tokens_needed} tokens")

seq = Sequence(seq_id=0, prompt_token_ids=input_ids,
               sampling_params=SamplingParams(temperature=0.0, max_tokens=max_tokens))

# FIX 1: block_manager.allocate(seq) is idempotent — it always computes
# _blocks_needed(seq.total_len) which equals ceil(T/block_size) = 2 for a
# 24-token prompt, so looping N times still allocates only 2 blocks.
# Instead, directly append_slot until we have blocks_needed entries.
while len(seq.block_table) < blocks_needed:
    seq.block_table.append(block_manager._allocator.allocate())

print(f"[DEBUG] block_table length: {len(seq.block_table)}")
print(f"[DEBUG] block_table: {list(seq.block_table)}")

# ---- Prefill ----
all_ids = jnp.array(input_ids, dtype=jnp.int32)
all_pos = jnp.arange(T, dtype=jnp.int32)
all_bi  = jnp.array([seq.block_table[i // block_size] for i in range(T)], dtype=jnp.int32)
all_bo  = jnp.array([i % block_size for i in range(T)], dtype=jnp.int32)
seq_lens    = jnp.array([T], dtype=jnp.int32)
last_indices = jnp.array([T - 1], dtype=jnp.int32)

T_pad = _next_pow2(T)
def pad1d(a, n, v=0):
    return jnp.pad(a, (0, n - a.shape[0]), constant_values=v)
if T_pad != T:
    all_ids = pad1d(all_ids, T_pad)
    all_pos = pad1d(all_pos, T_pad)
    all_bi  = pad1d(all_bi,  T_pad)
    all_bo  = pad1d(all_bo,  T_pad)

print(f"\n[DEBUG] Prefill: T_real={T}, T_pad={T_pad}")
print(f"[DEBUG] Prefill bi (first {T}): {list(np.array(all_bi[:T]))}")
logits = _jit_prefill(
    model, kv_cache,
    all_ids, all_pos, all_bi, all_bo, seq_lens, last_indices,
    num_real_tokens=T,
)
logits_np = np.array(logits, dtype=np.float32)
print(f"[DEBUG] Prefill logits shape: {logits_np.shape}")
print(f"[DEBUG] Prefill logits: min={logits_np[0].min():.4f}  max={logits_np[0].max():.4f}")
top5_ids = np.argsort(logits_np[0])[::-1][:5]
print(f"  Top-5 ids  : {top5_ids.tolist()}")
print(f"  Top-5 text : {[tok.decode([i]) for i in top5_ids]}")
print(f"  Top-5 logit: {logits_np[0][top5_ids].tolist()}")

argmax_token = int(np.argmax(logits_np[0]))
print(f"\n[DEBUG] Prefill argmax: id={argmax_token}  text={tok.decode([argmax_token])!r}")

# ---- Decode ----
print(f"\n[DEBUG] ===== Decode {max_tokens} steps =====")
seq.append_token(argmax_token)
generated = [argmax_token]

for step_i in range(max_tokens - 1):
    step = seq.total_len - 1
    blk_num = step // block_size
    blk_off = step % block_size

    # FIX 2: ensure the block for this step exists.
    # Pre-allocation covered prompt+max_tokens, but append_token grows
    # seq.total_len so we may need a slot beyond the pre-allocated range
    # if the caller changes max_tokens. append_slot is safe to call every
    # step — it only allocates when the current capacity is exceeded.
    block_manager.append_slot(seq)

    # Sanity check (should never trigger after the fixes above)
    if blk_num >= len(seq.block_table):
        print(f"  [WARN] step {step_i+1}: blk_num={blk_num} >= block_table len {len(seq.block_table)}, stopping")
        break

    dec_ids  = jnp.array([seq.last_token_id], dtype=jnp.int32)
    dec_pos  = jnp.array([step], dtype=jnp.int32)
    dec_bi   = jnp.array([seq.block_table[blk_num]], dtype=jnp.int32)
    dec_bo   = jnp.array([blk_off], dtype=jnp.int32)
    dec_lens = jnp.array([seq.total_len], dtype=jnp.int32)
    # Pass the full block_table so kv_cache.read() can access all context blocks
    dec_bt   = jnp.array([list(seq.block_table)], dtype=jnp.int32)

    print(f"  [DEBUG] step {step_i+1:3d}: pos={step} blk_num={blk_num} blk_off={blk_off} "
          f"bt_len={len(seq.block_table)} dec_bt_shape={dec_bt.shape}")

    dec_logits = _jit_decode(
        model, kv_cache,
        dec_ids, dec_pos, dec_bi, dec_bo, dec_lens, dec_bt,
        num_real_seqs=1,
    )
    dec_np = np.array(dec_logits, dtype=np.float32)
    top5_dec = np.argsort(dec_np[0])[::-1][:3]
    print(f"         logits min={dec_np[0].min():.3f} max={dec_np[0].max():.3f} "
          f"top3={[tok.decode([i]) for i in top5_dec]}")
    next_tok = int(np.argmax(dec_np[0]))
    print(f"  step {step_i+1:3d}: {tok.decode([seq.last_token_id])!r} -> {tok.decode([next_tok])!r}  (id={next_tok})")
    seq.append_token(next_tok)
    generated.append(next_tok)
    if next_tok == tok.eos_token_id:
        print("  [EOS reached]")
        break

full_response = tok.decode(generated, skip_special_tokens=True)
print(f"\n{'='*60}")
print(f"Q: {raw_question}")
print(f"A: {full_response}")
print(f"{'='*60}")
PYEOF
