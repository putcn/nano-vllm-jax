#!/usr/bin/env bash
# demo.gpu.sh — NVIDIA GPU demo (CUDA 12, requires nvidia-container-toolkit)
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MODEL_PATH="${MODEL_PATH:-}"
MAX_TOKENS="${MAX_TOKENS:-128}"
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

# 利用 Docker 层缓存：不加 --no-cache，依赖层在未变更时直接复用
# 只有源代码变更时才需重建最后的 COPY + pip install 层
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
import os, pathlib
import jax
import jax.numpy as jnp
import numpy as np

print(f"JAX devices: {jax.devices()}")

model_id   = os.environ["MODEL"]
model_path = os.environ.get("MODEL_PATH", "").strip()
max_tokens = int(os.environ.get("MAX_TOKENS", 128))
num_blocks = int(os.environ.get("NUM_BLOCKS", 512))

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

# 权重绑定诊断
print("\n[DIAG] === 权重绑定检查 ===")
embed_w = np.array(model.model.embed_tokens.weight_array, dtype=np.float32)
print(f"  embed_tokens  shape={embed_w.shape}  norm={np.linalg.norm(embed_w):.4f}  is_zero={np.abs(embed_w).max()==0}")
lmh_w = np.array(model.lm_head.effective_weight, dtype=np.float32)
print(f"  lm_head       shape={lmh_w.shape}  norm={np.linalg.norm(lmh_w):.4f}  is_zero={np.abs(lmh_w).max()==0}")
if mc.tie_word_embeddings:
    ok = np.allclose(embed_w, lmh_w, atol=1e-4)
    print(f"  tied weights match: {ok}" + ("" if ok else f"  !! max_diff={np.abs(embed_w-lmh_w).max():.6f}"))

kv_cache = PagedKVCache(
    num_layers=mc.num_hidden_layers,
    num_kv_heads=mc.num_key_value_heads,
    head_dim=mc.head_dim,
    num_blocks=num_blocks,
    block_size=16,
    dtype=jnp.bfloat16,
)
block_manager = BlockManager(num_blocks=num_blocks, block_size=16)

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
print(f"\n[DEBUG] Question : {raw_question!r}")
print(f"[DEBUG] Token IDs ({len(input_ids)}): {input_ids}")

seq = Sequence(seq_id=0, prompt_token_ids=input_ids,
               sampling_params=SamplingParams(temperature=0.0, max_tokens=max_tokens))
block_manager.allocate(seq)

T = len(input_ids)
all_ids = jnp.array(input_ids, dtype=jnp.int32)
all_pos = jnp.arange(T, dtype=jnp.int32)
all_bi  = jnp.array([seq.block_table[i // 16] for i in range(T)], dtype=jnp.int32)
all_bo  = jnp.array([i % 16 for i in range(T)], dtype=jnp.int32)
seq_lens    = jnp.array([T], dtype=jnp.int32)
last_indices = jnp.array([T - 1], dtype=jnp.int32)

T_pad = _next_pow2(T)
def pad1d(a, n, v=0):
    return jnp.pad(a, (0, n - a.shape[0]), constant_values=v)
if T_pad != T:
    all_ids = pad1d(all_ids, T_pad)
    all_pos = pad1d(all_pos, T_pad)
    all_bi  = pad1d(all_bi, T_pad)
    all_bo  = pad1d(all_bo, T_pad)

logits = _jit_prefill(
    model, kv_cache,
    all_ids, all_pos, all_bi, all_bo, seq_lens, last_indices,
    num_real_tokens=T,
)
logits_np = np.array(logits, dtype=np.float32)
print(f"\n[DEBUG] Prefill logits: min={logits_np[0].min():.4f}  max={logits_np[0].max():.4f}")
top5 = np.argsort(logits_np[0])[::-1][:5]
print(f"  Top-5: {[tok.decode([i]) for i in top5]}")

argmax_token = int(np.argmax(logits_np[0]))
print(f"\n[DEBUG] Prefill argmax: id={argmax_token}  text={tok.decode([argmax_token])!r}")

print(f"\n[DEBUG] ===== Decode 前10步 =====")
seq.append_token(argmax_token)
for step_i in range(10):
    step = seq.total_len - 1
    dec_ids  = jnp.array([seq.last_token_id], dtype=jnp.int32)
    dec_pos  = jnp.array([step], dtype=jnp.int32)
    dec_bi   = jnp.array([seq.block_table[step // 16]], dtype=jnp.int32)
    dec_bo   = jnp.array([step % 16], dtype=jnp.int32)
    dec_lens = jnp.array([seq.total_len], dtype=jnp.int32)
    dec_bt   = jnp.array([list(seq.block_table)], dtype=jnp.int32)
    dec_logits = _jit_decode(
        model, kv_cache,
        dec_ids, dec_pos, dec_bi, dec_bo, dec_lens, dec_bt,
        num_real_seqs=1,
    )
    dec_np = np.array(dec_logits, dtype=np.float32)
    next_tok = int(np.argmax(dec_np[0]))
    print(f"  step {step_i+1}: {tok.decode([seq.last_token_id])!r} -> {tok.decode([next_tok])!r}")
    seq.append_token(next_tok)
    if next_tok == tok.eos_token_id:
        break

print(f"\n[DEBUG] 生成结果: {tok.decode(seq.all_token_ids[T:])!r}")
PYEOF
