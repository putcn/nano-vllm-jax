#!/usr/bin/env bash
# demo.gpu.sh — NVIDIA GPU demo (CUDA 12, requires nvidia-container-toolkit)
#
# Usage:
#   bash demo.gpu.sh
#   GPU_ID=0 bash demo.gpu.sh
#   MODEL=Qwen/Qwen3-1.7B NUM_BLOCKS=1024 bash demo.gpu.sh
#   MODEL_PATH=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/latest \
#     bash demo.gpu.sh
#
# Prerequisites:
#   nvidia-smi
#   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
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
echo "    HF cache   : $HF_CACHE (mounted from host)"
echo ""

docker build -q -f docker/Dockerfile.gpu -t nano-vllm-jax:gpu . 2>&1 | tail -3

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

# ── 1. 模型路径解析 ─────────────────────────────────────────────────────────
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

# ── 2. 直接加载模型，跳过 LLM 包装层，方便插入 debug hook ──────────────────
from nanovllm_jax.loader.model_registry import load_model
from nanovllm_jax.layers.attention import PagedKVCache
from nanovllm_jax.engine.model_runner import ModelRunner, _jit_prefill, _jit_decode, _next_pow2
from nanovllm_jax.engine.sequence import Sequence, SequenceStatus
from nanovllm_jax.engine.block_manager import BlockManager
from nanovllm_jax.sampling_params import SamplingParams
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(resolved)
model = load_model(resolved, dtype="bfloat16")
mc = model.config

print(f"\n[DEBUG] 模型配置:")
print(f"  vocab_size         = {mc.vocab_size}")
print(f"  num_hidden_layers  = {mc.num_hidden_layers}")
print(f"  num_attention_heads= {mc.num_attention_heads}")
print(f"  num_key_value_heads= {mc.num_key_value_heads}")
print(f"  head_dim           = {mc.head_dim}")
print(f"  hidden_size        = {mc.hidden_size}")

kv_cache = PagedKVCache(
    num_layers=mc.num_hidden_layers,
    num_kv_heads=mc.num_key_value_heads,
    head_dim=mc.head_dim,
    num_blocks=num_blocks,
    block_size=16,
    dtype=jnp.bfloat16,
)

block_manager = BlockManager(num_blocks=num_blocks, block_size=16)

# ── 3. 只用第一条 prompt 做单条 prefill debug ────────────────────────────────
prompt = "The capital of France is"
input_ids = tok.encode(prompt)
print(f"\n[DEBUG] Prompt: {prompt!r}")
print(f"[DEBUG] Token IDs ({len(input_ids)}): {input_ids}")

sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
seq = Sequence(seq_id=0, prompt_token_ids=input_ids, sampling_params=sp)

# 分配 KV block
block_manager.allocate(seq)
print(f"[DEBUG] 分配后 block_table: {seq.block_table}")

# 手动构建 prefill 输入
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
    all_bi  = pad1d(all_bi,  T_pad)
    all_bo  = pad1d(all_bo,  T_pad)

print(f"\n[DEBUG] Prefill 输入:")
print(f"  T_real={T}, T_pad={T_pad}")
print(f"  ids      = {np.array(all_ids)}")
print(f"  pos      = {np.array(all_pos)}")
print(f"  bi       = {np.array(all_bi)}")
print(f"  bo       = {np.array(all_bo)}")
print(f"  seq_lens = {np.array(seq_lens)}")
print(f"  last_idx = {np.array(last_indices)}")

# ── 4. 运行 prefill，打印 logit 统计 ─────────────────────────────────────────
logits = _jit_prefill(
    model, kv_cache,
    all_ids, all_pos, all_bi, all_bo, seq_lens, last_indices,
    num_real_tokens=T,
)
logits_np = np.array(logits)  # shape: [1, vocab_size]
print(f"\n[DEBUG] Prefill logits shape: {logits_np.shape}")
print(f"  logits[0] min={logits_np[0].min():.4f}  max={logits_np[0].max():.4f}  mean={logits_np[0].mean():.4f}")
print(f"  有 NaN? {np.isnan(logits_np).any()}  有 Inf? {np.isinf(logits_np).any()}")

top5_idx = np.argsort(logits_np[0])[::-1][:5]
print(f"  Top-5 token ids: {top5_idx.tolist()}")
print(f"  Top-5 logit val: {logits_np[0][top5_idx].tolist()}")
print(f"  Top-5 tokens   : {[tok.decode([i]) for i in top5_idx]}")

argmax_token = int(np.argmax(logits_np[0]))
print(f"\n[DEBUG] Prefill argmax token: id={argmax_token}  text={tok.decode([argmax_token])!r}")

# ── 5. 多步 decode debug（前 5 步）──────────────────────────────────────────
print(f"\n[DEBUG] ===== Decode 前5步 =====")
seq.append_token(argmax_token)

for step_i in range(5):
    step = seq.total_len - 1
    blk_num = step // 16
    blk_off = step % 16

    dec_ids  = jnp.array([seq.last_token_id], dtype=jnp.int32)
    dec_pos  = jnp.array([step], dtype=jnp.int32)
    dec_bi   = jnp.array([seq.block_table[blk_num]], dtype=jnp.int32)
    dec_bo   = jnp.array([blk_off], dtype=jnp.int32)
    dec_lens = jnp.array([seq.total_len], dtype=jnp.int32)
    max_blocks = len(seq.block_table)
    dec_bt   = jnp.array([list(seq.block_table)], dtype=jnp.int32)

    dec_logits = _jit_decode(
        model, kv_cache,
        dec_ids, dec_pos, dec_bi, dec_bo, dec_lens, dec_bt,
        num_real_seqs=1,
    )
    dec_np = np.array(dec_logits)
    next_tok = int(np.argmax(dec_np[0]))

    print(f"  step {step_i+1}: in={tok.decode([seq.last_token_id])!r}  "
          f"logit_max={dec_np[0].max():.3f}  logit_min={dec_np[0].min():.3f}  "
          f"→ out id={next_tok}  text={tok.decode([next_tok])!r}")

    seq.append_token(next_tok)

# ── 6. KV cache 健康检查 ──────────────────────────────────────────────────
import jax
cache_arr = jax.device_get(kv_cache.cache.get_value() if hasattr(kv_cache.cache, 'get_value') else kv_cache.cache.value)
print(f"\n[DEBUG] KV cache shape: {cache_arr.shape}")
print(f"  layer0 key block0: mean abs = {np.abs(cache_arr[0, 0, 0]).mean():.6f}")
print(f"  layer0 key block1: mean abs = {np.abs(cache_arr[0, 0, 1]).mean():.6f}")
nonzero_blocks = (np.abs(cache_arr[0, 0]).sum(axis=(-1,-2,-3)) > 0).sum()
print(f"  非零 key blocks (layer0): {nonzero_blocks} / {cache_arr.shape[2]}")

print("\n[DEBUG] ===== 诊断完成 =====\n")
PYEOF
