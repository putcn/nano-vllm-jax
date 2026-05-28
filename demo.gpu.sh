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

docker run --rm -it $GPU_FLAG \
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
from huggingface_hub import snapshot_download
from nanovllm_jax.llm import LLM
from nanovllm_jax.sampling_params import SamplingParams
from transformers import AutoTokenizer
import jax

print(f"JAX devices: {jax.devices()}")

model_id   = os.environ["MODEL"]
model_path = os.environ.get("MODEL_PATH", "").strip()
max_tokens = int(os.environ.get("MAX_TOKENS", 128))
num_blocks = int(os.environ.get("NUM_BLOCKS", 512))

if model_path and pathlib.Path(model_path).exists():
    resolved = model_path
elif pathlib.Path(model_id).exists():
    resolved = model_id
else:
    print(f"Downloading {model_id} ...")
    resolved = snapshot_download(model_id)
    print(f"Saved to: {resolved}")

print(f"\nLoading model from {resolved} ...")
llm = LLM(
    model=resolved,
    num_gpu_blocks=num_blocks,
    block_size=16,
    max_num_seqs=16,
    max_num_batched_tokens=4096,
    max_model_len=4096,
    eos_token_id=151645,
)

tok = AutoTokenizer.from_pretrained(resolved)
prompts = [
    "The capital of France is",
    "What is JAX? Answer in one sentence:",
    "Write a haiku about programming:",
    "Explain paged attention in two sentences:",
]
sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
outputs = llm.generate(
    prompts, sp,
    prompt_token_ids=[tok.encode(p) for p in prompts],
)

print("\n" + "="*60)
for prompt, out in zip(prompts, outputs):
    text = tok.decode(out.token_ids, skip_special_tokens=True)
    print(f"Prompt : {prompt}")
    print(f"Output : {text}")
    print("-"*60)
PYEOF
