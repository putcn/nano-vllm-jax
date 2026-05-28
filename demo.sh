#!/usr/bin/env bash
# demo.sh — CPU-only demo (no GPU required)
# Runs nano-vllm-jax inside the CPU Docker image.
# HuggingFace model cache is shared from your host machine.
#
# Usage:
#   bash demo.sh
#   MODEL=Qwen/Qwen3-1.7B bash demo.sh
#   MODEL=/absolute/local/path bash demo.sh
#   HF_CACHE=/data/models bash demo.sh
#   MAX_TOKENS=128 bash demo.sh
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MAX_TOKENS="${MAX_TOKENS:-64}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

echo "==> nano-vllm-jax CPU demo"
echo "    model      : $MODEL"
echo "    max_tokens : $MAX_TOKENS"
echo "    HF cache   : $HF_CACHE (mounted from host)"
echo ""

docker build -q -f docker/Dockerfile.cpu -t nano-vllm-jax:cpu . 2>&1 | tail -3

docker run --rm -i \
  -e JAX_PLATFORMS=cpu \
  -e HF_HOME=/root/.cache/huggingface \
  -e MODEL="$MODEL" \
  -e MAX_TOKENS="$MAX_TOKENS" \
  -v "$(pwd)":/workspace \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -w /workspace \
  nano-vllm-jax:cpu \
  python - <<'PYEOF'
import os, pathlib
from huggingface_hub import snapshot_download
from nanovllm_jax.llm import LLM
from nanovllm_jax.sampling_params import SamplingParams
from transformers import AutoTokenizer

model_id   = os.environ["MODEL"]
max_tokens = int(os.environ.get("MAX_TOKENS", 64))

if pathlib.Path(model_id).exists():
    resolved = model_id
else:
    print(f"Downloading {model_id} ...")
    resolved = snapshot_download(model_id)
    print(f"Saved to: {resolved}")

print(f"\nLoading model from {resolved} ...")
llm = LLM(
    model=resolved,
    num_gpu_blocks=128,
    block_size=16,
    max_num_seqs=4,
    max_num_batched_tokens=512,
    max_model_len=512,
    eos_token_id=151645,
)

tok = AutoTokenizer.from_pretrained(resolved)
prompts = [
    "The capital of France is",
    "What is JAX? Answer in one sentence:",
    "Write a haiku about programming:",
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
