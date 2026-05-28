#!/usr/bin/env bash
# demo.apple.sh — Apple Silicon (M1/M2/M3/M4) Metal demo
#
# Uses JAX's Metal backend (Apple GPU via MPS). Runs natively on macOS.
# No Docker needed.
#
# Usage:
#   bash demo.apple.sh
#   MODEL=Qwen/Qwen3-0.6B bash demo.apple.sh
#   MODEL=~/models/Qwen3-0.6B bash demo.apple.sh
#   MAX_TOKENS=256 bash demo.apple.sh
#
# One-time setup:
#   brew install python@3.11
#   python3.11 -m venv .venv && source .venv/bin/activate
#   pip install jax-metal
#   pip install -e ".[dev]"
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]] || [[ "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: demo.apple.sh requires macOS on Apple Silicon (arm64)." >&2
  echo "       On NVIDIA GPU use demo.gpu.sh; on CPU use demo.sh." >&2
  exit 1
fi

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MAX_TOKENS="${MAX_TOKENS:-128}"
NUM_BLOCKS="${NUM_BLOCKS:-256}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export JAX_PLATFORMS=METAL

echo "==> nano-vllm-jax Apple Silicon demo"
echo "    model      : $MODEL"
echo "    max_tokens : $MAX_TOKENS"
echo "    HF cache   : $HF_HOME"
echo ""

# Activate venv if present at repo root
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

# Verify Metal backend — hard exit if not found
python - <<'PYEOF'
import sys
import jax
devices = jax.devices()
print(f"JAX devices: {devices}")
has_metal = any('metal' in str(d).lower() or 'gpu' in str(d).lower() for d in devices)
if not has_metal:
    print("ERROR: Metal GPU not detected.", file=sys.stderr)
    print("Install jax-metal: pip install jax-metal", file=sys.stderr)
    print("Then re-run: bash demo.apple.sh", file=sys.stderr)
    sys.exit(1)
print("Metal OK")
PYEOF

MODEL="$MODEL" MAX_TOKENS="$MAX_TOKENS" NUM_BLOCKS="$NUM_BLOCKS" \
python - <<'PYEOF'
import os, pathlib
from huggingface_hub import snapshot_download
from nanovllm_jax.llm import LLM
from nanovllm_jax.sampling_params import SamplingParams
from transformers import AutoTokenizer
import jax

print(f"JAX devices: {jax.devices()}")

model_id   = os.environ["MODEL"]
max_tokens = int(os.environ.get("MAX_TOKENS", 128))
num_blocks = int(os.environ.get("NUM_BLOCKS", 256))

if pathlib.Path(model_id).exists():
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
    max_num_seqs=8,
    max_num_batched_tokens=2048,
    max_model_len=2048,
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
