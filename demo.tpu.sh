#!/usr/bin/env bash
# demo.tpu.sh — Google Cloud TPU demo (TPU v2/v3/v4/v5)
#
# Runs directly on a GCE TPU VM (not inside Docker).
# TPU VMs expose /dev/accel* devices that are simpler to access natively.
#
# Usage (on a TPU VM):
#   bash demo.tpu.sh
#   MODEL=Qwen/Qwen3-0.6B bash demo.tpu.sh
#   HF_HOME=/mnt/gcs bash demo.tpu.sh     # GCS FUSE mount
#
# One-time setup on the TPU VM:
#   git clone https://github.com/putcn/nano-vllm-jax.git && cd nano-vllm-jax
#   pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
#   pip install -e ".[dev]"
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MAX_TOKENS="${MAX_TOKENS:-128}"
NUM_BLOCKS="${NUM_BLOCKS:-512}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export JAX_PLATFORMS=tpu

echo "==> nano-vllm-jax TPU demo"
echo "    model      : $MODEL"
echo "    max_tokens : $MAX_TOKENS"
echo "    HF cache   : $HF_HOME"
echo ""

python - <<'PYEOF'
import jax
devices = jax.devices()
print(f"JAX devices: {devices}")
assert any('tpu' in str(d).lower() for d in devices), \
    "No TPU devices found. Ensure jax[tpu] is installed and you are on a TPU VM."
print("TPU OK")
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
num_blocks = int(os.environ.get("NUM_BLOCKS", 512))

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
