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
#   MAX_TOKENS=128 bash demo.apple.sh
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
MODEL_PATH="${MODEL_PATH:-}"
MAX_TOKENS="${MAX_TOKENS:-128}"
NUM_BLOCKS="${NUM_BLOCKS:-256}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export JAX_PLATFORMS=METAL

echo "==> nano-vllm-jax Apple Silicon demo"
echo "    model      : ${MODEL_PATH:-$MODEL}"
echo "    max_tokens : $MAX_TOKENS"
echo "    num_blocks : $NUM_BLOCKS"
echo "    HF cache   : $HF_HOME"
echo ""

# Activate venv if present at repo root
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

# Verify Metal backend
python - <<'PYEOF'
import sys, jax
devices = jax.devices()
print(f"JAX devices: {devices}")
has_metal = any('metal' in str(d).lower() or 'gpu' in str(d).lower() for d in devices)
if not has_metal:
    print("ERROR: Metal GPU not detected.", file=sys.stderr)
    print("Install jax-metal: pip install jax-metal", file=sys.stderr)
    sys.exit(1)
print("Metal OK")
PYEOF

MODEL="$MODEL" MODEL_PATH="$MODEL_PATH" MAX_TOKENS="$MAX_TOKENS" NUM_BLOCKS="$NUM_BLOCKS" \
python - <<'PYEOF'
import os, pathlib, time
import jax
from huggingface_hub import snapshot_download
from nanovllm_jax.llm import LLM
from nanovllm_jax.sampling_params import SamplingParams
from transformers import AutoTokenizer

print(f"JAX devices: {jax.devices()}")

model_id   = os.environ["MODEL"]
model_path = os.environ.get("MODEL_PATH", "").strip()
max_tokens = int(os.environ.get("MAX_TOKENS", 128))
num_blocks = int(os.environ.get("NUM_BLOCKS", 256))

if model_path and pathlib.Path(model_path).exists():
    resolved = model_path
elif pathlib.Path(model_id).exists():
    resolved = model_id
else:
    print(f"Downloading {model_id} ...")
    resolved = snapshot_download(model_id)
    print(f"Saved to: {resolved}")

print(f"Loading model from {resolved} ...")
tok = AutoTokenizer.from_pretrained(resolved)
llm = LLM(
    model=resolved,
    num_gpu_blocks=num_blocks,
    block_size=16,
    max_num_seqs=8,
    max_num_batched_tokens=2048,
    max_model_len=2048,
)

QA_PAIRS = [
    ("What is the capital of France? Answer in one word.",            32),
    ("What is the capital of Japan? Answer in one word.",             32),
    ("Which planet is closest to the Sun? Answer in one word.",       32),
    ("What is 17 multiplied by 13? Answer with just the number.",     32),
    ("What is the square root of 144? Answer with just the number.",  32),
    ("Write a Python one-liner that prints the Fibonacci sequence up to 100.", 80),
    ("Write a Python function that checks if a string is a palindrome.",       128),
    ("If all roses are flowers and some flowers fade quickly, can we conclude "
     "that some roses fade quickly? Answer yes or no and explain briefly.",    96),
    ("Translate 'Good morning, how are you?' into French.",           48),
    ("Translate 'Thank you very much' into Japanese.",                 48),
]

def make_prompt(question):
    messages = [{"role": "user", "content": question}]
    try:
        return tok.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True,
        )

print("\n" + "="*70)
print(" nano-vllm-jax  —  Qwen3-0.6B  multi-Q&A demo  [Apple Metal]")
print("="*70)

# warm-up
print("\n[warm-up] compiling JIT kernels...")
wq, wmt = QA_PAIRS[0]
llm.generate(
    [make_prompt(wq)],
    SamplingParams(temperature=0.0, max_tokens=wmt),
)
print("[warm-up] done.\n")

for i, (question, mt) in enumerate(QA_PAIRS):
    prompt = make_prompt(question)
    sp = SamplingParams(temperature=0.0, max_tokens=mt)
    t0 = time.perf_counter()
    outputs = llm.generate([prompt], sp)
    elapsed = (time.perf_counter() - t0) * 1000
    answer = tok.decode(outputs[0].token_ids, skip_special_tokens=True).strip()
    print(f"Q{i+1}: {question}")
    print(f"A{i+1}: {answer}")
    print(f"     [{elapsed:.0f}ms total]")
    print()

print("="*70)
PYEOF
