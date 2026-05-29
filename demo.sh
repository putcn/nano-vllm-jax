#!/usr/bin/env bash
# demo.sh — CPU-only demo (no GPU required)
# Runs nano-vllm-jax inside the CPU Docker image.
#
# Usage:
#   bash demo.sh
#   MODEL=Qwen/Qwen3-1.7B bash demo.sh
#   MODEL=/absolute/local/path bash demo.sh
#   HF_CACHE=/data/models bash demo.sh
#   MAX_TOKENS=128 bash demo.sh
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MODEL_PATH="${MODEL_PATH:-}"
MAX_TOKENS="${MAX_TOKENS:-128}"
NUM_BLOCKS="${NUM_BLOCKS:-128}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

echo "==> nano-vllm-jax CPU demo"
echo "    model      : ${MODEL_PATH:-$MODEL}"
echo "    max_tokens : $MAX_TOKENS"
echo "    num_blocks : $NUM_BLOCKS"
echo "    HF cache   : $HF_CACHE"
echo ""

docker build -q -f docker/Dockerfile.cpu -t nano-vllm-jax:cpu . 2>&1 | tail -3

docker run --rm -i \
  -e JAX_PLATFORMS=cpu \
  -e HF_HOME=/root/.cache/huggingface \
  -e MODEL="$MODEL" \
  -e MODEL_PATH="$MODEL_PATH" \
  -e MAX_TOKENS="$MAX_TOKENS" \
  -e NUM_BLOCKS="$NUM_BLOCKS" \
  -v "$(pwd)":/workspace \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -w /workspace \
  nano-vllm-jax:cpu \
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
num_blocks = int(os.environ.get("NUM_BLOCKS", 128))

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
    max_num_seqs=4,
    max_num_batched_tokens=512,
    max_model_len=512,
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
print(" nano-vllm-jax  —  Qwen3-0.6B  multi-Q&A demo  [CPU]")
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
