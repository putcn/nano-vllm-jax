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
import os, pathlib, math, time
import jax
import jax.numpy as jnp
import numpy as np

print(f"JAX devices: {jax.devices()}")

model_id   = os.environ["MODEL"]
model_path = os.environ.get("MODEL_PATH", "").strip()
max_tokens = int(os.environ.get("MAX_TOKENS", 128))
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

print(f"Loading model from {resolved} ...")

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
print(f"Config: vocab={mc.vocab_size}  hidden={mc.hidden_size}  "
      f"layers={mc.num_hidden_layers}  heads={mc.num_attention_heads}  "
      f"kv={mc.num_key_value_heads}  head_dim={mc.head_dim}")

# ------------------------------------------------------------------ #
# Helper: run one full prefill+decode for a single question           #
# ------------------------------------------------------------------ #
def run_qa(question: str, max_tok: int = max_tokens) -> tuple[str, float, float]:
    """Returns (answer, prefill_ms, decode_tok_per_sec)."""
    messages = [{"role": "user", "content": question}]
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

    total_needed  = T + max_tok
    blocks_needed = math.ceil(total_needed / block_size) + 1

    kv_cache = PagedKVCache(
        num_layers=mc.num_hidden_layers,
        num_kv_heads=mc.num_key_value_heads,
        head_dim=mc.head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=jnp.bfloat16,
    )
    bm  = BlockManager(num_blocks=num_blocks, block_size=block_size)
    seq = Sequence(seq_id=0, prompt_token_ids=input_ids,
                   sampling_params=SamplingParams(temperature=0.0, max_tokens=max_tok))
    while len(seq.block_table) < blocks_needed:
        seq.block_table.append(bm._allocator.allocate())

    ids_j = jnp.array(input_ids, dtype=jnp.int32)
    pos_j = jnp.arange(T, dtype=jnp.int32)
    bi_j  = jnp.array([seq.block_table[i // block_size] for i in range(T)], jnp.int32)
    bo_j  = jnp.array([i % block_size for i in range(T)], jnp.int32)
    slens = jnp.array([T], dtype=jnp.int32)
    last  = jnp.array([T - 1], dtype=jnp.int32)

    T_pad = _next_pow2(T)
    def pad1d(a, n, v=0):
        return jnp.pad(a, (0, n - a.shape[0]), constant_values=v)
    if T_pad != T:
        ids_j = pad1d(ids_j, T_pad)
        pos_j = pad1d(pos_j, T_pad)
        bi_j  = pad1d(bi_j,  T_pad)
        bo_j  = pad1d(bo_j,  T_pad)

    t0 = time.perf_counter()
    logits = _jit_prefill(model, kv_cache, ids_j, pos_j, bi_j, bo_j, slens, last,
                          num_real_tokens=T)
    jax.block_until_ready(logits)
    prefill_ms = (time.perf_counter() - t0) * 1000

    logits_np  = np.array(logits, dtype=np.float32)
    next_id    = int(np.argmax(logits_np[0]))
    seq.append_token(next_id)
    generated  = [next_id]

    t1 = time.perf_counter()
    for _ in range(max_tok - 1):
        if next_id == tok.eos_token_id:
            break
        step    = seq.total_len - 1
        blk_num = step // block_size
        blk_off = step % block_size
        bm.append_slot(seq)
        if blk_num >= len(seq.block_table):
            break
        dec_logits = _jit_decode(
            model, kv_cache,
            jnp.array([seq.last_token_id], jnp.int32),
            jnp.array([step],              jnp.int32),
            jnp.array([seq.block_table[blk_num]], jnp.int32),
            jnp.array([blk_off],           jnp.int32),
            jnp.array([seq.total_len],     jnp.int32),
            jnp.array([list(seq.block_table)], jnp.int32),
            num_real_seqs=1,
        )
        next_id = int(np.argmax(np.array(dec_logits, dtype=np.float32)[0]))
        seq.append_token(next_id)
        generated.append(next_id)
    jax.block_until_ready(dec_logits)
    decode_s = time.perf_counter() - t1
    tps = len(generated) / decode_s if decode_s > 0 else 0.0

    answer = tok.decode(generated, skip_special_tokens=True).strip()
    return answer, prefill_ms, tps


# ------------------------------------------------------------------ #
# Q&A pairs                                                           #
# ------------------------------------------------------------------ #
QA_PAIRS = [
    # factual / geography
    ("What is the capital of France? Answer in one word.",            32),
    ("What is the capital of Japan? Answer in one word.",             32),
    ("Which planet is closest to the Sun? Answer in one word.",       32),
    # math
    ("What is 17 multiplied by 13? Answer with just the number.",     32),
    ("What is the square root of 144? Answer with just the number.",  32),
    # coding
    ("Write a Python one-liner that prints the Fibonacci sequence up to 100.", 80),
    ("Write a Python function that checks if a string is a palindrome.",       128),
    # reasoning
    ("If all roses are flowers and some flowers fade quickly, can we conclude "
     "that some roses fade quickly? Answer yes or no and explain briefly.",    96),
    # language
    ("Translate 'Good morning, how are you?' into French.",           48),
    ("Translate 'Thank you very much' into Japanese.",                 48),
]

print("\n" + "="*70)
print(" nano-vllm-jax  —  Qwen3-0.6B  multi-Q&A demo")
print("="*70)

# warm-up with first question (JIT compile)
print("\n[warm-up] compiling JIT kernels...")
_ = run_qa(QA_PAIRS[0][0], QA_PAIRS[0][1])
print("[warm-up] done.\n")

for i, (q, mt) in enumerate(QA_PAIRS):
    ans, pf_ms, tps = run_qa(q, mt)
    print(f"Q{i+1}: {q}")
    print(f"A{i+1}: {ans}")
    print(f"     [prefill {pf_ms:.0f}ms | decode {tps:.1f} tok/s]")
    print()

print("="*70)
PYEOF
