# nano-vllm-jax

A JAX/Flax rewrite of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), a minimal vLLM-style LLM inference engine.

> See [MIGRATION_PLAN.md](./MIGRATION_PLAN.md) for the full porting plan and progress tracker.

## Goals
- Drop-in API compatible with `nano-vllm`
- Pure JAX/Flax backend (no PyTorch at inference time)
- JIT-compiled prefill and decode paths
- Paged KV cache via functional `jax.lax` ops
- Multi-device support via `jax.sharding`

---

## Quick Demo

Four demo scripts are provided. Each downloads the model automatically on first run and
reuses your host's HuggingFace cache on subsequent runs.

| Script | Target hardware | Docker? | Requires |
|---|---|---|---|
| `demo.sh` | CPU | ✅ | Docker |
| `demo.gpu.sh` | NVIDIA GPU (CUDA 12) | ✅ | Docker + nvidia-container-toolkit |
| `demo.tpu.sh` | Google Cloud TPU | ❌ native | Run on TPU VM, `jax[tpu]` |
| `demo.apple.sh` | Apple Silicon M1–M4 | ❌ native | macOS arm64, `jax-metal` |

### Model cache — host vs Docker

The demo scripts mount your **host** HuggingFace cache directly into the container:

```
host: $HOME/.cache/huggingface   (default, override with HF_CACHE=...)
  └── mounted to → container: /root/.cache/huggingface
```

This means `huggingface-cli download` on the host and the demo scripts share the same
cache — download once, run anywhere. To use a different directory:

```bash
HF_CACHE=/data/models bash demo.gpu.sh
```

> **Note:** `docker-compose.yml` (used for dev) uses a Docker **named volume** (`hf_cache`)
> instead, which keeps the cache isolated inside Docker. The demo scripts intentionally
> use host-directory mounts for easier model reuse.

---

### `demo.sh` — CPU

```bash
# Default: Qwen3-0.6B
bash demo.sh

# Custom model or path
MODEL=Qwen/Qwen3-1.7B bash demo.sh
MODEL=/data/models/Qwen3-0.6B bash demo.sh

# Custom cache dir or token count
HF_CACHE=/data/models bash demo.sh
MAX_TOKENS=128 bash demo.sh
```

---

### `demo.gpu.sh` — NVIDIA GPU

**Prerequisites**
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

```bash
bash demo.gpu.sh

# Single GPU
GPU_ID=0 bash demo.gpu.sh

# Larger model
MODEL=Qwen/Qwen3-1.7B NUM_BLOCKS=1024 bash demo.gpu.sh

# Local path (skip download)
MODEL_PATH=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/latest \
  bash demo.gpu.sh
```

---

### `demo.tpu.sh` — Google Cloud TPU

Runs **natively on the TPU VM** (no Docker). TPU device files (`/dev/accel*`) are
simpler to access without container overhead.

```bash
# 1. Create a TPU VM
gcloud compute tpus tpu-vm create my-tpu \
  --zone=us-central2-b \
  --accelerator-type=v4-8 \
  --version=tpu-vm-pt-2.0

# 2. SSH in and clone repo
gcloud compute tpus tpu-vm ssh my-tpu --zone=us-central2-b
git clone https://github.com/putcn/nano-vllm-jax.git && cd nano-vllm-jax

# 3. One-time setup
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -e ".[dev]"

# 4. Run
bash demo.tpu.sh
MODEL=Qwen/Qwen3-0.6B bash demo.tpu.sh

# Use a GCS FUSE mount as cache
HF_HOME=/mnt/gcs-models bash demo.tpu.sh
```

---

### `demo.apple.sh` — Apple Silicon (M1/M2/M3/M4)

Runs natively on macOS using JAX's Metal backend. No Docker needed.

**One-time setup**
```bash
brew install python@3.11
python3.11 -m venv .venv && source .venv/bin/activate
pip install jax-metal
pip install -e ".[dev]"
```

```bash
bash demo.apple.sh
MODEL=Qwen/Qwen3-0.6B bash demo.apple.sh
MAX_TOKENS=256 bash demo.apple.sh
MODEL=~/models/Qwen3-0.6B bash demo.apple.sh
```

The script sets `JAX_PLATFORMS=METAL` automatically and activates `.venv` if present.
If `jax-metal` is not installed or no Metal GPU is detected, the script **exits with an error**.

---

## Development Environments

Two Docker images are provided: one for **GPU** (CUDA 12 + JAX GPU backend) and one for
**CPU** (lightweight, good for CI and unit tests without a GPU).

### Prerequisites

| | GPU image | CPU image |
|---|---|---|
| Docker | ✅ required | ✅ required |
| [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) | ✅ required | ❌ not needed |
| NVIDIA driver ≥ 525 | ✅ required | ❌ not needed |

---

## Docker Quick Start

### Option A — docker compose (recommended for dev)

```bash
git clone https://github.com/putcn/nano-vllm-jax.git
cd nano-vllm-jax

# GPU container
docker compose -f docker/docker-compose.yml up -d dev-gpu
docker compose -f docker/docker-compose.yml exec dev-gpu bash

# CPU container
docker compose -f docker/docker-compose.yml up -d dev-cpu
docker compose -f docker/docker-compose.yml exec dev-cpu bash
```

> **Note:** `docker-compose.yml` uses a Docker **named volume** (`hf_cache`) for the HF
> model cache. The cache persists across container rebuilds but is **not** shared with
> `~/.cache/huggingface` on the host. To share with the host, use `demo.gpu.sh` or mount
> the directory manually (see below).

### Option B — manual docker run

```bash
# GPU — all GPUs, host HF cache shared
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  nano-vllm-jax:gpu bash

# CPU
docker run -it --rm \
  -e JAX_PLATFORMS=cpu \
  -v $(pwd):/workspace \
  nano-vllm-jax:cpu bash
```

---

## Running Tests Inside Docker

```bash
# All unit tests (no GPU / model required)
pytest tests/unit/ -v

# Specific module
pytest tests/unit/layers/ -v
pytest tests/unit/engine/ -v

# Parallel run (faster)
pytest tests/unit/ -n auto -v

# E2e tests — auto-skipped if model not present
# Download first: huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/models/Qwen3-0.6B
MODEL_PATH=~/models/Qwen3-0.6B pytest tests/e2e/ -v -s

# Check JAX device visibility
python -c "import jax; print(jax.devices())"
# GPU: [CudaDevice(id=0), ...]
# CPU: [CpuDevice(id=0)]
```

---

## Local Setup (without Docker)

```bash
python3.11 -m venv .venv && source .venv/bin/activate

# NVIDIA GPU (CUDA 12)
pip install -e ".[dev]"

# CPU only
JAX_PLATFORMS=cpu pip install -e ".[dev]"

# Apple Silicon
pip install jax-metal && pip install -e ".[dev]"
```

---

## Quick API Example

```python
from nanovllm_jax.llm import LLM
from nanovllm_jax.sampling_params import SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",   # HF hub ID or local path
    num_gpu_blocks=512,
    block_size=16,
    eos_token_id=151645,       # Qwen3 <|im_end|>
)
outputs = llm.generate(
    ["The capital of France is"],
    SamplingParams(temperature=0.0, max_tokens=64),
)
print(outputs[0].text)
```

---

## Fixed Bugs

### Bug #1: KV Cache Not Persisted Across JIT Boundary (Critical)

**Symptom:** All output tokens are garbled / random-looking text regardless of the prompt.

**Root cause:** `PagedKVCache.write()` mutates `self.cache` (an `nnx.Variable`) inside a
`jax.jit`-compiled forward function. JAX JIT compiles **pure functions** — Python-level
object mutations inside a JIT call are **not** propagated back to the outer scope after
the call returns. Every decode step read an all-zero KV cache, so attention over zero
value vectors produced near-zero output and every generated token was effectively random.

**Fix:** Replace `jax.jit` with `nnx.jit` in `model_runner.py`. `nnx.jit` is NNX-aware:
it extracts `nnx.Variable` state before the call, passes it through XLA as mutable arrays,
and writes updates back into the Python objects after each call.

See [MIGRATION.md](MIGRATION.md#1-jax-functional-purity-and-nnxjit-critical) for a
detailed explanation.

### Bug #2: `last_indices` Not Passed to `lm_head` During Prefill

**Symptom:** In multi-sequence prefill batches, the wrong logit rows are selected,
producing wrong next tokens.

**Fix:** Pass `last_indices` from `_run_prefill` through `_jit_prefill` into
`LlamaForCausalLM.__call__`. `ParallelLMHead` now receives the correct slice indices and
returns shape `[num_seqs, vocab_size]`.

### Bug #3: `seq_lens` Padding with `val=1` in Decode

**Symptom:** Padding sequences in a padded decode batch could attend to position 0 of the
KV cache, potentially contaminating real sequences.

**Fix:** Pad `seq_lens` with `val=0` so padding sequences have no valid context and all
their attention logits become `-inf`.

---

## Status

See [MIGRATION_PLAN.md](./MIGRATION_PLAN.md) for detailed per-module progress.

| Phase | Content | Status |
|---|---|---|
| 0 | Project scaffold, CI | ✅ Done |
| 1 | Stateless layers (Norm, RoPE, Activation) | ✅ Done |
| 2 | Linear layers + TP wrappers | ✅ Done |
| 3 | PagedKVCache + Attention + Sampler | ✅ Done |
| 4 | Full LLaMA model | ✅ Done |
| 5 | HuggingFace weight loader + registry | ✅ Done |
| 6 | Engine (Sequence, BlockManager, Scheduler, ModelRunner) | ✅ Done |
| 7 | LLMEngine + public `LLM` API | ✅ Done |
| Bug fixes | KV cache persistence, last_indices, seq_lens padding | ✅ Fixed |
| 8 | Flash Attention + multi-device benchmark | ⏳ Next |

---

## Architecture

```
nanovllm_jax/
  config.py              Engine / model configuration
  llm.py                 High-level LLM API
  sampling_params.py     SamplingParams dataclass
  engine/
    llm_engine.py        Ties scheduler + block manager + model runner
    model_runner.py      Builds JAX inputs, calls nnx.jit forward
    scheduler.py         FCFS continuous-batching scheduler
    block_manager.py     Paged KV cache block allocator
    sequence.py          Sequence / SequenceGroup dataclasses
  layers/
    attention.py         PagedKVCache + Attention (prefill & decode)
    rotary_embedding.py  RoPE (GPT-NeoX style)
    layernorm.py         RMSNorm with fused residual
    linear.py            ColumnParallel / RowParallel / QKV linear
    embed_head.py        VocabParallelEmbedding + ParallelLMHead
    activation.py        SiluAndMul
    sampler.py           Greedy / temperature / top-k / top-p
  models/
    llama.py             LlamaForCausalLM (also Qwen2/Qwen3)
  loader/
    weight_loader.py     HuggingFace safetensors weight loader
    model_registry.py    Architecture → class mapping
tests/
  unit/
    layers/              Per-layer unit tests
    engine/              Engine component tests (incl. KV cache regression)
  e2e/                   End-to-end tests
```

---

## Design Differences from PyTorch nano-vllm

See [MIGRATION.md](MIGRATION.md) for a full list. Key points:

1. **`nnx.jit` instead of `jax.jit`** — required for KV cache mutations to persist across
   call boundaries.
2. **Functional tensor ops** — `tensor.at[i].set(val)` instead of `tensor[i] = val`.
3. **No `torch.no_grad()`** — JAX is functional; gradients are opt-in via `jax.grad`.
4. **bfloat16 via torch bridge** — `safetensors.numpy` returns bf16 as `uint16` bytes;
   we load via `safetensors.torch` and reinterpret.
