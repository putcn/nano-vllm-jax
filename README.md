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

## Development Environments

Two Docker images are provided: one for **GPU** (CUDA 12 + JAX GPU backend) and one for **CPU** (lightweight, good for CI and unit tests without a GPU).

### Prerequisites

| | GPU image | CPU image |
|---|---|---|
| Docker | ✅ required | ✅ required |
| [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) | ✅ required | ❌ not needed |
| NVIDIA driver ≥ 525 | ✅ required | ❌ not needed |

Verify your GPU setup before building:
```bash
nvidia-smi          # should show your GPU
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## Docker Quick Start

### Option A — docker compose (recommended)

```bash
# Clone the repo
git clone https://github.com/putcn/nano-vllm-jax.git
cd nano-vllm-jax

# Build + start GPU container (interactive shell)
docker compose -f docker/docker-compose.yml up -d dev-gpu
docker compose -f docker/docker-compose.yml exec dev-gpu bash

# Build + start CPU container
docker compose -f docker/docker-compose.yml up -d dev-cpu
docker compose -f docker/docker-compose.yml exec dev-cpu bash
```

### Option B — docker build manually

```bash
# GPU image
docker build -f docker/Dockerfile.gpu -t nano-vllm-jax:gpu .

# CPU image
docker build -f docker/Dockerfile.cpu -t nano-vllm-jax:cpu .
```

### Running containers

```bash
# GPU — interactive dev shell with all GPUs
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v hf_cache:/workspace/.cache/huggingface \
  nano-vllm-jax:gpu bash

# GPU — specific GPU (e.g. only GPU 0)
docker run --gpus '"device=0"' -it --rm \
  -v $(pwd):/workspace \
  nano-vllm-jax:gpu bash

# CPU — interactive dev shell
docker run -it --rm \
  -v $(pwd):/workspace \
  nano-vllm-jax:cpu bash
```

---

## Running Tests Inside Docker

Once inside the container (either GPU or CPU), your local code is mounted at `/workspace`, so edits on the host are reflected instantly.

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run tests for a specific module
pytest tests/unit/layers/test_activation.py -v
pytest tests/unit/layers/test_layernorm.py -v
pytest tests/unit/layers/test_rotary_embedding.py -v

# Run in parallel (faster)
pytest tests/unit/ -n auto -v

# Run e2e tests (requires HuggingFace model weights)
pytest tests/e2e/ -v

# Verify JAX sees the GPU
python -c "import jax; print(jax.devices())"
# GPU output: [CudaDevice(id=0), CudaDevice(id=1), ...]
# CPU output: [CpuDevice(id=0)]
```

---

## Local Setup (without Docker)

```bash
# Requires Python 3.11+
python3.11 -m venv .venv
source .venv/bin/activate

# GPU (CUDA 12)
pip install -e ".[dev]"

# CPU only
JAX_PLATFORMS=cpu pip install -e ".[dev]"
```

---

## Docker Tips

**Persist HuggingFace model cache** across container rebuilds:
```bash
# The docker-compose.yml already sets this up via the hf_cache volume.
# For manual docker run, mount a local directory:
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v $HOME/.cache/huggingface:/workspace/.cache/huggingface \
  nano-vllm-jax:gpu bash
```

**Rebuild after dependency changes** (e.g. new package added to `pyproject.toml`):
```bash
docker compose -f docker/docker-compose.yml build dev-gpu
# or
docker build --no-cache -f docker/Dockerfile.gpu -t nano-vllm-jax:gpu .
```

**Control VRAM pre-allocation** (useful when sharing GPU with other processes):
```bash
# Already set in Dockerfile.gpu; override at runtime:
docker run --gpus all -it --rm \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
  -v $(pwd):/workspace \
  nano-vllm-jax:gpu bash
```

---

## Quick Example

```python
from nanovllm_jax import LLM, SamplingParams

llm = LLM("meta-llama/Llama-3.2-1B")
outputs = llm.generate(["Hello, my name is"], SamplingParams(max_tokens=50))
print(outputs[0].text)
```

---

## Status

See [MIGRATION_PLAN.md](./MIGRATION_PLAN.md) for detailed per-module progress.
