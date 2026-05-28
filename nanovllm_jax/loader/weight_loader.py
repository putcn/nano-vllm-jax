"""HuggingFace weight loader (Phase 5).

Supports:
- safetensors shards (model-00001-of-NNNNN.safetensors)
- pytorch_model.bin shards
- single model.safetensors / pytorch_model.bin

Public API
----------
- load_hf_config(model_dir)       -- read config.json -> raw dict
- _shard_files(model_dir)         -- list of shard Paths in load order
- load_hf_weights(model, dir)     -- load all shards into model
- llama_config_from_hf(model_dir) -- Llama / Mistral / CodeLlama
- qwen_config_from_hf(model_dir)  -- Qwen2 / Qwen3

Bfloat16 handling
-----------------
numpy has no native bfloat16.  ``safetensors.numpy`` returns bf16 tensors as
``uint16`` raw bytes, so casting via numpy loses the values entirely.

Instead we use ``safetensors.torch`` (torch tensors keep real bf16) and
then convert to a JAX array with the correct dtype via::

    jnp.array(torch_tensor.numpy(), dtype=jnp.bfloat16)

For float32 weights we stay on the numpy path (faster, no torch needed).
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import json


# ---------------------------------------------------------------------------
# load_hf_config
# ---------------------------------------------------------------------------

def load_hf_config(model_dir: str | Path) -> dict:
    """Read and return the raw config.json as a dict.

    Raises:
        FileNotFoundError: if config.json does not exist.
    """
    model_dir = Path(model_dir)
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(cfg_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# _shard_files
# ---------------------------------------------------------------------------

def _shard_files(model_dir: Path) -> List[Path]:
    """Return an ordered list of shard file Paths (safetensors preferred).

    Raises:
        FileNotFoundError: if no weight files are found.
    """
    model_dir = Path(model_dir)
    index_st   = model_dir / "model.safetensors.index.json"
    index_bin  = model_dir / "pytorch_model.bin.index.json"
    single_st  = model_dir / "model.safetensors"
    single_bin = model_dir / "pytorch_model.bin"

    if index_st.exists():
        with open(index_st) as f:
            mapping = json.load(f)["weight_map"]
        seen: set[str] = set()
        result: List[Path] = []
        for fname in mapping.values():
            if fname not in seen:
                seen.add(fname)
                result.append(model_dir / fname)
        return result

    if index_bin.exists():
        with open(index_bin) as f:
            mapping = json.load(f)["weight_map"]
        seen = set()
        result = []
        for fname in mapping.values():
            if fname not in seen:
                seen.add(fname)
                result.append(model_dir / fname)
        return result

    if single_st.exists():
        return [single_st]

    if single_bin.exists():
        return [single_bin]

    raise FileNotFoundError(
        f"No weight files found in {model_dir}. "
        "Expected model.safetensors[.index.json] or pytorch_model.bin[.index.json]."
    )


# Keep old generator alias for internal compatibility
_iter_shard_paths = lambda d: iter(_shard_files(Path(d)))


# ---------------------------------------------------------------------------
# _load_shard  (dtype-aware)
# ---------------------------------------------------------------------------

def _load_shard(shard_path: Path, target_dtype: str, verbose: bool) -> dict:
    """Load one shard and return {name: numpy_array} in the correct dtype.

    Bfloat16 path
    ~~~~~~~~~~~~~
    numpy has no native bfloat16; ``safetensors.numpy`` silently returns bf16
    weights as ``uint16`` raw bytes.  Casting those bytes to float32 produces
    garbage.  We therefore use ``safetensors.torch`` which preserves true
    bfloat16 tensor values, then reinterpret via JAX.

    Float32 / other path
    ~~~~~~~~~~~~~~~~~~~~
    Plain numpy path — no torch import required.
    """
    import numpy as np
    import jax.numpy as jnp

    tensors_raw: dict  # str -> numpy array or torch tensor

    if shard_path.suffix == ".safetensors":
        # Always load via torch to preserve bfloat16 correctly.
        # torch is already a transitive dep (safetensors, transformers).
        from safetensors.torch import load_file as load_st_torch
        torch_tensors = load_st_torch(str(shard_path))
        tensors_raw = {k: v for k, v in torch_tensors.items()}
    else:
        import torch
        raw = torch.load(str(shard_path), map_location="cpu")
        tensors_raw = {k: v for k, v in raw.items()}

    result = {}
    for name, t in tensors_raw.items():
        if verbose:
            print(f"    {name}: {tuple(t.shape)} {t.dtype}")

        # Convert torch tensor -> JAX array with correct dtype.
        # t.numpy() on a bf16 tensor gives a uint16 view, but
        # jnp.array(..., dtype=jnp.bfloat16) reinterprets it correctly
        # because JAX and torch share the same bf16 bit layout.
        import torch
        if t.dtype == torch.bfloat16:
            arr = jnp.array(t.view(torch.int16).numpy()).view(jnp.bfloat16)
        else:
            arr = t.numpy()

        # Cast to target dtype
        if target_dtype == "bfloat16":
            arr = jnp.array(arr, dtype=jnp.bfloat16)
        elif target_dtype == "float32":
            arr = jnp.array(arr, dtype=jnp.float32)
        # else: keep as-is (e.g. int8 quantised weights in future)

        result[name] = arr

    return result


# ---------------------------------------------------------------------------
# load_hf_weights
# ---------------------------------------------------------------------------

def load_hf_weights(
    model,
    model_dir: str | Path,
    *,
    dtype: str = "bfloat16",
    verbose: bool = False,
) -> None:
    """Load all shards into *model* via ``model.load_weights(params_dict)``.

    Args:
        model:     An instance with a ``load_weights(dict)`` method.
        model_dir: HuggingFace checkpoint directory.
        dtype:     Target compute dtype: ``'bfloat16'`` (default) or ``'float32'``.
        verbose:   Print each tensor name as it is loaded.
    """
    model_dir = Path(model_dir)
    all_params: dict = {}

    for shard_path in _shard_files(model_dir):
        if verbose:
            print(f"  loading shard: {shard_path.name}")
        shard = _load_shard(shard_path, target_dtype=dtype, verbose=verbose)
        all_params.update(shard)

    model.load_weights(all_params)


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def llama_config_from_hf(model_dir: str | Path):
    """Build LlamaConfig from a HuggingFace Llama / Mistral config.json."""
    from nanovllm_jax.models.llama import LlamaConfig
    cfg = load_hf_config(model_dir)
    return LlamaConfig(
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        head_dim=cfg.get("head_dim"),
        max_position_embeddings=cfg.get("max_position_embeddings", 4096),
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-5),
        vocab_size=cfg["vocab_size"],
        rope_theta=cfg.get("rope_theta", 10000.0),
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
    )


def qwen_config_from_hf(model_dir: str | Path):
    """Build LlamaConfig from a HuggingFace Qwen2 / Qwen3 config.json.

    Key differences vs Llama defaults:
    - head_dim explicit (128 for most Qwen3 sizes)
    - rms_norm_eps = 1e-6
    - rope_theta = 1_000_000
    """
    from nanovllm_jax.models.llama import LlamaConfig
    cfg = load_hf_config(model_dir)
    num_heads = cfg["num_attention_heads"]
    num_kv    = cfg.get("num_key_value_heads", num_heads)
    hidden    = cfg["hidden_size"]
    head_dim  = cfg.get("head_dim", hidden // num_heads)
    return LlamaConfig(
        hidden_size=hidden,
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv,
        head_dim=head_dim,
        max_position_embeddings=cfg.get("max_position_embeddings", 32768),
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
        vocab_size=cfg["vocab_size"],
        rope_theta=cfg.get("rope_theta", 1_000_000.0),
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
    )
