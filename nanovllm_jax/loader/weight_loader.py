"""HuggingFace weight loader for nano-vllm-jax.

Status: ✅ Done (Phase 5)

Supports:
  - safetensors (preferred, mmap-friendly)
  - pytorch_model.bin / model.bin (torch.load fallback)
  - Sharded checkpoints via model.safetensors.index.json
    or pytorch_model.bin.index.json

Usage::

    from nanovllm_jax.loader.weight_loader import load_hf_weights
    from nanovllm_jax.models.llama import LlamaForCausalLM, LlamaConfig

    config = LlamaConfig(...)
    model  = LlamaForCausalLM(config)
    load_hf_weights(model, "/path/to/hf/model")
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Shard iterator
# ---------------------------------------------------------------------------

def _iter_safetensors(path: Path) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (name, numpy_array) from a single .safetensors file."""
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise ImportError(
            "safetensors not installed. Run: pip install safetensors"
        ) from e
    with safe_open(str(path), framework="numpy") as f:
        for key in f.keys():
            yield key, f.get_tensor(key)


def _iter_pytorch_bin(path: Path) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (name, numpy_array) from a pytorch .bin file."""
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "torch not installed. Run: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from e
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    for key, tensor in state.items():
        yield key, tensor.float().numpy()


def _shard_files(model_dir: Path) -> list[Path]:
    """Return ordered list of weight shard files in model_dir."""
    # 1. Prefer safetensors
    index_st = model_dir / "model.safetensors.index.json"
    single_st = model_dir / "model.safetensors"
    if index_st.exists():
        with open(index_st) as f:
            index = json.load(f)
        # weight_map values are filenames; deduplicate while preserving order
        seen: dict[str, None] = {}
        for fname in index["weight_map"].values():
            seen[fname] = None
        return [model_dir / fname for fname in seen]
    if single_st.exists():
        return [single_st]

    # 2. Fallback: pytorch bin
    index_pt = model_dir / "pytorch_model.bin.index.json"
    single_pt = model_dir / "pytorch_model.bin"
    if index_pt.exists():
        with open(index_pt) as f:
            index = json.load(f)
        seen = {}
        for fname in index["weight_map"].values():
            seen[fname] = None
        return [model_dir / fname for fname in seen]
    if single_pt.exists():
        return [single_pt]

    raise FileNotFoundError(
        f"No weight files found in {model_dir}. "
        "Expected model.safetensors[.index.json] or pytorch_model.bin[.index.json]."
    )


def _iter_shards(model_dir: Path) -> Iterator[tuple[str, np.ndarray]]:
    """Iterate over all (name, array) pairs across all shards."""
    for shard_path in _shard_files(model_dir):
        if shard_path.suffix == ".safetensors":
            yield from _iter_safetensors(shard_path)
        else:
            yield from _iter_pytorch_bin(shard_path)


# ---------------------------------------------------------------------------
# Key remapping
# ---------------------------------------------------------------------------

# Some HF checkpoints use slightly different key prefixes.
# Add entries here as new model variants are supported.
_KEY_REMAP: list[tuple[str, str]] = [
    # Llama-3 uses the same keys, no remap needed.
    # ("old_prefix.", "new_prefix."),
]


def _remap_key(key: str) -> str:
    for old, new in _KEY_REMAP:
        if key.startswith(old):
            return new + key[len(old):]
    return key


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_hf_weights(
    model,
    model_dir: str | Path,
    *,
    dtype: str = "float32",
    verbose: bool = False,
) -> None:
    """Load HuggingFace weights from *model_dir* into *model* in-place.

    Args:
        model:     A model instance with a ``load_weights(params)`` method
                   (e.g. ``LlamaForCausalLM``).
        model_dir: Directory containing the HF checkpoint.
        dtype:     Cast weights to this numpy dtype before passing to JAX.
                   Default ``'float32'``; use ``'bfloat16'`` for large models.
        verbose:   Print each loaded tensor name and shape.
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise NotADirectoryError(f"{model_dir} is not a directory")

    params: dict[str, np.ndarray] = {}
    np_dtype = np.dtype(dtype)

    for name, arr in _iter_shards(model_dir):
        name = _remap_key(name)
        params[name] = arr.astype(np_dtype)
        if verbose:
            print(f"  loaded {name:80s} {arr.shape}")

    model.load_weights(params)
    if verbose:
        print(f"\nLoaded {len(params)} tensors from {model_dir}")


def load_hf_config(model_dir: str | Path) -> dict:
    """Read config.json from a HuggingFace model directory.

    Returns the raw dict; callers map fields to ``LlamaConfig`` themselves.
    """
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(cfg_path) as f:
        return json.load(f)


def llama_config_from_hf(model_dir: str | Path):
    """Build a ``LlamaConfig`` from a HuggingFace config.json.

    Supports Llama-1/2/3 config field names.
    """
    from nanovllm_jax.models.llama import LlamaConfig

    raw = load_hf_config(model_dir)
    return LlamaConfig(
        hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw.get("num_key_value_heads", raw["num_attention_heads"]),
        head_dim=raw.get("head_dim"),  # None = infer
        max_position_embeddings=raw["max_position_embeddings"],
        rms_norm_eps=raw.get("rms_norm_eps", 1e-5),
        vocab_size=raw["vocab_size"],
        rope_theta=raw.get("rope_theta", 10000.0),
        tie_word_embeddings=raw.get("tie_word_embeddings", False),
    )
