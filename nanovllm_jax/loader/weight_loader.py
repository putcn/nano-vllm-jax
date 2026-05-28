"""HuggingFace weight loader (Phase 5).

Supports:
- safetensors shards (model-00001-of-NNNNN.safetensors)
- pytorch_model.bin shards
- single model.safetensors / pytorch_model.bin

Also exports config builders:
- llama_config_from_hf   -- Llama / Mistral / CodeLlama
- qwen_config_from_hf    -- Qwen2 / Qwen3 (same weight key layout)
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator
import json


def _iter_shard_paths(model_dir: Path) -> Iterator[Path]:
    """Yield shard file paths in order (safetensors preferred over bin)."""
    index_st = model_dir / "model.safetensors.index.json"
    index_bin = model_dir / "pytorch_model.bin.index.json"
    single_st = model_dir / "model.safetensors"
    single_bin = model_dir / "pytorch_model.bin"

    if index_st.exists():
        with open(index_st) as f:
            mapping = json.load(f)["weight_map"]
        seen: set[str] = set()
        for fname in mapping.values():
            if fname not in seen:
                seen.add(fname)
                yield model_dir / fname
    elif index_bin.exists():
        with open(index_bin) as f:
            mapping = json.load(f)["weight_map"]
        seen = set()
        for fname in mapping.values():
            if fname not in seen:
                seen.add(fname)
                yield model_dir / fname
    elif single_st.exists():
        yield single_st
    elif single_bin.exists():
        yield single_bin
    else:
        raise FileNotFoundError(
            f"No weight files found in {model_dir}. "
            "Expected model.safetensors[.index.json] or pytorch_model.bin[.index.json]."
        )


def load_hf_weights(
    model,
    model_dir: str | Path,
    *,
    dtype: str = "float32",
    verbose: bool = False,
) -> None:
    """Load all shards into *model* via ``model.load_weights(params_dict)``.

    Args:
        model:     An instance with a ``load_weights(dict)`` method.
        model_dir: HuggingFace checkpoint directory.
        dtype:     Target dtype (``'float32'`` or ``'bfloat16'``).
        verbose:   Print each tensor name as it is loaded.
    """
    import numpy as np
    model_dir = Path(model_dir)
    all_params: dict = {}

    for shard_path in _iter_shard_paths(model_dir):
        if verbose:
            print(f"  loading shard: {shard_path.name}")
        if shard_path.suffix == ".safetensors":
            from safetensors.numpy import load_file
            tensors = load_file(str(shard_path))
        else:
            import torch
            raw = torch.load(str(shard_path), map_location="cpu")
            tensors = {k: v.numpy() for k, v in raw.items()}

        for name, arr in tensors.items():
            if verbose:
                print(f"    {name}: {arr.shape} {arr.dtype}")
            if dtype == "bfloat16":
                arr = arr.astype(np.float32)  # JAX doesn't read bf16 from numpy
            all_params[name] = arr

    model.load_weights(all_params)


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def llama_config_from_hf(model_dir: str | Path):
    """Build LlamaConfig from a HuggingFace Llama / Mistral config.json."""
    from nanovllm_jax.models.llama import LlamaConfig
    model_dir = Path(model_dir)
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

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

    Qwen2 / Qwen3 use the identical Llama-style weight key layout but their
    ``config.json`` may have slightly different field names.  We normalise
    them here and re-use ``LlamaConfig`` as the internal representation.

    Notable differences:
    - ``head_dim`` is explicit in Qwen3 (128 for most sizes)
    - ``rms_norm_eps`` is typically 1e-6 instead of 1e-5
    - ``rope_theta`` is 1_000_000 for Qwen3
    - ``tie_word_embeddings`` is False for all public Qwen3 checkpoints
    """
    from nanovllm_jax.models.llama import LlamaConfig
    model_dir = Path(model_dir)
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    hidden_size = cfg["hidden_size"]
    # Qwen3 exposes head_dim directly; fall back to hidden_size // num_heads
    head_dim = cfg.get("head_dim", hidden_size // num_heads)

    return LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=cfg.get("max_position_embeddings", 32768),
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
        vocab_size=cfg["vocab_size"],
        rope_theta=cfg.get("rope_theta", 1_000_000.0),
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
    )
