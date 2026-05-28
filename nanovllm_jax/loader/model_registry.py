"""Model registry: maps HuggingFace architectures to nano-vllm-jax classes.

Status: ✅ Done (Phase 5)

Usage::

    from nanovllm_jax.loader.model_registry import load_model
    model = load_model("/path/to/hf/model")  # auto-detects architecture
"""
from __future__ import annotations
from pathlib import Path
from typing import Type


# Registry: HF architectures -> (model_class, config_builder)
# Extend this dict to add new model families.
_REGISTRY: dict[str, tuple] = {}


def register(hf_arch: str):
    """Decorator to register a (model_cls, config_fn) pair.

    Usage::

        @register("LlamaForCausalLM")
        def _():
            from nanovllm_jax.models.llama import LlamaForCausalLM
            from nanovllm_jax.loader.weight_loader import llama_config_from_hf
            return LlamaForCausalLM, llama_config_from_hf
    """
    def decorator(fn):
        _REGISTRY[hf_arch] = fn
        return fn
    return decorator


@register("LlamaForCausalLM")
def _llama():
    from nanovllm_jax.models.llama import LlamaForCausalLM
    from nanovllm_jax.loader.weight_loader import llama_config_from_hf
    return LlamaForCausalLM, llama_config_from_hf


# Mistral uses Llama architecture with GQA
@register("MistralForCausalLM")
def _mistral():
    from nanovllm_jax.models.llama import LlamaForCausalLM
    from nanovllm_jax.loader.weight_loader import llama_config_from_hf
    return LlamaForCausalLM, llama_config_from_hf


def get_model_class(hf_arch: str):
    """Return (ModelClass, config_builder_fn) for the given HF architecture."""
    if hf_arch not in _REGISTRY:
        supported = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unsupported architecture: {hf_arch!r}. "
            f"Supported: {supported}"
        )
    model_cls, config_fn = _REGISTRY[hf_arch]()
    return model_cls, config_fn


def load_model(
    model_dir: str | Path,
    *,
    dtype: str = "float32",
    verbose: bool = False,
):
    """Auto-detect architecture from config.json and return an initialized model.

    Args:
        model_dir: HuggingFace checkpoint directory.
        dtype:     Weight dtype (``'float32'`` or ``'bfloat16'``).
        verbose:   Print loaded tensor names.

    Returns:
        Initialized model with weights loaded.
    """
    import json
    from nanovllm_jax.loader.weight_loader import load_hf_weights

    cfg_path = Path(model_dir) / "config.json"
    with open(cfg_path) as f:
        raw_cfg = json.load(f)

    architectures = raw_cfg.get("architectures", [])
    if not architectures:
        raise ValueError(f"No 'architectures' field in {cfg_path}")

    arch = architectures[0]
    model_cls, config_fn = get_model_class(arch)
    config = config_fn(model_dir)
    model = model_cls(config)
    load_hf_weights(model, model_dir, dtype=dtype, verbose=verbose)
    return model
