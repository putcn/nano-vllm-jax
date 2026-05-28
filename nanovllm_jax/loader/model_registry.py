"""Model registry: maps HuggingFace architectures to nano-vllm-jax classes.

Status: ✅ Done (Phase 5)

Supported architectures
-----------------------
- LlamaForCausalLM   (Llama 2/3, CodeLlama, etc.)
- MistralForCausalLM (Mistral 7B, Mixtral via dense view)
- Qwen3ForCausalLM   (Qwen3-0.6B / 1.7B / 4B / 8B / 14B / 32B)
- Qwen2ForCausalLM   (Qwen2-0.5B / 1.5B / 7B / 72B)

All Qwen2/Qwen3 models share the LlamaForCausalLM backbone (identical
architecture: RMSNorm + GQA + SwiGLU + RoPE) with a compatible weight
key layout.  They only differ in config field names, handled by
``qwen_config_from_hf``.

Usage::

    from nanovllm_jax.loader.model_registry import load_model
    model = load_model("/path/to/hf/model")  # auto-detects architecture
"""
from __future__ import annotations
from pathlib import Path
from typing import Type


_REGISTRY: dict[str, tuple] = {}


def register(hf_arch: str):
    """Decorator to register a (model_cls, config_builder) pair."""
    def decorator(fn):
        _REGISTRY[hf_arch] = fn
        return fn
    return decorator


@register("LlamaForCausalLM")
def _llama():
    from nanovllm_jax.models.llama import LlamaForCausalLM
    from nanovllm_jax.loader.weight_loader import llama_config_from_hf
    return LlamaForCausalLM, llama_config_from_hf


@register("MistralForCausalLM")
def _mistral():
    from nanovllm_jax.models.llama import LlamaForCausalLM
    from nanovllm_jax.loader.weight_loader import llama_config_from_hf
    return LlamaForCausalLM, llama_config_from_hf


@register("Qwen2ForCausalLM")
def _qwen2():
    from nanovllm_jax.models.llama import LlamaForCausalLM
    from nanovllm_jax.loader.weight_loader import qwen_config_from_hf
    return LlamaForCausalLM, qwen_config_from_hf


@register("Qwen3ForCausalLM")
def _qwen3():
    from nanovllm_jax.models.llama import LlamaForCausalLM
    from nanovllm_jax.loader.weight_loader import qwen_config_from_hf
    return LlamaForCausalLM, qwen_config_from_hf


def get_model_class(hf_arch: str):
    if hf_arch not in _REGISTRY:
        supported = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unsupported architecture: {hf_arch!r}. Supported: {supported}"
        )
    model_cls, config_fn = _REGISTRY[hf_arch]()
    return model_cls, config_fn


def load_model(
    model_dir: str | Path,
    *,
    dtype: str = "float32",
    verbose: bool = False,
):
    """Auto-detect architecture from config.json and return an initialized model."""
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
