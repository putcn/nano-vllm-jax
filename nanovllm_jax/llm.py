"""Public LLM entry point (stub — implemented in Phase 6)."""
from nanovllm_jax.sampling_params import SamplingParams


class LLM:
    """High-level LLM inference API, mirrors nano-vllm's LLM class."""

    def __init__(self, model: str, **kwargs):
        raise NotImplementedError("LLM entry point will be implemented in Phase 6")

    def generate(self, prompts: list[str], sampling_params: SamplingParams):
        raise NotImplementedError
