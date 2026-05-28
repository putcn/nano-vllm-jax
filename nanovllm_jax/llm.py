"""High-level LLM class: drop-in replacement for nano-vllm's LLM.

Status: ✅ Done (Phase 7)

Usage::

    from nanovllm_jax.llm import LLM
    from nanovllm_jax.sampling_params import SamplingParams

    llm = LLM(model="meta-llama/Llama-3.2-1B", num_gpu_blocks=512)
    outputs = llm.generate(
        ["Tell me a joke", "What is JAX?"],
        SamplingParams(temperature=0.8, max_tokens=128),
    )
    for o in outputs:
        print(o.text)
"""
from __future__ import annotations
from typing import List, Optional, Union

from nanovllm_jax.config import EngineConfig
from nanovllm_jax.sampling_params import SamplingParams
from nanovllm_jax.engine.llm_engine import LLMEngine, RequestOutput


class LLM:
    """Public-facing LLM class matching the nano-vllm API.

    Args:
        model: HuggingFace model name or local path.
        **kwargs: Forwarded to ``EngineConfig`` (e.g. ``num_gpu_blocks``,
                  ``block_size``, ``max_model_len``, ``dtype``).
    """

    def __init__(self, model: str, **kwargs):
        config = EngineConfig(model=model, **kwargs)
        self.engine = LLMEngine(config)

    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        *,
        prompt_token_ids: Optional[List[List[int]]] = None,
    ) -> List[RequestOutput]:
        """Generate completions for one or more prompts.

        Args:
            prompts: A single string or list of strings.
            sampling_params: Shared params applied to all prompts.
                             Defaults to greedy decoding with max_tokens=256.
            prompt_token_ids: Optional pre-tokenised IDs per prompt.
                              Must have the same length as *prompts* if provided.

        Returns:
            List of ``RequestOutput`` in the same order as *prompts*.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if sampling_params is None:
            sampling_params = SamplingParams()
        if prompt_token_ids is None:
            prompt_token_ids = [None] * len(prompts)  # type: ignore[list-item]

        req_ids: List[str] = []
        for prompt, pids in zip(prompts, prompt_token_ids):
            rid = self.engine.add_request(
                prompt, sampling_params, prompt_token_ids=pids
            )
            req_ids.append(rid)

        self.engine.generate_all()

        return [self.engine.get_outputs(rid) for rid in req_ids]
