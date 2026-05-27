"""Sampling parameters (JAX port of nanovllm/sampling_params.py)."""
from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 = disabled
    max_tokens: int = 16
    stop: list[str] = None

    def __post_init__(self):
        if self.stop is None:
            self.stop = []
        assert 0.0 <= self.top_p <= 1.0, "top_p must be in [0, 1]"
        assert self.max_tokens > 0, "max_tokens must be positive"
