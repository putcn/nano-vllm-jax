"""Sampling parameters (JAX port of nanovllm/sampling_params.py)."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class SamplingParams:
    temperature: float = 0.0    # 0 = greedy
    top_p: float = 1.0
    top_k: int = 0              # 0 = disabled
    max_tokens: int = 256
    stop_token_ids: List[int] = field(default_factory=list)
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature >= 0, f"temperature must be >= 0, got {self.temperature}"
        assert 0 < self.top_p <= 1.0, f"top_p must be in (0, 1], got {self.top_p}"
        assert self.top_k >= 0, f"top_k must be >= 0, got {self.top_k}"
        assert self.max_tokens >= 1, f"max_tokens must be >= 1, got {self.max_tokens}"
