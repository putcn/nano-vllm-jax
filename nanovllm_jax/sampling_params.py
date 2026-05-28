"""Sampling parameters (JAX port of nanovllm/sampling_params.py)."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SamplingParams:
    temperature: float = 0.0    # 0 = greedy
    top_p: float = 1.0
    top_k: int = 0              # 0 = disabled
    max_tokens: int = 256
    stop_token_ids: List[int] = field(default_factory=list)
    ignore_eos: bool = False

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_p <= 0:
            raise ValueError(f"top_p must be > 0, got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
