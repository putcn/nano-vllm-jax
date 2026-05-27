"""Model and engine configuration (JAX port of nanovllm/config.py)."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model: str
    tokenizer: Optional[str] = None
    dtype: str = "bfloat16"  # jax default
    max_model_len: int = 4096
    enforce_eager: bool = False  # if True, skip jax.jit

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model


@dataclass
class CacheConfig:
    block_size: int = 16
    num_gpu_blocks: int = 0  # 0 = auto-detect
    max_num_seqs: int = 256


@dataclass
class EngineConfig:
    model_config: ModelConfig
    cache_config: CacheConfig = field(default_factory=CacheConfig)
