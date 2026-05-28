"""Model and engine configuration (JAX port of nanovllm/config.py)."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model: str
    tokenizer: Optional[str] = None
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    enforce_eager: bool = False

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model


@dataclass
class CacheConfig:
    block_size: int = 16
    num_gpu_blocks: int = 0
    max_num_seqs: int = 256


@dataclass
class EngineConfig:
    """Flat engine config used by Scheduler, BlockManager, and ModelRunner."""
    # ---- Model fields ----
    model: str = ""
    tokenizer: Optional[str] = None
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    enforce_eager: bool = False

    # ---- Cache / scheduling fields ----
    block_size: int = 16
    num_gpu_blocks: int = 0
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 2048

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model

    @classmethod
    def from_configs(cls, model_config: ModelConfig,
                     cache_config: Optional[CacheConfig] = None) -> "EngineConfig":
        cc = cache_config or CacheConfig()
        return cls(
            model=model_config.model,
            tokenizer=model_config.tokenizer,
            dtype=model_config.dtype,
            max_model_len=model_config.max_model_len,
            enforce_eager=model_config.enforce_eager,
            block_size=cc.block_size,
            num_gpu_blocks=cc.num_gpu_blocks,
            max_num_seqs=cc.max_num_seqs,
        )

    @property
    def model_config(self) -> ModelConfig:
        return ModelConfig(
            model=self.model,
            tokenizer=self.tokenizer,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            enforce_eager=self.enforce_eager,
        )

    @property
    def cache_config(self) -> CacheConfig:
        return CacheConfig(
            block_size=self.block_size,
            num_gpu_blocks=self.num_gpu_blocks,
            max_num_seqs=self.max_num_seqs,
        )
