"""Phase 0 smoke test: verify package imports correctly."""


def test_import_package():
    """Package must be importable without errors."""
    import nanovllm_jax  # noqa: F401


def test_import_config():
    from nanovllm_jax.config import ModelConfig, CacheConfig, EngineConfig
    cfg = ModelConfig(model="test-model")
    assert cfg.tokenizer == "test-model"
    assert cfg.dtype == "bfloat16"


def test_import_sampling_params():
    from nanovllm_jax.sampling_params import SamplingParams
    sp = SamplingParams(temperature=0.8, max_tokens=32)
    assert sp.temperature == 0.8
    assert sp.max_tokens == 32


def test_sampling_params_validation():
    import pytest
    from nanovllm_jax.sampling_params import SamplingParams
    with pytest.raises(AssertionError):
        SamplingParams(top_p=1.5)  # invalid
    with pytest.raises(AssertionError):
        SamplingParams(max_tokens=0)  # invalid
