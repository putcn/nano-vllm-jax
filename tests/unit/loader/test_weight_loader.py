"""Unit tests for weight_loader.py bfloat16 handling.

Covers:
- bfloat16 weights are loaded with correct numerical values (not garbage)
- float32 weights load correctly
- load_hf_config raises FileNotFoundError for missing config.json
- _shard_files raises FileNotFoundError when no weight files exist
"""
import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers to create minimal fake checkpoint directories
# ---------------------------------------------------------------------------

def _write_safetensors_bf16(path: Path, name: str, data_f32: np.ndarray):
    """Write a single bf16 tensor as a minimal safetensors file."""
    # Convert float32 -> bfloat16 via torch (which stores real bf16 bits)
    import torch
    t = torch.tensor(data_f32).to(torch.bfloat16)
    from safetensors.torch import save_file
    save_file({name: t}, str(path))


def _make_fake_checkpoint(tmp_dir: Path, dtype: str = "bfloat16") -> Path:
    """Create a minimal HF checkpoint with one weight tensor."""
    ckpt = tmp_dir / "model"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps({"model_type": "test"}))

    data = np.array([[1.0, 2.0, 3.0, 4.0],
                     [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    if dtype == "bfloat16":
        _write_safetensors_bf16(ckpt / "model.safetensors", "weight", data)
    else:
        import torch
        from safetensors.torch import save_file
        save_file({"weight": torch.tensor(data).float()},
                  str(ckpt / "model.safetensors"))
    return ckpt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadHfConfig:
    def test_reads_config(self, tmp_path):
        from nanovllm_jax.loader.weight_loader import load_hf_config
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({"hidden_size": 512}))
        cfg = load_hf_config(tmp_path)
        assert cfg["hidden_size"] == 512

    def test_missing_config_raises(self, tmp_path):
        from nanovllm_jax.loader.weight_loader import load_hf_config
        with pytest.raises(FileNotFoundError):
            load_hf_config(tmp_path)


class TestShardFiles:
    def test_single_safetensors(self, tmp_path):
        from nanovllm_jax.loader.weight_loader import _shard_files
        (tmp_path / "model.safetensors").touch()
        shards = _shard_files(tmp_path)
        assert len(shards) == 1
        assert shards[0].name == "model.safetensors"

    def test_no_weights_raises(self, tmp_path):
        from nanovllm_jax.loader.weight_loader import _shard_files
        with pytest.raises(FileNotFoundError):
            _shard_files(tmp_path)


class TestBfloat16Loading:
    """The critical fix: bf16 weights must load with correct numerical values."""

    def test_bf16_values_are_correct(self, tmp_path):
        """After loading, the tensor values must match the original float32 values
        within bfloat16 precision (~0.5% relative error)."""
        import jax.numpy as jnp
        from nanovllm_jax.loader.weight_loader import _load_shard

        expected = np.array([[1.0, 2.0, 3.0, 4.0],
                              [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
        st_path = tmp_path / "model.safetensors"
        _write_safetensors_bf16(st_path, "weight", expected)

        result = _load_shard(st_path, target_dtype="bfloat16", verbose=False)
        got = np.array(result["weight"], dtype=np.float32)

        # bfloat16 has ~0.5% relative error vs float32
        np.testing.assert_allclose(got, expected, rtol=0.01,
            err_msg="bfloat16 weight values are wrong — likely uint16 miscast")

    def test_bf16_not_uint16_garbage(self, tmp_path):
        """Explicitly check that the old broken path (uint16 cast) is NOT happening."""
        import jax.numpy as jnp
        from nanovllm_jax.loader.weight_loader import _load_shard

        # A float value of 1.0 in bfloat16 has bits 0x3F80.
        # If misinterpreted as uint16 -> float32, it becomes 16256.0
        expected = np.array([1.0], dtype=np.float32)
        st_path = tmp_path / "model.safetensors"
        _write_safetensors_bf16(st_path, "w", expected)

        result = _load_shard(st_path, target_dtype="bfloat16", verbose=False)
        got = float(np.array(result["w"], dtype=np.float32).flat[0])

        assert abs(got - 1.0) < 0.1, (
            f"Expected ~1.0 but got {got}. "
            f"This looks like a uint16->float32 miscast (expected 16256.0 for that bug)."
        )

    def test_float32_loading(self, tmp_path):
        import jax.numpy as jnp
        from nanovllm_jax.loader.weight_loader import _load_shard
        import torch
        from safetensors.torch import save_file

        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        st_path = tmp_path / "model.safetensors"
        save_file({"w": torch.tensor(expected)}, str(st_path))

        result = _load_shard(st_path, target_dtype="float32", verbose=False)
        got = np.array(result["w"], dtype=np.float32)
        np.testing.assert_allclose(got, expected, rtol=1e-6)
