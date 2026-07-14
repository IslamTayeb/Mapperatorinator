from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from utils.fixed_seed_inference import (
    SEED_POLICY_VERSION,
    fixed_seed_processor_generation,
    stage_seed,
)
from utils.run_batched_encoder_store_full_song import _validate_args


class _Profiler:
    def __init__(self):
        self.metadata = {}

    def set_metadata(self, **kwargs):
        self.metadata.update(kwargs)


class _Processor:
    def __init__(self):
        self.profiler = _Profiler()

    def generate(self, *, profile_label):
        return torch.rand(3)


def _args(**overrides):
    values = {
        "inference_engine": "optimized",
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "device": "cuda",
        "use_server": False,
        "parallel": False,
        "cfg_scale": 1.0,
        "num_beams": 1,
        "super_timing": False,
        "generate_positions": False,
        "profile_inference": True,
        "max_batch_size": 32,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_stage_seed_is_stable_and_label_specific():
    assert stage_seed(12345, "timing_context") == stage_seed(
        12345, "timing_context"
    )
    assert stage_seed(12345, "timing_context") != stage_seed(
        12345, "main_generation"
    )


def test_fixed_seed_hook_restores_and_publishes_rng_evidence():
    module = SimpleNamespace(Processor=_Processor)
    original = _Processor.generate
    processor = _Processor()
    with fixed_seed_processor_generation(module, base_seed=12345):
        first = processor.generate(profile_label="main_generation")
        second = processor.generate(profile_label="main_generation")
        assert torch.equal(first, second)
    assert _Processor.generate is original
    assert processor.profiler.metadata["reciprocal_seed_policy"] == SEED_POLICY_VERSION
    assert processor.profiler.metadata["reciprocal_seed_main_generation"] == stage_seed(
        12345, "main_generation"
    )


def test_runner_rejects_nonaccepted_configs_and_batch_overflow():
    _validate_args(_args(), mode="candidate", batch_size=16)
    with pytest.raises(ValueError, match="args changed"):
        _validate_args(_args(precision="fp16"), mode="candidate", batch_size=16)
    with pytest.raises(ValueError, match="max_batch_size"):
        _validate_args(_args(max_batch_size=8), mode="candidate", batch_size=16)
