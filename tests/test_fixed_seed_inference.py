from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utils.fixed_seed_inference import (
    SEED_POLICY_VERSION,
    fixed_seed_processor_generation,
    reset_rng,
    stage_seed,
)


def test_stage_seed_is_stable_and_label_specific() -> None:
    assert stage_seed(12345, "timing_context") == stage_seed(12345, "timing_context")
    assert stage_seed(12345, "timing_context") != stage_seed(12345, "main_generation")
    assert stage_seed(12345, "main_generation") != stage_seed(12346, "main_generation")
    with pytest.raises(TypeError, match="integer"):
        stage_seed(True, "main_generation")
    with pytest.raises(ValueError, match="non-empty"):
        stage_seed(12345, "")


def test_reset_rng_repeats_cpu_torch_and_numpy() -> None:
    reset_rng(717)
    first = (torch.rand(3), np.random.random(3))
    reset_rng(717)
    second = (torch.rand(3), np.random.random(3))
    assert torch.equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_processor_patch_reseeds_each_stage_and_restores(monkeypatch) -> None:
    calls = []

    class Processor:
        def generate(self, *args, **kwargs):
            del args
            calls.append((kwargs["profile_label"], torch.rand(2)))
            return kwargs["profile_label"]

    original = Processor.generate
    module = SimpleNamespace(Processor=Processor)
    metadata = {}
    instance = Processor()
    instance.profiler = SimpleNamespace(set_metadata=lambda **values: metadata.update(values))

    with fixed_seed_processor_generation(module, base_seed=12345):
        instance.generate(profile_label="main_generation")
        torch.rand(10)
        instance.generate(profile_label="main_generation")
        instance.generate(profile_label="timing_context")
        with pytest.raises(RuntimeError, match="explicit profile_label"):
            instance.generate()

    assert Processor.generate is original
    assert calls[0][0] == calls[1][0] == "main_generation"
    assert torch.equal(calls[0][1], calls[1][1])
    assert not torch.equal(calls[0][1], calls[2][1])
    assert metadata["reciprocal_seed_policy"] == SEED_POLICY_VERSION
    assert metadata["reciprocal_seed_main_generation"] == stage_seed(
        12345, "main_generation"
    )
    assert metadata["reciprocal_seed_timing_context"] == stage_seed(
        12345, "timing_context"
    )
