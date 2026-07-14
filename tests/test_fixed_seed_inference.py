import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utils.fixed_seed_inference import (
    DETERMINISM_POLICY_VERSION,
    REQUIRED_CUBLAS_WORKSPACE_CONFIG,
    SEED_POLICY_VERSION,
    SEED_POLICY_SOURCE_COMMIT,
    deterministic_inference_algorithms,
    fixed_seed_processor_generation,
    reset_rng,
    rng_state_fingerprints,
    seed_fingerprint,
    stage_seed,
)
from utils.run_fixed_seed_inference import _target_repo


def test_stage_seed_is_stable_and_label_specific() -> None:
    assert stage_seed(12345, "timing_context") == stage_seed(12345, "timing_context")
    assert stage_seed(12345, "timing_context") != stage_seed(12345, "main_generation")
    assert stage_seed(12345, "main_generation") != stage_seed(12346, "main_generation")
    with pytest.raises(TypeError, match="integer"):
        stage_seed(True, "main_generation")
    with pytest.raises(ValueError, match="non-empty"):
        stage_seed(12345, "")


def test_target_repo_requires_inference_and_config_tree(tmp_path) -> None:
    with pytest.raises(ValueError, match="lacks inference.py"):
        _target_repo(tmp_path)
    (tmp_path / "inference.py").write_text("", encoding="utf-8")
    (tmp_path / "configs" / "inference").mkdir(parents=True)
    assert _target_repo(tmp_path) == tmp_path.resolve()


def test_reset_rng_repeats_cpu_torch_and_numpy() -> None:
    reset_rng(717)
    first = (torch.rand(3), np.random.random(3))
    reset_rng(717)
    second = (torch.rand(3), np.random.random(3))
    assert torch.equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    reset_rng(717)
    first_fingerprint = rng_state_fingerprints()["cpu"]
    torch.rand(1)
    assert rng_state_fingerprints()["cpu"] != first_fingerprint
    reset_rng(717)
    assert rng_state_fingerprints()["cpu"] == first_fingerprint


def test_deterministic_algorithm_context_restores_global_policy(monkeypatch) -> None:
    monkeypatch.setenv(
        "CUBLAS_WORKSPACE_CONFIG",
        REQUIRED_CUBLAS_WORKSPACE_CONFIG,
    )
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic

    with deterministic_inference_algorithms():
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True

    assert torch.are_deterministic_algorithms_enabled() is previous_enabled
    assert torch.is_deterministic_algorithms_warn_only_enabled() is previous_warn_only
    assert torch.backends.cudnn.benchmark is previous_benchmark
    assert torch.backends.cudnn.deterministic is previous_cudnn_deterministic


def test_processor_patch_reseeds_each_stage_and_restores() -> None:
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
    instance.profiler = SimpleNamespace(
        set_metadata=lambda **values: metadata.update(values)
    )

    original_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = REQUIRED_CUBLAS_WORKSPACE_CONFIG
    try:
        with deterministic_inference_algorithms():
            with fixed_seed_processor_generation(module, base_seed=12345):
                instance.generate(profile_label="main_generation")
                torch.rand(10)
                instance.generate(profile_label="main_generation")
                instance.generate(profile_label="timing_context")
                with pytest.raises(RuntimeError, match="explicit profile_label"):
                    instance.generate()
    finally:
        if original_cublas is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = original_cublas

    assert Processor.generate is original
    assert calls[0][0] == calls[1][0] == "main_generation"
    assert torch.equal(calls[0][1], calls[1][1])
    assert not torch.equal(calls[0][1], calls[2][1])
    assert metadata["reciprocal_seed_policy"] == SEED_POLICY_VERSION
    assert metadata["stage_seed_policy"] == SEED_POLICY_VERSION
    assert metadata["stage_seed_policy_source_commit"] == SEED_POLICY_SOURCE_COMMIT
    assert metadata["deterministic_algorithm_policy"] == DETERMINISM_POLICY_VERSION
    assert metadata["torch_deterministic_algorithms"] is True
    assert metadata["cublas_workspace_config"] == REQUIRED_CUBLAS_WORKSPACE_CONFIG
    assert metadata["reciprocal_seed_main_generation"] == stage_seed(
        12345, "main_generation"
    )
    assert metadata["reciprocal_seed_timing_context"] == stage_seed(
        12345, "timing_context"
    )
    assert len(metadata["reciprocal_rng_cpu_sha256_main_generation"]) == 64
    stage_fingerprints = metadata["profile_label_seed_fingerprints"]
    assert set(stage_fingerprints) == {"timing_context", "main_generation"}
    main = stage_fingerprints["main_generation"]
    main_seed = stage_seed(12345, "main_generation")
    assert main["seed"] == main_seed
    assert main["seed_sha256"] == seed_fingerprint(
        12345,
        "main_generation",
        main_seed,
    )
    assert len(main["rng_state_sha256"]["cpu"]) == 64
