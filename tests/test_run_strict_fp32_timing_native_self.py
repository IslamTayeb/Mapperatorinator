from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.inference.engine_binding import InferenceEngineBinding
from osuT5.osuT5.inference.optimized.single import engine
from utils import run_strict_fp32_timing_native_self as runner


ROOT = Path(__file__).resolve().parents[1]


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    @property
    def dtype(self):
        return self.weight.dtype


def _binding() -> InferenceEngineBinding:
    return InferenceEngineBinding(
        raw_model=_Model(),
        runtime=engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS["fp32"]),
    )


def _strict_environment(monkeypatch) -> None:
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "highest")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)


def test_runner_wraps_only_distinct_timing_binding_and_restores_state(
    monkeypatch,
    tmp_path,
) -> None:
    _strict_environment(monkeypatch)
    bindings = [_binding(), _binding()]
    original_argv = sys.argv[:]
    load_index = 0

    def original_loader(*args, **kwargs):
        nonlocal load_index
        binding = bindings[load_index]
        load_index += 1
        return binding, object()

    def fake_main():
        inference.load_model_with_engine(auto_select_gamemode_model=True)
        inference.load_model_with_engine(auto_select_gamemode_model=False)

    inference = SimpleNamespace(
        load_model_with_engine=original_loader,
        main=fake_main,
    )
    monkeypatch.setitem(sys.modules, "inference", inference)
    output = tmp_path / "initialization.json"

    runner.run("profile_salvalai", ["seed=12345"], output)

    assert inference.load_model_with_engine is original_loader
    assert sys.argv == original_argv
    assert bindings[0].runtime._strict_fp32_timing_native_self_owner is None
    assert bindings[1].runtime._strict_fp32_timing_native_self_owner is (
        bindings[1].raw_model
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["model_loads"]["owners_distinct"] is True
    assert payload["strict_fp32"]["NVIDIA_TF32_OVERRIDE"] == "0"
    assert payload["timing_native_self"]["exactness_claim"] is True
    assert payload["timing_native_self"]["reduced_precision_weights"] is False
    assert payload["timing_native_self"]["counter_rng"] is False


def test_runner_restores_loader_and_argv_after_failure(monkeypatch, tmp_path) -> None:
    _strict_environment(monkeypatch)
    binding = _binding()
    original_argv = sys.argv[:]

    def original_loader(*args, **kwargs):
        return binding, object()

    def fake_main():
        inference.load_model_with_engine(auto_select_gamemode_model=True)
        raise RuntimeError("nested failure")

    inference = SimpleNamespace(
        load_model_with_engine=original_loader,
        main=fake_main,
    )
    monkeypatch.setitem(sys.modules, "inference", inference)

    with pytest.raises(RuntimeError, match="nested failure"):
        runner.run("profile_salvalai", [], tmp_path / "init.json")
    assert inference.load_model_with_engine is original_loader
    assert sys.argv == original_argv
    assert not (tmp_path / "init.json").exists()


def test_runner_rejects_non_strict_environment_and_non_fp32_overrides(
    monkeypatch,
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="rejected overrides"):
        runner.run(
            "profile_salvalai",
            ["precision=fp16"],
            tmp_path / "init.json",
        )
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "1")
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "high")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    with pytest.raises(RuntimeError, match="environment mismatch"):
        runner._require_strict_fp32_environment()


def test_runner_is_cold_and_has_no_reduced_precision_or_counter_rng_ancestry() -> None:
    source = (ROOT / "utils/run_strict_fp32_timing_native_self.py").read_text(
        encoding="utf-8"
    )
    engine_source = (
        ROOT / "osuT5/osuT5/inference/optimized/single/engine.py"
    ).read_text(encoding="utf-8")
    processor_source = (
        ROOT / "osuT5/osuT5/inference/processor.py"
    ).read_text(encoding="utf-8")
    candidate_region = engine_source[
        engine_source.index("STRICT_FP32_TIMING_NATIVE_SELF_VERSION") :
    ]
    for forbidden in (
        "weight_only_runtime",
        "k8_runtime",
        "counter_uniform",
        "fp16_packed",
        "initialize_approximate",
    ):
        assert forbidden not in source
        assert forbidden not in candidate_region.lower()
    assert "from utils.run_" not in source
    assert "module.forward =" not in candidate_region
    assert '"optimized_strict_fp32_timing_native_self"' in processor_source

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "utils/run_strict_fp32_timing_native_self.py"),
            "--help",
        ],
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
