from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.inference.engine_binding import InferenceEngineBinding
from osuT5.osuT5.inference.optimized.single import engine
from utils import run_fp32_timing_native_self as runner


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    @property
    def dtype(self):
        return self.weight.dtype


def _binding():
    return InferenceEngineBinding(
        raw_model=_Model(),
        runtime=engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS["fp32"]),
    )


def test_runner_wraps_only_second_binding_and_restores_loader(monkeypatch, tmp_path) -> None:
    bindings = [_binding(), _binding()]
    load_index = 0

    def original_loader(*args, **kwargs):
        nonlocal load_index
        binding = bindings[load_index]
        load_index += 1
        return binding, object()

    inference = SimpleNamespace(load_model_with_engine=original_loader)
    monkeypatch.setitem(sys.modules, "inference", inference)

    def fake_composed(config_name, overrides, output_init_json, *, mode):
        assert config_name == "profile_salvalai"
        assert overrides == ["seed=12345"]
        inference.load_model_with_engine(auto_select_gamemode_model=True)
        inference.load_model_with_engine(auto_select_gamemode_model=False)
        output_init_json.write_text(
            json.dumps(
                {
                    "combined_runtime": "selected-composition",
                    "cross_candidate": {"mode": mode},
                    "cross_runtime": {"mode": mode},
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(runner, "run_composed", fake_composed)
    output = tmp_path / "initialization.json"

    runner.run("profile_salvalai", ["seed=12345"], output)

    assert inference.load_model_with_engine is original_loader
    assert bindings[0].runtime._timing_native_self_owner is None
    assert bindings[1].runtime._timing_native_self_owner is bindings[1].raw_model
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["timing_native_self_scout"]["version"] == (
        engine.TIMING_NATIVE_SELF_SCOUT_VERSION
    )
    assert payload["timing_native_self_scout"]["incremental_control"] == (
        "selected-composition"
    )


def test_runner_restores_loader_when_nested_composition_fails(monkeypatch, tmp_path) -> None:
    binding = _binding()

    def original_loader(*args, **kwargs):
        return binding, object()

    inference = SimpleNamespace(load_model_with_engine=original_loader)
    monkeypatch.setitem(sys.modules, "inference", inference)

    def fail(*args, **kwargs):
        inference.load_model_with_engine(auto_select_gamemode_model=True)
        raise RuntimeError("nested failure")

    monkeypatch.setattr(runner, "run_composed", fail)

    with pytest.raises(RuntimeError, match="nested failure"):
        runner.run("profile_salvalai", [], tmp_path / "init.json")
    assert inference.load_model_with_engine is original_loader
