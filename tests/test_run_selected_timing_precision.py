from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.inference.engine_binding import InferenceEngineBinding
from osuT5.osuT5.inference.optimized.single import engine
from osuT5.osuT5.inference.optimized.single import timing_precision_matrix as matrix
from utils import run_selected_timing_precision as runner


class _Model(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros((), dtype=dtype))

    @property
    def dtype(self):
        return self.marker.dtype


def _binding(dtype=torch.float32):
    precision = "fp16" if dtype == torch.float16 else "fp32"
    return InferenceEngineBinding(
        _Model(dtype),
        engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS[precision]),
    )


def _selected_evidence() -> dict:
    return {
        "combined_runtime": runner.SELECTED_MAIN_COMPOSITION,
        "cross_candidate": {"mode": "fp16_packed_projections"},
    }


@pytest.mark.parametrize(
    ("mode", "timing_dtype"),
    (
        (matrix.FULL_FP16, torch.float16),
        (matrix.FP16_WEIGHTS_FP32_STATE, torch.float32),
    ),
)
def test_runner_changes_only_distinct_timing_binding(
    monkeypatch,
    tmp_path,
    mode,
    timing_dtype,
) -> None:
    bindings = [_binding(torch.float32), _binding(timing_dtype)]
    calls = []

    def original_loader(*args, **kwargs):
        calls.append(dict(kwargs))
        return bindings[len(calls) - 1], object()

    inference = SimpleNamespace(load_model_with_engine=original_loader)
    monkeypatch.setitem(sys.modules, "inference", inference)

    def fake_timed(binding, *, mode):
        return binding, {
            "version": "test",
            "mode": mode,
            "result_class": "documented-drift",
            "exactness_claim": False,
        }

    monkeypatch.setattr(runner, "_timed_binding", fake_timed)

    def fake_selected(config_name, overrides, output_init_json, *, mode):
        inference.load_model_with_engine(
            auto_select_gamemode_model=True,
            precision="fp32",
        )
        inference.load_model_with_engine(
            auto_select_gamemode_model=False,
            precision="fp32",
        )
        output_init_json.write_text(json.dumps(_selected_evidence()), encoding="utf-8")

    monkeypatch.setattr(runner, "run_selected", fake_selected)
    output = tmp_path / "init.json"
    runner.run("profile_salvalai", ["seed=12345"], output, mode=mode)

    assert inference.load_model_with_engine is original_loader
    assert calls[0]["precision"] == "fp32"
    assert calls[1]["precision"] == ("fp16" if mode == matrix.FULL_FP16 else "fp32")
    payload = json.loads(output.read_text(encoding="utf-8"))
    timing = payload["timing_precision_matrix"]
    assert timing["mode"] == mode
    assert timing["incremental_control"] == runner.SELECTED_MAIN_COMPOSITION
    assert timing["fixed_main_tokens"] == 8_294
    assert timing["fixed_timing_tokens"] == 821


def test_runner_restores_loader_after_failure(monkeypatch, tmp_path) -> None:
    original_loader = lambda *args, **kwargs: (_binding(), object())
    inference = SimpleNamespace(load_model_with_engine=original_loader)
    monkeypatch.setitem(sys.modules, "inference", inference)
    monkeypatch.setattr(
        runner,
        "run_selected",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("nested failure")),
    )

    with pytest.raises(RuntimeError, match="nested failure"):
        runner.run(
            "profile_salvalai",
            [],
            tmp_path / "init.json",
            mode=matrix.FULL_FP16,
        )
    assert inference.load_model_with_engine is original_loader
