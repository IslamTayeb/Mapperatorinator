from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils import run_main_shared_rope_delegate as composed


class _Stats:
    def as_dict(self):
        return {
            "module_count": 12,
            "group_count": 1,
            "forwards": 5,
            "computes": 5,
            "reuses": 55,
            "expected_computes": 5,
            "expected_reuses": 55,
            "eliminated_per_forward": 11,
        }


@pytest.mark.parametrize("arm", ["control", "candidate"])
def test_composition_patches_only_main_and_restores_loader(
    monkeypatch,
    tmp_path: Path,
    arm: str,
) -> None:
    import inference

    main_model = object()
    timing_model = object()
    models = iter((main_model, timing_model))
    events = []

    def original_loader(*args, **kwargs):
        del args
        model = next(models)
        events.append(("load", model, kwargs["auto_select_gamemode_model"]))
        return SimpleNamespace(raw_model=model), object()

    @contextmanager
    def fake_shared(model, *, stats):
        assert model is main_model
        assert isinstance(stats, _Stats)
        events.append(("shared-enter", model))
        try:
            yield stats
        finally:
            events.append(("shared-exit", model))

    def delegate(config_name, overrides, output):
        events.append(("delegate", config_name, tuple(overrides)))
        inference.load_model_with_engine(auto_select_gamemode_model=True)
        inference.load_model_with_engine(auto_select_gamemode_model=False)
        payload = {
            "result_class": "documented-drift",
            "exactness_claim": False,
        }
        if arm == "control":
            payload["decode_block_sizes"] = {
                "timing_context": 1,
                "main_generation": 4,
            }
        else:
            payload["stage_precision_hybrid"] = {
                "version": "timing-fp16-main-mixed-fp32-k4-v1",
                "result_class": "documented-drift",
                "exactness_claim": False,
                "decode_block_sizes": {
                    "timing_context": 1,
                    "main_generation": 4,
                },
            }
        output.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(inference, "load_model_with_engine", original_loader)
    monkeypatch.setattr(composed, "SharedRopeStats", _Stats)
    monkeypatch.setattr(composed, "shared_decoder_rope_context", fake_shared)
    output = tmp_path / f"{arm}.json"

    composed.run_with_main_shared_rope(
        delegate,
        "profile_salvalai",
        ["seed=12345"],
        output,
        arm=arm,
    )

    assert events == [
        ("delegate", "profile_salvalai", ("seed=12345",)),
        ("load", main_model, True),
        ("shared-enter", main_model),
        ("load", timing_model, False),
        ("shared-exit", main_model),
    ]
    assert inference.load_model_with_engine is original_loader
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["composition_arm"] == arm
    assert payload["combined_runtime"] == composed.SHARED_STAGE_COMPOSITION_VERSION
    assert payload["shared_rope"]["scope"] == "main-model-only"
    assert payload["shared_rope"]["stats"]["reuses"] == 55


def test_composition_restores_on_delegate_failure(monkeypatch, tmp_path: Path) -> None:
    import inference

    original = inference.load_model_with_engine

    def fail(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("delegate failed")

    with pytest.raises(RuntimeError, match="delegate failed"):
        composed.run_with_main_shared_rope(
            fail,
            "profile_salvalai",
            [],
            tmp_path / "missing.json",
            arm="candidate",
        )
    assert inference.load_model_with_engine is original


def test_composition_fails_on_reordered_bindings(monkeypatch, tmp_path: Path) -> None:
    import inference

    def delegate(config_name, overrides, output):
        del config_name, overrides, output
        inference.load_model_with_engine(auto_select_gamemode_model=False)

    with pytest.raises(RuntimeError, match="load order changed"):
        composed.run_with_main_shared_rope(
            delegate,
            "profile_salvalai",
            [],
            tmp_path / "missing.json",
            arm="control",
        )
