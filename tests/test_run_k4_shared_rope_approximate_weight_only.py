from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils import run_k4_shared_rope_approximate_weight_only as combined


ROOT = Path(__file__).resolve().parents[1]


class _Stats:
    def as_dict(self):
        return {
            "module_count": 12,
            "group_count": 1,
            "forwards": 3,
            "computes": 3,
            "reuses": 33,
            "eliminated_per_forward": 11,
            "expected_computes": 3,
            "expected_reuses": 33,
            "group_computes": {"rope-0": 3},
            "member_names": [f"layer-{index}" for index in range(12)],
            "group_members": {"rope-0": [f"layer-{index}" for index in range(12)]},
        }


def test_real_inference_loads_main_binding_before_separate_timing_binding() -> None:
    source = (ROOT / "inference.py").read_text(encoding="utf-8")
    main_stage = source.index('with profiler.stage("load_main_model")')
    timing_gate = source.index("if should_load_separate_timing_model(args):", main_stage)
    timing_stage = source.index('with profiler.stage("load_timing_model")', timing_gate)
    main_call = source.index("load_model_with_engine(", main_stage, timing_gate)
    timing_call = source.index("load_model_with_engine(", timing_stage)

    assert main_stage < main_call < timing_gate < timing_stage < timing_call


def test_combined_runner_scopes_shared_rope_to_main_and_restores_everything(
    monkeypatch,
    tmp_path,
) -> None:
    import inference

    events: list[object] = []
    original_loader = inference.load_model_with_engine
    main_model = object()
    timing_model = object()
    bindings = iter((main_model, timing_model))

    def fake_loader(*args, **kwargs):
        del args, kwargs
        model = next(bindings)
        events.append(("load", model))
        return SimpleNamespace(raw_model=model), object()

    @contextmanager
    def fake_rope(model, *, stats):
        assert model is main_model
        assert isinstance(stats, combined.SharedRopeStats)
        events.append(("rope-enter", model))
        try:
            yield _Stats()
        finally:
            events.append(("rope-exit", model))

    @contextmanager
    def fake_k4(*, block_size):
        events.append(("k4-enter", block_size))
        try:
            yield
        finally:
            events.append(("k4-exit", block_size))

    def fake_weight_run(config_name, overrides, output_init_json):
        events.append(("weight", config_name, list(overrides)))
        inference.load_model_with_engine("main")
        inference.load_model_with_engine("timing")
        output_init_json.write_text(
            json.dumps(
                {
                    "result_class": "documented-drift",
                    "exactness_claim": False,
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(inference, "load_model_with_engine", fake_loader)
    monkeypatch.setattr(combined, "shared_decoder_rope_context", fake_rope)
    monkeypatch.setattr(combined, "install_k8_candidate", fake_k4)
    monkeypatch.setattr(combined, "run_weight_only", fake_weight_run)
    monkeypatch.setattr(combined, "SharedRopeStats", _Stats)
    output = tmp_path / "init.json"

    combined.run("profile_salvalai", ["seed=12345"], output)

    assert events == [
        ("k4-enter", 4),
        ("weight", "profile_salvalai", ["seed=12345"]),
        ("load", main_model),
        ("rope-enter", main_model),
        ("load", timing_model),
        ("k4-exit", 4),
        ("rope-exit", main_model),
    ]
    assert inference.load_model_with_engine is fake_loader
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["combined_runtime"] == combined.COMPOSITION_VERSION
    assert payload["shared_rope"]["scope"] == "main-model-only"
    assert payload["shared_rope"]["incremental_exactness_claim"] is True
    assert payload["shared_rope"]["stats"]["reuses"] == 33
    monkeypatch.setattr(inference, "load_model_with_engine", original_loader)


def test_combined_runner_restores_loader_and_context_on_failure(monkeypatch) -> None:
    import inference

    original_loader = inference.load_model_with_engine
    events: list[str] = []

    @contextmanager
    def fake_k4(*, block_size):
        assert block_size == 4
        events.append("k4-enter")
        try:
            yield
        finally:
            events.append("k4-exit")

    def fail(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("candidate failure")

    monkeypatch.setattr(combined, "install_k8_candidate", fake_k4)
    monkeypatch.setattr(combined, "run_weight_only", fail)

    with pytest.raises(RuntimeError, match="candidate failure"):
        combined.run("profile_salvalai", [], Path("init.json"))

    assert inference.load_model_with_engine is original_loader
    assert events == ["k4-enter", "k4-exit"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("forwards", 0, "did not observe"),
        ("computes", 2, "compute accounting"),
        ("reuses", 32, "reuse accounting"),
    ),
)
def test_shared_rope_evidence_fails_loudly(field, value, message) -> None:
    stats = _Stats()
    payload = stats.as_dict()
    payload[field] = value
    stats.as_dict = lambda: payload

    with pytest.raises(RuntimeError, match=message):
        combined._validated_shared_rope_evidence(stats)
