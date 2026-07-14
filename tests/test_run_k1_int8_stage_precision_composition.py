from pathlib import Path

from utils import run_k4_shared_rope_k1_int8_stage_aware_control as control
from utils import run_k4_shared_rope_k1_int8_stage_precision_hybrid as candidate


def test_control_delegate_retains_int8_and_captured_remainders(monkeypatch, tmp_path):
    calls = []

    def fake_run(config_name, overrides, output, **kwargs):
        calls.append((config_name, overrides, output, kwargs))

    monkeypatch.setattr(control, "_control_runner", lambda: fake_run)
    output = tmp_path / "control.json"
    control._run_delegate("profile_salvalai", ["seed=12345"], output)

    assert calls == [
        (
            "profile_salvalai",
            ["seed=12345"],
            output,
            {
                "initializer_name": control.INITIALIZER,
                "graph_remainders": True,
            },
        )
    ]


def test_candidate_delegate_changes_only_timing_precision(monkeypatch, tmp_path):
    calls = []

    def fake_run(config_name, overrides, output, **kwargs):
        calls.append((config_name, overrides, output, kwargs))

    monkeypatch.setattr(candidate, "_hybrid_runner", lambda: fake_run)
    output = tmp_path / "candidate.json"
    candidate._run_delegate("profile_salvalai", ["seed=12345"], output)

    assert candidate.COMPOSITION_VERSION == control.COMPOSITION_VERSION
    assert candidate.INITIALIZER == control.INITIALIZER
    assert calls == [
        (
            "profile_salvalai",
            ["seed=12345"],
            output,
            {
                "initializer_name": control.INITIALIZER,
                "graph_remainders": True,
            },
        )
    ]


def test_both_arms_declare_one_shared_composition(monkeypatch, tmp_path: Path):
    calls = []

    def fake_shared(delegate, config_name, overrides, output, **kwargs):
        calls.append((delegate, config_name, overrides, output, kwargs))

    monkeypatch.setattr(control, "_shared_runner", lambda: fake_shared)
    monkeypatch.setattr(candidate, "_shared_runner", lambda: fake_shared)
    control_output = tmp_path / "control.json"
    candidate_output = tmp_path / "candidate.json"
    control.run("profile_salvalai", [], control_output)
    candidate.run("profile_salvalai", [], candidate_output)

    assert calls[0][-1] == {
        "arm": "control",
        "composition_version": control.COMPOSITION_VERSION,
    }
    assert calls[1][-1] == {
        "arm": "candidate",
        "composition_version": control.COMPOSITION_VERSION,
    }
