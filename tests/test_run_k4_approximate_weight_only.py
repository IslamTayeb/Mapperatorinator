from contextlib import contextmanager
from pathlib import Path

import pytest

from utils import run_k4_approximate_weight_only


def test_combined_runner_scopes_k4_around_weight_initialization(monkeypatch) -> None:
    events: list[object] = []

    @contextmanager
    def fake_install(*, block_size: int):
        events.append(("enter", block_size))
        try:
            yield
        finally:
            events.append(("exit", block_size))

    def fake_weight_run(config_name, overrides, output_init_json):
        events.append(("weight", config_name, list(overrides), output_init_json))

    monkeypatch.setattr(run_k4_approximate_weight_only, "install_k8_candidate", fake_install)
    monkeypatch.setattr(run_k4_approximate_weight_only, "run_weight_only", fake_weight_run)
    output = Path("/tmp/init.json")

    run_k4_approximate_weight_only.run("profile_salvalai", ["seed=12345"], output)

    assert events == [
        ("enter", 4),
        ("weight", "profile_salvalai", ["seed=12345"], output),
        ("exit", 4),
    ]


def test_combined_runner_restores_k4_context_when_weight_run_fails(monkeypatch) -> None:
    events: list[str] = []

    @contextmanager
    def fake_install(*, block_size: int):
        assert block_size == 4
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def fail(*args, **kwargs):
        raise RuntimeError("weight failure")

    monkeypatch.setattr(run_k4_approximate_weight_only, "install_k8_candidate", fake_install)
    monkeypatch.setattr(run_k4_approximate_weight_only, "run_weight_only", fail)

    with pytest.raises(RuntimeError, match="weight failure"):
        run_k4_approximate_weight_only.run("profile_salvalai", [], Path("init.json"))
    assert events == ["enter", "exit"]
