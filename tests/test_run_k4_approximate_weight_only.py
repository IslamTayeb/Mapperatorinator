from contextlib import contextmanager
from pathlib import Path

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
