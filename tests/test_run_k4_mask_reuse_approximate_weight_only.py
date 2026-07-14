from pathlib import Path

import pytest

from utils import run_k4_mask_reuse_approximate_weight_only as runner


def test_runner_composes_k4_mask_reuse_with_mixed_weights(monkeypatch, tmp_path):
    calls = []

    class Context:
        def __enter__(self):
            calls.append(("enter",))

        def __exit__(self, *args):
            calls.append(("exit",))

    def install(**kwargs):
        calls.append(("install", kwargs))
        return Context()

    def weight_run(config_name, overrides, output):
        calls.append(("run", config_name, overrides, output))

    monkeypatch.setattr(runner, "install_k8_candidate", install)
    monkeypatch.setattr(runner, "run_weight_only", weight_run)
    output = tmp_path / "init.json"

    runner.run("profile_salvalai", ["seed=12345"], output)

    assert calls == [
        (
            "install",
            {"block_size": 4, "reuse_decoder_attention_mask": True},
        ),
        ("enter",),
        ("run", "profile_salvalai", ["seed=12345"], output),
        ("exit",),
    ]


def test_runner_restores_context_after_weight_failure(monkeypatch, tmp_path):
    exited = []

    class Context:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            exited.append(True)

    monkeypatch.setattr(runner, "install_k8_candidate", lambda **kwargs: Context())
    monkeypatch.setattr(
        runner,
        "run_weight_only",
        lambda *args: (_ for _ in ()).throw(RuntimeError("failure")),
    )

    with pytest.raises(RuntimeError, match="failure"):
        runner.run("profile_salvalai", [], Path(tmp_path / "init.json"))
    assert exited == [True]
