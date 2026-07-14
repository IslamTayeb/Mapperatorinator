import pytest

from utils import run_k4_shared_rope_mask_reuse_approximate_weight_only as runner


def test_runner_adds_mask_reuse_to_shared_rope_stack(monkeypatch, tmp_path):
    calls = []

    def shared_run(config_name, overrides, output, **kwargs):
        calls.append(("run", config_name, overrides, output, kwargs))

    monkeypatch.setattr(runner, "run_shared_rope", shared_run)
    output = tmp_path / "init.json"

    runner.run("profile_salvalai", ["seed=12345"], output)

    assert calls == [
        (
            "run",
            "profile_salvalai",
            ["seed=12345"],
            output,
            {"reuse_decoder_attention_mask": True},
        )
    ]


def test_runner_propagates_shared_stack_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner,
        "run_shared_rope",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failure")),
    )

    with pytest.raises(RuntimeError, match="failure"):
        runner.run("profile_salvalai", [], tmp_path / "init.json")
