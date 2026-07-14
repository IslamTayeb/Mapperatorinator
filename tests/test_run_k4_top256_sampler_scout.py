from contextlib import contextmanager
import json
from pathlib import Path

from utils import run_k4_top256_sampler_scout as runner


ROOT = Path(__file__).resolve().parents[1]


def test_runner_uses_selected_cross_runtime_and_writes_plain_artifacts(
    monkeypatch,
    tmp_path,
) -> None:
    events = []

    @contextmanager
    def observer_context(observer):
        events.append("observer-enter")
        observer.samples = [object()]
        try:
            yield
        finally:
            events.append("observer-exit")

    def fake_cross(config_name, overrides, output_init_json, *, mode):
        events.append(("cross", config_name, overrides, output_init_json, mode))

    component = {
        "correctness": {
            "samples": 1,
            "overflow_count": 0,
            "selected_token_exact": True,
            "counter_threshold_consumed_unchanged": True,
        },
        "representatives": [
            {
                "baseline_ms_per_step_worst": 0.071114,
                "candidate_ms_per_step_worst": 0.03,
                "baseline_capture_seconds": 0.1,
                "candidate_capture_seconds": 0.1,
                "selected_token_exact": True,
            }
        ],
    }
    monkeypatch.setattr(runner, "install_vocab_sampling_observer", observer_context)
    monkeypatch.setattr(runner, "run_cross", fake_cross)
    monkeypatch.setattr(runner, "benchmark_candidate", lambda *args, **kwargs: component)
    monkeypatch.setattr(
        runner,
        "fixed_physical_work",
        lambda *args, **kwargs: {"fixed_physical_steps": 8294.0},
    )
    monkeypatch.setattr(runner, "_git_value", lambda *args: "test-value")
    output_json = tmp_path / "component.json"
    output_text = tmp_path / "component.txt"

    report = runner.run(
        "profile_salvalai",
        ["seed=12345"],
        output_init_json=tmp_path / "initialization.json",
        output_report_json=output_json,
        output_report_text=output_text,
        inference_output_dir=tmp_path / "output",
        max_samples=1,
        warmup=1,
        iterations=1,
        rounds=1,
        fixed_main_steps=8294,
        max_candidate_ms_per_step=0.035,
        minimum_saving_seconds=0.3,
    )

    assert events == [
        "observer-enter",
        (
            "cross",
            "profile_salvalai",
            ["seed=12345"],
            tmp_path / "initialization.json",
            runner.CROSS_FP16_PACKED,
        ),
        "observer-exit",
    ]
    assert json.loads(output_json.read_text(encoding="utf-8"))["gate"][
        "promotion_pass"
    ]
    assert output_text.read_text(encoding="utf-8").startswith(
        "decision=retain_for_full_runtime_scout\n"
    )
    assert report["metadata"]["sampler_runtime_installed"] is False


def test_dcc_wrapper_is_clean_pushed_2080ti_and_gate_bounded() -> None:
    source = (
        ROOT / "scripts/dcc/profile_top256_sampler_component.sbatch"
    ).read_text(encoding="utf-8")

    for required in (
        "MAPPERATORINATOR_REPO",
        "MAPPERATORINATOR_COMMIT",
        "MAPPERATORINATOR_BRANCH",
        'REMOTE=${MAPPERATORINATOR_REMOTE:-islamtayeb}',
        'status --porcelain',
        'refs/remotes/$REMOTE/$BRANCH',
        'NVIDIA GeForce RTX 2080 Ti',
        '--max-candidate-ms-per-step 0.035',
        '--minimum-saving-seconds 0.3',
        'output_path="$RUN_ROOT/output"',
        'profile_pass_kind=untraced_control',
        'STOP_COMPONENT_SCOUT',
        'PROMOTE_TO_RUNTIME_SCOUT',
        'sha256sums.txt',
    ):
        assert required in source
    assert "sbatch " not in source
    assert ".md" not in source
    assert ".html" not in source
    assert ".png" not in source
