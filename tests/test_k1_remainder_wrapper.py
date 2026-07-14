from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_wrapper_pins_reciprocal_runners_and_one_2080_ti() -> None:
    wrapper = (
        ROOT
        / "scripts/dcc/verify_k4_shared_rope_k1_remainder_reciprocal.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --gres=gpu:2080:1" in wrapper
    assert "#SBATCH --time=00:30:00" in wrapper
    assert (
        "BASELINE_RUNNER=utils/run_k4_shared_rope_approximate_weight_only.py"
        in wrapper
    )
    assert "CANDIDATE_RUNNER=utils/run_k4_shared_rope_k1_remainder.py" in wrapper
    assert "REQUIRE_K1_REMAINDER_INCREMENTAL=true" in wrapper


def test_base_wrapper_requires_exact_tokens_and_explicit_remainder_evidence() -> None:
    source = (
        ROOT / "scripts/dcc/verify_weight_only_full_song_reciprocal.sbatch"
    ).read_text(encoding="utf-8")

    assert "utils/validate_k1_remainder_profile.py" in source
    assert "--require-exact-label timing_context" in source
    assert "--require-exact-label main_generation" in source
    assert "EXPECTED_EXACT_DISPATCH_LABELS=none" in source
    assert "parity.cross_candidate_exact=true" in source


def test_candidate_runner_only_enables_opt_in_remainder_graphs() -> None:
    source = (
        ROOT / "utils/run_k4_shared_rope_k1_remainder.py"
    ).read_text(encoding="utf-8")

    assert "graph_remainders=True" in source
    assert "run_k4_shared_rope_approximate_weight_only import run" in source


def test_candidate_runner_is_directly_executable_outside_repo(tmp_path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "utils/run_k4_shared_rope_k1_remainder.py"),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
