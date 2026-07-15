from __future__ import annotations

from pathlib import Path
import subprocess

from utils.analyze_lazy_startup_full_song import _comparison


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "dcc" / "profile_lazy_startup_full_song.sbatch"


def test_full_song_wrapper_is_valid_and_serial() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    source = WRAPPER.read_text()
    assert "#SBATCH --gres=gpu:2080:1" in source
    assert "for precision in fp32 fp16" in source
    assert "for index in 01 02 03" in source
    assert "exactness_audit" in source
    assert "another user GPU job exists" in source
    assert "NVIDIA_TF32_OVERRIDE=0" in source
    assert "profile_cuda_capture=false" in source


def test_full_song_comparison_reports_candidate_delta() -> None:
    assert _comparison(10.0, 8.0) == {
        "baseline_seconds": 10.0,
        "candidate_seconds": 8.0,
        "saved_seconds": 2.0,
        "candidate_delta_pct": -20.0,
    }
