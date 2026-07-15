from __future__ import annotations

from pathlib import Path
import subprocess

from utils.analyze_lazy_startup_full_song import _comparison


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "dcc" / "profile_combined_startup_full_song.sbatch"


def test_combined_full_song_wrapper_is_serial_exact_and_not_submitted() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    source = WRAPPER.read_text()
    assert "#SBATCH --gres=gpu:2080:1" in source
    assert "for precision in fp32 fp16" in source
    assert "untraced_control" in source
    assert "exactness_audit" in source
    assert "combined_fallback" in source
    assert "MAPPERATORINATOR_NATIVE_EXTENSION_MANIFEST" in source
    assert "another user GPU job exists" in source
    assert "NVIDIA_TF32_OVERRIDE=0" in source
    assert "profile_cuda_capture=false" in source
    assert "analyze_combined_startup_full_song.py" in source
    assert "sbatch" not in source


def test_combined_full_song_comparison_uses_parent_as_baseline() -> None:
    assert _comparison(12.0, 9.0) == {
        "baseline_seconds": 12.0,
        "candidate_seconds": 9.0,
        "saved_seconds": 3.0,
        "candidate_delta_pct": -25.0,
    }
