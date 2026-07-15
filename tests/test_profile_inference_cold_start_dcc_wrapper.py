from __future__ import annotations

from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "dcc" / "profile_inference_cold_start.sbatch"


def test_wrapper_is_valid_bash_and_cpu_only() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    source = WRAPPER.read_text()
    assert "#SBATCH --partition=common" in source
    assert "#SBATCH --gres" not in source
    assert "nvidia-smi" not in source


def test_wrapper_requires_clean_pushed_candidate_and_exact_baseline() -> None:
    source = WRAPPER.read_text()
    for expected in (
        "MAPPERATORINATOR_BASELINE_COMMIT",
        "MAPPERATORINATOR_CANDIDATE_COMMIT",
        "MAPPERATORINATOR_CANDIDATE_REMOTE_REF",
        'status --porcelain',
        'show-ref --verify --quiet',
        'NVIDIA_TF32_OVERRIDE=0',
    ):
        assert expected in source
