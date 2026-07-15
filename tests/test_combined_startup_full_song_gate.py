from __future__ import annotations

from pathlib import Path
import subprocess

from utils import analyze_combined_startup_full_song as combined_analysis
from utils.analyze_lazy_startup_full_song import _comparison


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "dcc" / "profile_combined_startup_full_song.sbatch"


def test_combined_full_song_wrapper_is_serial_by_default_with_parallel_opt_in() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    source = WRAPPER.read_text()
    assert "#SBATCH --gres=gpu:2080:1" in source
    assert "for precision in fp32 fp16" in source
    assert "untraced_control" in source
    assert "exactness_audit" in source
    assert "combined_fallback" in source
    assert "MAPPERATORINATOR_NATIVE_EXTENSION_MANIFEST" in source
    assert "another user GPU job exists" in source
    assert "MAPPERATORINATOR_ALLOW_PARALLEL_RECIPROCAL:-0" in source
    assert "parallel_reciprocal_opt_in=$ALLOW_PARALLEL_RECIPROCAL" in source
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


def _fake_run(*, process: float, audit: bool) -> dict:
    return {
        "output_sha256": "output",
        "output_structure": {"objects": 1},
        "workload": {"audio": "same"},
        "preset": {"precision": "same"},
        "records": {"dispatch": "same"},
        "graph_sha256": "graph",
        "signatures": {"tokens": "same"},
        "strict_exactness": {"rng": "same", "cache": "same"} if audit else None,
        "process_wall_seconds": process,
        "request_wall_seconds": 40.0,
        "generation": {
            "main_generation": {"synchronized_model_seconds": 28.0},
            "timing_context": {"synchronized_model_seconds": 8.0},
        },
        "peak_cuda_memory_mb": 1000.0,
    }


def test_combined_full_song_gate_compares_both_parents_and_fallback(monkeypatch) -> None:
    process_by_variant = {
        "lazy_parent": 52.0,
        "aot_parent": 53.0,
        "combined": 49.0,
        "combined_fallback": 49.0,
    }

    def fake_load(run_dir, *, audit, **kwargs):
        del kwargs
        variant = run_dir.parent.name
        return _fake_run(process=process_by_variant[variant], audit=audit)

    monkeypatch.setattr(combined_analysis, "_load_run", fake_load)
    expected = {
        variant: ("a" * 40, "branch")
        for variant in combined_analysis.AUDIT_VARIANTS
    }

    report = combined_analysis._analyze_precision(
        Path("/runs"),
        precision="fp32",
        expected=expected,
    )

    assert report["status"] == "PASS"
    assert set(report["exactness_audits"]) == set(combined_analysis.AUDIT_VARIANTS)
    assert report["comparisons"]["lazy_parent"]["process_wall_seconds"][
        "saved_seconds"
    ] == 3.0
    assert report["comparisons"]["aot_parent"]["process_wall_seconds"][
        "saved_seconds"
    ] == 4.0
