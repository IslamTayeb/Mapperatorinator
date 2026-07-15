from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from utils.analyze_shared_rope_full_song import (
    FullSongGateError,
    _candidate_evidence_contract,
    _comparison,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "dcc" / "profile_shared_rope_full_song.sbatch"


def test_full_song_wrapper_is_serial_by_default_with_explicit_parallel_opt_in() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    source = WRAPPER.read_text()
    assert "#SBATCH --gres=gpu:2080:1" in source
    assert 'MAPPERATORINATOR_PRECISION:?Set MAPPERATORINATOR_PRECISION' in source
    assert "MAPPERATORINATOR_ALLOW_PARALLEL_RECIPROCAL:-0" in source
    assert "parallel_reciprocal_opt_in=$ALLOW_PARALLEL_RECIPROCAL" in source
    assert 'for precision in "$PRECISION"' in source
    assert "for index in 01 02 03" in source
    assert "exactness_audit" in source
    assert "another user GPU job exists" in source
    assert "NVIDIA_TF32_OVERRIDE=0" in source
    assert "profile_cuda_capture=false" in source
    assert "utils/run_strict_shared_rope.py" in source
    assert "candidate-evidence.json" in source
    assert "utils/analyze_shared_rope_full_song.py" in source


def test_full_song_comparison_reports_candidate_delta() -> None:
    assert _comparison(10.0, 8.0) == {
        "baseline_seconds": 10.0,
        "candidate_seconds": 8.0,
        "saved_seconds": 2.0,
        "candidate_delta_pct": -20.0,
    }


def _evidence(precision: str) -> dict:
    return {
        "candidate": {
            "version": "strict-shared-rope-dtype-generic-v2",
            "precision": precision,
            "dtype_generic": True,
            "original_decoder_forward_required": True,
            "production_wiring": False,
            "full_song_exactness_claim": False,
            "retained_specialized_dispatch": {
                "q1_bmm_cross_attention": True,
                "native_q1_rope_cache_self_attention": True,
                "native_q1_self_attention": True,
            },
            "stats": {"reuses": 11},
        }
    }


@pytest.mark.parametrize("precision", ("fp32", "fp16"))
def test_candidate_evidence_requires_dtype_and_specialized_topology(precision):
    payload = _evidence(precision)
    assert _candidate_evidence_contract(payload, precision=precision) is payload[
        "candidate"
    ]

    payload["candidate"]["retained_specialized_dispatch"][
        "native_q1_rope_cache_self_attention"
    ] = False
    with pytest.raises(FullSongGateError, match="candidate evidence mismatch"):
        _candidate_evidence_contract(payload, precision=precision)
