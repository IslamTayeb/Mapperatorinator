from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_thin_wrapper_pins_exact_incremental_pair() -> None:
    source = (
        ROOT
        / "scripts/dcc/verify_k4_shared_rope_coalesced_split_kv_reciprocal.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --time=00:30:00" in source
    assert (
        "BASELINE_RUNNER="
        "utils/run_k4_shared_rope_k1_remainder_int8_mlp_weight_only.py"
    ) in source
    assert "CANDIDATE_RUNNER=utils/run_k4_shared_rope_coalesced_split_kv.py" in source
    assert "REQUIRE_EXACT_TIMING=true" in source
    assert "REQUIRE_K4_CANDIDATE=true" in source
    assert "REQUIRE_COALESCED_SPLIT_KV_INCREMENTAL=true" in source


def test_base_wrapper_validates_init_dispatch_and_same_commit() -> None:
    source = (
        ROOT / "scripts/dcc/verify_weight_only_full_song_reciprocal.sbatch"
    ).read_text(encoding="utf-8")

    assert 'if [[ "$BASELINE_COMMIT" != "$CANDIDATE_COMMIT" ]]' in source
    assert "coalesced split-KV gate requires the K1+INT8 composition control" in source
    assert "utils/validate_coalesced_split_kv_profile.py" in source
    assert "main-model-mixed-weight-self-attention-only" in source
    assert (
        "k4-split-kv-mixed-weight-shared-rope-k1-remainder-int8-mlp-"
        "coalesced-split-kv-v1"
    ) in source
    assert "--require-exact-label timing_context" in source
    assert "coalesced-split-kv-validation.json" in source
    assert "metric.fixed_8294_main_seconds=" in source
    assert "metric.complete_request_wall_seconds=" in source
