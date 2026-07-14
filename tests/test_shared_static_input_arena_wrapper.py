from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_shared_arena_wrapper_owns_selected_stack_and_hard_gate():
    source = (
        ROOT / "scripts/dcc/verify_shared_static_input_arena_reciprocal.sbatch"
    ).read_text(encoding="utf-8")

    assert "run_k4_shared_rope_fp16_cross_transition_timing.py" in source
    assert "run_k4_shared_rope_fp16_cross_shared_arena_transition_timing.py" in source
    assert "REQUIRE_SHARED_ARENA_INCREMENTAL=true" in source
    assert "CROSS_CANDIDATE_MODE=fp16_packed_projections" in source
    assert "--gres=gpu:2080:1" in source


def test_general_wrapper_runs_arena_analyzer_and_retains_negative_report():
    source = (
        ROOT / "scripts/dcc/verify_weight_only_full_song_reciprocal.sbatch"
    ).read_text(encoding="utf-8")
    block = source[source.index('if [[ "$REQUIRE_SHARED_ARENA_INCREMENTAL" == true ]]; then') :]

    assert "utils/analyze_shared_static_input_arena.py" in block
    assert "shared-static-input-arena-analysis.json" in block
    assert 'SHARED_ARENA_STATUS" -ne 0' in block
    assert 'SHARED_ARENA_STATUS" -ne 3' in block
    assert 'SHARED_ARENA_STATUS" -eq 3' in block


def test_shared_arena_runners_differ_only_by_arena_policy():
    baseline = (
        ROOT / "utils/run_k4_shared_rope_fp16_cross_transition_timing.py"
    ).read_text(encoding="utf-8")
    candidate = (
        ROOT
        / "utils/run_k4_shared_rope_fp16_cross_shared_arena_transition_timing.py"
    ).read_text(encoding="utf-8")

    assert "mode=CROSS_FP16_PACKED" in baseline
    assert "transition_timing=True" in baseline
    assert "shared_static_input_arena=True" not in baseline
    assert "mode=CROSS_FP16_PACKED" in candidate
    assert "transition_timing=True" in candidate
    assert "shared_static_input_arena=True" in candidate
