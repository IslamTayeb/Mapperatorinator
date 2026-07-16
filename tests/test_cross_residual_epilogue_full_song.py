from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/dcc/verify_weight_only_full_song_reciprocal.sbatch"
THIN = ROOT / "scripts/dcc/verify_cross_residual_epilogue_full_song_reciprocal.sbatch"
SUBMIT = ROOT / "scripts/dcc/submit_cross_residual_epilogue_full_song.sh"
RUNNER = ROOT / "utils/run_k4_shared_rope_fp16_cross_shared_arena_layout_fused.py"


def test_layout_fused_full_song_wrapper_pins_shared_arena_control() -> None:
    source = THIN.read_text(encoding="utf-8")
    assert "run_k4_shared_rope_fp16_cross_shared_arena.py" in source
    assert "run_k4_shared_rope_fp16_cross_shared_arena_layout_fused.py" in source
    assert "REQUIRE_CROSS_OUT_LAYOUT_FUSED_INCREMENTAL=true" in source
    assert "REQUIRE_COMPILED_CROSS_INCREMENTAL=false" in source
    assert "h36-5" in source and "h36-9" in source
    assert "MAPPERATORINATOR_JOB_TMPDIR" in source
    assert "verify_weight_only_full_song_reciprocal.sbatch" in source


def test_layout_fused_runner_sets_env_and_forbids_compile() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "MAPPERATORINATOR_CROSS_OUT_LAYOUT_FUSED" in source or "LAYOUT_FUSED_ENV" in source
    assert "compiled_cross_bmm=False" in source
    assert "heads_view_fused" in source
    assert "compiled_cross_composed" in source


def test_weight_only_wrapper_owns_layout_fused_gate() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert (
        "REQUIRE_CROSS_OUT_LAYOUT_FUSED_INCREMENTAL="
        "${REQUIRE_CROSS_OUT_LAYOUT_FUSED_INCREMENTAL:-false}" in source
    )
    assert "run_k4_shared_rope_fp16_cross_shared_arena_layout_fused.py" in source
    assert "MAPPERATORINATOR_JOB_TMPDIR" in source
    assert "must not compose compiled-cross" in source
    assert "incremental full-song gates are mutually exclusive" in source


def test_submit_excludes_compile_cross_remeasure_node() -> None:
    source = SUBMIT.read_text(encoding="utf-8")
    assert "h36-5" in source
    assert "h36-9" in source
    assert "verify_cross_residual_epilogue_full_song_reciprocal.sbatch" in source
