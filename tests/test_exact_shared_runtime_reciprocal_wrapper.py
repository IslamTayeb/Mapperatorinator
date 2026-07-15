from pathlib import Path


SCRIPT = Path("scripts/dcc/profile_exact_shared_runtime_reciprocal.sbatch")


def test_exact_shared_runtime_wrapper_isolates_native_builds_and_temp_files() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'extension_dir="$WORK/torch_extensions/exact-shared-${commit}"' in source
    assert 'temp_dir="$WORK/tmp/exact-shared-${SLURM_JOB_ID}-${role}"' in source
    assert 'TORCH_EXTENSIONS_DIR="$extension_dir" TMPDIR="$temp_dir"' in source
    assert 'PYTHONPATH="$repo"' in source
    assert 'BASELINE_COMMIT=${MAPPERATORINATOR_BASELINE_COMMIT:?' in source
    assert 'CANDIDATE_COMMIT=${MAPPERATORINATOR_CANDIDATE_COMMIT:?' in source


def test_exact_shared_runtime_wrapper_uses_strict_reciprocal_order_and_gates() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    calls = [
        "run_role baseline_first baseline",
        "run_role candidate_first candidate",
        "run_role candidate_second candidate",
        "run_role baseline_second baseline",
    ]
    assert [source.index(call) for call in calls] == sorted(
        source.index(call) for call in calls
    )
    assert "--mode \"$ANALYSIS_MODE\"" in source
    assert 'report["parity"]["cross_candidate_exact"] is True' in source
    assert 'candidate_config["shared_decoder_rope"] is True' in source
    assert 'candidate_config["preallocated_batch1_sequence_state"] is True' in source
    assert "native_q1_rope_cache_self_attention" in source
    assert "q1_bmm_cross_attention" in source
