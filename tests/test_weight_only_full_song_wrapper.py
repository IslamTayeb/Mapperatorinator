import subprocess
from pathlib import Path

import pytest
import torch

from utils import run_approximate_weight_only


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/dcc/verify_weight_only_full_song_reciprocal.sbatch"
K4_WRAPPER = (
    ROOT / "scripts/dcc/verify_k4_split_weight_full_song_reciprocal.sbatch"
)
SHARED_ROPE_WRAPPER = (
    ROOT
    / "scripts/dcc/verify_k4_split_weight_shared_rope_full_song_reciprocal.sbatch"
)


def test_weight_only_full_song_wrapper_has_valid_bash_syntax() -> None:
    for wrapper in (WRAPPER, K4_WRAPPER, SHARED_ROPE_WRAPPER):
        subprocess.run(["bash", "-n", str(wrapper)], check=True)


def test_wrapper_uses_full_fp32_optimized_reciprocal_order_and_launchers() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert "PROFILE_CONFIG=${PROFILE_CONFIG:-profile_salvalai}" in source
    assert '[[ "$PROFILE_CONFIG" != profile_salvalai ]]' in source
    assert "precision=fp32" in source
    assert "inference_engine=optimized" in source
    assert "profile_detail_ranges=false" in source
    assert "profile_cuda_capture=false" in source
    assert "profile_pass_kind=untraced_control" in source
    assert '"$PYTHON" "$CANDIDATE_REPO/utils/run_fixed_seed_inference.py"' in source
    assert '--target-repo "$BASELINE_REPO"' in source
    assert (
        "CANDIDATE_RUNNER=${CANDIDATE_RUNNER:-utils/run_approximate_weight_only.py}"
        in source
    )
    assert '"$PYTHON" "$CANDIDATE_REPO/$runner"' in source
    assert source.count("run_profile baseline_first") == 1
    assert source.count("run_profile candidate_first") == 1
    assert source.count("run_profile candidate_second") == 1
    assert source.count("run_profile baseline_second") == 1
    assert source.index("run_profile baseline_first") < source.index(
        "run_profile candidate_first"
    )
    assert source.index("run_profile candidate_first") < source.index(
        "run_profile candidate_second"
    )
    assert source.index("run_profile candidate_second") < source.index(
        "run_profile baseline_second"
    )


def test_wrapper_requires_one_pushed_combined_commit_and_worktree() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert '[[ "$BASELINE_COMMIT" != "$CANDIDATE_COMMIT" ]]' in source
    assert '[[ "$BASELINE_BRANCH" != "$CANDIDATE_BRANCH" ]]' in source
    assert 'BASELINE_REPO_RESOLVED=$(realpath -e "$BASELINE_REPO")' in source
    assert 'CANDIDATE_REPO_RESOLVED=$(realpath -e "$CANDIDATE_REPO")' in source
    assert '[[ "$BASELINE_REPO_RESOLVED" != "$CANDIDATE_REPO_RESOLVED" ]]' in source
    assert 'echo "same_commit_gate=true"' in source


def test_wrapper_validates_opt_in_candidate_runner_inside_candidate_repo() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert '[[ "$CANDIDATE_RUNNER" = /* || "$CANDIDATE_RUNNER" == *".."* ]]' in source
    assert '[[ ! -f "$CANDIDATE_REPO/$CANDIDATE_RUNNER" ]]' in source
    assert 'echo "candidate_runner=$CANDIDATE_RUNNER"' in source


def test_wrapper_allows_declared_timing_drift_for_k4_composition() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert "REQUIRE_EXACT_TIMING=${REQUIRE_EXACT_TIMING:-true}" in source
    assert "REQUIRE_K4_CANDIDATE=${REQUIRE_K4_CANDIDATE:-false}" in source
    assert '[[ "$REQUIRE_EXACT_TIMING" != true' in source
    assert 'if [[ "$REQUIRE_EXACT_TIMING" == true ]]' in source
    assert "EXPECTED_EXACT_LABELS=none" in source
    assert "EXPECTED_EXACT_LABELS=timing_context" in source
    for pattern in (
        "'records.timing_context[[]*].optimized_dispatch_capture_hits'",
        "'records.timing_context[[]*].optimized_dispatch_capture_hits.*'",
        "'records.timing_context[[]*].optimized_cuda_graphs.*'",
    ):
        assert pattern in source
    assert "timing drift may be enabled only for the K4 candidate" in source


def test_dedicated_k4_wrapper_pins_combined_runner_and_contract() -> None:
    source = K4_WRAPPER.read_text(encoding="utf-8")

    assert "export CANDIDATE_RUNNER=utils/run_k4_approximate_weight_only.py" in source
    assert "export REQUIRE_EXACT_TIMING=false" in source
    assert "export REQUIRE_K4_CANDIDATE=true" in source
    assert "verify_weight_only_full_song_reciprocal.sbatch" in source


def test_shared_rope_wrapper_compares_against_exact_combined_control() -> None:
    source = SHARED_ROPE_WRAPPER.read_text(encoding="utf-8")

    assert "export BASELINE_RUNNER=utils/run_k4_approximate_weight_only.py" in source
    assert (
        "export CANDIDATE_RUNNER="
        "utils/run_k4_shared_rope_approximate_weight_only.py"
    ) in source
    assert "export REQUIRE_EXACT_TIMING=true" in source
    assert "export REQUIRE_K4_CANDIDATE=true" in source
    assert "export REQUIRE_SHARED_ROPE_INCREMENTAL=true" in source
    assert "verify_weight_only_full_song_reciprocal.sbatch" in source

    general = WRAPPER.read_text(encoding="utf-8")
    assert "ANALYSIS_MODE=exact-fp32" in general
    assert "--require-exact-label main_generation" in general
    assert "--require-exact-dispatch-label main_generation" in general
    assert "shared-RoPE did not execute" in general
    assert 'runtime_candidate=$candidate' in general


def test_general_wrapper_requires_k4_metadata_for_every_profile() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert "utils/validate_k4_profile_contract.py" in source
    assert "k4_validation_role=candidate" in source
    for role in (
        "baseline_first",
        "candidate_first",
        "candidate_second",
        "baseline_second",
    ):
        assert f'"$RUN_ROOT/{role}.k4-validation.json"' in source


def test_wrapper_isolates_native_extensions_and_keeps_compiler_caches_per_run() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert (
        'EXTENSION_JOB_ROOT="$WORK/torch_extensions/reciprocal-$EXTENSION_JOB_KEY"'
        in source
    )
    assert (
        'BASELINE_TORCH_EXTENSIONS_DIR="$EXTENSION_JOB_ROOT/$BASELINE_EXTENSION_KEY"'
        in source
    )
    assert (
        'CANDIDATE_TORCH_EXTENSIONS_DIR="$EXTENSION_JOB_ROOT/$CANDIDATE_EXTENSION_KEY"'
        in source
    )
    assert '[[ "$BASELINE_COMMIT" != "$CANDIDATE_COMMIT" ]]' in source
    assert '[[ "$BASELINE_EXTENSION_KEY" == "$CANDIDATE_EXTENSION_KEY" ]]' in source
    assert 'export TORCH_EXTENSIONS_DIR="$extension_dir"' in source
    assert 'local compiler_cache="$RUN_ROOT/compiler-cache/$role"' in source
    assert 'export TORCHINDUCTOR_CACHE_DIR="$compiler_cache/torch_inductor"' in source
    assert 'export TRITON_CACHE_DIR="$compiler_cache/triton"' in source


def test_wrapper_can_reuse_only_exact_commit_keyed_extension_caches() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert "EXTENSION_CACHE_ROOT_OVERRIDE=${EXTENSION_CACHE_ROOT_OVERRIDE:-}" in source
    assert (
        "BASELINE_EXTENSION_KEY_OVERRIDE=${BASELINE_EXTENSION_KEY_OVERRIDE:-}" in source
    )
    assert (
        "CANDIDATE_EXTENSION_KEY_OVERRIDE=${CANDIDATE_EXTENSION_KEY_OVERRIDE:-}"
        in source
    )
    assert "EXTENSION_CACHE_MODE=existing_override" in source
    assert (
        'EXTENSION_JOB_ROOT=$(realpath -e "$EXTENSION_CACHE_ROOT_OVERRIDE")' in source
    )
    assert (
        '[[ "$BASELINE_EXTENSION_KEY_OVERRIDE" != "$BASELINE_EXTENSION_KEY" ]]'
        in source
    )
    assert (
        '[[ "$CANDIDATE_EXTENSION_KEY_OVERRIDE" != "$CANDIDATE_EXTENSION_KEY" ]]'
        in source
    )
    assert "native extension cache path/commit pairing is not exact" in source


def test_baseline_extensions_preload_before_each_measured_baseline() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    first_preload = source.index("preload_baseline_extensions baseline_first")
    first_run = source.index("run_profile baseline_first")
    second_preload = source.index("preload_baseline_extensions baseline_second")
    second_run = source.index("run_profile baseline_second")
    assert first_preload < first_run
    assert second_preload < second_run
    assert source.count("preload_native_q1_attention()") == 1
    assert source.count("preload_native_decoder_layer()") == 1
    assert 'export TORCH_EXTENSIONS_DIR="$BASELINE_TORCH_EXTENSIONS_DIR"' in source

    for artifact in (
        "baseline-extension-setup/baseline_first.json",
        "baseline-extension-setup/baseline_second.json",
        "baseline-extension-setup-summary.json",
    ):
        assert artifact in source
    assert '"wall_seconds": wall_seconds' in source
    assert '"cuda_memory": {' in source
    assert '"wall_seconds": statistics.median(' in source


def test_wrapper_requires_initialization_evidence_and_relaxed_analyzer_outputs() -> (
    None
):
    source = WRAPPER.read_text(encoding="utf-8")

    for field in (
        "initialization_wall_seconds",
        "initialization_unattributed_seconds",
        "extension_init_seconds",
        "extension_allocated_bytes_delta",
        "extension_reserved_bytes_delta",
        "weight_pack_seconds",
        "weight_pack_allocated_bytes_delta",
        "weight_pack_reserved_bytes_delta",
        "initialization_cuda_memory",
    ):
        assert field in source
    assert "ANALYSIS_MODE=relaxed" in source
    assert '--mode "$ANALYSIS_MODE"' in source
    assert '"$RUN_ROOT/reciprocal-analysis.json"' in source
    assert '"$RUN_ROOT/reciprocal-analysis.txt"' in source
    assert "ANALYSIS_CLAIM=relaxed-nonexact" in source
    assert "--require-exact-label timing_context" in source
    assert "--require-exact-dispatch-label timing_context" in source
    assert "--require-dispatch-declaration" in source
    assert "parity.required_exact_labels_pass=true" in source
    assert "parity.required_exact_dispatch_labels_pass=true" in source
    assert "dispatch.declaration_pass=true" in source
    for metric in (
        "timing_model_seconds",
        "main_model_seconds",
        "complete_request_wall_seconds",
        "peak_cuda_memory_allocated_mb",
        "setup_plus_capture_seconds",
    ):
        assert f"metric.{metric}=" in source


def test_wrapper_declares_optional_result_class_and_full_main_dispatch_delta() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert "--allow-optional-dispatch-delta 'optimized_result_class'" in source
    assert "--allow-dispatch-delta 'optimized_result_class'" not in source
    aggregate = "'records.main_generation[[]*].optimized_dispatch_capture_hits'"
    children = "'records.main_generation[[]*].optimized_dispatch_capture_hits.*'"
    assert aggregate in source
    assert children in source
    assert source.index(aggregate) < source.index(children)


def test_wrapper_fails_loudly_on_weight_only_profile_dispatch_contract() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert "local validation_role=baseline" in source
    assert "validation_role=candidate" in source
    assert "utils/validate_weight_only_full_song_profile.py \\" in source
    assert '--profile "$profile"' in source
    assert '--role "$validation_role"' in source
    for role in (
        "baseline_first",
        "candidate_first",
        "candidate_second",
        "baseline_second",
    ):
        assert f'"$RUN_ROOT/{role}.weight-only-validation.json"' in source


def test_initialization_evidence_reconciles_walls_and_memory(monkeypatch) -> None:
    allocated = iter((100, 180))
    reserved = iter((200, 320))
    peak = iter((150, 240))
    clocks = iter((10.0, 12.0))
    synchronizations = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda: synchronizations.append(True)
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: next(allocated))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: next(reserved))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: next(peak))
    monkeypatch.setattr(
        run_approximate_weight_only.time,
        "perf_counter",
        lambda: next(clocks),
    )

    evidence = run_approximate_weight_only._initialize_with_evidence(
        lambda model: {
            "extension_init_seconds": 1.0,
            "weight_pack_seconds": 0.5,
            "exactness_claim": False,
        },
        object(),
    )

    assert len(synchronizations) == 2
    assert evidence["initialization_wall_seconds"] == pytest.approx(2.0)
    assert evidence["initialization_unattributed_seconds"] == pytest.approx(0.5)
    assert evidence["initialization_cuda_memory"] == {
        "allocated_bytes_before": 100,
        "allocated_bytes_after": 180,
        "allocated_bytes_delta": 80,
        "reserved_bytes_before": 200,
        "reserved_bytes_after": 320,
        "reserved_bytes_delta": 120,
        "max_allocated_bytes_before": 150,
        "max_allocated_bytes_after": 240,
        "max_allocated_bytes_delta": 90,
    }


def test_initialization_evidence_rejects_component_walls_larger_than_total(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    clocks = iter((1.0, 2.0))
    monkeypatch.setattr(
        run_approximate_weight_only.time,
        "perf_counter",
        lambda: next(clocks),
    )

    with pytest.raises(RuntimeError, match="component walls exceed"):
        run_approximate_weight_only._initialize_with_evidence(
            lambda model: {
                "extension_init_seconds": 0.8,
                "weight_pack_seconds": 0.5,
            },
            object(),
        )
