import subprocess
from pathlib import Path

import pytest
import torch

from utils import run_approximate_weight_only


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/dcc/verify_weight_only_full_song_reciprocal.sbatch"


def test_weight_only_full_song_wrapper_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


def test_wrapper_uses_full_fp32_optimized_reciprocal_order_and_launchers() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert "PROFILE_CONFIG=${PROFILE_CONFIG:-profile_salvalai}" in source
    assert '[[ "$PROFILE_CONFIG" != profile_salvalai ]]' in source
    assert "precision=fp32" in source
    assert "inference_engine=optimized" in source
    assert "profile_pass_kind=untraced_control" in source
    assert '"$PYTHON" inference.py --config-name "$PROFILE_CONFIG"' in source
    assert '"$PYTHON" utils/run_approximate_weight_only.py' in source
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


def test_wrapper_requires_initialization_evidence_and_relaxed_analyzer_outputs() -> None:
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
    assert "--mode relaxed" in source
    assert '"$RUN_ROOT/reciprocal-analysis.json"' in source
    assert '"$RUN_ROOT/reciprocal-analysis.txt"' in source
    assert "parity.claim=relaxed-nonexact" in source


def test_initialization_evidence_reconciles_walls_and_memory(monkeypatch) -> None:
    allocated = iter((100, 180))
    reserved = iter((200, 320))
    peak = iter((150, 240))
    clocks = iter((10.0, 12.0))
    synchronizations = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: synchronizations.append(True))
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
