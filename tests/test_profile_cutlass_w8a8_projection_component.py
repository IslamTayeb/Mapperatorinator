import csv
from pathlib import Path

import pytest
import torch

from osuT5.osuT5.inference.optimized.scout.int8_mlp import Int8PackedLinear

from utils.profile_cutlass_w8a8_projection_component import (
    CONFIGS,
    _quantized_reference,
    _write_csv,
    summarize_component,
)


SCRIPT = Path("scripts/dcc/profile_cutlass_w8a8_projection_component.sbatch")


def _bucket(*, baseline_ms: float = 1.0, candidate_ms: float = 0.5):
    regions = {}
    for region in ("self_norm_qkv", "cross_norm_q"):
        candidates = {
            config: {
                "finite": True,
                "repeat_bitwise_equal": True,
                "memory_stable": True,
                "kernel_vs_quantized_reference_pass": True,
                "active_warps": warps,
                "persistent_ctas": ctas,
                "candidate_vs_current_baseline": {"max_abs": 0.1},
                "kernel_vs_quantized_reference": {"max_abs": 0.0},
            }
            for config, (warps, ctas) in CONFIGS.items()
        }
        regions[region] = {
            "baseline_finite": True,
            "baseline_repeat_bitwise_equal": True,
            "baseline_memory_stable": True,
            "cross_configuration_bitwise_equal": True,
            "candidates": candidates,
            "l2_hot": {
                "baseline_ms_per_call": baseline_ms,
                "candidates": {
                    config: {"ms_per_call": candidate_ms}
                    for config in CONFIGS
                },
            },
            "cache_cold": {
                "baseline_ms_per_call": baseline_ms,
                "candidates": {
                    config: {"ms_per_call": candidate_ms}
                    for config in CONFIGS
                },
            },
        }
    return {"decode_replays": 50, "regions": regions}


def test_both_hot_and_cold_must_clear_fixed_main_gate() -> None:
    buckets = {"128": _bucket()}

    report = summarize_component(buckets, total_replays=100)

    assert report["modes"]["l2_hot"]["fixed_main_saving_seconds"] == pytest.approx(1.2)
    assert report["modes"]["cache_cold"]["fixed_main_saving_seconds"] == pytest.approx(1.2)
    assert report["conservative_fixed_main_saving_seconds"] == pytest.approx(1.2)
    assert report["component_pass"] is True
    assert report["retained_full_graph_required"] is True
    assert report["promotion_pass"] is False

    for region in buckets["128"]["regions"].values():
        region["l2_hot"]["candidates"] = {
            config: {"ms_per_call": 0.99} for config in CONFIGS
        }
    failed = summarize_component(buckets, total_replays=100)
    assert failed["modes"]["l2_hot"]["passes_saving_gate"] is False
    assert failed["component_pass"] is False
    assert failed["retained_full_graph_required"] is False


def test_implementation_or_cross_configuration_drift_blocks_component() -> None:
    buckets = {"128": _bucket()}
    buckets["128"]["regions"]["self_norm_qkv"]["candidates"]["w8_c68"][
        "kernel_vs_quantized_reference_pass"
    ] = False

    report = summarize_component(buckets, total_replays=100)

    assert report["invariants_pass"] is False
    assert report["component_pass"] is False


def test_quantized_reference_uses_signed_integer_dot_and_owned_scales() -> None:
    quantized = torch.tensor([3, -2, 5, -7], dtype=torch.int8)
    weight = torch.tensor(
        [[1, 2, -3, 4], [-5, 6, 7, -8]],
        dtype=torch.int8,
    )
    pack = Int8PackedLinear(
        weight=weight,
        scale=torch.tensor([0.25, 0.5]),
        bias=torch.tensor([1.0, -2.0]),
        source_weight_bytes=weight.numel() * 4,
    )

    reference, activation = _quantized_reference(
        torch.tensor([[[1.0, -2.0, 3.0, -4.0]]]),
        torch.ones(4),
        pack,
        eps=1e-5,
        quantized=quantized,
        activation_scale=torch.tensor([0.125]),
    )

    integer_dot = torch.tensor(
        [3 - 4 - 15 - 28, -15 - 12 + 35 + 56],
        dtype=torch.float32,
    )
    expected = integer_dot * torch.tensor([0.125 * 0.25, 0.125 * 0.5])
    expected += pack.bias
    assert torch.equal(reference.flatten(), expected)
    assert activation["activation_scale"] == pytest.approx(0.125)


def test_csv_contains_both_timing_modes_and_every_candidate(tmp_path) -> None:
    report = {"buckets": {"128": _bucket()}}
    path = tmp_path / "component.csv"

    _write_csv(report, path)

    rows = list(csv.DictReader(path.open()))
    assert {row["mode"] for row in rows} == {"l2_hot", "cache_cold"}
    assert {row["variant"] for row in rows} == {"baseline", *CONFIGS}
    assert len(rows) == 2 * 2 * (1 + len(CONFIGS))


def test_dcc_wrapper_pins_clean_remote_commit_and_bounded_scout() -> None:
    source = SCRIPT.read_text()

    assert "#SBATCH --time=00:30:00" in source
    assert "#SBATCH --gres=gpu:2080:1" in source
    for variable in (
        "MAPPERATORINATOR_REPO:?",
        "MAPPERATORINATOR_COMMIT:?",
        "MAPPERATORINATOR_BRANCH:?",
    ):
        assert variable in source
    assert "status --porcelain" in source
    assert "rev-parse HEAD" in source
    assert "branch --show-current" in source
    assert "show-ref --verify --quiet" in source
    assert "NVIDIA GeForce RTX 2080 Ti" in source
    assert "utils/profile_cutlass_w8a8_projection_component.py" in source
    assert "#SBATCH --exclude=dcc-core-gpu-ferc-s-h36-9" in source
    assert "cutlass-w8a8-projection-$COMMIT-$SLURM_JOB_ID" in source
    assert "cutlass-w8a8-projection-${SLURM_JOB_ID:-manual}" in source
    assert "--bucket-mode sentinel" in source
    assert "--eviction-bytes 67108864" in source
    assert "component.json" in source
    assert "component.csv" in source
    assert "component.txt" in source
    assert "diagnostics.jsonl" in source
    assert "MAPPERATORINATOR_CUDA_LAUNCH_BLOCKING" in source
    assert 'export CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING_VALUE"' in source
    assert "sbatch " not in source
