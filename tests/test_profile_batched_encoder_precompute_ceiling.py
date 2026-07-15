from types import SimpleNamespace

import pytest
import torch

from utils.profile_batched_encoder_precompute_ceiling import (
    SCHEMA_VERSION,
    _assert_accepted_args,
    _stack_window_kwargs,
    _validate_report,
    _window_input_manifest,
    encoder_drift,
    parse_batch_sizes,
    validate_batch_sizes,
)


def _accepted_args(**overrides):
    values = {
        "inference_engine": "optimized",
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "device": "cuda",
        "use_server": False,
        "parallel": False,
        "cfg_scale": 1.0,
        "num_beams": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _valid_report():
    live_windows = 3
    variants = {}
    for batch_size in (1, 2):
        variants[str(batch_size)] = {
            "live_window_count": live_windows,
            "batch_setup_seconds": 0.01,
            "input_copy_seconds": 0.02,
            "encoder_synchronized_seconds": 0.1,
            "storage_allocation_seconds": 0.003,
            "output_store_copy_seconds": 0.004,
            "complete_precompute_seconds": 0.14,
            "encoder_drift_vs_b1": {
                "window_count": live_windows,
                "max_abs": 0.0,
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "live_window_count": live_windows,
            "batch_sizes": [1, 2],
        },
        "variants": variants,
    }


def test_parse_batch_sizes_requires_ordered_unique_b1_reference():
    assert parse_batch_sizes("1,2,4,8,16") == (1, 2, 4, 8, 16)

    for invalid in ("", "2,4", "1,4,2", "1,2,2", "1,0", "1,nope"):
        with pytest.raises(ValueError):
            parse_batch_sizes(invalid)


def test_validate_batch_sizes_fails_loudly_above_configured_cap():
    assert validate_batch_sizes((1, 2, 4, 8), max_batch_size=8) == (1, 2, 4, 8)

    with pytest.raises(ValueError, match=r"16.*max_batch_size=8"):
        validate_batch_sizes((1, 2, 4, 8, 16), max_batch_size=8)


def test_accepted_args_accepts_shared_fp32_fp16_and_rejects_other_runtime():
    _assert_accepted_args(_accepted_args())
    _assert_accepted_args(_accepted_args(precision="fp16"))

    with pytest.raises(ValueError, match="fp32 or fp16"):
        _assert_accepted_args(_accepted_args(precision="bf16"))
    with pytest.raises(ValueError, match="accepted runtime"):
        _assert_accepted_args(_accepted_args(inference_engine="v32"))


def test_stack_window_kwargs_preserves_each_live_row_and_rejects_key_drift():
    rows = [
        {
            "song_position": torch.tensor([[0.0, 0.1]]),
            "timing": torch.tensor([[1, 2]]),
        },
        {
            "song_position": torch.tensor([[0.1, 0.2]]),
            "timing": torch.tensor([[3, 4]]),
        },
    ]

    stacked = _stack_window_kwargs(rows)

    assert torch.equal(
        stacked["song_position"],
        torch.tensor([[0.0, 0.1], [0.1, 0.2]]),
    )
    assert torch.equal(stacked["timing"], torch.tensor([[1, 2], [3, 4]]))

    with pytest.raises(ValueError, match="conditioning keys changed"):
        _stack_window_kwargs([rows[0], {"timing": rows[1]["timing"]}])


def test_window_manifest_is_stable_and_detects_input_changes():
    frames = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    kwargs = [
        {"song_position": torch.tensor([[index / 10, (index + 1) / 10]])}
        for index in range(3)
    ]

    first = _window_input_manifest(frames, kwargs)
    second = _window_input_manifest(frames.clone(), kwargs)
    changed_frames = frames.clone()
    changed_frames[1, 2] += 1
    changed = _window_input_manifest(changed_frames, kwargs)

    assert first == second
    assert first["live_window_count"] == 3
    assert first["combined_sha256"] != changed["combined_sha256"]

    with pytest.raises(ValueError, match="at least one live audio window"):
        _window_input_manifest(torch.empty((0, 4)), [])


def test_encoder_drift_reports_exact_windows_and_max_abs():
    reference = torch.zeros((3, 2, 2), dtype=torch.float32)
    candidate = reference.clone()
    candidate[1, 0, 0] = 0.25
    candidate[2, 1, 1] = -0.5

    drift = encoder_drift(reference, candidate)

    assert drift["max_abs"] == pytest.approx(0.5)
    assert drift["mean_window_max_abs"] == pytest.approx(0.25)
    assert drift["exact_window_count"] == 1
    assert drift["per_window_max_abs"] == pytest.approx([0.0, 0.25, 0.5])

    with pytest.raises(ValueError, match="shape changed"):
        encoder_drift(reference, candidate[:, :, :1])
    with pytest.raises(TypeError, match="dtype changed"):
        encoder_drift(reference, candidate.half())


def test_report_schema_requires_same_live_window_count_and_finite_timings():
    report = _valid_report()
    _validate_report(report)

    report["variants"]["2"]["live_window_count"] = 2
    with pytest.raises(ValueError, match="cover every live window"):
        _validate_report(report)

    report = _valid_report()
    report["variants"]["2"]["encoder_synchronized_seconds"] = float("nan")
    with pytest.raises(ValueError, match="missing or non-finite"):
        _validate_report(report)
