from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from utils.profile_device_resident_tail import (
    BASELINE_MAIN_SECONDS,
    FIXED_WORK_TOKENS,
    PROMOTION_SAVING_SECONDS,
    _RawTailCapture,
    _bucketed_prefix,
    _mutable_tensor_templates,
    summarize,
)


class _OffsetProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        del input_ids
        return scores + 1


class _NeverStop:
    def __call__(self, input_ids, scores):
        del scores
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def _entry(*, saved_us: float, passed: bool = True) -> dict:
    return {
        "saved_wall_us_per_token": saved_us,
        "pass": passed,
    }


def test_bucketed_prefix_rounds_up_and_caps() -> None:
    assert _bucketed_prefix(1, 1024) == 64
    assert _bucketed_prefix(64, 1024) == 64
    assert _bucketed_prefix(65, 1024) == 128
    assert _bucketed_prefix(1000, 1000) == 1000
    with pytest.raises(ValueError, match="positive"):
        _bucketed_prefix(0, 10)


def test_raw_capture_is_transparent_and_retains_one_contiguous_block() -> None:
    processors = LogitsProcessorList([_OffsetProcessor()])
    config = SimpleNamespace(
        _pad_token_tensor=torch.tensor([0]),
        _eos_token_tensor=torch.tensor([9, 10]),
        do_sample=False,
        max_length=64,
    )
    anchors = {}
    capture = _RawTailCapture(
        anchors=anchors,
        processors=processors,
        stopping_criteria=[_NeverStop()],
        generation_config=config,
        block_size=4,
    )
    processors.insert(0, capture)

    prompt = torch.arange(1, 49).reshape(1, -1)
    scores = torch.zeros((1, 16))
    for step in range(4):
        input_ids = torch.cat((prompt, torch.ones((1, step), dtype=torch.long)), dim=1)
        assert capture(input_ids, scores) is scores

    assert set(anchors) == {64}
    anchor = anchors[64]
    assert anchor.input_ids.shape == (1, 48)
    assert len(anchor.raw_logits_steps) == 4
    assert [item.__class__.__name__ for item in anchor.logits_processors] == [
        "_OffsetProcessor"
    ]
    assert capture not in anchor.logits_processors


def test_mutable_tensor_templates_cover_nested_state_and_restore_in_place() -> None:
    first = torch.tensor([1.0])
    second = torch.tensor([2.0])
    root = SimpleNamespace(state={"first": first, "nested": [second, first]})

    templates = _mutable_tensor_templates(root)
    first.add_(10)
    second.mul_(10)
    for tensor, template in templates:
        tensor.copy_(template)

    assert first.tolist() == [1.0]
    assert second.tolist() == [2.0]
    assert len(templates) == 2


def test_summary_charges_only_live_coverage_and_clears_gate() -> None:
    live = {64: 4_000, 128: 4_294}
    saved_us = PROMOTION_SAVING_SECONDS * 1_000_000 / sum(live.values())
    buckets = {
        "64": {"8": _entry(saved_us=saved_us), "16": _entry(saved_us=saved_us)},
        "128": {"8": _entry(saved_us=saved_us), "16": _entry(saved_us=saved_us)},
    }

    report = summarize(buckets, live_counts=live)

    for block in ("8", "16"):
        assert report[block]["coverage_fraction"] == 1.0
        assert report[block]["projected_main_saved_seconds"] == pytest.approx(
            PROMOTION_SAVING_SECONDS
        )
        assert report[block]["projected_main_seconds"] == pytest.approx(
            BASELINE_MAIN_SECONDS - PROMOTION_SAVING_SECONDS
        )
        assert report[block]["projected_fixed_work_tps"] == pytest.approx(
            FIXED_WORK_TOKENS / (BASELINE_MAIN_SECONDS - PROMOTION_SAVING_SECONDS)
        )
        assert report[block]["promotion_pass"]


def test_summary_conservatively_allows_partial_coverage_but_blocks_failure() -> None:
    live = {64: 4, 128: 4}
    buckets = {
        "64": {
            "8": _entry(saved_us=2_000_000),
            "16": _entry(saved_us=2_000_000, passed=False),
        }
    }

    report = summarize(buckets, live_counts=live)

    assert report["8"]["projected_main_saved_seconds"] == pytest.approx(8.0)
    assert report["8"]["coverage_fraction"] == 0.5
    assert report["8"]["unmeasured_prefixes_assumed_saving_seconds"] == 0.0
    assert report["8"]["promotion_pass"]
    assert not report["16"]["all_correctness_pass"]
    assert not report["16"]["promotion_pass"]


def test_summary_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="requires"):
        summarize({}, live_counts={64: 1})
    with pytest.raises(ValueError, match="requires"):
        summarize({"64": {}}, live_counts={})
