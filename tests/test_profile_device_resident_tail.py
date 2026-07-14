from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from utils.profile_device_resident_tail import (
    PROMOTION_SAVING_SECONDS,
    _RawTailCapture,
    _bucketed_prefix,
    _is_main_capture_label,
    _mutable_tensor_templates,
    _representative_anchors,
    WindowSchedule,
    schedule_block_manifest,
    summarize,
    validate_live_graph_cache,
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
        "conservative_saved_wall_us_per_token": saved_us,
        "pass": passed,
    }


def test_bucketed_prefix_rounds_up_and_caps() -> None:
    assert _bucketed_prefix(1, 1024) == 64
    assert _bucketed_prefix(64, 1024) == 64
    assert _bucketed_prefix(65, 1024) == 128
    assert _bucketed_prefix(1000, 1000) == 1000
    with pytest.raises(ValueError, match="positive"):
        _bucketed_prefix(0, 10)


def test_capture_provenance_accepts_only_main_generation() -> None:
    assert _is_main_capture_label("main_generation")
    assert not _is_main_capture_label("timing_context")
    assert not _is_main_capture_label(None)


def test_representative_anchor_selection_spans_first_middle_last() -> None:
    anchors = [object() for _ in range(9)]
    selected = _representative_anchors(anchors, limit=3)
    assert selected == [anchors[0], anchors[4], anchors[8]]


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
        block_sizes=(4,),
        window_index=7,
        samples_per_prefix=3,
    )
    processors.insert(0, capture)

    prompt = torch.arange(1, 49).reshape(1, -1)
    scores = torch.zeros((1, 16))
    for step in range(5):
        input_ids = torch.cat((prompt, torch.ones((1, step), dtype=torch.long)), dim=1)
        assert capture(input_ids, scores) is scores

    assert set(anchors) == {4}
    assert set(anchors[4]) == {64}
    anchor = anchors[4][64][0]
    assert anchor.input_ids.shape == (1, 49)
    assert len(anchor.raw_logits_steps) == 4
    assert [item.__class__.__name__ for item in anchor.logits_processors] == [
        "_OffsetProcessor"
    ]
    assert capture not in anchor.logits_processors
    assert capture.schedule() == WindowSchedule(
        window_index=7,
        generated_steps=5,
        decode_prefixes=[64, 64, 64, 64],
    )


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
    schedule = schedule_block_manifest(
        [
            WindowSchedule(
                window_index=0,
                generated_steps=sum(live.values()) + 1,
                decode_prefixes=[64] * live[64] + [128] * live[128],
            )
        ],
        live_counts=live,
    )
    eligible = schedule["blocks"]["8"]["eligible_replays"]
    saved_us = PROMOTION_SAVING_SECONDS * 1_000_000 / eligible
    buckets = {
        "64": {"8": _entry(saved_us=saved_us), "16": _entry(saved_us=saved_us)},
        "128": {"8": _entry(saved_us=saved_us), "16": _entry(saved_us=saved_us)},
    }

    report = summarize(buckets, live_counts=live, schedule_manifest=schedule)

    for block in ("8", "16"):
        assert report[block]["eligible_coverage_fraction"] == 1.0
        assert report[block]["tail_component_runtime_headroom_seconds"] == pytest.approx(
            PROMOTION_SAVING_SECONDS
        )
        assert report[block]["tail_component_headroom_pass"]
        assert report[block]["full_runtime_decision_available"] is False
        assert report[block]["full_runtime_promotion_pass"] is False


def test_summary_blocks_partial_eligible_coverage_and_correctness_failure() -> None:
    live = {64: 8, 128: 8}
    schedule = schedule_block_manifest(
        [WindowSchedule(0, 17, [64] * 8 + [128] * 8)],
        live_counts=live,
    )
    buckets = {
        "64": {
            "8": _entry(saved_us=2_000_000),
            "16": _entry(saved_us=2_000_000, passed=False),
        }
    }

    report = summarize(buckets, live_counts=live, schedule_manifest=schedule)

    assert report["8"]["tail_component_runtime_headroom_seconds"] == pytest.approx(16.0)
    assert report["8"]["eligible_coverage_fraction"] == 0.5
    assert report["8"]["unmeasured_prefixes_assumed_saving_seconds"] == 0.0
    assert not report["8"]["tail_component_headroom_pass"]
    assert not report["16"]["all_correctness_pass"]
    assert not report["16"]["tail_component_headroom_pass"]


def test_schedule_manifest_uses_floor_blocks_per_window_and_prefix_run() -> None:
    live = {64: 16, 128: 9}
    manifest = schedule_block_manifest(
        [
            WindowSchedule(0, 13, [64] * 7 + [128] * 5),
            WindowSchedule(1, 14, [64] * 8 + [128] * 4 + [64]),
        ],
        live_counts=live,
    )

    assert manifest["window_count"] == 2
    assert manifest["prefill_steps"] == 2
    assert manifest["decode_replays"] == 25
    assert manifest["generated_steps"] == 27
    assert manifest["run_count"] == 5
    assert manifest["blocks"]["8"]["eligible_replays"] == 8
    assert manifest["blocks"]["8"]["remainder_replays"] == 17
    assert manifest["blocks"]["16"]["eligible_replays"] == 0


def test_live_manifest_accepts_new_positive_prefixes_and_rejects_stale_shapes() -> None:
    cache = {
        (128,): {
            "active_prefix_length": 128,
            "decode_replays": 3,
            "graph": object(),
            "outputs": object(),
            "static_inputs": {},
        },
        (896,): {
            "active_prefix_length": 896,
            "decode_replays": 2,
            "graph": object(),
            "outputs": object(),
            "static_inputs": {},
        },
    }
    assert validate_live_graph_cache(cache) == {128: 3, 896: 2}
    cache[(896,)]["decode_replays"] = 0
    with pytest.raises(ValueError, match="invalid replay count"):
        validate_live_graph_cache(cache)


def test_summary_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="requires"):
        summarize(
            {},
            live_counts={64: 1},
            schedule_manifest={"blocks": {}},
        )
    with pytest.raises(ValueError, match="requires"):
        summarize(
            {"64": {}},
            live_counts={},
            schedule_manifest={"blocks": {}},
        )
