from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from utils.profile_device_resident_tail import (
    PROMOTION_SAVING_SECONDS,
    TailAnchor,
    _aggregate_anchor_samples,
    _BoundedAnchorSampler,
    _RawTailCapture,
    _bucketed_prefix,
    _eager_dynamic_tail,
    _is_main_capture_label,
    _mutable_tensor_templates,
    _post_block_status_report,
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


def _minimal_anchor(
    *,
    window_index: int,
    token: int = 1,
    eos: int = 9,
) -> TailAnchor:
    scores = torch.full((1, 11), -100.0)
    scores[:, token] = 100.0
    return TailAnchor(
        prefix=64,
        window_index=window_index,
        block_ordinal=-1,
        input_ids=torch.tensor([[3, 4]]),
        raw_logits_steps=[scores],
        rng_state=(torch.random.get_rng_state(), None),
        logits_processors=LogitsProcessorList(),
        stopping_criteria=[_NeverStop()],
        do_sample=False,
        pad_token_id=0,
        eos_token_ids=torch.tensor([eos]),
        max_length=128,
    )


def test_bounded_sampler_observes_whole_stream_and_retains_first_interior_last() -> None:
    def sample_ordinals():
        sampler = _BoundedAnchorSampler(prefix=64, block_size=8, limit=3)
        for ordinal in range(100):
            sampler.consider(_minimal_anchor(window_index=ordinal // 9))
        selected = sampler.selected()
        assert sampler.completed_blocks == 100
        assert sampler.retained_payloads == 3
        assert [role for role, _ in selected] == [
            "first",
            "deterministic_interior",
            "latest",
        ]
        assert selected[0][1].block_ordinal == 0
        assert 0 < selected[1][1].block_ordinal < 99
        assert selected[-1][1].block_ordinal == 99
        assert selected[-1][1].window_index == 11
        return [anchor.block_ordinal for _, anchor in selected]

    assert sample_ordinals() == sample_ordinals()


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
    sampler = anchors[4][64]
    assert sampler.completed_blocks == 1
    assert sampler.retained_payloads == 1
    [(role, anchor)] = sampler.selected()
    assert role == "first"
    assert anchor.block_ordinal == 0
    assert anchor.window_index == 7
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


def test_eager_control_records_first_logical_stop_length() -> None:
    class _EosStop:
        def __call__(self, input_ids, scores):
            del scores
            return input_ids[:, -1] == 9

    anchor = _minimal_anchor(window_index=0, token=9)
    second = torch.full((1, 11), -100.0)
    second[:, 7] = 100.0
    anchor.raw_logits_steps.append(second)
    anchor.stopping_criteria = [_EosStop()]

    eager = _eager_dynamic_tail(
        anchor,
        block_size=2,
        processors=LogitsProcessorList(),
        stopping_criteria=_EosStop(),
    )

    assert eager[0].tolist() == [[3, 4, 9, 0]]
    assert eager[2].tolist() == [False]
    assert eager[4].tolist() == [3]


def test_post_block_status_report_requires_one_consistent_matching_read() -> None:
    matching = _post_block_status_report(
        expected_unfinished=False,
        timing={
            "returned_post_block_unfinished": False,
            "returned_post_block_status_consistent": True,
            "post_block_status_reads": 1,
        },
    )
    mismatching = _post_block_status_report(
        expected_unfinished=True,
        timing={
            "returned_post_block_unfinished": False,
            "returned_post_block_status_consistent": True,
            "post_block_status_reads": 1,
        },
    )

    assert matching["parity"] is True
    assert matching["reads_per_block"] == 1
    assert mismatching["parity"] is False


def test_capture_accounting_is_tail_only_and_never_claims_a_vram_delta() -> None:
    aggregate = _aggregate_anchor_samples(
        [
            {
                "pass": True,
                "saved_wall_us_per_token": 8.0,
                "tail_only_capture_setup_seconds": 0.2,
                "absolute_peak_vram_bytes": 100,
            },
            {
                "pass": True,
                "saved_wall_us_per_token": 7.0,
                "tail_only_capture_setup_seconds": 0.3,
                "absolute_peak_vram_bytes": 120,
            },
        ]
    )

    assert aggregate["tail_only_capture_setup_seconds_sample_sum"] == pytest.approx(
        0.5
    )
    assert aggregate["absolute_peak_vram_bytes_max"] == 120
    assert aggregate["capture_vram_delta_available"] is False
    assert aggregate["capture_vram_delta_bytes"] is None


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
