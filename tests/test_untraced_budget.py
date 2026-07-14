from __future__ import annotations

import json

import pytest
import torch

from osuT5.osuT5.inference.optimized.single import budget as budget_module
from osuT5.osuT5.inference.optimized.single.budget import (
    BUDGET_REGION_NAMES,
    UntracedBudgetRecorder,
)
from utils.summarize_untraced_budget import summarize


def _finished_budget(monkeypatch, *, elapsed: float = 1.0, tokens: int = 10):
    monkeypatch.setattr(budget_module.torch.cuda, "is_available", lambda: False)
    recorder = UntracedBudgetRecorder()
    with recorder.region("prefill"):
        pass
    return recorder.finish(
        model_elapsed_seconds=elapsed,
        generated_tokens=tokens,
    )


def _profile_payload(budget):
    return {
        "schema_version": 1,
        "metadata": {
            "profile_pass_kind": "untraced_budget",
            "profile_detail_ranges": False,
            "profile_cuda_capture": False,
        },
        "generation": [
            {
                "profile_label": "main_generation",
                "context_type": "MAP",
                "generated_tokens": budget["generated_tokens"],
                "model_elapsed_seconds": budget["model_elapsed_seconds"],
                "untraced_budget": budget,
            }
        ],
    }


def test_recorder_is_complete_per_token_and_tracks_explicit_copy(monkeypatch):
    monkeypatch.setattr(budget_module.torch.cuda, "is_available", lambda: False)
    recorder = UntracedBudgetRecorder()
    source = torch.ones(8, dtype=torch.float32)
    destination = torch.empty_like(source)

    with recorder.region("graph_input_copies"):
        destination.copy_(source)
        recorder.record_copy(source, destination)
    recorder.record_external_copy(
        "model_input_to_device",
        source,
        destination,
        host_wall_seconds=0.01,
    )
    result = recorder.finish(model_elapsed_seconds=1.0, generated_tokens=4)

    assert tuple(result["regions"]) == BUDGET_REGION_NAMES
    copy = result["regions"]["graph_input_copies"]
    assert copy["calls"] == 1
    assert copy["copy_count"] == 1
    assert copy["copy_bytes"] == 32
    assert copy["copy_count_by_direction"]["h2h"] == 1
    assert result["external_transfers"]["model_input_to_device"][
        "host_wall_seconds"
    ] == pytest.approx(0.01)
    assert result["per_token"]["model_elapsed_ms"] == pytest.approx(250.0)
    assert result["reconciliation_pass"] is True


def test_recorder_rejects_overlap_and_double_finish(monkeypatch):
    monkeypatch.setattr(budget_module.torch.cuda, "is_available", lambda: False)
    recorder = UntracedBudgetRecorder()
    with recorder.region("prefill"):
        with pytest.raises(RuntimeError, match="must not overlap"):
            with recorder.region("sampling"):
                pass
    recorder.finish(model_elapsed_seconds=1.0, generated_tokens=1)
    with pytest.raises(RuntimeError, match="only finish once"):
        recorder.finish(model_elapsed_seconds=1.0, generated_tokens=1)


def test_reconciliation_fails_above_two_percent(monkeypatch):
    monkeypatch.setattr(budget_module.torch.cuda, "is_available", lambda: False)
    clock = iter((0.0, 1.1))
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: next(clock))
    recorder = UntracedBudgetRecorder()
    with recorder.region("prefill"):
        pass
    result = recorder.finish(model_elapsed_seconds=1.0, generated_tokens=1)

    assert result["reconciliation_error_fraction"] == pytest.approx(0.1)
    assert result["reconciliation_pass"] is False


def test_summarizer_validates_schema_and_reconciliation(monkeypatch, tmp_path):
    budget = _finished_budget(monkeypatch)
    payload = _profile_payload(budget)
    report = summarize(payload)

    assert report["pass"] is True
    assert report["overall"]["records"] == 1
    assert report["overall"]["generated_tokens"] == 10
    assert report["overall"]["reconciliation_error_fraction"] <= 0.02

    payload_path = tmp_path / "profile.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    broken = json.loads(payload_path.read_text(encoding="utf-8"))
    broken["generation"][0]["untraced_budget"]["regions"].pop("sampling")
    with pytest.raises(ValueError, match="complete budget schema"):
        summarize(broken)


def test_summarizer_rejects_traced_or_control_profiles(monkeypatch):
    payload = _profile_payload(_finished_budget(monkeypatch))
    payload["metadata"]["profile_pass_kind"] = "untraced_control"
    with pytest.raises(ValueError, match="untraced_budget"):
        summarize(payload)
