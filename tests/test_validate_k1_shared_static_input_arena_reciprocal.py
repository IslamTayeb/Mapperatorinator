from __future__ import annotations

import pytest

from utils.validate_k1_shared_static_input_arena_reciprocal import validate


def _row(label: str, *, candidate: bool):
    graphs = {
        "graph_count": 3,
        "decode_replays": 20,
        "capture_seconds": 0.2,
    }
    if candidate:
        graphs["shared_static_input_arena"] = {
            "enabled": True,
            "arena_count": 1,
            "graph_entries": 3,
            "refreshes": 17,
            "refresh_copy_calls": 68,
            "refresh_copy_bytes": 4096,
            "owned_tensor_bytes": 256,
            "avoided_duplicate_graph_input_bytes": 512,
        }
    return {
        "profile_label": label,
        "optimized_effective_config_version": "accepted",
        "optimized_dispatch_mode": "accepted_batch1",
        "optimized_dispatch_policy": {"native": True},
        "optimized_dispatch_capture_hits": {"native": 12},
        "optimized_cuda_graphs": graphs,
    }


def _profile(*, candidate: bool, precision: str = "fp32"):
    return {
        "metadata": {"precision": precision},
        "generation": [
            _row("timing_context", candidate=candidate),
            _row("main_generation", candidate=candidate),
        ],
    }


def _comparison(**overrides):
    result = {
        "same_calculation_pass": True,
        "token_equivalence_pass": True,
        "output_artifact_pass": True,
        "performance_pass": True,
    }
    result.update(overrides)
    return result


def _profiles(precision: str = "fp32"):
    return {
        "baseline_first": _profile(candidate=False, precision=precision),
        "candidate_second": _profile(candidate=True, precision=precision),
        "candidate_first": _profile(candidate=True, precision=precision),
        "baseline_second": _profile(candidate=False, precision=precision),
    }


@pytest.mark.parametrize("precision", ("fp32", "fp16"))
def test_validator_accepts_same_precision_exact_reciprocal_evidence(precision):
    result = validate(
        _comparison(),
        _profiles(precision),
        precision=precision,
    )
    assert result["status"] == "PASS"
    assert result["precision"] == precision
    assert result["dispatch_parity"] is True
    assert result["candidate_evidence"]["candidate_first"] == {
        "arena_rows": 2,
        "refreshes": 34,
        "graph_entries": 6,
        "owned_tensor_bytes": 512,
        "avoided_duplicate_graph_input_bytes": 1024,
    }


@pytest.mark.parametrize(
    "gate",
    (
        "same_calculation_pass",
        "token_equivalence_pass",
        "output_artifact_pass",
        "performance_pass",
    ),
)
def test_validator_rejects_every_reciprocal_gate(gate):
    with pytest.raises(ValueError, match=gate):
        validate(
            _comparison(**{gate: False}),
            _profiles(),
            precision="fp32",
        )


def test_validator_rejects_baseline_contamination_and_dispatch_drift():
    profiles = _profiles()
    profiles["baseline_first"] = _profile(candidate=True)
    with pytest.raises(ValueError, match="baseline unexpectedly"):
        validate(_comparison(), profiles, precision="fp32")

    profiles = _profiles()
    profiles["candidate_first"]["generation"][0][
        "optimized_dispatch_capture_hits"
    ] = {"native": 11}
    with pytest.raises(ValueError, match="dispatch parity"):
        validate(_comparison(), profiles, precision="fp32")


def test_validator_allows_prefill_only_row_but_requires_real_refresh():
    profiles = _profiles()
    for label in ("candidate_first", "candidate_second"):
        profiles[label]["generation"].append(
            {
                **_row("main_generation", candidate=False),
                "optimized_cuda_graphs": {"graph_count": 0},
            }
        )
    for label in ("baseline_first", "baseline_second"):
        profiles[label]["generation"].append(
            {
                **_row("main_generation", candidate=False),
                "optimized_cuda_graphs": {"graph_count": 0},
            }
        )
    assert validate(_comparison(), profiles, precision="fp32")["status"] == "PASS"

    profiles = _profiles()
    for label in ("candidate_first", "candidate_second"):
        for row in profiles[label]["generation"]:
            row["optimized_cuda_graphs"]["shared_static_input_arena"][
                "refreshes"
            ] = 0
    with pytest.raises(ValueError, match="did not exercise arena refresh"):
        validate(_comparison(), profiles, precision="fp32")
