from __future__ import annotations

import pytest

from utils.analyze_conditional_while_multinomial_probe import (
    INVALID_CONTROL,
    POSITIVE,
    VALID_NEGATIVE,
    analyze,
)


def _payload() -> dict:
    samples = [1, 2, 3, 4]
    state = "a" * 64
    return {
        "backend": "cuda_python_conditional_while_child_graphs",
        "rng_policy": "torch_multinomial_philox",
        "torch_version": "2.10.0+cu128",
        "torch_cuda_version": "12.8",
        "cuda_runtime_version": 12080,
        "cuda_driver_version": 12080,
        "device_name": "NVIDIA GeForce RTX 2080 Ti",
        "device_capability": [7, 5],
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "nvidia_tf32_override": "0",
        "seed": 12345,
        "iterations": 4,
        "eager_samples": samples,
        "ordinary_graph_samples": samples,
        "conditional_while_samples": samples,
        "eager_generator_state_sha256": state,
        "ordinary_graph_generator_state_sha256": state,
        "conditional_generator_state_sha256": state,
        "ordinary_matches_eager": True,
        "ordinary_generator_matches_eager": True,
        "conditional_matches_ordinary": True,
        "conditional_generator_matches_ordinary": True,
        "exact_conditional_while_feasible": True,
    }


def test_positive_result_requires_probe_exit_zero() -> None:
    report, exit_code = analyze(_payload(), probe_exit_code=0)

    assert report["classification"] == POSITIVE
    assert report["ordinary_control_pass"] is True
    assert report["exact_conditional_while_feasible"] is True
    assert exit_code == 0


def test_conditional_rng_mismatch_is_a_valid_negative() -> None:
    payload = _payload()
    payload["conditional_while_samples"] = [1, 1, 1, 1]
    payload["conditional_generator_state_sha256"] = "b" * 64
    payload["conditional_matches_ordinary"] = False
    payload["conditional_generator_matches_ordinary"] = False
    payload["exact_conditional_while_feasible"] = False

    report, exit_code = analyze(payload, probe_exit_code=2)

    assert report["classification"] == VALID_NEGATIVE
    assert report["ordinary_control_pass"] is True
    assert report["conditional_token_match"] is False
    assert report["conditional_generator_match"] is False
    assert exit_code == 0


def test_ordinary_graph_mismatch_is_an_invalid_control() -> None:
    payload = _payload()
    payload["ordinary_graph_samples"] = [4, 3, 2, 1]
    payload["ordinary_matches_eager"] = False
    payload["conditional_matches_ordinary"] = False
    payload["exact_conditional_while_feasible"] = False

    report, exit_code = analyze(payload, probe_exit_code=2)

    assert report["classification"] == INVALID_CONTROL
    assert report["ordinary_control_pass"] is False
    assert exit_code == 3


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("float32_matmul_precision", "high", "matmul precision"),
        ("cuda_matmul_allow_tf32", True, "CUDA matmul TF32"),
        ("nvidia_tf32_override", "1", "NVIDIA_TF32_OVERRIDE"),
        ("device_capability", [8, 0], "SM75"),
    ),
)
def test_environment_schema_fails_loudly(field, value, message) -> None:
    payload = _payload()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        analyze(payload, probe_exit_code=0)


def test_exit_code_must_match_the_recorded_result() -> None:
    with pytest.raises(ValueError, match="probe exit code"):
        analyze(_payload(), probe_exit_code=2)
