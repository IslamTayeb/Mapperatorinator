from copy import deepcopy

import pytest

from utils import validate_k1_int8_stage_precision_profile as validator


def _runtime(*, label: str) -> dict:
    main = label == "main_generation"
    return {
        "block_size": 4 if main else 1,
        "remainder_backend": "cuda_graph",
        "remainder_steps": 2 if main else 0,
        "remainder_graph_replays": 2 if main else 0,
        "eager_remainder_steps": 0,
    }


def _profile() -> dict:
    return {
        "generation": [
            {
                "profile_label": label,
                "optimized_cuda_graphs": {"k8_candidate": _runtime(label=label)},
            }
            for label in ("timing_context", "main_generation")
        ]
    }


def test_composition_validator_requires_all_four_owners(monkeypatch):
    calls = []
    monkeypatch.setattr(
        validator,
        "validate_stage",
        lambda payload, *, role: calls.append(("stage", role)) or {"pass": True},
    )
    monkeypatch.setattr(
        validator,
        "validate_k4",
        lambda payload, **kwargs: calls.append(("k4", kwargs)) or {"pass": True},
    )
    monkeypatch.setattr(
        validator,
        "validate_int8",
        lambda payload, *, role: calls.append(("int8", role)) or {"pass": True},
    )

    report = validator.validate(_profile(), role="candidate")

    assert report["pass"] is True
    assert report["remainder_ownership"]["main_generation"][
        "remainder_graph_replays"
    ] == 2
    assert calls == [
        ("stage", "candidate"),
        (
            "k4",
            {"role": "candidate", "block_size": 4, "timing_block_size": 1},
        ),
        ("int8", "candidate"),
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("remainder_backend", "eager", "did not enable"),
        ("eager_remainder_steps", 1, "used an eager"),
        ("remainder_graph_replays", 1, "accounting diverged"),
    ),
)
def test_remainder_ownership_fails_loudly(field, value, message):
    payload = deepcopy(_profile())
    payload["generation"][1]["optimized_cuda_graphs"]["k8_candidate"][field] = value

    with pytest.raises(ValueError, match=message):
        validator._validate_remainder_ownership(payload)


def test_main_must_exercise_a_captured_k1_remainder():
    payload = _profile()
    runtime = payload["generation"][1]["optimized_cuda_graphs"]["k8_candidate"]
    runtime["remainder_steps"] = 0
    runtime["remainder_graph_replays"] = 0

    with pytest.raises(ValueError, match="no captured K1"):
        validator._validate_remainder_ownership(payload)
