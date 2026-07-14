import pytest

from utils.validate_k1_remainder_profile import validate


def _runtime(*, backend: str, remainders: int = 2) -> dict:
    graph = remainders if backend == "cuda_graph" else 0
    eager = remainders if backend == "eager" else 0
    return {
        "block_size": 4,
        "remainder_steps": remainders,
        "remainder_backend": backend,
        "remainder_graph_replays": graph,
        "eager_remainder_steps": eager,
    }


def _profile(*, backend: str, remainders: int = 2) -> dict:
    return {
        "generation": [
            {
                "profile_label": label,
                "optimized_cuda_graphs": {
                    "k8_candidate": _runtime(
                        backend=backend,
                        remainders=remainders,
                    )
                },
            }
            for label in ("timing_context", "main_generation")
        ]
    }


def test_control_and_candidate_require_complete_remainder_ownership() -> None:
    control = validate(_profile(backend="eager"), role="control")
    candidate = validate(_profile(backend="cuda_graph"), role="candidate")

    assert control["labels"]["main_generation"]["eager_remainder_steps"] == 2
    assert candidate["labels"]["timing_context"]["remainder_graph_replays"] == 2


@pytest.mark.parametrize(
    ("role", "backend", "field", "value", "message"),
    (
        ("candidate", "eager", None, None, "backend must be cuda_graph"),
        ("control", "cuda_graph", None, None, "backend must be eager"),
        ("candidate", "cuda_graph", "remainder_graph_replays", 1, "accounting diverged"),
        ("candidate", "cuda_graph", "eager_remainder_steps", 1, "used an eager"),
        ("control", "eager", "eager_remainder_steps", 1, "accounting diverged"),
    ),
)
def test_invalid_remainder_ownership_fails_loudly(
    role, backend, field, value, message
) -> None:
    profile = _profile(backend=backend)
    if field is not None:
        profile["generation"][0]["optimized_cuda_graphs"]["k8_candidate"][field] = value

    with pytest.raises(ValueError, match=message):
        validate(profile, role=role)


def test_both_stages_must_have_real_remainder_opportunity() -> None:
    profile = _profile(backend="cuda_graph")
    profile["generation"][0]["optimized_cuda_graphs"]["k8_candidate"] = _runtime(
        backend="cuda_graph", remainders=0
    )

    with pytest.raises(ValueError, match="timing_context executed no remainder"):
        validate(profile, role="candidate")
