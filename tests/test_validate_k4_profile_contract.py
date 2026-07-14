import pytest

from utils.validate_k4_profile_contract import RNG_POLICY, validate


def _candidate(block_replays: int = 2) -> dict:
    eligible = 4 * block_replays
    logical = 1 + eligible
    return {
        "block_size": 4,
        "prefill_steps": 1,
        "eligible_steps": eligible,
        "block_replays": block_replays,
        "remainder_steps": 0,
        "physical_steps": logical,
        "logical_steps": logical,
        "wasted_steps": 0,
        "rng_policy": RNG_POLICY,
        "rng_exact": False,
        "rng_early_eos_isolation": True,
        "capture_state_restore_synchronized": True,
    }


def _profile(*, candidate: bool = True, block_replays: int = 2) -> dict:
    records = []
    for label in ("timing_context", "main_generation"):
        graphs = {}
        generated_tokens = 1 + 4 * block_replays
        if candidate:
            graphs["k8_candidate"] = _candidate(block_replays)
        records.append(
            {
                "profile_label": label,
                "generated_tokens": generated_tokens,
                "optimized_cuda_graphs": graphs,
            }
        )
    return {"generation": records}


def test_candidate_requires_positive_k4_usage_in_both_stages() -> None:
    report = validate(_profile(), role="candidate", block_size=4)

    assert report["pass"] is True
    assert report["labels"]["timing_context"]["eligible_steps"] == 8
    assert report["labels"]["main_generation"]["block_replays"] == 2
    assert report["labels"]["main_generation"]["physical_steps"] == 9
    assert report["labels"]["main_generation"]["logical_steps"] == 9


def test_baseline_forbids_k4_metadata() -> None:
    assert validate(_profile(candidate=False), role="baseline", block_size=4)["pass"]
    with pytest.raises(ValueError, match="baseline .* contains K4 metadata"):
        validate(_profile(candidate=True), role="baseline", block_size=4)


def test_candidate_fails_on_silent_all_remainder_fallback() -> None:
    with pytest.raises(ValueError, match="did not execute K4 blocks"):
        validate(_profile(block_replays=0), role="candidate", block_size=4)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("block_size", 8, "wrong block_size"),
        ("rng_policy", "wrong", "wrong RNG policy"),
        ("rng_exact", True, "declare RNG drift"),
        ("physical_steps", 99, "physical/logical accounting diverged"),
        ("eligible_steps", 7, "block accounting diverged"),
    ),
)
def test_candidate_fails_loudly_on_invalid_contract(field, value, message) -> None:
    profile = _profile()
    profile["generation"][0]["optimized_cuda_graphs"]["k8_candidate"][field] = value

    with pytest.raises(ValueError, match=message):
        validate(profile, role="candidate", block_size=4)


def test_candidate_logical_steps_match_profile_generated_tokens() -> None:
    profile = _profile()
    profile["generation"][0]["generated_tokens"] += 1

    with pytest.raises(ValueError, match="do not match generated tokens"):
        validate(profile, role="candidate", block_size=4)
