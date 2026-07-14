import pytest

from utils.validate_encoder_stabilization_profile import (
    EncoderStabilizationProfileError,
    validate_profile,
)


def _profile(*, skipped: int = 3) -> dict:
    metadata = {
        "profile_pass_kind": "untraced_control",
        "authoritative_performance": True,
        "profile_detail_ranges": False,
        "profile_cuda_capture": False,
    }
    records = []
    for label in ("timing_context", "main_generation"):
        records.append(
            {
                "profile_label": label,
                "decoder_loop_backend": "active_prefix_cuda_graph",
                "encoder_stabilization": {
                    "calls": skipped + 1,
                    "skipped_copies": skipped,
                    "skipped_identity_copies": skipped - 1,
                    "skipped_shared_storage_copies": 1,
                    "executed_copies": 1,
                    "executed_bytes": 4096,
                    "holder_replacements": 1,
                },
            }
        )
    return {"schema_version": 1, "metadata": metadata, "generation": records}


def test_candidate_profile_requires_positive_identity_or_is_set_to_skips() -> None:
    report = validate_profile(_profile())

    assert report["pass"] is True
    assert report["records"] == 2
    assert report["totals"]["skipped_copies"] == 6
    assert report["skip_policy"] == {
        "identity_only": True,
        "shared_storage_requires_is_set_to": True,
        "other_skip_reasons": 0,
    }


def test_candidate_profile_rejects_unreconciled_skip_reasons() -> None:
    profile = _profile()
    profile["generation"][0]["encoder_stabilization"][
        "skipped_identity_copies"
    ] = 0

    with pytest.raises(EncoderStabilizationProfileError, match="identity or exact"):
        validate_profile(profile)


def test_candidate_profile_rejects_zero_skip_and_traced_profile() -> None:
    no_skip = _profile(skipped=0)
    for record in no_skip["generation"]:
        record["encoder_stabilization"]["skipped_identity_copies"] = 0
        record["encoder_stabilization"]["skipped_shared_storage_copies"] = 0
    with pytest.raises(EncoderStabilizationProfileError, match="did not skip"):
        validate_profile(no_skip)

    traced = _profile()
    traced["metadata"]["profile_detail_ranges"] = True
    with pytest.raises(EncoderStabilizationProfileError, match="detail ranges"):
        validate_profile(traced)


def test_candidate_profile_requires_both_generation_stages() -> None:
    profile = _profile()
    profile["generation"] = profile["generation"][:1]

    with pytest.raises(EncoderStabilizationProfileError, match="must cover"):
        validate_profile(profile)
