import pytest

from utils.run_exact_rope_device_state import (
    _activation_payload,
    _parse_args,
    _run_setup_and_fingerprint,
)


def test_combined_activation_payload_preserves_decoder_and_dispatch() -> None:
    seed = {"python_sha256": "a"}
    final = {"python_sha256": "b"}
    assert _activation_payload(
        mode="candidate",
        calls=87,
        rng_after_seed=seed,
        rng_after_inference=final,
    ) == {
        "accepted_dispatch_replaced": False,
        "candidate": "shared_rope_plus_preallocated_batch1_device_sequence_state",
        "decode_loop_calls": 87,
        "decoder_layer_replaced": False,
        "device_sequence_state_enabled": True,
        "enabled": True,
        "rng_after_inference": final,
        "rng_after_seed": seed,
        "role": "candidate",
        "schema_version": 1,
        "shared_rope_enabled": True,
    }


@pytest.mark.parametrize("calls", [0, -1, True, 1.5])
def test_combined_activation_payload_rejects_invalid_counts(calls) -> None:
    with pytest.raises(RuntimeError, match="count|activation"):
        _activation_payload(
            mode="candidate",
            calls=calls,
            rng_after_seed={},
            rng_after_inference={},
        )


def test_combined_baseline_requires_zero_candidate_calls() -> None:
    payload = _activation_payload(
        mode="baseline",
        calls=0,
        rng_after_seed={},
        rng_after_inference={},
    )
    assert payload["enabled"] is False
    assert payload["device_sequence_state_enabled"] is False
    with pytest.raises(RuntimeError, match="activation"):
        _activation_payload(
            mode="baseline",
            calls=1,
            rng_after_seed={},
            rng_after_inference={},
        )


def test_combined_runner_requires_both_evidence_paths() -> None:
    parsed = _parse_args(
        [
            "--rope-evidence-path",
            "/tmp/rope.json",
            "--evidence-manifest",
            "/tmp/state.json",
            "--mode",
            "candidate",
            "precision=fp32",
        ]
    )
    assert parsed.config_name == "profile_salvalai"
    assert str(parsed.rope_evidence_path) == "/tmp/rope.json"
    assert str(parsed.evidence_manifest) == "/tmp/state.json"
    assert parsed.mode == "candidate"
    assert parsed.overrides == ["precision=fp32"]


def test_setup_observer_preserves_precision_policy_arguments(monkeypatch) -> None:
    calls = []
    expected = {"python_sha256": "a", "torch_cuda_sha256": ["b"]}
    monkeypatch.setattr(
        "utils.run_device_sequence_state_candidate._rng_fingerprint",
        lambda: expected,
    )

    def setup(seed, *, strict_fp32):
        calls.append((seed, strict_fp32))

    assert _run_setup_and_fingerprint(setup, 12345, strict_fp32=False) is expected
    assert calls == [(12345, False)]
