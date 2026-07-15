import pytest

from utils.run_exact_rope_device_state import _activation_payload, _parse_args


def test_combined_activation_payload_preserves_decoder_and_dispatch() -> None:
    assert _activation_payload(87) == {
        "accepted_dispatch_replaced": False,
        "candidate": "shared_rope_plus_preallocated_batch1_device_sequence_state",
        "decode_loop_calls": 87,
        "decoder_layer_replaced": False,
        "device_sequence_state_enabled": True,
        "enabled": True,
        "schema_version": 1,
        "shared_rope_enabled": True,
    }


@pytest.mark.parametrize("calls", [0, -1, True, 1.5])
def test_combined_activation_payload_rejects_invalid_counts(calls) -> None:
    with pytest.raises(RuntimeError, match="never invoked"):
        _activation_payload(calls)


def test_combined_runner_requires_both_evidence_paths() -> None:
    parsed = _parse_args(
        [
            "--rope-evidence-path",
            "/tmp/rope.json",
            "--activation-manifest",
            "/tmp/state.json",
            "precision=fp32",
        ]
    )
    assert parsed.config_name == "profile_salvalai"
    assert str(parsed.rope_evidence_path) == "/tmp/rope.json"
    assert str(parsed.activation_manifest) == "/tmp/state.json"
    assert parsed.overrides == ["precision=fp32"]
