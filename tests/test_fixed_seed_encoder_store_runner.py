from types import SimpleNamespace

import pytest

from utils.run_batched_encoder_store_full_song import _validate_args


def _args(**overrides):
    values = {
        "inference_engine": "optimized",
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "device": "cuda",
        "use_server": False,
        "parallel": False,
        "cfg_scale": 1.0,
        "num_beams": 1,
        "super_timing": False,
        "generate_positions": False,
        "profile_inference": True,
        "profile_pass_kind": "untraced_control",
        "max_batch_size": 32,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
@pytest.mark.parametrize("pass_kind", ["untraced_control", "exactness_audit"])
def test_runner_accepts_independent_same_precision_gates(precision, pass_kind):
    _validate_args(
        _args(precision=precision, profile_pass_kind=pass_kind),
        mode="candidate",
        batch_size=16,
    )


def test_runner_rejects_nonaccepted_configs_and_batch_overflow():
    with pytest.raises(ValueError, match="fp32 or fp16"):
        _validate_args(_args(precision="bf16"), mode="candidate", batch_size=16)
    with pytest.raises(ValueError, match="untraced_control or exactness_audit"):
        _validate_args(
            _args(profile_pass_kind="graph_trace"),
            mode="candidate",
            batch_size=16,
        )
    with pytest.raises(ValueError, match="max_batch_size"):
        _validate_args(_args(max_batch_size=8), mode="candidate", batch_size=16)
