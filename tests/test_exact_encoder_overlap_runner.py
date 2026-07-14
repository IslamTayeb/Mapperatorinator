from types import SimpleNamespace

import pytest

from utils.run_exact_encoder_overlap_full_song import _validate_args


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
        "auto_select_gamemode_model": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_runner_accepts_only_standard_salvalai_exact_control():
    _validate_args(_args(), mode="baseline")
    _validate_args(_args(), mode="candidate")
    with pytest.raises(ValueError, match="args changed"):
        _validate_args(_args(precision="fp16"), mode="candidate")
    with pytest.raises(ValueError, match="args changed"):
        _validate_args(
            _args(auto_select_gamemode_model=False),
            mode="candidate",
        )
    with pytest.raises(ValueError, match="mode"):
        _validate_args(_args(), mode="other")
