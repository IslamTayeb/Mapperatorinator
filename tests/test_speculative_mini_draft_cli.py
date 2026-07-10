from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest

from utils.verify_speculative_mini_draft import (
    EXPECTED_GAMEMODE0_TOKENIZER_SHA256,
    MINI_REVISION,
    TARGET_REPO,
    TARGET_REVISION,
    _artifact_compatibility,
    _validate_approved_contract,
)


def _artifact(*, hidden_size: int, layers: int, common_value: int = 7):
    return {
        "sha256": {
            "tokenizer.json": EXPECTED_GAMEMODE0_TOKENIZER_SHA256,
            "generation_config.json": "same-generation",
        },
        "config": {
            "model_type": "mapperatorinator",
            "backbone_model_name": "target-or-mini",
            "hidden_size": hidden_size,
            "num_hidden_layers": layers,
            "num_attention_heads": hidden_size // 64,
            "common_value": common_value,
            "backbone_config": {
                "d_model": hidden_size,
                "encoder_layers": layers,
                "decoder_layers": layers,
                "vocab_size": 4069,
            },
        },
    }


def test_artifact_compatibility_allows_only_capacity_differences():
    compatible = _artifact_compatibility(
        _artifact(hidden_size=768, layers=12),
        _artifact(hidden_size=512, layers=6),
    )
    incompatible = _artifact_compatibility(
        _artifact(hidden_size=768, layers=12),
        _artifact(hidden_size=512, layers=6, common_value=8),
    )

    assert compatible["pass"] is True
    assert compatible["non_capacity_config_match"] is True
    assert incompatible["pass"] is False
    assert incompatible["non_capacity_config_match"] is False


def test_cli_contract_rejects_any_second_shape_or_runtime_drift():
    cli = Namespace(
        config_name="profile_salvalai_smoke15",
        sequence_index=9,
        speculation_k=4,
        max_new_tokens=256,
        target_revision=TARGET_REVISION,
        mini_revision=MINI_REVISION,
    )
    args = SimpleNamespace(
        model_path=TARGET_REPO,
        seed=12345,
        precision="fp32",
        attn_implementation="sdpa",
        inference_engine="v32",
        optimized_inference_mode="single",
        gamemode=0,
        use_server=False,
        parallel=False,
        cfg_scale=1.0,
        num_beams=1,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        top_k=0,
        inference_generation_compile=False,
        inference_stateful_monotonic_logits_processor=True,
        start_time=71000,
        end_time=86000,
    )

    _validate_approved_contract(cli, args)
    cli.max_new_tokens = 255
    with pytest.raises(ValueError, match="contract changed"):
        _validate_approved_contract(cli, args)
