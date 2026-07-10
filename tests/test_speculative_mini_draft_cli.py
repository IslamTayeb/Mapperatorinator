from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest

from inference import compile_args
from osuT5.osuT5.inference.optimized.speculative import mini_draft_gpu
from utils.verify_speculative_mini_draft import (
    EXPECTED_GAMEMODE0_TOKENIZER_SHA256,
    MINI_REVISION,
    TARGET_REPO,
    TARGET_REVISION,
    _artifact_compatibility,
    _load_pinned_models,
    _load_args,
    _validate_approved_contract,
)


def test_scout_owns_stateful_monotonic_processor_without_public_legacy_flag(
    monkeypatch,
):
    calls = []

    def fake_builder(*args, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(
        mini_draft_gpu,
        "build_logits_processor_list",
        fake_builder,
    )
    args = SimpleNamespace(
        timeshift_bias=0.0,
        train=SimpleNamespace(data=SimpleNamespace(types_first=False)),
        temperature=0.9,
        timing_temperature=0.9,
        mania_column_temperature=0.9,
        taiko_hit_temperature=0.9,
        cfg_scale=1.0,
        top_k=0,
        top_p=1.0,
        inference_stateful_monotonic_logits_processor=False,
    )

    mini_draft_gpu._greedy_processors(
        args,
        object(),
        "cpu",
        lookback_time=0.0,
    )
    mini_draft_gpu._target_sampling_processors(
        args,
        object(),
        "cpu",
        lookback_time=0.0,
    )

    assert len(calls) == 2
    assert all(call["stateful_monotonic"] is True for call in calls)


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
        inference_active_prefix_decode_loop=False,
        inference_stateful_monotonic_logits_processor=False,
        start_time=71000,
        end_time=86000,
        device="cuda",
    )

    _validate_approved_contract(cli, args)
    cli.max_new_tokens = 255
    with pytest.raises(ValueError, match="contract changed"):
        _validate_approved_contract(cli, args)


def test_pinned_models_use_exact_resolved_paths_and_disable_local_gamemode_reselection(tmp_path):
    target_path = tmp_path / "target" / "gamemode=0"
    mini_path = tmp_path / "mini" / "gamemode=0"
    calls = []

    def fake_loader(path, **kwargs):
        calls.append((path, kwargs))
        return f"model-{len(calls)}", f"tokenizer-{len(calls)}"

    args = SimpleNamespace(
        train="train-config",
        device="cpu",
        max_batch_size=1,
        precision="fp32",
        attn_implementation="sdpa",
        gamemode=0,
    )
    loaded = _load_pinned_models(
        args,
        {"resolved_path": str(target_path)},
        {"resolved_path": str(mini_path)},
        loader=fake_loader,
    )

    assert loaded == (("model-1", "tokenizer-1"), ("model-2", "tokenizer-2"))
    assert [call[0] for call in calls] == [target_path, mini_path]
    assert all(call[1]["auto_select_gamemode_model"] is False for call in calls)
    assert all(call[1]["gamemode"] == 0 for call in calls)


def test_loader_preflight_contract_requires_cpu_and_remains_distinct_from_gpu_gate():
    cli = Namespace(
        config_name="profile_salvalai_smoke15",
        sequence_index=9,
        speculation_k=4,
        max_new_tokens=256,
        target_revision=TARGET_REVISION,
        mini_revision=MINI_REVISION,
        loaders_only=True,
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
        inference_active_prefix_decode_loop=False,
        inference_stateful_monotonic_logits_processor=False,
        start_time=71000,
        end_time=86000,
        device="cpu",
    )

    _validate_approved_contract(cli, args)
    args.device = "cuda"
    with pytest.raises(ValueError, match="contract changed"):
        _validate_approved_contract(cli, args)


def test_exact_sbatch_hydra_contract_passes_runtime_flag_validation(tmp_path):
    audio_path = tmp_path / "salvalai.mp3"
    audio_path.touch()
    args = _load_args(
        "profile_salvalai_smoke15",
        [
            f"audio_path={audio_path}",
            f"output_path={tmp_path / 'output'}",
            "device=cuda",
            "precision=fp32",
            "attn_implementation=sdpa",
            "inference_engine=v32",
            "optimized_inference_mode=single",
            "use_server=false",
            "parallel=false",
            "cfg_scale=1.0",
            "num_beams=1",
            "seed=12345",
            "inference_generation_compile=false",
            "inference_active_prefix_decode_loop=false",
            "inference_stateful_monotonic_logits_processor=false",
        ],
    )
    cli = Namespace(
        config_name="profile_salvalai_smoke15",
        sequence_index=9,
        speculation_k=4,
        max_new_tokens=256,
        target_revision=TARGET_REVISION,
        mini_revision=MINI_REVISION,
    )

    _validate_approved_contract(cli, args)
    compile_args(args, verbose=False)
