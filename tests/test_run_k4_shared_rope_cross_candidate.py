import json

import pytest

from osuT5.osuT5.inference.optimized.kernels.weight_only_runtime import (
    CROSS_FP16_PACKED,
)
from utils import run_k4_shared_rope_cross_candidate as candidate


def test_cross_runner_composes_k1_int8_and_records_incremental_mode(
    monkeypatch,
    tmp_path,
) -> None:
    mode = CROSS_FP16_PACKED
    calls = []
    def weight_run(
        config_name,
        overrides,
        output_init_json,
        *,
        initializer_name,
        initializer_kwargs,
    ):
        assert config_name == "profile_salvalai"
        assert overrides == ["seed=12345"]
        assert initializer_name == "initialize_approximate_int8_mlp_weight_only_cross"
        assert initializer_kwargs == {"mode": mode}
        output_init_json.write_text(
            json.dumps(
                {
                    "result_class": "documented-drift",
                    "exactness_claim": False,
                    "cross_candidate": {
                        "mode": mode,
                        "scope": "main-model-only",
                        "attention_accumulation": "fp32",
                        "production_selector_unchanged": True,
                    },
                    "int8_mlp_overlay": {
                        "dispatch_counter": "int8_weight_mlp_tail",
                    },
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(candidate, "run_with_initializer", weight_run)

    def combined_run(
        config_name,
        overrides,
        output_init_json,
        *,
        graph_remainders,
        weight_runner,
        composition_version,
        shared_static_input_arena,
        transition_timing,
    ):
        assert shared_static_input_arena is False
        assert transition_timing is False
        calls.append(
            (
                config_name,
                overrides,
                graph_remainders,
                composition_version,
            )
        )
        weight_runner(config_name, overrides, output_init_json)
        payload = json.loads(output_init_json.read_text(encoding="utf-8"))
        payload["combined_runtime"] = composition_version
        payload["shared_rope"] = {"scope": "main-model-only", "stats": {"reuses": 22}}
        output_init_json.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(candidate, "run_combined", combined_run)
    output = tmp_path / "init.json"

    candidate.run(
        "profile_salvalai",
        ["seed=12345"],
        output,
        mode=mode,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["cross_runtime"]["mode"] == mode
    assert payload["cross_runtime"]["incremental_control"] == (
        "k4-split-kv-mixed-weight-shared-rope-k1-remainder-int8-mlp-v1"
    )
    assert payload["cross_runtime"]["incremental_exactness_required"] is True
    assert payload["cross_runtime"]["packed_projection_delta_only"] is True
    assert payload["cross_runtime"]["accepted_q1_bmm_required"] is True
    assert payload["cross_runtime"]["original_decoder_forward_required"] is True
    assert payload["combined_runtime"].endswith(f"int8-mlp-{mode}-v1")
    assert payload["shared_rope"]["stats"]["reuses"] == 22
    assert calls == [
        (
            "profile_salvalai",
            ["seed=12345"],
            True,
            payload["combined_runtime"],
        )
    ]


def test_dp4a_cross_runner_records_selected_overlay(monkeypatch, tmp_path) -> None:
    calls = []

    def weight_run(
        config_name,
        overrides,
        output_init_json,
        *,
        initializer_name,
        initializer_kwargs,
    ):
        assert initializer_name.endswith("cross_dp4a_self_qkv")
        assert initializer_kwargs == {"mode": CROSS_FP16_PACKED}
        output_init_json.write_text(
            json.dumps(
                {
                    "cross_candidate": {"mode": CROSS_FP16_PACKED},
                    "dp4a_self_qkv_overlay": {
                        "dispatch_counter": "dp4a_self_qkv_projection",
                        "persistent_ctas": 68,
                    },
                }
            ),
            encoding="utf-8",
        )

    def combined_run(
        config_name,
        overrides,
        output_init_json,
        *,
        graph_remainders,
        weight_runner,
        composition_version,
        shared_static_input_arena,
        transition_timing,
    ):
        calls.append((shared_static_input_arena, composition_version))
        weight_runner(config_name, overrides, output_init_json)
        payload = json.loads(output_init_json.read_text(encoding="utf-8"))
        payload["combined_runtime"] = composition_version
        payload["shared_rope"] = {"scope": "main-model-only"}
        output_init_json.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(candidate, "run_with_initializer", weight_run)
    monkeypatch.setattr(candidate, "run_combined", combined_run)
    output = tmp_path / "dp4a.json"
    candidate.run(
        "profile_salvalai",
        [],
        output,
        mode=CROSS_FP16_PACKED,
        shared_static_input_arena=True,
        dp4a_self_qkv=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["dp4a_runtime"]["dispatch_counter"] == (
        "dp4a_self_qkv_projection"
    )
    assert payload["dp4a_runtime"]["full_graph_measurement_required"] is True
    assert calls == [(True, payload["combined_runtime"])]


@pytest.mark.parametrize("mode", ["accepted", "split8_attention", "bad"])
def test_cross_runner_rejects_unselected_mode_before_loading(tmp_path, mode) -> None:
    with pytest.raises(ValueError, match="cross candidate mode"):
        candidate.run(
            "profile_salvalai",
            [],
            tmp_path / "init.json",
            mode=mode,
        )
