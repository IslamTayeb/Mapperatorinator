import json

import pytest

from osuT5.osuT5.inference.optimized.kernels.weight_only_runtime import (
    CROSS_FP16_PACKED,
    CROSS_SPLIT8,
)
from utils import run_k4_shared_rope_cross_candidate as candidate


@pytest.mark.parametrize("mode", [CROSS_FP16_PACKED, CROSS_SPLIT8])
def test_cross_runner_composes_k1_int8_and_records_incremental_mode(
    monkeypatch,
    tmp_path,
    mode,
) -> None:
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
    ):
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


def test_cross_runner_rejects_control_mode_before_loading(tmp_path) -> None:
    with pytest.raises(ValueError, match="cross candidate mode"):
        candidate.run(
            "profile_salvalai",
            [],
            tmp_path / "init.json",
            mode="accepted",
        )
