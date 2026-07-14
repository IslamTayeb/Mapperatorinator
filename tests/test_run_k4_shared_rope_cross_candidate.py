from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest

from osuT5.osuT5.inference.optimized.kernels.weight_only_runtime import (
    CROSS_FP16_PACKED,
    CROSS_SPLIT8,
)
from utils import run_k4_shared_rope_cross_candidate as candidate


class _Stats:
    def as_dict(self):
        return {
            "module_count": 12,
            "group_count": 1,
            "forwards": 2,
            "computes": 2,
            "reuses": 22,
            "expected_computes": 2,
            "expected_reuses": 22,
            "group_computes": {"rope": 2},
            "member_names": [f"layer-{index}" for index in range(12)],
            "group_members": {"rope": [f"layer-{index}" for index in range(12)]},
        }


@pytest.mark.parametrize("mode", [CROSS_FP16_PACKED, CROSS_SPLIT8])
def test_cross_runner_scopes_main_model_and_records_incremental_mode(
    monkeypatch,
    tmp_path,
    mode,
) -> None:
    import inference

    main_model = object()
    timing_model = object()
    bindings = iter((main_model, timing_model))
    events = []

    def loader(*args, **kwargs):
        del args, kwargs
        model = next(bindings)
        events.append(("load", model))
        return SimpleNamespace(raw_model=model), object()

    @contextmanager
    def rope(model, *, stats):
        assert model is main_model
        events.append(("rope", model))
        yield stats

    @contextmanager
    def k4(*, block_size):
        events.append(("k4", block_size))
        yield

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
        assert initializer_name == "initialize_approximate_weight_only_cross"
        assert initializer_kwargs == {"mode": mode}
        inference.load_model_with_engine("main")
        inference.load_model_with_engine("timing")
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
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(inference, "load_model_with_engine", loader)
    monkeypatch.setattr(candidate, "shared_decoder_rope_context", rope)
    monkeypatch.setattr(candidate, "install_k8_candidate", k4)
    monkeypatch.setattr(candidate, "run_with_initializer", weight_run)
    monkeypatch.setattr(candidate, "SharedRopeStats", _Stats)
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
        "k4-split-kv-mixed-weight-shared-rope-v1"
    )
    assert payload["shared_rope"]["stats"]["reuses"] == 22
    assert events == [
        ("k4", 4),
        ("load", main_model),
        ("rope", main_model),
        ("load", timing_model),
    ]


def test_cross_runner_rejects_control_mode_before_loading(tmp_path) -> None:
    with pytest.raises(ValueError, match="cross candidate mode"):
        candidate.run(
            "profile_salvalai",
            [],
            tmp_path / "init.json",
            mode="accepted",
        )
