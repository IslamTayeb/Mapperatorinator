from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from utils import run_strict_shared_rope as candidate


class _Stats:
    def as_dict(self):
        return {
            "module_count": 12,
            "group_count": 1,
            "forwards": 3,
            "computes": 3,
            "reuses": 33,
            "eliminated_per_forward": 11,
            "expected_computes": 3,
            "expected_reuses": 33,
            "group_computes": {"rope-0": 3},
            "member_names": [f"layer-{index}" for index in range(12)],
            "group_members": {"rope-0": [f"layer-{index}" for index in range(12)]},
        }


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
        "profile_inference": True,
        "super_timing": False,
        "generate_positions": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _preset(precision: str) -> dict:
    return {
        "version": {
            "fp32": "accepted-fp32-native-cross-mlp-289-v3",
            "fp16": "accepted-fp16-all-fused-v2",
        }[precision],
        "precision": precision,
        "q1_bmm_cross_attention": True,
        "native_q1_self_attention": True,
        "native_q1_rope_cache_self_attention": True,
    }


def _binding(model, precision: str):
    runtime = SimpleNamespace(
        preset=SimpleNamespace(precision=precision),
        profile_metadata=lambda: {"optimized_effective_config": _preset(precision)},
    )
    return SimpleNamespace(raw_model=model, runtime=runtime)


@pytest.mark.parametrize(
    ("precision", "dtype"),
    (("fp32", torch.float32), ("fp16", torch.float16)),
)
def test_runner_scopes_candidate_to_first_binding_and_restores(
    precision,
    dtype,
    monkeypatch,
    tmp_path,
):
    import inference

    events = []
    main_model = SimpleNamespace(dtype=dtype)
    timing_model = SimpleNamespace(dtype=dtype)
    models = iter((main_model, timing_model))

    def fake_loader(*args, **kwargs):
        del args, kwargs
        model = next(models)
        events.append(("load", model))
        return _binding(model, precision), object()

    @contextmanager
    def fake_rope(model, *, stats):
        assert model is main_model
        assert isinstance(stats, _Stats)
        events.append(("enter", model))
        try:
            yield stats
        finally:
            events.append(("exit", model))

    def fake_main(args):
        assert args.precision == precision
        inference.load_model_with_engine("main")
        inference.load_model_with_engine("timing")
        events.append("main")

    original_loader = inference.load_model_with_engine
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    monkeypatch.setattr(inference, "load_model_with_engine", fake_loader)
    monkeypatch.setattr(inference, "main", fake_main)
    monkeypatch.setattr(
        candidate,
        "_load_args",
        lambda *args: _args(precision=precision),
    )
    monkeypatch.setattr(candidate, "SharedRopeStats", _Stats)
    monkeypatch.setattr(candidate, "shared_decoder_rope_context", fake_rope)
    monkeypatch.setattr(
        candidate,
        "_strict_environment_evidence",
        lambda requested: {
            "precision": requested,
            "float32_matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "nvidia_tf32_override": "0",
        },
    )
    output = tmp_path / "shared-rope.json"

    candidate.run("profile_salvalai", ["seed=12345"], output)

    assert events == [
        ("load", main_model),
        ("enter", main_model),
        ("load", timing_model),
        "main",
        ("exit", main_model),
    ]
    assert inference.load_model_with_engine is fake_loader
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["candidate"]["precision"] == precision
    assert payload["candidate"]["accepted_preset"] == _preset(precision)
    assert payload["candidate"]["dtype_generic"] is True
    assert payload["candidate"]["scope"] == "main-model-only"
    assert payload["candidate"]["full_song_exactness_claim"] is False
    assert payload["candidate"]["stats"]["reuses"] == 33
    monkeypatch.setattr(inference, "load_model_with_engine", original_loader)


def test_runner_restores_loader_and_context_on_failure(monkeypatch, tmp_path):
    import inference

    events = []
    original_loader = inference.load_model_with_engine

    def fake_loader(*args, **kwargs):
        del args, kwargs
        model = SimpleNamespace(dtype=torch.float32)
        return _binding(model, "fp32"), object()

    @contextmanager
    def fake_rope(model, *, stats):
        del model, stats
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def fail(args):
        del args
        inference.load_model_with_engine("main")
        raise RuntimeError("candidate failure")

    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    monkeypatch.setattr(inference, "load_model_with_engine", fake_loader)
    monkeypatch.setattr(inference, "main", fail)
    monkeypatch.setattr(candidate, "_load_args", lambda *args: _args())
    monkeypatch.setattr(candidate, "shared_decoder_rope_context", fake_rope)
    monkeypatch.setattr(
        candidate,
        "_strict_environment_evidence",
        lambda precision: {"precision": precision},
    )

    with pytest.raises(RuntimeError, match="candidate failure"):
        candidate.run("profile_salvalai", [], tmp_path / "never.json")

    assert inference.load_model_with_engine is fake_loader
    assert events == ["enter", "exit"]
    monkeypatch.setattr(inference, "load_model_with_engine", original_loader)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("precision", "bf16"),
        ("device", "cpu"),
        ("profile_inference", False),
        ("cfg_scale", 1.5),
        ("num_beams", 2),
    ),
)
def test_candidate_rejects_non_strict_config(field, value, monkeypatch):
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    with pytest.raises(ValueError, match="precision must|requirements changed"):
        candidate._validate_args(_args(**{field: value}))


def test_candidate_requires_process_level_tf32_override(monkeypatch):
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    with pytest.raises(RuntimeError, match="NVIDIA_TF32_OVERRIDE=0"):
        candidate._validate_args(_args())


def test_shared_rope_evidence_rejects_missing_reuse():
    class NoReuse(_Stats):
        def as_dict(self):
            payload = super().as_dict()
            payload["reuses"] = 0
            payload["expected_reuses"] = 0
            return payload

    with pytest.raises(RuntimeError, match="did not eliminate"):
        candidate._validated_shared_rope_evidence(
            NoReuse(),
            precision="fp32",
            preset=_preset("fp32"),
        )


def test_shared_rope_evidence_rejects_lost_specialized_dispatch():
    preset = _preset("fp16")
    preset["native_q1_rope_cache_self_attention"] = False
    with pytest.raises(RuntimeError, match="accepted specialized topology"):
        candidate._validated_shared_rope_evidence(
            _Stats(),
            precision="fp16",
            preset=preset,
        )


def test_source_has_no_selector_or_counter_rng_wiring():
    source = Path(candidate.__file__).read_text(encoding="utf-8").lower()
    forbidden = ("inference_engine =", "autocast", "counter_rng", "k4")
    assert all(term not in source for term in forbidden)
