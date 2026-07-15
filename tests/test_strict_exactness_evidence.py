from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.inference import processor as processor_module
from osuT5.osuT5.inference.optimized.single.exactness import (
    cache_write_signature,
    rng_progression_signature,
)
from osuT5.osuT5.inference.processor import Processor
from osuT5.osuT5.inference.profiler import InferenceProfiler


def _cache(*, dtype: torch.dtype = torch.float32):
    def layers(offset: float, active_length: int):
        result = []
        for layer_idx in range(2):
            keys = torch.zeros((1, 2, 6, 3), dtype=dtype)
            keys[..., :active_length, :] = (
                torch.arange(
                    1 * 2 * active_length * 3,
                    dtype=dtype,
                ).reshape(1, 2, active_length, 3)
                + offset
                + layer_idx
                + 1
            )
            result.append(
                SimpleNamespace(
                    is_initialized=True,
                    keys=keys,
                    values=torch.cat(
                        (
                            keys[..., :active_length, :] + 100,
                            torch.zeros_like(keys[..., active_length:, :]),
                        ),
                        dim=-2,
                    ),
                    get_seq_length=lambda length=active_length: length,
                )
            )
        return result

    return SimpleNamespace(
        self_attention_cache=SimpleNamespace(layers=layers(0, 3)),
        cross_attention_cache=SimpleNamespace(layers=layers(1000, 4)),
    )


def _signature(cache, *, self_sequence_length: int = 3):
    return cache_write_signature(
        cache,
        self_sequence_length=self_sequence_length,
        expected_dtype=torch.float32,
        expected_device="cpu",
    )


def test_cache_signature_hashes_written_self_slots_and_full_cross_cache():
    cache = _cache()
    baseline = _signature(cache)

    cache.self_attention_cache.layers[0].keys[..., 2, :].add_(1)
    active_changed = _signature(cache)
    assert active_changed["aggregate_sha256"] != baseline["aggregate_sha256"]

    cache.cross_attention_cache.layers[1].values[..., 3, :].add_(1)
    assert _signature(cache)["aggregate_sha256"] != active_changed["aggregate_sha256"]

    cache.self_attention_cache.layers[0].keys[..., 5, :].add_(1)
    with pytest.raises(RuntimeError, match="beyond written length"):
        _signature(cache)


def test_cache_signature_fails_loudly_on_invalid_contracts():
    cache = _cache(dtype=torch.float16)
    with pytest.raises(TypeError, match="expected torch.float32"):
        _signature(cache)
    with pytest.raises(ValueError, match="exceeds cache capacity"):
        _signature(_cache(), self_sequence_length=7)

    malformed = _cache()
    malformed.cross_attention_cache.layers[0].values = torch.zeros(1, 2, 6)
    with pytest.raises(ValueError, match="matching rank-4"):
        _signature(malformed)

    wrong_length = _cache()
    wrong_length.self_attention_cache.layers[0].get_seq_length = lambda: 2
    with pytest.raises(RuntimeError, match="expected result_length-1=3"):
        _signature(wrong_length)

    stale_cross = _cache()
    stale_cross.cross_attention_cache.layers[0].keys[..., 5, :].fill_(1)
    with pytest.raises(RuntimeError, match="beyond written length"):
        _signature(stale_cross)


def test_rng_signature_does_not_advance_cpu_rng_and_detects_progression():
    previous = torch.random.get_rng_state()
    try:
        torch.manual_seed(12345)
        before = rng_progression_signature("cpu")
        assert rng_progression_signature("cpu") == before
        torch.rand(1)
        after = rng_progression_signature("cpu")
        assert after["cpu_sha256"] != before["cpu_sha256"]
        assert after["cuda_sha256"] is None
    finally:
        torch.random.set_rng_state(previous)


@pytest.mark.parametrize(
    ("pass_kind", "expected"),
    (("untraced_control", False), ("exactness_audit", True)),
)
def test_processor_enables_evidence_only_for_exactness_audit(pass_kind, expected):
    captured = {}

    class Runtime:
        def new_context_state(self):
            return object()

        def generate_window(self, **kwargs):
            captured.update(kwargs["generate_kwargs"])
            return "result"

    processor = object.__new__(Processor)
    processor.precision = "fp32"
    processor.do_sample = True
    processor.num_beams = 1
    processor.top_p = 0.9
    processor.top_k = 0
    processor.tgt_seq_len = 64
    processor.cfg_scale = 1.0
    processor.timeshift_bias = 0.0
    processor.types_first = False
    processor.temperature = 1.0
    processor.timing_temperature = 1.0
    processor.mania_column_temperature = 1.0
    processor.taiko_hit_temperature = 1.0
    processor.profiler = InferenceProfiler(enabled=True, pass_kind=pass_kind)
    processor.inference_runtime = Runtime()
    processor.decode_session_state = None
    processor.model = object()
    processor.tokenizer = object()

    assert Processor.model_generate(processor, {"inputs": torch.ones(1, 1)}) == "result"
    assert captured["collect_strict_exactness"] is expected
    assert processor.profiler.metadata["strict_exactness_evidence"] is expected
    assert processor.profiler.metadata["authoritative_performance"] is (
        pass_kind == "untraced_control"
    )


def test_exactness_internal_flag_never_leaks_to_v32(monkeypatch):
    captured = {}

    def fake_model_generate(model, tokenizer, model_kwargs, generate_kwargs):
        captured.update(generate_kwargs)
        return "result"

    monkeypatch.setattr(processor_module, "model_generate", fake_model_generate)
    processor = object.__new__(Processor)
    processor.precision = "fp32"
    processor.do_sample = True
    processor.num_beams = 1
    processor.top_p = 0.9
    processor.top_k = 0
    processor.tgt_seq_len = 64
    processor.cfg_scale = 1.0
    processor.timeshift_bias = 0.0
    processor.types_first = False
    processor.temperature = 1.0
    processor.timing_temperature = 1.0
    processor.mania_column_temperature = 1.0
    processor.taiko_hit_temperature = 1.0
    processor.profiler = InferenceProfiler(
        enabled=True,
        pass_kind="exactness_audit",
    )
    processor.inference_runtime = None
    processor.model = object()
    processor.tokenizer = object()

    assert Processor.model_generate(processor, {"inputs": torch.ones(1, 1)}) == "result"
    assert "collect_strict_exactness" not in captured
