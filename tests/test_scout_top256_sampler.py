from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import TopPLogitsWarper

from osuT5.osuT5.inference.optimized.scout import top256_sampler
from utils.top256_sampler_scout import bounded_reference
from utils.vocab_sampling_scout import _sampling_with_threshold


ROOT = Path(__file__).resolve().parents[1]


def _actual(scores: torch.Tensor, threshold: float, top_p: float) -> tuple[int, int]:
    filtered = TopPLogitsWarper(top_p=top_p)(
        torch.zeros((1, 1), dtype=torch.long),
        scores,
    )
    token = int(
        _sampling_with_threshold(filtered, torch.tensor(threshold)).item()
    )
    return token, int(torch.isfinite(filtered).sum().item())


def test_import_is_cold_and_extension_preload_is_singleton(monkeypatch) -> None:
    calls = []
    extension = object()

    def fake_load_inline(**kwargs):
        calls.append(kwargs)
        return extension

    monkeypatch.setattr(top256_sampler, "_EXTENSION", None)
    monkeypatch.setattr(top256_sampler, "load_inline", fake_load_inline)

    assert top256_sampler.preload_top256_sampler_extension() is extension
    assert top256_sampler.preload_top256_sampler_extension() is extension
    assert len(calls) == 1
    assert calls[0]["name"] == "mapperatorinator_top256_sampler_scout_v1"
    assert calls[0]["with_cuda"] is True


def test_source_preserves_stable_boundary_and_original_id_sampling() -> None:
    source = top256_sampler._CUDA_SOURCE

    assert "constexpr int kTopK = 256" in source
    assert "constexpr int kCandidatesPerChunk = kTopK + 1" in source
    assert "left_score == right_score && left_id < right_id" in source
    assert "retained_scores[kTopK - 1] == retained_scores[kTopK]" in source
    assert "Sort only the retained bounded set by token id" in source
    assert "atomicMin(&selected_position, lane)" in source
    assert "int* __restrict__ kept_count_output" in source
    assert "unsigned char* __restrict__ overflow_output" in source

    package_init = (
        ROOT / "osuT5/osuT5/inference/optimized/scout/__init__.py"
    ).read_text(encoding="utf-8")
    assert "top256_sampler" not in package_init


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_validation_rejects_unsupported_score_dtype_before_build(
    monkeypatch,
    dtype,
) -> None:
    scores = torch.zeros((1, 8), dtype=dtype)
    threshold = torch.tensor([0.25], dtype=torch.float32)
    monkeypatch.setattr(
        top256_sampler,
        "_load_top256_extension",
        lambda: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    with pytest.raises(TypeError, match="FP32 storage"):
        top256_sampler.top256_sample(scores, threshold, top_p=0.9)


def test_validation_fails_loudly_for_shape_width_and_top_p(monkeypatch) -> None:
    monkeypatch.setattr(
        top256_sampler,
        "_load_top256_extension",
        lambda: (_ for _ in ()).throw(AssertionError("must not build")),
    )
    cpu = torch.zeros((1, 8), dtype=torch.float32)
    threshold = torch.tensor([0.25], dtype=torch.float32)
    with pytest.raises(RuntimeError, match="CUDA tensor"):
        top256_sampler.top256_sample(cpu, threshold, top_p=0.9)
    with pytest.raises(TypeError, match="top_p must be numeric"):
        top256_sampler._validate_inputs(cpu, threshold, True)
    with pytest.raises(ValueError, match=r"inside \(0, 1\)"):
        top256_sampler._validate_inputs(cpu, threshold, 1.0)
    with pytest.raises(ValueError, match=r"shape \[1, vocab\]"):
        top256_sampler._validate_inputs(torch.zeros(8), threshold, 0.9)
    with pytest.raises(ValueError, match="vocabulary width"):
        top256_sampler._validate_inputs(
            torch.zeros((1, top256_sampler.MAX_VOCAB + 1)), threshold, 0.9
        )


def test_extension_output_schema_is_checked(monkeypatch) -> None:
    scores = torch.zeros((1, 8), dtype=torch.float32)
    threshold = torch.tensor([0.25], dtype=torch.float32)
    monkeypatch.setattr(top256_sampler, "_validate_inputs", lambda *args: None)
    monkeypatch.setattr(
        top256_sampler,
        "_load_top256_extension",
        lambda: SimpleNamespace(
            top256_sample=lambda *args: (
                torch.tensor([1], dtype=torch.long),
                torch.tensor([2], dtype=torch.int16),
                torch.tensor([0], dtype=torch.uint8),
            )
        ),
    )

    with pytest.raises(RuntimeError, match="invalid output tensors"):
        top256_sampler.top256_sample(scores, threshold, top_p=0.9)


@pytest.mark.parametrize("threshold", [0.05, 0.25, 0.73, 0.99])
def test_bounded_reference_matches_real_top_p_and_original_order(threshold) -> None:
    scores = torch.tensor([[0.1, 4.0, -0.2, 3.0, 1.0]], dtype=torch.float32)
    top_p = 0.9

    reference = bounded_reference(scores, threshold, top_p=top_p)
    expected_token, expected_count = _actual(scores, threshold, top_p)

    assert not reference["overflow"]
    assert reference["token"] == expected_token
    assert reference["kept_count"] == expected_count


def test_reference_overflows_above_256_and_on_boundary_tie() -> None:
    large_nucleus = torch.zeros((1, 300), dtype=torch.float32)
    tie_boundary = torch.arange(300, 0, -1, dtype=torch.float32).reshape(1, 300)
    tie_boundary[0, 255:257] = tie_boundary[0, 255]

    assert bounded_reference(
        large_nucleus,
        0.25,
        top_p=0.9,
    )["overflow"]
    assert bounded_reference(
        tie_boundary,
        0.25,
        top_p=0.999999,
    )["overflow"]


def test_negative_infinity_padding_does_not_create_a_false_boundary_tie() -> None:
    scores = torch.full((1, 300), -torch.inf, dtype=torch.float32)
    scores[0, :8] = torch.arange(8, 0, -1, dtype=torch.float32)

    result = bounded_reference(scores, 0.25, top_p=0.9)

    assert not result["overflow"]
    assert result["kept_count"] <= 8


@pytest.mark.parametrize(
    "scores",
    [
        torch.full((1, 8), -torch.inf, dtype=torch.float32),
        torch.tensor([[0.0, torch.nan]], dtype=torch.float32),
        torch.tensor([[0.0, torch.inf]], dtype=torch.float32),
    ],
)
def test_reference_overflows_nonfinite_unsupported_inputs(scores) -> None:
    result = bounded_reference(scores, 0.25, top_p=0.9)

    assert result == {"token": -1, "kept_count": 0, "overflow": True}


def test_reference_kept_count_can_represent_256_without_uint8_wrap() -> None:
    scores = torch.linspace(5.0, -5.0, 300, dtype=torch.float32).reshape(1, 300)
    probabilities = torch.softmax(scores[0], dim=0)
    cumulative = probabilities.cumsum(0)
    top_p = float((cumulative[254] + cumulative[255]) / 2.0)

    result = bounded_reference(scores, 0.25, top_p=top_p)

    assert not result["overflow"]
    assert result["kept_count"] == 256
