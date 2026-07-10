from __future__ import annotations

from unittest.mock import patch

import torch
from transformers.modeling_outputs import BaseModelOutput

from osuT5.osuT5.inference.direct_decode import DecodeSession, OneTokenDecodeState
from osuT5.osuT5.inference.optimized.batch.merged_one_token import (
    MergedOneTokenConfig,
    compare_logits,
    repeat_batch_kwargs,
    repeat_batch_tensor,
    resolve_row_seeds,
    summarize_previous_gate_scaling,
    validate_previous_gate,
)


def _assert_raises(exc_type, fn, message_fragment):
    try:
        fn()
    except exc_type as exc:
        assert message_fragment in str(exc)
    else:
        raise AssertionError(f"Expected {exc_type.__name__}")


def test_repeat_batch_helpers_require_b1_and_preserve_non_tensor_values():
    tensor = torch.tensor([[1, 2, 3]])
    repeated = repeat_batch_tensor(tensor, 5, name="prompt")

    assert repeated.shape == (5, 3)
    assert repeated.is_contiguous()
    assert repeated.tolist() == [[1, 2, 3]] * 5
    values = repeat_batch_kwargs({"tensor": tensor, "flag": True}, 2)
    assert values["tensor"].shape == (2, 3)
    assert values["flag"] is True
    _assert_raises(
        ValueError,
        lambda: repeat_batch_tensor(torch.ones((2, 3)), 2, name="not-b1"),
        "batch dimension 1",
    )


def test_compare_logits_checks_allclose_nonfinite_layout_and_topk():
    reference = torch.tensor([[1.0, 4.0, float("-inf"), 3.0]])
    close = torch.tensor([[1.0, 4.00001, float("-inf"), 3.0]])
    changed_topk = torch.tensor([[1.0, 2.0, float("-inf"), 5.0]])

    passing = compare_logits(reference, close, atol=1e-4, rtol=1e-4, top_k=2)
    failing = compare_logits(reference, changed_topk, atol=1e-4, rtol=1e-4, top_k=2)

    assert passing["allclose"] is True
    assert passing["topk_match"] is True
    assert failing["allclose"] is False
    assert failing["topk_match"] is False


def test_row_seed_resolution_and_config_reject_ungraduated_shapes():
    assert resolve_row_seeds([], batch_size=2, default_seed=12345) == (12345, 12345)
    assert resolve_row_seeds([7], batch_size=2, default_seed=12345) == (7, 7)
    assert resolve_row_seeds([7, 8], batch_size=2, default_seed=12345) == (7, 8)
    _assert_raises(
        ValueError,
        lambda: resolve_row_seeds([1, 2, 3], batch_size=2, default_seed=0),
        "zero, one, or exactly batch_size",
    )
    _assert_raises(
        ValueError,
        lambda: MergedOneTokenConfig(
            batch_size=3,
            seeds=(1,) * 3,
            do_sample=True,
        ),
        "support B=1, B=2, B=5, or B=8",
    )


def test_previous_gate_enforces_b1_then_b2_then_b5_then_b8():
    b1_report = {
        "pass": True,
        "batch_size": 1,
        "observation": {"execution_family": "merged_batch", "parallelism": 1},
    }
    b2_report = {
        "pass": True,
        "batch_size": 2,
        "observation": {"execution_family": "merged_batch", "parallelism": 2},
    }
    b5_report = {
        "pass": True,
        "batch_size": 5,
        "observation": {"execution_family": "merged_batch", "parallelism": 5},
    }

    validate_previous_gate(1, None)
    validate_previous_gate(2, b1_report)
    validate_previous_gate(5, b2_report)
    validate_previous_gate(8, b5_report)
    _assert_raises(
        ValueError,
        lambda: validate_previous_gate(5, b1_report),
        "requires a B=2 previous report",
    )
    failed = dict(b1_report, **{"pass": False})
    _assert_raises(
        ValueError,
        lambda: validate_previous_gate(2, failed),
        "did not pass",
    )


def test_previous_gate_scaling_applies_keep_bar_and_ideal_capacity_loss():
    passing = summarize_previous_gate_scaling(
        previous_batch_size=5,
        previous_tokens_per_second=300.0,
        candidate_batch_size=8,
        candidate_tokens_per_second=420.0,
    )
    below_bar = summarize_previous_gate_scaling(
        previous_batch_size=5,
        previous_tokens_per_second=300.0,
        candidate_batch_size=8,
        candidate_tokens_per_second=312.0,
    )

    assert passing["clears_five_percent_gain_gate"] is True
    assert abs(passing["relative_complete_throughput_gain"] - 0.4) < 1e-12
    assert passing["ideal_scaled_previous_capacity_tokens_per_second"] == 480.0
    assert passing["ideal_capacity_loss"] == -0.125
    assert below_bar["clears_five_percent_gain_gate"] is False


def test_decode_session_prefill_allocates_cache_for_prompt_batch_size():
    for batch_size in (1, 2, 5):
        prompt = torch.ones((batch_size, 3), dtype=torch.long)
        prompt_mask = torch.ones_like(prompt)
        frames = torch.zeros((batch_size, 80, 16))
        fake_state = OneTokenDecodeState(
            cache=object(),
            encoder_outputs=BaseModelOutput(
                last_hidden_state=torch.zeros((batch_size, 4, 8))
            ),
            prompt_length=3,
            prefill_cache_position=torch.arange(3),
            prefill_prepared_inputs={},
            prefill_logits=torch.zeros((batch_size, 10)),
        )

        with patch(
                "osuT5.osuT5.inference.direct_decode.prefill_static_cache",
                return_value=fake_state,
        ) as prefill:
            session = DecodeSession.prefill(
                object(),
                prompt=prompt,
                prompt_attention_mask=prompt_mask,
                frames=frames,
                condition_kwargs={"test": True},
            )

        assert session.static_inputs.prompt.shape[0] == batch_size
        assert prefill.call_args.kwargs["batch_size"] == batch_size
