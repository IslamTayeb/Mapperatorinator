from __future__ import annotations

import pytest

from osuT5.osuT5.inference.optimized.speculative import (
    ACCEPTED_TARGET_Q1_MS,
    MEASURED_TARGET_K4_GRAPH_MS,
    ClosedLoopAcceptanceHistogram,
    MiniDraftProjection,
    draft_cost_surface_ms,
)


def test_k4_measured_break_even_and_five_percent_surface():
    expected = {
        1: (-4.638447951817797, -4.829713139226908),
        2: (-0.8131442036355958, -1.1956745784538167),
        3: (3.0121595445466056, 2.4383639823192738),
        4: (6.837463292728808, 6.072402543092366),
    }

    for committed_tokens_per_call, (break_even_ms, five_percent_ms) in expected.items():
        surface = draft_cost_surface_ms(committed_tokens_per_call)
        assert surface[0.0] == pytest.approx(break_even_ms)
        assert surface[0.05] == pytest.approx(five_percent_ms)


def test_zero_cost_k4_requires_more_than_58_percent_committed_span_efficiency_for_five_percent():
    minimum_effective_length = MEASURED_TARGET_K4_GRAPH_MS / (0.95 * ACCEPTED_TARGET_Q1_MS)
    minimum_efficiency = minimum_effective_length / 4

    assert minimum_effective_length == pytest.approx(2.3290208836417667)
    assert minimum_efficiency == pytest.approx(0.5822552209104417)


def test_same_marginal_acceptance_can_have_different_closed_loop_economics():
    first_mismatch_or_full = ClosedLoopAcceptanceHistogram(4, (1, 0, 0, 0, 1))
    two_then_mismatch = ClosedLoopAcceptanceHistogram(4, (0, 0, 2, 0, 0))

    assert first_mismatch_or_full.draft_token_acceptance_fraction == 0.5
    assert two_then_mismatch.draft_token_acceptance_fraction == 0.5
    assert first_mismatch_or_full.committed_tokens_per_target_call == 2.5
    assert two_then_mismatch.committed_tokens_per_target_call == 3.0
    assert (
        first_mismatch_or_full.committed_span_efficiency
        < two_then_mismatch.committed_span_efficiency
    )


def test_finite_projection_charges_encoder_or_rebuild_overhead_to_real_scout():
    histogram = ClosedLoopAcceptanceHistogram(4, (0, 0, 0, 0, 10))
    no_fixed_overhead = MiniDraftProjection(histogram, steady_draft_ms_per_call=6.0)
    with_fixed_overhead = MiniDraftProjection(
        histogram,
        steady_draft_ms_per_call=6.0,
        fixed_draft_overhead_ms=1.0,
    )

    assert no_fixed_overhead.clears_minimum_improvement is True
    assert with_fixed_overhead.clears_minimum_improvement is False
    assert no_fixed_overhead.minimum_improvement_draft_budget_ms_per_call == pytest.approx(
        6.072402543092366
    )
    assert with_fixed_overhead.minimum_improvement_draft_budget_ms_per_call == pytest.approx(
        5.972402543092366
    )


@pytest.mark.parametrize(
    "histogram",
    [
        (4, ()),
        (4, (0, 0, 0, 0, 0)),
        (4, (1, 2, 3, 4)),
        (4, (1, 0, -1, 0, 0)),
        (4, (1, 0, 0.5, 0, 0)),
    ],
)
def test_histogram_rejects_ambiguous_or_invalid_counts(histogram):
    with pytest.raises(ValueError):
        ClosedLoopAcceptanceHistogram(*histogram)

