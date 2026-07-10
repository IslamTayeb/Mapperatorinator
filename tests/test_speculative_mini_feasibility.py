from __future__ import annotations

import pytest

from osuT5.osuT5.inference.optimized.speculative import (
    ACCEPTED_TARGET_Q1_MS,
    MEASURED_TARGET_K4_GRAPH_MS,
    ClosedLoopAcceptanceHistogram,
    FiniteWindowProjection,
    MiniDraftProjection,
    cached_q1_forwards_for_proposal,
    clears_strict_cuda_and_wall_gate,
    draft_cost_surface_ms,
    replay_closed_loop_target_transcript,
    token_ids_sha256,
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


def test_closed_loop_replay_reconditions_every_proposal_on_corrected_target_prefix():
    target = (1, 2, 3, 4, 5, 6)
    observed_prefixes = []

    def propose(prefix, requested):
        observed_prefixes.append(prefix)
        if not prefix:
            return (99, 2, 3, 4)
        if prefix == (1,):
            return (2, 3, 4, 5)
        return (99,)[:requested]

    replay = replay_closed_loop_target_transcript(
        target,
        speculation_k=4,
        propose=propose,
    )

    assert replay.exact is True
    assert replay.replayed_token_ids == target
    assert observed_prefixes == [(), (1,), (1, 2, 3, 4, 5)]
    assert [event.accepted_prefix_length for event in replay.events] == [0, 4, 0]
    assert replay.full_span_histogram().counts == (1, 0, 0, 0, 1)
    assert replay.short_boundary_bins() == {"1": [1, 0]}
    assert replay.events[1].committed_prefix_sha256 == token_ids_sha256((1,))
    assert replay.events[1].committed_prefix_after_sha256 == token_ids_sha256((1, 2, 3, 4, 5))


def test_ready_prefix_k4_proposal_requires_only_three_q1_forwards():
    assert cached_q1_forwards_for_proposal(4) == 3
    assert cached_q1_forwards_for_proposal(1) == 0


def test_strict_gate_requires_cuda_and_conservative_wall_projection_to_clear():
    histogram = ClosedLoopAcceptanceHistogram(4, (0, 0, 0, 0, 10))
    cuda_projection = FiniteWindowProjection(
        target_output_tokens=40,
        full_span_histogram=histogram,
        full_span_draft_cost_ms=60.0,
    )
    wall_projection = FiniteWindowProjection(
        target_output_tokens=40,
        full_span_histogram=histogram,
        full_span_draft_cost_ms=61.0,
    )

    assert cuda_projection.clears_minimum_improvement is True
    assert wall_projection.clears_minimum_improvement is False
    assert clears_strict_cuda_and_wall_gate(cuda_projection, wall_projection) is False


def test_finite_window_projection_keeps_short_boundary_on_serial_q1_denominator():
    histogram = ClosedLoopAcceptanceHistogram(4, (0, 0, 0, 0, 2))
    projection = FiniteWindowProjection(
        target_output_tokens=9,
        full_span_histogram=histogram,
        full_span_draft_cost_ms=4.0,
        short_boundary_output_tokens=1,
        short_boundary_draft_cost_ms=0.5,
        fixed_draft_overhead_ms=1.0,
    )

    expected = 2 * MEASURED_TARGET_K4_GRAPH_MS + 4.0 + ACCEPTED_TARGET_Q1_MS + 0.5 + 1.0
    assert projection.candidate_ms == pytest.approx(expected)
    assert projection.to_manifest()["short_boundary_policy"] == (
        "serial_q1_no_unmeasured_q_len_lt_k_saving"
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
