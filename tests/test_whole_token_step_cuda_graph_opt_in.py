"""Cold-default and opt-in wiring for whole-token-step CUDA graph."""

from __future__ import annotations

from osuT5.osuT5.inference.optimized.kernels.whole_token_step_cuda_graph import (
    record_whole_token_step_cuda_graph_hit,
    reset_whole_token_step_cuda_graph_hits,
    whole_token_step_cuda_graph_candidate_context,
    whole_token_step_cuda_graph_hits,
    whole_token_step_cuda_graph_requested,
)


def test_cold_default_unrequested() -> None:
    assert whole_token_step_cuda_graph_requested() is False


def test_candidate_context_toggles_and_resets_hits() -> None:
    reset_whole_token_step_cuda_graph_hits()
    record_whole_token_step_cuda_graph_hit()
    assert whole_token_step_cuda_graph_hits() == 1
    with whole_token_step_cuda_graph_candidate_context():
        assert whole_token_step_cuda_graph_requested() is True
        assert whole_token_step_cuda_graph_hits() == 0
        record_whole_token_step_cuda_graph_hit()
        assert whole_token_step_cuda_graph_hits() == 1
    assert whole_token_step_cuda_graph_requested() is False
