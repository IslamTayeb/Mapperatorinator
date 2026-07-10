from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Callable, Sequence

import pytest

from osuT5.osuT5.inference.optimized.speculative import (
    DraftProposal,
    DecodeIterationKind,
    EvaluatedTargetSpan,
    MissingSpeculativeScoutAdapterError,
    ProposalSourceKind,
    SpeculationConfig,
    SpeculativeScoutConfig,
    compare_exact_transcript,
    load_scout_adapter_factory,
    run_bounded_speculative_scout,
    run_exact_speculative_decode,
    run_sequential_target_reference,
)


class CountingGenerator:
    """Tiny deterministic generator whose complete state is easy to compare."""

    def __init__(self, seed: int):
        self.state = seed & ((1 << 64) - 1)

    def draw(self) -> int:
        self.state = (6364136223846793005 * self.state + 1442695040888963407) & ((1 << 64) - 1)
        return self.state

    def state_hash(self) -> str:
        return hashlib.sha256(struct.pack("<Q", self.state)).hexdigest()


@dataclass(frozen=True)
class FixedTokenDistribution:
    token_id: int

    def sample(self, generator: CountingGenerator) -> int:
        generator.draw()
        return self.token_id


ProposalFunction = Callable[[tuple[int, ...], int, tuple[int, ...]], tuple[int, ...]]


class ScriptedAdapter:
    is_real_gpu_model_adapter = False
    proposal_source = ProposalSourceKind.NGRAM.value

    def __init__(
            self,
            target_tokens: Sequence[int],
            *,
            seed: int = 12345,
            proposal_function: ProposalFunction | None = None,
            draft_cost_seconds: float | None = None,
            mutate_rng_during_proposal: bool = False,
    ):
        self.target_tokens = tuple(target_tokens)
        self.seed = seed
        self.generator = CountingGenerator(seed)
        self.proposal_function = proposal_function or self._exact_proposal
        self.draft_cost_seconds = draft_cost_seconds
        self.mutate_rng_during_proposal = mutate_rng_during_proposal

    @staticmethod
    def _exact_proposal(
            prefix: tuple[int, ...],
            max_tokens: int,
            target_tokens: tuple[int, ...],
    ) -> tuple[int, ...]:
        start = len(prefix)
        return target_tokens[start:start + max_tokens]

    def propose(self, committed_tokens: Sequence[int], max_tokens: int) -> DraftProposal:
        if self.mutate_rng_during_proposal:
            self.generator.draw()
        tokens = self.proposal_function(tuple(committed_tokens), max_tokens, self.target_tokens)
        return DraftProposal(tokens, cost_seconds=self.draft_cost_seconds)

    def evaluate_target_span(
            self,
            committed_tokens: Sequence[int],
            draft_tokens: Sequence[int],
    ) -> EvaluatedTargetSpan:
        start = len(committed_tokens)
        positions = tuple(
            FixedTokenDistribution(self.target_tokens[start + offset])
            for offset in range(len(draft_tokens))
        )
        return EvaluatedTargetSpan(positions)

    def evaluate_target_next(self, committed_tokens: Sequence[int]) -> FixedTokenDistribution:
        return FixedTokenDistribution(self.target_tokens[len(committed_tokens)])

    def generator_state_hash(self) -> str:
        return self.generator.state_hash()

    def make_sequential_reference(self):
        return ScriptedAdapter(self.target_tokens, seed=self.seed)


def _assert_matches_sequential(
        speculative: ScriptedAdapter,
        baseline: ScriptedAdapter,
        config: SpeculationConfig,
):
    candidate = run_exact_speculative_decode(speculative, config)
    reference = run_sequential_target_reference(baseline, config)
    comparison = compare_exact_transcript(candidate, reference)
    comparison.assert_exact()
    assert candidate.target_positions_sampled == len(candidate.generated_token_ids)
    return candidate, reference


def test_exact_transcript_comparison_reports_each_failed_observable():
    config = SpeculationConfig(2, max_new_tokens=2, eos_token_id=99)
    candidate = run_exact_speculative_decode(ScriptedAdapter((1, 2)), config)
    different_reference = run_sequential_target_reference(ScriptedAdapter((1, 3)), config)

    comparison = compare_exact_transcript(candidate, different_reference)

    assert comparison.generated_tokens_match is False
    assert comparison.stop_reason_match is True
    assert comparison.final_rng_state_match is True
    assert comparison.exact is False
    with pytest.raises(AssertionError, match="tokens=False, stop=True, rng=True"):
        comparison.assert_exact()


@pytest.mark.parametrize("speculation_k", [2, 4, 8])
def test_all_accepted_k_blocks_match_sequential_tokens_and_rng(speculation_k):
    target = tuple(range(10, 18))
    config = SpeculationConfig(speculation_k, max_new_tokens=len(target), eos_token_id=999)

    result, _ = _assert_matches_sequential(
        ScriptedAdapter(target, draft_cost_seconds=0.125),
        ScriptedAdapter(target),
        config,
    )

    assert result.accepted_draft_tokens == len(target)
    assert result.target_model_calls == len(target) // speculation_k
    assert result.target_model_calls_saved == len(target) - len(target) // speculation_k
    assert result.serial_target_token_steps_saved == result.target_model_calls_saved
    assert result.draft_cost_status == "measured"
    assert result.draft_cost_seconds == pytest.approx(0.125 * (len(target) // speculation_k))
    assert all(event.discarded_uncommitted_suffix_length == 0 for event in result.cache_events)


@pytest.mark.parametrize("speculation_k", [2, 4, 8])
def test_every_first_block_mismatch_position_rolls_back_and_matches_sequential(speculation_k):
    target = tuple(range(20, 20 + speculation_k + 3))

    for mismatch_position in range(speculation_k):
        def proposal(prefix, max_tokens, target_tokens, mismatch=mismatch_position):
            proposed = list(target_tokens[len(prefix):len(prefix) + max_tokens])
            if not prefix and mismatch < len(proposed):
                proposed[mismatch] += 1000
            return tuple(proposed)

        config = SpeculationConfig(speculation_k, len(target), eos_token_id=9999)
        result, _ = _assert_matches_sequential(
            ScriptedAdapter(target, proposal_function=proposal),
            ScriptedAdapter(target),
            config,
        )

        first = result.cache_events[0]
        assert first.accepted_draft_length == mismatch_position
        assert first.mismatch_index == mismatch_position
        assert first.committed_token_ids == target[:mismatch_position + 1]
        assert first.prefix_length_after == mismatch_position + 1
        assert first.discarded_uncommitted_suffix_length == speculation_k - mismatch_position


@pytest.mark.parametrize(
    ("target", "proposal", "expected_sample_count", "expected_accepted", "expected_discarded"),
    [
        ((99, 1, 2, 3), None, 1, 1, 3),
        ((1, 2, 99, 3), None, 3, 3, 1),
        ((1, 2, 99, 3), (1, 2, 777, 3), 3, 2, 2),
    ],
)
def test_immediate_delayed_and_mismatched_eos_never_advance_rng_after_stop(
        target,
        proposal,
        expected_sample_count,
        expected_accepted,
        expected_discarded,
):
    proposal_function = None
    if proposal is not None:
        proposal_function = lambda prefix, max_tokens, _target: proposal[:max_tokens]
    config = SpeculationConfig(4, max_new_tokens=4, eos_token_id=99)

    result, _ = _assert_matches_sequential(
        ScriptedAdapter(target, proposal_function=proposal_function),
        ScriptedAdapter(target),
        config,
    )

    assert result.stop_reason == "eos"
    assert result.target_positions_sampled == expected_sample_count
    assert result.cache_events[0].accepted_draft_length == expected_accepted
    assert result.cache_events[0].discarded_uncommitted_suffix_length == expected_discarded
    assert result.cache_events[0].stopped_after_iteration is True


def test_max_token_boundary_shrinks_last_block_and_does_not_oversample():
    target = (1, 2, 3, 4, 5)
    config = SpeculationConfig(4, max_new_tokens=5, eos_token_id=99)

    result, _ = _assert_matches_sequential(ScriptedAdapter(target), ScriptedAdapter(target), config)

    assert result.stop_reason == "max_new_tokens"
    assert result.target_positions_sampled == 5
    assert [event.proposal_length for event in result.cache_events] == [4, 1]
    assert result.cache_events[-1].stopped_after_iteration is True
    assert result.unused_target_positions_evaluated == 0


@pytest.mark.parametrize(
    ("target", "max_new_tokens", "eos_token_id", "expected_stop", "expected_tokens"),
    [
        ((1, 2, 3), 3, 99, "max_new_tokens", (1, 2, 3)),
        ((1, 99, 3), 3, 99, "eos", (1, 99)),
    ],
)
def test_repeated_empty_ngram_proposals_fall_back_one_target_step_exactly(
        target,
        max_new_tokens,
        eos_token_id,
        expected_stop,
        expected_tokens,
):
    def no_ngram_match(_prefix, _max_tokens, _target_tokens):
        return ()

    config = SpeculationConfig(4, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id)
    result, _ = _assert_matches_sequential(
        ScriptedAdapter(target, proposal_function=no_ngram_match, draft_cost_seconds=0.0005),
        ScriptedAdapter(target),
        config,
    )

    assert result.generated_token_ids == expected_tokens
    assert result.stop_reason == expected_stop
    assert result.draft_calls == len(expected_tokens)
    assert result.target_model_calls == len(expected_tokens)
    assert result.target_model_calls_saved == 0
    assert result.proposed_draft_tokens == 0
    assert result.accepted_draft_tokens == 0
    assert result.draft_acceptance_rate == 0.0
    assert result.draft_cost_status == "measured"
    assert result.draft_cost_seconds == pytest.approx(0.0005 * len(expected_tokens))
    assert all(
        event.iteration_kind is DecodeIterationKind.TARGET_ONLY_FALLBACK
        for event in result.cache_events
    )
    for index, event in enumerate(result.cache_events):
        assert event.prefix_length_before == index
        assert event.prefix_length_after == index + 1
        assert event.proposal_length == 0
        assert event.target_positions_evaluated == 1
        assert event.target_positions_sampled == 1
        assert event.accepted_draft_length == 0
        assert event.discarded_uncommitted_suffix_length == 0
        assert event.emitted_target_token_on_mismatch is False
        assert event.mismatch_index is None
    assert result.cache_events[-1].stopped_after_iteration is True


@pytest.mark.parametrize("non_finite_cost", [float("inf"), float("nan")])
def test_draft_cost_must_be_finite_when_reported(non_finite_cost):
    with pytest.raises(ValueError, match="finite and non-negative"):
        DraftProposal((), cost_seconds=non_finite_cost)


def test_unmeasured_draft_cost_is_explicit_instead_of_zero():
    config = SpeculationConfig(2, max_new_tokens=2, eos_token_id=99)
    result = run_exact_speculative_decode(ScriptedAdapter((1, 2)), config)

    assert result.draft_cost_status == "not_measured"
    assert result.draft_cost_seconds is None


def test_target_generator_must_not_advance_during_proposal_or_evaluation():
    config = SpeculationConfig(2, max_new_tokens=2, eos_token_id=99)
    with pytest.raises(RuntimeError, match="advanced the baseline generator before sampling"):
        run_exact_speculative_decode(
            ScriptedAdapter((1, 2), mutate_rng_during_proposal=True),
            config,
        )


@pytest.mark.parametrize("proposal_source", list(ProposalSourceKind))
def test_bounded_scout_fails_loudly_without_real_gpu_model_adapter(proposal_source):
    config = SpeculativeScoutConfig(
        proposal_source=proposal_source,
        speculation=SpeculationConfig(2, max_new_tokens=2, eos_token_id=99),
        seed=12345,
    )
    with pytest.raises(MissingSpeculativeScoutAdapterError, match="real GPU/model adapter"):
        run_bounded_speculative_scout(config)
    with pytest.raises(MissingSpeculativeScoutAdapterError, match="is_real_gpu_model_adapter=True"):
        run_bounded_speculative_scout(config, ScriptedAdapter((1, 2)))


def test_bounded_scout_runs_and_asserts_serial_exactness_with_explicit_adapter():
    class DeclaredGpuAdapter(ScriptedAdapter):
        is_real_gpu_model_adapter = True

    config = SpeculativeScoutConfig(
        proposal_source=ProposalSourceKind.NGRAM,
        speculation=SpeculationConfig(2, max_new_tokens=4, eos_token_id=99),
        seed=12345,
    )
    comparison = run_bounded_speculative_scout(config, DeclaredGpuAdapter((1, 2, 3, 4)))

    assert comparison.exact is True
    assert comparison.speculative.target_model_calls_saved == 2


def test_scout_factory_parser_fails_before_importing_an_ambiguous_spec():
    with pytest.raises(ValueError, match="module:function"):
        load_scout_adapter_factory("missing_separator")


def test_v32_mini_scout_contract_records_default_model_and_ngram_rejects_one():
    config = SpeculativeScoutConfig(
        proposal_source=ProposalSourceKind.V32_MINI,
        speculation=SpeculationConfig(2, max_new_tokens=2, eos_token_id=99),
        seed=12345,
    )
    assert config.draft_model_path == "OliBomby/Mapperatorinator-v32-mini"

    with pytest.raises(ValueError, match="ngram.*must not declare"):
        SpeculativeScoutConfig(
            proposal_source=ProposalSourceKind.NGRAM,
            speculation=SpeculationConfig(2, max_new_tokens=2, eos_token_id=99),
            seed=12345,
            draft_model_path="unexpected-model",
        )


@pytest.mark.parametrize("speculation_k", [1, 3, 16])
def test_campaign_only_allows_planned_k_values(speculation_k):
    with pytest.raises(ValueError, match="2, 4, or 8"):
        SpeculationConfig(speculation_k, max_new_tokens=8, eos_token_id=99)
