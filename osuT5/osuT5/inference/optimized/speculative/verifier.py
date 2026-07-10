"""Exact transcript/commit/rollback state machine for speculative scouts."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from .contracts import (
    CacheCommitEvent,
    DecodeIterationKind,
    DraftProposal,
    EvaluatedTargetSpan,
    ExactTranscriptComparison,
    SpeculationConfig,
    SequentialTargetResult,
    SpeculativeDecodeResult,
    TargetTokenDistribution,
)


class ExactSpeculativeAdapter(Protocol):
    """Minimal adapter boundary shared by scripted tests and future GPU scouts."""

    generator: Any

    def propose(self, committed_tokens: Sequence[int], max_tokens: int) -> DraftProposal:
        """Return at most ``max_tokens`` proposed output tokens."""

    def evaluate_target_span(
            self,
            committed_tokens: Sequence[int],
            draft_tokens: Sequence[int],
    ) -> EvaluatedTargetSpan:
        """Evaluate target positions in one call, without sampling or changing RNG."""

    def evaluate_target_next(self, committed_tokens: Sequence[int]) -> TargetTokenDistribution:
        """Evaluate one target-only fallback position without changing target RNG."""

    def generator_state_hash(self) -> str:
        """Hash the complete request-private generator state."""


class SequentialTargetAdapter(Protocol):
    """Target-only denominator with a separately initialized identical generator."""

    generator: Any

    def evaluate_target_next(self, committed_tokens: Sequence[int]) -> TargetTokenDistribution:
        """Evaluate one next-token distribution without consuming target RNG."""

    def generator_state_hash(self) -> str:
        """Hash the complete request-private generator state."""


def run_sequential_target_reference(
        adapter: SequentialTargetAdapter,
        config: SpeculationConfig,
) -> SequentialTargetResult:
    """Run the one-target-call/one-sample-per-position exactness denominator."""

    generated: list[int] = []
    stop_reason: str | None = None
    while stop_reason is None:
        rng_before_evaluation = adapter.generator_state_hash()
        position = adapter.evaluate_target_next(tuple(generated))
        if adapter.generator_state_hash() != rng_before_evaluation:
            raise RuntimeError("Target evaluation advanced the baseline generator before sampling.")
        token = int(position.sample(adapter.generator))
        generated.append(token)
        if token == config.eos_token_id:
            stop_reason = "eos"
        elif len(generated) == config.max_new_tokens:
            stop_reason = "max_new_tokens"

    return SequentialTargetResult(
        generated_token_ids=tuple(generated),
        stop_reason=stop_reason,
        final_rng_state_hash=adapter.generator_state_hash(),
        target_model_calls=len(generated),
    )


def compare_exact_transcript(
        speculative: SpeculativeDecodeResult,
        sequential: SequentialTargetResult,
) -> ExactTranscriptComparison:
    """Compare all observable verifier-level exactness fields and fail separately."""

    return ExactTranscriptComparison(
        speculative=speculative,
        sequential=sequential,
        generated_tokens_match=(speculative.generated_token_ids == sequential.generated_token_ids),
        stop_reason_match=(speculative.stop_reason == sequential.stop_reason),
        final_rng_state_match=(speculative.final_rng_state_hash == sequential.final_rng_state_hash),
    )


def run_exact_speculative_decode(
        adapter: ExactSpeculativeAdapter,
        config: SpeculationConfig,
) -> SpeculativeDecodeResult:
    """Run exact speculative verification without advancing RNG past a stop.

    Target span evaluation may calculate K positions at once, but target sampling is
    intentionally serial.  A block ends on its first mismatch, EOS, or max-token
    boundary.  Any evaluated suffix after that point is logical rollback work and must
    not be committed by a concrete cache adapter.  An empty draft takes one explicit
    target-only step and never enters the span evaluator.
    """

    generated: list[int] = []
    cache_events: list[CacheCommitEvent] = []
    draft_costs: list[float | None] = []
    proposed_draft_tokens = 0
    accepted_draft_tokens = 0
    target_positions_evaluated = 0
    target_positions_sampled = 0
    stop_reason: str | None = None

    while stop_reason is None:
        remaining = config.max_new_tokens - len(generated)
        if remaining <= 0:
            stop_reason = "max_new_tokens"
            break

        requested = min(config.speculation_k, remaining)
        prefix_before = len(generated)
        rng_before_non_sampling_work = adapter.generator_state_hash()
        proposal = adapter.propose(tuple(generated), requested)
        if len(proposal.token_ids) > requested:
            raise ValueError(
                f"Draft adapter returned {len(proposal.token_ids)} tokens for a {requested}-token bound."
            )
        draft_costs.append(proposal.cost_seconds)
        if not proposal.token_ids:
            target_position = adapter.evaluate_target_next(tuple(generated))
            if adapter.generator_state_hash() != rng_before_non_sampling_work:
                raise RuntimeError(
                    "Empty draft fallback evaluation advanced the baseline generator before sampling."
                )
            target_token = int(target_position.sample(adapter.generator))
            if target_token < 0:
                raise ValueError("Target sampling returned a negative token ID.")
            generated.append(target_token)
            target_positions_evaluated += 1
            target_positions_sampled += 1
            if target_token == config.eos_token_id:
                stop_reason = "eos"
            elif len(generated) == config.max_new_tokens:
                stop_reason = "max_new_tokens"
            cache_events.append(
                CacheCommitEvent(
                    iteration=len(cache_events),
                    iteration_kind=DecodeIterationKind.TARGET_ONLY_FALLBACK,
                    prefix_length_before=prefix_before,
                    proposal_length=0,
                    target_positions_evaluated=1,
                    target_positions_sampled=1,
                    accepted_draft_length=0,
                    emitted_target_token_on_mismatch=False,
                    mismatch_index=None,
                    committed_token_ids=(target_token,),
                    prefix_length_after=len(generated),
                    discarded_uncommitted_suffix_length=0,
                    stopped_after_iteration=stop_reason is not None,
                )
            )
            continue

        span = adapter.evaluate_target_span(tuple(generated), proposal.token_ids)
        if len(span.positions) != len(proposal.token_ids):
            raise ValueError(
                "Target span length must exactly match the bounded draft proposal length "
                f"({len(span.positions)} != {len(proposal.token_ids)})."
            )
        if adapter.generator_state_hash() != rng_before_non_sampling_work:
            raise RuntimeError(
                "Draft proposal or target span evaluation advanced the baseline generator before sampling."
            )

        proposed_draft_tokens += len(proposal.token_ids)
        target_positions_evaluated += len(span.positions)
        committed_this_iteration: list[int] = []
        accepted_this_iteration = 0
        mismatch_index: int | None = None
        emitted_target_on_mismatch = False

        for index, (draft_token, target_position) in enumerate(zip(proposal.token_ids, span.positions)):
            # This is deliberately the only RNG-consuming operation in the loop.
            target_token = int(target_position.sample(adapter.generator))
            if target_token < 0:
                raise ValueError("Target sampling returned a negative token ID.")
            target_positions_sampled += 1

            if target_token == draft_token:
                accepted_this_iteration += 1
                accepted_draft_tokens += 1
                emitted_token = int(draft_token)
            else:
                mismatch_index = index
                emitted_target_on_mismatch = True
                emitted_token = target_token

            generated.append(emitted_token)
            committed_this_iteration.append(emitted_token)

            if emitted_token == config.eos_token_id:
                stop_reason = "eos"
                break
            if len(generated) == config.max_new_tokens:
                stop_reason = "max_new_tokens"
                break
            if mismatch_index is not None:
                break

        cache_events.append(
            CacheCommitEvent(
                iteration=len(cache_events),
                iteration_kind=DecodeIterationKind.SPECULATIVE_SPAN,
                prefix_length_before=prefix_before,
                proposal_length=len(proposal.token_ids),
                target_positions_evaluated=len(span.positions),
                target_positions_sampled=len(committed_this_iteration),
                accepted_draft_length=accepted_this_iteration,
                emitted_target_token_on_mismatch=emitted_target_on_mismatch,
                mismatch_index=mismatch_index,
                committed_token_ids=tuple(committed_this_iteration),
                prefix_length_after=len(generated),
                discarded_uncommitted_suffix_length=(
                    len(proposal.token_ids) - accepted_this_iteration
                ),
                stopped_after_iteration=stop_reason is not None,
            )
        )

    if stop_reason is None:
        raise AssertionError("Speculative verifier exited without a stop reason.")

    all_costs_measured = bool(draft_costs) and all(cost is not None for cost in draft_costs)
    draft_cost_seconds = (
        sum(float(cost) for cost in draft_costs if cost is not None)
        if all_costs_measured
        else None
    )
    target_model_calls = len(cache_events)
    serial_target_model_calls = len(generated)

    return SpeculativeDecodeResult(
        generated_token_ids=tuple(generated),
        stop_reason=stop_reason,
        final_rng_state_hash=adapter.generator_state_hash(),
        speculation_k=config.speculation_k,
        draft_calls=len(cache_events),
        proposed_draft_tokens=proposed_draft_tokens,
        accepted_draft_tokens=accepted_draft_tokens,
        target_model_calls=target_model_calls,
        target_positions_evaluated=target_positions_evaluated,
        target_positions_sampled=target_positions_sampled,
        serial_target_model_calls=serial_target_model_calls,
        target_model_calls_saved=serial_target_model_calls - target_model_calls,
        serial_target_token_steps_saved=serial_target_model_calls - target_model_calls,
        unused_target_positions_evaluated=target_positions_evaluated - target_positions_sampled,
        draft_cost_seconds=draft_cost_seconds,
        draft_cost_status="measured" if all_costs_measured else "not_measured",
        cache_events=tuple(cache_events),
    )
