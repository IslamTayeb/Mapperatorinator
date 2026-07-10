"""Model-free cost gate for a v32-mini K=4 speculative draft scout.

The helper intentionally contains no model loading, CUDA code, or runtime/cache
wiring.  It projects an *empirical closed-loop accepted-prefix histogram* onto
the accepted target q1 denominator and the measured fixed-shape K4 target graph.
Marginal draft-token acceptance alone is not a sufficient performance metric:
the first mismatch ends a span and emits one target fallback token.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence


FIVE_FULL_TARGET_SECONDS = 163.088
FIVE_FULL_MAIN_TOKENS = 42_634
ACCEPTED_TARGET_Q1_MS = FIVE_FULL_TARGET_SECONDS * 1000.0 / FIVE_FULL_MAIN_TOKENS

# Job 49547134, commit f4c2156.  This is the safe fixed-shape graph replay
# ceiling; it is not a production runtime measurement.
MEASURED_TARGET_K4_GRAPH_MS = 8.4637517
DEFAULT_SPECULATION_K = 4
CAMPAIGN_MINIMUM_IMPROVEMENT_FRACTION = 0.05


@dataclass(frozen=True)
class ClosedLoopAcceptanceHistogram:
    """Counts of accepted draft-prefix lengths from full K-token proposals.

    ``counts[j]`` means exactly ``j`` draft tokens matched before the first
    mismatch.  ``counts[K]`` means the complete K-token proposal matched.  A
    mismatch iteration commits ``j + 1`` target output tokens because the
    mismatching target token is emitted; a fully accepted iteration commits K.

    Short EOS/max-token boundary proposals must be reported separately instead
    of being mixed into this steady full-span histogram.
    """

    speculation_k: int
    counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.speculation_k <= 0:
            raise ValueError("speculation_k must be positive.")
        object.__setattr__(self, "counts", tuple(self.counts))
        if len(self.counts) != self.speculation_k + 1:
            raise ValueError("counts must contain one bin for every accepted prefix length 0..K.")
        if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in self.counts):
            raise ValueError("Histogram counts must be non-negative integers.")
        if self.full_span_calls == 0:
            raise ValueError("At least one full-span proposal is required.")

    @property
    def full_span_calls(self) -> int:
        return sum(self.counts)

    @property
    def accepted_draft_tokens(self) -> int:
        return sum(prefix_length * count for prefix_length, count in enumerate(self.counts))

    @property
    def proposed_draft_tokens(self) -> int:
        return self.speculation_k * self.full_span_calls

    @property
    def committed_output_tokens(self) -> int:
        return sum(
            (prefix_length + 1 if prefix_length < self.speculation_k else self.speculation_k) * count
            for prefix_length, count in enumerate(self.counts)
        )

    @property
    def draft_token_acceptance_fraction(self) -> float:
        return self.accepted_draft_tokens / self.proposed_draft_tokens

    @property
    def full_accept_fraction(self) -> float:
        return self.counts[-1] / self.full_span_calls

    @property
    def committed_tokens_per_target_call(self) -> float:
        return self.committed_output_tokens / self.full_span_calls

    @property
    def committed_span_efficiency(self) -> float:
        """Committed tokens per call divided by K, not marginal acceptance."""

        return self.committed_tokens_per_target_call / self.speculation_k

    def to_manifest(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            full_span_calls=self.full_span_calls,
            accepted_draft_tokens=self.accepted_draft_tokens,
            proposed_draft_tokens=self.proposed_draft_tokens,
            committed_output_tokens=self.committed_output_tokens,
            draft_token_acceptance_fraction=self.draft_token_acceptance_fraction,
            full_accept_fraction=self.full_accept_fraction,
            committed_tokens_per_target_call=self.committed_tokens_per_target_call,
            committed_span_efficiency=self.committed_span_efficiency,
        )
        result["counts"] = list(self.counts)
        return result


@dataclass(frozen=True)
class MiniDraftProjection:
    """Finite-scout cost projection using exact empirical closed-loop counts."""

    histogram: ClosedLoopAcceptanceHistogram
    steady_draft_ms_per_call: float
    fixed_draft_overhead_ms: float = 0.0
    target_span_ms_per_call: float = MEASURED_TARGET_K4_GRAPH_MS
    baseline_target_q1_ms: float = ACCEPTED_TARGET_Q1_MS
    minimum_improvement_fraction: float = CAMPAIGN_MINIMUM_IMPROVEMENT_FRACTION

    def __post_init__(self) -> None:
        for name in (
                "steady_draft_ms_per_call",
                "fixed_draft_overhead_ms",
                "target_span_ms_per_call",
                "baseline_target_q1_ms",
        ):
            value = getattr(self, name)
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative.")
        if not 0 <= self.minimum_improvement_fraction < 1:
            raise ValueError("minimum_improvement_fraction must be in [0, 1).")

    @property
    def baseline_ms(self) -> float:
        return self.histogram.committed_output_tokens * self.baseline_target_q1_ms

    @property
    def candidate_ms(self) -> float:
        return (
            self.histogram.full_span_calls
            * (self.target_span_ms_per_call + self.steady_draft_ms_per_call)
            + self.fixed_draft_overhead_ms
        )

    @property
    def candidate_ms_per_committed_token(self) -> float:
        return self.candidate_ms / self.histogram.committed_output_tokens

    @property
    def improvement_fraction(self) -> float:
        return (self.baseline_ms - self.candidate_ms) / self.baseline_ms

    @property
    def break_even_draft_budget_ms_per_call(self) -> float:
        return self._draft_budget_ms_per_call(0.0)

    @property
    def minimum_improvement_draft_budget_ms_per_call(self) -> float:
        return self._draft_budget_ms_per_call(self.minimum_improvement_fraction)

    @property
    def clears_minimum_improvement(self) -> bool:
        # The campaign threshold is strict: exactly 5% is not enough.
        return self.improvement_fraction > self.minimum_improvement_fraction

    def _draft_budget_ms_per_call(self, improvement_fraction: float) -> float:
        allowed_total_ms = (1.0 - improvement_fraction) * self.baseline_ms
        target_and_fixed_ms = (
            self.histogram.full_span_calls * self.target_span_ms_per_call
            + self.fixed_draft_overhead_ms
        )
        return (allowed_total_ms - target_and_fixed_ms) / self.histogram.full_span_calls

    def to_manifest(self) -> dict[str, Any]:
        return {
            "histogram": self.histogram.to_manifest(),
            "steady_draft_ms_per_call": self.steady_draft_ms_per_call,
            "fixed_draft_overhead_ms": self.fixed_draft_overhead_ms,
            "target_span_ms_per_call": self.target_span_ms_per_call,
            "baseline_target_q1_ms": self.baseline_target_q1_ms,
            "minimum_improvement_fraction": self.minimum_improvement_fraction,
            "baseline_ms": self.baseline_ms,
            "candidate_ms": self.candidate_ms,
            "candidate_ms_per_committed_token": self.candidate_ms_per_committed_token,
            "improvement_fraction": self.improvement_fraction,
            "break_even_draft_budget_ms_per_call": self.break_even_draft_budget_ms_per_call,
            "minimum_improvement_draft_budget_ms_per_call": (
                self.minimum_improvement_draft_budget_ms_per_call
            ),
            "clears_minimum_improvement": self.clears_minimum_improvement,
        }


def draft_cost_surface_ms(
        committed_tokens_per_target_call: float,
        *,
        target_span_ms_per_call: float = MEASURED_TARGET_K4_GRAPH_MS,
        baseline_target_q1_ms: float = ACCEPTED_TARGET_Q1_MS,
        improvement_fractions: Sequence[float] = (0.0, CAMPAIGN_MINIMUM_IMPROVEMENT_FRACTION),
) -> dict[float, float]:
    """Return the strict draft-cost boundary for each requested improvement.

    For improvement ``s`` and effective closed-loop length ``L``, the boundary
    is ``D = (1 - s) * q1_ms * L - target_span_ms``.  A measured steady draft
    cost must be strictly below the returned boundary.  Negative boundaries are
    impossible to clear with a non-negative-cost draft.
    """

    if committed_tokens_per_target_call <= 0 or not math.isfinite(committed_tokens_per_target_call):
        raise ValueError("committed_tokens_per_target_call must be finite and positive.")
    if target_span_ms_per_call < 0 or not math.isfinite(target_span_ms_per_call):
        raise ValueError("target_span_ms_per_call must be finite and non-negative.")
    if baseline_target_q1_ms <= 0 or not math.isfinite(baseline_target_q1_ms):
        raise ValueError("baseline_target_q1_ms must be finite and positive.")

    surface: dict[float, float] = {}
    for improvement_fraction in improvement_fractions:
        if not 0 <= improvement_fraction < 1:
            raise ValueError("Each improvement fraction must be in [0, 1).")
        surface[float(improvement_fraction)] = (
            (1.0 - improvement_fraction)
            * baseline_target_q1_ms
            * committed_tokens_per_target_call
            - target_span_ms_per_call
        )
    return surface

