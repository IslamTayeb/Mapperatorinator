"""Verifier-first exact speculative decoding contracts.

This package deliberately contains no model loading or production runtime wiring.  The
state machine can be exercised with scripted adapters before a GPU-backed target/draft
adapter is allowed to run a bounded scout.
"""

from .contracts import (
    CacheCommitEvent,
    DraftProposal,
    EvaluatedTargetSpan,
    ExactTranscriptComparison,
    SequentialTargetResult,
    SpeculationConfig,
    SpeculativeDecodeResult,
    TargetTokenDistribution,
)
from .scout import (
    MissingSpeculativeScoutAdapterError,
    ProposalSourceKind,
    SpeculativeScoutConfig,
    SpeculativeScoutAdapter,
    load_scout_adapter_factory,
    run_bounded_speculative_scout,
)
from .verifier import (
    ExactSpeculativeAdapter,
    SequentialTargetAdapter,
    compare_exact_transcript,
    run_exact_speculative_decode,
    run_sequential_target_reference,
)

__all__ = [
    "CacheCommitEvent",
    "DraftProposal",
    "EvaluatedTargetSpan",
    "ExactTranscriptComparison",
    "ExactSpeculativeAdapter",
    "MissingSpeculativeScoutAdapterError",
    "ProposalSourceKind",
    "SequentialTargetAdapter",
    "SequentialTargetResult",
    "SpeculationConfig",
    "SpeculativeDecodeResult",
    "SpeculativeScoutConfig",
    "SpeculativeScoutAdapter",
    "TargetTokenDistribution",
    "compare_exact_transcript",
    "load_scout_adapter_factory",
    "run_bounded_speculative_scout",
    "run_exact_speculative_decode",
    "run_sequential_target_reference",
]
