"""Verifier-first exact speculative decoding contracts.

This package deliberately contains no model loading or production runtime wiring.  The
state machine can be exercised with scripted adapters before a GPU-backed target/draft
adapter is allowed to run a bounded scout.
"""

from .contracts import (
    CacheCommitEvent,
    DecodeIterationKind,
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
from .ngram_acceptance import (
    CAMPAIGN_SPECULATION_K,
    DEFAULT_ADVANCE_THRESHOLD,
    DEFAULT_MAX_NGRAM_LENGTH,
    STRUCTURAL_CLAIM_SCOPE,
    STRUCTURAL_LIMITATIONS,
    STRUCTURAL_RESULT_CLASS,
    GeneratedPrefixNgramProposal,
    NgramAcceptanceMetrics,
    ProfileTokenWindow,
    aggregate_ngram_acceptance_metrics,
    find_generated_prefix_ngram_proposal,
    load_profile_token_windows,
    run_generated_prefix_ngram_profile_scout,
    simulate_generated_prefix_ngram_acceptance,
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
    "CAMPAIGN_SPECULATION_K",
    "DEFAULT_ADVANCE_THRESHOLD",
    "DEFAULT_MAX_NGRAM_LENGTH",
    "DecodeIterationKind",
    "DraftProposal",
    "EvaluatedTargetSpan",
    "ExactTranscriptComparison",
    "ExactSpeculativeAdapter",
    "GeneratedPrefixNgramProposal",
    "MissingSpeculativeScoutAdapterError",
    "NgramAcceptanceMetrics",
    "ProfileTokenWindow",
    "ProposalSourceKind",
    "SequentialTargetAdapter",
    "SequentialTargetResult",
    "SpeculationConfig",
    "SpeculativeDecodeResult",
    "SpeculativeScoutConfig",
    "SpeculativeScoutAdapter",
    "STRUCTURAL_CLAIM_SCOPE",
    "STRUCTURAL_LIMITATIONS",
    "STRUCTURAL_RESULT_CLASS",
    "TargetTokenDistribution",
    "compare_exact_transcript",
    "aggregate_ngram_acceptance_metrics",
    "find_generated_prefix_ngram_proposal",
    "load_profile_token_windows",
    "load_scout_adapter_factory",
    "run_bounded_speculative_scout",
    "run_exact_speculative_decode",
    "run_generated_prefix_ngram_profile_scout",
    "run_sequential_target_reference",
    "simulate_generated_prefix_ngram_acceptance",
]
