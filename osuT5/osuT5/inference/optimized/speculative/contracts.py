"""Model-independent contracts for exact fixed-K speculative verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence


class TargetTokenDistribution(Protocol):
    """One target position that can consume exactly one baseline RNG draw."""

    def sample(self, generator: Any) -> int:
        """Sample one token using the request-private baseline generator."""


@dataclass(frozen=True)
class DraftProposal:
    """A bounded draft suffix and optional measured draft cost."""

    token_ids: tuple[int, ...]
    cost_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(int(token) for token in self.token_ids))
        if not self.token_ids:
            raise ValueError("A speculative draft proposal must contain at least one token.")
        if any(token < 0 for token in self.token_ids):
            raise ValueError("Draft token IDs must be non-negative.")
        if self.cost_seconds is not None and self.cost_seconds < 0:
            raise ValueError("Draft proposal cost_seconds must be non-negative when measured.")


@dataclass(frozen=True)
class EvaluatedTargetSpan:
    """Target distributions evaluated in one model call against the draft prefix."""

    positions: tuple[TargetTokenDistribution, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", tuple(self.positions))
        if not self.positions:
            raise ValueError("An evaluated target span must contain at least one position.")


@dataclass(frozen=True)
class SpeculationConfig:
    """The deliberately small verifier gate; production scheduling is out of scope."""

    speculation_k: int
    max_new_tokens: int
    eos_token_id: int

    def __post_init__(self) -> None:
        if self.speculation_k not in (2, 4, 8):
            raise ValueError("speculation_k must be one of 2, 4, or 8 for this campaign.")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if self.eos_token_id < 0:
            raise ValueError("eos_token_id must be non-negative.")


@dataclass(frozen=True)
class CacheCommitEvent:
    """Logical prefix ownership for one evaluated speculative span.

    This does not claim a particular model KV layout.  A GPU adapter must map this
    logical commit/rollback contract to its concrete self/cross-cache representation.
    """

    iteration: int
    prefix_length_before: int
    proposal_length: int
    target_positions_evaluated: int
    target_positions_sampled: int
    accepted_draft_length: int
    emitted_target_token_on_mismatch: bool
    mismatch_index: int | None
    committed_token_ids: tuple[int, ...]
    prefix_length_after: int
    discarded_uncommitted_suffix_length: int
    stopped_after_iteration: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "committed_token_ids", tuple(self.committed_token_ids))
        expected_after = self.prefix_length_before + len(self.committed_token_ids)
        if self.prefix_length_after != expected_after:
            raise ValueError("Cache event prefix_length_after does not match committed tokens.")
        if self.target_positions_sampled != len(self.committed_token_ids):
            raise ValueError("Each committed output position must consume exactly one target sample.")
        if self.target_positions_evaluated != self.proposal_length:
            raise ValueError("Target evaluation length must equal proposal length.")
        if self.accepted_draft_length < 0 or self.accepted_draft_length > self.proposal_length:
            raise ValueError("accepted_draft_length is outside the proposal.")
        expected_discarded = self.proposal_length - self.accepted_draft_length
        if self.discarded_uncommitted_suffix_length != expected_discarded:
            raise ValueError("Discarded suffix must contain every draft token that was not accepted.")
        if self.emitted_target_token_on_mismatch != (self.mismatch_index is not None):
            raise ValueError("Mismatch index and target fallback emission must be reported together.")
        if self.mismatch_index is not None and not 0 <= self.mismatch_index < self.proposal_length:
            raise ValueError("mismatch_index is outside the proposal.")


@dataclass(frozen=True)
class SpeculativeDecodeResult:
    """Exactness and work-accounting evidence from the verifier state machine."""

    generated_token_ids: tuple[int, ...]
    stop_reason: str
    final_rng_state_hash: str
    speculation_k: int
    draft_calls: int
    proposed_draft_tokens: int
    accepted_draft_tokens: int
    target_model_calls: int
    target_positions_evaluated: int
    target_positions_sampled: int
    serial_target_model_calls: int
    target_model_calls_saved: int
    serial_target_token_steps_saved: int
    unused_target_positions_evaluated: int
    draft_cost_seconds: float | None
    draft_cost_status: str
    cache_events: tuple[CacheCommitEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "generated_token_ids", tuple(self.generated_token_ids))
        object.__setattr__(self, "cache_events", tuple(self.cache_events))
        if self.stop_reason not in ("eos", "max_new_tokens"):
            raise ValueError(f"Unsupported stop_reason: {self.stop_reason!r}")
        if self.target_positions_sampled != len(self.generated_token_ids):
            raise ValueError("Target sample count must equal emitted token count.")
        if self.serial_target_model_calls != len(self.generated_token_ids):
            raise ValueError("Serial target call denominator must equal emitted token count.")
        if self.target_model_calls_saved != self.serial_target_model_calls - self.target_model_calls:
            raise ValueError("target_model_calls_saved is inconsistent with its denominator.")
        if self.serial_target_token_steps_saved != self.target_model_calls_saved:
            raise ValueError("Serial target token-step savings must use the target-call denominator.")
        if self.unused_target_positions_evaluated != (
                self.target_positions_evaluated - self.target_positions_sampled
        ):
            raise ValueError("Unused target-position accounting is inconsistent.")

    @property
    def draft_acceptance_rate(self) -> float:
        if self.proposed_draft_tokens == 0:
            return 0.0
        return self.accepted_draft_tokens / self.proposed_draft_tokens

    def to_manifest(self) -> dict[str, Any]:
        manifest = asdict(self)
        manifest["generated_token_ids"] = list(self.generated_token_ids)
        manifest["draft_acceptance_rate"] = self.draft_acceptance_rate
        return manifest


@dataclass(frozen=True)
class SequentialTargetResult:
    """The serial target-only transcript used as the exactness denominator."""

    generated_token_ids: tuple[int, ...]
    stop_reason: str
    final_rng_state_hash: str
    target_model_calls: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "generated_token_ids", tuple(self.generated_token_ids))
        if self.stop_reason not in ("eos", "max_new_tokens"):
            raise ValueError(f"Unsupported stop_reason: {self.stop_reason!r}")
        if self.target_model_calls != len(self.generated_token_ids):
            raise ValueError("Sequential target reference requires one target call per emitted token.")


@dataclass(frozen=True)
class ExactTranscriptComparison:
    """Observable equivalence between a speculative run and its serial denominator."""

    speculative: SpeculativeDecodeResult
    sequential: SequentialTargetResult
    generated_tokens_match: bool
    stop_reason_match: bool
    final_rng_state_match: bool

    @property
    def exact(self) -> bool:
        return self.generated_tokens_match and self.stop_reason_match and self.final_rng_state_match

    def assert_exact(self) -> None:
        if not self.exact:
            raise AssertionError(
                "Speculative transcript is not exact: "
                f"tokens={self.generated_tokens_match}, stop={self.stop_reason_match}, "
                f"rng={self.final_rng_state_match}."
            )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "exact": self.exact,
            "generated_tokens_match": self.generated_tokens_match,
            "stop_reason_match": self.stop_reason_match,
            "final_rng_state_match": self.final_rng_state_match,
            "speculative": self.speculative.to_manifest(),
            "sequential": asdict(self.sequential),
        }
