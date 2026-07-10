"""CPU-only generated-prefix n-gram acceptance replay.

This module deliberately consumes recorded target transcripts rather than a model.
It can reject a weak prompt-lookup proposal source before GPU work, or justify the
next q_len=K verifier, but it cannot establish exact target numerics or throughput.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


CAMPAIGN_SPECULATION_K = (2, 4, 8)
DEFAULT_MAX_NGRAM_LENGTH = 4
DEFAULT_ADVANCE_THRESHOLD = 0.05
STRUCTURAL_RESULT_CLASS = "structural_ngram_acceptance_ceiling"
STRUCTURAL_CLAIM_SCOPE = "cpu_only_generated_prefix_replay_not_tps"
STRUCTURAL_LIMITATIONS = (
    "Only already committed generated_token_ids from each window are used; prompt-token history is not available.",
    "q_len=K target numerical equivalence and concrete cache commit/rollback behavior are not measured.",
    "N-gram proposal, draft-model, target-model, and q_len=K target costs are not measured; this is not a TPS claim.",
    "Call reduction assumes optimistically that one q_len=K target span costs one serial q_len=1 target call.",
)


def _validate_token_ids(token_ids: Sequence[int], *, field_name: str) -> tuple[int, ...]:
    tokens = tuple(token_ids)
    if not tokens:
        raise ValueError(f"{field_name} must contain at least one generated token ID.")
    if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in tokens):
        raise TypeError(f"{field_name} must contain only non-negative integer token IDs.")
    return tokens


@dataclass(frozen=True)
class GeneratedPrefixNgramProposal:
    """One causal prompt-lookup proposal backed only by committed output tokens."""

    token_ids: tuple[int, ...]
    matched_ngram_length: int
    source_start: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "token_ids",
            _validate_token_ids(self.token_ids, field_name="proposal token_ids"),
        )
        if self.matched_ngram_length <= 0:
            raise ValueError("matched_ngram_length must be positive.")
        if self.source_start < 0:
            raise ValueError("source_start must be non-negative.")


def find_generated_prefix_ngram_proposal(
        committed_token_ids: Sequence[int],
        *,
        max_proposal_tokens: int,
        max_ngram_length: int,
) -> GeneratedPrefixNgramProposal | None:
    """Find the longest generated-prefix suffix with an earlier continuation.

    The lookup is causal: both the matched occurrence and every proposed token are
    already present in ``committed_token_ids``.  For equal n-gram lengths, the most
    recent earlier occurrence wins so the tie-break is deterministic and local.
    """

    if max_proposal_tokens <= 0:
        raise ValueError("max_proposal_tokens must be positive.")
    if max_ngram_length <= 0:
        raise ValueError("max_ngram_length must be positive.")
    committed = tuple(committed_token_ids)
    if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in committed):
        raise TypeError("committed_token_ids must contain only non-negative integer token IDs.")
    if not committed:
        return None

    longest = min(max_ngram_length, len(committed))
    for ngram_length in range(longest, 0, -1):
        suffix = committed[-ngram_length:]
        # The current suffix occurrence has no committed continuation.  Search
        # only starts whose matching n-gram ends before the committed prefix does.
        latest_source_start = len(committed) - ngram_length - 1
        for source_start in range(latest_source_start, -1, -1):
            if committed[source_start:source_start + ngram_length] != suffix:
                continue
            continuation_start = source_start + ngram_length
            continuation = committed[
                continuation_start:continuation_start + max_proposal_tokens
            ]
            if continuation:
                return GeneratedPrefixNgramProposal(
                    token_ids=continuation,
                    matched_ngram_length=ngram_length,
                    source_start=source_start,
                )
    return None


@dataclass(frozen=True)
class NgramAcceptanceMetrics:
    """Count-based structural acceptance and target-call accounting."""

    speculation_k: int
    generated_tokens: int
    proposed_tokens: int
    accepted_tokens: int
    serial_target_calls: int
    speculative_target_calls: int
    target_calls_saved: int
    serial_target_token_steps_saved: int
    empty_proposal_calls: int
    mismatch_calls: int
    full_accept_calls: int

    def __post_init__(self) -> None:
        if self.speculation_k not in CAMPAIGN_SPECULATION_K:
            raise ValueError(f"speculation_k must be one of {CAMPAIGN_SPECULATION_K}.")
        count_fields = (
            "generated_tokens",
            "proposed_tokens",
            "accepted_tokens",
            "serial_target_calls",
            "speculative_target_calls",
            "target_calls_saved",
            "serial_target_token_steps_saved",
            "empty_proposal_calls",
            "mismatch_calls",
            "full_accept_calls",
        )
        if any(
                not isinstance(getattr(self, field_name), int)
                or isinstance(getattr(self, field_name), bool)
                or getattr(self, field_name) < 0
                for field_name in count_fields
        ):
            raise TypeError("N-gram acceptance counters must be non-negative integers.")
        if self.generated_tokens <= 0:
            raise ValueError("generated_tokens must be positive.")
        if self.serial_target_calls != self.generated_tokens:
            raise ValueError("Serial target calls must equal the recorded generated-token count.")
        if self.accepted_tokens > self.proposed_tokens:
            raise ValueError("accepted_tokens cannot exceed proposed_tokens.")
        if self.accepted_tokens > self.generated_tokens:
            raise ValueError("accepted_tokens cannot exceed generated_tokens.")
        classified_calls = self.empty_proposal_calls + self.mismatch_calls + self.full_accept_calls
        if self.speculative_target_calls != classified_calls:
            raise ValueError("Every speculative target call must have exactly one outcome classification.")
        if self.target_calls_saved != self.serial_target_calls - self.speculative_target_calls:
            raise ValueError("target_calls_saved is inconsistent with serial and speculative calls.")
        if self.serial_target_token_steps_saved != self.target_calls_saved:
            raise ValueError("Serial target token-step savings must use the target-call denominator.")

    @property
    def proposal_acceptance_rate(self) -> float:
        return self.accepted_tokens / self.proposed_tokens if self.proposed_tokens else 0.0

    @property
    def accepted_token_coverage(self) -> float:
        return self.accepted_tokens / self.generated_tokens

    @property
    def target_call_reduction_rate(self) -> float:
        return self.target_calls_saved / self.serial_target_calls

    @property
    def committed_tokens_per_speculative_target_call(self) -> float:
        return self.generated_tokens / self.speculative_target_calls

    def to_dict(self) -> dict[str, int | float]:
        return {
            "speculation_k": self.speculation_k,
            "generated_tokens": self.generated_tokens,
            "proposed_tokens": self.proposed_tokens,
            "accepted_tokens": self.accepted_tokens,
            "proposal_acceptance_rate": self.proposal_acceptance_rate,
            "accepted_token_coverage": self.accepted_token_coverage,
            "serial_target_calls": self.serial_target_calls,
            "speculative_target_calls": self.speculative_target_calls,
            "target_calls_saved": self.target_calls_saved,
            "serial_target_token_steps_saved": self.serial_target_token_steps_saved,
            "target_call_reduction_rate": self.target_call_reduction_rate,
            "committed_tokens_per_speculative_target_call": (
                self.committed_tokens_per_speculative_target_call
            ),
            "empty_proposal_calls": self.empty_proposal_calls,
            "mismatch_calls": self.mismatch_calls,
            "full_accept_calls": self.full_accept_calls,
        }


def simulate_generated_prefix_ngram_acceptance(
        generated_token_ids: Sequence[int],
        *,
        speculation_k: int,
        max_ngram_length: int,
) -> NgramAcceptanceMetrics:
    """Replay one exact transcript with causal generated-prefix proposals."""

    tokens = _validate_token_ids(generated_token_ids, field_name="generated_token_ids")
    if speculation_k not in CAMPAIGN_SPECULATION_K:
        raise ValueError(f"speculation_k must be one of {CAMPAIGN_SPECULATION_K}.")
    if max_ngram_length <= 0:
        raise ValueError("max_ngram_length must be positive.")

    cursor = 0
    proposed_tokens = 0
    accepted_tokens = 0
    speculative_target_calls = 0
    empty_proposal_calls = 0
    mismatch_calls = 0
    full_accept_calls = 0

    while cursor < len(tokens):
        remaining = len(tokens) - cursor
        proposal_match = find_generated_prefix_ngram_proposal(
            tokens[:cursor],
            max_proposal_tokens=min(speculation_k, remaining),
            max_ngram_length=max_ngram_length,
        )
        speculative_target_calls += 1
        if proposal_match is None:
            empty_proposal_calls += 1
            cursor += 1
            continue

        proposal = proposal_match.token_ids[:remaining]
        proposed_tokens += len(proposal)
        accepted = 0
        for proposed_token, target_token in zip(proposal, tokens[cursor:]):
            if proposed_token != target_token:
                break
            accepted += 1
        accepted_tokens += accepted

        if accepted == len(proposal):
            full_accept_calls += 1
            cursor += accepted
        else:
            mismatch_calls += 1
            # The same target span call emits the exact target token at the first
            # mismatch after committing the accepted proposal prefix.
            cursor += accepted + 1

    serial_target_calls = len(tokens)
    target_calls_saved = serial_target_calls - speculative_target_calls
    return NgramAcceptanceMetrics(
        speculation_k=speculation_k,
        generated_tokens=len(tokens),
        proposed_tokens=proposed_tokens,
        accepted_tokens=accepted_tokens,
        serial_target_calls=serial_target_calls,
        speculative_target_calls=speculative_target_calls,
        target_calls_saved=target_calls_saved,
        serial_target_token_steps_saved=target_calls_saved,
        empty_proposal_calls=empty_proposal_calls,
        mismatch_calls=mismatch_calls,
        full_accept_calls=full_accept_calls,
    )


def aggregate_ngram_acceptance_metrics(
        metrics: Iterable[NgramAcceptanceMetrics],
) -> NgramAcceptanceMetrics:
    """Aggregate raw counts so reported rates are token-weighted, not mean-of-means."""

    rows = tuple(metrics)
    if not rows:
        raise ValueError("At least one window metric is required for aggregation.")
    speculation_k = rows[0].speculation_k
    if any(row.speculation_k != speculation_k for row in rows):
        raise ValueError("Cannot aggregate N-gram metrics from different speculation_k values.")
    return NgramAcceptanceMetrics(
        speculation_k=speculation_k,
        generated_tokens=sum(row.generated_tokens for row in rows),
        proposed_tokens=sum(row.proposed_tokens for row in rows),
        accepted_tokens=sum(row.accepted_tokens for row in rows),
        serial_target_calls=sum(row.serial_target_calls for row in rows),
        speculative_target_calls=sum(row.speculative_target_calls for row in rows),
        target_calls_saved=sum(row.target_calls_saved for row in rows),
        serial_target_token_steps_saved=sum(row.serial_target_token_steps_saved for row in rows),
        empty_proposal_calls=sum(row.empty_proposal_calls for row in rows),
        mismatch_calls=sum(row.mismatch_calls for row in rows),
        full_accept_calls=sum(row.full_accept_calls for row in rows),
    )


@dataclass(frozen=True)
class ProfileTokenWindow:
    """Validated per-window transcript extracted from one inference profile."""

    profile_path: str
    generation_record_index: int
    profile_label: str
    context_type: str
    mode: str
    sequence_index: int | None
    batch_start_index: int | None
    generated_token_ids: tuple[int, ...]

    @property
    def window_id(self) -> str:
        if self.sequence_index is not None:
            unit = f"seq{self.sequence_index}"
        elif self.batch_start_index is not None:
            unit = f"batch{self.batch_start_index}"
        else:
            unit = f"record{self.generation_record_index}"
        return (
            f"{self.profile_path}:{self.profile_label}:{self.context_type}:"
            f"{self.mode}:{unit}"
        )


def load_profile_token_windows(
        profile_path: Path,
        *,
        profile_label: str,
) -> tuple[ProfileTokenWindow, ...]:
    """Load every selected per-window transcript, failing on partial token evidence."""

    if not profile_path.is_file():
        raise FileNotFoundError(f"Inference profile does not exist: {profile_path}")
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Inference profile is not valid JSON: {profile_path}: {error}") from error
    if not isinstance(profile, dict):
        raise TypeError(f"Inference profile root must be an object: {profile_path}")
    generation = profile.get("generation")
    if not isinstance(generation, list):
        raise ValueError(f"Inference profile must contain a generation list: {profile_path}")

    selected = [
        (index, record)
        for index, record in enumerate(generation)
        if isinstance(record, dict) and record.get("profile_label") == profile_label
    ]
    if not selected:
        raise ValueError(
            f"Inference profile has no generation records for profile_label={profile_label!r}: "
            f"{profile_path}"
        )

    windows: list[ProfileTokenWindow] = []
    for record_index, record in selected:
        if "generated_token_ids" not in record or record.get("generated_token_ids") is None:
            raise ValueError(
                f"Missing per-window generated_token_ids in {profile_path} generation record "
                f"{record_index}; rerun with profile_record_token_ids=true."
            )
        raw_tokens = record["generated_token_ids"]
        if not isinstance(raw_tokens, list):
            raise TypeError(
                f"generated_token_ids must be a list in {profile_path} generation record "
                f"{record_index}."
            )
        tokens = _validate_token_ids(
            raw_tokens,
            field_name=f"{profile_path} generation record {record_index} generated_token_ids",
        )
        reported_count = record.get("generated_tokens")
        if reported_count is not None and (
                not isinstance(reported_count, int)
                or isinstance(reported_count, bool)
                or reported_count != len(tokens)
        ):
            raise ValueError(
                f"generated_tokens does not match generated_token_ids length in {profile_path} "
                f"generation record {record_index}: {reported_count!r} != {len(tokens)}."
            )

        sequence_index = record.get("sequence_index")
        batch_start_index = record.get("batch_start_index")
        for field_name, value in (
                ("sequence_index", sequence_index),
                ("batch_start_index", batch_start_index),
        ):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                raise TypeError(
                    f"{field_name} must be an integer in {profile_path} generation record "
                    f"{record_index}."
                )
        windows.append(ProfileTokenWindow(
            profile_path=str(profile_path),
            generation_record_index=record_index,
            profile_label=profile_label,
            context_type=str(record.get("context_type", "unknown")),
            mode=str(record.get("mode", "unknown")),
            sequence_index=sequence_index,
            batch_start_index=batch_start_index,
            generated_token_ids=tokens,
        ))
    return tuple(windows)


def run_generated_prefix_ngram_profile_scout(
        profile_paths: Sequence[Path],
        *,
        profile_label: str = "main_generation",
        speculation_k_values: Sequence[int] = CAMPAIGN_SPECULATION_K,
        max_ngram_length: int = DEFAULT_MAX_NGRAM_LENGTH,
        advance_threshold: float = DEFAULT_ADVANCE_THRESHOLD,
) -> dict[str, Any]:
    """Build per-window and token-weighted structural acceptance evidence."""

    if not profile_paths:
        raise ValueError("At least one inference profile path is required.")
    if not profile_label:
        raise ValueError("profile_label must be non-empty.")
    if max_ngram_length <= 0:
        raise ValueError("max_ngram_length must be positive.")
    if not math.isfinite(advance_threshold) or not 0 <= advance_threshold <= 1:
        raise ValueError("advance_threshold must be finite and between 0 and 1.")
    k_values = tuple(speculation_k_values)
    if not k_values or len(set(k_values)) != len(k_values):
        raise ValueError("speculation_k_values must contain unique campaign K values.")
    if any(k not in CAMPAIGN_SPECULATION_K for k in k_values):
        raise ValueError(f"speculation_k_values must be drawn from {CAMPAIGN_SPECULATION_K}.")

    windows = tuple(
        window
        for path in profile_paths
        for window in load_profile_token_windows(Path(path), profile_label=profile_label)
    )
    metrics_by_window: list[dict[str, Any]] = []
    rows_by_k: dict[int, list[NgramAcceptanceMetrics]] = {k: [] for k in k_values}
    for window in windows:
        k_metrics: dict[str, Any] = {}
        for speculation_k in k_values:
            metrics = simulate_generated_prefix_ngram_acceptance(
                window.generated_token_ids,
                speculation_k=speculation_k,
                max_ngram_length=max_ngram_length,
            )
            rows_by_k[speculation_k].append(metrics)
            k_metrics[str(speculation_k)] = metrics.to_dict()
        metrics_by_window.append({
            "window_id": window.window_id,
            "profile_path": window.profile_path,
            "generation_record_index": window.generation_record_index,
            "profile_label": window.profile_label,
            "context_type": window.context_type,
            "mode": window.mode,
            "sequence_index": window.sequence_index,
            "batch_start_index": window.batch_start_index,
            "generated_tokens": len(window.generated_token_ids),
            "metrics_by_k": k_metrics,
        })

    weighted = {
        str(speculation_k): aggregate_ngram_acceptance_metrics(
            rows_by_k[speculation_k]
        ).to_dict()
        for speculation_k in k_values
    }
    best_k = max(
        k_values,
        key=lambda k: weighted[str(k)]["target_call_reduction_rate"],
    )
    best_reduction = float(weighted[str(best_k)]["target_call_reduction_rate"])
    if best_reduction < advance_threshold:
        triage_decision = "reject_before_gpu_below_structural_call_reduction_bar"
        triage_reason = (
            "Even the zero-cost generated-prefix structural replay is below the advance threshold; "
            "measured proposal and q_len=K target costs can only make the case weaker."
        )
    else:
        triage_decision = "advance_to_gpu_q_len_k_numeric_and_cost_gate"
        triage_reason = (
            "Generated-prefix acceptance clears the structural call-reduction bar, but GPU target "
            "numerics, cache behavior, and proposal/target costs must be measured before any speed claim."
        )

    return {
        "result_class": STRUCTURAL_RESULT_CLASS,
        "claim_scope": STRUCTURAL_CLAIM_SCOPE,
        "proposal_source": "causal_generated_prefix_ngram",
        "profile_label": profile_label,
        "profile_paths": [str(Path(path)) for path in profile_paths],
        "window_count": len(windows),
        "max_ngram_length": max_ngram_length,
        "speculation_k_values": list(k_values),
        "limitations": list(STRUCTURAL_LIMITATIONS),
        "target_call_accounting_assumption": (
            "one speculative q_len=K target span equals one serial q_len=1 target call; "
            "proposal cost is zero"
        ),
        "windows": metrics_by_window,
        "weighted_totals_by_k": weighted,
        "triage": {
            "advance_threshold": advance_threshold,
            "best_speculation_k": best_k,
            "best_structural_target_call_reduction_rate": best_reduction,
            "decision": triage_decision,
            "reason": triage_reason,
        },
    }
