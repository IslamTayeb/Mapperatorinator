"""Exactness labels and evidence shared by optimized inference experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class ExactnessResultClass(str, Enum):
    """The strongest equivalence claim supported by an optimized result."""

    BITWISE_CALCULATION_EXACT = "bitwise-calculation-exact"
    EXACT_OUTPUT = "exact-output"


@dataclass(frozen=True)
class ExactnessEvidence:
    """Observable generation evidence required for an exact-output claim."""

    result_class: ExactnessResultClass
    generated_token_ids: tuple[int, ...]
    stop_reason: str
    final_rng_state_hash: str
    result_file_sha256: str | None = None
    result_file_size_bytes: int | None = None
    intermediate_state_hashes: Mapping[str, str] = field(default_factory=dict)
    cache_state_hashes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "result_class", ExactnessResultClass(self.result_class))
        object.__setattr__(self, "generated_token_ids", tuple(self.generated_token_ids))
        if not self.stop_reason:
            raise ValueError("stop_reason must be non-empty.")
        if not self.final_rng_state_hash:
            raise ValueError("final_rng_state_hash must be non-empty.")
        artifact_fields = (self.result_file_sha256, self.result_file_size_bytes)
        if (artifact_fields[0] is None) != (artifact_fields[1] is None):
            raise ValueError("result_file_sha256 and result_file_size_bytes must be provided together.")
        if self.result_file_size_bytes is not None and self.result_file_size_bytes < 0:
            raise ValueError("result_file_size_bytes must be non-negative.")
        for evidence_name, hashes in (
                ("intermediate_state_hashes", self.intermediate_state_hashes),
                ("cache_state_hashes", self.cache_state_hashes),
        ):
            if any(not name or not value for name, value in hashes.items()):
                raise ValueError(f"{evidence_name} keys and values must be non-empty.")
        if self.result_class is ExactnessResultClass.BITWISE_CALCULATION_EXACT:
            if not self.intermediate_state_hashes or not self.cache_state_hashes:
                raise ValueError(
                    "bitwise-calculation-exact requires explicit intermediate_state_hashes "
                    "and cache_state_hashes evidence."
                )
