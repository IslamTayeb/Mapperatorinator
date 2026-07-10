"""Bounded scout interface for later n-gram and v32-mini GPU experiments."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

from .contracts import ExactTranscriptComparison, SpeculationConfig
from .verifier import (
    ExactSpeculativeAdapter,
    SequentialTargetAdapter,
    compare_exact_transcript,
    run_exact_speculative_decode,
    run_sequential_target_reference,
)


class ProposalSourceKind(str, Enum):
    NGRAM = "ngram"
    V32_MINI = "v32-mini"


V32_MINI_MODEL_ID = "OliBomby/Mapperatorinator-v32-mini"


@dataclass(frozen=True)
class SpeculativeScoutConfig:
    proposal_source: ProposalSourceKind
    speculation: SpeculationConfig
    seed: int
    draft_model_path: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "proposal_source", ProposalSourceKind(self.proposal_source))
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer.")
        if self.proposal_source is ProposalSourceKind.NGRAM and self.draft_model_path is not None:
            raise ValueError("The ngram proposal source must not declare a draft_model_path.")
        if self.proposal_source is ProposalSourceKind.V32_MINI:
            object.__setattr__(self, "draft_model_path", self.draft_model_path or V32_MINI_MODEL_ID)


class MissingSpeculativeScoutAdapterError(RuntimeError):
    """Raised before model loading when no explicit GPU scout adapter exists."""


class SpeculativeScoutAdapter(ExactSpeculativeAdapter, Protocol):
    """GPU adapter plus a fresh serial reference sharing immutable target weights."""

    is_real_gpu_model_adapter: bool
    proposal_source: str

    def make_sequential_reference(self) -> SequentialTargetAdapter:
        """Return private cache/generator state initialized to the identical seed."""


def load_scout_adapter_factory(spec: str) -> Callable[[SpeculativeScoutConfig], SpeculativeScoutAdapter]:
    """Load an explicit ``module:function`` factory without importing it by default."""

    module_name, separator, function_name = spec.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("adapter factory must use the form 'module:function'.")
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name, None)
    if factory is None or not callable(factory):
        raise ValueError(f"Adapter factory {spec!r} is missing or is not callable.")
    return factory


def run_bounded_speculative_scout(
        config: SpeculativeScoutConfig,
        adapter: SpeculativeScoutAdapter | None = None,
) -> ExactTranscriptComparison:
    """Run only with an adapter that explicitly declares real GPU/model ownership."""

    if adapter is None:
        raise MissingSpeculativeScoutAdapterError(
            "The speculative scout is verifier-only until a real GPU/model adapter is supplied. "
            "No model, draft source, or native extension was imported."
        )
    if getattr(adapter, "is_real_gpu_model_adapter", False) is not True:
        raise MissingSpeculativeScoutAdapterError(
            "The bounded scout requires adapter.is_real_gpu_model_adapter=True; "
            "scripted adapters belong in CPU verifier tests only."
        )
    adapter_source = getattr(adapter, "proposal_source", None)
    if adapter_source != config.proposal_source.value:
        raise ValueError(
            "Scout proposal-source mismatch: "
            f"config={config.proposal_source.value!r}, adapter={adapter_source!r}."
        )
    make_reference = getattr(adapter, "make_sequential_reference", None)
    if make_reference is None or not callable(make_reference):
        raise MissingSpeculativeScoutAdapterError(
            "The GPU scout adapter must provide make_sequential_reference() with private cache and RNG state."
        )
    initial_rng_hash = adapter.generator_state_hash()
    sequential_adapter = make_reference()
    if adapter.generator_state_hash() != initial_rng_hash:
        raise RuntimeError("Constructing the sequential reference mutated the candidate target generator.")
    if sequential_adapter.generator_state_hash() != initial_rng_hash:
        raise RuntimeError("Candidate and sequential reference did not start from identical RNG state.")

    speculative = run_exact_speculative_decode(adapter, config.speculation)
    sequential = run_sequential_target_reference(sequential_adapter, config.speculation)
    comparison = compare_exact_transcript(speculative, sequential)
    comparison.assert_exact()
    return comparison
