"""Dependency-light batch-physics state and RNG verifier.

This module deliberately does not run the target model.  It proves the request
isolation rules that a merged-batch or lane-pool GPU scout must preserve before
either execution family is allowed to become runtime code.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, Sequence

from ..exactness import ExactnessEvidence, ExactnessResultClass


def _rng_state_hash(generator: random.Random) -> str:
    return hashlib.sha256(repr(generator.getstate()).encode("utf-8")).hexdigest()


def _sample_token(generator: random.Random, weights: Sequence[float]) -> int:
    total = float(sum(weights))
    draw = generator.random() * total
    cumulative = 0.0
    for token_id, weight in enumerate(weights):
        cumulative += float(weight)
        if draw < cumulative:
            return token_id
    return len(weights) - 1


@dataclass(frozen=True)
class BatchPhysicsRequest:
    """One independently seeded scripted-score request for the CPU gate."""

    request_id: str
    seed: int
    score_steps: tuple[tuple[float, ...], ...]
    max_new_tokens: int
    eos_token_ids: tuple[int, ...] = ()
    arrival_step: int = 0

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer.")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if self.arrival_step < 0:
            raise ValueError("arrival_step must be non-negative.")
        if len(self.score_steps) < self.max_new_tokens:
            raise ValueError("score_steps must cover max_new_tokens.")
        vocabulary_size: int | None = None
        for weights in self.score_steps:
            if not weights:
                raise ValueError("each score step must contain at least one token weight.")
            if vocabulary_size is None:
                vocabulary_size = len(weights)
            elif len(weights) != vocabulary_size:
                raise ValueError("all score steps must use the same vocabulary size.")
            if any(not math.isfinite(weight) or weight < 0 for weight in weights):
                raise ValueError("score weights must be finite and non-negative.")
            if not any(weight > 0 for weight in weights):
                raise ValueError("each score step must contain a positive weight.")
            if not math.isfinite(sum(weights)):
                raise ValueError("the total score weight must be finite.")
        if vocabulary_size is not None and any(
                token_id < 0 or token_id >= vocabulary_size
                for token_id in self.eos_token_ids
        ):
            raise ValueError("eos_token_ids must be inside the score vocabulary.")


@dataclass(frozen=True)
class BatchPhysicsResult:
    """Observable per-request transcript and isolation evidence."""

    request_id: str
    seed: int
    generated_token_ids: tuple[int, ...]
    stop_reason: str
    initial_rng_state_hash: str
    final_rng_state_hash: str
    sample_calls: int
    slot_id: int
    slot_generation: int
    activation_step: int
    finish_step: int

    def exactness_evidence(self) -> ExactnessEvidence:
        """Expose the transcript in the optimized engine's shared evidence type."""

        return ExactnessEvidence(
            result_class=ExactnessResultClass.EXACT_OUTPUT,
            generated_token_ids=self.generated_token_ids,
            stop_reason=self.stop_reason,
            final_rng_state_hash=self.final_rng_state_hash,
        )


@dataclass(frozen=True)
class BatchPhysicsStep:
    """One iteration-level scheduling step."""

    step_index: int
    activated_request_ids: tuple[str, ...]
    decoded_request_ids: tuple[str, ...]
    finished_request_ids: tuple[str, ...]
    active_batch_size: int


@dataclass
class _RequestState:
    request: BatchPhysicsRequest
    generator: random.Random = field(init=False)
    initial_rng_state_hash: str = field(init=False)
    generated_token_ids: list[int] = field(default_factory=list)
    sample_calls: int = 0
    stop_reason: str | None = None
    slot_id: int | None = None
    slot_generation: int | None = None
    activation_step: int | None = None
    finish_step: int | None = None

    def __post_init__(self) -> None:
        self.generator = random.Random(self.request.seed)
        self.initial_rng_state_hash = _rng_state_hash(self.generator)


@dataclass
class _Slot:
    slot_id: int
    generation: int = 0
    request_id: str | None = None

    @property
    def free(self) -> bool:
        return self.request_id is None


class BatchPhysicsScheduler:
    """CPU-only verifier for exact iteration-level batch request semantics.

    Empty slots have no RNG. Finished requests are released before another step
    can sample them, so dummy and stopped rows cannot advance generator state.
    """

    def __init__(self, requests: Iterable[BatchPhysicsRequest], *, max_active_sequences: int):
        if max_active_sequences <= 0:
            raise ValueError("max_active_sequences must be positive.")
        request_list = list(requests)
        request_ids = [request.request_id for request in request_list]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request_id values must be unique.")

        self.max_active_sequences = int(max_active_sequences)
        self._states = {request.request_id: _RequestState(request) for request in request_list}
        self._arrival_order = {
            request.request_id: index
            for index, request in enumerate(request_list)
        }
        self._unarrived = sorted(
            request_list,
            key=lambda request: (request.arrival_step, self._arrival_order[request.request_id]),
        )
        self._pending: Deque[str] = deque()
        self._slots = [_Slot(slot_id=index) for index in range(self.max_active_sequences)]
        self._steps: list[BatchPhysicsStep] = []
        self._active_batch_histogram: Counter[int] = Counter()
        self._step_index = 0

    @property
    def current_step(self) -> int:
        return self._step_index

    @property
    def active_batch_size_histogram(self) -> dict[int, int]:
        return dict(sorted(self._active_batch_histogram.items()))

    @property
    def steps(self) -> tuple[BatchPhysicsStep, ...]:
        return tuple(self._steps)

    def has_work(self) -> bool:
        return bool(
            self._unarrived
            or self._pending
            or any(not slot.free for slot in self._slots)
        )

    def step(self) -> BatchPhysicsStep:
        self._release_arrivals()
        activated = self._activate_open_slots()
        active_slots = sorted(
            (slot for slot in self._slots if not slot.free),
            key=lambda slot: self._arrival_order[slot.request_id],
        )
        decoded: list[str] = []
        finished: list[str] = []
        if active_slots:
            self._active_batch_histogram[len(active_slots)] += 1

        for slot in active_slots:
            request_id = slot.request_id
            if request_id is None:
                raise AssertionError("active slot must own a request.")
            state = self._states[request_id]
            if state.stop_reason is not None:
                raise AssertionError("finished requests must never remain active.")
            token_index = len(state.generated_token_ids)
            token_id = _sample_token(state.generator, state.request.score_steps[token_index])
            state.sample_calls += 1
            state.generated_token_ids.append(token_id)
            decoded.append(request_id)

            if token_id in state.request.eos_token_ids:
                self._finish(slot, state, "eos")
                finished.append(request_id)
            elif len(state.generated_token_ids) >= state.request.max_new_tokens:
                self._finish(slot, state, "max_new_tokens")
                finished.append(request_id)

        # Stable slots may be reused immediately, but replacements start decoding
        # on the next iteration rather than receiving a second token this step.
        activated.extend(self._activate_open_slots())
        metadata = BatchPhysicsStep(
            step_index=self._step_index,
            activated_request_ids=tuple(activated),
            decoded_request_ids=tuple(decoded),
            finished_request_ids=tuple(finished),
            active_batch_size=len(active_slots),
        )
        self._steps.append(metadata)
        self._step_index += 1
        return metadata

    def run_until_idle(self, *, max_steps: int = 10_000) -> tuple[BatchPhysicsResult, ...]:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        for _ in range(max_steps):
            if not self.has_work():
                break
            self.step()
        if self.has_work():
            raise RuntimeError(f"batch-physics scheduler still has work after {max_steps} steps.")
        return self.results()

    def results(self) -> tuple[BatchPhysicsResult, ...]:
        unfinished = [state.request.request_id for state in self._states.values() if state.stop_reason is None]
        if unfinished:
            raise RuntimeError(f"requests have not finished: {unfinished}.")
        return tuple(
            self._result(self._states[request_id])
            for request_id in sorted(self._states)
        )

    def rng_state_hash(self, request_id: str) -> str:
        try:
            return _rng_state_hash(self._states[request_id].generator)
        except KeyError as exc:
            raise KeyError(f"unknown request_id: {request_id}") from exc

    def sample_calls(self, request_id: str) -> int:
        try:
            return self._states[request_id].sample_calls
        except KeyError as exc:
            raise KeyError(f"unknown request_id: {request_id}") from exc

    def _release_arrivals(self) -> None:
        while self._unarrived and self._unarrived[0].arrival_step <= self._step_index:
            self._pending.append(self._unarrived.pop(0).request_id)

    def _activate_open_slots(self) -> list[str]:
        activated: list[str] = []
        for slot in self._slots:
            if not self._pending:
                break
            if not slot.free:
                continue
            request_id = self._pending.popleft()
            state = self._states[request_id]
            slot.generation += 1
            slot.request_id = request_id
            state.slot_id = slot.slot_id
            state.slot_generation = slot.generation
            state.activation_step = self._step_index
            activated.append(request_id)
        return activated

    def _finish(self, slot: _Slot, state: _RequestState, stop_reason: str) -> None:
        state.stop_reason = stop_reason
        state.finish_step = self._step_index
        slot.request_id = None

    @staticmethod
    def _result(state: _RequestState) -> BatchPhysicsResult:
        if (
                state.stop_reason is None
                or state.slot_id is None
                or state.slot_generation is None
                or state.activation_step is None
                or state.finish_step is None
        ):
            raise RuntimeError(f"request {state.request.request_id} lacks completed state.")
        return BatchPhysicsResult(
            request_id=state.request.request_id,
            seed=state.request.seed,
            generated_token_ids=tuple(state.generated_token_ids),
            stop_reason=state.stop_reason,
            initial_rng_state_hash=state.initial_rng_state_hash,
            final_rng_state_hash=_rng_state_hash(state.generator),
            sample_calls=state.sample_calls,
            slot_id=state.slot_id,
            slot_generation=state.slot_generation,
            activation_step=state.activation_step,
            finish_step=state.finish_step,
        )
