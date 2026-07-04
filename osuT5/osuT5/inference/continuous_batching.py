from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque


@dataclass(frozen=True)
class ContinuousBatchSchedulerConfig:
    max_active_sequences: int
    max_wait_ms: int = 0
    prefill_policy: str = "serial"
    decode_order_policy: str = "arrival_order"
    rng_policy: str = "serial_global"


@dataclass
class ContinuousBatchRequest:
    request_id: str
    compatibility_key: tuple[Any, ...]
    prompt_tokens: int
    max_new_tokens: int
    eos_token_ids: tuple[int, ...] = ()
    script_tokens: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    planned_arrival_step: int | None = None
    initial_rng_state_hash: str | None = None
    final_rng_state_hash: str | None = None
    logits_processor_state_hash: str | None = None
    cache_state_hash: str | None = None
    enqueue_order: int | None = None
    enqueue_step: int | None = None
    activation_step: int | None = None
    finish_step: int | None = None
    generated_tokens: list[int] = field(default_factory=list)
    stop_reason: str | None = None
    slot_id: int | None = None
    slot_generation: int | None = None

    @property
    def stopped(self) -> bool:
        return self.stop_reason is not None


@dataclass
class ActiveSequenceSlot:
    slot_id: int
    slot_generation: int = 0
    request_id: str | None = None
    generated_tokens: list[int] = field(default_factory=list)
    stopped: bool = False
    stop_reason: str | None = None

    @property
    def is_free(self) -> bool:
        return self.request_id is None


@dataclass(frozen=True)
class ContinuousBatchStepMetadata:
    step_index: int
    activated: list[dict[str, Any]]
    decoded: list[dict[str, Any]]
    finished: list[dict[str, Any]]
    active_batch_size: int

    @property
    def is_noop(self) -> bool:
        return self.active_batch_size == 0 and not self.activated and not self.finished


@dataclass(frozen=True)
class ContinuousBatchReport:
    config: dict[str, Any]
    compatibility_key: tuple[Any, ...] | None
    active_batch_size_histogram: dict[str, int]
    steps: list[dict[str, Any]]
    requests: list[dict[str, Any]]
    cache_slot_events: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": dict(self.config),
            "compatibility_key": self.compatibility_key,
            "active_batch_size_histogram": dict(self.active_batch_size_histogram),
            "steps": [dict(step) for step in self.steps],
            "requests": [dict(request) for request in self.requests],
            "cache_slot_events": [dict(event) for event in self.cache_slot_events],
        }


class ContinuousBatchScheduler:
    """Deterministic CPU-only harness for future continuous batching.

    The harness only models request lifecycle, active slots, scripted tokens,
    stop reasons, and metadata. It intentionally does not run a model, sample,
    consume RNG, mutate logits processors, write KV cache tensors, or touch the
    static IPC server path.
    """

    def __init__(self, config: ContinuousBatchSchedulerConfig):
        if config.max_active_sequences <= 0:
            raise ValueError("max_active_sequences must be positive.")
        if config.max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative.")
        if config.prefill_policy not in {"serial", "batch_prefill"}:
            raise ValueError("prefill_policy must be one of: serial, batch_prefill.")
        if config.decode_order_policy not in {"arrival_order", "round_robin"}:
            raise ValueError("decode_order_policy must be one of: arrival_order, round_robin.")
        if config.rng_policy not in {"serial_global", "per_request_generator", "documented_drift"}:
            raise ValueError("rng_policy must be one of: serial_global, per_request_generator, documented_drift.")

        self.config = config
        self._pending: Deque[ContinuousBatchRequest] = deque()
        self._slots = [
            ActiveSequenceSlot(slot_id=slot_id)
            for slot_id in range(config.max_active_sequences)
        ]
        self._requests: dict[str, ContinuousBatchRequest] = {}
        self._finished_order: list[str] = []
        self._compatibility_key: tuple[Any, ...] | None = None
        self._next_enqueue_order = 0
        self._step_index = 0
        self._round_robin_cursor = 0
        self._active_batch_size_histogram: Counter[int] = Counter()
        self._step_metadata: list[ContinuousBatchStepMetadata] = []
        self.cache_slot_events: list[dict[str, Any]] = []

    def enqueue(self, request: ContinuousBatchRequest) -> None:
        if request.request_id in self._requests:
            raise ValueError(f"Duplicate request_id: {request.request_id}.")
        if request.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative.")
        if request.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if request.planned_arrival_step is not None and request.planned_arrival_step < 0:
            raise ValueError("planned_arrival_step must be non-negative.")
        if request.planned_arrival_step is not None and request.planned_arrival_step > self._step_index:
            raise ValueError(
                "ContinuousBatchScheduler.enqueue() received a request before its planned arrival step. "
                "Delay enqueue until planned_arrival_step or leave planned_arrival_step unset."
            )
        if self._compatibility_key is None:
            self._compatibility_key = request.compatibility_key
        elif request.compatibility_key != self._compatibility_key:
            raise ValueError("ContinuousBatchScheduler accepts one compatibility_key group at a time.")

        request.enqueue_order = self._next_enqueue_order
        request.enqueue_step = self._step_index
        self._next_enqueue_order += 1
        self._requests[request.request_id] = request
        self._pending.append(request)

    def step(self) -> ContinuousBatchStepMetadata:
        activated = self._activate_open_slots()
        active_slots = self._decode_ordered_active_slots()
        if not active_slots and not activated:
            metadata = ContinuousBatchStepMetadata(
                step_index=self._step_index,
                activated=[],
                decoded=[],
                finished=[],
                active_batch_size=0,
            )
            self._step_metadata.append(metadata)
            self._step_index += 1
            return metadata

        active_batch_size = len(active_slots)
        decoded: list[dict[str, Any]] = []
        finished: list[dict[str, Any]] = []
        if active_batch_size:
            self._active_batch_size_histogram[active_batch_size] += 1

        for slot in active_slots:
            request = self._requests[slot.request_id]
            token_index = len(request.generated_tokens)
            token_id: int | None = None
            stop_reason: str | None = None
            if token_index < len(request.script_tokens):
                token_id = int(request.script_tokens[token_index])
                request.generated_tokens.append(token_id)
                slot.generated_tokens.append(token_id)
                if token_id in request.eos_token_ids:
                    stop_reason = "eos"
                elif len(request.generated_tokens) >= request.max_new_tokens:
                    stop_reason = "max_new_tokens"
                elif len(request.generated_tokens) >= len(request.script_tokens):
                    stop_reason = "script_exhausted"
            else:
                stop_reason = "script_exhausted"

            decoded.append({
                "request_id": request.request_id,
                "slot_id": slot.slot_id,
                "slot_generation": slot.slot_generation,
                "token_id": token_id,
                "stop_reason": stop_reason,
            })
            if stop_reason is not None:
                finished.append(self._finish_slot(slot, request, stop_reason))

        activated.extend(self._activate_open_slots())

        metadata = ContinuousBatchStepMetadata(
            step_index=self._step_index,
            activated=activated,
            decoded=decoded,
            finished=finished,
            active_batch_size=active_batch_size,
        )
        self._step_metadata.append(metadata)
        self._step_index += 1
        return metadata

    def run_until_idle(self, *, max_steps: int = 1000) -> ContinuousBatchReport:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        for _ in range(max_steps):
            if not self.has_work():
                break
            self.step()
        if self.has_work():
            raise RuntimeError(f"ContinuousBatchScheduler still has work after {max_steps} steps.")
        return self.report()

    def has_work(self) -> bool:
        return bool(self._pending or any(not slot.is_free for slot in self._slots))

    @property
    def current_step(self) -> int:
        return self._step_index

    def active_slots(self) -> tuple[ActiveSequenceSlot, ...]:
        return tuple(slot for slot in self._slots if not slot.is_free)

    def report(self) -> ContinuousBatchReport:
        return ContinuousBatchReport(
            config={
                "max_active_sequences": self.config.max_active_sequences,
                "max_wait_ms": self.config.max_wait_ms,
                "prefill_policy": self.config.prefill_policy,
                "decode_order_policy": self.config.decode_order_policy,
                "rng_policy": self.config.rng_policy,
            },
            compatibility_key=self._compatibility_key,
            active_batch_size_histogram={
                str(size): count for size, count in sorted(self._active_batch_size_histogram.items())
            },
            steps=[
                {
                    "step_index": step.step_index,
                    "activated": list(step.activated),
                    "decoded": list(step.decoded),
                    "finished": list(step.finished),
                    "active_batch_size": step.active_batch_size,
                }
                for step in self._step_metadata
            ],
            requests=[
                self._request_report(self._requests[request_id])
                for request_id in sorted(self._requests, key=lambda key: self._requests[key].enqueue_order)
            ],
            cache_slot_events=list(self.cache_slot_events),
        )

    def _activate_open_slots(self) -> list[dict[str, Any]]:
        activated: list[dict[str, Any]] = []
        for slot in self._slots:
            if not slot.is_free:
                continue
            if not self._pending:
                break
            request = self._pending.popleft()
            slot.slot_generation += 1
            slot.request_id = request.request_id
            slot.generated_tokens = []
            slot.stopped = False
            slot.stop_reason = None
            request.slot_id = slot.slot_id
            request.slot_generation = slot.slot_generation
            request.activation_step = self._step_index
            event = {
                "event": "acquire",
                "step_index": self._step_index,
                "request_id": request.request_id,
                "cache_slot_id": slot.slot_id,
                "slot_generation": slot.slot_generation,
            }
            self.cache_slot_events.append(event)
            activated.append({
                "request_id": request.request_id,
                "cache_slot_id": slot.slot_id,
                "slot_generation": slot.slot_generation,
                "activation_step": request.activation_step,
                "queue_wait_steps": self._queue_wait_steps(request),
            })
        return activated

    def _decode_ordered_active_slots(self) -> list[ActiveSequenceSlot]:
        active = [slot for slot in self._slots if not slot.is_free]
        if self.config.decode_order_policy == "round_robin" and active:
            active.sort(key=lambda slot: slot.slot_id)
            offset = self._round_robin_cursor % len(active)
            active = active[offset:] + active[:offset]
            self._round_robin_cursor += 1
        else:
            active.sort(key=lambda slot: self._requests[slot.request_id].enqueue_order)
        return active

    def _finish_slot(
            self,
            slot: ActiveSequenceSlot,
            request: ContinuousBatchRequest,
            stop_reason: str,
    ) -> dict[str, Any]:
        request.stop_reason = stop_reason
        request.finish_step = self._step_index
        request.slot_id = slot.slot_id
        request.slot_generation = slot.slot_generation
        slot.stopped = True
        slot.stop_reason = stop_reason
        self._finished_order.append(request.request_id)

        event = {
            "event": "release",
            "step_index": self._step_index,
            "request_id": request.request_id,
            "cache_slot_id": slot.slot_id,
            "slot_generation": slot.slot_generation,
            "stop_reason": stop_reason,
        }
        self.cache_slot_events.append(event)
        finished = {
            "request_id": request.request_id,
            "cache_slot_id": slot.slot_id,
            "slot_generation": slot.slot_generation,
            "finish_step": request.finish_step,
            "decode_steps": self._decode_steps(request),
            "latency_steps": self._latency_steps(request),
            "stop_reason": stop_reason,
            "generated_tokens": list(request.generated_tokens),
        }
        slot.request_id = None
        slot.generated_tokens = []
        slot.stopped = False
        slot.stop_reason = None
        return finished

    @staticmethod
    def _request_report(request: ContinuousBatchRequest) -> dict[str, Any]:
        return {
            "request_id": request.request_id,
            "compatibility_key": request.compatibility_key,
            "prompt_tokens": request.prompt_tokens,
            "max_new_tokens": request.max_new_tokens,
            "eos_token_ids": request.eos_token_ids,
            "planned_arrival_step": request.planned_arrival_step,
            "generated_tokens": list(request.generated_tokens),
            "generated_token_count": len(request.generated_tokens),
            "stop_reason": request.stop_reason,
            "enqueue_step": request.enqueue_step,
            "activation_step": request.activation_step,
            "finish_step": request.finish_step,
            "queue_wait_steps": ContinuousBatchScheduler._queue_wait_steps(request),
            "decode_steps": ContinuousBatchScheduler._decode_steps(request),
            "latency_steps": ContinuousBatchScheduler._latency_steps(request),
            "cache_slot_id": request.slot_id,
            "slot_generation": request.slot_generation,
            "metadata": dict(request.metadata),
            "initial_rng_state_hash": request.initial_rng_state_hash,
            "final_rng_state_hash": request.final_rng_state_hash,
            "logits_processor_state_hash": request.logits_processor_state_hash,
            "cache_state_hash": request.cache_state_hash,
        }

    @staticmethod
    def _queue_wait_steps(request: ContinuousBatchRequest) -> int | None:
        if request.enqueue_step is None or request.activation_step is None:
            return None
        return request.activation_step - request.enqueue_step

    @staticmethod
    def _decode_steps(request: ContinuousBatchRequest) -> int | None:
        if request.activation_step is None or request.finish_step is None:
            return None
        return request.finish_step - request.activation_step + 1

    @staticmethod
    def _latency_steps(request: ContinuousBatchRequest) -> int | None:
        if request.enqueue_step is None or request.finish_step is None:
            return None
        return request.finish_step - request.enqueue_step + 1
