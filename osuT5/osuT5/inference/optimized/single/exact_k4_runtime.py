"""Opt-in strict-FP32 exact K=4 full-model CUDA-graph decode candidate.

The accepted K=1 loop remains the default.  This module is imported only by an
explicit reciprocal runner.  Eligible blocks capture four distinct
``model -> logits processors -> softmax -> torch.multinomial`` steps in one
ordinary PyTorch CUDA graph registered against the default CUDA generator.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Any, Iterator

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from ....runtime_profiling import profile_range
from .decode_loop import (
    _bucketed_prefix_length,
    _clone_static_graph_inputs,
    _copy_static_graph_inputs,
    _max_cache_shape,
    _stable_encoder_outputs,
)
from .runtime_context import active_prefix_self_attention_context


EXACT_K4_BLOCK_SIZE = 4
SUPPORTED_BLOCK_SIZES = (EXACT_K4_BLOCK_SIZE,)
RNG_POLICY = "default_cuda_generator_torch_multinomial_exact_offset_rollback"
_GENERIC_SCORE_PROCESSORS = {
    "TopKLogitsWarper",
    "TopPLogitsWarper",
    "MinPLogitsWarper",
    "TypicalLogitsWarper",
    "EpsilonLogitsWarper",
    "EtaLogitsWarper",
}


def _require_batch1_long(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype != torch.long:
        raise TypeError(f"{name} must use torch.long storage")
    if value.ndim != 2 or value.shape[0] != 1:
        raise ValueError(f"{name} must have shape [1, sequence]")
    return value


@dataclass(slots=True)
class ExactK4DeviceState:
    sequence: torch.Tensor
    physical_length: torch.Tensor
    logical_length: torch.Tensor
    unfinished: torch.Tensor
    pad_token: torch.Tensor
    eos_token_ids: torch.Tensor
    monotonic_has_time_shift: torch.Tensor
    monotonic_last_value: torch.Tensor
    stop_flags: torch.Tensor
    length_snapshots: torch.Tensor
    unfinished_snapshots: torch.Tensor
    max_length: int

    @classmethod
    def allocate(
        cls,
        input_ids: torch.Tensor,
        *,
        max_length: int,
        pad_token_id: int,
        eos_token_ids: torch.Tensor,
    ) -> "ExactK4DeviceState":
        input_ids = _require_batch1_long("input_ids", input_ids)
        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise TypeError("max_length must be an integer")
        if max_length <= input_ids.shape[1]:
            raise ValueError("max_length must exceed the prompt length")
        if (
            not isinstance(eos_token_ids, torch.Tensor)
            or eos_token_ids.ndim != 1
            or eos_token_ids.numel() < 1
        ):
            raise ValueError("eos_token_ids must be a non-empty vector")
        sequence = torch.full(
            (1, max_length),
            int(pad_token_id),
            dtype=torch.long,
            device=input_ids.device,
        )
        sequence[:, : input_ids.shape[1]].copy_(input_ids)
        length = torch.full(
            (1,), input_ids.shape[1], dtype=torch.long, device=input_ids.device
        )
        return cls(
            sequence=sequence,
            physical_length=length,
            logical_length=torch.full_like(length, max_length),
            unfinished=torch.ones((1,), dtype=torch.bool, device=input_ids.device),
            pad_token=torch.full_like(length, int(pad_token_id)),
            eos_token_ids=eos_token_ids.to(
                device=input_ids.device, dtype=torch.long
            ).contiguous(),
            monotonic_has_time_shift=torch.zeros(
                (1,), dtype=torch.bool, device=input_ids.device
            ),
            monotonic_last_value=torch.zeros_like(length),
            stop_flags=torch.zeros(
                (EXACT_K4_BLOCK_SIZE,),
                dtype=torch.bool,
                device=input_ids.device,
            ),
            length_snapshots=torch.zeros(
                (EXACT_K4_BLOCK_SIZE,),
                dtype=torch.long,
                device=input_ids.device,
            ),
            unfinished_snapshots=torch.zeros(
                (EXACT_K4_BLOCK_SIZE,),
                dtype=torch.bool,
                device=input_ids.device,
            ),
            max_length=max_length,
        )

    def validate(self) -> None:
        _require_batch1_long("sequence", self.sequence)
        device = self.sequence.device
        for name, value, dtype in (
            ("physical_length", self.physical_length, torch.long),
            ("logical_length", self.logical_length, torch.long),
            ("unfinished", self.unfinished, torch.bool),
            ("pad_token", self.pad_token, torch.long),
            ("monotonic_has_time_shift", self.monotonic_has_time_shift, torch.bool),
            ("monotonic_last_value", self.monotonic_last_value, torch.long),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if value.shape != (1,) or value.dtype != dtype or value.device != device:
                raise ValueError(
                    f"{name} must have shape [1], dtype {dtype}, and sequence device"
                )
        for name, value, dtype in (
            ("stop_flags", self.stop_flags, torch.bool),
            ("length_snapshots", self.length_snapshots, torch.long),
            ("unfinished_snapshots", self.unfinished_snapshots, torch.bool),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != (EXACT_K4_BLOCK_SIZE,)
                or value.dtype != dtype
                or value.device != device
            ):
                raise ValueError(
                    f"{name} must have shape [{EXACT_K4_BLOCK_SIZE}], dtype "
                    f"{dtype}, and sequence device"
                )
        if self.sequence.shape[1] != self.max_length:
            raise ValueError("sequence width must equal max_length")

    def reset(
        self,
        input_ids: torch.Tensor,
        *,
        pad_token_id: int,
        eos_token_ids: torch.Tensor,
    ) -> None:
        input_ids = _require_batch1_long("input_ids", input_ids)
        if input_ids.device != self.sequence.device:
            raise ValueError("ExactK4 state reset requires the original CUDA device")
        if input_ids.shape[1] >= self.max_length:
            raise ValueError("ExactK4 state reset prompt exceeds sequence capacity")
        if eos_token_ids.shape != self.eos_token_ids.shape or not torch.equal(
            eos_token_ids.to(device=self.eos_token_ids.device), self.eos_token_ids
        ):
            raise ValueError("ExactK4 state reset EOS policy changed")
        self.sequence.fill_(int(pad_token_id))
        self.sequence[:, : input_ids.shape[1]].copy_(input_ids)
        self.physical_length.fill_(input_ids.shape[1])
        self.logical_length.fill_(self.max_length)
        self.unfinished.fill_(True)
        self.pad_token.fill_(int(pad_token_id))
        self.monotonic_has_time_shift.zero_()
        self.monotonic_last_value.zero_()
        self.stop_flags.zero_()
        self.length_snapshots.zero_()
        self.unfinished_snapshots.zero_()

    def initialize_monotonic(
        self,
        *,
        time_shift_start: int,
        time_shift_end: int,
        sos_ids: torch.Tensor,
        prompt_length: int,
    ) -> None:
        tokens = self.sequence[0, :prompt_length]
        indices = torch.arange(prompt_length, device=tokens.device)
        is_time = (tokens >= time_shift_start) & (tokens < time_shift_end)
        is_sos = (tokens[:, None] == sos_ids[None, :]).any(dim=1)
        last_time = torch.max(torch.where(is_time, indices, -1), dim=0).values
        last_sos = torch.max(torch.where(is_sos, indices, -1), dim=0).values
        value = torch.where(
            last_time != -1,
            tokens[last_time] - time_shift_start,
            torch.zeros((), dtype=torch.long, device=tokens.device),
        )
        self.monotonic_has_time_shift.copy_(
            ((last_time != -1) & (last_time > last_sos)).reshape(1)
        )
        self.monotonic_last_value.copy_(value.reshape(1))

    def update_monotonic(
        self,
        token: torch.Tensor,
        *,
        time_shift_start: int,
        time_shift_end: int,
        sos_ids: torch.Tensor,
    ) -> None:
        is_time = (token >= time_shift_start) & (token < time_shift_end)
        is_sos = (token[:, None] == sos_ids[None, :]).any(dim=1)
        self.monotonic_has_time_shift.copy_(torch.where(
            is_sos,
            torch.zeros_like(self.monotonic_has_time_shift),
            torch.where(
                is_time,
                torch.ones_like(self.monotonic_has_time_shift),
                self.monotonic_has_time_shift,
            ),
        ))
        self.monotonic_last_value.copy_(torch.where(
            is_time,
            token - time_shift_start,
            self.monotonic_last_value,
        ))

    def append(
        self,
        next_token: torch.Tensor,
        *,
        snapshot_index: int | None = None,
    ) -> torch.Tensor:
        if next_token.shape != (1,) or next_token.dtype != torch.long:
            raise ValueError("next_token must be one torch.long token")
        active = torch.where(self.unfinished, next_token, self.pad_token)
        self.sequence.scatter_(
            1, self.physical_length.view(1, 1), active.view(1, 1)
        )
        next_length = self.physical_length + 1
        eos = (active[:, None] == self.eos_token_ids[None, :]).any(dim=1)
        stopped = eos | (next_length >= self.max_length)
        just_stopped = self.unfinished & stopped
        self.logical_length.copy_(
            torch.where(just_stopped, next_length, self.logical_length)
        )
        self.unfinished.bitwise_and_(~just_stopped)
        self.physical_length.copy_(next_length)
        if snapshot_index is not None:
            if not 0 <= snapshot_index < EXACT_K4_BLOCK_SIZE:
                raise ValueError("snapshot_index is outside the K4 block")
            self.stop_flags[snapshot_index].copy_(just_stopped[0])
            self.length_snapshots[snapshot_index].copy_(self.physical_length[0])
            self.unfinished_snapshots[snapshot_index].copy_(self.unfinished[0])
        return active

    def restore_terminal(self, *, valid_steps: int) -> None:
        if isinstance(valid_steps, bool) or not isinstance(valid_steps, int):
            raise TypeError("valid_steps must be an integer")
        if not 1 <= valid_steps <= EXACT_K4_BLOCK_SIZE:
            raise ValueError("valid_steps is outside the K4 block")
        logical_length = int(self.length_snapshots[valid_steps - 1].item())
        self.sequence[:, logical_length:].fill_(int(self.pad_token.item()))
        self.physical_length.fill_(logical_length)
        self.logical_length.fill_(logical_length)
        self.unfinished.copy_(
            self.unfinished_snapshots[valid_steps - 1].reshape(1)
        )
        self.stop_flags[valid_steps:].zero_()
        self.length_snapshots[valid_steps:].zero_()
        self.unfinished_snapshots[valid_steps:].zero_()


@dataclass(slots=True)
class _LookbackState:
    processor: Any
    last_scores: torch.Tensor
    has_last_scores: torch.Tensor


@dataclass(slots=True)
class ExactK4LogitsProcessor:
    processors: list[Any]
    state: ExactK4DeviceState
    time_shift_start: int
    time_shift_end: int
    sos_ids: torch.Tensor
    offsets: torch.Tensor
    lookbacks: dict[int, _LookbackState] = field(default_factory=dict)
    snapshots: dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def compile(
        cls,
        processors: Any,
        *,
        state: ExactK4DeviceState,
        vocab_size: int,
        prompt_length: int,
    ) -> "ExactK4LogitsProcessor":
        items = list(processors)
        monotonic = [
            item for item in items
            if type(item).__name__ == "MonotonicTimeShiftLogitsProcessor"
        ]
        if len(monotonic) != 1:
            raise ValueError("ExactK4 requires exactly one monotonic logits processor")
        owner = monotonic[0]
        time_start = int(owner.time_shift_start)
        time_end = int(owner.time_shift_end)
        sos_ids = owner._sos_ids(state.sequence.device).to(dtype=torch.long)
        state.initialize_monotonic(
            time_shift_start=time_start,
            time_shift_end=time_end,
            sos_ids=sos_ids,
            prompt_length=prompt_length,
        )
        compiled = cls(
            processors=items,
            state=state,
            time_shift_start=time_start,
            time_shift_end=time_end,
            sos_ids=sos_ids,
            offsets=torch.arange(
                time_end - time_start,
                dtype=torch.long,
                device=state.sequence.device,
            ),
        )
        for index, item in enumerate(items):
            name = type(item).__name__
            if name in {
                "MonotonicTimeShiftLogitsProcessor",
                "TimeshiftBias",
                "TemperatureLogitsWarper",
                "ConditionalTemperatureLogitsWarper",
            } or name in _GENERIC_SCORE_PROCESSORS:
                continue
            if name == "LookbackBiasLogitsWarper":
                compiled.lookbacks[index] = _LookbackState(
                    processor=item,
                    last_scores=torch.zeros(
                        (1, vocab_size),
                        dtype=torch.float32,
                        device=state.sequence.device,
                    ),
                    has_last_scores=torch.zeros(
                        (1,), dtype=torch.bool, device=state.sequence.device
                    ),
                )
                continue
            raise ValueError(f"ExactK4 does not support logits processor {name}")
        compiled.snapshots = {
            name: torch.zeros(
                (EXACT_K4_BLOCK_SIZE, *value.shape),
                dtype=value.dtype,
                device=value.device,
            )
            for name, value in compiled.mutable_tensors().items()
        }
        return compiled

    def reset(self, *, prompt_length: int) -> None:
        self.state.initialize_monotonic(
            time_shift_start=self.time_shift_start,
            time_shift_end=self.time_shift_end,
            sos_ids=self.sos_ids,
            prompt_length=prompt_length,
        )
        for lookback in self.lookbacks.values():
            lookback.last_scores.zero_()
            lookback.has_last_scores.zero_()
        for snapshots in self.snapshots.values():
            snapshots.zero_()

    def mutable_tensors(self) -> dict[str, torch.Tensor]:
        values = {
            "monotonic_has_time_shift": self.state.monotonic_has_time_shift,
            "monotonic_last_value": self.state.monotonic_last_value,
        }
        for index, lookback in self.lookbacks.items():
            values[f"lookback_{index}_scores"] = lookback.last_scores
            values[f"lookback_{index}_has"] = lookback.has_last_scores
        return values

    def snapshot(self, step: int) -> None:
        if not 0 <= step < EXACT_K4_BLOCK_SIZE:
            raise ValueError("processor snapshot step is outside the K4 block")
        current = self.mutable_tensors()
        if set(current) != set(self.snapshots):
            raise RuntimeError("processor mutable-state schema changed after capture")
        for name, value in current.items():
            self.snapshots[name][step].copy_(value)

    def restore_terminal(self, *, valid_steps: int) -> None:
        if not 1 <= valid_steps <= EXACT_K4_BLOCK_SIZE:
            raise ValueError("valid_steps is outside the K4 block")
        for name, value in self.mutable_tensors().items():
            value.copy_(self.snapshots[name][valid_steps - 1])
            self.snapshots[name][valid_steps:].zero_()

    def _conditional_temperature(self, item: Any, scores: torch.Tensor) -> torch.Tensor:
        temperature = torch.full(
            (1,), float(item.temperature), dtype=scores.dtype, device=scores.device
        )
        assigned = torch.zeros((1,), dtype=torch.bool, device=scores.device)
        for index, (value, tokens, offset) in enumerate(item.conditionals):
            token_values = torch.as_tensor(
                tokens, dtype=torch.long, device=scores.device
            )
            position = (self.state.physical_length - int(offset)).clamp(min=0)
            prior = self.state.sequence.gather(1, position.view(1, 1)).reshape(1)
            matches = (prior[:, None] == token_values[None, :]).any(dim=1)
            matches &= self.state.physical_length >= int(offset)
            matches &= ~assigned
            temperature = torch.where(
                matches,
                torch.full_like(temperature, float(value)),
                temperature,
            )
            assigned |= matches
        return scores / temperature.unsqueeze(1)

    def _lookback(self, slot: _LookbackState, scores: torch.Tensor) -> torch.Tensor:
        item = slot.processor
        if not bool(item.types_first):
            result = scores.clone()
            result[:, item.lookback_range] = -torch.inf
            return result
        last_position = (self.state.physical_length - 1).clamp(min=0)
        last_token = self.state.sequence.gather(
            1, last_position.view(1, 1)
        ).reshape(1)
        last_timed = (last_token[:, None] == item.timed_tokens[None, :]).any(dim=1)
        last_probs = nn.functional.softmax(slot.last_scores, dim=-1)
        probs = nn.functional.softmax(scores, dim=-1)
        prob_eos = last_probs[:, item.eos_ids].sum(dim=-1)
        prob_event = 1 - prob_eos
        scale = 1 / (
            probs[:, item.other_range].sum(dim=-1) * prob_event + prob_eos
        )
        adjusted = probs.clone()
        adjusted[:, item.lookback_range] = 0
        adjusted[:, item.other_range] *= scale.unsqueeze(1)
        eos_extra = torch.clip((scale - 1) * prob_eos / prob_event, 0, 1)
        adjusted[:, item.lookback_start] = eos_extra
        apply = last_timed & slot.has_last_scores
        result = torch.where(apply.unsqueeze(1), torch.log(adjusted), scores)
        slot.last_scores.copy_(scores)
        slot.has_last_scores.fill_(True)
        return result

    def __call__(self, raw_logits: torch.Tensor) -> torch.Tensor:
        scores = raw_logits.to(
            copy=True, dtype=torch.float32, device=self.state.sequence.device
        )
        for index, item in enumerate(self.processors):
            name = type(item).__name__
            if name == "MonotonicTimeShiftLogitsProcessor":
                invalid = self.offsets < self.state.monotonic_last_value.reshape(1)
                invalid &= self.state.monotonic_has_time_shift.reshape(1)
                scores[:, self.time_shift_start:self.time_shift_end].masked_fill_(
                    invalid.unsqueeze(0), -torch.inf
                )
            elif name == "TimeshiftBias":
                scores = scores.clone()
                scores[:, item.time_range] += float(item.timeshift_bias)
            elif name == "TemperatureLogitsWarper":
                scores = scores / float(item.temperature)
            elif name == "ConditionalTemperatureLogitsWarper":
                scores = self._conditional_temperature(item, scores)
            elif name == "LookbackBiasLogitsWarper":
                scores = self._lookback(self.lookbacks[index], scores)
            elif name in _GENERIC_SCORE_PROCESSORS:
                scores = item(self.state.sequence, scores)
            else:
                raise RuntimeError(f"uncompiled ExactK4 logits processor {name}")
        return scores

    def observe(self, token: torch.Tensor) -> None:
        self.state.update_monotonic(
            token,
            time_shift_start=self.time_shift_start,
            time_shift_end=self.time_shift_end,
            sos_ids=self.sos_ids,
        )


def _sample_exact(scores: torch.Tensor) -> torch.Tensor:
    probabilities = nn.functional.softmax(scores, dim=-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(1)


def _update_static_model_inputs(
    static_inputs: dict[str, Any],
    token: torch.Tensor,
) -> None:
    decoder_ids = static_inputs.get("decoder_input_ids")
    cache_position = static_inputs.get("cache_position")
    position_ids = static_inputs.get("decoder_position_ids")
    if (
        not isinstance(decoder_ids, torch.Tensor)
        or decoder_ids.shape != (1, 1)
        or decoder_ids.dtype != torch.long
    ):
        raise ValueError("ExactK4 static decoder_input_ids must be [1, 1] long")
    if (
        not isinstance(cache_position, torch.Tensor)
        or cache_position.shape != (1,)
        or cache_position.dtype != torch.long
    ):
        raise ValueError("ExactK4 static cache_position must be [1] long")
    if (
        not isinstance(position_ids, torch.Tensor)
        or position_ids.shape != (1, 1)
        or position_ids.dtype != torch.long
    ):
        raise ValueError("ExactK4 static decoder_position_ids must be [1, 1] long")
    decoder_ids.copy_(token.view(1, 1))
    cache_position.add_(1)
    position_ids.add_(1)
    mask = static_inputs.get("decoder_attention_mask")
    if isinstance(mask, torch.Tensor) and mask.ndim == 4:
        mask.scatter_(
            -1,
            cache_position.view(1, 1, 1, 1),
            torch.zeros((1, 1, 1, 1), dtype=mask.dtype, device=mask.device),
        )


def _one_step_tail(
    *,
    logits: torch.Tensor,
    processor: ExactK4LogitsProcessor,
    state: ExactK4DeviceState,
    static_inputs: dict[str, Any],
    do_sample: bool,
    snapshot_index: int | None = None,
) -> torch.Tensor:
    scores = processor(logits[:, -1, :])
    next_token = (
        _sample_exact(scores)
        if do_sample
        else torch.argmax(scores, dim=-1)
    )
    active = state.append(next_token, snapshot_index=snapshot_index)
    processor.observe(active)
    if snapshot_index is not None:
        processor.snapshot(snapshot_index)
    _update_static_model_inputs(static_inputs, active)
    return active


_EXACT_K4_INSTALLED = False


@dataclass(slots=True)
class _ExactK4GraphEntry:
    graph: torch.cuda.CUDAGraph
    static_inputs: dict[str, Any]
    state: ExactK4DeviceState
    processor: ExactK4LogitsProcessor
    generator: torch.Generator
    model_dtype: torch.dtype
    offset_increment: int
    capture_seconds: float
    peak_vram_bytes: int
    block_replays: int = 0


@dataclass(slots=True)
class _ExactK4RuntimeSlot:
    state: ExactK4DeviceState
    processor: ExactK4LogitsProcessor
    signature: tuple[Any, ...]


def _freeze_processor_value(
    value: Any,
    *,
    d2h: dict[str, int] | None = None,
) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, slice):
        return (value.start, value.stop, value.step)
    if isinstance(value, torch.Tensor):
        if value.numel() > 4096:
            return (tuple(value.shape), str(value.dtype), str(value.device))
        if d2h is not None and value.device.type != "cpu":
            d2h["calls"] = d2h.get("calls", 0) + 1
            d2h["bytes"] = d2h.get("bytes", 0) + (
                value.numel() * value.element_size()
            )
        return (
            tuple(value.shape),
            str(value.dtype),
            tuple(value.detach().cpu().reshape(-1).tolist()),
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_processor_value(item, d2h=d2h) for item in value
        )
    if isinstance(value, dict):
        return tuple(sorted(
            (str(key), _freeze_processor_value(item, d2h=d2h))
            for key, item in value.items()
        ))
    return type(value).__name__


def _processor_signature(
    processors: Any,
    *,
    d2h: dict[str, int] | None = None,
) -> tuple[Any, ...]:
    rows = []
    for processor in processors:
        attributes = getattr(processor, "__dict__", {})
        rows.append((
            type(processor).__name__,
            tuple(sorted(
                (name, _freeze_processor_value(value, d2h=d2h))
                for name, value in attributes.items()
                if not name.startswith("_state_")
                and not name.endswith("_by_device")
                and name not in {"last_scores"}
            )),
        ))
    return tuple(rows)


def _default_cuda_generator(
    explicit_generator: Any,
    device: torch.device,
) -> torch.Generator:
    if explicit_generator is not None:
        raise ValueError(
            "ExactK4 preserves the accepted implicit default CUDA generator; "
            "explicit generators are unsupported"
        )
    if device.type != "cuda":
        raise RuntimeError("ExactK4 requires a CUDA device")
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    generator = torch.cuda.default_generators[index]
    if not hasattr(generator, "get_offset") or not hasattr(generator, "set_offset"):
        raise RuntimeError("ExactK4 requires CUDA generator offset APIs")
    return generator


def _runtime_slot(
    holder: dict[str, Any],
    *,
    input_ids: torch.Tensor,
    max_length: int,
    pad_token_id: int,
    eos_token_ids: torch.Tensor,
    logits_processor: Any,
    processor_signature: tuple[Any, ...],
    vocab_size: int,
    do_sample: bool,
) -> _ExactK4RuntimeSlot:
    signature = (
        max_length,
        tuple(int(value) for value in eos_token_ids.detach().cpu().tolist()),
        processor_signature,
        str(input_ids.device),
        bool(do_sample),
    )
    slots = holder.setdefault("__exact_k4_runtime_slots__", {})
    slot = slots.get(signature)
    if slot is None:
        state = ExactK4DeviceState.allocate(
            input_ids,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_ids=eos_token_ids,
        )
        processor = ExactK4LogitsProcessor.compile(
            logits_processor,
            state=state,
            vocab_size=vocab_size,
            prompt_length=input_ids.shape[1],
        )
        slot = _ExactK4RuntimeSlot(state, processor, signature)
        slots[signature] = slot
    else:
        slot.state.reset(
            input_ids,
            pad_token_id=pad_token_id,
            eos_token_ids=eos_token_ids,
        )
        slot.processor.reset(prompt_length=input_ids.shape[1])
    return slot


def _snapshot_tensors(
    state: ExactK4DeviceState,
    processor: ExactK4LogitsProcessor,
    static_inputs: dict[str, Any],
    *,
    cache_position_start: int,
    model_dtype: torch.dtype,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    tensors: dict[str, torch.Tensor] = {
        name: value
        for name in (
            "sequence",
            "physical_length",
            "logical_length",
            "unfinished",
            "stop_flags",
            "length_snapshots",
            "unfinished_snapshots",
        )
        if isinstance((value := getattr(state, name)), torch.Tensor)
    }
    tensors.update({
        **processor.mutable_tensors(),
        **{
            f"processor_snapshot:{name}": value
            for name, value in processor.snapshots.items()
        },
        **{
            f"input:{name}": value
            for name, value in static_inputs.items()
            if isinstance(value, torch.Tensor)
        },
    })
    cache = static_inputs.get("past_key_values")
    layers = getattr(getattr(cache, "self_attention_cache", None), "layers", None)
    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError("ExactK4 requires initialized self-attention cache layers")
    for layer_index, layer in enumerate(layers):
        for kind in ("keys", "values"):
            value = getattr(layer, kind, None)
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != model_dtype
                or value.device != state.sequence.device
                or value.ndim != 4
                or cache_position_start < 0
                or cache_position_start + EXACT_K4_BLOCK_SIZE > value.shape[-2]
            ):
                raise ValueError(
                    f"ExactK4 cache self[{layer_index}].{kind} cannot own the block"
                )
            tensors[f"cache:{layer_index}:{kind}"] = value[
                ...,
                cache_position_start : cache_position_start + EXACT_K4_BLOCK_SIZE,
                :,
            ]
    return [(value, value.detach().clone()) for value in tensors.values()]


def _restore_snapshot_tensors(
    snapshots: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> None:
    """Order capture-time writes before restoring persistent ExactK4 state."""

    # torch.cuda.graph captures on a side stream and capture_end does not make
    # the caller's stream wait for that work.  Without this synchronization,
    # graph-capture writes to sequence/cache/processor state can race these copies.
    torch.cuda.synchronize(device)
    for tensor, snapshot in snapshots:
        tensor.copy_(snapshot)
    torch.cuda.synchronize(device)


def _capture_exact_k4_entry(
    model,
    model_inputs: dict[str, Any],
    *,
    active_prefix_length: int,
    state: ExactK4DeviceState,
    processor: ExactK4LogitsProcessor,
    generator: torch.Generator,
    cache_position_start: int,
    do_sample: bool,
    block_size: int = EXACT_K4_BLOCK_SIZE,
) -> _ExactK4GraphEntry:
    if not torch.cuda.is_available():
        raise RuntimeError("ExactK4 graph capture requires CUDA")
    started = time.perf_counter()
    device = state.sequence.device
    block_size = _validate_block_size(block_size)
    if not do_sample:
        raise ValueError("ExactK4 currently requires accepted sampling mode")
    with torch.cuda.device(device):
        torch.cuda.reset_peak_memory_stats(device)
        static_inputs = _clone_static_graph_inputs(model_inputs)
        snapshots = _snapshot_tensors(
            state,
            processor,
            static_inputs,
            cache_position_start=cache_position_start,
            model_dtype=model.dtype,
        )
        generator_state = generator.get_state().clone()
        generator_offset = int(generator.get_offset())
        probe_scores = torch.zeros(
            (1, int(model.config.vocab_size)),
            dtype=torch.float32,
            device=device,
        )
        _sample_exact(probe_scores)
        torch.cuda.synchronize(device)
        offset_increment = int(generator.get_offset()) - generator_offset
        generator.set_state(generator_state)
        generator.set_offset(generator_offset)
        if offset_increment != 4:
            raise RuntimeError(
                "accepted FP32 [1, vocab] multinomial must advance the CUDA RNG "
                f"offset by four, got {offset_increment}"
            )

        graph = torch.cuda.CUDAGraph()
        graph.register_generator_state(generator)
        with torch.cuda.graph(graph):
            for step in range(block_size):
                with active_prefix_self_attention_context(active_prefix_length):
                    outputs = model(**static_inputs, return_dict=True)
                _one_step_tail(
                    logits=outputs.logits,
                    processor=processor,
                    state=state,
                    static_inputs=static_inputs,
                    do_sample=True,
                    snapshot_index=step,
                )
        _restore_snapshot_tensors(snapshots, device=device)
        generator.set_state(generator_state)
        generator.set_offset(generator_offset)
        torch.cuda.synchronize(device)
        return _ExactK4GraphEntry(
            graph=graph,
            static_inputs=static_inputs,
            state=state,
            processor=processor,
            generator=generator,
            model_dtype=model.dtype,
            offset_increment=offset_increment,
            capture_seconds=time.perf_counter() - started,
            peak_vram_bytes=torch.cuda.max_memory_allocated(device),
        )


def _validate_block_size(block_size: int) -> int:
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise TypeError("candidate block size must be an integer")
    if block_size not in SUPPORTED_BLOCK_SIZES:
        raise ValueError(
            f"candidate block size must be one of {SUPPORTED_BLOCK_SIZES}, got {block_size}"
        )
    return block_size


def _exact_k4_eligible(
    cur_len: int,
    max_length: int,
    bucket_size: int,
    block_size: int = EXACT_K4_BLOCK_SIZE,
) -> bool:
    block_size = _validate_block_size(block_size)
    if cur_len + block_size > max_length:
        return False
    start = _bucketed_prefix_length(cur_len, bucket_size, max_length)
    end = _bucketed_prefix_length(
        cur_len + block_size - 1, bucket_size, max_length
    )
    return start == end


def _eos_tensor(
    stopping_criteria: Any,
    device: torch.device,
    *,
    max_length: int,
) -> torch.Tensor:
    eos = [
        item.eos_token_id
        for item in stopping_criteria
        if type(item).__name__ == "EosTokenCriteria"
    ]
    maximum = [
        int(item.max_length)
        for item in stopping_criteria
        if type(item).__name__ == "MaxLengthCriteria"
    ]
    unsupported = [
        type(item).__name__
        for item in stopping_criteria
        if type(item).__name__ not in {"EosTokenCriteria", "MaxLengthCriteria"}
    ]
    if unsupported or len(eos) != 1 or len(maximum) != 1:
        raise ValueError(
            "ExactK4 requires exactly EosTokenCriteria and MaxLengthCriteria; "
            f"unsupported={unsupported}"
        )
    if not isinstance(eos[0], torch.Tensor):
        raise TypeError("ExactK4 EOS ids must be a tensor")
    if maximum[0] != max_length:
        raise ValueError("ExactK4 MaxLengthCriteria must match cache capacity")
    return eos[0].to(device=device, dtype=torch.long).reshape(-1).contiguous()


def _sync_model_kwargs_after_exact_k4(
    model_kwargs: dict[str, Any],
    entry: _ExactK4GraphEntry,
    *,
    cur_len: int,
) -> None:
    model_kwargs["cache_position"] = entry.static_inputs["cache_position"]
    if "decoder_attention_mask" in model_kwargs:
        original = model_kwargs["decoder_attention_mask"]
        if (
            not isinstance(original, torch.Tensor)
            or original.ndim != 2
            or tuple(original.shape[:1]) != (1,)
            or original.shape[1] > cur_len
        ):
            raise ValueError("ExactK4 decoder attention mask must be [1, <=cur_len]")
        extended = torch.ones(
            (1, cur_len), dtype=original.dtype, device=original.device
        )
        extended[:, : original.shape[1]].copy_(original)
        model_kwargs["decoder_attention_mask"] = extended


def _terminal_steps(stop_flags: torch.Tensor) -> int | None:
    if (
        not isinstance(stop_flags, torch.Tensor)
        or stop_flags.shape != (EXACT_K4_BLOCK_SIZE,)
        or stop_flags.dtype != torch.bool
    ):
        raise ValueError("ExactK4 stopping flags have an invalid schema")
    indices = torch.nonzero(stop_flags, as_tuple=False).reshape(-1)
    if indices.numel() == 0:
        return None
    return int(indices[0].item()) + 1


def _restore_terminal_block(
    entry: _ExactK4GraphEntry,
    model_inputs: dict[str, Any],
    *,
    valid_steps: int,
    block_cache_start: int,
    block_sequence_start: int,
    generator_start_offset: int,
) -> None:
    """Restore the exact accepted state after a terminal partial block."""

    entry.state.restore_terminal(valid_steps=valid_steps)
    entry.processor.restore_terminal(valid_steps=valid_steps)
    _copy_static_graph_inputs(entry.static_inputs, model_inputs)
    for step in range(valid_steps):
        _update_static_model_inputs(
            entry.static_inputs,
            entry.state.sequence[:, block_sequence_start + step],
        )

    cache = model_inputs.get("past_key_values")
    layers = getattr(getattr(cache, "self_attention_cache", None), "layers", None)
    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError("ExactK4 terminal rollback lost the self-attention cache")
    tail_start = block_cache_start + valid_steps
    tail_end = block_cache_start + EXACT_K4_BLOCK_SIZE
    for layer_index, layer in enumerate(layers):
        for kind in ("keys", "values"):
            value = getattr(layer, kind, None)
            if not isinstance(value, torch.Tensor) or value.dtype != entry.model_dtype:
                raise TypeError(
                    f"ExactK4 terminal cache self[{layer_index}].{kind} has dtype "
                    f"{getattr(value, 'dtype', None)}, expected {entry.model_dtype}"
                )
            value[..., tail_start:tail_end, :].zero_()

    entry.generator.set_offset(
        generator_start_offset + entry.offset_increment * valid_steps
    )


def exact_k4_active_prefix_decode_generate(
    model,
    input_ids: torch.LongTensor,
    logits_processor,
    stopping_criteria,
    generation_config,
    synced_gpus: bool = False,
    streamer=None,
    *,
    active_prefix_bucket_size: int = 64,
    cuda_graph_forward: bool = True,
    cuda_graph_warmup: int = 0,
    cuda_graph_min_decode_steps: int = 1,
    shared_graph_cache: dict[Any, dict[str, Any]] | None = None,
    stable_encoder_holder: dict[str, BaseModelOutput] | None = None,
    **model_kwargs,
) -> torch.LongTensor:
    """Opt-in full-model+tail ExactK4 loop; defaults never select this callable."""

    del cuda_graph_warmup, cuda_graph_min_decode_steps
    if synced_gpus or streamer is not None:
        raise ValueError("ExactK4 does not support synced_gpus or streamers")
    if input_ids.device.type != "cuda" or not cuda_graph_forward:
        raise RuntimeError("ExactK4 requires CUDA graph decode")
    if input_ids.shape[0] != 1:
        raise ValueError("ExactK4 requires batch size one")
    if getattr(model, "dtype", None) != torch.float32 or input_ids.device.type != "cuda":
        raise TypeError("ExactK4 requires strict FP32 model storage on CUDA")
    if generation_config.return_dict_in_generate or any((
        generation_config.output_attentions,
        generation_config.output_hidden_states,
        generation_config.output_scores,
        generation_config.output_logits,
    )):
        raise ValueError("ExactK4 supports tensor outputs only")
    block_size = EXACT_K4_BLOCK_SIZE
    if not _EXACT_K4_INSTALLED:
        raise RuntimeError("ExactK4 decode must run inside install_exact_k4_candidate()")
    pad_token_id = int(generation_config._pad_token_tensor.reshape(-1)[0].item())
    _, cur_len = input_ids.shape
    generator = _default_cuda_generator(
        model_kwargs.pop("generator", None),
        input_ids.device,
    )
    model_kwargs = model._get_initial_cache_position(
        cur_len, input_ids.device, model_kwargs
    )
    max_length = _max_cache_shape(model_kwargs, generation_config)
    eos_ids = _eos_tensor(
        stopping_criteria,
        input_ids.device,
        max_length=max_length,
    )
    if stable_encoder_holder is None:
        raise RuntimeError("ExactK4 requires persistent request-local runtime state")
    graph_cache = shared_graph_cache if shared_graph_cache is not None else {}
    window_identity = int(
        stable_encoder_holder.get("__exact_k4_window_serial__", 0)
    ) + 1
    stable_encoder_holder["__exact_k4_window_serial__"] = window_identity
    processor_d2h: dict[str, int] = {"calls": 0, "bytes": 0}
    processor_signature_started = time.perf_counter()
    processor_signature = _processor_signature(
        logits_processor,
        d2h=processor_d2h,
    )
    processor_signature_seconds = (
        time.perf_counter() - processor_signature_started
    )
    do_sample = bool(generation_config.do_sample)
    if not do_sample:
        raise ValueError("ExactK4 currently supports accepted sampling mode only")
    slot = _runtime_slot(
        stable_encoder_holder,
        input_ids=input_ids,
        max_length=max_length,
        pad_token_id=pad_token_id,
        eos_token_ids=eos_ids,
        logits_processor=logits_processor,
        processor_signature=processor_signature,
        vocab_size=int(model.config.vocab_size),
        do_sample=do_sample,
    )
    state = slot.state
    processor = slot.processor
    prompt_length = int(cur_len)
    stats = {
        "exact_k4_candidate": True,
        "window_serial": window_identity,
        "block_size": block_size,
        "prefill_steps": 0,
        "eligible_steps": 0,
        "block_replays": 0,
        "remainder_steps": 0,
        "status_reads": 0,
        "copy_calls": 0,
        "copy_bytes": 0,
        "capture_seconds": 0.0,
        "peak_vram_bytes": 0,
        "physical_steps": 0,
        "logical_steps": 0,
        "wasted_steps": 0,
        "rng_policy": RNG_POLICY,
        "rng_exact": True,
        "rng_drift": None,
        "rng_request_seed": int(generator.initial_seed()),
        "rng_window_identity": window_identity,
        "rng_early_eos_isolation": True,
        "sampling_mode": "sample",
        "prompt_seed_d2h_copy_calls": 0,
        "prompt_seed_d2h_copy_bytes": 0,
        "prompt_seed_setup_seconds": 0.0,
        "processor_signature_d2h_copy_calls": processor_d2h["calls"],
        "processor_signature_d2h_copy_bytes": processor_d2h["bytes"],
        "processor_signature_setup_seconds": processor_signature_seconds,
        "parent_backend": "ordinary_pytorch_cuda_graph_four_distinct_steps",
        "capture_state_restore_synchronized": True,
        "terminal_rollbacks": 0,
        "generator_offset_corrections": 0,
        "generator_offset_increment": None,
    }
    stable_encoder_holder["__exact_k4_latest_stats__"] = stats

    is_prefill = True
    finished = False
    scores = None
    while not finished:
        if is_prefill or not _exact_k4_eligible(
            cur_len,
            max_length,
            active_prefix_bucket_size,
            block_size,
        ):
            active_ids = state.sequence[:, :cur_len]
            model_inputs = model.prepare_inputs_for_generation(
                active_ids, **model_kwargs
            )
            if is_prefill:
                with profile_range("generation.encoder_prefill"):
                    outputs = model(**model_inputs, return_dict=True)
                is_prefill = False
                stats["prefill_steps"] += 1
            else:
                prefix = _bucketed_prefix_length(
                    cur_len, active_prefix_bucket_size, max_length
                )
                with active_prefix_self_attention_context(prefix):
                    outputs = model(**model_inputs, return_dict=True)
                stats["remainder_steps"] += 1
            model_kwargs = model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=model.config.is_encoder_decoder,
            )
            if stable_encoder_holder is not None:
                encoder_outputs = model_kwargs.get("encoder_outputs")
                if encoder_outputs is not None:
                    model_kwargs["encoder_outputs"] = _stable_encoder_outputs(
                        stable_encoder_holder, encoder_outputs
                    )
            scores_step = processor(outputs.logits[:, -1, :])
            token = _sample_exact(scores_step)
            active = state.append(token)
            processor.observe(active)
            stats["status_reads"] += 1
            unfinished = bool(state.unfinished.item())
            cur_len += 1
            finished = not unfinished
            del outputs
            continue

        prefix = _bucketed_prefix_length(
            cur_len, active_prefix_bucket_size, max_length
        )
        active_ids = state.sequence[:, :cur_len]
        model_inputs = model.prepare_inputs_for_generation(
            active_ids, **model_kwargs
        )
        key = ("exact_k4_candidate", block_size, id(state), id(generator), prefix, tuple(
            (name, tuple(value.shape), str(value.dtype), str(value.device))
            if isinstance(value, torch.Tensor)
            else (name, type(value).__name__, id(value))
            for name, value in sorted(model_inputs.items())
        ))
        raw_entry = graph_cache.get(key)
        if raw_entry is None:
            for value in model_inputs.values():
                if isinstance(value, torch.Tensor):
                    stats["copy_calls"] += 1
                    stats["copy_bytes"] += value.numel() * value.element_size()
            entry = _capture_exact_k4_entry(
                model,
                model_inputs,
                active_prefix_length=prefix,
                state=state,
                processor=processor,
                generator=generator,
                cache_position_start=cur_len - 1,
                do_sample=do_sample,
                block_size=block_size,
            )
            raw_entry = {
                "exact_k4_candidate": True,
                "runtime_entry": entry,
                "active_prefix_length": prefix,
                "capture_seconds": entry.capture_seconds,
                "decode_replays": 0,
                "exact_k4_stats": stats,
            }
            graph_cache[key] = raw_entry
            stats["capture_seconds"] += entry.capture_seconds
            stats["peak_vram_bytes"] = max(
                stats["peak_vram_bytes"], entry.peak_vram_bytes
            )
            stats["generator_offset_increment"] = entry.offset_increment
        else:
            entry = raw_entry["runtime_entry"]
            raw_entry["exact_k4_stats"] = stats
            stats["generator_offset_increment"] = entry.offset_increment
            if (
                entry.state is not state
                or entry.processor is not processor
                or entry.generator is not generator
            ):
                raise RuntimeError("ExactK4 graph resolved the wrong persistent state")
            for name, static_value in entry.static_inputs.items():
                if not isinstance(static_value, torch.Tensor):
                    continue
                current = model_inputs.get(name)
                if not isinstance(current, torch.Tensor):
                    raise RuntimeError(f"ExactK4 model input {name} stopped being a tensor")
                if (
                    current.shape != static_value.shape
                    or current.dtype != static_value.dtype
                    or current.device != static_value.device
                ):
                    raise RuntimeError(f"ExactK4 model input {name} changed signature")
                static_value.copy_(current)
                stats["copy_calls"] += 1
                stats["copy_bytes"] += current.numel() * current.element_size()
        block_sequence_start = cur_len
        block_cache_start = cur_len - 1
        generator_start_offset = int(generator.get_offset())
        stats["eligible_steps"] += block_size
        with profile_range("generation.exact_k4_graph_replay"):
            entry.graph.replay()
        entry.block_replays += 1
        raw_entry["decode_replays"] += block_size
        stats["block_replays"] += 1
        stats["status_reads"] += 1
        status = torch.cat((
            state.physical_length,
            state.logical_length,
            state.unfinished.to(dtype=torch.long),
            state.stop_flags.to(dtype=torch.long),
        )).cpu()
        generator_end_offset = int(generator.get_offset())
        expected_offset = entry.offset_increment * block_size
        if generator_end_offset - generator_start_offset != expected_offset:
            raise RuntimeError(
                "ExactK4 graph RNG offset disagrees with accepted multinomial: "
                f"expected {expected_offset}, got "
                f"{generator_end_offset - generator_start_offset}"
            )
        valid_steps = _terminal_steps(status[3:].to(dtype=torch.bool))
        if valid_steps is None:
            physical_length = int(status[0].item())
            logical_length = int(status[1].item())
            unfinished = bool(status[2].item())
            if not unfinished or physical_length != block_sequence_start + block_size:
                raise RuntimeError("ExactK4 nonterminal block state is inconsistent")
            cur_len = physical_length
        else:
            _restore_terminal_block(
                entry,
                model_inputs,
                valid_steps=valid_steps,
                block_cache_start=block_cache_start,
                block_sequence_start=block_sequence_start,
                generator_start_offset=generator_start_offset,
            )
            stats["terminal_rollbacks"] += 1
            stats["generator_offset_corrections"] += 1
            stats["wasted_steps"] += block_size - valid_steps
            cur_len = block_sequence_start + valid_steps
            unfinished = False
        _sync_model_kwargs_after_exact_k4(model_kwargs, entry, cur_len=cur_len)
        finished = not unfinished

    stats["logical_steps"] = cur_len - prompt_length
    stats["physical_steps"] = stats["logical_steps"] + stats["wasted_steps"]
    if stats["physical_steps"] - stats["logical_steps"] != stats["wasted_steps"]:
        raise RuntimeError("ExactK4 physical/logical/wasted step accounting diverged")
    accounted_steps = (
        stats["prefill_steps"]
        + stats["eligible_steps"]
        + stats["remainder_steps"]
    )
    if stats["physical_steps"] != accounted_steps:
        raise RuntimeError("ExactK4 prefill/eligible/remainder work accounting diverged")
    return state.sequence[:, :cur_len]


@contextmanager
def install_exact_k4_candidate() -> Iterator[None]:
    """Temporarily install a bounded block candidate for an explicit runner."""

    global _EXACT_K4_INSTALLED
    from . import engine

    if _EXACT_K4_INSTALLED:
        raise RuntimeError("ExactK4 candidate context cannot be nested")
    previous = engine.active_prefix_decode_generate
    _EXACT_K4_INSTALLED = True
    engine.active_prefix_decode_generate = exact_k4_active_prefix_decode_generate
    try:
        yield
    finally:
        engine.active_prefix_decode_generate = previous
        _EXACT_K4_INSTALLED = False


__all__ = [
    "ExactK4DeviceState",
    "ExactK4LogitsProcessor",
    "EXACT_K4_BLOCK_SIZE",
    "RNG_POLICY",
    "SUPPORTED_BLOCK_SIZES",
    "install_exact_k4_candidate",
    "exact_k4_active_prefix_decode_generate",
]
