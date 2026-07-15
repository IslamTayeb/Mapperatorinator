"""Stateful logits processing owned by optimized single inference."""

from __future__ import annotations

import torch

from ...logit_processors import (
    ConditionalTemperatureLogitsWarper as _V32ConditionalTemperatureLogitsWarper,
    MonotonicTimeShiftLogitsProcessor as _V32MonotonicTimeShiftLogitsProcessor,
    build_logits_processor_list,
)


class ConditionalTemperatureLogitsWarper(
    _V32ConditionalTemperatureLogitsWarper
):
    """Device-only batch-1 specialization with V32 batched fallback."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.specialized_calls = 0
        self.v32_fallback_calls = 0
        self._condition_tokens_by_device: dict[
            tuple[int, torch.device, torch.dtype], torch.Tensor
        ] = {}
        self._temperatures_by_device: dict[
            tuple[float, torch.device, torch.dtype], torch.Tensor
        ] = {}

    def _temperature_tensor(
        self,
        temperature: float,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        key = (float(temperature), scores.device, scores.dtype)
        value = self._temperatures_by_device.get(key)
        if value is None:
            value = torch.tensor(
                temperature,
                # A zero-dimensional FP32 scalar preserves PyTorch's Python
                # float division semantics for FP16 scores. Storing this as
                # FP16 would round the temperature before division and break
                # same-precision V32 parity.
                dtype=torch.float32,
                device=scores.device,
            )
            self._temperatures_by_device[key] = value
        return value

    def _condition_tokens(
        self,
        index: int,
        tokens: tuple[int, ...],
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        key = (index, input_ids.device, input_ids.dtype)
        value = self._condition_tokens_by_device.get(key)
        if value is None:
            value = torch.tensor(
                tokens,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            self._condition_tokens_by_device[key] = value
        return value

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or scores.ndim != 2:
            raise ValueError(
                "optimized conditional temperature requires rank-2 tokens and scores"
            )
        if not self.conditionals or input_ids.shape[0] != 1:
            self.v32_fallback_calls += 1
            return super().__call__(input_ids, scores)
        if scores.shape[0] != 1:
            raise ValueError(
                "optimized conditional temperature requires matching batch-1 scores"
            )
        if input_ids.dtype != torch.long:
            raise TypeError(
                "optimized conditional temperature requires int64 token storage"
            )
        if scores.dtype not in {torch.float32, torch.float16}:
            raise TypeError(
                "optimized conditional temperature supports FP32 or FP16 scores"
            )
        if input_ids.device != scores.device:
            raise ValueError(
                "optimized conditional temperature requires tokens and scores "
                "on one device"
            )

        self.specialized_calls += 1

        selected = self._temperature_tensor(self.temperature, scores)
        # Reverse traversal preserves the V32 first-match precedence without a
        # data-dependent host branch: earlier conditions overwrite later ones.
        for index in range(len(self.conditionals) - 1, -1, -1):
            temperature, tokens, offset = self.conditionals[index]
            if input_ids.shape[1] < offset:
                continue
            condition_tokens = self._condition_tokens(index, tokens, input_ids)
            matches = torch.eq(
                input_ids[0, -offset],
                condition_tokens,
            ).any()
            selected = torch.where(
                matches,
                self._temperature_tensor(temperature, scores),
                selected,
            )
        return scores / selected

    def profile_stats(self) -> dict[str, int]:
        """Return dispatch evidence without synchronizing device state."""

        return {
            "condition_count": len(self.conditionals),
            "specialized_calls": int(self.specialized_calls),
            "v32_fallback_calls": int(self.v32_fallback_calls),
        }


class MonotonicTimeShiftLogitsProcessor(_V32MonotonicTimeShiftLogitsProcessor):
    """Batch-1 stateful specialization with V32 full-scan fallback."""

    def __init__(self, tokenizer, *, stateful_batch1: bool = True):
        super().__init__(tokenizer)
        self.stateful_batch1 = stateful_batch1
        self._time_shift_offsets_by_device: dict[torch.device, torch.Tensor] = {}
        self._state_seq_len: int | None = None
        self._state_has_time_shift: torch.Tensor | None = None
        self._state_last_time_shift_value: torch.Tensor | None = None

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self.stateful_batch1 and input_ids.shape[0] == 1 and scores.shape[0] == 1:
            return self._stateful_batch1_call(input_ids, scores)
        return super().__call__(input_ids, scores)

    def _stateful_batch1_call(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        seq_len = input_ids.shape[1]
        if self._state_seq_len is None or seq_len != self._state_seq_len + 1:
            self._initialize_batch1_state(input_ids)
        else:
            self._update_batch1_state(input_ids[:, -1])
            self._state_seq_len = seq_len

        if self._state_has_time_shift is None or self._state_last_time_shift_value is None:
            return scores

        invalid_mask = (
            self._time_shift_offsets(input_ids.device)
            < self._state_last_time_shift_value.reshape(1)
        )
        invalid_mask = invalid_mask & self._state_has_time_shift.reshape(1)
        scores[:, self.time_shift_start:self.time_shift_end].masked_fill_(
            invalid_mask.unsqueeze(0),
            -torch.inf,
        )
        return scores

    def _initialize_batch1_state(self, input_ids: torch.LongTensor) -> None:
        tokens = input_ids[0]
        seq_len = tokens.shape[0]
        indices = torch.arange(seq_len, device=input_ids.device)

        is_time_shift = (
            (tokens >= self.time_shift_start) & (tokens < self.time_shift_end)
        )
        is_sos = self._is_sos(tokens)
        last_time_shift_idx = torch.max(
            torch.where(is_time_shift, indices, -1),
            dim=0,
        ).values
        last_sos_idx = torch.max(torch.where(is_sos, indices, -1), dim=0).values
        last_time_shift_value = torch.where(
            last_time_shift_idx != -1,
            tokens[last_time_shift_idx] - self.time_shift_start,
            torch.zeros((), dtype=tokens.dtype, device=tokens.device),
        )

        self._state_seq_len = seq_len
        self._state_has_time_shift = (
            (last_time_shift_idx != -1) & (last_time_shift_idx > last_sos_idx)
        ).reshape(1)
        self._state_last_time_shift_value = last_time_shift_value.reshape(1)

    def _update_batch1_state(self, last_token: torch.LongTensor) -> None:
        is_time_shift = (
            (last_token >= self.time_shift_start)
            & (last_token < self.time_shift_end)
        )
        is_sos = self._is_sos(last_token)
        last_time_shift_value = last_token - self.time_shift_start
        self._state_has_time_shift = torch.where(
            is_sos,
            torch.zeros_like(self._state_has_time_shift, dtype=torch.bool),
            torch.where(
                is_time_shift,
                torch.ones_like(self._state_has_time_shift, dtype=torch.bool),
                self._state_has_time_shift,
            ),
        )
        self._state_last_time_shift_value = torch.where(
            is_time_shift,
            last_time_shift_value,
            self._state_last_time_shift_value,
        )

    def _time_shift_offsets(self, device: torch.device) -> torch.Tensor:
        if device not in self._time_shift_offsets_by_device:
            self._time_shift_offsets_by_device[device] = torch.arange(
                self.time_shift_end - self.time_shift_start,
                device=device,
            )
        return self._time_shift_offsets_by_device[device]


def build_single_logits_processor_list(
    tokenizer,
    *,
    cfg_scale: float = 1.0,
    timeshift_bias: float = 0.0,
    types_first: bool = False,
    temperature: float = 1.0,
    timing_temperature: float | None = None,
    mania_column_temperature: float | None = None,
    taiko_hit_temperature: float | None = None,
    lookback_time: float = 0.0,
    device=None,
    stateful_monotonic: bool = True,
    device_conditional_temperature: bool = True,
):
    monotonic_factory = (
        MonotonicTimeShiftLogitsProcessor
        if stateful_monotonic
        else _V32MonotonicTimeShiftLogitsProcessor
    )
    conditional_temperature_factory = (
        ConditionalTemperatureLogitsWarper
        if device_conditional_temperature
        else _V32ConditionalTemperatureLogitsWarper
    )
    return build_logits_processor_list(
        tokenizer,
        cfg_scale=cfg_scale,
        timeshift_bias=timeshift_bias,
        types_first=types_first,
        temperature=temperature,
        timing_temperature=timing_temperature,
        mania_column_temperature=mania_column_temperature,
        taiko_hit_temperature=taiko_hit_temperature,
        lookback_time=lookback_time,
        device=device,
        monotonic_processor_factory=monotonic_factory,
        conditional_temperature_factory=conditional_temperature_factory,
    )


def conditional_temperature_profile_stats(processors) -> dict[str, int]:
    """Extract one candidate processor's host-side dispatch counters."""

    candidates = [
        processor
        for processor in processors
        if isinstance(processor, ConditionalTemperatureLogitsWarper)
    ]
    if len(candidates) > 1:
        raise RuntimeError(
            "optimized logits chain contains multiple conditional-temperature "
            "processors"
        )
    if not candidates:
        return {
            "condition_count": 0,
            "specialized_calls": 0,
            "v32_fallback_calls": 0,
        }
    return candidates[0].profile_stats()
