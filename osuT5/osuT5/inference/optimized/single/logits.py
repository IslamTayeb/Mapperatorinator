"""Stateful logits processing owned by optimized single inference."""

from __future__ import annotations

import torch

from ...logit_processors import (
    MonotonicTimeShiftLogitsProcessor as _V32MonotonicTimeShiftLogitsProcessor,
    build_logits_processor_list,
)
from ..kernels.whole_token_step_cuda_graph import (
    whole_token_active_length_tensor,
    whole_token_step_cuda_graph_requested,
)


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

    def _full_scan_call(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Capture-safe full scan (gather + where; no data-dependent index)."""

        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        is_time_shift = (
            (input_ids >= self.time_shift_start) & (input_ids < self.time_shift_end)
        )
        is_sos = self._is_sos(input_ids)
        indices = torch.arange(seq_len, device=device).expand(batch_size, -1)
        active_length = whole_token_active_length_tensor()
        if active_length is not None:
            if (
                not isinstance(active_length, torch.Tensor)
                or active_length.shape != (1,)
                or active_length.dtype != torch.long
                or active_length.device != device
            ):
                raise RuntimeError(
                    "whole-token active length must be int64 CUDA tensor shape [1]"
                )
            valid = indices < active_length.reshape(1, 1)
            is_time_shift = is_time_shift & valid
            is_sos = is_sos & valid
        last_time_shift_idx = torch.max(
            torch.where(is_time_shift, indices, -1),
            dim=1,
        ).values
        last_sos_idx = torch.max(torch.where(is_sos, indices, -1), dim=1).values
        safe_idx = last_time_shift_idx.clamp(min=0).unsqueeze(1)
        gathered = input_ids.gather(1, safe_idx).squeeze(1)
        last_time_shift_values = torch.where(
            last_time_shift_idx != -1,
            gathered - self.time_shift_start,
            torch.zeros(batch_size, dtype=input_ids.dtype, device=device),
        )
        apply_mask = (last_time_shift_idx != -1) & (last_time_shift_idx > last_sos_idx)
        time_shift_vocab = torch.arange(
            self.time_shift_start,
            self.time_shift_end,
            device=device,
        )
        invalid_mask = time_shift_vocab.unsqueeze(0) < (
            self.time_shift_start + last_time_shift_values
        ).unsqueeze(1)
        scores_ts = scores[:, self.time_shift_start : self.time_shift_end]
        masked_ts = scores_ts.masked_fill(invalid_mask, -torch.inf)
        scores = scores.clone()
        scores[:, self.time_shift_start : self.time_shift_end] = torch.where(
            apply_mask.unsqueeze(1),
            masked_ts,
            scores_ts,
        )
        return scores

    def _stateful_batch1_call(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # Whole-token CUDA graphs pass a fixed-width pad-filled buffer and must
        # avoid data-dependent indexing / boolean row assigns during capture.
        if whole_token_step_cuda_graph_requested():
            return self._full_scan_call(input_ids, scores)

        seq_len = input_ids.shape[1]
        if self._state_seq_len is not None and seq_len == self._state_seq_len:
            return self._full_scan_call(input_ids, scores)
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
        safe_idx = last_time_shift_idx.clamp(min=0)
        gathered = tokens.gather(0, safe_idx.reshape(1)).squeeze(0)
        last_time_shift_value = torch.where(
            last_time_shift_idx != -1,
            gathered - self.time_shift_start,
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
):
    monotonic_factory = (
        MonotonicTimeShiftLogitsProcessor
        if stateful_monotonic
        else _V32MonotonicTimeShiftLogitsProcessor
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
    )
