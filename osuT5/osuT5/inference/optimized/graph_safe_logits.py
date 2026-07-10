"""Graph-capturable logits processors used only by optimized verifiers."""

from __future__ import annotations

from typing import Any, MutableSequence

import torch

from osuT5.osuT5.inference.logit_processors import (
    MonotonicTimeShiftLogitsProcessor as _V32MonotonicTimeShiftLogitsProcessor,
)


class MonotonicTimeShiftLogitsProcessor(_V32MonotonicTimeShiftLogitsProcessor):
    """Graph-safe full-scan variant with the V32 calculation unchanged."""

    def _full_scan_call(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        is_time_shift = (
            (input_ids >= self.time_shift_start)
            & (input_ids < self.time_shift_end)
        )
        is_sos = self._is_sos(input_ids)
        indices = torch.arange(seq_len, device=device).expand(batch_size, -1)
        last_time_shift_idx = torch.max(
            torch.where(is_time_shift, indices, -1),
            dim=1,
        ).values
        last_sos_idx = torch.max(torch.where(is_sos, indices, -1), dim=1).values
        last_time_shift_values = torch.where(
            last_time_shift_idx != -1,
            input_ids[
                torch.arange(batch_size, device=device),
                last_time_shift_idx,
            ] - self.time_shift_start,
            0,
        )
        apply_mask = (last_time_shift_idx != -1) & (
            last_time_shift_idx > last_sos_idx
        )
        time_shift_vocab = torch.arange(self.time_shift_start, self.time_shift_end, device=device)
        invalid_mask = time_shift_vocab.unsqueeze(0) < (
            self.time_shift_start + last_time_shift_values
        ).unsqueeze(1)
        batch_mask = torch.full_like(scores, False, dtype=torch.bool)
        batch_mask[:, self.time_shift_start:self.time_shift_end] = invalid_mask
        scores.masked_fill_(batch_mask & apply_mask.unsqueeze(1), -torch.inf)
        return scores


def make_graph_safe_logits_processor_list(
    processors: MutableSequence[Any],
) -> MutableSequence[Any]:
    """Replace pristine V32 monotonic processors for optimized graph capture."""

    monotonic_count = 0
    for index, processor in enumerate(processors):
        if type(processor) is MonotonicTimeShiftLogitsProcessor:
            monotonic_count += 1
            continue
        if type(processor) is not _V32MonotonicTimeShiftLogitsProcessor:
            if isinstance(processor, _V32MonotonicTimeShiftLogitsProcessor):
                raise TypeError(
                    "optimized graph-safe replacement does not accept unknown "
                    "monotonic processor subclasses."
                )
            continue
        monotonic_count += 1
        if (
            processor._state_seq_len is not None
            or processor._state_has_time_shift is not None
            or processor._state_last_time_shift_value is not None
            or processor._sos_ids_by_device
            or processor._time_shift_offsets_by_device
        ):
            raise ValueError(
                "optimized graph-safe replacement requires a pristine monotonic processor."
            )
        processors[index] = MonotonicTimeShiftLogitsProcessor(
            processor.tokenizer,
            stateful_batch1=processor.stateful_batch1,
        )
    if monotonic_count != 1:
        raise ValueError(
            "optimized graph-safe processor list requires exactly one monotonic processor."
        )
    return processors
