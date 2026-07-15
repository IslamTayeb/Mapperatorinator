"""Fixed-address token storage for optimized batch-one decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class BatchOneStoppingPolicy:
    eos_token_ids: torch.Tensor
    max_length: int

    @classmethod
    def from_transformers(
        cls,
        stopping_criteria: Any,
        *,
        device: torch.device,
    ) -> "BatchOneStoppingPolicy":
        from transformers.generation.stopping_criteria import (
            EosTokenCriteria,
            MaxLengthCriteria,
            StoppingCriteriaList,
        )

        if not isinstance(stopping_criteria, StoppingCriteriaList):
            raise TypeError(
                "optimized batch-one state requires StoppingCriteriaList"
            )
        eos_criteria = []
        length_criteria = []
        unsupported = []
        for criterion in stopping_criteria:
            if type(criterion) is EosTokenCriteria:
                eos_criteria.append(criterion)
            elif type(criterion) is MaxLengthCriteria:
                length_criteria.append(criterion)
            else:
                unsupported.append(type(criterion).__name__)
        if unsupported:
            raise ValueError(
                "optimized batch-one state does not support stopping criteria: "
                + ", ".join(unsupported)
            )
        if len(eos_criteria) != 1 or len(length_criteria) != 1:
            raise ValueError(
                "optimized batch-one state requires exactly one "
                "EosTokenCriteria and one MaxLengthCriteria"
            )

        max_length = length_criteria[0].max_length
        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise TypeError("MaxLengthCriteria.max_length must be an integer")
        if max_length <= 0:
            raise ValueError("MaxLengthCriteria.max_length must be positive")

        eos_token_ids = eos_criteria[0].eos_token_id
        if not isinstance(eos_token_ids, torch.Tensor):
            raise TypeError("EosTokenCriteria.eos_token_id must be a tensor")
        if eos_token_ids.ndim != 1 or eos_token_ids.numel() == 0:
            raise ValueError("EosTokenCriteria token ids must be a non-empty vector")
        if eos_token_ids.dtype != torch.long:
            raise TypeError("EosTokenCriteria token ids must use torch.long")
        return cls(
            eos_token_ids=eos_token_ids.to(device=device).contiguous(),
            max_length=max_length,
        )

    def stopped(self, next_tokens: torch.Tensor, *, next_length: int) -> bool:
        if (
            not isinstance(next_tokens, torch.Tensor)
            or next_tokens.shape != (1,)
            or next_tokens.dtype != torch.long
        ):
            raise ValueError("next_tokens must have shape [1] and torch.long dtype")
        if next_tokens.device != self.eos_token_ids.device:
            raise ValueError("next_tokens and EOS ids must share a device")
        if isinstance(next_length, bool) or not isinstance(next_length, int):
            raise TypeError("next_length must be an integer")
        if next_length <= 0 or next_length > self.max_length:
            raise ValueError("next_length must be inside the stopping-policy range")
        if next_length >= self.max_length:
            return True
        return bool((next_tokens[:, None] == self.eos_token_ids[None, :]).any())


@dataclass(slots=True)
class BatchOneSequenceState:
    sequence: torch.Tensor
    length: int
    stopping_policy: BatchOneStoppingPolicy

    @classmethod
    def allocate(
        cls,
        input_ids: torch.Tensor,
        *,
        stopping_policy: BatchOneStoppingPolicy,
    ) -> "BatchOneSequenceState":
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a tensor")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("input_ids must have shape [1, sequence]")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long")
        if input_ids.device != stopping_policy.eos_token_ids.device:
            raise ValueError("input_ids and stopping policy must share a device")
        prompt_length = int(input_ids.shape[1])
        if prompt_length <= 0:
            raise ValueError("input_ids must contain at least one prompt token")
        if prompt_length >= stopping_policy.max_length:
            raise ValueError("prompt length must be smaller than max_length")
        sequence = input_ids.new_empty((1, stopping_policy.max_length))
        sequence[:, :prompt_length].copy_(input_ids)
        return cls(
            sequence=sequence,
            length=prompt_length,
            stopping_policy=stopping_policy,
        )

    @property
    def active(self) -> torch.Tensor:
        return self.sequence[:, : self.length]

    def append(self, next_tokens: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if self.length >= self.stopping_policy.max_length:
            raise RuntimeError("cannot append after max_length")
        if (
            not isinstance(next_tokens, torch.Tensor)
            or next_tokens.shape != (1,)
            or next_tokens.dtype != torch.long
            or next_tokens.device != self.sequence.device
        ):
            raise ValueError(
                "next_tokens must have shape [1], torch.long dtype, and the state device"
            )
        next_length = self.length + 1
        self.sequence[:, self.length : next_length].copy_(next_tokens[:, None])
        self.length = next_length
        stopped = self.stopping_policy.stopped(
            next_tokens,
            next_length=next_length,
        )
        return self.active, stopped


def allocate_batch_one_sequence_state(
    input_ids: torch.Tensor,
    stopping_criteria: Any,
) -> BatchOneSequenceState:
    policy = BatchOneStoppingPolicy.from_transformers(
        stopping_criteria,
        device=input_ids.device,
    )
    return BatchOneSequenceState.allocate(
        input_ids,
        stopping_policy=policy,
    )


__all__ = [
    "BatchOneSequenceState",
    "BatchOneStoppingPolicy",
    "allocate_batch_one_sequence_state",
]
