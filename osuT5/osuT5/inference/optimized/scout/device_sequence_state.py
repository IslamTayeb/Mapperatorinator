"""Opt-in batch-one sequence storage for exact one-step decoding.

The accepted decoder loop deliberately remains the default.  This module owns
only a narrow experiment which replaces per-token ``torch.cat`` and generic
Transformers stopping dispatch with fixed-address token storage plus the exact
EOS/max-length policy used by optimized single-song generation.

§46 adds an optional pinned-host EOS flag: device-side EOS match is copied to
pinned memory with a recorded event instead of ``unfinished.max() == 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from functools import wraps
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
                "preallocated batch-one state requires StoppingCriteriaList"
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
                "preallocated batch-one state does not support stopping criteria: "
                + ", ".join(unsupported)
            )
        if len(eos_criteria) != 1 or len(length_criteria) != 1:
            raise ValueError(
                "preallocated batch-one state requires exactly one "
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

    def eos_hit_device(self, next_tokens: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(next_tokens, torch.Tensor)
            or next_tokens.shape != (1,)
            or next_tokens.dtype != torch.long
        ):
            raise ValueError("next_tokens must have shape [1] and torch.long dtype")
        if next_tokens.device != self.eos_token_ids.device:
            raise ValueError("next_tokens and EOS ids must share a device")
        return (next_tokens[:, None] == self.eos_token_ids[None, :]).any(dim=-1)

    def stopped(self, next_tokens: torch.Tensor, *, next_length: int) -> bool:
        if isinstance(next_length, bool) or not isinstance(next_length, int):
            raise TypeError("next_length must be an integer")
        if next_length <= 0 or next_length > self.max_length:
            raise ValueError("next_length must be inside the stopping-policy range")
        if next_length >= self.max_length:
            return True
        # Equality against the tiny fixed EOS vector preserves EosTokenCriteria
        # semantics without invoking torch.isin.
        return bool(self.eos_hit_device(next_tokens).item())


@dataclass(slots=True)
class PinnedEosFlag:
    """Device→pinned host EOS/max-length flag with event sync (no .max()==0)."""

    device_flag: torch.Tensor
    host_flag: torch.Tensor
    event: torch.cuda.Event

    @classmethod
    def allocate(cls, device: torch.device) -> "PinnedEosFlag":
        if device.type != "cuda":
            raise RuntimeError("pinned EOS flag requires a CUDA device")
        return cls(
            device_flag=torch.zeros((1,), dtype=torch.bool, device=device),
            host_flag=torch.zeros((1,), dtype=torch.bool, pin_memory=True),
            event=torch.cuda.Event(),
        )

    def publish(self, stopped_device: torch.Tensor) -> None:
        if (
            not isinstance(stopped_device, torch.Tensor)
            or stopped_device.numel() != 1
            or stopped_device.dtype != torch.bool
        ):
            raise ValueError("stopped_device must be a bool tensor with one element")
        self.device_flag.copy_(stopped_device.reshape(1))
        self.host_flag.copy_(self.device_flag, non_blocking=True)
        self.event.record()

    def read(self) -> bool:
        self.event.synchronize()
        return bool(self.host_flag.item())


@dataclass(slots=True)
class BatchOneSequenceState:
    sequence: torch.Tensor
    length: int
    stopping_policy: BatchOneStoppingPolicy
    pinned_eos: PinnedEosFlag | None = None

    @classmethod
    def allocate(
        cls,
        input_ids: torch.Tensor,
        *,
        stopping_policy: BatchOneStoppingPolicy,
        pinned_eos: bool = False,
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
        flag = None
        if pinned_eos:
            flag = PinnedEosFlag.allocate(input_ids.device)
        return cls(
            sequence=sequence,
            length=prompt_length,
            stopping_policy=stopping_policy,
            pinned_eos=flag,
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
        if self.pinned_eos is not None:
            if next_length >= self.stopping_policy.max_length:
                self.pinned_eos.publish(
                    torch.ones((1,), dtype=torch.bool, device=self.sequence.device)
                )
            else:
                self.pinned_eos.publish(
                    self.stopping_policy.eos_hit_device(next_tokens)
                )
            stopped = self.pinned_eos.read()
        else:
            stopped = self.stopping_policy.stopped(
                next_tokens,
                next_length=next_length,
            )
        return self.active, stopped


def allocate_batch_one_sequence_state(
    input_ids: torch.Tensor,
    stopping_criteria: Any,
    *,
    pinned_eos: bool = False,
) -> BatchOneSequenceState:
    policy = BatchOneStoppingPolicy.from_transformers(
        stopping_criteria,
        device=input_ids.device,
    )
    return BatchOneSequenceState.allocate(
        input_ids,
        stopping_policy=policy,
        pinned_eos=pinned_eos,
    )


@dataclass(slots=True)
class DeviceSequenceStateActivation:
    calls: int = 0


@contextmanager
def device_sequence_state_candidate_context():
    """Temporarily route optimized generation through the opt-in state.

    Only the function object imported by the optimized single engine is
    replaced.  Decoder layers, dispatch contexts, accepted presets, and model
    methods remain untouched and the binding is restored on every exit path.
    """

    from ..single import engine

    original = engine.active_prefix_decode_generate
    activation = DeviceSequenceStateActivation()

    @wraps(original)
    def candidate(*args, **kwargs):
        if "preallocated_batch1_state" in kwargs:
            raise RuntimeError(
                "device-state candidate owns preallocated_batch1_state"
            )
        activation.calls += 1
        return original(
            *args,
            preallocated_batch1_state=True,
            **kwargs,
        )

    engine.active_prefix_decode_generate = candidate
    try:
        yield activation
    finally:
        engine.active_prefix_decode_generate = original


@contextmanager
def baseline_glue_v2_candidate_context():
    """§46 W-BASE: device buffer + pinned EOS + persistent sampling-tail graph."""

    from ..kernels.sampling_tail_cuda_graph import (
        sampling_tail_cuda_graph_candidate_context,
    )
    from ..single import engine

    original = engine.active_prefix_decode_generate
    activation = DeviceSequenceStateActivation()

    @wraps(original)
    def candidate(*args, **kwargs):
        owned = {
            "preallocated_batch1_state",
            "pinned_eos_flag",
            "sampling_tail_cuda_graph",
        }
        conflict = owned.intersection(kwargs)
        if conflict:
            raise RuntimeError(
                "baseline-glue-v2 candidate owns "
                + ", ".join(sorted(conflict))
            )
        activation.calls += 1
        return original(
            *args,
            preallocated_batch1_state=True,
            pinned_eos_flag=True,
            sampling_tail_cuda_graph=True,
            **kwargs,
        )

    engine.active_prefix_decode_generate = candidate
    try:
        with sampling_tail_cuda_graph_candidate_context():
            yield activation
    finally:
        engine.active_prefix_decode_generate = original


__all__ = [
    "BatchOneSequenceState",
    "BatchOneStoppingPolicy",
    "DeviceSequenceStateActivation",
    "PinnedEosFlag",
    "allocate_batch_one_sequence_state",
    "baseline_glue_v2_candidate_context",
    "device_sequence_state_candidate_context",
]
