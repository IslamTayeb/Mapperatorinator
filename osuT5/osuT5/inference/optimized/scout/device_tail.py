"""Opt-in device-resident decode-tail building blocks.

This module deliberately owns no production dispatch.  It provides static
device state and a fixed-block tail callable for component verification before
any full decoder graph is considered for promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn


SUPPORTED_BLOCK_SIZES = (1, 4, 8, 16)


@dataclass(frozen=True, slots=True)
class GraphCompatibleStoppingPolicy:
    """Capture-safe batch-one EOS/max-length stopping.

    This is deliberately narrower than Transformers' general stopping API.
    The accepted Mapperatorinator configuration contains exactly one
    ``MaxLengthCriteria`` and one ``EosTokenCriteria``.  We validate that
    configuration before capture, then use equality/reduction operations that
    CUDA graphs support instead of ``torch.isin``.
    """

    eos_token_ids: torch.Tensor
    max_length: int

    @classmethod
    def from_transformers(
        cls,
        stopping_criteria: Any,
        *,
        state: "DeviceSequenceState",
    ) -> "GraphCompatibleStoppingPolicy":
        # Keep Transformers out of this opt-in module's import path until a
        # caller explicitly asks to adapt its stopping criteria.
        from transformers.generation.stopping_criteria import (
            EosTokenCriteria,
            MaxLengthCriteria,
            StoppingCriteriaList,
        )

        state.validate()
        if not isinstance(stopping_criteria, StoppingCriteriaList):
            raise TypeError(
                "device-tail capture requires a Transformers StoppingCriteriaList"
            )
        max_length_criteria = []
        eos_criteria = []
        unsupported = []
        for criterion in stopping_criteria:
            if type(criterion) is MaxLengthCriteria:
                max_length_criteria.append(criterion)
            elif type(criterion) is EosTokenCriteria:
                eos_criteria.append(criterion)
            else:
                unsupported.append(type(criterion).__name__)
        if unsupported:
            raise ValueError(
                "device-tail capture does not support stopping criteria: "
                + ", ".join(unsupported)
            )
        if len(max_length_criteria) != 1 or len(eos_criteria) != 1:
            raise ValueError(
                "device-tail capture requires exactly one MaxLengthCriteria "
                "and one EosTokenCriteria"
            )

        criterion_max_length = max_length_criteria[0].max_length
        if (
            isinstance(criterion_max_length, bool)
            or not isinstance(criterion_max_length, int)
            or criterion_max_length != state.max_length
        ):
            raise ValueError(
                "MaxLengthCriteria must match the device sequence max_length"
            )
        criterion_eos = eos_criteria[0].eos_token_id
        if not isinstance(criterion_eos, torch.Tensor):
            raise TypeError("EosTokenCriteria must expose tensor token ids")
        if criterion_eos.ndim != 1 or criterion_eos.numel() < 1:
            raise ValueError("EosTokenCriteria token ids must be a non-empty vector")
        criterion_values = sorted(
            set(int(value) for value in criterion_eos.detach().cpu().tolist())
        )
        state_values = sorted(
            set(int(value) for value in state.eos_token_ids.detach().cpu().tolist())
        )
        if criterion_values != state_values:
            raise ValueError(
                "EosTokenCriteria token ids must match the device sequence state"
            )
        return cls(
            # Reuse the state's fixed-address device storage.  The adapter is
            # built before capture, so no device transfer occurs in the graph.
            eos_token_ids=state.eos_token_ids,
            max_length=state.max_length,
        )

    def stopped(
        self,
        active_tokens: torch.Tensor,
        next_length: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-row stop state using graph-compatible tensor operations."""

        if (
            active_tokens.shape != (1,)
            or active_tokens.dtype != torch.long
            or active_tokens.device != self.eos_token_ids.device
        ):
            raise ValueError("active_tokens must be one device torch.long token")
        if (
            next_length.shape != (1,)
            or next_length.dtype != torch.long
            or next_length.device != self.eos_token_ids.device
        ):
            raise ValueError("next_length must be one device torch.long value")
        eos = (active_tokens[:, None] == self.eos_token_ids[None, :]).any(dim=1)
        return eos | (next_length >= self.max_length)


def _require_batch1_long(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype != torch.long:
        raise TypeError(f"{name} must use torch.long storage")
    if value.ndim != 2 or value.shape[0] != 1:
        raise ValueError(f"{name} must have shape [1, sequence]")
    return value


def _small_long_vector(
    name: str,
    values: torch.Tensor | Iterable[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    tensor = (
        values.to(device=device, dtype=torch.long)
        if isinstance(values, torch.Tensor)
        else torch.tensor(tuple(int(value) for value in values), device=device)
    )
    if tensor.ndim != 1 or tensor.numel() < 1:
        raise ValueError(f"{name} must be a non-empty vector")
    return tensor.contiguous()


@dataclass(slots=True)
class DeviceSequenceState:
    """Fixed-address batch-one sequence and stopping state.

    ``physical_length`` continues to advance during a fixed graph block even
    after EOS.  ``logical_length`` records the first stop and is read by the
    host once after the block.  Finished rows write padding only.
    """

    sequence: torch.Tensor
    physical_length: torch.Tensor
    logical_length: torch.Tensor
    unfinished: torch.Tensor
    pad_token: torch.Tensor
    eos_token_ids: torch.Tensor
    max_length: int

    @classmethod
    def allocate(
        cls,
        input_ids: torch.Tensor,
        *,
        max_length: int,
        pad_token_id: int,
        eos_token_ids: torch.Tensor | Iterable[int],
    ) -> "DeviceSequenceState":
        input_ids = _require_batch1_long("input_ids", input_ids)
        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise TypeError("max_length must be an integer")
        if max_length <= input_ids.shape[1]:
            raise ValueError("max_length must exceed the prompt length")
        sequence = torch.full(
            (1, max_length),
            int(pad_token_id),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        sequence[:, : input_ids.shape[1]].copy_(input_ids)
        length = torch.full(
            (1,),
            int(input_ids.shape[1]),
            dtype=torch.long,
            device=input_ids.device,
        )
        return cls(
            sequence=sequence,
            physical_length=length,
            logical_length=torch.full_like(length, max_length),
            unfinished=torch.ones(
                (1,), dtype=torch.bool, device=input_ids.device
            ),
            pad_token=torch.full_like(length, int(pad_token_id)),
            eos_token_ids=_small_long_vector(
                "eos_token_ids",
                eos_token_ids,
                device=input_ids.device,
            ),
            max_length=max_length,
        )

    def validate(self) -> None:
        _require_batch1_long("sequence", self.sequence)
        device = self.sequence.device
        for name, tensor, dtype in (
            ("physical_length", self.physical_length, torch.long),
            ("logical_length", self.logical_length, torch.long),
            ("unfinished", self.unfinished, torch.bool),
            ("pad_token", self.pad_token, torch.long),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if tensor.shape != (1,) or tensor.dtype != dtype:
                raise ValueError(f"{name} must have shape [1] and dtype {dtype}")
            if tensor.device != device:
                raise ValueError(f"{name} must share the sequence device")
        if (
            self.eos_token_ids.ndim != 1
            or self.eos_token_ids.numel() < 1
            or self.eos_token_ids.dtype != torch.long
            or self.eos_token_ids.device != device
        ):
            raise ValueError("eos_token_ids must be a non-empty device long vector")
        if self.sequence.shape[1] != self.max_length:
            raise ValueError("sequence width must equal max_length")

    def append(
        self,
        next_tokens: torch.Tensor,
        *,
        stopping_policy: GraphCompatibleStoppingPolicy | None = None,
    ) -> torch.Tensor:
        """Append one sampled token without a host read or dynamic allocation."""

        self.validate()
        if not isinstance(next_tokens, torch.Tensor):
            raise TypeError("next_tokens must be a tensor")
        if (
            next_tokens.shape != (1,)
            or next_tokens.dtype != torch.long
            or next_tokens.device != self.sequence.device
        ):
            raise ValueError(
                "next_tokens must have shape [1], torch.long dtype, and the state device"
            )
        active_tokens = torch.where(
            self.unfinished,
            next_tokens,
            self.pad_token,
        )
        self.sequence.scatter_(
            1,
            self.physical_length.view(1, 1),
            active_tokens.view(1, 1),
        )
        next_length = self.physical_length + 1
        if stopping_policy is None:
            stopped = (
                active_tokens.unsqueeze(1) == self.eos_token_ids.unsqueeze(0)
            ).any(dim=1) | (next_length >= self.max_length)
        else:
            if (
                stopping_policy.eos_token_ids is not self.eos_token_ids
                or stopping_policy.max_length != self.max_length
            ):
                raise ValueError("stopping policy must be compiled for this state")
            stopped = stopping_policy.stopped(active_tokens, next_length)
        just_stopped = self.unfinished & stopped
        self.logical_length.copy_(
            torch.where(just_stopped, next_length, self.logical_length)
        )
        self.unfinished.bitwise_and_(~just_stopped)
        self.physical_length.copy_(next_length)
        return active_tokens


def update_one_token_model_inputs(
    static_inputs: dict[str, Any],
    next_tokens: torch.Tensor,
) -> None:
    """Feed a sampled token and device positions into the next graph step."""

    if next_tokens.shape != (1,) or next_tokens.dtype != torch.long:
        raise ValueError("next_tokens must be one torch.long token")
    decoder_ids = static_inputs.get("decoder_input_ids")
    if decoder_ids is None:
        decoder_ids = static_inputs.get("input_ids")
    if not isinstance(decoder_ids, torch.Tensor):
        raise RuntimeError("static model inputs expose no token-id tensor")
    if decoder_ids.shape != (1, 1) or decoder_ids.dtype != torch.long:
        raise ValueError("static decoder token ids must have shape [1, 1]")
    if decoder_ids.device != next_tokens.device:
        raise ValueError("static decoder token ids and next token must share a device")
    decoder_ids.copy_(next_tokens.view(1, 1))
    for name in ("cache_position", "position_ids"):
        value = static_inputs.get(name)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"static model inputs expose no tensor {name}")
        if value.dtype != torch.long or value.device != next_tokens.device:
            raise ValueError(f"static {name} must be a device torch.long tensor")
        value.add_(1)


def fixed_block_tail(
    *,
    state: DeviceSequenceState,
    start_length: int,
    raw_logits_steps: list[torch.Tensor],
    logits_processor,
    stopping_policy: GraphCompatibleStoppingPolicy,
    do_sample: bool,
    static_model_inputs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run an unrolled tail block over fixed raw logits.

    The fixed-logit boundary is intentional: this callable sizes removable
    processor/sampling/stopping overhead before a full model+tail graph exists.
    """

    if not raw_logits_steps:
        raise ValueError("raw_logits_steps must be non-empty")
    if len(raw_logits_steps) not in SUPPORTED_BLOCK_SIZES:
        raise ValueError(
            f"block size must be one of {SUPPORTED_BLOCK_SIZES}, got "
            f"{len(raw_logits_steps)}"
        )
    state.validate()
    if not isinstance(stopping_policy, GraphCompatibleStoppingPolicy):
        raise TypeError("stopping_policy must be graph-compatible")
    if (
        stopping_policy.eos_token_ids is not state.eos_token_ids
        or stopping_policy.max_length != state.max_length
    ):
        raise ValueError("stopping policy must be compiled for this state")
    if isinstance(start_length, bool) or not isinstance(start_length, int):
        raise TypeError("start_length must be an integer")
    if start_length < 0:
        raise ValueError("start_length must be non-negative")
    if start_length + len(raw_logits_steps) > state.max_length:
        raise ValueError("tail block exceeds the preallocated sequence")
    generated = torch.empty(
        (len(raw_logits_steps), 1),
        dtype=torch.long,
        device=state.sequence.device,
    )
    continue_values = torch.empty(
        (len(raw_logits_steps), 1),
        dtype=torch.bool,
        device=state.sequence.device,
    )
    for step, raw_logits in enumerate(raw_logits_steps):
        if (
            not isinstance(raw_logits, torch.Tensor)
            or raw_logits.ndim != 2
            or raw_logits.shape[0] != 1
            or not raw_logits.is_floating_point()
            or raw_logits.device != state.sequence.device
        ):
            raise ValueError("each raw-logit tensor must have shape [1, vocab]")
        active_length = start_length + step
        scores = logits_processor(
            state.sequence[:, :active_length],
            raw_logits.clone(memory_format=torch.contiguous_format),
        )
        if do_sample:
            probabilities = nn.functional.softmax(scores, dim=-1)
            next_tokens = torch.multinomial(
                probabilities,
                num_samples=1,
            ).squeeze(1)
        else:
            next_tokens = torch.argmax(scores, dim=-1)
        active_tokens = state.append(
            next_tokens,
            stopping_policy=stopping_policy,
        )
        if static_model_inputs is not None:
            update_one_token_model_inputs(static_model_inputs, active_tokens)
        generated[step].copy_(active_tokens)
        continue_values[step].copy_(state.unfinished)
    return generated, continue_values


__all__ = [
    "DeviceSequenceState",
    "GraphCompatibleStoppingPolicy",
    "SUPPORTED_BLOCK_SIZES",
    "fixed_block_tail",
    "update_one_token_model_inputs",
]
