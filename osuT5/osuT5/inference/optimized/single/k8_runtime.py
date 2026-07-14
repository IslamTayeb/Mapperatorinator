"""Opt-in batch-one K=8 full-model CUDA-graph decode candidate."""

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
    _capture_decode_cuda_graph,
    _clone_static_graph_inputs,
    _max_cache_shape,
    _stable_encoder_outputs,
)
from .runtime_context import active_prefix_self_attention_context


K8_BLOCK_SIZE = 8
RNG_POLICY = "counter_prompt_hash_v1"
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


def _prompt_seed(input_ids: torch.Tensor) -> int:
    """Stable per-window seed; the one host copy is charged as setup."""

    values = input_ids.detach().cpu().reshape(-1).tolist()
    state = 0xCBF29CE484222325
    for value in values:
        state ^= int(value) & 0xFFFFFFFFFFFFFFFF
        state = (state * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    state ^= len(values)
    return state & 0x7FFFFFFFFFFFFFFF


@dataclass(slots=True)
class K8DeviceState:
    sequence: torch.Tensor
    physical_length: torch.Tensor
    logical_length: torch.Tensor
    unfinished: torch.Tensor
    pad_token: torch.Tensor
    eos_token_ids: torch.Tensor
    rng_seed: torch.Tensor
    rng_counter: torch.Tensor
    monotonic_has_time_shift: torch.Tensor
    monotonic_last_value: torch.Tensor
    max_length: int

    @classmethod
    def allocate(
        cls,
        input_ids: torch.Tensor,
        *,
        max_length: int,
        pad_token_id: int,
        eos_token_ids: torch.Tensor,
    ) -> "K8DeviceState":
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
            rng_seed=torch.full_like(length, _prompt_seed(input_ids)),
            rng_counter=torch.zeros_like(length),
            monotonic_has_time_shift=torch.zeros(
                (1,), dtype=torch.bool, device=input_ids.device
            ),
            monotonic_last_value=torch.zeros_like(length),
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
            ("rng_seed", self.rng_seed, torch.long),
            ("rng_counter", self.rng_counter, torch.long),
            ("monotonic_has_time_shift", self.monotonic_has_time_shift, torch.bool),
            ("monotonic_last_value", self.monotonic_last_value, torch.long),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if value.shape != (1,) or value.dtype != dtype or value.device != device:
                raise ValueError(
                    f"{name} must have shape [1], dtype {dtype}, and sequence device"
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
            raise ValueError("K8 state reset requires the original CUDA device")
        if input_ids.shape[1] >= self.max_length:
            raise ValueError("K8 state reset prompt exceeds sequence capacity")
        if eos_token_ids.shape != self.eos_token_ids.shape or not torch.equal(
            eos_token_ids.to(device=self.eos_token_ids.device), self.eos_token_ids
        ):
            raise ValueError("K8 state reset EOS policy changed")
        self.sequence.fill_(int(pad_token_id))
        self.sequence[:, : input_ids.shape[1]].copy_(input_ids)
        self.physical_length.fill_(input_ids.shape[1])
        self.logical_length.fill_(self.max_length)
        self.unfinished.fill_(True)
        self.pad_token.fill_(int(pad_token_id))
        self.rng_seed.fill_(_prompt_seed(input_ids))
        self.rng_counter.zero_()
        self.monotonic_has_time_shift.zero_()
        self.monotonic_last_value.zero_()

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

    def append(self, next_token: torch.Tensor) -> torch.Tensor:
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
        return active

    def counter_uniform(self) -> torch.Tensor:
        value = self.rng_seed + self.rng_counter * 2862933555777941757
        value = torch.bitwise_xor(value, torch.bitwise_right_shift(value, 21))
        value = torch.bitwise_xor(value, torch.bitwise_left_shift(value, 17))
        bits = torch.bitwise_and(value, (1 << 53) - 1)
        self.rng_counter.add_(1)
        return (bits.to(torch.float64) + 0.5) / float(1 << 53)


@dataclass(slots=True)
class _LookbackState:
    processor: Any
    last_scores: torch.Tensor
    has_last_scores: torch.Tensor


@dataclass(slots=True)
class K8LogitsProcessor:
    processors: list[Any]
    state: K8DeviceState
    time_shift_start: int
    time_shift_end: int
    sos_ids: torch.Tensor
    offsets: torch.Tensor
    lookbacks: dict[int, _LookbackState] = field(default_factory=dict)

    @classmethod
    def compile(
        cls,
        processors: Any,
        *,
        state: K8DeviceState,
        vocab_size: int,
        prompt_length: int,
    ) -> "K8LogitsProcessor":
        items = list(processors)
        monotonic = [
            item for item in items
            if type(item).__name__ == "MonotonicTimeShiftLogitsProcessor"
        ]
        if len(monotonic) != 1:
            raise ValueError("K8 requires exactly one monotonic logits processor")
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
            raise ValueError(f"K8 does not support logits processor {name}")
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

    def mutable_tensors(self) -> dict[str, torch.Tensor]:
        values = {
            "monotonic_has_time_shift": self.state.monotonic_has_time_shift,
            "monotonic_last_value": self.state.monotonic_last_value,
        }
        for index, lookback in self.lookbacks.items():
            values[f"lookback_{index}_scores"] = lookback.last_scores
            values[f"lookback_{index}_has"] = lookback.has_last_scores
        return values

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
                raise RuntimeError(f"uncompiled K8 logits processor {name}")
        return scores

    def observe(self, token: torch.Tensor) -> None:
        self.state.update_monotonic(
            token,
            time_shift_start=self.time_shift_start,
            time_shift_end=self.time_shift_end,
            sos_ids=self.sos_ids,
        )


def _sample_counter(scores: torch.Tensor, state: K8DeviceState) -> torch.Tensor:
    probabilities = nn.functional.softmax(scores, dim=-1)
    threshold = state.counter_uniform().to(dtype=probabilities.dtype)
    cdf = probabilities.cumsum(dim=-1)
    return torch.searchsorted(cdf[0], threshold).clamp(
        max=probabilities.shape[-1] - 1
    ).to(dtype=torch.long)


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
        raise ValueError("K8 static decoder_input_ids must be [1, 1] long")
    if (
        not isinstance(cache_position, torch.Tensor)
        or cache_position.shape != (1,)
        or cache_position.dtype != torch.long
    ):
        raise ValueError("K8 static cache_position must be [1] long")
    if (
        not isinstance(position_ids, torch.Tensor)
        or position_ids.shape != (1, 1)
        or position_ids.dtype != torch.long
    ):
        raise ValueError("K8 static decoder_position_ids must be [1, 1] long")
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
    processor: K8LogitsProcessor,
    state: K8DeviceState,
    static_inputs: dict[str, Any],
    do_sample: bool,
) -> torch.Tensor:
    scores = processor(logits[:, -1, :])
    next_token = (
        _sample_counter(scores, state)
        if do_sample
        else torch.argmax(scores, dim=-1)
    )
    active = state.append(next_token)
    processor.observe(active)
    _update_static_model_inputs(static_inputs, active)
    return active


def _cuda_check(result: tuple[Any, ...], operation: str) -> tuple[Any, ...]:
    from cuda.bindings import runtime

    if not isinstance(result, tuple) or not result:
        raise RuntimeError(f"{operation} returned an invalid CUDA result")
    if result[0] != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed with {result[0]}")
    return result[1:]


@dataclass(slots=True)
class _ChildGraphSequence:
    parent_graph: Any
    executable: Any
    model_graph: torch.cuda.CUDAGraph
    tail_graph: torch.cuda.CUDAGraph
    closed: bool = False

    @classmethod
    def build(
        cls,
        model_graph: torch.cuda.CUDAGraph,
        tail_graph: torch.cuda.CUDAGraph,
    ) -> "_ChildGraphSequence":
        if not hasattr(model_graph, "raw_cuda_graph") or not hasattr(
            tail_graph, "raw_cuda_graph"
        ):
            raise RuntimeError(
                "K8 requires CUDAGraph(keep_graph=True).raw_cuda_graph()"
            )
        from cuda.bindings import runtime

        parent, = _cuda_check(runtime.cudaGraphCreate(0), "cudaGraphCreate")
        previous = None
        try:
            for _ in range(K8_BLOCK_SIZE):
                for child in (model_graph, tail_graph):
                    dependencies = None if previous is None else (previous,)
                    node, = _cuda_check(
                        runtime.cudaGraphAddChildGraphNode(
                            parent,
                            dependencies,
                            0 if previous is None else 1,
                            runtime.cudaGraph_t(child.raw_cuda_graph()),
                        ),
                        "cudaGraphAddChildGraphNode",
                    )
                    previous = node
            executable, = _cuda_check(
                runtime.cudaGraphInstantiate(parent, 0),
                "cudaGraphInstantiate",
            )
        except Exception:
            runtime.cudaGraphDestroy(parent)
            raise
        return cls(parent, executable, model_graph, tail_graph)

    def replay(self) -> None:
        if self.closed:
            raise RuntimeError("K8 parent graph is closed")
        from cuda.bindings import runtime

        stream = runtime.cudaStream_t(torch.cuda.current_stream().cuda_stream)
        _cuda_check(
            (runtime.cudaGraphLaunch(self.executable, stream),),
            "cudaGraphLaunch",
        )

    def close(self) -> None:
        if self.closed:
            return
        from cuda.bindings import runtime

        _cuda_check(
            (runtime.cudaGraphExecDestroy(self.executable),),
            "cudaGraphExecDestroy",
        )
        _cuda_check(
            (runtime.cudaGraphDestroy(self.parent_graph),),
            "cudaGraphDestroy",
        )
        self.closed = True


@dataclass(slots=True)
class _K8GraphEntry:
    parent: _ChildGraphSequence
    static_inputs: dict[str, Any]
    state: K8DeviceState
    processor: K8LogitsProcessor
    outputs: Any
    capture_seconds: float
    peak_vram_bytes: int
    block_replays: int = 0


@dataclass(slots=True)
class _K8RuntimeSlot:
    state: K8DeviceState
    processor: K8LogitsProcessor
    signature: tuple[Any, ...]


def _freeze_processor_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, slice):
        return (value.start, value.stop, value.step)
    if isinstance(value, torch.Tensor):
        if value.numel() > 4096:
            return (tuple(value.shape), str(value.dtype), str(value.device))
        return (
            tuple(value.shape),
            str(value.dtype),
            tuple(value.detach().cpu().reshape(-1).tolist()),
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_processor_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted(
            (str(key), _freeze_processor_value(item))
            for key, item in value.items()
        ))
    return type(value).__name__


def _processor_signature(processors: Any) -> tuple[Any, ...]:
    rows = []
    for processor in processors:
        attributes = getattr(processor, "__dict__", {})
        rows.append((
            type(processor).__name__,
            tuple(sorted(
                (name, _freeze_processor_value(value))
                for name, value in attributes.items()
                if not name.startswith("_state_")
                and not name.endswith("_by_device")
                and name not in {"last_scores"}
            )),
        ))
    return tuple(rows)


def _runtime_slot(
    holder: dict[str, Any],
    *,
    input_ids: torch.Tensor,
    max_length: int,
    pad_token_id: int,
    eos_token_ids: torch.Tensor,
    logits_processor: Any,
    vocab_size: int,
) -> _K8RuntimeSlot:
    signature = (
        max_length,
        tuple(int(value) for value in eos_token_ids.detach().cpu().tolist()),
        _processor_signature(logits_processor),
        str(input_ids.device),
    )
    slots = holder.setdefault("__k8_runtime_slots__", {})
    slot = slots.get(signature)
    if slot is None:
        state = K8DeviceState.allocate(
            input_ids,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_ids=eos_token_ids,
        )
        processor = K8LogitsProcessor.compile(
            logits_processor,
            state=state,
            vocab_size=vocab_size,
            prompt_length=input_ids.shape[1],
        )
        slot = _K8RuntimeSlot(state, processor, signature)
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
    state: K8DeviceState,
    processor: K8LogitsProcessor,
    static_inputs: dict[str, Any],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    tensors = {
        "sequence": state.sequence,
        "physical_length": state.physical_length,
        "logical_length": state.logical_length,
        "unfinished": state.unfinished,
        "rng_counter": state.rng_counter,
        **processor.mutable_tensors(),
        **{
            f"input:{name}": value
            for name, value in static_inputs.items()
            if isinstance(value, torch.Tensor)
        },
    }
    return [(value, value.detach().clone()) for value in tensors.values()]


def _capture_k8_entry(
    model,
    model_inputs: dict[str, Any],
    *,
    active_prefix_length: int,
    state: K8DeviceState,
    processor: K8LogitsProcessor,
    do_sample: bool,
) -> _K8GraphEntry:
    if not torch.cuda.is_available():
        raise RuntimeError("K8 graph capture requires CUDA")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(state.sequence.device)
    static_inputs = _clone_static_graph_inputs(model_inputs)
    model_graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(model_graph):
        with active_prefix_self_attention_context(active_prefix_length):
            outputs = model(**static_inputs, return_dict=True)
    snapshots = _snapshot_tensors(state, processor, static_inputs)
    tail_graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(tail_graph):
        _one_step_tail(
            logits=outputs.logits,
            processor=processor,
            state=state,
            static_inputs=static_inputs,
            do_sample=do_sample,
        )
    for tensor, snapshot in snapshots:
        tensor.copy_(snapshot)
    parent = _ChildGraphSequence.build(model_graph, tail_graph)
    torch.cuda.synchronize(state.sequence.device)
    return _K8GraphEntry(
        parent=parent,
        static_inputs=static_inputs,
        state=state,
        processor=processor,
        outputs=outputs,
        capture_seconds=time.perf_counter() - started,
        peak_vram_bytes=torch.cuda.max_memory_allocated(state.sequence.device),
    )


def _k8_eligible(cur_len: int, max_length: int, bucket_size: int) -> bool:
    if cur_len + K8_BLOCK_SIZE > max_length:
        return False
    start = _bucketed_prefix_length(cur_len, bucket_size, max_length)
    end = _bucketed_prefix_length(
        cur_len + K8_BLOCK_SIZE - 1, bucket_size, max_length
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
            "K8 requires exactly EosTokenCriteria and MaxLengthCriteria; "
            f"unsupported={unsupported}"
        )
    if not isinstance(eos[0], torch.Tensor):
        raise TypeError("K8 EOS ids must be a tensor")
    if maximum[0] != max_length:
        raise ValueError("K8 MaxLengthCriteria must match cache capacity")
    return eos[0].to(device=device, dtype=torch.long).reshape(-1).contiguous()


def _sync_model_kwargs_after_k8(
    model_kwargs: dict[str, Any],
    entry: _K8GraphEntry,
    *,
    cur_len: int,
) -> None:
    model_kwargs["cache_position"] = entry.static_inputs["cache_position"]
    if "decoder_attention_mask" in model_kwargs:
        original = model_kwargs["decoder_attention_mask"]
        model_kwargs["decoder_attention_mask"] = torch.ones(
            (1, cur_len), dtype=original.dtype, device=original.device
        )


def k8_active_prefix_decode_generate(
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
    """Opt-in full-model+tail K8 loop; defaults never select this callable."""

    del cuda_graph_warmup, cuda_graph_min_decode_steps
    if synced_gpus or streamer is not None:
        raise ValueError("K8 does not support synced_gpus or streamers")
    if input_ids.device.type != "cuda" or not cuda_graph_forward:
        raise RuntimeError("K8 requires CUDA graph decode")
    if input_ids.shape[0] != 1:
        raise ValueError("K8 requires batch size one")
    if generation_config.return_dict_in_generate or any((
        generation_config.output_attentions,
        generation_config.output_hidden_states,
        generation_config.output_scores,
        generation_config.output_logits,
    )):
        raise ValueError("K8 supports tensor outputs only")
    pad_token_id = int(generation_config._pad_token_tensor.reshape(-1)[0].item())
    batch_size, cur_len = input_ids.shape
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
        raise RuntimeError("K8 requires persistent request-local runtime state")
    slot = _runtime_slot(
        stable_encoder_holder,
        input_ids=input_ids,
        max_length=max_length,
        pad_token_id=pad_token_id,
        eos_token_ids=eos_ids,
        logits_processor=logits_processor,
        vocab_size=int(model.config.vocab_size),
    )
    state = slot.state
    processor = slot.processor
    graph_cache = shared_graph_cache if shared_graph_cache is not None else {}
    stats = {
        "k8_candidate": True,
        "window_serial": int(
            stable_encoder_holder.get("__k8_window_serial__", 0)
        ) + 1,
        "block_size": K8_BLOCK_SIZE,
        "eligible_steps": 0,
        "block_replays": 0,
        "remainder_steps": 0,
        "status_reads": 0,
        "copy_calls": 0,
        "copy_bytes": 0,
        "capture_seconds": 0.0,
        "peak_vram_bytes": 0,
        "wasted_steps": 0,
        "rng_policy": RNG_POLICY,
        "parent_backend": "cuda_python_child_graphs",
    }
    stable_encoder_holder["__k8_window_serial__"] = stats["window_serial"]

    is_prefill = True
    finished = False
    scores = None
    while not finished:
        if is_prefill or not _k8_eligible(
            cur_len, max_length, active_prefix_bucket_size
        ):
            active_ids = state.sequence[:, :cur_len]
            model_inputs = model.prepare_inputs_for_generation(
                active_ids, **model_kwargs
            )
            if is_prefill:
                with profile_range("generation.encoder_prefill"):
                    outputs = model(**model_inputs, return_dict=True)
                is_prefill = False
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
            token = (
                _sample_counter(scores_step, state)
                if generation_config.do_sample
                else torch.argmax(scores_step, dim=-1)
            )
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
        key = ("k8_candidate", id(state), prefix, tuple(
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
            entry = _capture_k8_entry(
                model,
                model_inputs,
                active_prefix_length=prefix,
                state=state,
                processor=processor,
                do_sample=bool(generation_config.do_sample),
            )
            raw_entry = {
                "k8_candidate": True,
                "runtime_entry": entry,
                "active_prefix_length": prefix,
                "capture_seconds": entry.capture_seconds,
                "decode_replays": 0,
                "k8_stats": stats,
            }
            graph_cache[key] = raw_entry
            stats["capture_seconds"] += entry.capture_seconds
            stats["peak_vram_bytes"] = max(
                stats["peak_vram_bytes"], entry.peak_vram_bytes
            )
        else:
            entry = raw_entry["runtime_entry"]
            raw_entry["k8_stats"] = stats
            if entry.state is not state or entry.processor is not processor:
                raise RuntimeError("K8 graph resolved the wrong persistent state")
            for name, static_value in entry.static_inputs.items():
                if not isinstance(static_value, torch.Tensor):
                    continue
                current = model_inputs.get(name)
                if not isinstance(current, torch.Tensor):
                    raise RuntimeError(f"K8 model input {name} stopped being a tensor")
                if (
                    current.shape != static_value.shape
                    or current.dtype != static_value.dtype
                    or current.device != static_value.device
                ):
                    raise RuntimeError(f"K8 model input {name} changed signature")
                static_value.copy_(current)
                stats["copy_calls"] += 1
                stats["copy_bytes"] += current.numel() * current.element_size()
        stats["eligible_steps"] += K8_BLOCK_SIZE
        with profile_range("generation.k8_parent_graph_replay"):
            entry.parent.replay()
        entry.block_replays += 1
        raw_entry["decode_replays"] += K8_BLOCK_SIZE
        stats["block_replays"] += 1
        stats["status_reads"] += 1
        status = torch.stack((
            state.physical_length[0],
            state.logical_length[0],
            state.unfinished[0].to(dtype=torch.long),
        )).cpu().tolist()
        physical_length, logical_length = (int(status[0]), int(status[1]))
        unfinished = bool(status[2])
        stats["wasted_steps"] += max(0, physical_length - logical_length)
        cur_len = physical_length if unfinished else logical_length
        _sync_model_kwargs_after_k8(model_kwargs, entry, cur_len=cur_len)
        finished = not unfinished

    return state.sequence[:, :cur_len]


@contextmanager
def install_k8_candidate() -> Iterator[None]:
    """Temporarily install K8 only for an explicit candidate runner."""

    from . import engine

    previous = engine.active_prefix_decode_generate
    engine.active_prefix_decode_generate = k8_active_prefix_decode_generate
    try:
        yield
    finally:
        engine.active_prefix_decode_generate = previous


__all__ = [
    "K8DeviceState",
    "K8LogitsProcessor",
    "K8_BLOCK_SIZE",
    "RNG_POLICY",
    "install_k8_candidate",
    "k8_active_prefix_decode_generate",
]
