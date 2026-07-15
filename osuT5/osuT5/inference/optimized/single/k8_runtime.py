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
    _copy_static_graph_inputs,
    _max_cache_shape,
    _stable_encoder_outputs,
)
from .runtime_context import active_prefix_self_attention_context


K8_BLOCK_SIZE = 8
SUPPORTED_BLOCK_SIZES = (1, 4, K8_BLOCK_SIZE)
RNG_POLICY = "counter_request_seed_window_prompt_v2"
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


def _prompt_seed(
    input_ids: torch.Tensor,
    *,
    request_seed: int,
    window_identity: int,
) -> int:
    """Stable request/window stream seed; the host copy is charged as setup."""

    values = input_ids.detach().cpu().reshape(-1).tolist()
    state = 0xCBF29CE484222325
    for value in (request_seed, window_identity):
        state ^= int(value) & 0xFFFFFFFFFFFFFFFF
        state = (state * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
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
        rng_seed: int,
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
            rng_seed=torch.full_like(length, int(rng_seed)),
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
        rng_seed: int,
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
        self.rng_seed.fill_(int(rng_seed))
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


def _cuda_status(result: tuple[Any, ...], operation: str) -> None:
    payload = _cuda_check(result, operation)
    if payload:
        raise RuntimeError(f"{operation} returned unexpected payload {payload!r}")


@dataclass(slots=True)
class _ChildGraphSequence:
    parent_graph: Any
    executable: Any
    model_graph: torch.cuda.CUDAGraph
    tail_graph: torch.cuda.CUDAGraph
    device: torch.device
    closed: bool = False
    executable_closed: bool = False
    parent_closed: bool = False

    @classmethod
    def build(
        cls,
        model_graph: torch.cuda.CUDAGraph,
        tail_graph: torch.cuda.CUDAGraph,
        *,
        block_size: int = K8_BLOCK_SIZE,
        device: torch.device | None = None,
    ) -> "_ChildGraphSequence":
        block_size = _validate_block_size(block_size)
        if not hasattr(model_graph, "raw_cuda_graph") or not hasattr(
            tail_graph, "raw_cuda_graph"
        ):
            raise RuntimeError(
                "K8 requires CUDAGraph(keep_graph=True).raw_cuda_graph()"
            )
        from cuda.bindings import runtime

        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        device = torch.device(device)
        with torch.cuda.device(device):
            parent, = _cuda_check(runtime.cudaGraphCreate(0), "cudaGraphCreate")
            previous = None
            try:
                for _ in range(block_size):
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
                _cuda_status(runtime.cudaGraphDestroy(parent), "cudaGraphDestroy")
                raise
        return cls(parent, executable, model_graph, tail_graph, device)

    def replay(self) -> None:
        if self.closed:
            raise RuntimeError("K8 parent graph is closed")
        from cuda.bindings import runtime

        with torch.cuda.device(self.device):
            stream = runtime.cudaStream_t(torch.cuda.current_stream().cuda_stream)
            _cuda_status(
                runtime.cudaGraphLaunch(self.executable, stream),
                "cudaGraphLaunch",
            )

    def close(self) -> None:
        if self.closed:
            return
        from cuda.bindings import runtime

        with torch.cuda.device(self.device):
            if not self.executable_closed:
                _cuda_status(
                    runtime.cudaGraphExecDestroy(self.executable),
                    "cudaGraphExecDestroy",
                )
                self.executable_closed = True
            if not self.parent_closed:
                _cuda_status(
                    runtime.cudaGraphDestroy(self.parent_graph),
                    "cudaGraphDestroy",
                )
                self.parent_closed = True
        self.closed = self.executable_closed and self.parent_closed


@dataclass(slots=True)
class _K8GraphLifecycle:
    active_cache_id: int | None = None
    graphs: dict[int, _ChildGraphSequence] = field(default_factory=dict)

    def activate(self, graph_cache: dict[Any, Any]) -> None:
        cache_id = id(graph_cache)
        if self.active_cache_id is not None and self.active_cache_id != cache_id:
            self.close_all()
        self.active_cache_id = cache_id

    def register(self, graph: _ChildGraphSequence) -> None:
        if graph.closed:
            raise RuntimeError("cannot register a closed K8 child graph")
        self.graphs[id(graph)] = graph

    def close_all(self) -> None:
        first_error: Exception | None = None
        remaining: dict[int, _ChildGraphSequence] = {}
        for identity, graph in tuple(self.graphs.items()):
            try:
                graph.close()
            except Exception as exc:
                remaining[identity] = graph
                if first_error is None:
                    first_error = exc
        self.graphs = remaining
        self.active_cache_id = None
        if first_error is not None:
            raise first_error


_ACTIVE_K8_LIFECYCLE: _K8GraphLifecycle | None = None
_ACTIVE_BLOCK_SIZE: int | None = None
_ACTIVE_GRAPH_REMAINDERS: bool | None = None
_ACTIVE_SHARED_STATIC_INPUT_ARENA: bool | None = None
_ACTIVE_TRANSITION_TIMING: bool | None = None


_TRANSITION_STAGES = (
    "prepare_inputs_for_generation",
    "graph_key_lookup",
    "static_input_refresh",
    "static_input_copies",
    "graph_replay",
    "status_d2h",
    "sync_model_kwargs",
)


def _static_input_signature(model_inputs: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        (name, tuple(value.shape), str(value.dtype), str(value.device))
        if isinstance(value, torch.Tensor)
        else (name, type(value).__name__, id(value))
        for name, value in sorted(model_inputs.items())
    )


def _tensor_copy_size(model_inputs: dict[str, Any]) -> tuple[int, int]:
    tensors = [
        value for value in model_inputs.values() if isinstance(value, torch.Tensor)
    ]
    return (
        len(tensors),
        sum(value.numel() * value.element_size() for value in tensors),
    )


def _expected_static_input_arena_refreshes(
    *,
    block_replays: int,
    remainder_graph_replays: int,
) -> int:
    """Return one only when this window actually entered a captured graph."""

    for name, value in (
        ("block_replays", block_replays),
        ("remainder_graph_replays", remainder_graph_replays),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"{name} must be a non-negative integer")
    return int(block_replays + remainder_graph_replays > 0)


def _validate_static_input_arena_refreshes(
    stats: dict[str, Any],
    *,
    enabled: bool,
) -> None:
    if not enabled:
        return
    expected = _expected_static_input_arena_refreshes(
        block_replays=stats["block_replays"],
        remainder_graph_replays=stats["remainder_graph_replays"],
    )
    if stats["static_input_arena_refreshes"] != expected:
        raise RuntimeError(
            "shared static-input arena requires one refresh for graph-using "
            "windows and zero refreshes for prefill-only windows"
        )


@dataclass(slots=True)
class _StaticInputArena:
    static_inputs: dict[str, Any]
    signature: tuple[Any, ...]
    tensor_addresses: tuple[tuple[str, int], ...]
    refreshed_window_identity: tuple[int, int]
    content_match: torch.Tensor

    @staticmethod
    def _tensor_addresses(values: dict[str, Any]) -> tuple[tuple[str, int], ...]:
        return tuple(
            (name, value.data_ptr())
            for name, value in sorted(values.items())
            if isinstance(value, torch.Tensor)
        )

    def validate_tensor_addresses(self) -> None:
        if self._tensor_addresses(self.static_inputs) != self.tensor_addresses:
            raise RuntimeError("shared static-input arena tensor address changed")

    @classmethod
    def create(
        cls,
        model_inputs: dict[str, Any],
        *,
        window_identity: tuple[int, int],
    ) -> "_StaticInputArena":
        static_inputs = _clone_static_graph_inputs(model_inputs)
        tensor_values = [
            value for value in static_inputs.values() if isinstance(value, torch.Tensor)
        ]
        if not tensor_values:
            raise RuntimeError("shared static-input arena requires tensor inputs")
        arena = cls(
            static_inputs=static_inputs,
            signature=_static_input_signature(model_inputs),
            tensor_addresses=cls._tensor_addresses(static_inputs),
            refreshed_window_identity=window_identity,
            content_match=torch.ones(
                (),
                dtype=torch.bool,
                device=tensor_values[0].device,
            ),
        )
        arena.accumulate_content_match(model_inputs)
        return arena

    def accumulate_content_match(self, model_inputs: dict[str, Any]) -> int:
        """Asynchronously prove every copied tensor matches its live source."""

        self.validate_tensor_addresses()
        checks = 0
        for name, static_value in self.static_inputs.items():
            if not isinstance(static_value, torch.Tensor):
                continue
            current = model_inputs.get(name)
            if not isinstance(current, torch.Tensor):
                raise RuntimeError(f"shared arena input {name} stopped being a tensor")
            if (
                current.shape != static_value.shape
                or current.dtype != static_value.dtype
                or current.device != static_value.device
            ):
                raise RuntimeError(f"shared arena input {name} changed tensor signature")
            self.content_match.logical_and_(torch.eq(static_value, current).all())
            checks += 1
        if checks <= 0:
            raise RuntimeError("shared static-input arena validated no tensors")
        return checks

    def refresh(
        self,
        model_inputs: dict[str, Any],
        *,
        window_identity: tuple[int, int],
    ) -> None:
        self.validate_tensor_addresses()
        if window_identity == self.refreshed_window_identity:
            raise RuntimeError("shared static-input arena refreshed twice in one window")
        signature = _static_input_signature(model_inputs)
        if signature != self.signature:
            raise RuntimeError("shared static-input arena signature changed")
        _copy_static_graph_inputs(self.static_inputs, model_inputs)
        self.accumulate_content_match(model_inputs)
        self.validate_tensor_addresses()
        self.refreshed_window_identity = window_identity


@dataclass(slots=True)
class _K8GraphEntry:
    parent: _ChildGraphSequence
    remainder_parent: _ChildGraphSequence | None
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
    static_input_arena: _StaticInputArena | None = None


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


def _request_seed(generator: Any, device: torch.device) -> int:
    if isinstance(generator, (list, tuple)):
        if len(generator) != 1:
            raise ValueError("K8 batch-one decoding accepts one generator")
        generator = generator[0]
    if generator is not None:
        if not isinstance(generator, torch.Generator):
            raise TypeError("K8 generator must be a torch.Generator")
        return int(generator.initial_seed())
    if device.type != "cuda":
        raise RuntimeError("K8 request seed fallback requires a CUDA device")
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return int(torch.cuda.default_generators[index].initial_seed())


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
    rng_seed: int,
    do_sample: bool,
) -> _K8RuntimeSlot:
    signature = (
        max_length,
        tuple(int(value) for value in eos_token_ids.detach().cpu().tolist()),
        processor_signature,
        str(input_ids.device),
        bool(do_sample),
    )
    slots = holder.setdefault("__k8_runtime_slots__", {})
    slot = slots.get(signature)
    if slot is None:
        state = K8DeviceState.allocate(
            input_ids,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_ids=eos_token_ids,
            rng_seed=rng_seed,
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
            rng_seed=rng_seed,
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


def _restore_snapshot_tensors(
    snapshots: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> None:
    """Order capture-time writes before restoring persistent K8 state."""

    # torch.cuda.graph captures on a side stream and capture_end does not make
    # the caller's stream wait for that work.  Without this synchronization,
    # tail-capture writes to sequence/RNG/processor state can race these copies.
    torch.cuda.synchronize(device)
    for tensor, snapshot in snapshots:
        tensor.copy_(snapshot)
    torch.cuda.synchronize(device)


def _capture_k8_entry(
    model,
    model_inputs: dict[str, Any],
    *,
    active_prefix_length: int,
    state: K8DeviceState,
    processor: K8LogitsProcessor,
    do_sample: bool,
    block_size: int = K8_BLOCK_SIZE,
    graph_remainders: bool = False,
    static_input_arena: dict[str, Any] | None = None,
) -> _K8GraphEntry:
    if not torch.cuda.is_available():
        raise RuntimeError("K8 graph capture requires CUDA")
    started = time.perf_counter()
    device = state.sequence.device
    with torch.cuda.device(device):
        torch.cuda.reset_peak_memory_stats(device)
        static_inputs = (
            _clone_static_graph_inputs(model_inputs)
            if static_input_arena is None
            else static_input_arena
        )
        if not isinstance(static_inputs, dict):
            raise TypeError("K8 static-input arena must be a dict")
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
        _restore_snapshot_tensors(snapshots, device=device)
        parent = _ChildGraphSequence.build(
            model_graph,
            tail_graph,
            block_size=block_size,
            device=device,
        )
        remainder_parent = None
        try:
            if graph_remainders:
                remainder_parent = _ChildGraphSequence.build(
                    model_graph,
                    tail_graph,
                    block_size=1,
                    device=device,
                )
            torch.cuda.synchronize(device)
            return _K8GraphEntry(
                parent=parent,
                remainder_parent=remainder_parent,
                static_inputs=static_inputs,
                state=state,
                processor=processor,
                outputs=outputs,
                capture_seconds=time.perf_counter() - started,
                peak_vram_bytes=torch.cuda.max_memory_allocated(device),
            )
        except Exception:
            if remainder_parent is not None:
                remainder_parent.close()
            parent.close()
            raise


def _validate_block_size(block_size: int) -> int:
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise TypeError("candidate block size must be an integer")
    if block_size not in SUPPORTED_BLOCK_SIZES:
        raise ValueError(
            f"candidate block size must be one of {SUPPORTED_BLOCK_SIZES}, got {block_size}"
        )
    return block_size


def _k8_eligible(
    cur_len: int,
    max_length: int,
    bucket_size: int,
    block_size: int = K8_BLOCK_SIZE,
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
        if (
            not isinstance(original, torch.Tensor)
            or original.ndim != 2
            or tuple(original.shape[:1]) != (1,)
            or original.shape[1] > cur_len
        ):
            raise ValueError("K8 decoder attention mask must be [1, <=cur_len]")
        extended = torch.ones(
            (1, cur_len), dtype=original.dtype, device=original.device
        )
        extended[:, : original.shape[1]].copy_(original)
        model_kwargs["decoder_attention_mask"] = extended


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
    k8_request_state: dict[str, Any] | None = None,
    k8_graph_lifecycle: _K8GraphLifecycle | None = None,
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
    lifecycle = (
        k8_graph_lifecycle
        if k8_graph_lifecycle is not None
        else _ACTIVE_K8_LIFECYCLE
    )
    block_size = _ACTIVE_BLOCK_SIZE
    graph_remainders = _ACTIVE_GRAPH_REMAINDERS
    shared_static_input_arena = _ACTIVE_SHARED_STATIC_INPUT_ARENA
    transition_timing = _ACTIVE_TRANSITION_TIMING
    if (
        lifecycle is None
        or block_size is None
        or graph_remainders is None
        or shared_static_input_arena is None
        or transition_timing is None
    ):
        raise RuntimeError("K8 decode must run inside install_k8_candidate()")
    pad_token_id = int(generation_config._pad_token_tensor.reshape(-1)[0].item())
    _, cur_len = input_ids.shape
    generator = model_kwargs.pop("generator", None)
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
    request_state = (
        stable_encoder_holder
        if k8_request_state is None
        else k8_request_state
    )
    if not isinstance(request_state, dict):
        raise TypeError("K8 request state must be a dictionary")
    graph_cache = shared_graph_cache if shared_graph_cache is not None else {}
    lifecycle.activate(graph_cache)
    window_identity = int(
        request_state.get("__k8_window_serial__", 0)
    ) + 1
    request_state["__k8_window_serial__"] = window_identity
    persistent_request_serial = request_state.get(
        "__persistent_request_serial__",
        0,
    )
    if (
        isinstance(persistent_request_serial, bool)
        or not isinstance(persistent_request_serial, int)
        or persistent_request_serial < 0
    ):
        raise RuntimeError(
            "persistent request serial must be a non-negative integer"
        )
    arena_window_identity = (persistent_request_serial, window_identity)
    current_request_seed = _request_seed(generator, input_ids.device)
    stored_request_seed = request_state.setdefault(
        "__k8_request_seed__",
        current_request_seed,
    )
    if int(stored_request_seed) != current_request_seed:
        raise RuntimeError("K8 request-local generator seed changed mid-session")

    prompt_seed_started = time.perf_counter()
    rng_seed = _prompt_seed(
        input_ids,
        request_seed=current_request_seed,
        window_identity=window_identity,
    )
    prompt_seed_seconds = time.perf_counter() - prompt_seed_started
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
    slot = _runtime_slot(
        stable_encoder_holder,
        input_ids=input_ids,
        max_length=max_length,
        pad_token_id=pad_token_id,
        eos_token_ids=eos_ids,
        logits_processor=logits_processor,
        processor_signature=processor_signature,
        vocab_size=int(model.config.vocab_size),
        rng_seed=rng_seed,
        do_sample=do_sample,
    )
    state = slot.state
    processor = slot.processor
    prompt_length = int(cur_len)
    stats = {
        "k8_candidate": True,
        "window_serial": window_identity,
        "block_size": block_size,
        "prefill_steps": 0,
        "eligible_steps": 0,
        "block_replays": 0,
        "remainder_steps": 0,
        "remainder_backend": "cuda_graph" if graph_remainders else "eager",
        "remainder_graph_replays": 0,
        "eager_remainder_steps": 0,
        "status_reads": 0,
        "copy_calls": 0,
        "copy_bytes": 0,
        "capture_seconds": 0.0,
        "peak_vram_bytes": 0,
        "physical_steps": 0,
        "logical_steps": 0,
        "wasted_steps": 0,
        "rng_policy": RNG_POLICY,
        "rng_exact": False,
        "rng_drift": "documented_counter_based_per_window",
        "rng_request_seed": current_request_seed,
        "rng_window_identity": window_identity,
        "rng_early_eos_isolation": True,
        "sampling_mode": "sample" if do_sample else "greedy",
        "prompt_seed_d2h_copy_calls": 1,
        "prompt_seed_d2h_copy_bytes": input_ids.numel() * input_ids.element_size(),
        "prompt_seed_setup_seconds": prompt_seed_seconds,
        "processor_signature_d2h_copy_calls": processor_d2h["calls"],
        "processor_signature_d2h_copy_bytes": processor_d2h["bytes"],
        "processor_signature_setup_seconds": processor_signature_seconds,
        "parent_backend": "cuda_python_child_graphs",
        "capture_state_restore_synchronized": True,
        "shared_static_input_arena": shared_static_input_arena,
        "static_input_arena_refreshes": 0,
        "static_input_arena_refresh_copy_calls": 0,
        "static_input_arena_refresh_copy_bytes": 0,
        "static_input_arena_graph_entries": 0,
        "static_input_arena_content_checks": 0,
        "static_input_arena_content_match": True,
    }
    if transition_timing:
        stats["transition_timing"] = {
            "schema_version": 1,
            "enabled": True,
            "model_wall_seconds": 0.0,
            "accounted_wall_seconds": 0.0,
            "unattributed_wall_seconds": 0.0,
            "reconciliation_error_fraction": 0.0,
            "stages": {
                name: {"calls": 0, "wall_seconds": 0.0}
                for name in _TRANSITION_STAGES
            },
            "device": {
                "static_input_copies": {"calls": 0, "seconds": 0.0},
                "graph_replay": {"calls": 0, "seconds": 0.0},
            },
        }
    stable_encoder_holder["__k8_latest_stats__"] = stats

    is_prefill = True
    finished = False
    scores = None
    copy_events = (
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        if transition_timing
        else None
    )
    replay_events = (
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        if transition_timing
        else None
    )
    while not finished:
        if is_prefill:
            active_ids = state.sequence[:, :cur_len]
            model_inputs = model.prepare_inputs_for_generation(
                active_ids, **model_kwargs
            )
            with profile_range("generation.encoder_prefill"):
                outputs = model(**model_inputs, return_dict=True)
            is_prefill = False
            stats["prefill_steps"] += 1
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
                if do_sample
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

        block_eligible = _k8_eligible(
            cur_len,
            max_length,
            active_prefix_bucket_size,
            block_size,
        )
        if not block_eligible and not graph_remainders:
            active_ids = state.sequence[:, :cur_len]
            model_inputs = model.prepare_inputs_for_generation(
                active_ids, **model_kwargs
            )
            prefix = _bucketed_prefix_length(
                cur_len, active_prefix_bucket_size, max_length
            )
            with active_prefix_self_attention_context(prefix):
                outputs = model(**model_inputs, return_dict=True)
            stats["remainder_steps"] += 1
            stats["eager_remainder_steps"] += 1
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
                if do_sample
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

        model_wall_started = time.perf_counter()
        prefix = _bucketed_prefix_length(
            cur_len, active_prefix_bucket_size, max_length
        )
        timing_stats = stats.get("transition_timing")
        local_stage_wall = {name: 0.0 for name in _TRANSITION_STAGES}
        local_stage_calls = {name: 0 for name in _TRANSITION_STAGES}
        copy_events_recorded = False

        def record_stage(name: str, started: float) -> None:
            if timing_stats is None:
                return
            local_stage_calls[name] += 1
            local_stage_wall[name] += time.perf_counter() - started

        prepare_started = time.perf_counter()
        arena = slot.static_input_arena
        first_arena_use_this_window = (
            shared_static_input_arena
            and (
                arena is None
                or arena.refreshed_window_identity != arena_window_identity
            )
        )
        if not shared_static_input_arena or first_arena_use_this_window:
            active_ids = state.sequence[:, :cur_len]
            model_inputs = model.prepare_inputs_for_generation(
                active_ids, **model_kwargs
            )
            record_stage("prepare_inputs_for_generation", prepare_started)
        else:
            if arena is None:
                raise RuntimeError("shared static-input arena disappeared")
            model_inputs = arena.static_inputs

        refresh_started = time.perf_counter()
        if first_arena_use_this_window:
            if copy_events is not None:
                copy_events[0].record()
            copy_calls, copy_bytes = _tensor_copy_size(model_inputs)
            if arena is None:
                arena = _StaticInputArena.create(
                    model_inputs,
                    window_identity=arena_window_identity,
                )
                slot.static_input_arena = arena
                stats["static_input_arena_content_checks"] += len(
                    arena.tensor_addresses
                )
            else:
                arena.refresh(
                    model_inputs,
                    window_identity=arena_window_identity,
                )
                stats["static_input_arena_content_checks"] += len(
                    arena.tensor_addresses
                )
            stats["static_input_arena_refreshes"] += 1
            stats["static_input_arena_refresh_copy_calls"] += copy_calls
            stats["static_input_arena_refresh_copy_bytes"] += copy_bytes
            stats["copy_calls"] += copy_calls
            stats["copy_bytes"] += copy_bytes
            model_inputs = arena.static_inputs
            if copy_events is not None:
                copy_events[1].record()
                copy_events_recorded = True
            record_stage("static_input_refresh", refresh_started)

        key_started = time.perf_counter()
        key = (
            "k8_candidate",
            block_size,
            graph_remainders,
            shared_static_input_arena,
            id(state),
            prefix,
            do_sample,
            (
                arena.signature
                if shared_static_input_arena and arena is not None
                else _static_input_signature(model_inputs)
            ),
        )
        raw_entry = graph_cache.get(key)
        steady_state_replay = raw_entry is not None
        record_stage("graph_key_lookup", key_started)
        if raw_entry is None:
            if not shared_static_input_arena:
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
                do_sample=do_sample,
                block_size=block_size,
                graph_remainders=graph_remainders,
                static_input_arena=(
                    arena.static_inputs
                    if shared_static_input_arena and arena is not None
                    else None
                ),
            )
            if shared_static_input_arena:
                if arena is None:
                    raise RuntimeError(
                        "shared static-input arena disappeared during capture"
                    )
                arena.validate_tensor_addresses()
                if entry.static_inputs is not arena.static_inputs:
                    raise RuntimeError(
                        "K8 graph capture did not retain the shared input arena"
                    )
            try:
                lifecycle.register(entry.parent)
                if entry.remainder_parent is not None:
                    lifecycle.register(entry.remainder_parent)
            except Exception:
                if entry.remainder_parent is not None:
                    entry.remainder_parent.close()
                entry.parent.close()
                raise
            raw_entry = {
                "k8_candidate": True,
                "runtime_entry": entry,
                "active_prefix_length": prefix,
                "capture_seconds": entry.capture_seconds,
                "decode_replays": 0,
                "k8_stats": stats,
                "last_request_serial": request_state.get(
                    "__persistent_request_serial__"
                ),
                "cross_request_reuse_hits": 0,
            }
            graph_cache[key] = raw_entry
            if shared_static_input_arena:
                stats["static_input_arena_graph_entries"] += 1
            stats["capture_seconds"] += entry.capture_seconds
            stats["peak_vram_bytes"] = max(
                stats["peak_vram_bytes"], entry.peak_vram_bytes
            )
        else:
            entry = raw_entry["runtime_entry"]
            raw_entry["k8_stats"] = stats
            request_serial = request_state.get("__persistent_request_serial__")
            if (
                request_serial is not None
                and raw_entry.get("last_request_serial") != request_serial
            ):
                raw_entry["cross_request_reuse_hits"] = int(
                    raw_entry.get("cross_request_reuse_hits", 0)
                ) + 1
                raw_entry["last_request_serial"] = request_serial
            if entry.state is not state or entry.processor is not processor:
                raise RuntimeError("K8 graph resolved the wrong persistent state")
            copy_started = time.perf_counter()
            if shared_static_input_arena:
                if arena is None:
                    raise RuntimeError("shared static-input arena disappeared")
                arena.validate_tensor_addresses()
                if entry.static_inputs is not arena.static_inputs:
                    raise RuntimeError("K8 graph did not retain the shared input arena")
            else:
                if copy_events is not None:
                    copy_events[0].record()
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
                if copy_events is not None:
                    copy_events[1].record()
                    copy_events_recorded = True
                record_stage("static_input_copies", copy_started)
        replay_steps = block_size if block_eligible else 1
        if block_eligible:
            stats["eligible_steps"] += block_size
            parent = entry.parent
            entry.block_replays += 1
            stats["block_replays"] += 1
            replay_range = "generation.k8_parent_graph_replay"
        else:
            parent = entry.remainder_parent
            if parent is None:
                raise RuntimeError("K1 remainder graph was not captured")
            stats["remainder_steps"] += 1
            stats["remainder_graph_replays"] += 1
            replay_range = "generation.k1_remainder_graph_replay"
        with profile_range(replay_range):
            replay_started = time.perf_counter()
            if replay_events is not None:
                replay_events[0].record()
            parent.replay()
            if replay_events is not None:
                replay_events[1].record()
            record_stage("graph_replay", replay_started)
        raw_entry["decode_replays"] += replay_steps
        stats["status_reads"] += 1
        status_started = time.perf_counter()
        status = torch.stack((
            state.physical_length[0],
            state.logical_length[0],
            state.unfinished[0].to(dtype=torch.long),
        )).cpu().tolist()
        record_stage("status_d2h", status_started)
        if timing_stats is not None and steady_state_replay:
            if copy_events_recorded and copy_events is not None:
                device_copy = timing_stats["device"]["static_input_copies"]
                device_copy["calls"] += 1
                device_copy["seconds"] += copy_events[0].elapsed_time(
                    copy_events[1]
                ) / 1_000.0
            if replay_events is None:
                raise RuntimeError("transition replay timing events disappeared")
            device_replay = timing_stats["device"]["graph_replay"]
            device_replay["calls"] += 1
            device_replay["seconds"] += replay_events[0].elapsed_time(
                replay_events[1]
            ) / 1_000.0
        physical_length, logical_length = (int(status[0]), int(status[1]))
        unfinished = bool(status[2])
        stats["wasted_steps"] += max(0, physical_length - logical_length)
        cur_len = physical_length if unfinished else logical_length
        sync_started = time.perf_counter()
        if not shared_static_input_arena:
            _sync_model_kwargs_after_k8(model_kwargs, entry, cur_len=cur_len)
            record_stage("sync_model_kwargs", sync_started)
        finished = not unfinished
        if timing_stats is not None and steady_state_replay:
            model_wall = time.perf_counter() - model_wall_started
            timing_stats["model_wall_seconds"] += model_wall
            for name in _TRANSITION_STAGES:
                stage = timing_stats["stages"][name]
                stage["calls"] += local_stage_calls[name]
                stage["wall_seconds"] += local_stage_wall[name]

    stats["logical_steps"] = cur_len - prompt_length
    stats["physical_steps"] = stats["logical_steps"] + stats["wasted_steps"]
    if stats["physical_steps"] - stats["logical_steps"] != stats["wasted_steps"]:
        raise RuntimeError("K8 physical/logical/wasted step accounting diverged")
    accounted_steps = (
        stats["prefill_steps"]
        + stats["eligible_steps"]
        + stats["remainder_steps"]
    )
    if stats["physical_steps"] != accounted_steps:
        raise RuntimeError("K8 prefill/eligible/remainder work accounting diverged")
    _validate_static_input_arena_refreshes(
        stats,
        enabled=shared_static_input_arena,
    )
    if shared_static_input_arena and stats["static_input_arena_refreshes"] > 0:
        arena = slot.static_input_arena
        if arena is None:
            raise RuntimeError("shared static-input arena disappeared before validation")
        stats["static_input_arena_content_match"] = bool(
            arena.content_match.item()
        )
        if not stats["static_input_arena_content_match"]:
            raise RuntimeError("shared static-input arena content copy diverged")
    if transition_timing:
        timing_stats = stats["transition_timing"]
        accounted = sum(
            float(stage["wall_seconds"])
            for stage in timing_stats["stages"].values()
        )
        model_wall = float(timing_stats["model_wall_seconds"])
        unattributed = model_wall - accounted
        timing_stats["accounted_wall_seconds"] = accounted
        timing_stats["unattributed_wall_seconds"] = unattributed
        timing_stats["reconciliation_error_fraction"] = (
            abs(unattributed) / model_wall
            if model_wall > 0.0
            else 0.0
        )
    return state.sequence[:, :cur_len]


@contextmanager
def install_k8_candidate(
    *,
    block_size: int = K8_BLOCK_SIZE,
    graph_remainders: bool = False,
    shared_static_input_arena: bool = False,
    transition_timing: bool = False,
) -> Iterator[None]:
    """Temporarily install a bounded block candidate for an explicit runner."""

    global _ACTIVE_BLOCK_SIZE, _ACTIVE_GRAPH_REMAINDERS, _ACTIVE_K8_LIFECYCLE
    global _ACTIVE_SHARED_STATIC_INPUT_ARENA, _ACTIVE_TRANSITION_TIMING
    from . import engine

    block_size = _validate_block_size(block_size)
    if not isinstance(graph_remainders, bool):
        raise TypeError("graph_remainders must be a bool")
    if not isinstance(shared_static_input_arena, bool):
        raise TypeError("shared_static_input_arena must be a bool")
    if not isinstance(transition_timing, bool):
        raise TypeError("transition_timing must be a bool")
    if shared_static_input_arena and not graph_remainders:
        raise ValueError("shared static-input arena requires graphed remainders")
    if (
        _ACTIVE_K8_LIFECYCLE is not None
        or _ACTIVE_BLOCK_SIZE is not None
        or _ACTIVE_GRAPH_REMAINDERS is not None
        or _ACTIVE_SHARED_STATIC_INPUT_ARENA is not None
        or _ACTIVE_TRANSITION_TIMING is not None
    ):
        raise RuntimeError("K8 candidate context cannot be nested")
    lifecycle = _K8GraphLifecycle()
    previous = engine.active_prefix_decode_generate
    _ACTIVE_K8_LIFECYCLE = lifecycle
    _ACTIVE_BLOCK_SIZE = block_size
    _ACTIVE_GRAPH_REMAINDERS = graph_remainders
    _ACTIVE_SHARED_STATIC_INPUT_ARENA = shared_static_input_arena
    _ACTIVE_TRANSITION_TIMING = transition_timing
    engine.active_prefix_decode_generate = k8_active_prefix_decode_generate
    try:
        yield
    finally:
        engine.active_prefix_decode_generate = previous
        try:
            lifecycle.close_all()
        finally:
            _ACTIVE_K8_LIFECYCLE = None
            _ACTIVE_BLOCK_SIZE = None
            _ACTIVE_GRAPH_REMAINDERS = None
            _ACTIVE_SHARED_STATIC_INPUT_ARENA = None
            _ACTIVE_TRANSITION_TIMING = None


__all__ = [
    "K8DeviceState",
    "K8LogitsProcessor",
    "K8_BLOCK_SIZE",
    "RNG_POLICY",
    "SUPPORTED_BLOCK_SIZES",
    "install_k8_candidate",
    "k8_active_prefix_decode_generate",
]
