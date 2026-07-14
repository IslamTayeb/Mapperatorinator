from __future__ import annotations

from bisect import bisect_left
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from types import MethodType
from typing import Any, Iterator

import torch

from ....event import ContextType, EventType
from ....tokenizer import MILISECONDS_PER_STEP


SCOUT_VERSION = "exact-main-prompt-context-v1"


@dataclass
class _SortedTimesState:
    values: list[float]
    length: int
    first: float | None
    last: float | None


@dataclass
class _ContextEncodingState:
    context: dict[str, Any]
    events: list[Any]
    event_times: list[float]
    encoded_tokens: list[int | None]
    length: int
    first_signature: tuple[Any, Any] | None
    last_signature: tuple[Any, Any] | None


@dataclass
class PromptContextStats:
    main_generate_calls: int = 0
    prompt_calls: int = 0
    prompt_retries: int = 0
    range_queries: int = 0
    full_order_validations: int = 0
    incremental_order_validations: int = 0
    validated_time_values: int = 0
    prompt_buffer_resizes: int = 0
    prompt_buffer_capacity: int = 0
    prompt_tokens_written: int = 0
    padding_tokens_written: int = 0
    vectorized_encode_calls: int = 0
    cached_static_events: int = 0
    dynamic_time_events: int = 0
    appended_events: int = 0

    def as_dict(self) -> dict[str, int | str | bool]:
        if self.main_generate_calls != 1:
            raise RuntimeError(
                "prompt-context scout expected exactly one main generation call"
            )
        if self.prompt_calls <= 0 or self.range_queries <= 0:
            raise RuntimeError("prompt-context scout did not prepare any prompts")
        return {
            "version": SCOUT_VERSION,
            "scope": "main_generation_only",
            "exactness_required": True,
            "binary_search_ranges": True,
            "reused_cpu_prompt_buffer": True,
            "main_generate_calls": self.main_generate_calls,
            "prompt_calls": self.prompt_calls,
            "prompt_retries": self.prompt_retries,
            "range_queries": self.range_queries,
            "full_order_validations": self.full_order_validations,
            "incremental_order_validations": self.incremental_order_validations,
            "validated_time_values": self.validated_time_values,
            "prompt_buffer_resizes": self.prompt_buffer_resizes,
            "prompt_buffer_capacity": self.prompt_buffer_capacity,
            "prompt_tokens_written": self.prompt_tokens_written,
            "padding_tokens_written": self.padding_tokens_written,
            "vectorized_encode_calls": self.vectorized_encode_calls,
            "cached_static_events": self.cached_static_events,
            "dynamic_time_events": self.dynamic_time_events,
            "appended_events": self.appended_events,
        }


@dataclass
class PromptContextScoutState:
    stats: PromptContextStats = field(default_factory=PromptContextStats)
    summaries: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        if len(self.summaries) != 1:
            raise RuntimeError(
                "prompt-context scout expected one completed main-generation summary"
            )
        summary = self.summaries[0]
        if summary != self.stats.as_dict():
            raise RuntimeError("prompt-context scout summary changed after completion")
        return dict(summary)


class _ExactPromptContextSession:
    def __init__(self, processor: Any, stats: PromptContextStats):
        if processor.parallel:
            raise ValueError("prompt-context scout requires sequential generation")
        if processor.inference_runtime is None:
            raise ValueError("prompt-context scout requires the optimized runtime")
        if float(processor.cfg_scale) != 1.0:
            raise ValueError("prompt-context scout requires cfg_scale=1")
        if int(processor.num_beams) != 1:
            raise ValueError("prompt-context scout requires num_beams=1")
        if int(processor.tgt_seq_len) <= 1:
            raise ValueError("prompt-context scout requires a positive prompt capacity")
        self.processor = processor
        self.stats = stats
        self._sorted_times: dict[int, _SortedTimesState] = {}
        self._context_encodings: dict[int, _ContextEncodingState] = {}
        self._token_constants: dict[int, torch.Tensor] = {}
        self._empty_tokens = torch.empty((1, 0), dtype=torch.long)
        self._prompt_buffer = torch.empty(
            (1, int(processor.tgt_seq_len) * 2),
            dtype=torch.long,
        )
        self.stats.prompt_buffer_capacity = int(self._prompt_buffer.shape[1])
        self._last_prompt: torch.Tensor | None = None
        self._original_add_predicted = type(
            processor
        ).add_predicted_tokens_to_context

    @staticmethod
    def _require_finite_time(value: Any, *, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite number")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    def _validate_span(
        self,
        values: list[float],
        *,
        start_index: int,
        full: bool,
    ) -> None:
        if full:
            self.stats.full_order_validations += 1
        else:
            self.stats.incremental_order_validations += 1
        previous: float | None = None
        if start_index > 0:
            previous = self._require_finite_time(
                values[start_index - 1],
                name="event_times value",
            )
        for index in range(start_index, len(values)):
            current = self._require_finite_time(
                values[index],
                name="event_times value",
            )
            if previous is not None and current < previous:
                raise ValueError("event_times must be sorted in nondecreasing order")
            previous = current
            self.stats.validated_time_values += 1

    def _validate_sorted_times(self, values: Any) -> list[float]:
        if not isinstance(values, list):
            raise TypeError("prompt-context scout requires event_times to be a list")
        key = id(values)
        state = self._sorted_times.get(key)
        length = len(values)
        first = values[0] if length else None
        last = values[-1] if length else None
        if state is None or state.values is not values:
            self._validate_span(values, start_index=0, full=True)
            self._sorted_times[key] = _SortedTimesState(
                values=values,
                length=length,
                first=first,
                last=last,
            )
            return values

        unchanged = (
            length == state.length
            and first == state.first
            and last == state.last
        )
        if unchanged:
            return values
        if length > state.length and first == state.first:
            self._validate_span(
                values,
                start_index=max(0, state.length),
                full=False,
            )
        else:
            self._validate_span(values, start_index=0, full=True)
        state.length = length
        state.first = first
        state.last = last
        return values

    def get_events_time_range(
        self,
        _processor: Any,
        event_times: list[float],
        start_time: float,
        end_time: float,
    ) -> tuple[int, int]:
        values = self._validate_sorted_times(event_times)
        start = self._require_finite_time(start_time, name="start_time")
        end = self._require_finite_time(end_time, name="end_time")
        if end < start:
            raise ValueError("event time range end must not precede start")
        self.stats.range_queries += 1
        return bisect_left(values, start), bisect_left(values, end)

    @staticmethod
    def _event_signature(event: Any) -> tuple[Any, Any]:
        if not hasattr(event, "type") or not hasattr(event, "value"):
            raise TypeError("context events must expose type and value")
        return event.type, event.value

    def _encode_static_event(self, event: Any) -> int | None:
        if event.type == EventType.TIME_SHIFT:
            return None
        token = self.processor.tokenizer.encode(event)
        if isinstance(token, bool) or not isinstance(token, int):
            raise TypeError("tokenizer.encode must return an integer")
        self.stats.cached_static_events += 1
        return token

    def _new_encoding_state(
        self,
        context: dict[str, Any],
    ) -> _ContextEncodingState:
        events = context.get("events")
        event_times = context.get("event_times")
        if not isinstance(events, list) or not isinstance(event_times, list):
            raise TypeError("prompt-context scout requires event and time lists")
        if len(events) != len(event_times):
            raise ValueError("context events and event_times lengths differ")
        self._validate_sorted_times(event_times)
        encoded = [self._encode_static_event(event) for event in events]
        state = _ContextEncodingState(
            context=context,
            events=events,
            event_times=event_times,
            encoded_tokens=encoded,
            length=len(events),
            first_signature=(self._event_signature(events[0]) if events else None),
            last_signature=(self._event_signature(events[-1]) if events else None),
        )
        self._context_encodings[id(context)] = state
        return state

    def _encoding_state(
        self,
        context: dict[str, Any],
    ) -> _ContextEncodingState:
        state = self._context_encodings.get(id(context))
        if state is None or state.context is not context:
            return self._new_encoding_state(context)
        events = context.get("events")
        event_times = context.get("event_times")
        if events is not state.events or event_times is not state.event_times:
            raise RuntimeError("context replaced event-list storage during generation")
        if len(events) != state.length or len(event_times) != state.length:
            raise RuntimeError("context changed without prompt-context ownership")
        first = self._event_signature(events[0]) if events else None
        last = self._event_signature(events[-1]) if events else None
        if first != state.first_signature or last != state.last_signature:
            raise RuntimeError("context mutated cached events without ownership")
        return state

    def _encode_context_slice(
        self,
        context: dict[str, Any],
        start: int,
        end: int,
        frame_time: float,
    ) -> torch.Tensor:
        state = self._encoding_state(context)
        if not 0 <= start <= end <= state.length:
            raise ValueError("context token slice is out of range")
        timeshift_range = self.processor.tokenizer.event_range[EventType.TIME_SHIFT]
        timeshift_start = self.processor.tokenizer.event_start[EventType.TIME_SHIFT]
        values: list[int] = []
        for index in range(start, end):
            cached = state.encoded_tokens[index]
            if cached is not None:
                values.append(cached)
                continue
            event = state.events[index]
            relative = int(
                (event.value - frame_time) / MILISECONDS_PER_STEP
            )
            relative = max(
                int(timeshift_range.min_value),
                min(int(timeshift_range.max_value), relative),
            )
            values.append(
                int(timeshift_start)
                + relative
                - int(timeshift_range.min_value)
            )
            self.stats.dynamic_time_events += 1
        self.stats.vectorized_encode_calls += 1
        return torch.tensor([values], dtype=torch.long)

    def prepare_context_sequence(
        self,
        _processor: Any,
        context: dict[str, Any],
        frame_time: float,
    ) -> dict[str, Any]:
        state = self._encoding_state(context)
        result = context.copy()
        result["frame_time"] = frame_time
        if context["add_pre_tokens"]:
            start, end = self.get_events_time_range(
                self.processor,
                state.event_times,
                frame_time - self.processor.miliseconds_per_sequence,
                frame_time,
            )
            pre_tokens = self._encode_context_slice(
                context,
                start,
                end,
                frame_time,
            )
            if (
                0 <= self.processor.max_pre_token_len < pre_tokens.shape[1]
            ):
                pre_tokens = pre_tokens[:, -self.processor.max_pre_token_len :]
            result["pre_tokens"] = pre_tokens

        start, end = self.get_events_time_range(
            self.processor,
            state.event_times,
            frame_time,
            frame_time + self.processor.miliseconds_per_sequence,
        )
        result["tokens"] = self._encode_context_slice(
            context,
            start,
            end,
            frame_time,
        )

        extra_special_events = {}
        context_type = context["context_type"]
        if self.processor.add_kiai_special_token and (
            context_type == ContextType.KIAI
            or (
                self.processor.add_kiai
                and context_type in (ContextType.GD, ContextType.MAP)
            )
        ):
            extra_special_events["last_kiai"] = self.processor._kiai_before_time(
                state.events,
                state.event_times,
                frame_time,
            )
        if self.processor.add_sv_special_token and (
            context_type == ContextType.SV
            or (
                (self.processor.add_sv or self.processor.add_mania_sv)
                and context_type in (ContextType.GD, ContextType.MAP)
            )
        ):
            extra_special_events["last_sv"] = self.processor._sv_before_time(
                state.events,
                state.event_times,
                frame_time,
            )
        if self.processor.add_song_position_token and "class" in context:
            extra_special_events["song_position"] = (
                self.processor.tokenizer.encode_song_position_event(
                    frame_time,
                    context["song_length"],
                )
            )
        result["extra_special_events"] = extra_special_events
        return result

    def add_predicted_tokens_to_context(
        self,
        _processor: Any,
        context: dict[str, Any],
        predicted_tokens: torch.Tensor,
        frame_time: float,
        trim_lookback: bool = False,
        trim_lookahead: bool = False,
    ) -> None:
        state = self._encoding_state(context)
        old_length = state.length
        old_last = state.last_signature
        self._original_add_predicted(
            self.processor,
            context,
            predicted_tokens,
            frame_time,
            trim_lookback,
            trim_lookahead,
        )
        if (
            context.get("events") is not state.events
            or context.get("event_times") is not state.event_times
        ):
            raise RuntimeError("predicted-token update replaced context storage")
        if len(state.events) != len(state.event_times):
            raise RuntimeError("predicted-token update desynchronized event times")
        if len(state.events) < old_length:
            raise RuntimeError("predicted-token update removed an existing event")
        if (
            old_length
            and self._event_signature(state.events[old_length - 1]) != old_last
        ):
            raise RuntimeError("predicted-token update changed an existing event")
        appended = state.events[old_length:]
        state.encoded_tokens.extend(
            self._encode_static_event(event) for event in appended
        )
        self.stats.appended_events += len(appended)
        state.length = len(state.events)
        state.first_signature = (
            self._event_signature(state.events[0]) if state.events else None
        )
        state.last_signature = (
            self._event_signature(state.events[-1]) if state.events else None
        )

    @staticmethod
    def _require_piece(value: Any, *, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if value.device.type != "cpu" or value.dtype != torch.long:
            raise TypeError(f"{name} must be a CPU int64 tensor")
        if value.ndim != 2 or value.shape[0] != 1:
            raise ValueError(f"{name} must have shape [1, N]")
        return value

    def _constant_token(self, token: Any) -> torch.Tensor:
        if isinstance(token, bool) or not isinstance(token, int):
            raise TypeError("prompt token IDs must be integers")
        result = self._token_constants.get(token)
        if result is None:
            result = torch.tensor([[token]], dtype=torch.long)
            self._token_constants[token] = result
        return result

    def _context_pieces(
        self,
        context: dict[str, Any],
        *,
        max_token_length: int | None,
        add_type_end: bool,
    ) -> list[torch.Tensor]:
        if not isinstance(context, dict):
            raise TypeError("prepared context must be a dict")
        context_type = context.get("context_type")
        tokens = self._require_piece(context.get("tokens"), name="context tokens")
        if max_token_length is not None and tokens.shape[1] > max_token_length:
            tokens = tokens[:, -max_token_length:]
        pieces: list[torch.Tensor] = []
        if context.get("add_type"):
            try:
                type_token = self.processor.tokenizer.context_sos[context_type]
            except KeyError as exc:
                raise ValueError("context type has no SOS token") from exc
            pieces.append(self._constant_token(type_token))
        if context.get("add_class"):
            if "class" in context:
                pieces.append(
                    self._require_piece(context["class"], name="context class")
                )
            if "extra_special_tokens" in context:
                pieces.append(
                    self._require_piece(
                        context["extra_special_tokens"],
                        name="extra special tokens",
                    )
                )
        pieces.append(tokens)
        if context.get("add_type") and add_type_end:
            try:
                type_token = self.processor.tokenizer.context_eos[context_type]
            except KeyError as exc:
                raise ValueError("context type has no EOS token") from exc
            pieces.append(self._constant_token(type_token))
        return pieces

    def _ensure_capacity(self, required: int) -> None:
        if required <= self._prompt_buffer.shape[1]:
            return
        capacity = int(self._prompt_buffer.shape[1])
        while capacity < required:
            capacity *= 2
        self._prompt_buffer = torch.empty((1, capacity), dtype=torch.long)
        self.stats.prompt_buffer_resizes += 1
        self.stats.prompt_buffer_capacity = capacity

    def _write_piece(
        self,
        piece: torch.Tensor,
        *,
        position: int,
        skip: int = 0,
    ) -> int:
        width = int(piece.shape[1])
        if skip >= width:
            return position
        source = piece[:, skip:]
        next_position = position + int(source.shape[1])
        self._prompt_buffer[:, position:next_position].copy_(source)
        return next_position

    def _assemble_prompt(
        self,
        in_context: list[dict[str, Any]],
        out_context: list[dict[str, Any]],
        *,
        max_token_length: int | None,
    ) -> torch.Tensor:
        if not out_context:
            raise ValueError("prompt-context scout requires an output context")
        class_container = out_context[0]
        user_prompt = self._require_piece(
            class_container.get("class"),
            name="conditional class",
        )
        extra = self._require_piece(
            class_container.get(
                "extra_special_tokens",
                self._empty_tokens,
            ),
            name="output extra special tokens",
        )
        pre_tokens = self._require_piece(
            class_container.get(
                "pre_tokens",
                self._empty_tokens,
            ),
            name="output pre tokens",
        )
        if max_token_length is not None:
            pre_tokens = pre_tokens[:, -max_token_length:]

        prefix_pieces: list[torch.Tensor] = []
        for context in in_context:
            prefix_pieces.extend(
                self._context_pieces(
                    context,
                    max_token_length=max_token_length,
                    add_type_end=True,
                )
            )
        prefix_pieces.extend((user_prompt, extra, pre_tokens))
        output_pieces: list[torch.Tensor] = []
        for index, context in enumerate(out_context):
            output_pieces.extend(
                self._context_pieces(
                    context,
                    max_token_length=max_token_length,
                    add_type_end=index != len(out_context) - 1,
                )
            )

        prefix_width = sum(int(piece.shape[1]) for piece in prefix_pieces)
        if self.processor.center_pad_decoder:
            centered_width = int(self.processor.tgt_seq_len) // 2
            prefix_output_width = centered_width
        else:
            centered_width = 0
            prefix_output_width = prefix_width
        output_width = sum(int(piece.shape[1]) for piece in output_pieces)
        required = prefix_output_width + 1 + output_width
        self._ensure_capacity(required)

        position = 0
        if self.processor.center_pad_decoder and prefix_width < centered_width:
            padding = centered_width - prefix_width
            self._prompt_buffer[:, :padding].fill_(
                int(self.processor.tokenizer.pad_id)
            )
            position = padding
            self.stats.padding_tokens_written += padding
            prefix_skip = 0
        elif self.processor.center_pad_decoder:
            prefix_skip = prefix_width - centered_width
        else:
            prefix_skip = 0
        for piece in prefix_pieces:
            width = int(piece.shape[1])
            skip = min(prefix_skip, width)
            prefix_skip -= skip
            position = self._write_piece(piece, position=position, skip=skip)
        if prefix_skip != 0 or position != prefix_output_width:
            raise RuntimeError("prompt prefix assembly did not reconcile")
        self._prompt_buffer[0, position] = int(self.processor.tokenizer.sos_id)
        position += 1
        for piece in output_pieces:
            position = self._write_piece(piece, position=position)
        if position != required:
            raise RuntimeError("prompt assembly width did not reconcile")
        self.stats.prompt_tokens_written += required
        return self._prompt_buffer[:, :required]

    def get_prompts(
        self,
        _processor: Any,
        in_context: list[dict[str, Any]],
        out_context: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, None]:
        self.stats.prompt_calls += 1
        max_length = int(self.processor.tgt_seq_len)
        for retry in range(11):
            prompt = self._assemble_prompt(
                in_context,
                out_context,
                max_token_length=None if retry == 0 else max_length,
            )
            if prompt.shape[1] < self.processor.tgt_seq_len:
                self.stats.prompt_retries += retry
                self._last_prompt = prompt
                return prompt, None
            max_length //= 2
        raise ValueError("Prompt is too long.")

    def pad_prompts(
        self,
        _processor: Any,
        prompts: list[torch.Tensor | None],
    ) -> tuple[list[torch.Tensor | None], int]:
        if len(prompts) != 2 or prompts[1] is not None:
            raise ValueError("prompt-context scout supports cfg_scale=1 only")
        prompt = prompts[0]
        if prompt is None or prompt is not self._last_prompt:
            raise RuntimeError("prompt-context scout lost prompt-buffer ownership")
        return prompts, int(prompt.shape[1])

    @contextmanager
    def install(self) -> Iterator[None]:
        names = (
            "_get_events_time_range",
            "prepare_context_sequence",
            "get_prompts",
            "pad_prompts",
            "add_predicted_tokens_to_context",
        )
        if any(name in self.processor.__dict__ for name in names):
            raise RuntimeError("prompt-context scout refuses existing instance patches")
        self.processor._get_events_time_range = MethodType(  # type: ignore[attr-defined]
            self.get_events_time_range,
            self.processor,
        )
        self.processor.get_prompts = MethodType(  # type: ignore[method-assign]
            self.get_prompts,
            self.processor,
        )
        self.processor.pad_prompts = MethodType(  # type: ignore[method-assign]
            self.pad_prompts,
            self.processor,
        )
        self.processor.prepare_context_sequence = MethodType(  # type: ignore[method-assign]
            self.prepare_context_sequence,
            self.processor,
        )
        self.processor.add_predicted_tokens_to_context = MethodType(  # type: ignore[method-assign]
            self.add_predicted_tokens_to_context,
            self.processor,
        )
        try:
            yield
        finally:
            for name in names:
                del self.processor.__dict__[name]


@contextmanager
def install_exact_prompt_context_preparation() -> Iterator[PromptContextScoutState]:
    """Install the exact prompt candidate only around labeled main generation."""

    from ...processor import Processor

    if "generate" in Processor.__dict__ and getattr(
        Processor.generate,
        "_prompt_context_scout",
        False,
    ):
        raise RuntimeError("prompt-context scout cannot be nested")
    original_generate = Processor.generate
    state = PromptContextScoutState()

    def generate_with_exact_prompt_context(processor, *args, **kwargs):
        profile_label = kwargs.get("profile_label")
        if profile_label != "main_generation":
            return original_generate(processor, *args, **kwargs)
        if state.stats.main_generate_calls != 0:
            raise RuntimeError("prompt-context scout observed repeated main generation")
        state.stats.main_generate_calls += 1
        session = _ExactPromptContextSession(processor, state.stats)
        with session.install():
            result = original_generate(processor, *args, **kwargs)
        summary = state.stats.as_dict()
        state.summaries.append(summary)
        processor.profiler.set_metadata(prompt_context_preparation=summary)
        return result

    generate_with_exact_prompt_context._prompt_context_scout = True  # type: ignore[attr-defined]
    Processor.generate = generate_with_exact_prompt_context
    try:
        yield state
    finally:
        Processor.generate = original_generate


__all__ = [
    "PromptContextScoutState",
    "PromptContextStats",
    "SCOUT_VERSION",
    "install_exact_prompt_context_preparation",
]
