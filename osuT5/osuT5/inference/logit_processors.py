import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

from osuT5.osuT5.dataset.data_utils import TIMED_EVENTS
from osuT5.osuT5.event import EventType, Event
from osuT5.osuT5.tokenizer import MILISECONDS_PER_STEP, Tokenizer

MILISECONDS_PER_SECOND = 1000


def get_beat_type_tokens(tokenizer: Tokenizer) -> tuple[int, ...]:
    beat_range = [
        tokenizer.event_start[EventType.BEAT],
        tokenizer.event_start[EventType.MEASURE],
    ]
    if EventType.TIMING_POINT in tokenizer.event_start:
        beat_range.append(tokenizer.event_start[EventType.TIMING_POINT])
    return tuple(beat_range)


def get_mania_type_tokens(tokenizer: Tokenizer) -> tuple[int, ...]:
    return tuple([
        tokenizer.event_start[EventType.CIRCLE],
        tokenizer.event_start[EventType.HOLD_NOTE],
        tokenizer.event_start[EventType.HOLD_NOTE_END],
    ] if EventType.HOLD_NOTE_END in tokenizer.event_start else [])


def get_scroll_speed_tokens(tokenizer: Tokenizer) -> tuple[int, ...]:
    return tuple(range(tokenizer.event_start[EventType.SCROLL_SPEED], tokenizer.event_end[EventType.SCROLL_SPEED])
                 if EventType.SCROLL_SPEED in tokenizer.event_start else [])


class TimeshiftBias(LogitsProcessor):
    def __init__(self, timeshift_bias: float, time_start: int, time_end: int):
        self.timeshift_bias = timeshift_bias
        self.time_range = slice(time_start, time_end)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        scores_processed = scores.clone()
        scores_processed[:, self.time_range] += self.timeshift_bias
        return scores_processed


class ConditionalTemperatureLogitsWarper(LogitsProcessor):
    def __init__(
            self,
            temperature: float,
            timing_temperature: float,
            mania_column_temperature: float,
            taiko_hit_temperature: float,
            types_first: bool,
            beat_type_tokens: tuple[int, ...],
            mania_type_tokens: tuple[int, ...],
            scroll_speed_tokens: tuple[int, ...],
    ):
        self.temperature = temperature
        self.conditionals = []

        if timing_temperature != temperature and len(beat_type_tokens) > 0:
            self.conditionals.append((timing_temperature, beat_type_tokens, 1))
        if mania_column_temperature != temperature and len(mania_type_tokens) > 0:
            self.conditionals.append((mania_column_temperature, mania_type_tokens, 3))
        if taiko_hit_temperature != temperature and len(scroll_speed_tokens) > 0:
            self.conditionals.append((taiko_hit_temperature, scroll_speed_tokens, 1))

        if not types_first:
            print("WARNING: Conditional temperature is not supported for types_first=False. Ignoring.")
            self.conditionals = []

        self.max_offset = max([offset for _, _, offset in self.conditionals], default=0)
        self._tokens_by_device: dict[tuple[int, torch.device], torch.Tensor] = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        if len(self.conditionals) > 0 and input_ids.shape[0] == 1:
            lookback = input_ids[0, -self.max_offset:].cpu()
            for temperature, tokens, offset in self.conditionals:
                if len(lookback) >= offset and lookback[-offset] in tokens:
                    return scores / temperature

        if len(self.conditionals) > 0 and input_ids.shape[0] > 1:
            temperatures = torch.full(
                (scores.shape[0],),
                self.temperature,
                dtype=scores.dtype,
                device=scores.device,
            )
            assigned = torch.zeros(
                scores.shape[0],
                dtype=torch.bool,
                device=input_ids.device,
            )
            for index, (temperature, tokens, offset) in enumerate(self.conditionals):
                if input_ids.shape[1] < offset:
                    continue
                cache_key = (index, input_ids.device)
                token_tensor = self._tokens_by_device.get(cache_key)
                if token_tensor is None:
                    token_tensor = torch.tensor(
                        tokens,
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    )
                    self._tokens_by_device[cache_key] = token_tensor
                matches = torch.isin(input_ids[:, -offset], token_tensor) & ~assigned
                temperatures = torch.where(
                    matches,
                    torch.as_tensor(
                        temperature,
                        dtype=scores.dtype,
                        device=scores.device,
                    ),
                    temperatures,
                )
                assigned |= matches
            return scores / temperatures.unsqueeze(1)

        return scores / self.temperature


class LookbackBiasLogitsWarper(LogitsProcessor):
    """This logit processor adjusts for bias in the frequency of generated events in case there is a lookback window,
    and it is not full of generated tokens. In this case, the lookback window will be considered multiple times for
    generating the next token, so we nill the scores of the lookback tokens and increase the chance of eos.
    """
    def __init__(self, lookback_max_time: float, tokenizer: Tokenizer, types_first: bool, device):
        self.types_first = types_first  # Lookback bias is only supported for types_first=True
        self.lookback_start = tokenizer.event_start[EventType.TIME_SHIFT]
        self.lookback_end = tokenizer.encode(Event(EventType.TIME_SHIFT, int(lookback_max_time / MILISECONDS_PER_STEP)))
        self.lookback_range = torch.full((tokenizer.vocab_size_out,), False, dtype=torch.bool, device=device)
        self.lookback_range[self.lookback_start:self.lookback_end] = True
        self.other_range = ~self.lookback_range
        self.eos_ids = torch.tensor([tokenizer.eos_id] + [tokenizer.context_eos[context] for context in tokenizer.context_eos], dtype=torch.long, device=device)

        self.last_scores = None
        self.timed_tokens = []
        for event_type in TIMED_EVENTS:
            if event_type in tokenizer.event_start:
                self.timed_tokens.extend(list(range(tokenizer.event_start[event_type], tokenizer.event_end[event_type])))
        self.timed_tokens = torch.tensor(self.timed_tokens, dtype=torch.long, device=device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        if not self.types_first:
            scores_processed = scores.clone()
            scores_processed[:, self.lookback_range] = -torch.inf
            return scores_processed

        scores_processed = scores

        if input_ids.shape[1] != 0 and self.last_scores is not None:
            last_token = input_ids[:, -1]
            last_timed = torch.isin(last_token, self.timed_tokens)
            if last_timed.any():
                # The scores are for a timeshift event
                last_probs = F.softmax(self.last_scores, dim=-1)
                probs = F.softmax(scores, dim=-1)
                prob_eos = last_probs[:, self.eos_ids].sum(dim=-1)
                prob_event = 1 - prob_eos
                s = 1 / (probs[:, self.other_range].sum(dim=-1) * prob_event + prob_eos)
                probs[:, self.lookback_range] = 0
                probs[:, self.other_range] *= s.unsqueeze(1)
                # Probability of eos now which should have been at the previous token
                prob_eos_extra = torch.clip((s - 1) * prob_eos / prob_event, 0, 1)  # Clip to avoid numerical instability
                probs[:, self.lookback_start] = prob_eos_extra  # This will be treated as eos if trim lookback is true

                scores_processed = torch.where(last_timed.unsqueeze(1), torch.log(probs), scores)

        self.last_scores = scores
        return scores_processed


class MonotonicTimeShiftLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, *, stateful_batch1: bool = False):
        if stateful_batch1:
            raise ValueError(
                "stateful_batch1 moved to optimized single inference; use "
                "osuT5.osuT5.inference.optimized.single.logits."
            )
        self.tokenizer = tokenizer
        self.time_shift_start = tokenizer.event_start[EventType.TIME_SHIFT]
        self.time_shift_end = tokenizer.event_end[EventType.TIME_SHIFT]
        self.sos_ids = torch.tensor([tokenizer.sos_id] + list(getattr(tokenizer, "context_sos", {}).values()))
        self.stateful_batch1 = False
        self._sos_ids_by_device: dict[torch.device, torch.Tensor] = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return self._full_scan_call(input_ids, scores)

    def _full_scan_call(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # Create masks for time_shift and sos tokens
        is_time_shift = (input_ids >= self.time_shift_start) & (input_ids < self.time_shift_end)
        is_sos = self._is_sos(input_ids)

        # Create a sequence of indices [0, 1, ..., seq_len-1]
        indices = torch.arange(seq_len, device=device).expand(batch_size, -1)

        # Find the index of the last occurrence of each event type
        # If not found, it will be -1
        last_time_shift_idx = torch.max(torch.where(is_time_shift, indices, -1), dim=1).values
        last_sos_idx = torch.max(torch.where(is_sos, indices, -1), dim=1).values

        # Get the value of the last time_shift token for each sequence
        # If no time_shift token, value is 0
        last_time_shift_values = torch.where(
            last_time_shift_idx != -1,
            input_ids[torch.arange(batch_size), last_time_shift_idx] - self.time_shift_start,
            0
        )

        # Determine which sequences in the batch need masking
        # Mask if a time_shift token was found and it did not appear after the last SOS token
        apply_mask = (last_time_shift_idx != -1) & (last_time_shift_idx > last_sos_idx)

        # Create the mask for invalid time_shift tokens
        time_shift_vocab = torch.arange(self.time_shift_start, self.time_shift_end, device=device)
        # Shape: (batch_size, num_time_shift_tokens)
        invalid_mask = time_shift_vocab.unsqueeze(0) < (self.time_shift_start + last_time_shift_values).unsqueeze(1)

        # Apply the mask to the scores for the relevant sequences
        # Shape: (batch_size, vocab_size)
        batch_mask = torch.full_like(scores, False, dtype=torch.bool)
        batch_mask[:, self.time_shift_start:self.time_shift_end] = invalid_mask
        scores[apply_mask] = scores[apply_mask].masked_fill(batch_mask[apply_mask], -torch.inf)

        return scores

    def _is_sos(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return (input_ids.unsqueeze(-1) == self._sos_ids(input_ids.device)).any(dim=-1)

    def _sos_ids(self, device: torch.device) -> torch.Tensor:
        if device not in self._sos_ids_by_device:
            self._sos_ids_by_device[device] = self.sos_ids.to(device)
        return self._sos_ids_by_device[device]


def build_logits_processor_list(
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
    monotonic_processor_factory=MonotonicTimeShiftLogitsProcessor,
):
    """Build the shared processor chain with a runtime-owned monotonic policy."""

    from transformers import (
        ClassifierFreeGuidanceLogitsProcessor,
        LogitsProcessorList,
        TemperatureLogitsWarper,
    )

    timing_temperature = temperature if timing_temperature is None else timing_temperature
    mania_column_temperature = (
        temperature if mania_column_temperature is None else mania_column_temperature
    )
    taiko_hit_temperature = (
        temperature if taiko_hit_temperature is None else taiko_hit_temperature
    )
    device = device if device is not None else getattr(tokenizer, "device", None)

    processors = LogitsProcessorList()
    if cfg_scale > 1.0:
        processors.append(ClassifierFreeGuidanceLogitsProcessor(cfg_scale))
    processors.append(monotonic_processor_factory(tokenizer))
    if timeshift_bias != 0:
        processors.append(
            TimeshiftBias(
                timeshift_bias,
                tokenizer.event_start[EventType.TIME_SHIFT],
                tokenizer.event_end[EventType.TIME_SHIFT],
            )
        )
    if types_first:
        processors.append(
            ConditionalTemperatureLogitsWarper(
                temperature,
                timing_temperature,
                mania_column_temperature,
                taiko_hit_temperature,
                types_first,
                get_beat_type_tokens(tokenizer),
                get_mania_type_tokens(tokenizer),
                get_scroll_speed_tokens(tokenizer),
            )
        )
    else:
        processors.append(TemperatureLogitsWarper(temperature))
    if lookback_time > 0:
        processors.append(
            LookbackBiasLogitsWarper(
                lookback_time,
                tokenizer,
                types_first,
                device,
            )
        )
    return processors
