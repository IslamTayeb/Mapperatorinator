from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers.modeling_outputs import BaseModelOutput

from osuT5.osuT5.inference.cache_utils import MapperatorinatorCache, get_cache


@dataclass
class OneTokenDecodeState:
    cache: MapperatorinatorCache
    encoder_outputs: BaseModelOutput
    prompt_length: int
    prefill_cache_position: torch.LongTensor
    prefill_prepared_inputs: dict[str, Any]


@dataclass
class OneTokenDecodeResult:
    logits: torch.Tensor
    cache_position: torch.LongTensor
    prepared_inputs: dict[str, Any]


def last_token_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, -1, :].detach().to(torch.float32)


@torch.no_grad()
def prefill_static_cache(
        model,
        *,
        prompt: torch.LongTensor,
        prompt_attention_mask: torch.Tensor,
        frames: torch.Tensor,
        condition_kwargs: dict[str, Any] | None = None,
        batch_size: int = 1,
) -> OneTokenDecodeState:
    """Run the exact production prefill path into a fresh static cache."""
    condition_kwargs = condition_kwargs or {}
    prompt_length = int(prompt.shape[-1])
    cache = get_cache(model, batch_size=batch_size, num_beams=1, cfg_scale=1.0)
    cache_position = torch.arange(prompt_length, device=prompt.device)
    prepared_inputs = model.prepare_inputs_for_generation(
        prompt,
        past_key_values=cache,
        use_cache=True,
        decoder_attention_mask=prompt_attention_mask,
        cache_position=cache_position,
        frames=frames,
        **condition_kwargs,
    )
    outputs = model(**prepared_inputs)
    encoder_outputs = BaseModelOutput(last_hidden_state=outputs.encoder_last_hidden_state)
    return OneTokenDecodeState(
        cache=cache,
        encoder_outputs=encoder_outputs,
        prompt_length=prompt_length,
        prefill_cache_position=cache_position,
        prefill_prepared_inputs=prepared_inputs,
    )


@torch.no_grad()
def decode_one_token_raw_logits(
        model,
        state: OneTokenDecodeState,
        *,
        full_prefix: torch.LongTensor,
        full_attention_mask: torch.Tensor,
        condition_kwargs: dict[str, Any] | None = None,
        cache_position: torch.LongTensor | None = None,
) -> OneTokenDecodeResult:
    """Run one cached decoder step and return raw last-token logits."""
    condition_kwargs = condition_kwargs or {}
    if cache_position is None:
        cache_position = torch.arange(
            state.prompt_length,
            state.prompt_length + 1,
            device=full_prefix.device,
        )
    prepared_inputs = model.prepare_inputs_for_generation(
        full_prefix,
        past_key_values=state.cache,
        use_cache=True,
        encoder_outputs=state.encoder_outputs,
        decoder_attention_mask=full_attention_mask,
        cache_position=cache_position,
        **condition_kwargs,
    )
    outputs = model(**prepared_inputs)
    return OneTokenDecodeResult(
        logits=last_token_logits(outputs.logits),
        cache_position=cache_position,
        prepared_inputs=prepared_inputs,
    )
