from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers.modeling_outputs import BaseModelOutput

from osuT5.osuT5.inference.cache_utils import MapperatorinatorCache, get_cache
from osuT5.osuT5.runtime_profiling import active_prefix_self_attention_context


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


@dataclass
class EncoderState:
    encoder_outputs: BaseModelOutput


@dataclass
class DecoderCacheState:
    cache: MapperatorinatorCache
    prompt_length: int
    prefill_cache_position: torch.LongTensor
    prefill_prepared_inputs: dict[str, Any]


@dataclass
class StaticInputBuffers:
    prompt: torch.LongTensor
    prompt_attention_mask: torch.Tensor
    frames: torch.Tensor


@dataclass
class SamplerState:
    logits_processor: Any | None = None
    stopping_criteria: Any | None = None
    rng_state: torch.Tensor | None = None


@dataclass
class GraphState:
    graphs: dict[tuple[Any, ...], Any]


@dataclass
class DecodeSession:
    """Verifier-first ABI for future single-song decode runtime work.

    This wrapper intentionally delegates to the existing exact one-token helpers.
    It gives runtime/kernel experiments stable ownership boundaries without
    changing production inference behavior.
    """

    model: Any
    encoder_state: EncoderState
    cache_state: DecoderCacheState
    static_inputs: StaticInputBuffers
    condition_kwargs: dict[str, Any]
    sampler_state: SamplerState
    graph_state: GraphState

    @classmethod
    @torch.no_grad()
    def prefill(
            cls,
            model,
            *,
            prompt: torch.LongTensor,
            prompt_attention_mask: torch.Tensor,
            frames: torch.Tensor,
            condition_kwargs: dict[str, Any] | None = None,
            active_prefix_self_attention: bool = False,
    ) -> "DecodeSession":
        condition_kwargs = condition_kwargs or {}
        state = prefill_static_cache(
            model,
            prompt=prompt,
            prompt_attention_mask=prompt_attention_mask,
            frames=frames,
            condition_kwargs=condition_kwargs,
            active_prefix_self_attention=active_prefix_self_attention,
        )
        return cls(
            model=model,
            encoder_state=EncoderState(encoder_outputs=state.encoder_outputs),
            cache_state=DecoderCacheState(
                cache=state.cache,
                prompt_length=state.prompt_length,
                prefill_cache_position=state.prefill_cache_position,
                prefill_prepared_inputs=state.prefill_prepared_inputs,
            ),
            static_inputs=StaticInputBuffers(
                prompt=prompt,
                prompt_attention_mask=prompt_attention_mask,
                frames=frames,
            ),
            condition_kwargs=condition_kwargs,
            sampler_state=SamplerState(),
            graph_state=GraphState(graphs={}),
        )

    def _one_token_state(self) -> OneTokenDecodeState:
        return OneTokenDecodeState(
            cache=self.cache_state.cache,
            encoder_outputs=self.encoder_state.encoder_outputs,
            prompt_length=self.cache_state.prompt_length,
            prefill_cache_position=self.cache_state.prefill_cache_position,
            prefill_prepared_inputs=self.cache_state.prefill_prepared_inputs,
        )

    def one_token_state(self) -> OneTokenDecodeState:
        return self._one_token_state()

    @torch.no_grad()
    def decode_one_token_raw_logits(
            self,
            *,
            full_prefix: torch.LongTensor,
            full_attention_mask: torch.Tensor,
            cache_position: torch.LongTensor | None = None,
            active_prefix_self_attention: bool = False,
            active_prefix_self_attention_length: int | None = None,
    ) -> OneTokenDecodeResult:
        return decode_one_token_raw_logits(
            self.model,
            self._one_token_state(),
            full_prefix=full_prefix,
            full_attention_mask=full_attention_mask,
            condition_kwargs=self.condition_kwargs,
            cache_position=cache_position,
            active_prefix_self_attention=active_prefix_self_attention,
            active_prefix_self_attention_length=active_prefix_self_attention_length,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "prompt_length": int(self.cache_state.prompt_length),
            "graph_count": len(self.graph_state.graphs),
            "prefill_cache_position_shape": list(self.cache_state.prefill_cache_position.shape),
            "prompt_shape": list(self.static_inputs.prompt.shape),
            "prompt_attention_mask_shape": list(self.static_inputs.prompt_attention_mask.shape),
            "frames_shape": list(self.static_inputs.frames.shape),
        }


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
        active_prefix_self_attention: bool = False,
) -> OneTokenDecodeState:
    """Run the exact production prefill path into a fresh static cache."""
    condition_kwargs = condition_kwargs or {}
    prompt_length = int(prompt.shape[-1])
    cache = get_cache(model, batch_size=batch_size, num_beams=1, cfg_scale=1.0)
    cache_position = torch.arange(prompt_length, device=prompt.device)
    with active_prefix_self_attention_context(prompt_length if active_prefix_self_attention else None):
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
        active_prefix_self_attention: bool = False,
        active_prefix_self_attention_length: int | None = None,
) -> OneTokenDecodeResult:
    """Run one cached decoder step and return raw last-token logits."""
    condition_kwargs = condition_kwargs or {}
    if cache_position is None:
        cache_position = torch.arange(
            state.prompt_length,
            state.prompt_length + 1,
            device=full_prefix.device,
        )
    prefix_length = active_prefix_self_attention_length
    if prefix_length is None and active_prefix_self_attention:
        prefix_length = int(cache_position[-1].detach().cpu().item()) + 1
    with active_prefix_self_attention_context(prefix_length if active_prefix_self_attention else None):
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
