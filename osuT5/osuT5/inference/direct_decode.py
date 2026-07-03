from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers.cache_utils import EncoderDecoderCache, StaticCache
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
    prefill_logits: torch.Tensor | None = None


@dataclass
class OneTokenDecodeResult:
    logits: torch.Tensor
    cache_position: torch.LongTensor
    prepared_inputs: dict[str, Any]


@dataclass
class EncoderState:
    encoder_outputs: BaseModelOutput | None


@dataclass
class DecoderCacheState:
    cache: MapperatorinatorCache | None
    prompt_length: int
    prefill_cache_position: torch.LongTensor | None
    prefill_prepared_inputs: dict[str, Any]
    prefill_logits: torch.Tensor | None = None
    current_cache_position: torch.LongTensor | None = None


@dataclass
class StaticInputBuffers:
    prompt: torch.LongTensor
    prompt_attention_mask: torch.Tensor | None
    frames: torch.Tensor | None


@dataclass
class SamplerState:
    logits_processor: Any | None = None
    stopping_criteria: Any | None = None
    rng_state: torch.Tensor | None = None
    cuda_rng_state: list[torch.Tensor] | None = None


@dataclass
class GraphState:
    graphs: dict[tuple[Any, ...], Any]


@dataclass
class GeneratedTokenState:
    output_shape: list[int] | None = None
    generated_steps: int = 0


@dataclass
class DecodeLoopState:
    model_kwargs: dict[str, Any]
    cur_len: int
    max_cache_len: int | None
    sequence_buffer_shape: list[int] | None = None
    generated_steps: int = 0


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
    generated_state: GeneratedTokenState = field(default_factory=GeneratedTokenState)
    loop_state: DecodeLoopState | None = None

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
                prefill_logits=state.prefill_logits,
                current_cache_position=state.prefill_cache_position,
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

    @classmethod
    def for_generation_loop(
            cls,
            model,
            *,
            prompt: torch.LongTensor,
            prompt_attention_mask: torch.Tensor | None,
            frames: torch.Tensor | None,
            model_kwargs: dict[str, Any],
            max_cache_len: int | None,
            sequence_buffer: torch.Tensor | None = None,
            condition_kwargs: dict[str, Any] | None = None,
            sampler_state: SamplerState | None = None,
    ) -> "DecodeSession":
        """Create a verifier-only session wrapper around an HF-style decode loop.

        This path intentionally does not change computation. It gives the direct
        loop verifier a stable state owner before future runtime work moves
        individual operations behind the session boundary.
        """
        cache_position = model_kwargs.get("cache_position")
        return cls(
            model=model,
            encoder_state=EncoderState(encoder_outputs=model_kwargs.get("encoder_outputs")),
            cache_state=DecoderCacheState(
                cache=model_kwargs.get("past_key_values"),
                prompt_length=int(prompt.shape[-1]),
                prefill_cache_position=cache_position if isinstance(cache_position, torch.Tensor) else None,
                prefill_prepared_inputs={},
                current_cache_position=cache_position if isinstance(cache_position, torch.Tensor) else None,
            ),
            static_inputs=StaticInputBuffers(
                prompt=prompt,
                prompt_attention_mask=prompt_attention_mask,
                frames=frames,
            ),
            condition_kwargs=condition_kwargs or {},
            sampler_state=sampler_state or SamplerState(),
            graph_state=GraphState(graphs={}),
            loop_state=DecodeLoopState(
                model_kwargs=model_kwargs,
                cur_len=int(prompt.shape[-1]),
                max_cache_len=max_cache_len,
                sequence_buffer_shape=(
                    list(sequence_buffer.shape)
                    if isinstance(sequence_buffer, torch.Tensor)
                    else None
                ),
            ),
        )

    def _one_token_state(self) -> OneTokenDecodeState:
        if self.cache_state.cache is None:
            raise RuntimeError("DecodeSession does not own a static cache for one-token decode")
        if self.encoder_state.encoder_outputs is None:
            raise RuntimeError("DecodeSession does not own encoder outputs for one-token decode")
        if self.cache_state.prefill_cache_position is None:
            raise RuntimeError("DecodeSession does not own a prefill cache position")
        return OneTokenDecodeState(
            cache=self.cache_state.cache,
            encoder_outputs=self.encoder_state.encoder_outputs,
            prompt_length=self.cache_state.prompt_length,
            prefill_cache_position=self.cache_state.prefill_cache_position,
            prefill_prepared_inputs=self.cache_state.prefill_prepared_inputs,
            prefill_logits=self.cache_state.prefill_logits,
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

    def record_generated_token(self, next_tokens: torch.LongTensor, output: torch.LongTensor) -> None:
        self.generated_state.output_shape = list(output.shape)
        self.generated_state.generated_steps += int(next_tokens.numel())
        if self.loop_state is not None:
            self.loop_state.generated_steps += int(next_tokens.numel())
            self.loop_state.cur_len = int(output.shape[-1])

    def update_loop_model_kwargs(self, model_kwargs: dict[str, Any]) -> None:
        if self.loop_state is not None:
            self.loop_state.model_kwargs = model_kwargs
            cache_position = model_kwargs.get("cache_position")
            if isinstance(cache_position, torch.Tensor):
                self.cache_state.current_cache_position = cache_position
            self.cache_state.cache = model_kwargs.get("past_key_values", self.cache_state.cache)
            encoder_outputs = model_kwargs.get("encoder_outputs")
            if encoder_outputs is not None:
                self.encoder_state.encoder_outputs = encoder_outputs

    def metadata(self) -> dict[str, Any]:
        loop_state = self.loop_state
        return {
            "prompt_length": int(self.cache_state.prompt_length),
            "graph_count": len(self.graph_state.graphs),
            "prefill_cache_position_shape": (
                list(self.cache_state.prefill_cache_position.shape)
                if isinstance(self.cache_state.prefill_cache_position, torch.Tensor)
                else None
            ),
            "current_cache_position_shape": (
                list(self.cache_state.current_cache_position.shape)
                if isinstance(self.cache_state.current_cache_position, torch.Tensor)
                else None
            ),
            "prompt_shape": list(self.static_inputs.prompt.shape),
            "prompt_attention_mask_shape": (
                list(self.static_inputs.prompt_attention_mask.shape)
                if isinstance(self.static_inputs.prompt_attention_mask, torch.Tensor)
                else None
            ),
            "frames_shape": (
                list(self.static_inputs.frames.shape)
                if isinstance(self.static_inputs.frames, torch.Tensor)
                else None
            ),
            "generated_steps": int(self.generated_state.generated_steps),
            "generated_token_count": int(self.generated_state.generated_steps),
            "output_shape": self.generated_state.output_shape,
            "sampler_state": {
                "has_logits_processor": self.sampler_state.logits_processor is not None,
                "has_stopping_criteria": self.sampler_state.stopping_criteria is not None,
                "rng_state_shape": (
                    list(self.sampler_state.rng_state.shape)
                    if isinstance(self.sampler_state.rng_state, torch.Tensor)
                    else None
                ),
                "cuda_rng_state_count": (
                    len(self.sampler_state.cuda_rng_state)
                    if self.sampler_state.cuda_rng_state is not None
                    else 0
                ),
            },
            "loop_state": (
                {
                    "cur_len": int(loop_state.cur_len),
                    "max_cache_len": loop_state.max_cache_len,
                    "sequence_buffer_shape": loop_state.sequence_buffer_shape,
                    "generated_steps": int(loop_state.generated_steps),
                    "model_kwargs_keys": sorted(loop_state.model_kwargs.keys()),
                }
                if loop_state is not None
                else None
            ),
        }


def last_token_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, -1, :].detach().to(dtype=torch.float32, copy=True)


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
        prefill_logits=last_token_logits(outputs.logits),
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


@torch.no_grad()
def prepare_one_token_decode_inputs_fast(
        model,
        input_ids: torch.LongTensor,
        model_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Verifier-only fast decode input builder for the simple batch-1 cached path.

    This mirrors VarWhisper.prepare_inputs_for_generation for post-prefill
    one-token decode. It intentionally keeps the full static-cache 4D mask
    construction for now, so verifier work can separate "can we prepare the
    same tensors?" from later incremental-mask/runtime experiments.
    """
    past_key_values = model_kwargs.get("past_key_values")
    cache_position = model_kwargs.get("cache_position")
    decoder_attention_mask = model_kwargs.get("decoder_attention_mask")
    use_cache = model_kwargs.get("use_cache")
    encoder_outputs = model_kwargs.get("encoder_outputs")

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("fast one-token decode input builder currently supports batch_size=1 only")
    if past_key_values is None:
        raise ValueError("fast one-token decode input builder requires past_key_values")
    if not isinstance(cache_position, torch.Tensor):
        raise ValueError("fast one-token decode input builder requires tensor cache_position")
    if cache_position.numel() < 1:
        raise ValueError("cache_position must contain at least one position")

    decoder_input_ids = input_ids[:, -1:].contiguous()
    next_cache_position = cache_position[-decoder_input_ids.shape[1]:]

    decoder_position_ids = None
    if decoder_attention_mask is not None:
        if decoder_attention_mask.ndim != 2:
            raise ValueError("fast one-token decode input builder expects a 2D decoder_attention_mask")
        decoder_position_ids = (decoder_attention_mask.cumsum(-1) - 1).clamp(min=0)
        decoder_position_ids = decoder_position_ids[:, -decoder_input_ids.shape[1]:]
        decoder_position_ids = decoder_position_ids.clone(memory_format=torch.contiguous_format)

    prepared_attention_mask = decoder_attention_mask
    if (
            isinstance(past_key_values, EncoderDecoderCache)
            and (
                isinstance(past_key_values.self_attention_cache, StaticCache)
                or isinstance(past_key_values.cross_attention_cache, StaticCache)
            )
            and decoder_attention_mask is not None
            and decoder_attention_mask.ndim == 2
    ):
        prepared_attention_mask = model.get_decoder()._prepare_4d_causal_attention_mask_with_cache_position(
            decoder_attention_mask,
            sequence_length=decoder_input_ids.shape[1],
            target_length=past_key_values.self_attention_cache.get_max_cache_shape(),
            dtype=model.proj_out.weight.dtype,
            device=decoder_input_ids.device,
            cache_position=next_cache_position,
            batch_size=decoder_input_ids.shape[0],
        )

    return {
        "encoder_outputs": encoder_outputs,
        "past_key_values": past_key_values,
        "decoder_input_ids": decoder_input_ids,
        "use_cache": use_cache,
        "decoder_attention_mask": prepared_attention_mask,
        "decoder_position_ids": decoder_position_ids,
        "cache_position": next_cache_position,
    }
