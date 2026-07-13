"""Small generation helpers shared by V32 and opt-in runtimes."""

from __future__ import annotations

from typing import Any

import torch

from ..event import ContextType, EventType
from ..tokenizer import MILISECONDS_PER_STEP


def build_generation_stats(
    result: torch.Tensor,
    model_kwargs: dict[str, Any],
    pad_token_id: int | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    prompt_counts = _prompt_token_counts(model_kwargs, pad_token_id)
    output_counts = _output_token_counts(result, pad_token_id).cpu()
    generated_counts = output_counts.clone()
    if prompt_counts is not None:
        generated_counts = torch.clamp(generated_counts - prompt_counts, min=0)

    generated_tokens = int(generated_counts.sum().item())
    return {
        "batch_size": int(result.shape[0]),
        "prompt_tokens": (
            int(prompt_counts.sum().item()) if prompt_counts is not None else None
        ),
        "prompt_tokens_per_sample": (
            prompt_counts.tolist() if prompt_counts is not None else None
        ),
        "output_tokens": int(output_counts.sum().item()),
        "output_tokens_per_sample": output_counts.tolist(),
        "generated_tokens": generated_tokens,
        "generated_tokens_per_sample": generated_counts.tolist(),
        "elapsed_seconds": float(elapsed_seconds),
        "tokens_per_second": (
            generated_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
        ),
    }


def eos_token_ids(
    tokenizer,
    *,
    lookback_time: float = 0,
    lookahead_time: float = 0,
    context_type: ContextType | None = None,
) -> list[int]:
    token_ids = [tokenizer.eos_id]
    if context_type is not None and context_type in tokenizer.context_eos:
        token_ids.append(tokenizer.context_eos[context_type])
    if lookback_time > 0:
        token_ids.extend(
            range(
                tokenizer.event_start[EventType.TIME_SHIFT],
                tokenizer.event_start[EventType.TIME_SHIFT]
                + int(lookback_time / MILISECONDS_PER_STEP),
            )
        )
    if lookahead_time > 0:
        token_ids.extend(
            range(
                tokenizer.event_end[EventType.TIME_SHIFT]
                - int(lookahead_time / MILISECONDS_PER_STEP),
                tokenizer.event_end[EventType.TIME_SHIFT],
            )
        )
    return token_ids


def sync_cuda_for_model(model) -> None:
    if (
        torch.cuda.is_available()
        and getattr(getattr(model, "device", None), "type", None) == "cuda"
    ):
        torch.cuda.synchronize(model.device)


def _prompt_token_counts(
    model_kwargs: dict[str, Any],
    pad_token_id: int | None,
) -> torch.Tensor | None:
    attention_mask = model_kwargs.get("decoder_attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        return attention_mask.to(torch.long).sum(dim=-1).cpu()

    input_ids = model_kwargs.get("decoder_input_ids")
    if not isinstance(input_ids, torch.Tensor):
        return None
    if pad_token_id is None:
        return torch.full(
            (input_ids.shape[0],),
            input_ids.shape[1],
            dtype=torch.long,
        )
    return input_ids.ne(pad_token_id).to(torch.long).sum(dim=-1).cpu()


def _output_token_counts(
    result: torch.Tensor,
    pad_token_id: int | None,
) -> torch.Tensor:
    if pad_token_id is None:
        return torch.full(
            (result.shape[0],),
            result.shape[1],
            dtype=torch.long,
        )
    return result.ne(pad_token_id).to(torch.long).sum(dim=-1)
