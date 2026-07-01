import os

import torch
from torch import nn

from ..runtime_profiling import active_prefix_self_attention_context


def _bucketed_prefix_length(cur_len: int, bucket_size: int, max_cache_len: int) -> int:
    if bucket_size <= 0:
        raise ValueError("active-prefix decode bucket size must be positive")
    bucketed = ((cur_len + bucket_size - 1) // bucket_size) * bucket_size
    return min(bucketed, max_cache_len)


def _max_cache_shape(model_kwargs: dict, generation_config) -> int:
    cache = model_kwargs.get("past_key_values")
    if cache is not None and hasattr(cache, "get_max_cache_shape"):
        return int(cache.get_max_cache_shape())
    return int(generation_config.max_length)


def active_prefix_decode_generate(
        model,
        input_ids: torch.LongTensor,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus: bool = False,
        streamer=None,
        *,
        active_prefix_bucket_size: int = 128,
        **model_kwargs,
) -> torch.LongTensor:
    """HF custom_generate loop with normal prefill and bucketed active-prefix one-token decode."""
    if synced_gpus:
        raise ValueError("active_prefix_decode_generate does not support synced_gpus")
    if streamer is not None:
        raise ValueError("active_prefix_decode_generate does not support streamers")
    if generation_config.return_dict_in_generate:
        raise ValueError("active_prefix_decode_generate currently supports tensor outputs only")
    if (
            generation_config.output_attentions
            or generation_config.output_hidden_states
            or generation_config.output_scores
            or generation_config.output_logits
    ):
        raise ValueError("active_prefix_decode_generate does not support auxiliary generation outputs")
    if generation_config.prefill_chunk_size is not None:
        raise ValueError("active_prefix_decode_generate does not support prefill_chunk_size")

    pad_token_id = generation_config._pad_token_tensor
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    batch_size, cur_len = input_ids.shape[:2]
    if batch_size != 1:
        raise ValueError("active_prefix_decode_generate currently supports batch_size=1 only")

    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = model._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
    max_cache_len = _max_cache_shape(model_kwargs, generation_config)

    model_forward = model.__call__
    if model._valid_auto_compile_criteria(model_kwargs, generation_config):
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        if model.config._attn_implementation == "flash_attention_2":
            compile_config = generation_config.compile_config
            if compile_config is not None and compile_config.fullgraph:
                compile_config.fullgraph = False
        model_forward = model.get_compiled_call(generation_config.compile_config)

    is_prefill = True
    scores = None
    while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        if is_prefill:
            outputs = model(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            prefix_length = _bucketed_prefix_length(cur_len, active_prefix_bucket_size, max_cache_len)
            with active_prefix_self_attention_context(prefix_length):
                outputs = model_forward(**model_inputs, return_dict=True)

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )

        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
        next_token_scores = logits_processor(input_ids, next_token_logits)

        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        del outputs

    return input_ids
