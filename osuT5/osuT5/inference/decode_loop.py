import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn

from ..runtime_profiling import active_prefix_self_attention_context, profile_range


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


def _add_diagnostic_wall(diagnostics: dict[str, Any], key: str, seconds: float) -> None:
    diagnostics[key] = float(diagnostics.get(key, 0.0)) + float(seconds)


@contextmanager
def _diagnostic_range(
        diagnostics: dict[str, Any] | None,
        name: str,
        *,
        wall_key: str | None = None,
        aggregate_wall_key: str | None = None,
) -> Iterator[None]:
    if diagnostics is None:
        yield
        return

    start = time.perf_counter()
    with profile_range(f"active_prefix.{name}"):
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if wall_key is not None:
                _add_diagnostic_wall(diagnostics, wall_key, elapsed)
            if aggregate_wall_key is not None:
                _add_diagnostic_wall(diagnostics, aggregate_wall_key, elapsed)


def _record_decode_bucket(diagnostics: dict[str, Any] | None, prefix_length: int) -> None:
    if diagnostics is None:
        return

    diagnostics["decode_steps"] = int(diagnostics.get("decode_steps", 0)) + 1
    buckets = diagnostics.setdefault("bucket_lengths_seen", [])
    if not buckets or buckets[-1] != prefix_length:
        if buckets:
            diagnostics["bucket_transition_count"] = int(diagnostics.get("bucket_transition_count", 0)) + 1
        buckets.append(prefix_length)


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
        active_prefix_decode_diagnostics: dict[str, Any] | None = None,
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
    if active_prefix_decode_diagnostics is not None:
        with _diagnostic_range(
                active_prefix_decode_diagnostics,
                "setup.cache_position",
                wall_key="cache_position_wall_cpu_s",
        ):
            model_kwargs = model._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
        with _diagnostic_range(
                active_prefix_decode_diagnostics,
                "setup.max_cache_shape",
                wall_key="max_cache_shape_wall_cpu_s",
        ):
            max_cache_len = _max_cache_shape(model_kwargs, generation_config)
    else:
        model_kwargs = model._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
        max_cache_len = _max_cache_shape(model_kwargs, generation_config)

    model_forward = model.__call__
    if active_prefix_decode_diagnostics is not None:
        with _diagnostic_range(
                active_prefix_decode_diagnostics,
                "setup.compile_lookup",
                wall_key="compile_lookup_wall_cpu_s",
        ):
            if model._valid_auto_compile_criteria(model_kwargs, generation_config):
                os.environ["TOKENIZERS_PARALLELISM"] = "0"
                if model.config._attn_implementation == "flash_attention_2":
                    compile_config = generation_config.compile_config
                    if compile_config is not None and compile_config.fullgraph:
                        compile_config.fullgraph = False
                model_forward = model.get_compiled_call(generation_config.compile_config)
    elif model._valid_auto_compile_criteria(model_kwargs, generation_config):
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        if model.config._attn_implementation == "flash_attention_2":
            compile_config = generation_config.compile_config
            if compile_config is not None and compile_config.fullgraph:
                compile_config.fullgraph = False
        model_forward = model.get_compiled_call(generation_config.compile_config)

    is_prefill = True
    scores = None
    while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        if active_prefix_decode_diagnostics is not None:
            with _diagnostic_range(
                    active_prefix_decode_diagnostics,
                    "prepare_inputs",
                    wall_key="prepare_inputs_wall_cpu_s",
            ):
                model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        else:
            model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        if is_prefill:
            if active_prefix_decode_diagnostics is not None:
                with _diagnostic_range(
                        active_prefix_decode_diagnostics,
                        "prefill_forward",
                        wall_key="prefill_forward_wall_cpu_s",
                ):
                    outputs = model(**model_inputs, return_dict=True)
            else:
                outputs = model(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            prefix_length = _bucketed_prefix_length(cur_len, active_prefix_bucket_size, max_cache_len)
            _record_decode_bucket(active_prefix_decode_diagnostics, prefix_length)
            with active_prefix_self_attention_context(prefix_length):
                if active_prefix_decode_diagnostics is not None:
                    first_decode = int(active_prefix_decode_diagnostics.get("decode_steps", 0)) == 1
                    wall_key = (
                        "first_decode_forward_wall_cpu_s"
                        if first_decode
                        else "steady_decode_forward_wall_cpu_s"
                    )
                    with _diagnostic_range(
                            active_prefix_decode_diagnostics,
                            "decode_forward",
                            wall_key=wall_key,
                            aggregate_wall_key="decode_forward_wall_cpu_s",
                    ):
                        outputs = model_forward(**model_inputs, return_dict=True)
                else:
                    outputs = model_forward(**model_inputs, return_dict=True)

        if active_prefix_decode_diagnostics is not None:
            with _diagnostic_range(
                    active_prefix_decode_diagnostics,
                    "update_model_kwargs",
                    wall_key="update_kwargs_wall_cpu_s",
            ):
                model_kwargs = model._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=model.config.is_encoder_decoder,
                )
        else:
            model_kwargs = model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=model.config.is_encoder_decoder,
            )

        if active_prefix_decode_diagnostics is not None:
            with _diagnostic_range(
                    active_prefix_decode_diagnostics,
                    "logits_extract",
                    wall_key="logits_extract_wall_cpu_s",
            ):
                next_token_logits = outputs.logits[:, -1, :].to(
                    copy=True,
                    dtype=torch.float32,
                    device=input_ids.device,
                )
            with _diagnostic_range(
                    active_prefix_decode_diagnostics,
                    "logits_processor",
                    wall_key="logits_processor_wall_cpu_s",
            ):
                next_token_scores = logits_processor(input_ids, next_token_logits)
        else:
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

        if do_sample:
            if active_prefix_decode_diagnostics is not None:
                with _diagnostic_range(
                        active_prefix_decode_diagnostics,
                        "sampling.softmax",
                        wall_key="softmax_wall_cpu_s",
                        aggregate_wall_key="sampling_wall_cpu_s",
                ):
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                with _diagnostic_range(
                        active_prefix_decode_diagnostics,
                        "sampling.multinomial",
                        wall_key="multinomial_wall_cpu_s",
                        aggregate_wall_key="sampling_wall_cpu_s",
                ):
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            if active_prefix_decode_diagnostics is not None:
                with _diagnostic_range(
                        active_prefix_decode_diagnostics,
                        "sampling.argmax",
                        wall_key="argmax_wall_cpu_s",
                        aggregate_wall_key="sampling_wall_cpu_s",
                ):
                    next_tokens = torch.argmax(next_token_scores, dim=-1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

        if has_eos_stopping_criteria:
            if active_prefix_decode_diagnostics is not None:
                with _diagnostic_range(
                        active_prefix_decode_diagnostics,
                        "eos_mask",
                        wall_key="eos_mask_wall_cpu_s",
                ):
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            else:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        if active_prefix_decode_diagnostics is not None:
            with _diagnostic_range(
                    active_prefix_decode_diagnostics,
                    "append_token",
                    wall_key="append_token_wall_cpu_s",
                    aggregate_wall_key="token_append_stop_wall_cpu_s",
            ):
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            with _diagnostic_range(
                    active_prefix_decode_diagnostics,
                    "stopping_criteria",
                    wall_key="stopping_criteria_wall_cpu_s",
                    aggregate_wall_key="token_append_stop_wall_cpu_s",
            ):
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            with _diagnostic_range(
                    active_prefix_decode_diagnostics,
                    "finished_check",
                    wall_key="finished_check_wall_cpu_s",
                    aggregate_wall_key="token_append_stop_wall_cpu_s",
            ):
                this_peer_finished = unfinished_sequences.max() == 0
        else:
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        del outputs

    return input_ids
