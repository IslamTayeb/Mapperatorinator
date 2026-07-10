from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import transformers
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from config import InferenceConfig
from inference import compile_args, load_model_with_server, setup_inference_environment
from osuT5.osuT5.inference.cache_utils import get_cache
from osuT5.osuT5.inference.optimized.single.state import ProductionDecodeSession
from osuT5.osuT5.runtime_profiling import active_prefix_self_attention_context, generation_profile_context
from osuT5.osuT5.tokenizer import ContextType
from utils.verify_direct_decode_loop import (
    _CaptureRawLogitsProcessor,
    _active_prefix_length,
    _capture_decode_cuda_graph,
    _clone_static_graph_inputs,
    _compare_logits,
    _condition_kwargs,
    _copy_static_graph_inputs,
    _cuda_graph_signature,
    _max_cache_shape,
    _prepare_sequence_buffer,
    _rng_state,
    _rng_states_equal,
    _set_rng_state,
    _stop_reason,
)
from utils.verify_one_token_decode import (
    _assert_supported_probe,
    _build_generation_logits_processors,
    _build_probe_inputs,
    _load_args,
    _move_kwargs_for_model,
)
from osuT5.osuT5.inference.server import get_eos_token_id


def _parse_sequence_indices(value: str) -> list[int]:
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        raise ValueError("--sequence-indices must contain at least one integer")
    return indices


def _generated_token_ids(output: torch.Tensor, prompt_len: int) -> list[int]:
    return [int(item) for item in output[0, prompt_len:].tolist()]


def _stable_encoder_outputs(
        holder: dict[str, BaseModelOutput],
        encoder_outputs: Any,
) -> BaseModelOutput:
    if not isinstance(encoder_outputs, BaseModelOutput):
        raise RuntimeError(f"Expected BaseModelOutput encoder_outputs, got {type(encoder_outputs).__name__}")
    source = encoder_outputs.last_hidden_state
    if not isinstance(source, torch.Tensor):
        raise RuntimeError("encoder_outputs.last_hidden_state must be a tensor")
    current = holder.get("encoder_outputs")
    if current is None or tuple(current.last_hidden_state.shape) != tuple(source.shape):
        holder["encoder_outputs"] = BaseModelOutput(
            last_hidden_state=source.clone(memory_format=torch.contiguous_format),
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    else:
        holder["encoder_outputs"].last_hidden_state.copy_(source)
    return holder["encoder_outputs"]


def _graph_summary(graph_cache: dict[tuple[Any, ...], dict[str, Any]]) -> dict[str, Any]:
    by_prefix: dict[str, dict[str, float | int]] = {}
    for entry in graph_cache.values():
        prefix = str(entry["active_prefix_length"])
        item = by_prefix.setdefault(prefix, {"graphs": 0, "capture_seconds": 0.0, "decode_replays": 0})
        item["graphs"] = int(item["graphs"]) + 1
        item["capture_seconds"] = float(item["capture_seconds"]) + float(entry["capture_seconds"])
        item["decode_replays"] = int(item["decode_replays"]) + int(entry["decode_replays"])
    return {
        "graphs": len(graph_cache),
        "capture_seconds": float(sum(float(entry["capture_seconds"]) for entry in graph_cache.values())),
        "decode_replays": int(sum(int(entry["decode_replays"]) for entry in graph_cache.values())),
        "by_prefix": dict(sorted(by_prefix.items(), key=lambda item: int(item[0]) if item[0].isdigit() else 10**9)),
    }


@torch.no_grad()
def persistent_decode_loop_generate(
        model,
        input_ids: torch.LongTensor,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus: bool = False,
        streamer=None,
        *,
        shared_graph_cache: dict[tuple[Any, ...], dict[str, Any]],
        stable_encoder_holder: dict[str, BaseModelOutput],
        buffered_input_ids: bool,
        active_prefix_decode: bool,
        active_prefix_decode_length: int | None,
        active_prefix_bucket_size: int,
        cuda_graph_forward: bool,
        cuda_graph_warmup: int,
        **model_kwargs,
) -> torch.LongTensor:
    """Verifier-only persistent graph/cache decode loop.

    This intentionally mirrors the direct-loop verifier and is not wired into
    production inference. It proves whether one stable cache/encoder object can
    safely reuse CUDA graph captures across multiple generation windows.
    """
    if synced_gpus:
        raise ValueError("persistent_decode_loop_generate does not support synced_gpus")
    if streamer is not None:
        raise ValueError("persistent_decode_loop_generate does not support streamers")
    if generation_config.return_dict_in_generate:
        raise ValueError("persistent_decode_loop_generate currently supports tensor outputs only")
    if (
            generation_config.output_attentions
            or generation_config.output_hidden_states
            or generation_config.output_scores
            or generation_config.output_logits
    ):
        raise ValueError("persistent_decode_loop_generate does not support auxiliary generation outputs")
    if generation_config.prefill_chunk_size is not None:
        raise ValueError("persistent_decode_loop_generate does not support prefill_chunk_size")

    batch_size, cur_len = input_ids.shape[:2]
    if batch_size != 1:
        raise ValueError("persistent_decode_loop_generate currently supports batch_size=1 only")
    if cuda_graph_forward and not torch.cuda.is_available():
        raise RuntimeError("cuda_graph_forward requires CUDA")

    pad_token_id = generation_config._pad_token_tensor
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample
    max_length = int(generation_config.max_length)
    sequence_buffer = (
        _prepare_sequence_buffer(input_ids, max_length, pad_token_id)
        if buffered_input_ids
        else None
    )

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
        current_input_ids = sequence_buffer[:, :cur_len] if sequence_buffer is not None else input_ids
        model_inputs = model.prepare_inputs_for_generation(current_input_ids, **model_kwargs)
        if is_prefill:
            outputs = model(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            prefix_length = _active_prefix_length(
                cur_len,
                active_prefix_decode=active_prefix_decode,
                active_prefix_decode_length=active_prefix_decode_length,
                active_prefix_bucket_size=active_prefix_bucket_size,
                max_cache_len=max_cache_len,
            )
            if cuda_graph_forward:
                graph_key = _cuda_graph_signature(prefix_length, model_inputs)
                graph_entry = shared_graph_cache.get(graph_key)
                if graph_entry is None:
                    graph_static_inputs = _clone_static_graph_inputs(model_inputs)
                    graph, graph_outputs, capture_seconds = _capture_decode_cuda_graph(
                        model,
                        graph_static_inputs,
                        active_prefix_length=prefix_length,
                        warmup=cuda_graph_warmup,
                    )
                    graph_entry = {
                        "graph": graph,
                        "outputs": graph_outputs,
                        "static_inputs": graph_static_inputs,
                        "active_prefix_length": prefix_length,
                        "capture_seconds": capture_seconds,
                        "decode_replays": 0,
                    }
                    shared_graph_cache[graph_key] = graph_entry
                else:
                    _copy_static_graph_inputs(graph_entry["static_inputs"], model_inputs)
                    graph_entry["graph"].replay()
                graph_entry["decode_replays"] = int(graph_entry["decode_replays"]) + 1
                outputs = graph_entry["outputs"]
            else:
                with active_prefix_self_attention_context(prefix_length):
                    outputs = model_forward(**model_inputs, return_dict=True)

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        encoder_outputs = model_kwargs.get("encoder_outputs")
        if encoder_outputs is not None:
            model_kwargs["encoder_outputs"] = _stable_encoder_outputs(stable_encoder_holder, encoder_outputs)

        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
        next_token_scores = logits_processor(current_input_ids, next_token_logits)

        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        if sequence_buffer is None:
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        else:
            sequence_buffer[:, cur_len].copy_(next_tokens)
            input_ids = sequence_buffer[:, :cur_len + 1]

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        del outputs

    if sequence_buffer is not None:
        return sequence_buffer[:, :cur_len].clone(memory_format=torch.contiguous_format)
    return input_ids


def _window_metadata(model_inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "sequence_index": model_inputs.pop("sequence_index"),
        "frame_time_ms": model_inputs.pop("frame_time_ms"),
        "context_type": model_inputs.pop("context_type"),
        "lookback_time": model_inputs.pop("lookback_time"),
        "lookahead_time": model_inputs.pop("lookahead_time"),
    }


def _eos_token_ids(tokenizer, metadata: dict[str, Any]) -> list[int]:
    context_type = ContextType(metadata["context_type"])
    return [
        int(item)
        for item in get_eos_token_id(
            tokenizer,
            lookback_time=float(metadata["lookback_time"]),
            lookahead_time=float(metadata["lookahead_time"]),
            context_type=context_type,
        )
    ]


@torch.no_grad()
def verify_persistent_decode_session(
        args: InferenceConfig,
        *,
        sequence_indices: list[int],
        max_new_tokens: int,
        active_prefix_bucket_size: int,
        cuda_graph_warmup: int,
        buffered_input_ids: bool,
        q1_bmm_cross_attention: bool,
        atol: float,
        rtol: float,
        top_k: int,
) -> dict[str, Any]:
    _assert_supported_probe(args)
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    compile_args(args, verbose=False)
    setup_inference_environment(args.seed)
    model, tokenizer = load_model_with_server(
        args.model_path,
        args.train,
        args.device,
        max_batch_size=args.max_batch_size,
        use_server=False,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
        lora_path=args.lora_path,
        gamemode=args.gamemode,
        auto_select_gamemode_model=args.auto_select_gamemode_model,
        generation_compile=args.inference_generation_compile,
    )
    model.eval()

    windows: list[dict[str, Any]] = []
    for sequence_index in sequence_indices:
        model_inputs = _move_kwargs_for_model(
            model,
            _build_probe_inputs(args, model, tokenizer, sequence_index=sequence_index),
        )
        metadata = _window_metadata(model_inputs)
        prompt = model_inputs["decoder_input_ids"]
        windows.append({
            "metadata": metadata,
            "prompt_len": int(prompt.shape[-1]),
            "model_inputs": model_inputs,
            "condition_kwargs": _condition_kwargs(model_inputs),
            "eos_token_ids": _eos_token_ids(tokenizer, metadata),
        })

    initial_rng_state = _rng_state()

    reference_results: list[dict[str, Any]] = []
    _set_rng_state(initial_rng_state)
    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=args.precision == "amp"), \
            generation_profile_context(sdpa_backend=args.profile_sdpa_backend):
        for window in windows:
            model_inputs = window["model_inputs"]
            captures: list[torch.Tensor] = []
            processors = _build_generation_logits_processors(
                args,
                tokenizer,
                model.device,
                lookback_time=float(window["metadata"]["lookback_time"]),
            )
            processors.insert(0, _CaptureRawLogitsProcessor(captures))
            cache = get_cache(model, batch_size=1, num_beams=1, cfg_scale=1.0)
            output = model.generate(
                inputs=model_inputs["frames"],
                decoder_input_ids=model_inputs["decoder_input_ids"],
                decoder_attention_mask=model_inputs["decoder_attention_mask"],
                **window["condition_kwargs"],
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                past_key_values=cache,
                logits_processor=processors,
                eos_token_id=window["eos_token_ids"],
            )
            reference_results.append({
                "output": output,
                "captures": captures,
                "tokens": _generated_token_ids(output, int(window["prompt_len"])),
                "stop_reason": _stop_reason(output, int(window["prompt_len"]), max_new_tokens, window["eos_token_ids"]),
            })
        reference_end_rng_state = _rng_state()

    candidate_results: list[dict[str, Any]] = []
    session = ProductionDecodeSession()
    candidate_cache_ids: list[int] = []
    _set_rng_state(initial_rng_state)
    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=args.precision == "amp"), \
            generation_profile_context(
                sdpa_backend=args.profile_sdpa_backend,
                q1_bmm_cross_attention=q1_bmm_cross_attention,
            ):
        for window in windows:
            shared_cache = session.cache_for_window(
                model,
                batch_size=1,
                num_beams=1,
                cfg_scale=1.0,
            )
            candidate_cache_ids.append(id(shared_cache))
            model_inputs = window["model_inputs"]
            captures = []
            processors = _build_generation_logits_processors(
                args,
                tokenizer,
                model.device,
                lookback_time=float(window["metadata"]["lookback_time"]),
            )
            processors.insert(0, _CaptureRawLogitsProcessor(captures))
            output = model.generate(
                inputs=model_inputs["frames"],
                decoder_input_ids=model_inputs["decoder_input_ids"],
                decoder_attention_mask=model_inputs["decoder_attention_mask"],
                **window["condition_kwargs"],
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                past_key_values=shared_cache,
                logits_processor=processors,
                eos_token_id=window["eos_token_ids"],
                custom_generate=partial(
                    persistent_decode_loop_generate,
                    **session.active_prefix_decode_kwargs(),
                    buffered_input_ids=buffered_input_ids,
                    active_prefix_decode=True,
                    active_prefix_decode_length=None,
                    active_prefix_bucket_size=active_prefix_bucket_size,
                    cuda_graph_forward=True,
                    cuda_graph_warmup=cuda_graph_warmup,
                ),
            )
            candidate_results.append({
                "output": output,
                "captures": captures,
                "tokens": _generated_token_ids(output, int(window["prompt_len"])),
                "stop_reason": _stop_reason(output, int(window["prompt_len"]), max_new_tokens, window["eos_token_ids"]),
            })
        candidate_end_rng_state = _rng_state()

    window_reports: list[dict[str, Any]] = []
    all_pass = True
    for window, reference, candidate in zip(windows, reference_results, candidate_results):
        logit_reports = []
        captures_match = len(reference["captures"]) == len(candidate["captures"])
        for step, (reference_logits, candidate_logits) in enumerate(zip(reference["captures"], candidate["captures"])):
            comparison = _compare_logits(reference_logits, candidate_logits, atol=atol, rtol=rtol, top_k=top_k)
            comparison["step"] = step
            logit_reports.append(comparison)
        token_match = reference["tokens"] == candidate["tokens"]
        logits_pass = captures_match and all(item["allclose"] and item["topk_match"] for item in logit_reports)
        window_pass = token_match and logits_pass and reference["stop_reason"] == candidate["stop_reason"]
        all_pass = all_pass and window_pass
        max_abs = max((float(item["max_abs"]) for item in logit_reports), default=0.0)
        window_reports.append({
            "sequence_index": int(window["metadata"]["sequence_index"]),
            "frame_time_ms": float(window["metadata"]["frame_time_ms"]),
            "prompt_len": int(window["prompt_len"]),
            "reference_generated_tokens": len(reference["tokens"]),
            "candidate_generated_tokens": len(candidate["tokens"]),
            "token_match": token_match,
            "captures_match": captures_match,
            "logits_pass": logits_pass,
            "max_logit_abs_diff": max_abs,
            "reference_stop_reason": reference["stop_reason"],
            "candidate_stop_reason": candidate["stop_reason"],
            "pass": window_pass,
        })

    rng_match = _rng_states_equal(reference_end_rng_state, candidate_end_rng_state)
    all_pass = all_pass and rng_match
    graph_summary = _graph_summary(session.graph_cache)
    unique_prefixes = len(graph_summary["by_prefix"])
    graph_capture_gate = int(graph_summary["graphs"]) == unique_prefixes
    cache_identity_reused = len(set(candidate_cache_ids)) == 1
    stable_encoder_output_shape = (
        list(session.stable_encoder_holder["encoder_outputs"].last_hidden_state.shape)
        if "encoder_outputs" in session.stable_encoder_holder
        else None
    )
    all_pass = (
        all_pass
        and graph_capture_gate
        and cache_identity_reused
        and stable_encoder_output_shape is not None
    )

    return {
        "pass": all_pass,
        "metadata": {
            "model_path": args.model_path,
            "precision": args.precision,
            "attn_implementation": args.attn_implementation,
            "profile_sdpa_backend": args.profile_sdpa_backend,
            "generation_compile_enabled": not bool(getattr(getattr(model, "generation_config", None), "disable_compile", True)),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "sequence_indices": sequence_indices,
            "max_new_tokens": max_new_tokens,
            "active_prefix_bucket_size": active_prefix_bucket_size,
            "cuda_graph_warmup": cuda_graph_warmup,
            "buffered_input_ids": bool(buffered_input_ids),
            "q1_bmm_cross_attention": bool(q1_bmm_cross_attention),
            "atol": atol,
            "rtol": rtol,
            "top_k": top_k,
        },
        "rng_state_match": rng_match,
        "production_session_cache_identity_reused": cache_identity_reused,
        "production_session_cache_ids": candidate_cache_ids,
        "windows": window_reports,
        "graph_summary": graph_summary,
        "graph_capture_count_equals_unique_prefix_count": graph_capture_gate,
        "stable_encoder_output_shape": stable_encoder_output_shape,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify persistent DecodeSession graph reuse across decode windows.")
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--sequence-indices", default="9,10")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--active-prefix-bucket-size", type=int, default=64)
    parser.add_argument("--cuda-graph-warmup", type=int, default=0)
    parser.add_argument("--buffered-input-ids", action="store_true")
    parser.add_argument("--q1-bmm-cross-attention", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model_path=/path/to/model")
    cli_args = parser.parse_args()

    start = time.perf_counter()
    args = _load_args(cli_args.config_name, cli_args.overrides)
    result = verify_persistent_decode_session(
        args,
        sequence_indices=_parse_sequence_indices(cli_args.sequence_indices),
        max_new_tokens=cli_args.max_new_tokens,
        active_prefix_bucket_size=cli_args.active_prefix_bucket_size,
        cuda_graph_warmup=cli_args.cuda_graph_warmup,
        buffered_input_ids=cli_args.buffered_input_ids,
        q1_bmm_cross_attention=cli_args.q1_bmm_cross_attention,
        atol=cli_args.atol,
        rtol=cli_args.rtol,
        top_k=cli_args.top_k,
    )
    result["metadata"]["config_name"] = cli_args.config_name
    result["wall_seconds"] = time.perf_counter() - start

    if cli_args.report_path is not None:
        cli_args.report_path.parent.mkdir(parents=True, exist_ok=True)
        cli_args.report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
