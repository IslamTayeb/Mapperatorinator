from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from inference import compile_args, load_model_with_server, setup_inference_environment
from osuT5.osuT5.inference.cache_utils import get_cache
from osuT5.osuT5.inference.decode_loop import (
    _capture_decode_cuda_graph,
    _clone_static_graph_inputs,
    _copy_static_graph_inputs,
)
from osuT5.osuT5.inference.direct_decode import (
    decode_one_token_raw_logits,
    last_token_logits,
    prefill_static_cache,
)
from osuT5.osuT5.inference.server import get_eos_token_id
from osuT5.osuT5.runtime_profiling import generation_profile_context
from osuT5.osuT5.tokenizer import ContextType
from utils.profile_decode_decoder_stack_island import _decoder_stack_hidden, _project_logits
from utils.profile_decode_linear_kernels import _allclose, _bucketed_prefix_length, _max_abs
from utils.verify_one_token_decode import (
    _build_generation_logits_processors,
    _build_probe_inputs,
    _condition_kwargs,
    _load_args,
    _move_kwargs_for_model,
)


def _project_seconds(
        ms_per_call: float,
        *,
        full_song_decode_steps: int,
) -> float:
    return float(ms_per_call) * int(full_song_decode_steps) / 1000.0


def _project_row(
        ms_per_call: float,
        *,
        full_song_decode_steps: int,
        full_song_main_tokens: int,
        full_song_model_time_s: float,
) -> dict[str, Any]:
    seconds = _project_seconds(ms_per_call, full_song_decode_steps=full_song_decode_steps)
    remaining = float(full_song_model_time_s) - seconds
    return {
        "ms_per_call": float(ms_per_call),
        "projected_full_song_seconds": seconds,
        "fraction_of_model_time": seconds / float(full_song_model_time_s),
        "ideal_free_component_tps": (
            int(full_song_main_tokens) / remaining
            if remaining > 0
            else None
        ),
    }


def _project_delta_row(
        ms_per_call: float,
        *,
        full_song_decode_steps: int,
        full_song_main_tokens: int,
        full_song_model_time_s: float,
) -> dict[str, Any]:
    seconds = _project_seconds(ms_per_call, full_song_decode_steps=full_song_decode_steps)
    remaining = float(full_song_model_time_s) - seconds
    return {
        "ms_per_call_delta": float(ms_per_call),
        "projected_full_song_seconds_delta": seconds,
        "fraction_of_model_time": seconds / float(full_song_model_time_s),
        "tps_if_delta_removed": (
            int(full_song_main_tokens) / remaining
            if remaining > 0
            else None
        ),
    }


def _cuda_graph_replay_ms_from_graph(
        graph: torch.cuda.CUDAGraph,
        *,
        iters: int,
) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / max(int(iters), 1)


def _cuda_graph_replay_tensor_ms(
        fn: Callable[[], torch.Tensor],
        *,
        warmup: int,
        iters: int,
) -> tuple[float, torch.Tensor]:
    output = None
    for _ in range(max(int(warmup), 0)):
        output = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    ms = _cuda_graph_replay_ms_from_graph(graph, iters=iters)
    if output is None:
        raise RuntimeError("CUDA graph replay benchmark did not execute")
    return ms, output


def _empty_graph_replay_ms(*, iters: int) -> float | None:
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            pass
        return _cuda_graph_replay_ms_from_graph(graph, iters=iters)
    except Exception:
        return None


def _static_input_copy_ms(
        static_inputs: dict[str, Any],
        model_inputs: dict[str, Any],
        *,
        warmup: int,
        iters: int,
) -> float:
    for _ in range(max(int(warmup), 0)):
        _copy_static_graph_inputs(static_inputs, model_inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(int(iters), 1)):
        _copy_static_graph_inputs(static_inputs, model_inputs)
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / max(int(iters), 1)


@torch.no_grad()
def profile_decode_replay_gap(
        args,
        *,
        sequence_index: int,
        active_prefix_bucket_size: int,
        active_prefix_decode_length: int | None,
        native_q1_rope_cache_self_attention: bool,
        capture_warmup: int,
        replay_warmup: int,
        iters: int,
        atol: float,
        rtol: float,
        full_song_decode_steps: int,
        full_song_main_tokens: int,
        full_song_model_time_s: float,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Decode replay-gap profiling requires CUDA")

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
    model_inputs = _move_kwargs_for_model(
        model,
        _build_probe_inputs(args, model, tokenizer, sequence_index=sequence_index),
    )
    metadata = {
        "model_path": args.model_path,
        "precision": args.precision,
        "attn_implementation": args.attn_implementation,
        "profile_sdpa_backend": args.profile_sdpa_backend,
        "torch_version": torch.__version__,
        "sequence_index": model_inputs.pop("sequence_index"),
        "frame_time_ms": model_inputs.pop("frame_time_ms"),
        "context_type": model_inputs.pop("context_type"),
        "lookback_time": model_inputs.pop("lookback_time"),
        "lookahead_time": model_inputs.pop("lookahead_time"),
        "capture_warmup": int(capture_warmup),
        "replay_warmup": int(replay_warmup),
        "iters": int(iters),
        "active_prefix_bucket_size": int(active_prefix_bucket_size),
        "active_prefix_decode_length_override": active_prefix_decode_length,
        "native_q1_rope_cache_self_attention": bool(native_q1_rope_cache_self_attention),
        "full_song_decode_steps": int(full_song_decode_steps),
        "full_song_main_tokens": int(full_song_main_tokens),
        "full_song_model_time_s": float(full_song_model_time_s),
    }
    prompt = model_inputs["decoder_input_ids"]
    prompt_mask = model_inputs["decoder_attention_mask"]
    prompt_len = int(prompt.shape[-1])
    condition_kwargs = _condition_kwargs(model_inputs)
    logits_processors = _build_generation_logits_processors(
        args,
        tokenizer,
        model.device,
        lookback_time=float(metadata["lookback_time"]),
    )
    context_type = ContextType(metadata["context_type"])
    eos_token_ids = get_eos_token_id(
        tokenizer,
        lookback_time=float(metadata["lookback_time"]),
        lookahead_time=float(metadata["lookahead_time"]),
        context_type=context_type,
    )

    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=args.precision == "amp"), \
            generation_profile_context(
                sdpa_backend=args.profile_sdpa_backend,
                q1_bmm_cross_attention=True,
                native_q1_self_attention=True,
                native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
            ):
        hf_cache = get_cache(model, batch_size=1, num_beams=1, cfg_scale=1.0)
        hf_generate_outputs = model.generate(
            inputs=model_inputs["frames"],
            decoder_input_ids=prompt,
            decoder_attention_mask=prompt_mask,
            **condition_kwargs,
            do_sample=args.do_sample,
            num_beams=args.num_beams,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=1,
            use_cache=True,
            past_key_values=hf_cache,
            logits_processor=logits_processors,
            eos_token_id=eos_token_ids,
            return_dict_in_generate=True,
            output_logits=False,
        )
        probe_token = hf_generate_outputs.sequences[:, prompt_len:prompt_len + 1].to(torch.long)
        if probe_token.numel() != 1:
            raise RuntimeError(f"expected one generated probe token, got shape {list(probe_token.shape)}")
        full_prefix = torch.cat([prompt, probe_token], dim=-1)
        full_mask = torch.cat([prompt_mask, torch.ones_like(probe_token, dtype=prompt_mask.dtype)], dim=-1)

        state = prefill_static_cache(
            model,
            prompt=prompt,
            prompt_attention_mask=prompt_mask,
            frames=model_inputs["frames"],
            condition_kwargs=condition_kwargs,
            active_prefix_self_attention=False,
        )
        cache_position = torch.arange(state.prompt_length, state.prompt_length + 1, device=prompt.device)
        max_cache_len = int(state.cache.get_max_cache_shape())
        computed_active_prefix_length = _bucketed_prefix_length(
            int(full_prefix.shape[-1]),
            active_prefix_bucket_size,
            max_cache_len,
        )
        active_prefix_length = (
            int(active_prefix_decode_length)
            if active_prefix_decode_length is not None
            else computed_active_prefix_length
        )
        direct_result = decode_one_token_raw_logits(
            model,
            state,
            full_prefix=full_prefix,
            full_attention_mask=full_mask,
            condition_kwargs=condition_kwargs,
            cache_position=cache_position,
            active_prefix_self_attention=True,
            active_prefix_self_attention_length=active_prefix_length,
        )
        prepared_inputs = direct_result.prepared_inputs
        expected_logits = direct_result.logits.detach()
        static_inputs = _clone_static_graph_inputs(prepared_inputs)

        def full_forward_logits() -> torch.Tensor:
            with generation_profile_context(
                sdpa_backend=args.profile_sdpa_backend,
                active_prefix_self_attention_length=active_prefix_length,
                q1_bmm_cross_attention=True,
                native_q1_self_attention=True,
                native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
            ):
                outputs = model(**prepared_inputs, return_dict=True)
            return last_token_logits(outputs.logits)

        def decoder_stack_plus_projection_logits() -> torch.Tensor:
            with generation_profile_context(
                sdpa_backend=args.profile_sdpa_backend,
                active_prefix_self_attention_length=active_prefix_length,
                q1_bmm_cross_attention=True,
                native_q1_self_attention=True,
                native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
            ):
                return _project_logits(model, _decoder_stack_hidden(model, prepared_inputs))

        with generation_profile_context(
            sdpa_backend=args.profile_sdpa_backend,
            q1_bmm_cross_attention=True,
            native_q1_self_attention=True,
            native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
        ):
            production_graph, production_outputs, capture_seconds = _capture_decode_cuda_graph(
                model,
                static_inputs,
                active_prefix_length=active_prefix_length,
                warmup=capture_warmup,
            )
        production_ms = _cuda_graph_replay_ms_from_graph(production_graph, iters=iters)
        production_logits = last_token_logits(production_outputs.logits)

        full_forward_ms, full_forward_output = _cuda_graph_replay_tensor_ms(
            full_forward_logits,
            warmup=replay_warmup,
            iters=iters,
        )
        decoder_stack_projection_ms, decoder_stack_projection_output = _cuda_graph_replay_tensor_ms(
            decoder_stack_plus_projection_logits,
            warmup=replay_warmup,
            iters=iters,
        )
        copy_static_inputs = _clone_static_graph_inputs(prepared_inputs)
        static_copy_ms = _static_input_copy_ms(
            copy_static_inputs,
            prepared_inputs,
            warmup=replay_warmup,
            iters=iters,
        )
    empty_graph_ms = _empty_graph_replay_ms(iters=iters)

    measurements = {
        "production_graph_replay": _project_row(
            production_ms,
            full_song_decode_steps=full_song_decode_steps,
            full_song_main_tokens=full_song_main_tokens,
            full_song_model_time_s=full_song_model_time_s,
        ),
        "full_forward_graph_replay": _project_row(
            full_forward_ms,
            full_song_decode_steps=full_song_decode_steps,
            full_song_main_tokens=full_song_main_tokens,
            full_song_model_time_s=full_song_model_time_s,
        ),
        "decoder_stack_plus_projection_graph_replay": _project_row(
            decoder_stack_projection_ms,
            full_song_decode_steps=full_song_decode_steps,
            full_song_main_tokens=full_song_main_tokens,
            full_song_model_time_s=full_song_model_time_s,
        ),
        "static_input_copy": _project_row(
            static_copy_ms,
            full_song_decode_steps=full_song_decode_steps,
            full_song_main_tokens=full_song_main_tokens,
            full_song_model_time_s=full_song_model_time_s,
        ),
    }
    if empty_graph_ms is not None:
        measurements["empty_graph_shell"] = _project_row(
            empty_graph_ms,
            full_song_decode_steps=full_song_decode_steps,
            full_song_main_tokens=full_song_main_tokens,
            full_song_model_time_s=full_song_model_time_s,
        )
    else:
        measurements["empty_graph_shell"] = {
            "ms_per_call": None,
            "projected_full_song_seconds": None,
            "note": "Empty CUDA graph capture/replay was unavailable; this optional floor is not used.",
        }
    deltas = {
        "production_minus_full_forward": _project_delta_row(
            production_ms - full_forward_ms,
            full_song_decode_steps=full_song_decode_steps,
            full_song_main_tokens=full_song_main_tokens,
            full_song_model_time_s=full_song_model_time_s,
        ),
        "production_minus_decoder_stack_plus_projection": _project_delta_row(
            production_ms - decoder_stack_projection_ms,
            full_song_decode_steps=full_song_decode_steps,
            full_song_main_tokens=full_song_main_tokens,
            full_song_model_time_s=full_song_model_time_s,
        ),
        "full_forward_minus_decoder_stack_plus_projection": _project_delta_row(
            full_forward_ms - decoder_stack_projection_ms,
            full_song_decode_steps=full_song_decode_steps,
            full_song_main_tokens=full_song_main_tokens,
            full_song_model_time_s=full_song_model_time_s,
        ),
        "production_minus_full_forward_minus_static_copy": _project_delta_row(
            production_ms - full_forward_ms - static_copy_ms,
            full_song_decode_steps=full_song_decode_steps,
            full_song_main_tokens=full_song_main_tokens,
            full_song_model_time_s=full_song_model_time_s,
        ),
    }
    five_percent_s = float(full_song_model_time_s) * 0.05
    ten_percent_s = float(full_song_model_time_s) * 0.10
    return {
        "pass": (
            _allclose(expected_logits, production_logits, atol=atol, rtol=rtol)
            and _allclose(expected_logits, full_forward_output, atol=atol, rtol=rtol)
            and _allclose(expected_logits, decoder_stack_projection_output, atol=atol, rtol=rtol)
        ),
        "prompt_tokens": prompt_len,
        "probe_token_id": int(probe_token.item()),
        "full_prefix_tokens": int(full_prefix.shape[-1]),
        "cache_position": [int(item) for item in cache_position.detach().cpu().tolist()],
        "max_cache_len": max_cache_len,
        "computed_active_prefix_length": computed_active_prefix_length,
        "active_prefix_length": active_prefix_length,
        "prepared_input_shapes": {
            key: list(value.shape)
            for key, value in prepared_inputs.items()
            if isinstance(value, torch.Tensor)
        },
        "correctness": {
            "production_graph_max_abs": _max_abs(expected_logits, production_logits),
            "full_forward_graph_max_abs": _max_abs(expected_logits, full_forward_output),
            "decoder_stack_plus_projection_graph_max_abs": _max_abs(
                expected_logits,
                decoder_stack_projection_output,
            ),
            "atol": float(atol),
            "rtol": float(rtol),
        },
        "capture_seconds": {
            "production_graph_capture": float(capture_seconds),
        },
        "measurements": measurements,
        "deltas": deltas,
        "thresholds": {
            "five_percent_full_song_seconds": five_percent_s,
            "ten_percent_full_song_seconds": ten_percent_s,
            "seconds_needed_for_500_tps": (
                float(full_song_model_time_s)
                - (int(full_song_main_tokens) / 500.0)
            ),
        },
        "decision_hints": {
            "production_minus_full_forward_clears_5_percent": (
                deltas["production_minus_full_forward"]["projected_full_song_seconds_delta"] >= five_percent_s
            ),
            "production_minus_full_forward_clears_10_percent": (
                deltas["production_minus_full_forward"]["projected_full_song_seconds_delta"] >= ten_percent_s
            ),
            "positive_exclusive_gap_after_static_copy": (
                deltas["production_minus_full_forward_minus_static_copy"]["projected_full_song_seconds_delta"] > 0
            ),
            "diagnostic_only": True,
        },
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare production-style DecodeSession graph replay against isolated same-input "
            "full-forward and decoder-stack graph replay. Diagnostic only; not an inference speed claim."
        )
    )
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--sequence-index", type=int, default=9)
    parser.add_argument("--active-prefix-bucket-size", type=int, default=64)
    parser.add_argument("--active-prefix-decode-length", type=int, default=None)
    parser.add_argument("--native-q1-rope-cache-self-attention", action="store_true")
    parser.add_argument("--capture-warmup", type=int, default=0)
    parser.add_argument("--replay-warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--full-song-decode-steps", type=int, default=7552)
    parser.add_argument("--full-song-main-tokens", type=int, default=7639)
    parser.add_argument("--full-song-model-time-s", type=float, default=28.243)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model_path=/path/to/model")
    cli_args = parser.parse_args()

    start = time.perf_counter()
    args = _load_args(cli_args.config_name, cli_args.overrides)
    result = profile_decode_replay_gap(
        args,
        sequence_index=cli_args.sequence_index,
        active_prefix_bucket_size=cli_args.active_prefix_bucket_size,
        active_prefix_decode_length=cli_args.active_prefix_decode_length,
        native_q1_rope_cache_self_attention=cli_args.native_q1_rope_cache_self_attention,
        capture_warmup=cli_args.capture_warmup,
        replay_warmup=cli_args.replay_warmup,
        iters=cli_args.iters,
        atol=cli_args.atol,
        rtol=cli_args.rtol,
        full_song_decode_steps=cli_args.full_song_decode_steps,
        full_song_main_tokens=cli_args.full_song_main_tokens,
        full_song_model_time_s=cli_args.full_song_model_time_s,
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
