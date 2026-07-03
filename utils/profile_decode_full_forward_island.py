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
from osuT5.osuT5.inference.direct_decode import decode_one_token_raw_logits, last_token_logits, prefill_static_cache
from osuT5.osuT5.inference.server import get_eos_token_id
from osuT5.osuT5.runtime_profiling import generation_profile_context
from osuT5.osuT5.tokenizer import ContextType
from utils.profile_decode_linear_kernels import (
    _allclose,
    _bucketed_prefix_length,
    _cuda_event_time_ms,
    _cuda_graph_replay_time_ms,
    _max_abs,
)
from utils.verify_one_token_decode import (
    _build_generation_logits_processors,
    _build_probe_inputs,
    _condition_kwargs,
    _load_args,
    _move_kwargs_for_model,
)


def _project_full_song(
        ms_per_call: float,
        *,
        full_song_decode_steps: int,
        full_song_main_tokens: int,
        full_song_model_time_s: float,
) -> dict[str, Any]:
    seconds = float(ms_per_call) * int(full_song_decode_steps) / 1000.0
    remaining = float(full_song_model_time_s) - seconds
    return {
        "full_song_seconds_at_decode_steps": seconds,
        "fraction_of_model_time": seconds / float(full_song_model_time_s),
        "projection_valid": remaining > 0,
        "ideal_free_component_tps": (
            int(full_song_main_tokens) / remaining
            if remaining > 0
            else None
        ),
    }


def _benchmark(
        name: str,
        fn: Callable[[], torch.Tensor],
        expected: torch.Tensor,
        *,
        warmup: int,
        iters: int,
        atol: float,
        rtol: float,
        cuda_graph_replay: bool,
        full_song_decode_steps: int,
        full_song_main_tokens: int,
        full_song_model_time_s: float,
) -> dict[str, Any]:
    ms, output = _cuda_event_time_ms(fn, warmup=warmup, iters=iters)
    result: dict[str, Any] = {
        "name": name,
        "ms_per_call": ms,
        "allclose": _allclose(expected, output, atol=atol, rtol=rtol),
        "max_abs": _max_abs(expected, output),
        "output_shape": list(output.shape),
        "projection": _project_full_song(
            ms,
            full_song_decode_steps=full_song_decode_steps,
            full_song_main_tokens=full_song_main_tokens,
            full_song_model_time_s=full_song_model_time_s,
        ),
    }
    if cuda_graph_replay:
        try:
            graph_ms, graph_output = _cuda_graph_replay_time_ms(fn, warmup=warmup, iters=iters)
            result["cuda_graph_replay_ms_per_call"] = graph_ms
            result["cuda_graph_replay_allclose"] = _allclose(expected, graph_output, atol=atol, rtol=rtol)
            result["cuda_graph_replay_max_abs"] = _max_abs(expected, graph_output)
            result["cuda_graph_replay_projection"] = _project_full_song(
                graph_ms,
                full_song_decode_steps=full_song_decode_steps,
                full_song_main_tokens=full_song_main_tokens,
                full_song_model_time_s=full_song_model_time_s,
            )
        except Exception as exc:
            result["cuda_graph_replay_error"] = f"{type(exc).__name__}: {exc}"
    return result


@torch.no_grad()
def profile_decode_full_forward_island(
        args,
        *,
        sequence_index: int,
        active_prefix_bucket_size: int,
        active_prefix_decode_length: int | None,
        native_q1_rope_cache_self_attention: bool,
        warmup: int,
        iters: int,
        atol: float,
        rtol: float,
        cuda_graph_replay: bool,
        full_song_decode_steps: int,
        full_song_main_tokens: int,
        full_song_model_time_s: float,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Full-forward decode profiling requires CUDA")

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
        "warmup": int(warmup),
        "iters": int(iters),
        "active_prefix_bucket_size": int(active_prefix_bucket_size),
        "active_prefix_decode_length_override": active_prefix_decode_length,
        "native_q1_rope_cache_self_attention": bool(native_q1_rope_cache_self_attention),
        "cuda_graph_replay": bool(cuda_graph_replay),
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

        results = {
            "full_model_forward": _benchmark(
                "full_model_forward",
                full_forward_logits,
                expected_logits,
                warmup=warmup,
                iters=iters,
                atol=atol,
                rtol=rtol,
                cuda_graph_replay=cuda_graph_replay,
                full_song_decode_steps=full_song_decode_steps,
                full_song_main_tokens=full_song_main_tokens,
                full_song_model_time_s=full_song_model_time_s,
            )
        }

    full_graph_projection = results["full_model_forward"].get("cuda_graph_replay_projection")
    full_graph_s = (
        full_graph_projection.get("full_song_seconds_at_decode_steps")
        if isinstance(full_graph_projection, dict)
        else None
    )
    outside_full_forward_s = (
        float(full_song_model_time_s) - float(full_graph_s)
        if isinstance(full_graph_s, float)
        else None
    )
    return {
        "pass": bool(results["full_model_forward"].get("allclose"))
                and bool(results["full_model_forward"].get("cuda_graph_replay_allclose", True)),
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
        "results": results,
        "outside_full_forward_s": outside_full_forward_s,
        "outside_full_forward_fraction": (
            outside_full_forward_s / float(full_song_model_time_s)
            if outside_full_forward_s is not None
            else None
        ),
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a real one-token full model forward under active-prefix/native q1 settings. "
            "Diagnostic only; not an inference speed claim."
        )
    )
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--sequence-index", type=int, default=9)
    parser.add_argument("--active-prefix-bucket-size", type=int, default=64)
    parser.add_argument("--active-prefix-decode-length", type=int, default=None)
    parser.add_argument("--native-q1-rope-cache-self-attention", action="store_true")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--cuda-graph-replay", action="store_true")
    parser.add_argument("--full-song-decode-steps", type=int, default=7552)
    parser.add_argument("--full-song-main-tokens", type=int, default=7639)
    parser.add_argument("--full-song-model-time-s", type=float, default=28.243)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model_path=/path/to/model")
    cli_args = parser.parse_args()

    start = time.perf_counter()
    args = _load_args(cli_args.config_name, cli_args.overrides)
    result = profile_decode_full_forward_island(
        args,
        sequence_index=cli_args.sequence_index,
        active_prefix_bucket_size=cli_args.active_prefix_bucket_size,
        active_prefix_decode_length=cli_args.active_prefix_decode_length,
        native_q1_rope_cache_self_attention=cli_args.native_q1_rope_cache_self_attention,
        warmup=cli_args.warmup,
        iters=cli_args.iters,
        atol=cli_args.atol,
        rtol=cli_args.rtol,
        cuda_graph_replay=cli_args.cuda_graph_replay,
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
