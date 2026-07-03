from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

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
from utils.profile_decode_full_forward_island import _benchmark
from utils.profile_decode_linear_kernels import _allclose, _bucketed_prefix_length, _max_abs
from utils.verify_one_token_decode import (
    _build_generation_logits_processors,
    _build_probe_inputs,
    _condition_kwargs,
    _load_args,
    _move_kwargs_for_model,
)


def _is_conditional_generation_model(model) -> bool:
    return hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model, "proj_out")


def _conditional_generation_model(model):
    transformer = getattr(model, "transformer", None)
    if transformer is not None and _is_conditional_generation_model(transformer):
        return transformer
    if _is_conditional_generation_model(model):
        return model
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "model"):
        candidate = base_model.model
        transformer = getattr(candidate, "transformer", None)
        if transformer is not None and _is_conditional_generation_model(transformer):
            return transformer
        if _is_conditional_generation_model(candidate):
            return candidate
    raise RuntimeError("could not locate VarWhisperForConditionalGeneration-style model")


def _encoder_last_hidden_state(encoder_outputs: Any) -> torch.Tensor:
    if hasattr(encoder_outputs, "last_hidden_state"):
        return encoder_outputs.last_hidden_state
    if isinstance(encoder_outputs, (tuple, list)) and encoder_outputs:
        return encoder_outputs[0]
    raise RuntimeError("prepared inputs did not contain usable encoder_outputs")


def _decoder_inputs_for_transformer(model, prepared_inputs: dict[str, Any]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    decoder_input_ids = prepared_inputs.get("decoder_input_ids")
    decoder_inputs_embeds = prepared_inputs.get("decoder_inputs_embeds")
    if (
            decoder_inputs_embeds is None
            and bool(getattr(model, "embed_decoder_input", False))
    ):
        if decoder_input_ids is None:
            raise RuntimeError("outer decoder embedding is enabled but decoder_input_ids are missing")
        decoder_inputs_embeds = model.decoder_embedder(decoder_input_ids)
        decoder_input_ids = None
    return decoder_input_ids, decoder_inputs_embeds


def _decoder_stack_hidden(model, prepared_inputs: dict[str, Any]) -> torch.Tensor:
    conditional_model = _conditional_generation_model(model)
    decoder_input_ids, decoder_inputs_embeds = _decoder_inputs_for_transformer(model, prepared_inputs)
    encoder_outputs = prepared_inputs.get("encoder_outputs")
    if encoder_outputs is None:
        raise RuntimeError("decoder-stack probe requires prepared encoder_outputs")
    decoder_outputs = conditional_model.model.decoder(
        input_ids=decoder_input_ids,
        attention_mask=prepared_inputs.get("decoder_attention_mask"),
        encoder_hidden_states=_encoder_last_hidden_state(encoder_outputs),
        past_key_values=prepared_inputs.get("past_key_values"),
        inputs_embeds=decoder_inputs_embeds,
        position_ids=prepared_inputs.get("decoder_position_ids"),
        use_cache=prepared_inputs.get("use_cache"),
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        cache_position=prepared_inputs.get("cache_position"),
    )
    return decoder_outputs.last_hidden_state


def _model_core_hidden(model, prepared_inputs: dict[str, Any]) -> torch.Tensor:
    conditional_model = _conditional_generation_model(model)
    decoder_input_ids, decoder_inputs_embeds = _decoder_inputs_for_transformer(model, prepared_inputs)
    outputs = conditional_model.model(
        input_features=prepared_inputs.get("input_features"),
        attention_mask=prepared_inputs.get("attention_mask"),
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=prepared_inputs.get("decoder_attention_mask"),
        encoder_outputs=prepared_inputs.get("encoder_outputs"),
        past_key_values=prepared_inputs.get("past_key_values"),
        decoder_inputs_embeds=decoder_inputs_embeds,
        decoder_position_ids=prepared_inputs.get("decoder_position_ids"),
        use_cache=prepared_inputs.get("use_cache"),
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        cache_position=prepared_inputs.get("cache_position"),
    )
    return outputs.last_hidden_state


def _project_logits(model, hidden_states: torch.Tensor) -> torch.Tensor:
    conditional_model = _conditional_generation_model(model)
    return last_token_logits(conditional_model.proj_out(hidden_states))


def _graph_projection_seconds(result: dict[str, Any]) -> float | None:
    projection = result.get("cuda_graph_replay_projection")
    if isinstance(projection, dict):
        seconds = projection.get("full_song_seconds_at_decode_steps")
        if isinstance(seconds, (int, float)):
            return float(seconds)
    return None


@torch.no_grad()
def profile_decode_decoder_stack_island(
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
        raise RuntimeError("Decoder-stack island profiling requires CUDA")

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

        def model_core_hidden() -> torch.Tensor:
            with generation_profile_context(
                sdpa_backend=args.profile_sdpa_backend,
                active_prefix_self_attention_length=active_prefix_length,
                q1_bmm_cross_attention=True,
                native_q1_self_attention=True,
                native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
            ):
                return _model_core_hidden(model, prepared_inputs)

        def decoder_stack_hidden() -> torch.Tensor:
            with generation_profile_context(
                sdpa_backend=args.profile_sdpa_backend,
                active_prefix_self_attention_length=active_prefix_length,
                q1_bmm_cross_attention=True,
                native_q1_self_attention=True,
                native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
            ):
                return _decoder_stack_hidden(model, prepared_inputs)

        def output_projection_logits() -> torch.Tensor:
            return _project_logits(model, expected_decoder_hidden)

        def decoder_stack_plus_projection_logits() -> torch.Tensor:
            return _project_logits(model, decoder_stack_hidden())

        def model_core_plus_projection_logits() -> torch.Tensor:
            return _project_logits(model, model_core_hidden())

        expected_model_core_hidden = model_core_hidden().detach()
        expected_decoder_hidden = decoder_stack_hidden().detach()
        expected_projection_logits = _project_logits(model, expected_decoder_hidden).detach()

        results = {
            "full_model_forward_logits": _benchmark(
                "full_model_forward_logits",
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
            ),
            "model_core_decoder_hidden": _benchmark(
                "model_core_decoder_hidden",
                model_core_hidden,
                expected_model_core_hidden,
                warmup=warmup,
                iters=iters,
                atol=atol,
                rtol=rtol,
                cuda_graph_replay=cuda_graph_replay,
                full_song_decode_steps=full_song_decode_steps,
                full_song_main_tokens=full_song_main_tokens,
                full_song_model_time_s=full_song_model_time_s,
            ),
            "decoder_stack_hidden": _benchmark(
                "decoder_stack_hidden",
                decoder_stack_hidden,
                expected_decoder_hidden,
                warmup=warmup,
                iters=iters,
                atol=atol,
                rtol=rtol,
                cuda_graph_replay=cuda_graph_replay,
                full_song_decode_steps=full_song_decode_steps,
                full_song_main_tokens=full_song_main_tokens,
                full_song_model_time_s=full_song_model_time_s,
            ),
            "output_projection_logits": _benchmark(
                "output_projection_logits",
                output_projection_logits,
                expected_projection_logits,
                warmup=warmup,
                iters=iters,
                atol=atol,
                rtol=rtol,
                cuda_graph_replay=cuda_graph_replay,
                full_song_decode_steps=full_song_decode_steps,
                full_song_main_tokens=full_song_main_tokens,
                full_song_model_time_s=full_song_model_time_s,
            ),
            "model_core_plus_projection_logits": _benchmark(
                "model_core_plus_projection_logits",
                model_core_plus_projection_logits,
                expected_logits,
                warmup=warmup,
                iters=iters,
                atol=atol,
                rtol=rtol,
                cuda_graph_replay=cuda_graph_replay,
                full_song_decode_steps=full_song_decode_steps,
                full_song_main_tokens=full_song_main_tokens,
                full_song_model_time_s=full_song_model_time_s,
            ),
            "decoder_stack_plus_projection_logits": _benchmark(
                "decoder_stack_plus_projection_logits",
                decoder_stack_plus_projection_logits,
                expected_logits,
                warmup=warmup,
                iters=iters,
                atol=atol,
                rtol=rtol,
                cuda_graph_replay=cuda_graph_replay,
                full_song_decode_steps=full_song_decode_steps,
                full_song_main_tokens=full_song_main_tokens,
                full_song_model_time_s=full_song_model_time_s,
            ),
        }

    full_model_graph_s = _graph_projection_seconds(results["full_model_forward_logits"])
    model_core_projection_graph_s = _graph_projection_seconds(results["model_core_plus_projection_logits"])
    decoder_stack_projection_graph_s = _graph_projection_seconds(results["decoder_stack_plus_projection_logits"])
    decoder_stack_graph_s = _graph_projection_seconds(results["decoder_stack_hidden"])
    output_projection_graph_s = _graph_projection_seconds(results["output_projection_logits"])
    graph_replay_gaps: dict[str, float | None] = {
        "full_minus_model_core_plus_projection_s": (
            full_model_graph_s - model_core_projection_graph_s
            if full_model_graph_s is not None and model_core_projection_graph_s is not None
            else None
        ),
        "full_minus_decoder_stack_plus_projection_s": (
            full_model_graph_s - decoder_stack_projection_graph_s
            if full_model_graph_s is not None and decoder_stack_projection_graph_s is not None
            else None
        ),
        "decoder_stack_plus_projection_minus_component_sum_s": (
            decoder_stack_projection_graph_s - decoder_stack_graph_s - output_projection_graph_s
            if (
                decoder_stack_projection_graph_s is not None
                and decoder_stack_graph_s is not None
                and output_projection_graph_s is not None
            )
            else None
        ),
    }
    pass_result = all(
        bool(result.get("allclose")) and bool(result.get("cuda_graph_replay_allclose", True))
        for result in results.values()
    )
    return {
        "pass": pass_result,
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
        "decoder_hidden_allclose_to_model_core": bool(
            _allclose(expected_model_core_hidden, expected_decoder_hidden, atol=atol, rtol=rtol)
        ),
        "decoder_hidden_vs_model_core_max_abs": _max_abs(expected_model_core_hidden, expected_decoder_hidden),
        "projection_logits_allclose_to_expected": bool(
            _allclose(expected_logits, expected_projection_logits, atol=atol, rtol=rtol)
        ),
        "projection_logits_vs_expected_max_abs": _max_abs(expected_logits, expected_projection_logits),
        "results": results,
        "graph_replay_gaps": graph_replay_gaps,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a real one-token decoder-stack boundary under active-prefix/native q1 settings. "
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
    result = profile_decode_decoder_stack_island(
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
