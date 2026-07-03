from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from inference import compile_args, load_model_with_server, setup_inference_environment
from osuT5.osuT5.inference.cache_utils import get_cache
from osuT5.osuT5.inference.direct_decode import (
    decode_one_token_raw_logits,
    last_token_logits,
    prefill_static_cache,
)
from osuT5.osuT5.inference.server import get_eos_token_id
from osuT5.osuT5.model.custom_transformers import modeling_varwhisper
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


@dataclass
class DecoderLayerCapture:
    name: str
    layer_idx: int
    module: torch.nn.Module
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor | None
    encoder_hidden_states: torch.Tensor | None
    past_key_value: Any
    cache_position: torch.Tensor
    position_ids: torch.Tensor | None
    cu_seqlens: torch.Tensor | None
    max_seqlen: int | None
    encoder_cu_seqlens: torch.Tensor | None
    encoder_max_seqlen: int | None
    output: torch.Tensor
    active_prefix_length: int


def _clone_tensor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    return value


def _decoder_layer_signature(capture: DecoderLayerCapture) -> str:
    mask_shape = "none" if capture.attention_mask is None else "x".join(str(dim) for dim in capture.attention_mask.shape)
    encoder_shape = (
        "none"
        if capture.encoder_hidden_states is None
        else "x".join(str(dim) for dim in capture.encoder_hidden_states.shape)
    )
    return (
        f"hidden{'x'.join(str(dim) for dim in capture.hidden_states.shape)}_"
        f"encoder{encoder_shape}_prefix{capture.active_prefix_length}_mask{mask_shape}"
    )


def _layer_forward(capture: DecoderLayerCapture) -> torch.Tensor:
    return capture.module(
        hidden_states=capture.hidden_states,
        attention_mask=capture.attention_mask,
        encoder_hidden_states=capture.encoder_hidden_states,
        past_key_value=capture.past_key_value,
        cache_position=capture.cache_position,
        position_ids=capture.position_ids,
        cu_seqlens=capture.cu_seqlens,
        max_seqlen=capture.max_seqlen,
        encoder_cu_seqlens=capture.encoder_cu_seqlens,
        encoder_max_seqlen=capture.encoder_max_seqlen,
        output_attentions=False,
    )[0]


def _norm_only(norm_module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    return norm_module(hidden_states)


def _benchmark_capture(
        capture: DecoderLayerCapture,
        *,
        warmup: int,
        iters: int,
        atol: float,
        rtol: float,
        cuda_graph_replay: bool,
) -> dict[str, Any]:
    with generation_profile_context(
            active_prefix_self_attention_length=capture.active_prefix_length,
            q1_bmm_cross_attention=True,
            native_q1_self_attention=True,
    ):
        variants: dict[str, tuple[Callable[[], torch.Tensor], torch.Tensor]] = {
            "repo_decoder_layer": (lambda: _layer_forward(capture), capture.output),
            "self_attn_norm_only": (
                lambda: _norm_only(capture.module.self_attn_layer_norm, capture.hidden_states),
                _norm_only(capture.module.self_attn_layer_norm, capture.hidden_states).detach(),
            ),
        }
        if capture.encoder_hidden_states is not None:
            self_attn_output = capture.module.self_attn(
                hidden_states=capture.module.self_attn_layer_norm(capture.hidden_states),
                past_key_value=capture.past_key_value,
                attention_mask=capture.attention_mask,
                cache_position=capture.cache_position,
                position_ids=capture.position_ids,
                cu_seqlens=capture.cu_seqlens,
                max_seqlen=capture.max_seqlen,
                output_attentions=False,
            )[0]
            after_self = capture.hidden_states + self_attn_output
            variants["cross_attn_norm_only"] = (
                lambda: _norm_only(capture.module.cross_attn_layer_norm, after_self),
                _norm_only(capture.module.cross_attn_layer_norm, after_self).detach(),
            )
        variants["mlp_norm_only"] = (
            lambda: _norm_only(capture.module.final_layer_norm, capture.output),
            _norm_only(capture.module.final_layer_norm, capture.output).detach(),
        )

        results: dict[str, dict[str, Any]] = {}
        for name, (fn, expected) in variants.items():
            ms, output = _cuda_event_time_ms(fn, warmup=warmup, iters=iters)
            results[name] = {
                "ms_per_call": ms,
                "allclose": _allclose(expected, output, atol=atol, rtol=rtol),
                "max_abs": _max_abs(expected, output),
                "output_shape": list(output.shape),
            }
            if cuda_graph_replay:
                try:
                    graph_ms, graph_output = _cuda_graph_replay_time_ms(fn, warmup=warmup, iters=iters)
                    results[name]["cuda_graph_replay_ms_per_call"] = graph_ms
                    results[name]["cuda_graph_replay_allclose"] = _allclose(
                        expected,
                        graph_output,
                        atol=atol,
                        rtol=rtol,
                    )
                    results[name]["cuda_graph_replay_max_abs"] = _max_abs(expected, graph_output)
                except Exception as exc:
                    results[name]["cuda_graph_replay_error"] = f"{type(exc).__name__}: {exc}"
    return results


def _capture_decoder_layers(
        model,
        prepared_inputs: dict[str, Any],
        *,
        active_prefix_length: int,
        sdpa_backend: str | None,
) -> tuple[dict[str, DecoderLayerCapture], torch.Tensor]:
    captures: dict[str, DecoderLayerCapture] = {}
    handles = []

    def should_capture(name: str, module: torch.nn.Module) -> bool:
        return (
            isinstance(module, modeling_varwhisper.VarWhisperDecoderLayer)
            and (
                name.startswith("model.decoder.layers.")
                or ".model.decoder.layers." in name
            )
        )

    def hook_for(name: str):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            if name in captures:
                return
            hidden_states = kwargs.get("hidden_states")
            if not isinstance(hidden_states, torch.Tensor):
                if not inputs or not isinstance(inputs[0], torch.Tensor):
                    return
                hidden_states = inputs[0]
            if hidden_states.shape[0] != 1 or hidden_states.shape[1] != 1:
                return
            if not isinstance(output, tuple) or not output or not isinstance(output[0], torch.Tensor):
                return
            match = re.search(r"(^|\.)model\.decoder\.layers\.(\d+)$", name)
            if match is None:
                return
            past_key_value = kwargs.get("past_key_value")
            cache_position = kwargs.get("cache_position")
            if past_key_value is None:
                raise RuntimeError(f"{name} did not receive past_key_value")
            if not isinstance(cache_position, torch.Tensor):
                raise RuntimeError(f"{name} did not receive tensor cache_position")
            captures[name] = DecoderLayerCapture(
                name=name,
                layer_idx=int(match.group(2)),
                module=module,
                hidden_states=hidden_states.detach(),
                attention_mask=_clone_tensor(kwargs.get("attention_mask")),
                encoder_hidden_states=_clone_tensor(kwargs.get("encoder_hidden_states")),
                past_key_value=past_key_value,
                cache_position=cache_position.detach(),
                position_ids=_clone_tensor(kwargs.get("position_ids")),
                cu_seqlens=_clone_tensor(kwargs.get("cu_seqlens")),
                max_seqlen=kwargs.get("max_seqlen"),
                encoder_cu_seqlens=_clone_tensor(kwargs.get("encoder_cu_seqlens")),
                encoder_max_seqlen=kwargs.get("encoder_max_seqlen"),
                output=output[0].detach(),
                active_prefix_length=active_prefix_length,
            )

        return hook

    for name, module in model.named_modules():
        if should_capture(name, module):
            handles.append(module.register_forward_hook(hook_for(name), with_kwargs=True))

    try:
        with generation_profile_context(
                sdpa_backend=sdpa_backend,
                active_prefix_self_attention_length=active_prefix_length,
                q1_bmm_cross_attention=True,
                native_q1_self_attention=True,
        ):
            outputs = model(**prepared_inputs)
            logits = last_token_logits(outputs.logits)
    finally:
        for handle in handles:
            handle.remove()

    return captures, logits


@torch.no_grad()
def profile_decode_decoder_layer_island(
        args,
        *,
        sequence_index: int,
        active_prefix_bucket_size: int,
        active_prefix_decode_length: int | None,
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
        raise RuntimeError("Decode decoder-layer island profiling requires CUDA")

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
        "warmup": warmup,
        "iters": iters,
        "active_prefix_bucket_size": active_prefix_bucket_size,
        "active_prefix_decode_length_override": active_prefix_decode_length,
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
        captures, replay_logits = _capture_decoder_layers(
            model,
            direct_result.prepared_inputs,
            active_prefix_length=active_prefix_length,
            sdpa_backend=args.profile_sdpa_backend,
        )

    if not captures:
        raise RuntimeError("did not capture any one-token decoder layer calls")

    signature_members: dict[str, list[str]] = defaultdict(list)
    for name, capture in captures.items():
        signature_members[_decoder_layer_signature(capture)].append(name)

    signature_reports: dict[str, Any] = {}
    projected_full_song: dict[str, Any] = {}
    for signature, names in sorted(signature_members.items()):
        representative = captures[sorted(names)[0]]
        benchmark = _benchmark_capture(
            representative,
            warmup=warmup,
            iters=iters,
            atol=atol,
            rtol=rtol,
            cuda_graph_replay=cuda_graph_replay,
        )
        member_count = len(names)
        signature_reports[signature] = {
            "representative": representative.name,
            "members": sorted(names),
            "member_count": member_count,
            "layer_idx": representative.layer_idx,
            "hidden_shape": list(representative.hidden_states.shape),
            "encoder_hidden_shape": (
                list(representative.encoder_hidden_states.shape)
                if isinstance(representative.encoder_hidden_states, torch.Tensor)
                else None
            ),
            "attention_mask_shape": (
                list(representative.attention_mask.shape)
                if isinstance(representative.attention_mask, torch.Tensor)
                else None
            ),
            "results": benchmark,
        }
        repo_ms = float(benchmark["repo_decoder_layer"]["ms_per_call"])
        repo_seconds = repo_ms * member_count * full_song_decode_steps / 1000.0
        repo_graph_ms = benchmark["repo_decoder_layer"].get("cuda_graph_replay_ms_per_call")
        projected_full_song[signature] = {
            "member_count": member_count,
            "decode_steps": int(full_song_decode_steps),
            "main_tokens": int(full_song_main_tokens),
            "repo_decoder_layer_s": repo_seconds,
            "repo_decoder_layer_fraction_of_model_time": repo_seconds / full_song_model_time_s,
            "ungraphed_projection_valid": repo_seconds < full_song_model_time_s,
        }
        if isinstance(repo_graph_ms, float):
            repo_graph_seconds = repo_graph_ms * member_count * full_song_decode_steps / 1000.0
            graph_projection_valid = repo_graph_seconds < full_song_model_time_s
            graph_ideal_time_s = (
                full_song_model_time_s - repo_graph_seconds
                if graph_projection_valid
                else None
            )
            projected_full_song[signature]["cuda_graph_repo_decoder_layer_s"] = repo_graph_seconds
            projected_full_song[signature]["cuda_graph_repo_fraction_of_model_time"] = (
                repo_graph_seconds / full_song_model_time_s
            )
            projected_full_song[signature]["cuda_graph_projection_valid"] = graph_projection_valid
            projected_full_song[signature]["cuda_graph_ideal_free_decoder_layers_tps"] = (
                full_song_main_tokens / graph_ideal_time_s
                if graph_ideal_time_s is not None and graph_ideal_time_s > 0
                else None
            )

    return {
        "pass": bool(_allclose(direct_result.logits, replay_logits, atol=atol, rtol=rtol)),
        "prompt_tokens": prompt_len,
        "probe_token_id": int(probe_token.item()),
        "full_prefix_tokens": int(full_prefix.shape[-1]),
        "cache_position": [int(item) for item in cache_position.detach().cpu().tolist()],
        "max_cache_len": max_cache_len,
        "computed_active_prefix_length": computed_active_prefix_length,
        "active_prefix_length": active_prefix_length,
        "captured_decoder_layer_count": len(captures),
        "signature_reports": signature_reports,
        "projected_full_song": projected_full_song,
        "logits_replay_allclose": bool(_allclose(direct_result.logits, replay_logits, atol=atol, rtol=rtol)),
        "logits_replay_max_abs": _max_abs(direct_result.logits, replay_logits),
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the real one-token decoder layer under active-prefix/native q1 settings. "
            "Diagnostic only; not an inference speed claim."
        )
    )
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--sequence-index", type=int, default=9)
    parser.add_argument("--active-prefix-bucket-size", type=int, default=64)
    parser.add_argument("--active-prefix-decode-length", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--cuda-graph-replay", action="store_true")
    parser.add_argument("--full-song-decode-steps", type=int, default=7552)
    parser.add_argument("--full-song-main-tokens", type=int, default=7639)
    parser.add_argument("--full-song-model-time-s", type=float, default=32.217)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model_path=/path/to/model")
    cli_args = parser.parse_args()

    start = time.perf_counter()
    args = _load_args(cli_args.config_name, cli_args.overrides)
    result = profile_decode_decoder_layer_island(
        args,
        sequence_index=cli_args.sequence_index,
        active_prefix_bucket_size=cli_args.active_prefix_bucket_size,
        active_prefix_decode_length=cli_args.active_prefix_decode_length,
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
