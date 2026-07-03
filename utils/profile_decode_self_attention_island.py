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
from osuT5.osuT5.inference.native_q1_attention import native_q1_attention, native_q1_rope_cache_attention
from osuT5.osuT5.inference.server import get_eos_token_id
from osuT5.osuT5.model.custom_transformers import modeling_varwhisper
from osuT5.osuT5.runtime_profiling import generation_profile_context
from osuT5.osuT5.tokenizer import ContextType
from utils.profile_decode_linear_kernels import (
    _allclose,
    _bucketed_prefix_length,
    _cuda_event_time_ms,
    _max_abs,
    _safe_ratio,
)
from utils.verify_one_token_decode import (
    _build_generation_logits_processors,
    _build_probe_inputs,
    _condition_kwargs,
    _load_args,
    _move_kwargs_for_model,
)


@dataclass
class SelfAttentionIslandCapture:
    name: str
    layer_idx: int
    module: torch.nn.Module
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor | None
    cache_position: torch.Tensor
    position_ids: torch.Tensor
    past_key_value: Any
    output: torch.Tensor
    active_prefix_length: int


def _clone_tensor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    return value


def _self_attention_signature(capture: SelfAttentionIslandCapture) -> str:
    mask_shape = "none" if capture.attention_mask is None else "x".join(str(dim) for dim in capture.attention_mask.shape)
    return (
        f"hidden{'x'.join(str(dim) for dim in capture.hidden_states.shape)}_"
        f"prefix{capture.active_prefix_length}_mask{mask_shape}"
    )


def _effective_self_attention_inputs(
        capture: SelfAttentionIslandCapture,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    module = capture.module
    hidden_states = capture.hidden_states
    bs = hidden_states.shape[0]
    qkv = module.Wqkv(hidden_states).view(bs, -1, 3, module.num_heads, module.head_dim)
    query_states, key_states, value_states = qkv.transpose(1, 3).unbind(dim=2)
    cos, sin = module.rotary_emb(query_states, position_ids=capture.position_ids)
    query_states, key_states = modeling_varwhisper.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": capture.cache_position}
    self_cache = capture.past_key_value.self_attention_cache
    key_states, value_states = self_cache.update(key_states, value_states, module.layer_idx, cache_kwargs)

    attention_mask = capture.attention_mask
    prefix_length = capture.active_prefix_length
    if prefix_length > 0 and key_states.shape[-2] > prefix_length:
        key_states = key_states[:, :, :prefix_length, :]
        value_states = value_states[:, :, :prefix_length, :]
        if isinstance(attention_mask, torch.Tensor) and attention_mask.shape[-1] > prefix_length:
            attention_mask = attention_mask[..., :prefix_length]
    return query_states, key_states, value_states, attention_mask


def _active_prefix_attention_mask(capture: SelfAttentionIslandCapture) -> torch.Tensor | None:
    attention_mask = capture.attention_mask
    if isinstance(attention_mask, torch.Tensor) and attention_mask.shape[-1] > capture.active_prefix_length:
        attention_mask = attention_mask[..., :capture.active_prefix_length]
    return attention_mask


def _self_attention_cache_tensors(capture: SelfAttentionIslandCapture) -> tuple[torch.Tensor, torch.Tensor]:
    cache_layer = capture.past_key_value.self_attention_cache.layers[capture.layer_idx]
    if not getattr(cache_layer, "is_initialized", False):
        raise RuntimeError(f"self-attention cache layer {capture.layer_idx} is not initialized")
    return cache_layer.keys, cache_layer.values


def _qkv_and_rope(capture: SelfAttentionIslandCapture) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    module = capture.module
    hidden_states = capture.hidden_states
    bs = hidden_states.shape[0]
    qkv = module.Wqkv(hidden_states).view(bs, -1, 3, module.num_heads, module.head_dim)
    cos, sin = module.rotary_emb(qkv, position_ids=capture.position_ids)
    return qkv, cos, sin


def _native_attention_module_output(capture: SelfAttentionIslandCapture) -> torch.Tensor:
    query, key, value, attention_mask = _effective_self_attention_inputs(capture)
    hidden_states = native_q1_attention(query, key, value, attention_mask).transpose(1, 2).contiguous()
    hidden_states = hidden_states.view(capture.hidden_states.shape[0], -1, capture.module.all_head_size)
    return capture.module.out_drop(capture.module.Wo(hidden_states))


def _pre_attention_setup_only(capture: SelfAttentionIslandCapture) -> torch.Tensor:
    query, _key, _value, _attention_mask = _effective_self_attention_inputs(capture)
    return query


def _native_attention_only(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    return native_q1_attention(query, key, value, attention_mask)


def _fused_rope_cache_attention_only(
        capture: SelfAttentionIslandCapture,
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
) -> torch.Tensor:
    cache_keys, cache_values = _self_attention_cache_tensors(capture)
    return native_q1_rope_cache_attention(
        qkv,
        cache_keys,
        cache_values,
        cos,
        sin,
        capture.cache_position,
        _active_prefix_attention_mask(capture),
        capture.active_prefix_length,
    )


def _fused_rope_cache_attention_module_output(capture: SelfAttentionIslandCapture) -> torch.Tensor:
    qkv, cos, sin = _qkv_and_rope(capture)
    attention_output = _fused_rope_cache_attention_only(capture, qkv, cos, sin)
    hidden_states = attention_output.transpose(1, 2).contiguous()
    hidden_states = hidden_states.view(capture.hidden_states.shape[0], -1, capture.module.all_head_size)
    return capture.module.out_drop(capture.module.Wo(hidden_states))


def _out_projection_only(capture: SelfAttentionIslandCapture, attention_output: torch.Tensor) -> torch.Tensor:
    hidden_states = attention_output.transpose(1, 2).contiguous()
    hidden_states = hidden_states.view(capture.hidden_states.shape[0], -1, capture.module.all_head_size)
    return capture.module.out_drop(capture.module.Wo(hidden_states))


def _module_forward(capture: SelfAttentionIslandCapture) -> torch.Tensor:
    return capture.module(
        hidden_states=capture.hidden_states,
        past_key_value=capture.past_key_value,
        attention_mask=capture.attention_mask,
        cache_position=capture.cache_position,
        position_ids=capture.position_ids,
        output_attentions=False,
    )[0]


def _cuda_graph_replay_time_ms(
        fn: Callable[[], torch.Tensor],
        *,
        warmup: int,
        iters: int,
) -> tuple[float, torch.Tensor]:
    output = None
    for _ in range(warmup):
        output = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    if output is None:
        raise RuntimeError("CUDA graph benchmark did not execute")
    return float(start.elapsed_time(end)) / max(iters, 1), output


def _benchmark_capture(
        capture: SelfAttentionIslandCapture,
        *,
        native_q1_rope_cache_self_attention: bool,
        warmup: int,
        iters: int,
        atol: float,
        rtol: float,
        cuda_graph_replay: bool,
) -> dict[str, Any]:
    with generation_profile_context(
            active_prefix_self_attention_length=capture.active_prefix_length,
            native_q1_self_attention=True,
            native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
    ):
        reference = capture.output
        precomputed_query, precomputed_key, precomputed_value, precomputed_mask = _effective_self_attention_inputs(capture)
        precomputed_attention = _native_attention_only(
            precomputed_query,
            precomputed_key,
            precomputed_value,
            precomputed_mask,
        ).detach()
        precomputed_qkv, precomputed_cos, precomputed_sin = _qkv_and_rope(capture)
        variants: dict[str, tuple[Callable[[], torch.Tensor], torch.Tensor]] = {
            "repo_module_forward": (lambda: _module_forward(capture), reference),
            "manual_native_attention_island": (lambda: _native_attention_module_output(capture), reference),
            "fused_rope_cache_attention_island": (
                lambda: _fused_rope_cache_attention_module_output(capture),
                reference,
            ),
            "pre_attention_setup_only": (lambda: _pre_attention_setup_only(capture), _pre_attention_setup_only(capture)),
            "native_attention_only": (
                lambda: _native_attention_only(
                    precomputed_query,
                    precomputed_key,
                    precomputed_value,
                    precomputed_mask,
                ),
                precomputed_attention,
            ),
            "fused_post_wqkv_attention_only": (
                lambda: _fused_rope_cache_attention_only(
                    capture,
                    precomputed_qkv,
                    precomputed_cos,
                    precomputed_sin,
                ),
                precomputed_attention,
            ),
            "out_projection_only": (
                lambda: _out_projection_only(capture, precomputed_attention),
                _out_projection_only(capture, precomputed_attention).detach(),
            ),
        }

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

    base_ms = results["repo_module_forward"]["ms_per_call"]
    for result in results.values():
        result["speedup_vs_repo_module_forward"] = _safe_ratio(base_ms, float(result["ms_per_call"]))

    full_ms = results["repo_module_forward"]["ms_per_call"]
    full_graph_ms = results["repo_module_forward"].get("cuda_graph_replay_ms_per_call")
    attention_ms = results["native_attention_only"]["ms_per_call"]
    projection_ms = results["out_projection_only"]["ms_per_call"]
    pre_attention_ms = results["pre_attention_setup_only"]["ms_per_call"]
    results["estimated_component_sum"] = {
        "ms_per_call": pre_attention_ms + attention_ms + projection_ms,
        "pre_attention_setup_ms": pre_attention_ms,
        "native_attention_ms": attention_ms,
        "out_projection_ms": projection_ms,
        "unexplained_vs_repo_ms": full_ms - (pre_attention_ms + attention_ms + projection_ms),
    }
    if isinstance(full_graph_ms, float):
        graph_component_sum = 0.0
        graph_components: dict[str, float] = {}
        for component_name in ("pre_attention_setup_only", "native_attention_only", "out_projection_only"):
            component_ms = results[component_name].get("cuda_graph_replay_ms_per_call")
            if isinstance(component_ms, float):
                graph_components[component_name] = component_ms
                graph_component_sum += component_ms
        results["estimated_cuda_graph_component_sum"] = {
            "ms_per_call": graph_component_sum,
            "repo_module_cuda_graph_replay_ms": full_graph_ms,
            "components": graph_components,
            "unexplained_vs_repo_ms": full_graph_ms - graph_component_sum,
        }
    return results


def _capture_self_attention_islands(
        model,
        prepared_inputs: dict[str, Any],
        *,
        active_prefix_length: int,
        q1_bmm_cross_attention: bool,
        sdpa_backend: str | None,
) -> tuple[dict[str, SelfAttentionIslandCapture], torch.Tensor]:
    captures: dict[str, SelfAttentionIslandCapture] = {}
    handles = []

    def should_capture(name: str, module: torch.nn.Module) -> bool:
        return (
            isinstance(module, modeling_varwhisper.VarWhisperAttention)
            and not module.is_cross_attention
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
            match = re.search(r"(^|\.)model\.decoder\.layers\.(\d+)\.self_attn$", name)
            if match is None:
                return
            cache_position = kwargs.get("cache_position")
            position_ids = kwargs.get("position_ids")
            past_key_value = kwargs.get("past_key_value")
            if not isinstance(cache_position, torch.Tensor):
                raise RuntimeError(f"{name} did not receive tensor cache_position")
            if not isinstance(position_ids, torch.Tensor):
                raise RuntimeError(f"{name} did not receive tensor position_ids")
            if past_key_value is None or not hasattr(past_key_value, "self_attention_cache"):
                raise RuntimeError(f"{name} did not receive an EncoderDecoderCache-like past_key_value")
            captures[name] = SelfAttentionIslandCapture(
                name=name,
                layer_idx=int(match.group(2)),
                module=module,
                hidden_states=hidden_states.detach(),
                attention_mask=_clone_tensor(kwargs.get("attention_mask")),
                cache_position=cache_position.detach(),
                position_ids=position_ids.detach(),
                past_key_value=past_key_value,
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
                q1_bmm_cross_attention=q1_bmm_cross_attention,
                native_q1_self_attention=True,
        ):
            outputs = model(**prepared_inputs)
            logits = last_token_logits(outputs.logits)
    finally:
        for handle in handles:
            handle.remove()

    return captures, logits


@torch.no_grad()
def profile_decode_self_attention_island(
        args,
        *,
        sequence_index: int,
        active_prefix_bucket_size: int,
        active_prefix_decode_length: int | None,
        q1_bmm_cross_attention: bool,
        native_q1_rope_cache_self_attention: bool,
        warmup: int,
        iters: int,
        atol: float,
        rtol: float,
        per_layer: bool,
        full_song_decode_steps: int,
        full_song_main_tokens: int,
        full_song_model_time_s: float,
        cuda_graph_replay: bool,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Decode self-attention island profiling requires CUDA")

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
        "q1_bmm_cross_attention": bool(q1_bmm_cross_attention),
        "native_q1_rope_cache_self_attention": bool(native_q1_rope_cache_self_attention),
        "per_layer": bool(per_layer),
        "full_song_decode_steps": int(full_song_decode_steps),
        "full_song_main_tokens": int(full_song_main_tokens),
        "full_song_model_time_s": float(full_song_model_time_s),
        "cuda_graph_replay": bool(cuda_graph_replay),
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
                q1_bmm_cross_attention=q1_bmm_cross_attention,
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
        captures, replay_logits = _capture_self_attention_islands(
            model,
            direct_result.prepared_inputs,
            active_prefix_length=active_prefix_length,
            q1_bmm_cross_attention=q1_bmm_cross_attention,
            sdpa_backend=args.profile_sdpa_backend,
        )

    if not captures:
        raise RuntimeError("did not capture any one-token decoder self-attention calls")

    signature_members: dict[str, list[str]] = defaultdict(list)
    for name, capture in captures.items():
        signature_members[_self_attention_signature(capture)].append(name)

    signature_reports: dict[str, Any] = {}
    representative_results: dict[str, dict[str, Any]] = {}
    layer_reports: dict[str, Any] = {}
    for signature, names in sorted(signature_members.items()):
        representative = captures[sorted(names)[0]]
        benchmark = _benchmark_capture(
            representative,
            native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
            warmup=warmup,
            iters=iters,
            atol=atol,
            rtol=rtol,
            cuda_graph_replay=cuda_graph_replay,
        )
        representative_results[signature] = benchmark
        signature_reports[signature] = {
            "representative": representative.name,
            "members": sorted(names),
            "member_count": len(names),
            "layer_idx": representative.layer_idx,
            "hidden_shape": list(representative.hidden_states.shape),
            "attention_mask_shape": (
                list(representative.attention_mask.shape)
                if isinstance(representative.attention_mask, torch.Tensor)
                else None
            ),
            "active_prefix_length": representative.active_prefix_length,
            "results": benchmark,
        }

    if per_layer:
        for name, capture in sorted(captures.items()):
            layer_reports[name] = {
                "signature": _self_attention_signature(capture),
                "layer_idx": capture.layer_idx,
                "results": _benchmark_capture(
                    capture,
                    native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
                    warmup=warmup,
                    iters=iters,
                    atol=atol,
                    rtol=rtol,
                    cuda_graph_replay=cuda_graph_replay,
                ),
            }

    projected_full_song: dict[str, Any] = {}
    for signature, report in signature_reports.items():
        member_count = int(report["member_count"])
        results = report["results"]
        repo_ms = float(results["repo_module_forward"]["ms_per_call"])
        repo_graph_ms = results["repo_module_forward"].get("cuda_graph_replay_ms_per_call")
        manual_ms = float(results["manual_native_attention_island"]["ms_per_call"])
        manual_graph_ms = results["manual_native_attention_island"].get("cuda_graph_replay_ms_per_call")
        fused_ms = float(results["fused_rope_cache_attention_island"]["ms_per_call"])
        fused_graph_ms = results["fused_rope_cache_attention_island"].get("cuda_graph_replay_ms_per_call")
        fused_post_wqkv_ms = float(results["fused_post_wqkv_attention_only"]["ms_per_call"])
        fused_post_wqkv_graph_ms = results["fused_post_wqkv_attention_only"].get("cuda_graph_replay_ms_per_call")
        pre_ms = float(results["pre_attention_setup_only"]["ms_per_call"])
        attention_ms = float(results["native_attention_only"]["ms_per_call"])
        out_ms = float(results["out_projection_only"]["ms_per_call"])
        repo_seconds = repo_ms * member_count * full_song_decode_steps / 1000.0
        projection_with_ungraphed_module_valid = repo_seconds < full_song_model_time_s
        ideal_free_island_time_s = (
            full_song_model_time_s - repo_seconds
            if projection_with_ungraphed_module_valid
            else None
        )
        manual_delta_s = (repo_ms - manual_ms) * member_count * full_song_decode_steps / 1000.0
        fused_delta_s = (repo_ms - fused_ms) * member_count * full_song_decode_steps / 1000.0
        fused_vs_manual_delta_s = (manual_ms - fused_ms) * member_count * full_song_decode_steps / 1000.0
        pre_attention_ceiling_s = pre_ms * member_count * full_song_decode_steps / 1000.0
        out_projection_ceiling_s = out_ms * member_count * full_song_decode_steps / 1000.0
        native_attention_ceiling_s = attention_ms * member_count * full_song_decode_steps / 1000.0
        fused_post_wqkv_attention_ceiling_s = fused_post_wqkv_ms * member_count * full_song_decode_steps / 1000.0
        projected_full_song[signature] = {
            "member_count": member_count,
            "decode_steps": int(full_song_decode_steps),
            "main_tokens": int(full_song_main_tokens),
            "repo_self_attention_island_s": repo_seconds,
            "repo_self_attention_island_fraction_of_model_time": repo_seconds / full_song_model_time_s,
            "manual_native_island_delta_s": manual_delta_s,
            "manual_native_island_delta_fraction_of_model_time": manual_delta_s / full_song_model_time_s,
            "fused_rope_cache_island_delta_s": fused_delta_s,
            "fused_rope_cache_island_delta_fraction_of_model_time": fused_delta_s / full_song_model_time_s,
            "fused_rope_cache_vs_manual_delta_s": fused_vs_manual_delta_s,
            "fused_rope_cache_vs_manual_delta_fraction_of_model_time": (
                fused_vs_manual_delta_s / full_song_model_time_s
            ),
            "pre_attention_setup_ceiling_s": pre_attention_ceiling_s,
            "native_attention_ceiling_s": native_attention_ceiling_s,
            "fused_post_wqkv_attention_ceiling_s": fused_post_wqkv_attention_ceiling_s,
            "out_projection_ceiling_s": out_projection_ceiling_s,
            "ungraphed_projection_valid": projection_with_ungraphed_module_valid,
            "ideal_free_island_tps": (
                full_song_main_tokens / ideal_free_island_time_s
                if ideal_free_island_time_s is not None and ideal_free_island_time_s > 0
                else None
            ),
        }
        if isinstance(repo_graph_ms, float):
            repo_graph_seconds = repo_graph_ms * member_count * full_song_decode_steps / 1000.0
            graph_projection_valid = repo_graph_seconds < full_song_model_time_s
            graph_ideal_time_s = (
                full_song_model_time_s - repo_graph_seconds
                if graph_projection_valid
                else None
            )
            projected_full_song[signature]["cuda_graph_repo_self_attention_island_s"] = repo_graph_seconds
            projected_full_song[signature]["cuda_graph_repo_fraction_of_model_time"] = (
                repo_graph_seconds / full_song_model_time_s
            )
            projected_full_song[signature]["cuda_graph_projection_valid"] = graph_projection_valid
            projected_full_song[signature]["cuda_graph_ideal_free_island_tps"] = (
                full_song_main_tokens / graph_ideal_time_s
                if graph_ideal_time_s is not None and graph_ideal_time_s > 0
                else None
            )
        if isinstance(repo_graph_ms, float) and isinstance(manual_graph_ms, float):
            graph_delta_s = (repo_graph_ms - manual_graph_ms) * member_count * full_song_decode_steps / 1000.0
            projected_full_song[signature]["cuda_graph_manual_native_island_delta_s"] = graph_delta_s
            projected_full_song[signature]["cuda_graph_manual_native_island_delta_fraction_of_model_time"] = (
                graph_delta_s / full_song_model_time_s
            )
        if isinstance(repo_graph_ms, float) and isinstance(fused_graph_ms, float):
            graph_fused_delta_s = (repo_graph_ms - fused_graph_ms) * member_count * full_song_decode_steps / 1000.0
            projected_full_song[signature]["cuda_graph_fused_rope_cache_island_delta_s"] = graph_fused_delta_s
            projected_full_song[signature]["cuda_graph_fused_rope_cache_island_delta_fraction_of_model_time"] = (
                graph_fused_delta_s / full_song_model_time_s
            )
        if isinstance(manual_graph_ms, float) and isinstance(fused_graph_ms, float):
            graph_fused_vs_manual_delta_s = (
                (manual_graph_ms - fused_graph_ms) * member_count * full_song_decode_steps / 1000.0
            )
            projected_full_song[signature]["cuda_graph_fused_rope_cache_vs_manual_delta_s"] = (
                graph_fused_vs_manual_delta_s
            )
            projected_full_song[signature]["cuda_graph_fused_rope_cache_vs_manual_delta_fraction_of_model_time"] = (
                graph_fused_vs_manual_delta_s / full_song_model_time_s
            )
        if isinstance(fused_post_wqkv_graph_ms, float):
            graph_fused_post_wqkv_s = fused_post_wqkv_graph_ms * member_count * full_song_decode_steps / 1000.0
            projected_full_song[signature]["cuda_graph_fused_post_wqkv_attention_ceiling_s"] = (
                graph_fused_post_wqkv_s
            )
            projected_full_song[signature]["cuda_graph_fused_post_wqkv_attention_fraction_of_model_time"] = (
                graph_fused_post_wqkv_s / full_song_model_time_s
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
        "captured_self_attention_count": len(captures),
        "signature_reports": signature_reports,
        "layer_reports": layer_reports,
        "projected_full_song": projected_full_song,
        "logits_replay_allclose": bool(_allclose(direct_result.logits, replay_logits, atol=atol, rtol=rtol)),
        "logits_replay_max_abs": _max_abs(direct_result.logits, replay_logits),
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the real one-token decoder self-attention island around native q1 attention. "
            "Diagnostic only; not an inference speed claim."
        )
    )
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--sequence-index", type=int, default=9)
    parser.add_argument("--active-prefix-bucket-size", type=int, default=64)
    parser.add_argument("--active-prefix-decode-length", type=int, default=None)
    parser.add_argument("--q1-bmm-cross-attention", action="store_true")
    parser.add_argument("--native-q1-rope-cache-self-attention", action="store_true")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--per-layer", action="store_true")
    parser.add_argument(
        "--cuda-graph-replay",
        action="store_true",
        help="Also capture each diagnostic variant in a CUDA graph and benchmark replay time.",
    )
    parser.add_argument(
        "--full-song-decode-steps",
        type=int,
        default=7552,
        help="Decode replay count used only for full-song ceiling projections.",
    )
    parser.add_argument(
        "--full-song-main-tokens",
        type=int,
        default=7639,
        help="Generated main token count used only for full-song TPS ceiling projections.",
    )
    parser.add_argument(
        "--full-song-model-time-s",
        type=float,
        default=28.243,
        help="Accepted full-song model time used only for full-song ceiling projections.",
    )
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model_path=/path/to/model")
    cli_args = parser.parse_args()

    start = time.perf_counter()
    args = _load_args(cli_args.config_name, cli_args.overrides)
    result = profile_decode_self_attention_island(
        args,
        sequence_index=cli_args.sequence_index,
        active_prefix_bucket_size=cli_args.active_prefix_bucket_size,
        active_prefix_decode_length=cli_args.active_prefix_decode_length,
        q1_bmm_cross_attention=cli_args.q1_bmm_cross_attention,
        native_q1_rope_cache_self_attention=cli_args.native_q1_rope_cache_self_attention,
        warmup=cli_args.warmup,
        iters=cli_args.iters,
        atol=cli_args.atol,
        rtol=cli_args.rtol,
        per_layer=cli_args.per_layer,
        full_song_decode_steps=cli_args.full_song_decode_steps,
        full_song_main_tokens=cli_args.full_song_main_tokens,
        full_song_model_time_s=cli_args.full_song_model_time_s,
        cuda_graph_replay=cli_args.cuda_graph_replay,
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
