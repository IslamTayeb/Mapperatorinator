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
import torch.nn.functional as F

from inference import compile_args, load_model_with_server, setup_inference_environment
from osuT5.osuT5.inference.cache_utils import get_cache
from osuT5.osuT5.inference.direct_decode import (
    decode_one_token_raw_logits,
    last_token_logits,
    prefill_static_cache,
)
from osuT5.osuT5.inference.server import get_eos_token_id
from osuT5.osuT5.runtime_profiling import generation_profile_context
from osuT5.osuT5.tokenizer import ContextType
from utils.verify_one_token_decode import (
    _build_generation_logits_processors,
    _build_probe_inputs,
    _condition_kwargs,
    _load_args,
    _move_kwargs_for_model,
)


@dataclass
class LinearCapture:
    name: str
    input: torch.Tensor
    output: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor | None


def _bucketed_prefix_length(cur_len: int, bucket_size: int, max_cache_len: int) -> int:
    if bucket_size <= 0:
        raise ValueError("active-prefix decode bucket size must be positive")
    bucketed = ((cur_len + bucket_size - 1) // bucket_size) * bucket_size
    return min(bucketed, max_cache_len)


def _cuda_event_time_ms(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> tuple[float, torch.Tensor]:
    output = None
    for _ in range(warmup):
        output = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        output = fn()
    end.record()
    torch.cuda.synchronize()
    if output is None:
        raise RuntimeError("benchmark did not execute")
    return float(start.elapsed_time(end)) / max(iters, 1), output


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


def _max_abs(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float((reference.detach().to(torch.float32) - candidate.detach().to(torch.float32)).abs().max().item())


def _allclose(reference: torch.Tensor, candidate: torch.Tensor, *, atol: float, rtol: float) -> bool:
    return bool(torch.allclose(reference.detach().to(torch.float32), candidate.detach().to(torch.float32), atol=atol, rtol=rtol))


def _safe_ratio(reference_ms: float, candidate_ms: float) -> float | None:
    if candidate_ms <= 0:
        return None
    return reference_ms / candidate_ms


def _linear_variants(
        capture: LinearCapture,
        *,
        native_linear_variant: bool,
) -> dict[str, Callable[[], torch.Tensor]]:
    x = capture.input
    weight = capture.weight
    bias = capture.bias
    output_shape = (*x.shape[:-1], weight.shape[0])
    x2d = x.reshape(-1, x.shape[-1])
    if x2d.shape[0] != 1:
        raise ValueError(f"expected one-token linear input for {capture.name}, got {list(x.shape)}")
    x1d = x2d.reshape(-1)
    weight_t = weight.t()

    def module_linear() -> torch.Tensor:
        return F.linear(x, weight, bias)

    def matmul_linear() -> torch.Tensor:
        output = torch.matmul(x, weight_t)
        if bias is not None:
            output = output + bias
        return output

    def addmm_linear() -> torch.Tensor:
        if bias is None:
            output = torch.mm(x2d, weight_t)
        else:
            output = torch.addmm(bias, x2d, weight_t)
        return output.reshape(output_shape)

    def mv_linear() -> torch.Tensor:
        output = torch.mv(weight, x1d)
        if bias is not None:
            output = output + bias
        return output.reshape(output_shape)

    variants: dict[str, Callable[[], torch.Tensor]] = {
        "functional_linear": module_linear,
        "matmul": matmul_linear,
        "addmm": addmm_linear,
        "mv": mv_linear,
    }
    if native_linear_variant:
        from osuT5.osuT5.inference.native_linear import native_one_token_linear, preload_native_linear

        preload_native_linear()

        def native_linear_bs128() -> torch.Tensor:
            return native_one_token_linear(x, weight, bias, block_size=128)

        def native_linear_bs256() -> torch.Tensor:
            return native_one_token_linear(x, weight, bias, block_size=256)

        def native_linear_bs512() -> torch.Tensor:
            return native_one_token_linear(x, weight, bias, block_size=512)

        variants.update({
            "native_linear_bs128": native_linear_bs128,
            "native_linear_bs256": native_linear_bs256,
            "native_linear_bs512": native_linear_bs512,
        })
    return variants


def _benchmark_variants(
        *,
        reference: torch.Tensor,
        variants: dict[str, Callable[[], torch.Tensor]],
        warmup: int,
        iters: int,
        atol: float,
        rtol: float,
        cuda_graph_replay: bool,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for name, fn in variants.items():
        ms, output = _cuda_event_time_ms(fn, warmup=warmup, iters=iters)
        results[name] = {
            "ms_per_call": ms,
            "allclose": _allclose(reference, output, atol=atol, rtol=rtol),
            "max_abs": _max_abs(reference, output),
            "output_shape": list(output.shape),
        }
        if cuda_graph_replay:
            try:
                graph_ms, graph_output = _cuda_graph_replay_time_ms(fn, warmup=warmup, iters=iters)
                results[name]["cuda_graph_replay_ms_per_call"] = graph_ms
                results[name]["cuda_graph_replay_allclose"] = _allclose(
                    reference,
                    graph_output,
                    atol=atol,
                    rtol=rtol,
                )
                results[name]["cuda_graph_replay_max_abs"] = _max_abs(reference, graph_output)
            except Exception as exc:
                results[name]["cuda_graph_replay_error"] = f"{type(exc).__name__}: {exc}"
    base_name = next(iter(variants))
    base_ms = results[base_name]["ms_per_call"]
    for result in results.values():
        result[f"speedup_vs_{base_name}"] = _safe_ratio(base_ms, float(result["ms_per_call"]))
        base_graph_ms = results[base_name].get("cuda_graph_replay_ms_per_call")
        result_graph_ms = result.get("cuda_graph_replay_ms_per_call")
        if isinstance(base_graph_ms, float) and isinstance(result_graph_ms, float):
            result[f"cuda_graph_replay_speedup_vs_{base_name}"] = _safe_ratio(base_graph_ms, result_graph_ms)
    return results


def _capture_decoder_linears(
        model,
        prepared_inputs: dict[str, Any],
        *,
        active_prefix_length: int,
        q1_bmm_cross_attention: bool,
        native_q1_self_attention: bool,
        native_q1_rope_cache_self_attention: bool,
        sdpa_backend: str | None,
) -> tuple[dict[str, LinearCapture], torch.Tensor]:
    captures: dict[str, LinearCapture] = {}
    handles = []

    def should_capture(name: str, module: torch.nn.Module) -> bool:
        return isinstance(module, torch.nn.Linear) and (
            name == "proj_out"
            or name.endswith(".proj_out")
            or name.startswith("model.decoder.layers.")
            or ".model.decoder.layers." in name
        )

    def hook_for(name: str):
        def hook(module: torch.nn.Linear, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            if name in captures:
                return
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            captures[name] = LinearCapture(
                name=name,
                input=inputs[0].detach(),
                output=output.detach(),
                weight=module.weight.detach(),
                bias=module.bias.detach() if module.bias is not None else None,
            )

        return hook

    for name, module in model.named_modules():
        if should_capture(name, module):
            handles.append(module.register_forward_hook(hook_for(name)))

    try:
        with generation_profile_context(
                sdpa_backend=sdpa_backend,
                active_prefix_self_attention_length=active_prefix_length,
                q1_bmm_cross_attention=q1_bmm_cross_attention,
                native_q1_self_attention=native_q1_self_attention,
                native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
        ):
            outputs = model(**prepared_inputs)
            logits = last_token_logits(outputs.logits)
    finally:
        for handle in handles:
            handle.remove()

    return captures, logits


def _signature_for_capture(capture: LinearCapture) -> str:
    return (
        f"in{capture.input.shape[-1]}_out{capture.weight.shape[0]}_"
        f"bias{int(capture.bias is not None)}_shape{'x'.join(str(dim) for dim in capture.input.shape)}"
    )


def _operation_kind(name: str) -> str:
    if name == "proj_out" or name.endswith(".proj_out"):
        return "decoder.output_projection"
    if ".self_attn.Wqkv" in name:
        return "decoder.self_attn.qkv"
    if ".self_attn.Wo" in name:
        return "decoder.self_attn.out"
    if ".cross_attn.Wq" in name:
        return "decoder.cross_attn.q"
    if ".cross_attn.Wo" in name:
        return "decoder.cross_attn.out"
    if ".fc1" in name:
        return "decoder.mlp.fc1"
    if ".fc2" in name:
        return "decoder.mlp.fc2"
    return "decoder.other_linear"


def _find_layer_mlp_inputs(captures: dict[str, LinearCapture]) -> dict[str, tuple[LinearCapture, LinearCapture]]:
    by_layer: dict[str, dict[str, LinearCapture]] = defaultdict(dict)
    for name, capture in captures.items():
        match = re.search(r"(^|\.)model\.decoder\.layers\.(\d+)\.(fc1|fc2)$", name)
        if match is None:
            continue
        layer_key = f"decoder.layers.{match.group(2)}"
        by_layer[layer_key][match.group(3)] = capture
    return {
        layer: (items["fc1"], items["fc2"])
        for layer, items in by_layer.items()
        if "fc1" in items and "fc2" in items
    }


def _find_decoder_layer_modules(model) -> dict[int, torch.nn.Module]:
    layers: dict[int, torch.nn.Module] = {}
    for name, module in model.named_modules():
        match = re.search(r"(^|\.)model\.decoder\.layers\.(\d+)$", name)
        if match is None:
            continue
        if hasattr(module, "activation_fn") and hasattr(module, "fc1") and hasattr(module, "fc2"):
            layers[int(match.group(2))] = module
    return layers


def _benchmark_mlp_variants(
        *,
        fc1_capture: LinearCapture,
        fc2_capture: LinearCapture,
        activation: Callable[[torch.Tensor], torch.Tensor],
        compile_variant: bool,
        warmup: int,
        iters: int,
        atol: float,
        rtol: float,
        cuda_graph_replay: bool,
) -> dict[str, dict[str, Any]]:
    x = fc1_capture.input
    fc1_weight = fc1_capture.weight
    fc1_bias = fc1_capture.bias
    fc2_weight = fc2_capture.weight
    fc2_bias = fc2_capture.bias
    x2d = x.reshape(-1, x.shape[-1])
    x1d = x2d.reshape(-1)
    fc1_weight_t = fc1_weight.t()
    fc2_weight_t = fc2_weight.t()

    def functional_mlp() -> torch.Tensor:
        return F.linear(activation(F.linear(x, fc1_weight, fc1_bias)), fc2_weight, fc2_bias)

    def addmm_mlp() -> torch.Tensor:
        if fc1_bias is None:
            hidden = torch.mm(x2d, fc1_weight_t)
        else:
            hidden = torch.addmm(fc1_bias, x2d, fc1_weight_t)
        hidden = activation(hidden)
        if fc2_bias is None:
            output = torch.mm(hidden, fc2_weight_t)
        else:
            output = torch.addmm(fc2_bias, hidden, fc2_weight_t)
        return output.reshape(*x.shape[:-1], fc2_weight.shape[0])

    def mv_mlp() -> torch.Tensor:
        hidden = torch.mv(fc1_weight, x1d)
        if fc1_bias is not None:
            hidden = hidden + fc1_bias
        hidden = activation(hidden)
        output = torch.mv(fc2_weight, hidden)
        if fc2_bias is not None:
            output = output + fc2_bias
        return output.reshape(*x.shape[:-1], fc2_weight.shape[0])

    reference = functional_mlp()
    variants = {
        "functional_mlp": functional_mlp,
        "addmm_mlp": addmm_mlp,
        "mv_mlp": mv_mlp,
    }
    if compile_variant:
        variants["compiled_functional_mlp"] = torch.compile(
            functional_mlp,
            mode="reduce-overhead",
        )
    return _benchmark_variants(
        reference=reference,
        variants=variants,
        warmup=warmup,
        iters=iters,
        atol=atol,
        rtol=rtol,
        cuda_graph_replay=cuda_graph_replay,
    )


@torch.no_grad()
def profile_decode_linear_kernels(
        args,
        *,
        sequence_index: int,
        active_prefix_bucket_size: int,
        active_prefix_decode_length: int | None,
        q1_bmm_cross_attention: bool,
        native_q1_self_attention: bool,
        native_q1_rope_cache_self_attention: bool,
        compile_mlp_variant: bool,
        native_linear_variant: bool,
        cuda_graph_replay: bool,
        warmup: int,
        iters: int,
        atol: float,
        rtol: float,
        per_module: bool,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Decode linear kernel profiling requires CUDA")

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
        "native_q1_self_attention": bool(native_q1_self_attention),
        "native_q1_rope_cache_self_attention": bool(native_q1_rope_cache_self_attention),
        "compile_mlp_variant": bool(compile_mlp_variant),
        "native_linear_variant": bool(native_linear_variant),
        "cuda_graph_replay": bool(cuda_graph_replay),
        "per_module": bool(per_module),
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
                native_q1_self_attention=native_q1_self_attention,
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
        captures, replay_logits = _capture_decoder_linears(
            model,
            direct_result.prepared_inputs,
            active_prefix_length=active_prefix_length,
            q1_bmm_cross_attention=q1_bmm_cross_attention,
            native_q1_self_attention=native_q1_self_attention,
            native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
            sdpa_backend=args.profile_sdpa_backend,
        )

    signatures: dict[str, dict[str, Any]] = {}
    signature_members: dict[str, list[str]] = defaultdict(list)
    for name, capture in captures.items():
        signature = _signature_for_capture(capture)
        signature_members[signature].append(name)
        signatures.setdefault(
            signature,
            {
                "input_shape": list(capture.input.shape),
                "weight_shape": list(capture.weight.shape),
                "bias": capture.bias is not None,
                "operation_kinds": sorted({_operation_kind(name)}),
            },
        )
        kind = _operation_kind(name)
        if kind not in signatures[signature]["operation_kinds"]:
            signatures[signature]["operation_kinds"].append(kind)
            signatures[signature]["operation_kinds"].sort()

    unique_benchmarks: dict[str, dict[str, Any]] = {}
    per_module_benchmarks: dict[str, dict[str, Any]] = {}
    operation_counts: dict[str, int] = defaultdict(int)
    for name in captures:
        operation_counts[_operation_kind(name)] += 1

    for signature, names in sorted(signature_members.items()):
        capture = captures[names[0]]
        variants = _linear_variants(capture, native_linear_variant=native_linear_variant)
        unique_benchmarks[signature] = {
            "representative": names[0],
            "member_count": len(names),
            "members": sorted(names),
            "operation_kinds": signatures[signature]["operation_kinds"],
            "results": _benchmark_variants(
                reference=capture.output,
                variants=variants,
                warmup=warmup,
                iters=iters,
                atol=atol,
                rtol=rtol,
                cuda_graph_replay=cuda_graph_replay,
            ),
        }

    if per_module:
        for name, capture in sorted(captures.items()):
            per_module_benchmarks[name] = {
                "operation_kind": _operation_kind(name),
                "signature": _signature_for_capture(capture),
                "results": _benchmark_variants(
                    reference=capture.output,
                    variants=_linear_variants(capture, native_linear_variant=native_linear_variant),
                    warmup=warmup,
                    iters=iters,
                    atol=atol,
                    rtol=rtol,
                    cuda_graph_replay=cuda_graph_replay,
                ),
            }

    mlp_benchmarks: dict[str, dict[str, Any]] = {}
    mlp_inputs = _find_layer_mlp_inputs(captures)
    decoder_layers = _find_decoder_layer_modules(model)
    for layer_key, (fc1_capture, fc2_capture) in sorted(mlp_inputs.items()):
        if mlp_benchmarks:
            break
        layer_index = int(layer_key.split(".")[-1])
        if layer_index not in decoder_layers:
            raise RuntimeError(f"captured MLP inputs for decoder layer {layer_index}, but could not find that layer module")
        mlp_benchmarks[layer_key] = {
            "fc1_signature": _signature_for_capture(fc1_capture),
            "fc2_signature": _signature_for_capture(fc2_capture),
            "results": _benchmark_mlp_variants(
                fc1_capture=fc1_capture,
                fc2_capture=fc2_capture,
                activation=decoder_layers[layer_index].activation_fn,
                compile_variant=compile_mlp_variant,
                warmup=warmup,
                iters=iters,
                atol=atol,
                rtol=rtol,
                cuda_graph_replay=cuda_graph_replay,
            ),
        }

    return {
        "pass": bool(_allclose(direct_result.logits, replay_logits, atol=atol, rtol=rtol)),
        "prompt_tokens": prompt_len,
        "probe_token_id": int(probe_token.item()),
        "full_prefix_tokens": int(full_prefix.shape[-1]),
        "cache_position": [int(item) for item in cache_position.detach().cpu().tolist()],
        "max_cache_len": max_cache_len,
        "computed_active_prefix_length": computed_active_prefix_length,
        "active_prefix_length": active_prefix_length,
        "captured_linear_count": len(captures),
        "operation_counts": dict(sorted(operation_counts.items())),
        "signatures": signatures,
        "unique_linear_benchmarks": unique_benchmarks,
        "per_module_benchmarks": per_module_benchmarks,
        "mlp_benchmarks": mlp_benchmarks,
        "logits_replay_allclose": bool(_allclose(direct_result.logits, replay_logits, atol=atol, rtol=rtol)),
        "logits_replay_max_abs": _max_abs(direct_result.logits, replay_logits),
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark real one-token decoder Linear/MLP call shapes captured from a direct "
            "Mapperatorinator decode step. Diagnostic only; not an inference speed claim."
        )
    )
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--sequence-index", type=int, default=9)
    parser.add_argument("--active-prefix-bucket-size", type=int, default=64)
    parser.add_argument("--active-prefix-decode-length", type=int, default=None)
    parser.add_argument("--q1-bmm-cross-attention", action="store_true")
    parser.add_argument("--native-q1-self-attention", action="store_true")
    parser.add_argument("--native-q1-rope-cache-self-attention", action="store_true")
    parser.add_argument(
        "--compile-mlp-variant",
        action="store_true",
        help="Also benchmark a diagnostic torch.compile(mode='reduce-overhead') MLP island variant.",
    )
    parser.add_argument(
        "--native-linear-variant",
        action="store_true",
        help="Also benchmark diagnostic native CUDA one-token linear kernels for captured Linear shapes.",
    )
    parser.add_argument(
        "--cuda-graph-replay",
        action="store_true",
        help="Also capture each linear/MLP diagnostic variant in a CUDA graph and benchmark replay time.",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--per-module", action="store_true")
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model_path=/path/to/model")
    cli_args = parser.parse_args()

    start = time.perf_counter()
    args = _load_args(cli_args.config_name, cli_args.overrides)
    result = profile_decode_linear_kernels(
        args,
        sequence_index=cli_args.sequence_index,
        active_prefix_bucket_size=cli_args.active_prefix_bucket_size,
        active_prefix_decode_length=cli_args.active_prefix_decode_length,
        q1_bmm_cross_attention=cli_args.q1_bmm_cross_attention,
        native_q1_self_attention=cli_args.native_q1_self_attention,
        native_q1_rope_cache_self_attention=cli_args.native_q1_rope_cache_self_attention,
        compile_mlp_variant=cli_args.compile_mlp_variant,
        native_linear_variant=cli_args.native_linear_variant,
        cuda_graph_replay=cli_args.cuda_graph_replay,
        warmup=cli_args.warmup,
        iters=cli_args.iters,
        atol=cli_args.atol,
        rtol=cli_args.rtol,
        per_module=cli_args.per_module,
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
