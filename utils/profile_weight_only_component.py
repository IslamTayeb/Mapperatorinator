"""Size FP16-stored decoder matrices against accepted FP32 real tensors."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from osuT5.osuT5.inference.optimized.scout.weight_only import (  # noqa: E402
    DecoderWeightPack,
    PackedLinear,
    preload_weight_only_extension,
    weight_only_linear_residual,
    weight_only_mlp_residual,
    weight_only_rmsnorm_linear,
)
from utils.profile_native_prefix_dtype_scout import (  # noqa: E402
    CapturedGraph,
    LayerCapture,
    _accepted_main_session_run,
    _all_cache_snapshots,
    _assert_scout_args,
    _cache_from_static_inputs,
    _capture_cuda_graph,
    _capture_representative_layer,
    _load_args,
    _max_abs,
    _observe_prefix_graph,
    _reciprocal_graph_rounds,
    _restore_all_cache,
)


SCHEMA_VERSION = 2
BASELINE_MAIN_SECONDS = 30.068768849
BASELINE_MAIN_TOKENS = 8_294
EXPECTED_DECODE_REPLAYS = 8_207
EXPECTED_PREFILLS = 87
SAVING_TARGET_SECONDS = 3.5
DECODER_LAYERS = 12
OUTPUTS_PER_BLOCK_CHOICES = (2, 4, 8)
REGIONS = (
    "self_norm_qkv",
    "self_out_residual",
    "cross_norm_q",
    "cross_out_residual",
    "mlp",
    "final_norm_logits",
)
SELECTIVE_WEIGHT_ONLY_REGIONS = frozenset(
    ("self_norm_qkv", "self_out_residual", "mlp", "final_norm_logits")
)


def _candidate_configurations(region: str) -> dict[str, dict[str, int]]:
    if region not in REGIONS:
        raise ValueError(f"unknown weight-only region {region!r}")
    if region == "mlp":
        return {
            f"fc1_{fc1}_fc2_{fc2}": {
                "fc1_outputs_per_block": fc1,
                "fc2_outputs_per_block": fc2,
            }
            for fc1 in OUTPUTS_PER_BLOCK_CHOICES
            for fc2 in OUTPUTS_PER_BLOCK_CHOICES
        }
    return {
        str(value): {"outputs_per_block": value}
        for value in OUTPUTS_PER_BLOCK_CHOICES
    }


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _eps(module: Any) -> float:
    value = getattr(module, "eps", None)
    return float(torch.finfo(torch.float32).eps if value is None else value)


def _validate_live_workload(
    run: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    processor = run.get("processor")
    stats = getattr(processor, "last_generation_stats", None)
    if not isinstance(stats, dict):
        raise RuntimeError("weight-only scout requires main generation statistics")
    generated_tokens = int(stats.get("generated_tokens", -1))
    if generated_tokens != BASELINE_MAIN_TOKENS:
        raise RuntimeError(
            "weight-only scout main token count changed: "
            f"expected {BASELINE_MAIN_TOKENS}, got {generated_tokens}"
        )

    session = run.get("session")
    graph_cache = getattr(session, "graph_cache", None)
    if not isinstance(graph_cache, dict) or not graph_cache:
        raise RuntimeError("weight-only scout requires a non-empty live graph cache")
    entries: dict[int, dict[str, Any]] = {}
    for raw_entry in graph_cache.values():
        if not isinstance(raw_entry, dict):
            raise TypeError("live graph-cache entries must be dictionaries")
        required = (
            "active_prefix_length",
            "graph",
            "outputs",
            "static_inputs",
            "decode_replays",
        )
        missing = [name for name in required if name not in raw_entry]
        if missing:
            raise RuntimeError(f"live graph-cache entry is missing {missing}")
        prefix = int(raw_entry["active_prefix_length"])
        if prefix <= 0:
            raise RuntimeError(f"live graph-cache prefix must be positive, got {prefix}")
        if prefix in entries:
            raise RuntimeError(f"live graph cache repeats prefix {prefix}")
        count = int(raw_entry["decode_replays"])
        if count <= 0:
            raise RuntimeError(
                f"live graph-cache prefix {prefix} has non-positive replay count {count}"
            )
        entries[prefix] = raw_entry

    entries = dict(sorted(entries.items()))
    decode_replays = sum(int(entry["decode_replays"]) for entry in entries.values())
    if decode_replays != EXPECTED_DECODE_REPLAYS:
        raise RuntimeError(
            "weight-only scout decode workload changed: "
            f"expected {EXPECTED_DECODE_REPLAYS}, got {decode_replays}"
        )
    prefills = generated_tokens - decode_replays
    if prefills != EXPECTED_PREFILLS:
        raise RuntimeError(
            "weight-only scout prefill count changed: "
            f"expected {EXPECTED_PREFILLS}, got {prefills}"
        )
    return entries, {
        "generated_tokens": generated_tokens,
        "decode_replays": decode_replays,
        "prefills": prefills,
        "bucket_counts": {
            str(prefix): int(entry["decode_replays"])
            for prefix, entry in entries.items()
        },
    }


def _locate_decoder_and_projection(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Module], torch.nn.Module, torch.nn.Module]:
    from osuT5.osuT5.model.custom_transformers.modeling_varwhisper import (
        VarWhisperDecoderLayer,
    )

    layers = [
        module
        for module in model.modules()
        if isinstance(module, VarWhisperDecoderLayer)
    ]
    if len(layers) != DECODER_LAYERS:
        raise RuntimeError(
            f"weight-only scout requires {DECODER_LAYERS} decoder layers, got {len(layers)}"
        )
    named = dict(model.named_modules())
    layer_names = [name for name, value in named.items() if id(value) == id(layers[0])]
    if len(layer_names) != 1 or ".layers." not in layer_names[0]:
        raise RuntimeError("could not uniquely locate decoder parent")
    decoder_name = layer_names[0].rsplit(".layers.", 1)[0]
    decoder = named[decoder_name]
    final_norm = getattr(decoder, "layer_norm", None)
    projections = {
        id(module): module
        for name, module in named.items()
        if name == "proj_out" or name.endswith(".proj_out")
    }
    if not isinstance(final_norm, torch.nn.Module) or len(projections) != 1:
        raise RuntimeError(
            "could not uniquely locate decoder final norm and output projection"
        )
    return layers, final_norm, next(iter(projections.values()))


def _capture_final_norm_input(
    model: torch.nn.Module,
    final_norm: torch.nn.Module,
    static_inputs: dict[str, Any],
) -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def hook(module, args):
        del module
        if not captured:
            if not args or not isinstance(args[0], torch.Tensor):
                raise TypeError("decoder final norm did not receive a tensor")
            captured.append(args[0].detach().clone())

    handle = final_norm.register_forward_pre_hook(hook)
    try:
        model(**static_inputs, return_dict=True)
        torch.cuda.synchronize()
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("did not capture exactly one decoder final norm input")
    return captured[0]


def _pack_model_weights(
    layers: list[torch.nn.Module],
    projection: torch.nn.Module,
) -> tuple[list[DecoderWeightPack], PackedLinear, dict[str, Any]]:
    torch.cuda.synchronize()
    allocated_before = int(torch.cuda.memory_allocated())
    started = time.perf_counter()
    packs = [DecoderWeightPack.from_layer(layer) for layer in layers]
    output = PackedLinear.from_module(projection)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    allocated_after = int(torch.cuda.memory_allocated())
    layer_source = sum(
        pack.memory_report()["source_weight_bytes"] for pack in packs
    )
    layer_packed = sum(
        pack.memory_report()["packed_weight_bytes"] for pack in packs
    )
    source = layer_source + output.source_weight_bytes
    packed = layer_packed + output.packed_weight_bytes
    return packs, output, {
        "setup_seconds": elapsed,
        "allocated_before_bytes": allocated_before,
        "allocated_after_bytes": allocated_after,
        "observed_resident_increment_bytes": allocated_after - allocated_before,
        "retained_fp32_source_weight_bytes": source,
        "packed_weight_bytes": packed,
        "hypothetical_saving_only_if_fp32_source_freed_bytes": source - packed,
        "fp32_source_weights_retained": True,
    }


def _linear_shape(packed: PackedLinear) -> dict[str, int]:
    if packed.weight.ndim != 2:
        raise ValueError("packed matrix weight must be two-dimensional")
    return {
        "input_features": int(packed.weight.shape[1]),
        "output_features": int(packed.weight.shape[0]),
    }


def _region_matrix_shapes(
    pack: DecoderWeightPack,
    output: PackedLinear,
) -> dict[str, list[dict[str, int]]]:
    return {
        "self_norm_qkv": [_linear_shape(pack.self_qkv)],
        "self_out_residual": [_linear_shape(pack.self_out)],
        "cross_norm_q": [_linear_shape(pack.cross_q)],
        "cross_out_residual": [_linear_shape(pack.cross_out)],
        "mlp": [_linear_shape(pack.fc1), _linear_shape(pack.fc2)],
        "final_norm_logits": [_linear_shape(output)],
    }


def _validate_uniform_region_shapes(
    packs: list[DecoderWeightPack],
    output: PackedLinear,
) -> dict[str, list[dict[str, int]]]:
    if not packs:
        raise ValueError("weight-only scout requires at least one decoder weight pack")
    expected = _region_matrix_shapes(packs[0], output)
    for layer_idx, pack in enumerate(packs[1:], start=1):
        actual = _region_matrix_shapes(pack, output)
        if actual != expected:
            raise RuntimeError(
                f"decoder layer {layer_idx} matrix shapes differ from layer zero"
            )
    return expected


def _prepare_region_inputs(capture: LayerCapture) -> dict[str, torch.Tensor]:
    from osuT5.osuT5.inference.optimized.scout.native_prefix import (
        _cache_tensors,
        _trim_mask,
        fp32_rms_norm,
        framework_q1_attention,
    )
    from osuT5.osuT5.inference.optimized.kernels.q1_attention import (
        native_q1_rope_cache_attention,
    )
    from osuT5.osuT5.inference.optimized.kernels.decoder_layer import (
        native_one_token_linear_residual,
        native_one_token_rmsnorm_linear,
    )

    module = capture.module
    hidden = capture.hidden_states
    self_attn = module.self_attn
    qkv = F.linear(
        fp32_rms_norm(
            hidden,
            module.self_attn_layer_norm.weight,
            getattr(module.self_attn_layer_norm, "eps", None),
        ),
        self_attn.Wqkv.weight,
        self_attn.Wqkv.bias,
    ).view(1, 1, 3, self_attn.num_heads, self_attn.head_dim)
    self_keys, self_values = _cache_tensors(
        capture.past_key_value, "self", capture.layer_idx
    )
    cos, sin = self_attn.rotary_emb(qkv, position_ids=capture.position_ids)
    self_output = native_q1_rope_cache_attention(
        qkv,
        self_keys,
        self_values,
        cos,
        sin,
        capture.cache_position,
        _trim_mask(capture.attention_mask, capture.active_prefix_length),
        capture.active_prefix_length,
    ).transpose(1, 2).contiguous().view(1, 1, self_attn.all_head_size)
    after_self = hidden + F.linear(
        self_output,
        self_attn.Wo.weight,
        self_attn.Wo.bias,
    )

    cross_attn = module.cross_attn
    query = native_one_token_rmsnorm_linear(
        after_self,
        module.cross_attn_layer_norm.weight,
        cross_attn.Wq.weight,
        cross_attn.Wq.bias,
        eps=_eps(module.cross_attn_layer_norm),
        outputs_per_block=8,
    ).view(1, 1, cross_attn.num_heads, cross_attn.head_dim).transpose(1, 2)
    cross_keys, cross_values = _cache_tensors(
        capture.past_key_value, "cross", capture.layer_idx
    )
    cross_output = framework_q1_attention(
        query, cross_keys, cross_values, None
    ).transpose(1, 2).contiguous().view(1, 1, cross_attn.all_head_size)
    after_cross = native_one_token_linear_residual(
        cross_output,
        after_self,
        cross_attn.Wo.weight,
        cross_attn.Wo.bias,
        outputs_per_block=8,
    )
    return {
        "layer_input": hidden,
        "self_attention_output": self_output,
        "after_self": after_self,
        "cross_attention_output": cross_output,
        "after_cross": after_cross,
    }


def _accepted_component_prefix(capture: LayerCapture) -> torch.Tensor:
    """Mirror the accepted FP32 layer topology for one captured real input."""

    from osuT5.osuT5.inference.optimized.kernels.decoder_layer import (
        native_one_token_linear_residual,
        native_one_token_mlp_residual,
        native_one_token_rmsnorm_linear,
    )
    from osuT5.osuT5.inference.optimized.kernels.q1_attention import (
        native_q1_rope_cache_attention,
    )
    from osuT5.osuT5.inference.optimized.scout.native_prefix import (
        _cache_tensors,
        _trim_mask,
        fp32_rms_norm,
        framework_q1_attention,
    )

    module = capture.module
    self_attn = module.self_attn
    residual = capture.hidden_states
    qkv = F.linear(
        fp32_rms_norm(
            residual,
            module.self_attn_layer_norm.weight,
            getattr(module.self_attn_layer_norm, "eps", None),
        ),
        self_attn.Wqkv.weight,
        self_attn.Wqkv.bias,
    ).view(1, 1, 3, self_attn.num_heads, self_attn.head_dim)
    self_keys, self_values = _cache_tensors(
        capture.past_key_value, "self", capture.layer_idx
    )
    cos, sin = self_attn.rotary_emb(qkv, position_ids=capture.position_ids)
    self_output = native_q1_rope_cache_attention(
        qkv,
        self_keys,
        self_values,
        cos,
        sin,
        capture.cache_position,
        _trim_mask(capture.attention_mask, capture.active_prefix_length),
        capture.active_prefix_length,
    ).transpose(1, 2).contiguous().view(1, 1, self_attn.all_head_size)
    hidden = residual + F.linear(
        self_output, self_attn.Wo.weight, self_attn.Wo.bias
    )

    cross_attn = module.cross_attn
    query = native_one_token_rmsnorm_linear(
        hidden,
        module.cross_attn_layer_norm.weight,
        cross_attn.Wq.weight,
        cross_attn.Wq.bias,
        eps=_eps(module.cross_attn_layer_norm),
        outputs_per_block=8,
    ).view(1, 1, cross_attn.num_heads, cross_attn.head_dim).transpose(1, 2)
    cross_keys, cross_values = _cache_tensors(
        capture.past_key_value, "cross", capture.layer_idx
    )
    cross_output = framework_q1_attention(
        query, cross_keys, cross_values, None
    ).transpose(1, 2).contiguous().view(1, 1, cross_attn.all_head_size)
    hidden = native_one_token_linear_residual(
        cross_output,
        hidden,
        cross_attn.Wo.weight,
        cross_attn.Wo.bias,
        outputs_per_block=8,
    )
    return native_one_token_mlp_residual(
        hidden,
        module.final_layer_norm.weight,
        module.fc1.weight,
        module.fc1.bias,
        module.fc2.weight,
        module.fc2.bias,
        eps=_eps(module.final_layer_norm),
        outputs_per_block=8,
    )


def _accepted_region_functions(
    capture: LayerCapture,
    inputs: dict[str, torch.Tensor],
    final_input: torch.Tensor,
    final_norm: torch.nn.Module,
    projection: torch.nn.Module,
) -> dict[str, Callable[[], torch.Tensor]]:
    from osuT5.osuT5.inference.optimized.kernels.decoder_layer import (
        native_one_token_linear_residual,
        native_one_token_mlp_residual,
        native_one_token_rmsnorm_linear,
    )
    from osuT5.osuT5.inference.optimized.scout.native_prefix import fp32_rms_norm

    module = capture.module
    return {
        "self_norm_qkv": lambda: F.linear(
            fp32_rms_norm(
                inputs["layer_input"],
                module.self_attn_layer_norm.weight,
                getattr(module.self_attn_layer_norm, "eps", None),
            ),
            module.self_attn.Wqkv.weight,
            module.self_attn.Wqkv.bias,
        ),
        "self_out_residual": lambda: inputs["layer_input"]
        + F.linear(
            inputs["self_attention_output"],
            module.self_attn.Wo.weight,
            module.self_attn.Wo.bias,
        ),
        "cross_norm_q": lambda: native_one_token_rmsnorm_linear(
            inputs["after_self"],
            module.cross_attn_layer_norm.weight,
            module.cross_attn.Wq.weight,
            module.cross_attn.Wq.bias,
            eps=_eps(module.cross_attn_layer_norm),
            outputs_per_block=8,
        ),
        "cross_out_residual": lambda: native_one_token_linear_residual(
            inputs["cross_attention_output"],
            inputs["after_self"],
            module.cross_attn.Wo.weight,
            module.cross_attn.Wo.bias,
            outputs_per_block=8,
        ),
        "mlp": lambda: native_one_token_mlp_residual(
            inputs["after_cross"],
            module.final_layer_norm.weight,
            module.fc1.weight,
            module.fc1.bias,
            module.fc2.weight,
            module.fc2.bias,
            eps=_eps(module.final_layer_norm),
            outputs_per_block=8,
        ),
        "final_norm_logits": lambda: F.linear(
            fp32_rms_norm(
                final_input,
                final_norm.weight,
                getattr(final_norm, "eps", None),
            ),
            projection.weight,
            projection.bias,
        ),
    }


def _candidate_region_functions(
    capture: LayerCapture,
    pack: DecoderWeightPack,
    inputs: dict[str, torch.Tensor],
    final_input: torch.Tensor,
    final_norm: torch.nn.Module,
    output_pack: PackedLinear,
    *,
    outputs_per_block: int,
    mlp_fc2_outputs_per_block: int | None = None,
) -> dict[str, Callable[[], torch.Tensor]]:
    module = capture.module
    return {
        "self_norm_qkv": lambda: weight_only_rmsnorm_linear(
            inputs["layer_input"],
            module.self_attn_layer_norm.weight,
            pack.self_qkv,
            eps=_eps(module.self_attn_layer_norm),
            outputs_per_block=outputs_per_block,
        ),
        "self_out_residual": lambda: weight_only_linear_residual(
            inputs["self_attention_output"],
            inputs["layer_input"],
            pack.self_out,
            outputs_per_block=outputs_per_block,
        ),
        "cross_norm_q": lambda: weight_only_rmsnorm_linear(
            inputs["after_self"],
            module.cross_attn_layer_norm.weight,
            pack.cross_q,
            eps=_eps(module.cross_attn_layer_norm),
            outputs_per_block=outputs_per_block,
        ),
        "cross_out_residual": lambda: weight_only_linear_residual(
            inputs["cross_attention_output"],
            inputs["after_self"],
            pack.cross_out,
            outputs_per_block=outputs_per_block,
        ),
        "mlp": lambda: weight_only_mlp_residual(
            inputs["after_cross"],
            module.final_layer_norm.weight,
            pack.fc1,
            pack.fc2,
            eps=_eps(module.final_layer_norm),
            fc1_outputs_per_block=outputs_per_block,
            fc2_outputs_per_block=(
                outputs_per_block
                if mlp_fc2_outputs_per_block is None
                else mlp_fc2_outputs_per_block
            ),
        ),
        "final_norm_logits": lambda: weight_only_rmsnorm_linear(
            final_input,
            final_norm.weight,
            output_pack,
            eps=_eps(final_norm),
            outputs_per_block=outputs_per_block,
        ),
    }


def _observe_graph_tensor(graph: CapturedGraph) -> torch.Tensor:
    graph.graph.replay()
    torch.cuda.synchronize()
    if not isinstance(graph.outputs, torch.Tensor):
        raise TypeError("region graph must expose one tensor output")
    return graph.outputs.detach().float().cpu().clone()


def _profile_candidate_sweep(
    accepted: Callable[[], torch.Tensor],
    candidates: dict[str, Callable[[], torch.Tensor]],
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("candidate sweep requires at least one configuration")
    graphs = {"accepted": _capture_cuda_graph(accepted, context=nullcontext, warmup=0)}
    graphs.update(
        {
            f"candidate_{key}": _capture_cuda_graph(
                candidate,
                context=nullcontext,
                warmup=0,
            )
            for key, candidate in sorted(candidates.items())
        }
    )
    timings, rounds, stable = _reciprocal_graph_rounds(
        {name: value.graph for name, value in graphs.items()},
        restore=lambda: None,
        warmup=warmup,
        iters=iters,
    )
    observations = {name: _observe_graph_tensor(value) for name, value in graphs.items()}
    accepted_observation = observations["accepted"]
    result = {
        "accepted": {
            "ms_per_call": float(timings["accepted"]),
            "graph_capture_seconds": float(graphs["accepted"].setup_seconds),
            "absolute_peak_vram_bytes": int(graphs["accepted"].peak_vram_bytes),
            "memory_stable": bool(stable["accepted"]),
            "finite": bool(torch.isfinite(accepted_observation).all()),
        },
        "candidates": {
            key: {
                "ms_per_call": float(timings[f"candidate_{key}"]),
                "saving_ms_per_call": float(
                    timings["accepted"] - timings[f"candidate_{key}"]
                ),
                "graph_capture_seconds": float(
                    graphs[f"candidate_{key}"].setup_seconds
                ),
                "absolute_peak_vram_bytes": int(
                    graphs[f"candidate_{key}"].peak_vram_bytes
                ),
                "memory_stable": bool(stable[f"candidate_{key}"]),
                "finite": bool(
                    torch.isfinite(observations[f"candidate_{key}"]).all()
                ),
                "output_max_abs_drift": _max_abs(
                    accepted_observation,
                    observations[f"candidate_{key}"],
                ),
            }
            for key in sorted(candidates)
        },
        "rounds": rounds,
    }
    del graphs
    torch.cuda.empty_cache()
    return result


def _selected_region_result(
    sweep: dict[str, Any],
    *,
    candidate_key: str,
    configuration: dict[str, int],
) -> dict[str, Any]:
    accepted = sweep["accepted"]
    candidate = sweep["candidates"][candidate_key]
    return {
        "selected_candidate_key": candidate_key,
        "selected_configuration": configuration,
        "accepted_ms_per_call": float(accepted["ms_per_call"]),
        "candidate_ms_per_call": float(candidate["ms_per_call"]),
        "saving_ms_per_call": float(candidate["saving_ms_per_call"]),
        "accepted_graph_capture_seconds": float(accepted["graph_capture_seconds"]),
        "candidate_graph_capture_seconds": float(candidate["graph_capture_seconds"]),
        "accepted_absolute_peak_vram_bytes": int(
            accepted["absolute_peak_vram_bytes"]
        ),
        "candidate_absolute_peak_vram_bytes": int(
            candidate["absolute_peak_vram_bytes"]
        ),
        "accepted_memory_stable": bool(accepted["memory_stable"]),
        "candidate_memory_stable": bool(candidate["memory_stable"]),
        "accepted_finite": bool(accepted["finite"]),
        "candidate_finite": bool(candidate["finite"]),
        "output_max_abs_drift": float(candidate["output_max_abs_drift"]),
        "opb_sweep": sweep,
    }


def _select_weighted_outputs_per_block(
    buckets: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not buckets:
        raise ValueError("outputs-per-block selection requires live bucket evidence")
    result: dict[str, dict[str, Any]] = {}
    for region in REGIONS:
        multiplier = 1 if region == "final_norm_logits" else DECODER_LAYERS
        configurations = _candidate_configurations(region)
        candidate_seconds: dict[str, float] = {}
        for candidate_key in configurations:
            weighted_ms = 0.0
            valid = True
            for entry in buckets.values():
                count = int(entry["decode_replays"])
                candidates = entry["regions"][region]["candidates"]
                if set(candidates) != set(configurations):
                    raise ValueError(
                        f"bucket candidate configurations changed for {region}"
                    )
                candidate = candidates[candidate_key]
                valid &= bool(candidate["memory_stable"] and candidate["finite"])
                weighted_ms += (
                    count * multiplier * float(candidate["ms_per_call"])
                )
            candidate_seconds[candidate_key] = (
                weighted_ms / 1_000.0 if valid else math.inf
            )
        selected_key = min(
            configurations,
            key=lambda value: (candidate_seconds[value], value),
        )
        if not math.isfinite(candidate_seconds[selected_key]):
            raise RuntimeError(
                f"no structurally valid outputs-per-block candidate for {region}"
            )
        result[region] = {
            "selected_candidate_key": selected_key,
            "selected_configuration": configurations[selected_key],
            "weighted_candidate_seconds": candidate_seconds,
            "selective_combined_dispatch": (
                "fp16_weight" if region in SELECTIVE_WEIGHT_ONLY_REGIONS else "accepted_fp32"
            ),
        }
    return result


def _selective_weight_only_prefix(
    capture: LayerCapture,
    pack: DecoderWeightPack,
    *,
    configurations: dict[str, dict[str, int]],
) -> torch.Tensor:
    """Measure selected weight-only self/MLP regions with accepted FP32 cross attention."""

    from osuT5.osuT5.inference.optimized.kernels.decoder_layer import (
        native_one_token_linear_residual,
        native_one_token_rmsnorm_linear,
    )
    from osuT5.osuT5.inference.optimized.kernels.q1_attention import (
        native_q1_rope_cache_attention,
    )
    from osuT5.osuT5.inference.optimized.scout.native_prefix import (
        _cache_tensors,
        _trim_mask,
        framework_q1_attention,
    )

    module = capture.module
    self_attn = module.self_attn
    residual = capture.hidden_states
    qkv = weight_only_rmsnorm_linear(
        residual,
        module.self_attn_layer_norm.weight,
        pack.self_qkv,
        eps=_eps(module.self_attn_layer_norm),
        outputs_per_block=configurations["self_norm_qkv"]["outputs_per_block"],
    ).view(1, 1, 3, self_attn.num_heads, self_attn.head_dim)
    self_keys, self_values = _cache_tensors(
        capture.past_key_value,
        "self",
        capture.layer_idx,
    )
    cos, sin = self_attn.rotary_emb(qkv, position_ids=capture.position_ids)
    self_output = native_q1_rope_cache_attention(
        qkv,
        self_keys,
        self_values,
        cos,
        sin,
        capture.cache_position,
        _trim_mask(capture.attention_mask, capture.active_prefix_length),
        capture.active_prefix_length,
    ).transpose(1, 2).contiguous().view(1, 1, self_attn.all_head_size)
    hidden = weight_only_linear_residual(
        self_output,
        residual,
        pack.self_out,
        outputs_per_block=configurations["self_out_residual"]["outputs_per_block"],
    )

    cross_attn = module.cross_attn
    query = native_one_token_rmsnorm_linear(
        hidden,
        module.cross_attn_layer_norm.weight,
        cross_attn.Wq.weight,
        cross_attn.Wq.bias,
        eps=_eps(module.cross_attn_layer_norm),
        outputs_per_block=8,
    ).view(1, 1, cross_attn.num_heads, cross_attn.head_dim).transpose(1, 2)
    cross_keys, cross_values = _cache_tensors(
        capture.past_key_value,
        "cross",
        capture.layer_idx,
    )
    cross_output = framework_q1_attention(
        query,
        cross_keys,
        cross_values,
        None,
    ).transpose(1, 2).contiguous().view(1, 1, cross_attn.all_head_size)
    hidden = native_one_token_linear_residual(
        cross_output,
        hidden,
        cross_attn.Wo.weight,
        cross_attn.Wo.bias,
        outputs_per_block=8,
    )
    return weight_only_mlp_residual(
        hidden,
        module.final_layer_norm.weight,
        pack.fc1,
        pack.fc2,
        eps=_eps(module.final_layer_norm),
        fc1_outputs_per_block=configurations["mlp"]["fc1_outputs_per_block"],
        fc2_outputs_per_block=configurations["mlp"]["fc2_outputs_per_block"],
    )


def summarize_weight_only_component(
    buckets: dict[str, dict[str, Any]],
    *,
    total_replays: int,
) -> dict[str, Any]:
    if not buckets or total_replays <= 0:
        raise ValueError("weight-only summary requires buckets and positive total replays")
    measured_replays = 0
    layer_accepted_ms = 0.0
    layer_candidate_ms = 0.0
    logits_accepted_ms = 0.0
    logits_candidate_ms = 0.0
    region_seconds = {region: 0.0 for region in REGIONS}
    structural_failures: dict[str, list[str]] = {}
    selected_drift = {
        "prefix_output_max_abs": 0.0,
        "cache_key_slot_max_abs": 0.0,
        "cache_value_slot_max_abs": 0.0,
        "logits_max_abs": 0.0,
    }
    diagnostic_region_drift = {region: 0.0 for region in REGIONS}
    selected_graph_recapture_seconds = 0.0
    paired_selected_graph_capture_seconds = 0.0
    diagnostic_candidate_graph_capture_seconds = 0.0
    selected_candidate_absolute_peak_vram_bytes = 0
    diagnostic_candidate_absolute_peak_vram_bytes = 0
    for bucket, entry in sorted(buckets.items(), key=lambda item: int(item[0])):
        count = int(entry["decode_replays"])
        if count <= 0:
            raise ValueError(f"bucket {bucket} has non-positive replay count")
        measured_replays += count
        prefix = entry["prefix_total"]
        layer_accepted_ms += count * DECODER_LAYERS * float(
            prefix["accepted_ms_per_call"]
        )
        layer_candidate_ms += count * DECODER_LAYERS * float(
            prefix["candidate_ms_per_call"]
        )
        bucket_failures: list[str] = []
        for name, region in entry["regions_selected"].items():
            multiplier = 1 if name == "final_norm_logits" else DECODER_LAYERS
            region_seconds[name] += (
                count * multiplier * float(region["saving_ms_per_call"]) / 1_000.0
            )
            diagnostic_region_drift[name] = max(
                diagnostic_region_drift[name],
                float(region["output_max_abs_drift"]),
            )
            if (
                name in SELECTIVE_WEIGHT_ONLY_REGIONS
                and not region["candidate_memory_stable"]
            ):
                bucket_failures.append(f"{name}:memory_unstable")
            if name in SELECTIVE_WEIGHT_ONLY_REGIONS and not region["candidate_finite"]:
                bucket_failures.append(f"{name}:nonfinite")
            for candidate in entry["regions"][name]["candidates"].values():
                diagnostic_candidate_graph_capture_seconds += float(
                    candidate["graph_capture_seconds"]
                )
                diagnostic_candidate_absolute_peak_vram_bytes = max(
                    diagnostic_candidate_absolute_peak_vram_bytes,
                    int(candidate["absolute_peak_vram_bytes"]),
                )
        logits = entry["selected_final_norm_logits"]
        logits_accepted_ms += count * float(logits["accepted_ms_per_call"])
        logits_candidate_ms += count * float(logits["candidate_ms_per_call"])
        selected_drift["prefix_output_max_abs"] = max(
            selected_drift["prefix_output_max_abs"],
            float(prefix["output_max_abs_drift"]),
        )
        selected_drift["cache_key_slot_max_abs"] = max(
            selected_drift["cache_key_slot_max_abs"],
            float(prefix["cache_key_slot_max_abs_drift"]),
        )
        selected_drift["cache_value_slot_max_abs"] = max(
            selected_drift["cache_value_slot_max_abs"],
            float(prefix["cache_value_slot_max_abs_drift"]),
        )
        selected_drift["logits_max_abs"] = max(
            selected_drift["logits_max_abs"],
            float(logits["output_max_abs_drift"]),
        )
        if not prefix["candidate_memory_stable"]:
            bucket_failures.append("prefix_total:memory_unstable")
        if not prefix["candidate_finite"]:
            bucket_failures.append("prefix_total:nonfinite")
        if not prefix.get("candidate_cache_behavior_pass", False):
            bucket_failures.append("prefix_total:cache_behavior_failed")
        if bucket_failures:
            structural_failures[bucket] = bucket_failures
        selected_graph_recapture_seconds += float(
            prefix["candidate_graph_capture_seconds"]
        ) + float(logits["candidate_graph_capture_seconds"])
        paired_selected_graph_capture_seconds += (
            float(prefix["accepted_graph_capture_seconds"])
            + float(prefix["candidate_graph_capture_seconds"])
            + float(logits["accepted_graph_capture_seconds"])
            + float(logits["candidate_graph_capture_seconds"])
        )
        selected_candidate_absolute_peak_vram_bytes = max(
            selected_candidate_absolute_peak_vram_bytes,
            int(prefix["candidate_absolute_peak_vram_bytes"]),
            int(logits["candidate_absolute_peak_vram_bytes"]),
        )
    if measured_replays != total_replays:
        raise ValueError(
            "weight-only summary requires all live replay buckets: "
            f"measured {measured_replays}, expected {total_replays}"
        )
    accepted_seconds = (layer_accepted_ms + logits_accepted_ms) / 1_000.0
    candidate_seconds = (layer_candidate_ms + logits_candidate_ms) / 1_000.0
    saving_seconds = accepted_seconds - candidate_seconds
    projected_main = BASELINE_MAIN_SECONDS - saving_seconds
    return {
        "measured_replays": measured_replays,
        "total_replays": int(total_replays),
        "coverage_fraction": measured_replays / total_replays,
        "weighted_fixed_work_accepted_seconds": accepted_seconds,
        "weighted_fixed_work_candidate_seconds": candidate_seconds,
        "weighted_fixed_work_saving_seconds": saving_seconds,
        "region_saving_seconds_diagnostic": region_seconds,
        "saving_target_seconds": SAVING_TARGET_SECONDS,
        "sizing_pass": saving_seconds >= SAVING_TARGET_SECONDS,
        "projected_main_seconds": projected_main,
        "projected_main_tps_at_8294_tokens": (
            BASELINE_MAIN_TOKENS / projected_main if projected_main > 0 else math.inf
        ),
        "structural_runtime_failures": structural_failures,
        "structural_runtime_pass": not structural_failures,
        "numeric_drift": {
            "selected_combined": {
                **selected_drift,
                "maximum_observed_abs": max(selected_drift.values()),
            },
            "diagnostic_region_output_max_abs": diagnostic_region_drift,
            "diagnostic_maximum_region_output_abs": max(
                diagnostic_region_drift.values()
            ),
            "exactness_claim": False,
            "interpretation": "documented FP16-weight numerical drift",
        },
        "setup": {
            "selected_candidate_graph_recapture_seconds": (
                selected_graph_recapture_seconds
            ),
            "paired_selected_graph_capture_seconds": (
                paired_selected_graph_capture_seconds
            ),
            "diagnostic_candidate_graph_capture_seconds": (
                diagnostic_candidate_graph_capture_seconds
            ),
        },
        "vram": {
            "selected_candidate_absolute_peak_bytes": (
                selected_candidate_absolute_peak_vram_bytes
            ),
            "diagnostic_candidate_absolute_peak_bytes": (
                diagnostic_candidate_absolute_peak_vram_bytes
            ),
        },
    }


@torch.no_grad()
def profile_weight_only_component(
    args,
    *,
    output_path: Path,
    bucket_mode: str,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("weight-only component profiler requires CUDA")
    if warmup < 1 or iters < 1:
        raise ValueError("warmup and iters must be positive")
    _assert_scout_args(args)
    if bucket_mode != "all":
        raise ValueError("weight-only component profiling requires all live buckets")
    output_path.mkdir(parents=True, exist_ok=True)
    run = _accepted_main_session_run(args, output_path=output_path)
    model = run["model"]
    model.eval()
    accepted_entries, live_workload = _validate_live_workload(run)
    total_replays = int(live_workload["decode_replays"])
    layers, final_norm, projection = _locate_decoder_and_projection(model)

    from osuT5.osuT5.inference.optimized.kernels.decoder_layer import (
        preload_native_decoder_layer,
    )

    torch.cuda.synchronize()
    scout_allocated_before = int(torch.cuda.memory_allocated())
    scout_reserved_before = int(torch.cuda.memory_reserved())
    extension_started = time.perf_counter()
    preload_native_decoder_layer()
    preload_weight_only_extension()
    torch.cuda.synchronize()
    extension_preload_seconds = time.perf_counter() - extension_started
    packs, output_pack, memory = _pack_model_weights(layers, projection)
    region_shapes = _validate_uniform_region_shapes(packs, output_pack)

    buckets: dict[str, dict[str, Any]] = {}
    bucket_states: dict[int, dict[str, Any]] = {}
    for prefix, accepted in accepted_entries.items():
        static_inputs = accepted["static_inputs"]
        cache = _cache_from_static_inputs(static_inputs)
        cache_position = static_inputs.get("cache_position")
        if not isinstance(cache_position, torch.Tensor) or cache_position.numel() != 1:
            raise RuntimeError(f"prefix {prefix} has invalid cache_position")
        snapshots = _all_cache_snapshots(cache, cache_position)
        _restore_all_cache(cache, snapshots)
        capture = _capture_representative_layer(model, static_inputs, prefix=prefix)
        _restore_all_cache(cache, snapshots)
        final_input = _capture_final_norm_input(model, final_norm, static_inputs)
        _restore_all_cache(cache, snapshots)
        region_inputs = _prepare_region_inputs(capture)
        _restore_all_cache(cache, snapshots)
        pack = packs[capture.layer_idx]
        accepted_regions = _accepted_region_functions(
            capture,
            region_inputs,
            final_input,
            final_norm,
            projection,
        )
        candidate_regions: dict[str, dict[str, Callable[[], torch.Tensor]]] = {}
        for name in REGIONS:
            candidate_regions[name] = {}
            for candidate_key, configuration in _candidate_configurations(name).items():
                outputs_per_block = int(
                    configuration.get(
                        "outputs_per_block",
                        configuration.get("fc1_outputs_per_block", -1),
                    )
                )
                if outputs_per_block not in OUTPUTS_PER_BLOCK_CHOICES:
                    raise RuntimeError(
                        f"invalid outputs-per-block configuration for {name}"
                    )
                functions = _candidate_region_functions(
                    capture,
                    pack,
                    region_inputs,
                    final_input,
                    final_norm,
                    output_pack,
                    outputs_per_block=outputs_per_block,
                    mlp_fc2_outputs_per_block=configuration.get(
                        "fc2_outputs_per_block"
                    ),
                )
                candidate_regions[name][candidate_key] = functions[name]
        regions = {
            name: _profile_candidate_sweep(
                accepted_regions[name],
                candidate_regions[name],
                warmup=warmup,
                iters=iters,
            )
            for name in REGIONS
        }
        buckets[str(prefix)] = {
            "decode_replays": int(accepted["decode_replays"]),
            "regions": regions,
        }
        bucket_states[prefix] = {
            "capture": capture,
            "cache": cache,
            "snapshots": snapshots,
            "pack": pack,
            "accepted_regions": accepted_regions,
            "candidate_regions": candidate_regions,
        }

    opb_selection = _select_weighted_outputs_per_block(buckets)
    selected_configurations = {
        region: dict(selection["selected_configuration"])
        for region, selection in opb_selection.items()
    }

    for prefix, accepted in accepted_entries.items():
        state = bucket_states[prefix]
        capture = state["capture"]
        cache = state["cache"]
        snapshots = state["snapshots"]
        pack = state["pack"]
        entry = buckets[str(prefix)]
        entry["regions_selected"] = {
            name: _selected_region_result(
                entry["regions"][name],
                candidate_key=str(opb_selection[name]["selected_candidate_key"]),
                configuration=selected_configurations[name],
            )
            for name in REGIONS
        }

        from osuT5.osuT5.inference.optimized.scout.verifier import (
            verify_candidate_cache_behavior,
        )

        def accepted_prefix() -> torch.Tensor:
            return _accepted_component_prefix(capture)

        def candidate_prefix() -> torch.Tensor:
            return _selective_weight_only_prefix(
                capture,
                pack,
                configurations=selected_configurations,
            )

        _restore_all_cache(cache, snapshots)
        candidate_checks = verify_candidate_cache_behavior(
            cache,
            layer_idx=capture.layer_idx,
            cache_position=capture.cache_position,
            candidate=candidate_prefix,
            repeats=2,
        )
        prefix_graphs = {
            "accepted": _capture_cuda_graph(
                accepted_prefix, context=nullcontext, warmup=0
            ),
            "candidate": _capture_cuda_graph(
                candidate_prefix, context=nullcontext, warmup=0
            ),
        }
        prefix_times, prefix_rounds, prefix_stable = _reciprocal_graph_rounds(
            {name: value.graph for name, value in prefix_graphs.items()},
            restore=lambda: _restore_all_cache(cache, snapshots),
            warmup=warmup,
            iters=iters,
        )
        observations = {
            name: _observe_prefix_graph(graph, capture=capture)
            for name, graph in prefix_graphs.items()
        }
        reference = observations["accepted"]
        candidate = observations["candidate"]
        prefix_total = {
            "accepted_ms_per_call": float(prefix_times["accepted"]),
            "candidate_ms_per_call": float(prefix_times["candidate"]),
            "saving_ms_per_call": float(
                prefix_times["accepted"] - prefix_times["candidate"]
            ),
            "accepted_graph_capture_seconds": float(
                prefix_graphs["accepted"].setup_seconds
            ),
            "candidate_graph_capture_seconds": float(
                prefix_graphs["candidate"].setup_seconds
            ),
            "accepted_absolute_peak_vram_bytes": int(
                prefix_graphs["accepted"].peak_vram_bytes
            ),
            "candidate_absolute_peak_vram_bytes": int(
                prefix_graphs["candidate"].peak_vram_bytes
            ),
            "accepted_memory_stable": bool(prefix_stable["accepted"]),
            "candidate_memory_stable": bool(prefix_stable["candidate"]),
            "candidate_finite": bool(
                all(
                    all(repeat["finite"].values())
                    for repeat in candidate_checks["repeats"]
                )
            ),
            "candidate_cache_behavior_pass": bool(candidate_checks["pass"]),
            "output_max_abs_drift": _max_abs(reference.output, candidate.output),
            "cache_key_slot_max_abs_drift": _max_abs(
                reference.key_slot, candidate.key_slot
            ),
            "cache_value_slot_max_abs_drift": _max_abs(
                reference.value_slot, candidate.value_slot
            ),
            "candidate_checks": candidate_checks,
            "rounds": prefix_rounds,
        }
        selected_final_key = str(
            opb_selection["final_norm_logits"]["selected_candidate_key"]
        )
        final_sweep = _profile_candidate_sweep(
            state["accepted_regions"]["final_norm_logits"],
            {
                selected_final_key: state["candidate_regions"]["final_norm_logits"][
                    selected_final_key
                ]
            },
            warmup=warmup,
            iters=iters,
        )
        entry["selected_final_norm_logits"] = _selected_region_result(
            final_sweep,
            candidate_key=selected_final_key,
            configuration=selected_configurations["final_norm_logits"],
        )
        entry["prefix_total"] = prefix_total
        del prefix_graphs
        torch.cuda.empty_cache()

    summary = summarize_weight_only_component(
        buckets,
        total_replays=total_replays,
    )
    torch.cuda.synchronize()
    scout_allocated_after = int(torch.cuda.memory_allocated())
    scout_reserved_after = int(torch.cuda.memory_reserved())
    absolute_peak_vram_bytes = max(
        int(summary["vram"]["diagnostic_candidate_absolute_peak_bytes"]),
        int(summary["vram"]["selected_candidate_absolute_peak_bytes"]),
        scout_allocated_before,
        scout_allocated_after,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "candidate": "fp16_weights_fp32_state_cuda_core_gemv",
            "commit": _git_head(),
            "precision": "fp32_activations_fp16_weights",
            "attention": "unchanged_accepted_fp32",
            "bucket_mode": bucket_mode,
            "live_workload": live_workload,
            "warmup": warmup,
            "iters": iters,
            "outputs_per_block_choices": list(OUTPUTS_PER_BLOCK_CHOICES),
            "region_matrix_shapes": region_shapes,
            "weighted_outputs_per_block_selection": opb_selection,
            "selective_combined_policy": {
                region: (
                    "fp16_weight"
                    if region in SELECTIVE_WEIGHT_ONLY_REGIONS
                    else "accepted_fp32"
                )
                for region in REGIONS
            },
            "setup": {
                "jit_extension_preload_seconds": extension_preload_seconds,
                "weight_pack_seconds": float(memory["setup_seconds"]),
                **summary["setup"],
            },
            "weight_pack": memory,
            "vram": {
                "scout_allocated_before_bytes": scout_allocated_before,
                "scout_allocated_after_bytes": scout_allocated_after,
                "scout_allocated_increment_bytes": (
                    scout_allocated_after - scout_allocated_before
                ),
                "scout_reserved_before_bytes": scout_reserved_before,
                "scout_reserved_after_bytes": scout_reserved_after,
                "scout_reserved_increment_bytes": (
                    scout_reserved_after - scout_reserved_before
                ),
                "absolute_peak_observed_bytes": absolute_peak_vram_bytes,
                **summary["vram"],
            },
        },
        "summary": summary,
        "buckets": buckets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--bucket-mode", choices=("all",), default="all"
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1_000)
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()
    overrides = list(cli.overrides)
    if not any(value.startswith("output_path=") for value in overrides):
        overrides.append(f"output_path={cli.output_path}")
    args = _load_args(cli.config_name, overrides)
    result = profile_weight_only_component(
        args,
        output_path=cli.output_path,
        bucket_mode=cli.bucket_mode,
        warmup=cli.warmup,
        iters=cli.iters,
    )
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    if not result["summary"]["structural_runtime_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
