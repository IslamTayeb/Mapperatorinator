"""Opt-in torch.compile decode-only (1,1) self-attn Wqkv/Wo GEMVs before outer CUDA capture.

Exact FP16/FP32 path only — owned F.linear regions, no native_one_token_linear / INT8.
Call sites must refuse variable-seq prefill (see modeling_varwhisper Wo gate).
"""

from __future__ import annotations

import time
from typing import Callable

import torch

CompiledLinear = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor | None],
    torch.Tensor,
]

_COMPILED_CACHE: dict[
    tuple[int, torch.dtype, int, int],
    tuple[CompiledLinear, dict[str, float | bool | str | int]],
] = {}


def prepare_compiled_one_token_linear(
    device: torch.device,
    dtype: torch.dtype,
    out_features: int,
    in_features: int,
) -> tuple[CompiledLinear, dict[str, float | bool | str | int]]:
    """Compile and warm an exact F.linear region outside graph capture."""

    if not isinstance(device, torch.device) or device.type != "cuda":
        raise RuntimeError("compiled one-token linear requires one CUDA device")
    if dtype not in (torch.float32, torch.float16):
        raise TypeError("compiled one-token linear supports only float32 or float16")
    if (
        isinstance(out_features, bool)
        or isinstance(in_features, bool)
        or not isinstance(out_features, int)
        or not isinstance(in_features, int)
        or out_features <= 0
        or in_features <= 0
    ):
        raise ValueError("compiled one-token linear requires positive feature sizes")
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 5):
        raise RuntimeError(
            f"compiled one-token linear requires SM75, got sm_{capability[0]}{capability[1]}"
        )

    cache_key = (
        device.index if device.index is not None else torch.cuda.current_device(),
        dtype,
        int(out_features),
        int(in_features),
    )
    cached = _COMPILED_CACHE.get(cache_key)
    if cached is not None:
        compiled, evidence = cached
        return compiled, {
            **evidence,
            "cache_hit": True,
            "compile_and_first_call_seconds": 0.0,
        }

    def _linear_region(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return torch.nn.functional.linear(x, weight, bias)

    compiled = torch.compile(
        _linear_region,
        fullgraph=True,
        dynamic=False,
        mode="default",
    )
    x = torch.zeros((1, 1, in_features), device=device, dtype=dtype)
    weight = torch.zeros((out_features, in_features), device=device, dtype=dtype)
    bias = torch.zeros((out_features,), device=device, dtype=dtype)
    eager = _linear_region(x, weight, bias)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    actual = compiled(x, weight, bias)
    torch.cuda.synchronize(device)
    compile_seconds = time.perf_counter() - started
    if not torch.isfinite(actual).all().item():
        raise RuntimeError("compiled one-token linear produced non-finite warmup output")
    bitwise_equal = bool(torch.equal(eager, actual))
    max_abs = float((eager.float() - actual.float()).abs().max().item())
    if not bitwise_equal:
        raise RuntimeError(
            "compiled one-token linear warmup is not bitwise equal to eager "
            f"(max_abs={max_abs})"
        )
    evidence: dict[str, float | bool | str | int] = {
        "mode": "default",
        "fullgraph": True,
        "dynamic": False,
        "dtype": str(dtype).removeprefix("torch."),
        "out_features": int(out_features),
        "in_features": int(in_features),
        "outer_cuda_graph_owned": True,
        "compile_and_first_call_seconds": compile_seconds,
        "warmup_max_abs": max_abs,
        "warmup_bitwise_equal": bitwise_equal,
        "cache_hit": False,
    }
    _COMPILED_CACHE[cache_key] = (compiled, evidence)
    return compiled, dict(evidence)


def prepare_compiled_self_projections(
    model: torch.nn.Module,
    dtype: torch.dtype,
) -> tuple[CompiledLinear, CompiledLinear, dict[str, object]]:
    """Prepare compiled Wqkv and Wo callables from decoder self-attn shapes."""

    from osuT5.osuT5.model.custom_transformers.modeling_varwhisper import (
        VarWhisperDecoderLayer,
    )

    decoder_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, VarWhisperDecoderLayer)
    ]
    if not decoder_layers:
        raise RuntimeError("compiled self proj found no VarWhisper decoder layers")

    shapes: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    first_attn = None
    for _name, layer in decoder_layers:
        self_attn = getattr(layer, "self_attn", None)
        wqkv = getattr(self_attn, "Wqkv", None)
        wo = getattr(self_attn, "Wo", None)
        if wqkv is None or wo is None:
            raise RuntimeError("decoder self-attn missing Wqkv/Wo for compiled self proj")
        shapes.add(
            (
                (int(wqkv.weight.shape[0]), int(wqkv.weight.shape[1])),
                (int(wo.weight.shape[0]), int(wo.weight.shape[1])),
            )
        )
        if first_attn is None:
            first_attn = self_attn
    if len(shapes) != 1:
        raise RuntimeError(
            f"compiled self proj requires uniform Wqkv/Wo shapes, got {shapes}"
        )
    assert first_attn is not None
    device = first_attn.Wqkv.weight.device
    if first_attn.Wo.weight.device != device:
        raise RuntimeError("compiled self proj Wqkv/Wo devices disagree")
    if first_attn.Wqkv.weight.dtype != dtype or first_attn.Wo.weight.dtype != dtype:
        raise TypeError(
            "compiled self proj weight dtype must match expected generation dtype"
        )

    wqkv_out, wqkv_in = (int(first_attn.Wqkv.weight.shape[0]), int(first_attn.Wqkv.weight.shape[1]))
    wo_out, wo_in = (int(first_attn.Wo.weight.shape[0]), int(first_attn.Wo.weight.shape[1]))
    compiled_wqkv, wqkv_evidence = prepare_compiled_one_token_linear(
        device, dtype, wqkv_out, wqkv_in
    )
    compiled_wo, wo_evidence = prepare_compiled_one_token_linear(
        device, dtype, wo_out, wo_in
    )
    evidence: dict[str, object] = {
        "enabled": True,
        "decode_only": True,
        "decode_shape": [1, 1],
        "layer_count": len(decoder_layers),
        "wqkv": wqkv_evidence,
        "wo": wo_evidence,
    }
    return compiled_wqkv, compiled_wo, evidence


__all__ = [
    "CompiledLinear",
    "prepare_compiled_one_token_linear",
    "prepare_compiled_self_projections",
]
