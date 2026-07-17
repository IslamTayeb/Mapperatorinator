"""Opt-in torch.compile decode-only (1,1) self Wo + residual (exact F.linear+add).

Gemm/projection family lever: fuse attention-out GEMV with the following residual
add before outer CUDA-graph capture. Not native_one_token_linear_residual (§6),
not compiled Wqkv (§14), not LM proj_out (§15/§20/§21).
"""

from __future__ import annotations

import time
from typing import Callable

import torch
import torch.nn.functional as F

CompiledWoResidual = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None],
    torch.Tensor,
]

_COMPILED_CACHE: dict[
    tuple[int, torch.dtype, int, int],
    tuple[CompiledWoResidual, dict[str, float | bool | str | int]],
] = {}


def prepare_compiled_self_wo_residual(
    device: torch.device,
    dtype: torch.dtype,
    out_features: int,
    in_features: int,
) -> tuple[CompiledWoResidual, dict[str, float | bool | str | int]]:
    """Compile and warm exact F.linear + residual outside graph capture."""

    if not isinstance(device, torch.device) or device.type != "cuda":
        raise RuntimeError("compiled self Wo+residual requires one CUDA device")
    if dtype not in (torch.float32, torch.float16):
        raise TypeError("compiled self Wo+residual supports only float32 or float16")
    if (
        isinstance(out_features, bool)
        or isinstance(in_features, bool)
        or not isinstance(out_features, int)
        or not isinstance(in_features, int)
        or out_features <= 0
        or in_features <= 0
    ):
        raise ValueError("compiled self Wo+residual requires positive feature sizes")
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 5):
        raise RuntimeError(
            f"compiled self Wo+residual requires SM75, got sm_{capability[0]}{capability[1]}"
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

    def _wo_residual_region(
        attn_flat: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return residual + F.linear(attn_flat, weight, bias)

    compiled = torch.compile(
        _wo_residual_region,
        fullgraph=True,
        dynamic=False,
        mode="default",
    )
    x = torch.zeros((1, 1, in_features), device=device, dtype=dtype)
    residual = torch.zeros((1, 1, out_features), device=device, dtype=dtype)
    weight = torch.zeros((out_features, in_features), device=device, dtype=dtype)
    bias = torch.zeros((out_features,), device=device, dtype=dtype)
    eager = _wo_residual_region(x, residual, weight, bias)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    actual = compiled(x, residual, weight, bias)
    torch.cuda.synchronize(device)
    compile_seconds = time.perf_counter() - started
    if not torch.isfinite(actual).all().item():
        raise RuntimeError("compiled self Wo+residual produced non-finite warmup output")
    bitwise_equal = bool(torch.equal(eager, actual))
    max_abs = float((eager.float() - actual.float()).abs().max().item())
    if not bitwise_equal:
        raise RuntimeError(
            "compiled self Wo+residual warmup is not bitwise equal to eager "
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


def prepare_compiled_self_wo_residual_from_model(
    model: torch.nn.Module,
    dtype: torch.dtype,
) -> tuple[CompiledWoResidual, dict[str, object]]:
    """Prepare compiled Wo+residual from decoder self-attn Wo shapes."""

    from osuT5.osuT5.model.custom_transformers.modeling_varwhisper import (
        VarWhisperDecoderLayer,
    )

    decoder_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, VarWhisperDecoderLayer)
    ]
    if not decoder_layers:
        raise RuntimeError("compiled self Wo+residual found no VarWhisper decoder layers")

    shapes: set[tuple[int, int]] = set()
    first_attn = None
    for _name, layer in decoder_layers:
        self_attn = getattr(layer, "self_attn", None)
        wo = getattr(self_attn, "Wo", None)
        if wo is None:
            raise RuntimeError("decoder self-attn missing Wo for compiled Wo+residual")
        shapes.add((int(wo.weight.shape[0]), int(wo.weight.shape[1])))
        if first_attn is None:
            first_attn = self_attn
    if len(shapes) != 1:
        raise RuntimeError(
            f"compiled self Wo+residual requires uniform Wo shapes, got {shapes}"
        )
    assert first_attn is not None
    device = first_attn.Wo.weight.device
    if first_attn.Wo.weight.dtype != dtype:
        raise TypeError(
            "compiled self Wo+residual weight dtype must match expected generation dtype"
        )
    wo_out, wo_in = (int(first_attn.Wo.weight.shape[0]), int(first_attn.Wo.weight.shape[1]))
    compiled, wo_evidence = prepare_compiled_self_wo_residual(
        device, dtype, wo_out, wo_in
    )
    evidence: dict[str, object] = {
        "enabled": True,
        "decode_only": True,
        "decode_shape": [1, 1],
        "layer_count": len(decoder_layers),
        "wo_residual": wo_evidence,
    }
    return compiled, evidence


def compiled_self_wo_residual_forward(
    *,
    module: torch.nn.Module,
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    layer_name: str,
    dispatch_counts: dict[str, int] | None = None,
) -> torch.Tensor | None:
    """Decoder-layer hook: compiled Wo+residual when decode inputs qualify."""

    del layer_name
    from .compiled_self_wo_residual_activation import active_compiled_self_wo_residual

    compiled = active_compiled_self_wo_residual()
    if compiled is None:
        return None
    self_attn = module.self_attn
    if (
        module.training
        or self_attn.training
        or getattr(module, "dropout", 0) != 0
        or getattr(self_attn, "dropout", 0) != 0
        or residual.device.type != "cuda"
        or residual.shape[:2] != (1, 1)
        or residual.dtype not in (torch.float32, torch.float16)
        or attn_output.dtype != residual.dtype
        or attn_output.device != residual.device
    ):
        return None
    if getattr(self_attn, "out_drop", None) is not None:
        p = getattr(self_attn.out_drop, "p", 0.0)
        if float(p) != 0.0:
            return None
    if attn_output.dim() == 4 and attn_output.shape[0] == 1 and attn_output.shape[2] == 1:
        flat = attn_output.reshape(1, 1, -1)
    elif attn_output.dim() == 3 and attn_output.shape[:2] == (1, 1):
        flat = attn_output
    else:
        return None
    if int(flat.shape[-1]) != int(self_attn.Wo.weight.shape[1]):
        raise RuntimeError(
            "compiled self Wo+residual attn width mismatch: "
            f"{tuple(flat.shape)} vs Wo in={int(self_attn.Wo.weight.shape[1])}"
        )
    hidden = compiled(flat, residual, self_attn.Wo.weight, self_attn.Wo.bias)
    if dispatch_counts is not None:
        dispatch_counts["compiled_self_wo_residual"] = (
            dispatch_counts.get("compiled_self_wo_residual", 0) + 1
        )
    return hidden


__all__ = [
    "CompiledWoResidual",
    "compiled_self_wo_residual_forward",
    "prepare_compiled_self_wo_residual",
    "prepare_compiled_self_wo_residual_from_model",
]
