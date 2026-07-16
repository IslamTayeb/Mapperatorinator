"""Opt-in torch.compile decode-only (1,1) LM-head F.linear before outer CUDA capture."""

from __future__ import annotations

import time
from typing import Callable

import torch

CompiledLinear = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor
]
_COMPILED_CACHE: dict[
    tuple[int, torch.dtype, int, int],
    tuple[CompiledLinear, dict[str, float | bool | str | int]],
] = {}


def prepare_compiled_one_token_linear(
    device: torch.device, dtype: torch.dtype, out_features: int, in_features: int
) -> tuple[CompiledLinear, dict[str, float | bool | str | int]]:
    """Compile and bitwise-warm an exact F.linear region outside graph capture."""
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise RuntimeError("compiled one-token linear requires one CUDA device")
    if dtype not in (torch.float32, torch.float16):
        raise TypeError("compiled one-token linear supports only float32 or float16")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (out_features, in_features)
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
        x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        return torch.nn.functional.linear(x, weight, bias)

    compiled = torch.compile(
        _linear_region, fullgraph=True, dynamic=False, mode="default"
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
        raise RuntimeError(
            "compiled one-token linear produced non-finite warmup output"
        )
    bitwise_equal = bool(torch.equal(eager, actual))
    max_abs = float((eager.float() - actual.float()).abs().max().item())
    if not bitwise_equal:
        raise RuntimeError(
            f"compiled one-token linear warmup is not bitwise equal to eager (max_abs={max_abs})"
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


def prepare_compiled_proj_out(
    model: torch.nn.Module, dtype: torch.dtype
) -> tuple[CompiledLinear, dict[str, object]]:
    """Prepare the tip-exact LM-head projection from model.proj_out weight shape."""
    proj_out = getattr(model, "proj_out", None)
    weight = getattr(proj_out, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise RuntimeError(
            "compiled proj_out requires model.proj_out with rank-2 weight"
        )
    if weight.dtype != dtype:
        raise TypeError(
            "compiled proj_out weight dtype must match expected generation dtype"
        )
    bias = getattr(proj_out, "bias", None)
    if bias is not None and (
        not isinstance(bias, torch.Tensor)
        or bias.dtype != dtype
        or bias.device != weight.device
    ):
        raise TypeError(
            "compiled proj_out bias must match LM-head weight device and dtype"
        )
    compiled, linear_evidence = prepare_compiled_one_token_linear(
        weight.device, dtype, int(weight.shape[0]), int(weight.shape[1])
    )
    return compiled, {
        "enabled": True,
        "decode_only": True,
        "decode_shape": [1, 1],
        "proj_out": linear_evidence,
    }


__all__ = [
    "CompiledLinear",
    "prepare_compiled_one_token_linear",
    "prepare_compiled_proj_out",
]
