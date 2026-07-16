"""Opt-in torch.compile q1 cross BMM prepared before outer CUDA capture.

Exact FP16/FP32 path only — no INT8 / weight-only ownership.
"""

from __future__ import annotations

import time
from typing import Callable

import torch

from .dispatch import _q1_bmm_cross_attention


CompiledCross = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

_COMPILED_CACHE: dict[
    tuple[int, torch.dtype],
    tuple[CompiledCross, dict[str, float | bool | str]],
] = {}


def prepare_compiled_q1_cross_bmm(
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[CompiledCross, dict[str, float | bool | str]]:
    """Compile and warm the exact q1 cross-BMM signature outside graph capture."""

    if not isinstance(device, torch.device) or device.type != "cuda":
        raise RuntimeError("compiled q1 cross BMM requires one CUDA device")
    if dtype not in (torch.float32, torch.float16):
        raise TypeError("compiled q1 cross BMM supports only float32 or float16")
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 5):
        raise RuntimeError(
            f"compiled q1 cross BMM requires SM75, got sm_{capability[0]}{capability[1]}"
        )

    cache_key = (
        device.index if device.index is not None else torch.cuda.current_device(),
        dtype,
    )
    cached = _COMPILED_CACHE.get(cache_key)
    if cached is not None:
        compiled, evidence = cached
        return compiled, {
            **evidence,
            "cache_hit": True,
            "compile_and_first_call_seconds": 0.0,
        }

    def _cross_region(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return _q1_bmm_cross_attention(
            query,
            key,
            value,
            expected_dtype=dtype,
        )

    compiled = torch.compile(
        _cross_region,
        fullgraph=True,
        dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )
    query = torch.zeros((1, 1, 12, 64), device=device, dtype=dtype).transpose(1, 2)
    key = torch.zeros((1, 12, 1024, 64), device=device, dtype=dtype)
    value = torch.zeros_like(key)
    eager = _cross_region(query, key, value)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    actual = compiled(query, key, value)
    torch.cuda.synchronize(device)
    compile_seconds = time.perf_counter() - started
    max_abs = float((eager.float() - actual.float()).abs().max().item())
    if not torch.isfinite(actual).all().item():
        raise RuntimeError("compiled q1 cross BMM produced non-finite warmup output")
    evidence: dict[str, float | bool | str] = {
        "mode": "max-autotune-no-cudagraphs",
        "fullgraph": True,
        "dynamic": False,
        "dtype": str(dtype).removeprefix("torch."),
        "outer_cuda_graph_owned": True,
        "compile_and_first_call_seconds": compile_seconds,
        "warmup_max_abs": max_abs,
        "warmup_bitwise_equal": bool(torch.equal(eager, actual)),
        "cache_hit": False,
    }
    _COMPILED_CACHE[cache_key] = (compiled, evidence)
    return compiled, dict(evidence)


__all__ = ["CompiledCross", "prepare_compiled_q1_cross_bmm"]
