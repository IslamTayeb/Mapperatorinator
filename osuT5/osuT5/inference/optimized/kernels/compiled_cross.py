"""Opt-in torch.compile q1 cross BMM prepared before outer CUDA capture."""

from __future__ import annotations

import time
from typing import Callable

import torch

from .dispatch import _q1_bmm_cross_attention


CompiledCross = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _cross_region(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    return _q1_bmm_cross_attention(
        query,
        key,
        value,
        expected_dtype=torch.float32,
    )


def prepare_compiled_q1_cross_bmm(
    device: torch.device,
) -> tuple[CompiledCross, dict[str, float | bool | str]]:
    """Compile and warm the exact selected tensor signature outside graph capture."""

    if not isinstance(device, torch.device) or device.type != "cuda":
        raise RuntimeError("compiled q1 cross BMM requires one CUDA device")
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 5):
        raise RuntimeError(
            f"compiled q1 cross BMM requires SM75, got sm_{capability[0]}{capability[1]}"
        )
    compiled = torch.compile(
        _cross_region,
        fullgraph=True,
        dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )
    query = torch.zeros((1, 1, 12, 64), device=device, dtype=torch.float32).transpose(
        1, 2
    )
    key = torch.zeros((1, 12, 1024, 64), device=device, dtype=torch.float32)
    value = torch.zeros_like(key)
    eager = _cross_region(query, key, value)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    actual = compiled(query, key, value)
    torch.cuda.synchronize(device)
    compile_seconds = time.perf_counter() - started
    max_abs = float((eager - actual).abs().max().item())
    if not torch.isfinite(actual).all().item():
        raise RuntimeError("compiled q1 cross BMM produced non-finite warmup output")
    return compiled, {
        "mode": "max-autotune-no-cudagraphs",
        "fullgraph": True,
        "dynamic": False,
        "outer_cuda_graph_owned": True,
        "compile_and_first_call_seconds": compile_seconds,
        "warmup_max_abs": max_abs,
        "warmup_bitwise_equal": bool(torch.equal(eager, actual)),
    }


__all__ = ["CompiledCross", "prepare_compiled_q1_cross_bmm"]
