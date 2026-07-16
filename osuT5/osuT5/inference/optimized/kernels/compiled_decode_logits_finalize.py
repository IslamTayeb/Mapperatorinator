"""Tip-exact decode logits workspace + compiled float32 softmax (static 1×V).

Sampling / elementwise-memory lever: reuse one float32 vocab buffer and
torch.compile softmax at fixed decode shape. Multinomial/argmax stay eager
for RNG exactness. Not allocator-env; not native Wo/proj_out/Wqkv.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F

CompiledSoftmax = Callable[[torch.Tensor], torch.Tensor]

_COMPILED_SOFTMAX_CACHE: dict[
    tuple[int, int],
    tuple[CompiledSoftmax, dict[str, float | bool | str | int]],
] = {}


@dataclass
class DecodeLogitsFinalize:
    workspace: torch.Tensor
    compiled_softmax: CompiledSoftmax
    hits: dict[str, int] = field(
        default_factory=lambda: {"workspace_copy": 0, "compiled_softmax": 0}
    )

    def materialize_logits(self, last_row: torch.Tensor) -> torch.Tensor:
        if not isinstance(last_row, torch.Tensor) or last_row.ndim != 2:
            raise RuntimeError(
                "decode logits finalize expects rank-2 last-token logits"
            )
        if tuple(last_row.shape) != tuple(self.workspace.shape):
            raise RuntimeError(
                "decode logits finalize shape mismatch: "
                f"{tuple(last_row.shape)} vs {tuple(self.workspace.shape)}"
            )
        if last_row.device != self.workspace.device:
            raise RuntimeError("decode logits finalize device mismatch")
        if last_row.dtype == torch.float32:
            self.workspace.copy_(last_row)
        else:
            self.workspace.copy_(last_row.to(dtype=torch.float32))
        self.hits["workspace_copy"] += 1
        return self.workspace

    def softmax(self, scores: torch.Tensor) -> torch.Tensor:
        if not isinstance(scores, torch.Tensor) or scores.ndim != 2:
            raise RuntimeError("compiled softmax expects rank-2 scores")
        if tuple(scores.shape) != tuple(self.workspace.shape):
            raise RuntimeError(
                "compiled softmax shape mismatch: "
                f"{tuple(scores.shape)} vs {tuple(self.workspace.shape)}"
            )
        if scores.dtype != torch.float32 or scores.device != self.workspace.device:
            raise RuntimeError("compiled softmax requires float32 on workspace device")
        probabilities = self.compiled_softmax(scores)
        self.hits["compiled_softmax"] += 1
        return probabilities


def prepare_compiled_softmax(
    device: torch.device, vocab_size: int
) -> tuple[CompiledSoftmax, dict[str, float | bool | str | int]]:
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise RuntimeError("compiled decode softmax requires one CUDA device")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 1:
        raise ValueError("compiled decode softmax requires vocab_size > 1")
    capability = torch.cuda.get_device_capability(device)
    if capability != (7, 5):
        raise RuntimeError(
            f"compiled decode softmax requires SM75, got sm_{capability[0]}{capability[1]}"
        )
    cache_key = (
        device.index if device.index is not None else torch.cuda.current_device(),
        int(vocab_size),
    )
    cached = _COMPILED_SOFTMAX_CACHE.get(cache_key)
    if cached is not None:
        compiled, evidence = cached
        return compiled, {
            **evidence,
            "cache_hit": True,
            "compile_and_first_call_seconds": 0.0,
        }

    def _softmax_region(scores: torch.Tensor) -> torch.Tensor:
        return F.softmax(scores, dim=-1)

    compiled = torch.compile(
        _softmax_region, fullgraph=True, dynamic=False, mode="default"
    )
    scores = torch.zeros((1, vocab_size), device=device, dtype=torch.float32)
    eager = _softmax_region(scores)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    actual = compiled(scores)
    torch.cuda.synchronize(device)
    compile_seconds = time.perf_counter() - started
    if not torch.isfinite(actual).all().item():
        raise RuntimeError("compiled decode softmax produced non-finite warmup output")
    bitwise_equal = bool(torch.equal(eager, actual))
    max_abs = float((eager - actual).abs().max().item())
    if not bitwise_equal:
        raise RuntimeError(
            f"compiled decode softmax warmup is not bitwise equal (max_abs={max_abs})"
        )
    evidence: dict[str, float | bool | str | int] = {
        "mode": "default",
        "fullgraph": True,
        "dynamic": False,
        "dtype": "float32",
        "vocab_size": int(vocab_size),
        "compile_and_first_call_seconds": compile_seconds,
        "warmup_max_abs": max_abs,
        "warmup_bitwise_equal": bitwise_equal,
        "cache_hit": False,
    }
    _COMPILED_SOFTMAX_CACHE[cache_key] = (compiled, evidence)
    return compiled, dict(evidence)


def prepare_decode_logits_finalize(
    model: torch.nn.Module,
) -> tuple[DecodeLogitsFinalize, dict[str, object]]:
    device = getattr(model, "device", None)
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise RuntimeError("decode logits finalize requires model.device CUDA")
    config = getattr(model, "config", None)
    vocab_size = getattr(config, "vocab_size", None)
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 1:
        raise RuntimeError("decode logits finalize requires model.config.vocab_size")
    compiled, softmax_evidence = prepare_compiled_softmax(device, int(vocab_size))
    workspace = torch.empty((1, int(vocab_size)), device=device, dtype=torch.float32)
    finalize = DecodeLogitsFinalize(workspace=workspace, compiled_softmax=compiled)
    return finalize, {
        "enabled": True,
        "vocab_size": int(vocab_size),
        "workspace_shape": [1, int(vocab_size)],
        "compiled_softmax": softmax_evidence,
    }


__all__ = [
    "DecodeLogitsFinalize",
    "prepare_compiled_softmax",
    "prepare_decode_logits_finalize",
]
