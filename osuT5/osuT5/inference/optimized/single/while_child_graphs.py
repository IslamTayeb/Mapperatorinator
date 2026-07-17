"""Minimal child-graph helpers for §32 WHILE step wrap (cold; not production).

Extracted from the older K8 scout so tip `55949274` can wrap keep_graph
model+tail children without importing the full K8 runtime or Inductor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

SUPPORTED_BLOCK_SIZES = (1, 4, 8)


def _cuda_check(result: tuple[Any, ...], operation: str) -> tuple[Any, ...]:
    from cuda.bindings import runtime

    if not isinstance(result, tuple) or not result:
        raise RuntimeError(f"{operation} returned an invalid CUDA result")
    if result[0] != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed with {result[0]}")
    return result[1:]


def _cuda_status(result: tuple[Any, ...], operation: str) -> None:
    payload = _cuda_check(result, operation)
    if payload:
        raise RuntimeError(f"{operation} returned unexpected payload {payload!r}")


def _validate_block_size(block_size: int) -> int:
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise TypeError("block size must be an integer")
    if block_size not in SUPPORTED_BLOCK_SIZES:
        raise ValueError(
            f"block size must be one of {SUPPORTED_BLOCK_SIZES}, got {block_size}"
        )
    return block_size


def update_static_model_inputs(
    static_inputs: dict[str, Any],
    token: torch.Tensor,
) -> None:
    decoder_ids = static_inputs.get("decoder_input_ids")
    cache_position = static_inputs.get("cache_position")
    position_ids = static_inputs.get("decoder_position_ids")
    if (
        not isinstance(decoder_ids, torch.Tensor)
        or decoder_ids.shape != (1, 1)
        or decoder_ids.dtype != torch.long
    ):
        raise ValueError("static decoder_input_ids must be [1, 1] long")
    if (
        not isinstance(cache_position, torch.Tensor)
        or cache_position.shape != (1,)
        or cache_position.dtype != torch.long
    ):
        raise ValueError("static cache_position must be [1] long")
    if (
        not isinstance(position_ids, torch.Tensor)
        or position_ids.shape != (1, 1)
        or position_ids.dtype != torch.long
    ):
        raise ValueError("static decoder_position_ids must be [1, 1] long")
    decoder_ids.copy_(token.view(1, 1))
    cache_position.add_(1)
    position_ids.add_(1)
    mask = static_inputs.get("decoder_attention_mask")
    if isinstance(mask, torch.Tensor) and mask.ndim == 4:
        mask.scatter_(
            -1,
            cache_position.view(1, 1, 1, 1),
            torch.zeros((1, 1, 1, 1), dtype=mask.dtype, device=mask.device),
        )


@dataclass(slots=True)
class ChildGraphSequence:
    """Fixed-length unrolled parent of model+tail child graphs (k1/k4/k8)."""

    parent_graph: Any
    executable: Any
    model_graph: torch.cuda.CUDAGraph
    tail_graph: torch.cuda.CUDAGraph
    device: torch.device
    closed: bool = False
    executable_closed: bool = False
    parent_closed: bool = False

    @classmethod
    def build(
        cls,
        model_graph: torch.cuda.CUDAGraph,
        tail_graph: torch.cuda.CUDAGraph,
        *,
        block_size: int,
        device: torch.device | None = None,
    ) -> "ChildGraphSequence":
        block_size = _validate_block_size(block_size)
        if not hasattr(model_graph, "raw_cuda_graph") or not hasattr(
            tail_graph, "raw_cuda_graph"
        ):
            raise RuntimeError(
                "child graphs require CUDAGraph(keep_graph=True).raw_cuda_graph()"
            )
        from cuda.bindings import runtime

        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        device = torch.device(device)
        with torch.cuda.device(device):
            (parent,) = _cuda_check(runtime.cudaGraphCreate(0), "cudaGraphCreate")
            previous = None
            try:
                for _ in range(block_size):
                    for child in (model_graph, tail_graph):
                        dependencies = None if previous is None else (previous,)
                        (node,) = _cuda_check(
                            runtime.cudaGraphAddChildGraphNode(
                                parent,
                                dependencies,
                                0 if previous is None else 1,
                                runtime.cudaGraph_t(child.raw_cuda_graph()),
                            ),
                            "cudaGraphAddChildGraphNode",
                        )
                        previous = node
                (executable,) = _cuda_check(
                    runtime.cudaGraphInstantiate(parent, 0),
                    "cudaGraphInstantiate",
                )
            except Exception:
                _cuda_status(runtime.cudaGraphDestroy(parent), "cudaGraphDestroy")
                raise
        return cls(parent, executable, model_graph, tail_graph, device)

    def replay(self) -> None:
        if self.closed:
            raise RuntimeError("child-graph parent is closed")
        from cuda.bindings import runtime

        with torch.cuda.device(self.device):
            stream = runtime.cudaStream_t(torch.cuda.current_stream().cuda_stream)
            _cuda_status(
                runtime.cudaGraphLaunch(self.executable, stream),
                "cudaGraphLaunch",
            )

    def close(self) -> None:
        if self.closed:
            return
        from cuda.bindings import runtime

        with torch.cuda.device(self.device):
            if not self.executable_closed:
                _cuda_status(
                    runtime.cudaGraphExecDestroy(self.executable),
                    "cudaGraphExecDestroy",
                )
                self.executable_closed = True
            if not self.parent_closed:
                _cuda_status(
                    runtime.cudaGraphDestroy(self.parent_graph),
                    "cudaGraphDestroy",
                )
                self.parent_closed = True
        self.closed = self.executable_closed and self.parent_closed


__all__ = [
    "ChildGraphSequence",
    "SUPPORTED_BLOCK_SIZES",
    "update_static_model_inputs",
]
