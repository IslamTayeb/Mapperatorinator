"""Opt-in CUDA conditional-WHILE/torch.multinomial exactness probe.

This module is deliberately not imported by ``optimized.scout``.  It only
answers whether a raw CUDA conditional WHILE may repeat a PyTorch-captured
``torch.multinomial`` child graph while preserving the ordinary PyTorch graph
replay token stream *and* Philox generator progression.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
import re
from typing import Any

import torch
from torch.utils.cpp_extension import load_inline as _load_inline


MIN_CUDA_VERSION = (12, 8)
REQUIRED_CAPABILITY = (7, 5)
RNG_POLICY = "torch_multinomial_philox"
PROBE_BACKEND = "cuda_python_conditional_while_child_graphs"


_CPP_SOURCE = r"""
#include <torch/extension.h>

void conditional_probe_record_and_continue(
    torch::Tensor sampled,
    torch::Tensor samples,
    torch::Tensor iteration,
    int64_t iterations,
    int64_t conditional_handle);
"""


_CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conditional_probe_control_kernel(
    const int64_t* sampled,
    int64_t* samples,
    int32_t* iteration,
    int32_t iterations,
    cudaGraphConditionalHandle conditional_handle) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int32_t current = *iteration;
  if (current < 0 || current >= iterations) {
    cudaGraphSetConditional(conditional_handle, 0);
    return;
  }
  samples[current] = sampled[0];
  const int32_t next = current + 1;
  *iteration = next;
  cudaGraphSetConditional(conditional_handle, next < iterations ? 1 : 0);
}

void conditional_probe_record_and_continue(
    torch::Tensor sampled,
    torch::Tensor samples,
    torch::Tensor iteration,
    int64_t iterations,
    int64_t conditional_handle) {
  TORCH_CHECK(sampled.is_cuda(), "sampled must be a CUDA tensor");
  TORCH_CHECK(samples.is_cuda(), "samples must be a CUDA tensor");
  TORCH_CHECK(iteration.is_cuda(), "iteration must be a CUDA tensor");
  TORCH_CHECK(sampled.scalar_type() == torch::kInt64, "sampled must be int64");
  TORCH_CHECK(samples.scalar_type() == torch::kInt64, "samples must be int64");
  TORCH_CHECK(iteration.scalar_type() == torch::kInt32, "iteration must be int32");
  TORCH_CHECK(sampled.numel() == 1, "sampled must contain one token");
  TORCH_CHECK(samples.dim() == 1, "samples must be a vector");
  TORCH_CHECK(iteration.numel() == 1, "iteration must be a scalar tensor");
  TORCH_CHECK(iterations >= 2, "iterations must be at least two");
  TORCH_CHECK(iterations <= samples.numel(), "samples storage is too small");
  TORCH_CHECK(conditional_handle != 0, "conditional handle must be nonzero");
  TORCH_CHECK(sampled.get_device() == samples.get_device(), "device mismatch");
  TORCH_CHECK(sampled.get_device() == iteration.get_device(), "device mismatch");

  c10::cuda::CUDAGuard device_guard(sampled.device());
  auto stream = at::cuda::getCurrentCUDAStream(sampled.get_device());
  conditional_probe_control_kernel<<<1, 1, 0, stream.stream()>>>(
      sampled.data_ptr<int64_t>(),
      samples.data_ptr<int64_t>(),
      iteration.data_ptr<int32_t>(),
      static_cast<int32_t>(iterations),
      reinterpret_cast<cudaGraphConditionalHandle>(conditional_handle));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


_CONTROL_EXTENSION: Any | None = None


@dataclass(frozen=True)
class ConditionalWhileProbeResult:
    backend: str
    rng_policy: str
    torch_version: str
    torch_cuda_version: str
    cuda_runtime_version: int
    cuda_driver_version: int
    device_name: str
    device_capability: tuple[int, int]
    float32_matmul_precision: str
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    nvidia_tf32_override: str
    seed: int
    iterations: int
    eager_samples: list[int]
    ordinary_graph_samples: list[int]
    conditional_while_samples: list[int]
    eager_generator_state_sha256: str
    ordinary_graph_generator_state_sha256: str
    conditional_generator_state_sha256: str
    ordinary_matches_eager: bool
    ordinary_generator_matches_eager: bool
    conditional_matches_ordinary: bool
    conditional_generator_matches_ordinary: bool
    exact_conditional_while_feasible: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cuda_version_tuple(value: str | None) -> tuple[int, int]:
    if not isinstance(value, str):
        raise RuntimeError("PyTorch does not report a CUDA build version")
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", value)
    if match is None:
        raise RuntimeError(f"unrecognized PyTorch CUDA version {value!r}")
    return int(match.group(1)), int(match.group(2))


def _state_sha256(state: torch.Tensor) -> str:
    if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8:
        raise TypeError("generator state must be a uint8 tensor")
    payload = state.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _enforce_strict_fp32_policy() -> dict[str, Any]:
    override = os.environ.get("NVIDIA_TF32_OVERRIDE")
    if override != "0":
        raise RuntimeError(
            "strict FP32 probe requires NVIDIA_TF32_OVERRIDE=0 before process start"
        )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    precision = torch.get_float32_matmul_precision()
    cuda_allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    cudnn_allow_tf32 = bool(torch.backends.cudnn.allow_tf32)
    if precision != "highest" or cuda_allow_tf32 or cudnn_allow_tf32:
        raise RuntimeError("strict FP32 policy did not remain highest/TF32-disabled")
    return {
        "float32_matmul_precision": precision,
        "cuda_matmul_allow_tf32": cuda_allow_tf32,
        "cudnn_allow_tf32": cudnn_allow_tf32,
        "nvidia_tf32_override": override,
    }


def _cuda_check(result: tuple[Any, ...], operation: str) -> tuple[Any, ...]:
    from cuda.bindings import runtime

    if not isinstance(result, tuple) or not result:
        raise RuntimeError(f"{operation} returned an invalid CUDA result")
    if result[0] != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed with {result[0]}")
    return result[1:]


def _require_environment(device: torch.device) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("conditional-WHILE probe requires an allocated CUDA GPU")
    if device.type != "cuda":
        raise RuntimeError("conditional-WHILE probe requires a CUDA device")
    cuda_version = _cuda_version_tuple(torch.version.cuda)
    if cuda_version < MIN_CUDA_VERSION:
        raise RuntimeError(
            "conditional-WHILE probe requires a PyTorch CUDA 12.8+ build; "
            f"found {torch.version.cuda}"
        )
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    if capability != REQUIRED_CAPABILITY:
        raise RuntimeError(
            "conditional-WHILE probe is an RTX 2080 Ti/SM75 experiment; "
            f"found capability {capability}"
        )
    if not hasattr(torch.cuda.CUDAGraph, "raw_cuda_graph"):
        raise RuntimeError("PyTorch does not expose CUDAGraph.raw_cuda_graph")
    if not hasattr(torch.cuda.CUDAGraph, "register_generator_state"):
        raise RuntimeError("PyTorch does not expose CUDAGraph.register_generator_state")

    try:
        from cuda.bindings import runtime
    except ImportError as exc:
        raise RuntimeError("cuda-python runtime bindings are unavailable") from exc
    required = (
        "cudaGraphAddChildGraphNode",
        "cudaGraphAddNode",
        "cudaGraphConditionalHandleCreate",
        "cudaGraphConditionalNodeType",
        "cudaGraphNodeParams",
        "cudaGraphNodeType",
    )
    missing = [name for name in required if not hasattr(runtime, name)]
    if missing:
        raise RuntimeError(f"cuda-python lacks conditional graph APIs: {missing}")

    (runtime_version,) = _cuda_check(
        runtime.cudaRuntimeGetVersion(), "cudaRuntimeGetVersion"
    )
    (driver_version,) = _cuda_check(
        runtime.cudaDriverGetVersion(), "cudaDriverGetVersion"
    )
    if int(runtime_version) < 12080 or int(driver_version) < 12080:
        raise RuntimeError(
            "conditional-WHILE probe requires CUDA runtime and driver API 12.8+; "
            f"found runtime={runtime_version}, driver={driver_version}"
        )
    return {
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_runtime_version": int(runtime_version),
        "cuda_driver_version": int(driver_version),
        "device_name": str(torch.cuda.get_device_name(device)),
        "device_capability": capability,
    }


def _load_control_extension() -> Any:
    global _CONTROL_EXTENSION
    if _CONTROL_EXTENSION is None:
        _CONTROL_EXTENSION = _load_inline(
            name="mapperatorinator_conditional_while_multinomial_probe",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["conditional_probe_record_and_continue"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            with_cuda=True,
            verbose=False,
        )
    return _CONTROL_EXTENSION


def _capture_multinomial_graph(
    probabilities: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    graph.register_generator_state(generator)
    sampled: torch.Tensor | None = None
    with torch.cuda.graph(graph):
        sampled = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        ).reshape(-1)
    if sampled is None or sampled.shape != (1,) or sampled.dtype != torch.long:
        raise RuntimeError("captured torch.multinomial returned an invalid sample")
    return graph, sampled


def _capture_control_graph(
    extension: Any,
    *,
    sampled: torch.Tensor,
    samples: torch.Tensor,
    iteration: torch.Tensor,
    iterations: int,
    conditional_handle: Any,
) -> torch.cuda.CUDAGraph:
    handle_value = int(conditional_handle)
    if handle_value == 0:
        raise RuntimeError("CUDA returned a null conditional handle")
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        extension.conditional_probe_record_and_continue(
            sampled,
            samples,
            iteration,
            iterations,
            handle_value,
        )
    return graph


def _add_while_node(parent: Any) -> tuple[Any, Any]:
    from cuda.bindings import runtime

    (handle,) = _cuda_check(
        runtime.cudaGraphConditionalHandleCreate(
            parent,
            1,
            int(runtime.cudaGraphConditionalHandleFlags.cudaGraphCondAssignDefault),
        ),
        "cudaGraphConditionalHandleCreate",
    )
    params = runtime.cudaGraphNodeParams()
    params.type = runtime.cudaGraphNodeType.cudaGraphNodeTypeConditional
    params.conditional.handle = handle
    params.conditional.type = (
        runtime.cudaGraphConditionalNodeType.cudaGraphCondTypeWhile
    )
    params.conditional.size = 1
    _cuda_check(runtime.cudaGraphAddNode(parent, None, 0, params), "cudaGraphAddNode")
    body_graphs = list(params.conditional.phGraph_out)
    if len(body_graphs) != 1:
        raise RuntimeError(
            f"CUDA conditional WHILE returned {len(body_graphs)} body graphs"
        )
    return handle, body_graphs[0]


def _add_child_sequence(
    body: Any,
    sampler_graph: torch.cuda.CUDAGraph,
    control_graph: torch.cuda.CUDAGraph,
) -> None:
    from cuda.bindings import runtime

    previous = None
    for name, child in (("sampler", sampler_graph), ("control", control_graph)):
        dependencies = None if previous is None else (previous,)
        (previous,) = _cuda_check(
            runtime.cudaGraphAddChildGraphNode(
                body,
                dependencies,
                0 if dependencies is None else 1,
                runtime.cudaGraph_t(child.raw_cuda_graph()),
            ),
            f"cudaGraphAddChildGraphNode({name})",
        )


def _ordinary_graph_samples(
    graph: torch.cuda.CUDAGraph,
    sampled: torch.Tensor,
    iterations: int,
) -> list[int]:
    result: list[int] = []
    for _ in range(iterations):
        graph.replay()
        result.append(int(sampled.item()))
    return result


def run_conditional_while_multinomial_probe(
    *,
    seed: int = 12345,
    iterations: int = 8,
    probabilities: torch.Tensor | None = None,
) -> ConditionalWhileProbeResult:
    """Run one bounded probe; never install or modify an inference selector."""

    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise TypeError("iterations must be an integer")
    if not 2 <= iterations <= 32:
        raise ValueError("iterations must be between 2 and 32")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    precision_policy = _enforce_strict_fp32_policy()
    if not torch.cuda.is_available():
        raise RuntimeError("conditional-WHILE probe requires an allocated CUDA GPU")
    device = torch.device("cuda", torch.cuda.current_device())
    environment = _require_environment(device)
    if probabilities is None:
        probabilities = torch.tensor(
            [[0.03, 0.07, 0.11, 0.17, 0.23, 0.39]],
            dtype=torch.float32,
            device=device,
        )
    if not isinstance(probabilities, torch.Tensor):
        raise TypeError("probabilities must be a tensor")
    if probabilities.device != device or probabilities.dtype != torch.float32:
        raise ValueError("probabilities must be FP32 on the current CUDA device")
    if probabilities.ndim != 2 or probabilities.shape[0] != 1:
        raise ValueError("probabilities must have shape [1, vocabulary]")
    if probabilities.shape[1] < 2 or not bool(torch.isfinite(probabilities).all()):
        raise ValueError("probabilities must contain at least two finite values")
    if not bool((probabilities > 0).all()):
        raise ValueError("probabilities must be strictly positive")
    normalized = probabilities / probabilities.sum(dim=-1, keepdim=True)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    sampler_graph, sampled = _capture_multinomial_graph(normalized, generator)
    post_capture_state = generator.get_state().clone()

    eager_generator = torch.Generator(device=device)
    eager_generator.set_state(post_capture_state)
    eager_samples = [
        int(
            torch.multinomial(
                normalized,
                num_samples=1,
                generator=eager_generator,
            ).item()
        )
        for _ in range(iterations)
    ]
    eager_state = eager_generator.get_state()

    generator.set_state(post_capture_state)
    ordinary_samples = _ordinary_graph_samples(
        sampler_graph,
        sampled,
        iterations,
    )
    ordinary_state = generator.get_state()

    from cuda.bindings import runtime

    (parent,) = _cuda_check(runtime.cudaGraphCreate(0), "cudaGraphCreate")
    executable = None
    try:
        handle, body = _add_while_node(parent)
        samples = torch.full(
            (iterations,),
            -1,
            dtype=torch.long,
            device=device,
        )
        iteration = torch.zeros((1,), dtype=torch.int32, device=device)
        extension = _load_control_extension()
        control_graph = _capture_control_graph(
            extension,
            sampled=sampled,
            samples=samples,
            iteration=iteration,
            iterations=iterations,
            conditional_handle=handle,
        )
        _add_child_sequence(body, sampler_graph, control_graph)
        (executable,) = _cuda_check(
            runtime.cudaGraphInstantiate(parent, 0),
            "cudaGraphInstantiate",
        )

        generator.set_state(post_capture_state)
        samples.fill_(-1)
        iteration.zero_()
        stream = runtime.cudaStream_t(torch.cuda.current_stream(device).cuda_stream)
        _cuda_check(
            runtime.cudaGraphLaunch(executable, stream),
            "cudaGraphLaunch",
        )
        torch.cuda.synchronize(device)
        completed = int(iteration.item())
        if completed != iterations:
            raise RuntimeError(
                f"conditional WHILE executed {completed} of {iterations} iterations"
            )
        conditional_samples = [int(value) for value in samples.cpu().tolist()]
        conditional_state = generator.get_state()
    finally:
        if executable is not None:
            _cuda_check(
                runtime.cudaGraphExecDestroy(executable),
                "cudaGraphExecDestroy",
            )
        _cuda_check(runtime.cudaGraphDestroy(parent), "cudaGraphDestroy")

    eager_sha = _state_sha256(eager_state)
    ordinary_sha = _state_sha256(ordinary_state)
    conditional_sha = _state_sha256(conditional_state)
    ordinary_matches_eager = ordinary_samples == eager_samples
    ordinary_generator_matches_eager = ordinary_sha == eager_sha
    conditional_matches_ordinary = conditional_samples == ordinary_samples
    conditional_generator_matches_ordinary = conditional_sha == ordinary_sha
    return ConditionalWhileProbeResult(
        backend=PROBE_BACKEND,
        rng_policy=RNG_POLICY,
        torch_version=str(torch.__version__),
        seed=seed,
        iterations=iterations,
        eager_samples=eager_samples,
        ordinary_graph_samples=ordinary_samples,
        conditional_while_samples=conditional_samples,
        eager_generator_state_sha256=eager_sha,
        ordinary_graph_generator_state_sha256=ordinary_sha,
        conditional_generator_state_sha256=conditional_sha,
        ordinary_matches_eager=ordinary_matches_eager,
        ordinary_generator_matches_eager=ordinary_generator_matches_eager,
        conditional_matches_ordinary=conditional_matches_ordinary,
        conditional_generator_matches_ordinary=conditional_generator_matches_ordinary,
        exact_conditional_while_feasible=(
            ordinary_matches_eager
            and ordinary_generator_matches_eager
            and conditional_matches_ordinary
            and conditional_generator_matches_ordinary
        ),
        **precision_policy,
        **environment,
    )


__all__ = [
    "ConditionalWhileProbeResult",
    "PROBE_BACKEND",
    "RNG_POLICY",
    "_enforce_strict_fp32_policy",
    "run_conditional_while_multinomial_probe",
]
