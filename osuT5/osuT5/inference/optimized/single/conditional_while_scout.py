"""Opt-in CUDA 12.8 conditional-WHILE graph scout.

This module is deliberately absent from production imports.  It builds a
top-level conditional graph around two already-captured PyTorch CUDA graphs:
one model step and one device-resident tail step.  The conditional parent owns
the loop; the source PyTorch graphs stay alive in the Python wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import torch
from torch.utils.cpp_extension import load_inline


MINIMUM_WHILE_CUDA_VERSION = 12_030
EXTENSION_NAME = "mapperatorinator_conditional_while_cuda128_v1"


_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <cstdint>

std::uint64_t conditional_build_probe(torch::Tensor counter, torch::Tensor limit);
std::uint64_t conditional_build_while(
    std::uint64_t model_graph,
    std::uint64_t tail_graph,
    torch::Tensor unfinished,
    torch::Tensor physical_step,
    torch::Tensor stop_step);
void conditional_launch(std::uint64_t handle);
void conditional_destroy(std::uint64_t handle);
std::int64_t conditional_runtime_version();
std::int64_t conditional_driver_version();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("build_probe", &conditional_build_probe);
  module.def("build_while", &conditional_build_while);
  module.def("launch", &conditional_launch);
  module.def("destroy", &conditional_destroy);
  module.def("runtime_version", &conditional_runtime_version);
  module.def("driver_version", &conditional_driver_version);
}
"""


_CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#if CUDART_VERSION < 12030
#error "CUDA conditional WHILE graph nodes require CUDA toolkit 12.3 or newer"
#endif

namespace {

struct ConditionalExecutable {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  int device = -1;
};

std::mutex registry_mutex;
std::unordered_set<ConditionalExecutable*> registry;

void check_cuda(cudaError_t status, const char* operation) {
  TORCH_CHECK(
      status == cudaSuccess,
      operation,
      " failed: ",
      cudaGetErrorName(status),
      " (",
      cudaGetErrorString(status),
      ")");
}

void require_cuda128_runtime() {
  int runtime_version = 0;
  int driver_version = 0;
  check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
  check_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
  TORCH_CHECK(
      runtime_version >= 12030,
      "CUDA conditional WHILE requires runtime >= 12.3; got ",
      runtime_version);
  TORCH_CHECK(
      driver_version >= 12030,
      "CUDA conditional WHILE requires driver API >= 12.3; got ",
      driver_version);
}

void validate_scalar_tensor(
    const torch::Tensor& value,
    const char* name,
    c10::ScalarType dtype) {
  TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(value.numel() == 1, name, " must contain one element");
  TORCH_CHECK(value.scalar_type() == dtype, name, " has the wrong dtype");
}

void validate_same_device(
    const torch::Tensor& first,
    const torch::Tensor& second,
    const char* name) {
  TORCH_CHECK(
      first.device() == second.device(),
      name,
      " must use device ",
      first.device(),
      "; got ",
      second.device());
}

void validate_body_graph(cudaGraph_t graph, const char* label, int depth = 0) {
  TORCH_CHECK(graph != nullptr, label, " graph handle is null");
  TORCH_CHECK(depth < 32, label, " graph nesting exceeds 31 child levels");
  size_t count = 0;
  check_cuda(cudaGraphGetNodes(graph, nullptr, &count), "cudaGraphGetNodes(count)");
  std::vector<cudaGraphNode_t> nodes(count);
  if (count != 0) {
    check_cuda(cudaGraphGetNodes(graph, nodes.data(), &count), "cudaGraphGetNodes");
  }
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type = cudaGraphNodeTypeCount;
    check_cuda(cudaGraphNodeGetType(node, &type), "cudaGraphNodeGetType");
    switch (type) {
      case cudaGraphNodeTypeKernel:
      case cudaGraphNodeTypeMemcpy:
      case cudaGraphNodeTypeMemset:
      case cudaGraphNodeTypeEmpty:
        break;
      case cudaGraphNodeTypeGraph: {
        cudaGraph_t child = nullptr;
        check_cuda(
            cudaGraphChildGraphNodeGetGraph(node, &child),
            "cudaGraphChildGraphNodeGetGraph");
        validate_body_graph(child, label, depth + 1);
        break;
      }
      default:
        TORCH_CHECK(
            false,
            label,
            " graph contains CUDA node type ",
            static_cast<int>(type),
            ", which is forbidden recursively inside a conditional body");
    }
  }
}

ConditionalExecutable* lookup(std::uint64_t raw) {
  auto* pointer = reinterpret_cast<ConditionalExecutable*>(raw);
  std::lock_guard<std::mutex> guard(registry_mutex);
  TORCH_CHECK(
      pointer != nullptr && registry.count(pointer) == 1,
      "unknown or already-destroyed conditional graph handle");
  return pointer;
}

void register_executable(ConditionalExecutable* pointer) {
  std::lock_guard<std::mutex> guard(registry_mutex);
  const bool inserted = registry.insert(pointer).second;
  TORCH_CHECK(inserted, "conditional graph registry collision");
}

void unregister_executable(ConditionalExecutable* pointer) {
  std::lock_guard<std::mutex> guard(registry_mutex);
  const size_t erased = registry.erase(pointer);
  TORCH_CHECK(erased == 1, "conditional graph registry lost its handle");
}

void destroy_partial(ConditionalExecutable* pointer) noexcept {
  if (pointer == nullptr) {
    return;
  }
  if (pointer->executable != nullptr) {
    cudaGraphExecDestroy(pointer->executable);
    pointer->executable = nullptr;
  }
  if (pointer->graph != nullptr) {
    cudaGraphDestroy(pointer->graph);
    pointer->graph = nullptr;
  }
}

__global__ void counter_probe_kernel(
    std::int64_t* counter,
    const std::int64_t* limit,
    cudaGraphConditionalHandle condition) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const std::int64_t next = *counter + 1;
    *counter = next;
    cudaGraphSetConditional(condition, next < *limit ? 1U : 0U);
  }
}

__global__ void update_while_condition_kernel(
    const bool* unfinished,
    const std::int64_t* physical_step,
    const std::int64_t* stop_step,
    cudaGraphConditionalHandle condition) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const bool should_continue = *unfinished && *physical_step < *stop_step;
    cudaGraphSetConditional(condition, should_continue ? 1U : 0U);
  }
}

cudaGraph_t add_while_node(
    cudaGraph_t parent,
    cudaGraphConditionalHandle* condition_out) {
  check_cuda(
      cudaGraphConditionalHandleCreate(
          condition_out,
          parent,
          1U,
          cudaGraphCondAssignDefault),
      "cudaGraphConditionalHandleCreate");
  cudaGraphNodeParams parameters{};
  parameters.type = cudaGraphNodeTypeConditional;
  parameters.conditional.handle = *condition_out;
  parameters.conditional.type = cudaGraphCondTypeWhile;
  parameters.conditional.size = 1U;
  cudaGraphNode_t conditional_node = nullptr;
  check_cuda(
      cudaGraphAddNode(&conditional_node, parent, nullptr, 0, &parameters),
      "cudaGraphAddNode(WHILE)");
  TORCH_CHECK(
      parameters.conditional.phGraph_out != nullptr,
      "CUDA did not return the WHILE body graph array");
  cudaGraph_t body = parameters.conditional.phGraph_out[0];
  TORCH_CHECK(body != nullptr, "CUDA did not return a WHILE body graph");
  return body;
}

std::uint64_t finalize(ConditionalExecutable* pointer) {
  check_cuda(
      cudaGraphInstantiate(&pointer->executable, pointer->graph, 0),
      "cudaGraphInstantiate(conditional)");
  register_executable(pointer);
  return reinterpret_cast<std::uint64_t>(pointer);
}

}  // namespace

std::uint64_t conditional_build_probe(
    torch::Tensor counter,
    torch::Tensor limit) {
  require_cuda128_runtime();
  validate_scalar_tensor(counter, "counter", c10::ScalarType::Long);
  validate_scalar_tensor(limit, "limit", c10::ScalarType::Long);
  validate_same_device(counter, limit, "limit");
  c10::cuda::CUDAGuard device_guard(counter.device());
  auto* pointer = new ConditionalExecutable();
  pointer->device = counter.get_device();
  try {
    check_cuda(cudaGraphCreate(&pointer->graph, 0), "cudaGraphCreate");
    cudaGraphConditionalHandle condition{};
    cudaGraph_t body = add_while_node(pointer->graph, &condition);
    std::int64_t* counter_pointer = counter.data_ptr<std::int64_t>();
    std::int64_t* limit_pointer = limit.data_ptr<std::int64_t>();
    void* arguments[] = {&counter_pointer, &limit_pointer, &condition};
    cudaKernelNodeParams kernel{};
    kernel.func = reinterpret_cast<void*>(counter_probe_kernel);
    kernel.gridDim = dim3(1, 1, 1);
    kernel.blockDim = dim3(1, 1, 1);
    kernel.sharedMemBytes = 0;
    kernel.kernelParams = arguments;
    cudaGraphNode_t node = nullptr;
    check_cuda(
        cudaGraphAddKernelNode(&node, body, nullptr, 0, &kernel),
        "cudaGraphAddKernelNode(counter probe)");
    return finalize(pointer);
  } catch (...) {
    destroy_partial(pointer);
    delete pointer;
    throw;
  }
}

std::uint64_t conditional_build_while(
    std::uint64_t model_graph_raw,
    std::uint64_t tail_graph_raw,
    torch::Tensor unfinished,
    torch::Tensor physical_step,
    torch::Tensor stop_step) {
  require_cuda128_runtime();
  TORCH_CHECK(model_graph_raw != 0, "model graph handle is null");
  TORCH_CHECK(tail_graph_raw != 0, "tail graph handle is null");
  validate_scalar_tensor(unfinished, "unfinished", c10::ScalarType::Bool);
  validate_scalar_tensor(physical_step, "physical_step", c10::ScalarType::Long);
  validate_scalar_tensor(stop_step, "stop_step", c10::ScalarType::Long);
  validate_same_device(unfinished, physical_step, "physical_step");
  validate_same_device(unfinished, stop_step, "stop_step");
  c10::cuda::CUDAGuard device_guard(unfinished.device());
  auto model_graph = reinterpret_cast<cudaGraph_t>(model_graph_raw);
  auto tail_graph = reinterpret_cast<cudaGraph_t>(tail_graph_raw);
  validate_body_graph(model_graph, "model");
  validate_body_graph(tail_graph, "tail");

  auto* pointer = new ConditionalExecutable();
  pointer->device = unfinished.get_device();
  try {
    check_cuda(cudaGraphCreate(&pointer->graph, 0), "cudaGraphCreate");
    cudaGraphConditionalHandle condition{};
    cudaGraph_t body = add_while_node(pointer->graph, &condition);
    cudaGraphNode_t model_node = nullptr;
    check_cuda(
        cudaGraphAddChildGraphNode(
            &model_node, body, nullptr, 0, model_graph),
        "cudaGraphAddChildGraphNode(model)");
    cudaGraphNode_t tail_node = nullptr;
    check_cuda(
        cudaGraphAddChildGraphNode(
            &tail_node, body, &model_node, 1, tail_graph),
        "cudaGraphAddChildGraphNode(tail)");
    bool* unfinished_pointer = unfinished.data_ptr<bool>();
    std::int64_t* step_pointer = physical_step.data_ptr<std::int64_t>();
    std::int64_t* stop_pointer = stop_step.data_ptr<std::int64_t>();
    void* arguments[] = {
        &unfinished_pointer, &step_pointer, &stop_pointer, &condition};
    cudaKernelNodeParams kernel{};
    kernel.func = reinterpret_cast<void*>(update_while_condition_kernel);
    kernel.gridDim = dim3(1, 1, 1);
    kernel.blockDim = dim3(1, 1, 1);
    kernel.sharedMemBytes = 0;
    kernel.kernelParams = arguments;
    cudaGraphNode_t updater_node = nullptr;
    check_cuda(
        cudaGraphAddKernelNode(
            &updater_node, body, &tail_node, 1, &kernel),
        "cudaGraphAddKernelNode(condition updater)");
    return finalize(pointer);
  } catch (...) {
    destroy_partial(pointer);
    delete pointer;
    throw;
  }
}

void conditional_launch(std::uint64_t handle) {
  ConditionalExecutable* pointer = lookup(handle);
  c10::cuda::CUDAGuard device_guard(c10::Device(c10::kCUDA, pointer->device));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(pointer->device).stream();
  check_cuda(
      cudaGraphLaunch(pointer->executable, stream),
      "cudaGraphLaunch(conditional)");
}

void conditional_destroy(std::uint64_t handle) {
  ConditionalExecutable* pointer = lookup(handle);
  c10::cuda::CUDAGuard device_guard(c10::Device(c10::kCUDA, pointer->device));
  unregister_executable(pointer);
  cudaError_t executable_status = cudaSuccess;
  cudaError_t graph_status = cudaSuccess;
  if (pointer->executable != nullptr) {
    executable_status = cudaGraphExecDestroy(pointer->executable);
    pointer->executable = nullptr;
  }
  if (pointer->graph != nullptr) {
    graph_status = cudaGraphDestroy(pointer->graph);
    pointer->graph = nullptr;
  }
  delete pointer;
  check_cuda(executable_status, "cudaGraphExecDestroy(conditional)");
  check_cuda(graph_status, "cudaGraphDestroy(conditional)");
}

std::int64_t conditional_runtime_version() {
  int version = 0;
  check_cuda(cudaRuntimeGetVersion(&version), "cudaRuntimeGetVersion");
  return version;
}

std::int64_t conditional_driver_version() {
  int version = 0;
  check_cuda(cudaDriverGetVersion(&version), "cudaDriverGetVersion");
  return version;
}
"""


_EXTENSION: Any | None = None


def _load_extension() -> Any:
    global _EXTENSION
    if _EXTENSION is None:
        _EXTENSION = load_inline(
            name=EXTENSION_NAME,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=None,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--threads", "2"],
            with_cuda=True,
            verbose=os.environ.get("MAPPERATORINATOR_VERBOSE_NATIVE_BUILD") == "1",
        )
    return _EXTENSION


def _require_cuda_scalar(
    name: str,
    value: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> torch.device:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if value.numel() != 1 or value.dtype != dtype or not value.is_contiguous():
        raise ValueError(
            f"{name} must be one contiguous {dtype} element; got "
            f"shape={tuple(value.shape)}, dtype={value.dtype}"
        )
    if device is not None and value.device != device:
        raise ValueError(f"{name} must use {device}; got {value.device}")
    return value.device


def _raw_graph(name: str, graph: Any) -> int:
    raw = getattr(graph, "raw_cuda_graph", None)
    if not callable(raw):
        raise RuntimeError(
            f"{name} must be torch.cuda.CUDAGraph(keep_graph=True) with "
            "raw_cuda_graph()"
        )
    handle = int(raw())
    if handle == 0:
        raise RuntimeError(f"{name}.raw_cuda_graph() returned a null handle")
    return handle


@dataclass(slots=True)
class ConditionalGraph:
    """Owned native parent graph plus references to its source child graphs."""

    handle: int
    extension: Any
    device: torch.device
    source_graphs: tuple[Any, ...] = ()
    closed: bool = False

    @classmethod
    def counter_probe(
        cls,
        counter: torch.Tensor,
        limit: torch.Tensor,
    ) -> "ConditionalGraph":
        device = _require_cuda_scalar("counter", counter, dtype=torch.long)
        _require_cuda_scalar("limit", limit, dtype=torch.long, device=device)
        extension = _load_extension()
        with torch.cuda.device(device):
            handle = int(extension.build_probe(counter, limit))
        if handle == 0:
            raise RuntimeError("conditional counter probe returned a null handle")
        return cls(handle, extension, device)

    @classmethod
    def while_children(
        cls,
        model_graph: Any,
        tail_graph: Any,
        *,
        unfinished: torch.Tensor,
        physical_step: torch.Tensor,
        stop_step: torch.Tensor,
    ) -> "ConditionalGraph":
        device = _require_cuda_scalar("unfinished", unfinished, dtype=torch.bool)
        _require_cuda_scalar(
            "physical_step", physical_step, dtype=torch.long, device=device
        )
        _require_cuda_scalar("stop_step", stop_step, dtype=torch.long, device=device)
        model_handle = _raw_graph("model_graph", model_graph)
        tail_handle = _raw_graph("tail_graph", tail_graph)
        extension = _load_extension()
        with torch.cuda.device(device):
            handle = int(extension.build_while(
                model_handle,
                tail_handle,
                unfinished,
                physical_step,
                stop_step,
            ))
        if handle == 0:
            raise RuntimeError("conditional WHILE graph returned a null handle")
        return cls(
            handle,
            extension,
            device,
            source_graphs=(model_graph, tail_graph),
        )

    def replay(self) -> None:
        if self.closed:
            raise RuntimeError("conditional graph is closed")
        with torch.cuda.device(self.device):
            self.extension.launch(self.handle)

    def close(self) -> None:
        if self.closed:
            return
        with torch.cuda.device(self.device):
            self.extension.destroy(self.handle)
        self.closed = True
        self.handle = 0

    def versions(self) -> dict[str, int]:
        if self.closed:
            raise RuntimeError("conditional graph is closed")
        return {
            "runtime": int(self.extension.runtime_version()),
            "driver": int(self.extension.driver_version()),
        }


__all__ = [
    "ConditionalGraph",
    "EXTENSION_NAME",
    "MINIMUM_WHILE_CUDA_VERSION",
]
