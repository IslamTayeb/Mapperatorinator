"""Verifier-only INT8 MLP weight-storage scout.

This module is deliberately outside production dispatch. Importing it is cold:
the CUDA extension is built only by ``preload_int8_mlp_extension`` or the
explicit candidate wrapper. Activations, normalization, biases, reductions,
accumulators, residuals, and outputs remain FP32; only FC1/FC2 matrix weights
use per-output-row symmetric INT8 storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


_INT8_MLP_EXTENSION = None


def _load_inline(**kwargs):
    from torch.utils.cpp_extension import load_inline

    return load_inline(**kwargs)


_CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor int8_weight_mlp_residual(
    torch::Tensor input, torch::Tensor norm_weight,
    torch::Tensor fc1_weight, torch::Tensor fc1_scale,
    c10::optional<torch::Tensor> fc1_bias,
    torch::Tensor fc2_weight, torch::Tensor fc2_scale,
    c10::optional<torch::Tensor> fc2_bias,
    double eps, int64_t fc1_outputs_per_block,
    int64_t fc2_outputs_per_block);

torch::Tensor int8_weight_rmsnorm_linear(
    torch::Tensor input, torch::Tensor norm_weight,
    torch::Tensor weight, torch::Tensor scale,
    c10::optional<torch::Tensor> bias,
    double eps, int64_t outputs_per_block);

torch::Tensor int8_weight_linear_residual(
    torch::Tensor input, torch::Tensor residual,
    torch::Tensor weight, torch::Tensor scale,
    c10::optional<torch::Tensor> bias,
    int64_t outputs_per_block);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "int8_weight_mlp_residual",
        &int8_weight_mlp_residual,
        "INT8 weight-only one-token MLP residual (CUDA)");
    module.def(
        "int8_weight_rmsnorm_linear",
        &int8_weight_rmsnorm_linear,
        "INT8 weight-only one-token RMSNorm plus linear (CUDA)");
    module.def(
        "int8_weight_linear_residual",
        &int8_weight_linear_residual,
        "INT8 weight-only one-token linear plus residual (CUDA)");
}
"""


_CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float int8_weight_dot(
        const float* __restrict__ input,
        const int8_t* __restrict__ weight,
        int input_dim,
        int lane) {
    const char4* weight4 = reinterpret_cast<const char4*>(weight);
    const int quads = input_dim / 4;
    float sum = 0.0f;
    for (int quad = lane; quad < quads; quad += 32) {
        const char4 q = weight4[quad];
        const int index = 4 * quad;
        sum = fmaf(input[index], static_cast<float>(q.x), sum);
        sum = fmaf(input[index + 1], static_cast<float>(q.y), sum);
        sum = fmaf(input[index + 2], static_cast<float>(q.z), sum);
        sum = fmaf(input[index + 3], static_cast<float>(q.w), sum);
    }
    return warp_sum(sum);
}

template<int OUTPUTS_PER_BLOCK, bool ADD_RESIDUAL, bool APPLY_GELU>
__global__ void int8_weight_linear_kernel(
        const float* __restrict__ input,
        const int8_t* __restrict__ weight,
        const float* __restrict__ scale,
        const float* __restrict__ bias,
        const float* __restrict__ residual,
        float* __restrict__ output,
        int input_dim,
        int output_dim,
        int has_bias) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int out = blockIdx.x * OUTPUTS_PER_BLOCK + warp;
    if (warp >= OUTPUTS_PER_BLOCK || out >= output_dim) {
        return;
    }
    float sum = int8_weight_dot(
        input, weight + static_cast<int64_t>(out) * input_dim,
        input_dim, lane);
    if (lane == 0) {
        sum *= scale[out];
        if (has_bias) {
            sum += bias[out];
        }
        if (APPLY_GELU) {
            sum = 0.5f * sum *
                (1.0f + erff(sum * 0.7071067811865475244f));
        }
        if (ADD_RESIDUAL) {
            sum += residual[out];
        }
        output[out] = sum;
    }
}

__global__ void rmsnorm_fp32_kernel(
        const float* __restrict__ input,
        const float* __restrict__ weight,
        float* __restrict__ output,
        int hidden_dim,
        float eps) {
    float square = 0.0f;
    for (int index = threadIdx.x; index < hidden_dim; index += blockDim.x) {
        const float value = input[index];
        square = fmaf(value, value, square);
    }
    __shared__ float scratch[256];
    scratch[threadIdx.x] = square;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inverse = rsqrtf(
        scratch[0] / static_cast<float>(hidden_dim) + eps);
    for (int index = threadIdx.x; index < hidden_dim; index += blockDim.x) {
        output[index] = input[index] * weight[index] * inverse;
    }
}

void check_outputs_per_block(int64_t value) {
    TORCH_CHECK(value == 2 || value == 4 || value == 8,
        "outputs_per_block must be 2, 4, or 8");
}

void check_quantized_linear(
        const torch::Tensor& input,
        const torch::Tensor& weight,
        const torch::Tensor& scale,
        const char* name) {
    TORCH_CHECK(weight.is_cuda() && weight.device() == input.device(),
        name, " weight must use the input CUDA device");
    TORCH_CHECK(weight.scalar_type() == torch::kInt8 && weight.is_contiguous(),
        name, " weight must be contiguous INT8");
    TORCH_CHECK(weight.dim() == 2 && weight.size(1) % 4 == 0,
        name, " weight must be [output_dim, input_dim] with input_dim divisible by 4");
    TORCH_CHECK(scale.is_cuda() && scale.device() == input.device(),
        name, " scale must use the input CUDA device");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32 && scale.is_contiguous(),
        name, " scale must remain contiguous FP32");
    TORCH_CHECK(scale.dim() == 1 && scale.numel() == weight.size(0),
        name, " scale must contain one value per output row");
}

const float* check_bias(
        const c10::optional<torch::Tensor>& bias,
        const torch::Tensor& input,
        int output_dim,
        const char* name) {
    if (!bias.has_value()) {
        return nullptr;
    }
    const auto value = bias.value();
    TORCH_CHECK(value.is_cuda() && value.device() == input.device(),
        name, " bias must use the input CUDA device");
    TORCH_CHECK(value.scalar_type() == torch::kFloat32
        && value.is_contiguous() && value.numel() == output_dim,
        name, " bias must remain contiguous FP32 with one value per output");
    return value.data_ptr<float>();
}

template<bool ADD_RESIDUAL, bool APPLY_GELU>
void launch_int8_linear(
        const torch::Tensor& input,
        const torch::Tensor& weight,
        const torch::Tensor& scale,
        const float* bias,
        const float* residual,
        torch::Tensor& output,
        int64_t outputs_per_block,
        cudaStream_t stream) {
    const int input_dim = static_cast<int>(weight.size(1));
    const int output_dim = static_cast<int>(weight.size(0));
    const int threads = static_cast<int>(outputs_per_block) * 32;
    const int blocks = (output_dim + static_cast<int>(outputs_per_block) - 1)
        / static_cast<int>(outputs_per_block);
    if (outputs_per_block == 2) {
        int8_weight_linear_kernel<2, ADD_RESIDUAL, APPLY_GELU>
            <<<blocks, threads, 0, stream>>>(
                input.data_ptr<float>(), weight.data_ptr<int8_t>(),
                scale.data_ptr<float>(), bias, residual,
                output.data_ptr<float>(), input_dim, output_dim,
                bias != nullptr);
    } else if (outputs_per_block == 4) {
        int8_weight_linear_kernel<4, ADD_RESIDUAL, APPLY_GELU>
            <<<blocks, threads, 0, stream>>>(
                input.data_ptr<float>(), weight.data_ptr<int8_t>(),
                scale.data_ptr<float>(), bias, residual,
                output.data_ptr<float>(), input_dim, output_dim,
                bias != nullptr);
    } else {
        int8_weight_linear_kernel<8, ADD_RESIDUAL, APPLY_GELU>
            <<<blocks, threads, 0, stream>>>(
                input.data_ptr<float>(), weight.data_ptr<int8_t>(),
                scale.data_ptr<float>(), bias, residual,
                output.data_ptr<float>(), input_dim, output_dim,
                bias != nullptr);
    }
}

torch::Tensor int8_weight_mlp_residual(
        torch::Tensor input, torch::Tensor norm_weight,
        torch::Tensor fc1_weight, torch::Tensor fc1_scale,
        c10::optional<torch::Tensor> fc1_bias,
        torch::Tensor fc2_weight, torch::Tensor fc2_scale,
        c10::optional<torch::Tensor> fc2_bias,
        double eps, int64_t fc1_outputs_per_block,
        int64_t fc2_outputs_per_block) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32 && input.is_contiguous(),
        "input activations must remain contiguous FP32");
    TORCH_CHECK(input.dim() == 3 && input.size(0) == 1 && input.size(1) == 1,
        "input must have shape [1, 1, hidden_dim]");
    TORCH_CHECK(norm_weight.is_cuda() && norm_weight.device() == input.device(),
        "norm weight must use the input CUDA device");
    TORCH_CHECK(norm_weight.scalar_type() == torch::kFloat32
        && norm_weight.is_contiguous(),
        "norm weight must remain contiguous FP32");
    check_outputs_per_block(fc1_outputs_per_block);
    check_outputs_per_block(fc2_outputs_per_block);
    check_quantized_linear(input, fc1_weight, fc1_scale, "fc1");
    check_quantized_linear(input, fc2_weight, fc2_scale, "fc2");
    const int hidden_dim = static_cast<int>(input.size(2));
    const int intermediate_dim = static_cast<int>(fc1_weight.size(0));
    TORCH_CHECK(fc1_weight.size(1) == hidden_dim,
        "fc1 input dimension must equal hidden dimension");
    TORCH_CHECK(fc2_weight.size(0) == hidden_dim
        && fc2_weight.size(1) == intermediate_dim,
        "fc2 weight shape must be [hidden_dim, intermediate_dim]");
    TORCH_CHECK(norm_weight.numel() == hidden_dim,
        "norm weight/input dimension mismatch");
    const float* fc1_bias_ptr = check_bias(
        fc1_bias, input, intermediate_dim, "fc1");
    const float* fc2_bias_ptr = check_bias(
        fc2_bias, input, hidden_dim, "fc2");
    c10::cuda::CUDAGuard guard(input.device());
    auto normalized = torch::empty_like(input);
    auto activated = torch::empty({1, 1, intermediate_dim}, input.options());
    auto output = torch::empty_like(input);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rmsnorm_fp32_kernel<<<1, 256, 0, stream>>>(
        input.data_ptr<float>(), norm_weight.data_ptr<float>(),
        normalized.data_ptr<float>(), hidden_dim, static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    launch_int8_linear<false, true>(
        normalized, fc1_weight, fc1_scale, fc1_bias_ptr, nullptr,
        activated, fc1_outputs_per_block, stream);
    launch_int8_linear<true, false>(
        activated, fc2_weight, fc2_scale, fc2_bias_ptr,
        input.data_ptr<float>(), output, fc2_outputs_per_block, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor int8_weight_rmsnorm_linear(
        torch::Tensor input, torch::Tensor norm_weight,
        torch::Tensor weight, torch::Tensor scale,
        c10::optional<torch::Tensor> bias,
        double eps, int64_t outputs_per_block) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32 && input.is_contiguous(),
        "input activations must remain contiguous FP32");
    TORCH_CHECK(input.dim() == 3 && input.size(0) == 1 && input.size(1) == 1,
        "input must have shape [1, 1, input_dim]");
    TORCH_CHECK(norm_weight.is_cuda() && norm_weight.device() == input.device(),
        "norm weight must use the input CUDA device");
    TORCH_CHECK(norm_weight.scalar_type() == torch::kFloat32
        && norm_weight.is_contiguous(),
        "norm weight must remain contiguous FP32");
    check_outputs_per_block(outputs_per_block);
    check_quantized_linear(input, weight, scale, "linear");
    const int input_dim = static_cast<int>(input.size(2));
    const int output_dim = static_cast<int>(weight.size(0));
    TORCH_CHECK(weight.size(1) == input_dim,
        "linear input dimension must equal activation dimension");
    TORCH_CHECK(norm_weight.numel() == input_dim,
        "norm weight/input dimension mismatch");
    const float* bias_ptr = check_bias(bias, input, output_dim, "linear");
    c10::cuda::CUDAGuard guard(input.device());
    auto normalized = torch::empty_like(input);
    auto output = torch::empty({1, 1, output_dim}, input.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rmsnorm_fp32_kernel<<<1, 256, 0, stream>>>(
        input.data_ptr<float>(), norm_weight.data_ptr<float>(),
        normalized.data_ptr<float>(), input_dim, static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    launch_int8_linear<false, false>(
        normalized, weight, scale, bias_ptr, nullptr,
        output, outputs_per_block, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor int8_weight_linear_residual(
        torch::Tensor input, torch::Tensor residual,
        torch::Tensor weight, torch::Tensor scale,
        c10::optional<torch::Tensor> bias,
        int64_t outputs_per_block) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32 && input.is_contiguous(),
        "input activations must remain contiguous FP32");
    TORCH_CHECK(input.dim() == 3 && input.size(0) == 1 && input.size(1) == 1,
        "input must have shape [1, 1, input_dim]");
    TORCH_CHECK(residual.is_cuda() && residual.device() == input.device(),
        "residual must use the input CUDA device");
    TORCH_CHECK(residual.scalar_type() == torch::kFloat32
        && residual.is_contiguous(),
        "residual must remain contiguous FP32");
    TORCH_CHECK(residual.dim() == 3 && residual.size(0) == 1
        && residual.size(1) == 1,
        "residual must have shape [1, 1, output_dim]");
    check_outputs_per_block(outputs_per_block);
    check_quantized_linear(input, weight, scale, "linear");
    const int input_dim = static_cast<int>(input.size(2));
    const int output_dim = static_cast<int>(weight.size(0));
    TORCH_CHECK(weight.size(1) == input_dim,
        "linear input dimension must equal activation dimension");
    TORCH_CHECK(residual.size(2) == output_dim,
        "residual output dimension mismatch");
    const float* bias_ptr = check_bias(bias, input, output_dim, "linear");
    c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::empty_like(residual);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_int8_linear<true, false>(
        input, weight, scale, bias_ptr, residual.data_ptr<float>(),
        output, outputs_per_block, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""


def _load_int8_mlp_extension():
    global _INT8_MLP_EXTENSION
    if _INT8_MLP_EXTENSION is None:
        _INT8_MLP_EXTENSION = _load_inline(
            name="mapperatorinator_int8_projection_scout_v2",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_75,code=sm_75"],
            verbose=False,
        )
    return _INT8_MLP_EXTENSION


def preload_int8_mlp_extension():
    """Build/load the opt-in verifier extension without changing dispatch."""

    return _load_int8_mlp_extension()


def _require_fp32_cuda(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must retain FP32 storage")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return value


def symmetric_int8_per_row(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D FP32 matrix with one symmetric FP32 scale per row."""

    if not isinstance(weight, torch.Tensor):
        raise TypeError("weight must be a tensor")
    if weight.dtype != torch.float32:
        raise TypeError("weight must use FP32 source storage")
    if weight.dim() != 2:
        raise ValueError("weight must be two-dimensional")
    if weight.size(1) % 4:
        raise ValueError("weight input dimension must be divisible by four")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous")
    maximum = weight.abs().amax(dim=1)
    scale = torch.where(maximum > 0, maximum / 127.0, torch.ones_like(maximum))
    quantized = torch.round(weight / scale[:, None]).clamp(-127, 127)
    return quantized.to(dtype=torch.int8).contiguous(), scale.float().contiguous()


@dataclass(frozen=True)
class Int8PackedLinear:
    weight: torch.Tensor
    scale: torch.Tensor
    bias: torch.Tensor | None
    source_weight_bytes: int

    @classmethod
    def from_module(cls, module: Any) -> "Int8PackedLinear":
        source = _require_fp32_cuda(getattr(module, "weight", None), name="source weight")
        bias = getattr(module, "bias", None)
        if bias is not None:
            _require_fp32_cuda(bias, name="source bias")
            if bias.device != source.device:
                raise ValueError("linear bias/weight device mismatch")
        weight, scale = symmetric_int8_per_row(source)
        return cls(
            weight=weight,
            scale=scale,
            bias=bias,
            source_weight_bytes=source.numel() * source.element_size(),
        )

    @property
    def packed_weight_bytes(self) -> int:
        return (
            self.weight.numel() * self.weight.element_size()
            + self.scale.numel() * self.scale.element_size()
        )


@dataclass(frozen=True)
class Int8MlpPack:
    fc1: Int8PackedLinear
    fc2: Int8PackedLinear

    @classmethod
    def from_layer(cls, layer: Any) -> "Int8MlpPack":
        return cls(
            fc1=Int8PackedLinear.from_module(layer.fc1),
            fc2=Int8PackedLinear.from_module(layer.fc2),
        )

    def memory_report(self) -> dict[str, int]:
        source = self.fc1.source_weight_bytes + self.fc2.source_weight_bytes
        packed = self.fc1.packed_weight_bytes + self.fc2.packed_weight_bytes
        return {
            "retained_fp32_source_weight_bytes": source,
            "packed_weight_bytes": packed,
            "resident_increment_bytes": packed,
            "hypothetical_saving_if_fp32_sources_freed_bytes": source - packed,
        }


def _require_int8_pack(value: Int8PackedLinear, *, name: str) -> None:
    if not isinstance(value, Int8PackedLinear):
        raise TypeError(f"{name} must be an Int8PackedLinear")
    if value.weight.dtype != torch.int8:
        raise TypeError(f"{name} weight must use INT8 storage")
    if value.scale.dtype != torch.float32:
        raise TypeError(f"{name} scale must retain FP32 storage")
    if value.weight.device.type != "cuda" or value.scale.device.type != "cuda":
        raise RuntimeError(f"{name} weight and scale must be CUDA tensors")
    if value.weight.device != value.scale.device:
        raise ValueError(f"{name} weight/scale device mismatch")
    if not value.weight.is_contiguous() or not value.scale.is_contiguous():
        raise ValueError(f"{name} weight and scale must be contiguous")


def int8_weight_mlp_residual(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    pack: Int8MlpPack,
    *,
    eps: float,
    fc1_outputs_per_block: int = 2,
    fc2_outputs_per_block: int = 4,
) -> torch.Tensor:
    """Run RMSNorm + INT8-weight FC1/GELU + INT8-weight FC2/residual."""

    _require_fp32_cuda(input, name="input")
    _require_fp32_cuda(norm_weight, name="norm weight")
    if not isinstance(pack, Int8MlpPack):
        raise TypeError("pack must be an Int8MlpPack")
    _require_int8_pack(pack.fc1, name="fc1")
    _require_int8_pack(pack.fc2, name="fc2")
    if pack.fc1.weight.device != input.device or pack.fc2.weight.device != input.device:
        raise ValueError("packed weights must use the input device")
    if fc1_outputs_per_block not in (2, 4, 8) or fc2_outputs_per_block not in (2, 4, 8):
        raise ValueError("outputs_per_block must be 2, 4, or 8")
    return _load_int8_mlp_extension().int8_weight_mlp_residual(
        input,
        norm_weight,
        pack.fc1.weight,
        pack.fc1.scale,
        pack.fc1.bias,
        pack.fc2.weight,
        pack.fc2.scale,
        pack.fc2.bias,
        float(eps),
        int(fc1_outputs_per_block),
        int(fc2_outputs_per_block),
    )


def int8_weight_rmsnorm_linear(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    pack: Int8PackedLinear,
    *,
    eps: float,
    outputs_per_block: int = 4,
) -> torch.Tensor:
    """Run FP32 RMSNorm and one INT8-weight linear projection."""

    _require_fp32_cuda(input, name="input")
    _require_fp32_cuda(norm_weight, name="norm weight")
    _require_int8_pack(pack, name="linear")
    if pack.weight.device != input.device or norm_weight.device != input.device:
        raise ValueError("projection tensors must use the input device")
    if outputs_per_block not in (2, 4, 8):
        raise ValueError("outputs_per_block must be 2, 4, or 8")
    return _load_int8_mlp_extension().int8_weight_rmsnorm_linear(
        input,
        norm_weight,
        pack.weight,
        pack.scale,
        pack.bias,
        float(eps),
        int(outputs_per_block),
    )


def int8_weight_linear_residual(
    input: torch.Tensor,
    residual: torch.Tensor,
    pack: Int8PackedLinear,
    *,
    outputs_per_block: int = 4,
) -> torch.Tensor:
    """Run one INT8-weight linear projection and add an FP32 residual."""

    _require_fp32_cuda(input, name="input")
    _require_fp32_cuda(residual, name="residual")
    _require_int8_pack(pack, name="linear")
    if pack.weight.device != input.device or residual.device != input.device:
        raise ValueError("projection tensors must use the input device")
    if outputs_per_block not in (2, 4, 8):
        raise ValueError("outputs_per_block must be 2, 4, or 8")
    return _load_int8_mlp_extension().int8_weight_linear_residual(
        input,
        residual,
        pack.weight,
        pack.scale,
        pack.bias,
        int(outputs_per_block),
    )


__all__ = [
    "Int8MlpPack",
    "Int8PackedLinear",
    "int8_weight_linear_residual",
    "int8_weight_mlp_residual",
    "int8_weight_rmsnorm_linear",
    "preload_int8_mlp_extension",
    "symmetric_int8_per_row",
]
