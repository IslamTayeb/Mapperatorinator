"""Opt-in FP16-weight/FP32-activation one-token CUDA kernels.

This module is deliberately outside the production kernel package.  Importing
it does not build an extension, convert model parameters, or change dispatch.
The scout stores only matrix weights in FP16; activations, normalization
weights, biases, residuals, outputs, and every reduction remain FP32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


_WEIGHT_ONLY_EXTENSION = None


def _load_inline(**kwargs):
    from torch.utils.cpp_extension import load_inline

    return load_inline(**kwargs)


_CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor weight_only_linear(
    torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias,
    int64_t outputs_per_block);
torch::Tensor weight_only_rmsnorm_linear(
    torch::Tensor input, torch::Tensor norm_weight, torch::Tensor weight,
    c10::optional<torch::Tensor> bias, double eps, int64_t outputs_per_block);
torch::Tensor weight_only_linear_residual(
    torch::Tensor input, torch::Tensor residual, torch::Tensor weight,
    c10::optional<torch::Tensor> bias, int64_t outputs_per_block);
torch::Tensor weight_only_mlp_residual(
    torch::Tensor input, torch::Tensor norm_weight, torch::Tensor fc1_weight,
    c10::optional<torch::Tensor> fc1_bias, torch::Tensor fc2_weight,
    c10::optional<torch::Tensor> fc2_bias, double eps,
    int64_t outputs_per_block);
"""


_CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float half_weight_dot(
        const float* __restrict__ input,
        const __half* __restrict__ weight,
        int input_dim,
        int lane) {
    const __half2* weight2 = reinterpret_cast<const __half2*>(weight);
    const int pairs = input_dim / 2;
    float sum = 0.0f;
    for (int pair = lane; pair < pairs; pair += 32) {
        const float2 w = __half22float2(weight2[pair]);
        sum = fmaf(input[2 * pair], w.x, sum);
        sum = fmaf(input[2 * pair + 1], w.y, sum);
    }
    return warp_sum(sum);
}

template<int OUTPUTS_PER_BLOCK, bool ADD_RESIDUAL, bool APPLY_GELU>
__global__ void weight_only_linear_kernel(
        const float* __restrict__ input,
        const __half* __restrict__ weight,
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
    float sum = half_weight_dot(
        input, weight + static_cast<int64_t>(out) * input_dim, input_dim, lane);
    if (lane == 0) {
        if (has_bias) {
            sum += bias[out];
        }
        if (APPLY_GELU) {
            sum = 0.5f * sum * (1.0f + erff(sum * 0.7071067811865475244f));
        }
        if (ADD_RESIDUAL) {
            sum += residual[out];
        }
        output[out] = sum;
    }
}

template<int OUTPUTS_PER_BLOCK>
__global__ void weight_only_rmsnorm_linear_kernel(
        const float* __restrict__ input,
        const float* __restrict__ norm_weight,
        const __half* __restrict__ weight,
        const float* __restrict__ bias,
        float* __restrict__ output,
        int hidden_dim,
        int output_dim,
        int has_bias,
        float eps) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int out = blockIdx.x * OUTPUTS_PER_BLOCK + warp;
    if (warp >= OUTPUTS_PER_BLOCK || out >= output_dim) {
        return;
    }
    const __half* row = weight + static_cast<int64_t>(out) * hidden_dim;
    const __half2* row2 = reinterpret_cast<const __half2*>(row);
    float local_square = 0.0f;
    float local_dot = 0.0f;
    const int pairs = hidden_dim / 2;
    for (int pair = lane; pair < pairs; pair += 32) {
        const int idx = 2 * pair;
        const float x0 = input[idx];
        const float x1 = input[idx + 1];
        const float2 w = __half22float2(row2[pair]);
        local_square = fmaf(x0, x0, local_square);
        local_square = fmaf(x1, x1, local_square);
        local_dot = fmaf(x0 * norm_weight[idx], w.x, local_dot);
        local_dot = fmaf(x1 * norm_weight[idx + 1], w.y, local_dot);
    }
    const float square = warp_sum(local_square);
    const float dot = warp_sum(local_dot);
    if (lane == 0) {
        float value = dot * rsqrtf(square / static_cast<float>(hidden_dim) + eps);
        if (has_bias) {
            value += bias[out];
        }
        output[out] = value;
    }
}

__global__ void rmsnorm_fp32_kernel(
        const float* __restrict__ input,
        const float* __restrict__ weight,
        float* __restrict__ output,
        int hidden_dim,
        float eps) {
    float square = 0.0f;
    for (int idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
        const float value = input[idx];
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
    const float inverse = rsqrtf(scratch[0] / static_cast<float>(hidden_dim) + eps);
    for (int idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
        output[idx] = input[idx] * weight[idx] * inverse;
    }
}

void check_common(torch::Tensor input, torch::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.device() == weight.device(), "input/weight device mismatch");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
        "input activations must use FP32 storage");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16,
        "matrix weights must use FP16 storage");
    TORCH_CHECK(input.dim() == 3 && input.size(0) == 1 && input.size(1) == 1,
        "input must have shape [1, 1, input_dim]");
    TORCH_CHECK(weight.dim() == 2, "weight must be two-dimensional");
    TORCH_CHECK(input.size(2) == weight.size(1), "input/weight dimension mismatch");
    TORCH_CHECK(input.size(2) % 2 == 0,
        "input dimension must be even for half2 weight loads");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(),
        "input and weight must be contiguous");
}

const float* check_bias(
        const c10::optional<torch::Tensor>& bias,
        const torch::Tensor& input,
        int output_dim) {
    if (!bias.has_value()) {
        return nullptr;
    }
    const auto value = bias.value();
    TORCH_CHECK(value.is_cuda() && value.device() == input.device(),
        "bias must use the input CUDA device");
    TORCH_CHECK(value.scalar_type() == torch::kFloat32,
        "bias must retain FP32 storage");
    TORCH_CHECK(value.is_contiguous() && value.numel() == output_dim,
        "bias must be contiguous with one value per output");
    return value.data_ptr<float>();
}

void check_outputs_per_block(int64_t value) {
    TORCH_CHECK(value == 2 || value == 4 || value == 8,
        "outputs_per_block must be 2, 4, or 8");
}

template<bool ADD_RESIDUAL, bool APPLY_GELU>
torch::Tensor launch_linear(
        torch::Tensor input,
        torch::Tensor weight,
        c10::optional<torch::Tensor> bias,
        c10::optional<torch::Tensor> residual,
        int64_t outputs_per_block) {
    check_common(input, weight);
    check_outputs_per_block(outputs_per_block);
    c10::cuda::CUDAGuard guard(input.device());
    const int input_dim = static_cast<int>(input.size(2));
    const int output_dim = static_cast<int>(weight.size(0));
    const float* bias_ptr = check_bias(bias, input, output_dim);
    const float* residual_ptr = nullptr;
    if (ADD_RESIDUAL) {
        TORCH_CHECK(residual.has_value(), "residual is required");
        const auto value = residual.value();
        TORCH_CHECK(value.is_cuda() && value.device() == input.device(),
            "residual must use the input CUDA device");
        TORCH_CHECK(value.scalar_type() == torch::kFloat32 && value.is_contiguous(),
            "residual must be contiguous FP32");
        TORCH_CHECK(value.numel() == output_dim, "residual/output dimension mismatch");
        residual_ptr = value.data_ptr<float>();
    }
    auto output = torch::empty({1, 1, output_dim}, input.options());
    const int threads = static_cast<int>(outputs_per_block) * 32;
    const int blocks = (output_dim + static_cast<int>(outputs_per_block) - 1)
        / static_cast<int>(outputs_per_block);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (outputs_per_block == 2) {
        weight_only_linear_kernel<2, ADD_RESIDUAL, APPLY_GELU>
            <<<blocks, threads, 0, stream>>>(input.data_ptr<float>(),
                reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
                bias_ptr, residual_ptr, output.data_ptr<float>(), input_dim,
                output_dim, bias_ptr != nullptr);
    } else if (outputs_per_block == 4) {
        weight_only_linear_kernel<4, ADD_RESIDUAL, APPLY_GELU>
            <<<blocks, threads, 0, stream>>>(input.data_ptr<float>(),
                reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
                bias_ptr, residual_ptr, output.data_ptr<float>(), input_dim,
                output_dim, bias_ptr != nullptr);
    } else {
        weight_only_linear_kernel<8, ADD_RESIDUAL, APPLY_GELU>
            <<<blocks, threads, 0, stream>>>(input.data_ptr<float>(),
                reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
                bias_ptr, residual_ptr, output.data_ptr<float>(), input_dim,
                output_dim, bias_ptr != nullptr);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor weight_only_linear(
        torch::Tensor input, torch::Tensor weight,
        c10::optional<torch::Tensor> bias, int64_t outputs_per_block) {
    return launch_linear<false, false>(input, weight, bias, c10::nullopt,
        outputs_per_block);
}

torch::Tensor weight_only_linear_residual(
        torch::Tensor input, torch::Tensor residual, torch::Tensor weight,
        c10::optional<torch::Tensor> bias, int64_t outputs_per_block) {
    return launch_linear<true, false>(input, weight, bias, residual,
        outputs_per_block);
}

torch::Tensor weight_only_rmsnorm_linear(
        torch::Tensor input, torch::Tensor norm_weight, torch::Tensor weight,
        c10::optional<torch::Tensor> bias, double eps,
        int64_t outputs_per_block) {
    check_common(input, weight);
    check_outputs_per_block(outputs_per_block);
    TORCH_CHECK(norm_weight.is_cuda() && norm_weight.device() == input.device(),
        "norm weight must use the input CUDA device");
    TORCH_CHECK(norm_weight.scalar_type() == torch::kFloat32
        && norm_weight.is_contiguous(), "norm weight must remain contiguous FP32");
    TORCH_CHECK(norm_weight.numel() == input.size(2),
        "norm weight/input dimension mismatch");
    c10::cuda::CUDAGuard guard(input.device());
    const int hidden_dim = static_cast<int>(input.size(2));
    const int output_dim = static_cast<int>(weight.size(0));
    const float* bias_ptr = check_bias(bias, input, output_dim);
    auto output = torch::empty({1, 1, output_dim}, input.options());
    const int threads = static_cast<int>(outputs_per_block) * 32;
    const int blocks = (output_dim + static_cast<int>(outputs_per_block) - 1)
        / static_cast<int>(outputs_per_block);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (outputs_per_block == 2) {
        weight_only_rmsnorm_linear_kernel<2><<<blocks, threads, 0, stream>>>(
            input.data_ptr<float>(), norm_weight.data_ptr<float>(),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()), bias_ptr,
            output.data_ptr<float>(), hidden_dim, output_dim, bias_ptr != nullptr,
            static_cast<float>(eps));
    } else if (outputs_per_block == 4) {
        weight_only_rmsnorm_linear_kernel<4><<<blocks, threads, 0, stream>>>(
            input.data_ptr<float>(), norm_weight.data_ptr<float>(),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()), bias_ptr,
            output.data_ptr<float>(), hidden_dim, output_dim, bias_ptr != nullptr,
            static_cast<float>(eps));
    } else {
        weight_only_rmsnorm_linear_kernel<8><<<blocks, threads, 0, stream>>>(
            input.data_ptr<float>(), norm_weight.data_ptr<float>(),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()), bias_ptr,
            output.data_ptr<float>(), hidden_dim, output_dim, bias_ptr != nullptr,
            static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor weight_only_mlp_residual(
        torch::Tensor input, torch::Tensor norm_weight,
        torch::Tensor fc1_weight, c10::optional<torch::Tensor> fc1_bias,
        torch::Tensor fc2_weight, c10::optional<torch::Tensor> fc2_bias,
        double eps, int64_t outputs_per_block) {
    check_common(input, fc1_weight);
    check_outputs_per_block(outputs_per_block);
    TORCH_CHECK(fc2_weight.is_cuda() && fc2_weight.device() == input.device(),
        "fc2 weight must use the input CUDA device");
    TORCH_CHECK(fc2_weight.scalar_type() == torch::kFloat16
        && fc2_weight.is_contiguous(), "fc2 weight must be contiguous FP16");
    const int hidden_dim = static_cast<int>(input.size(2));
    const int intermediate_dim = static_cast<int>(fc1_weight.size(0));
    TORCH_CHECK(fc2_weight.dim() == 2 && fc2_weight.size(0) == hidden_dim
        && fc2_weight.size(1) == intermediate_dim,
        "fc2 weight shape must be [hidden_dim, intermediate_dim]");
    TORCH_CHECK(intermediate_dim % 2 == 0,
        "intermediate dimension must be even for half2 weight loads");
    TORCH_CHECK(norm_weight.is_cuda() && norm_weight.device() == input.device()
        && norm_weight.scalar_type() == torch::kFloat32
        && norm_weight.is_contiguous() && norm_weight.numel() == hidden_dim,
        "norm weight must remain contiguous FP32 with hidden_dim values");
    c10::cuda::CUDAGuard guard(input.device());
    const float* fc1_bias_ptr = check_bias(fc1_bias, input, intermediate_dim);
    const float* fc2_bias_ptr = check_bias(fc2_bias, input, hidden_dim);
    auto normalized = torch::empty({1, 1, hidden_dim}, input.options());
    auto activated = torch::empty({1, 1, intermediate_dim}, input.options());
    auto output = torch::empty_like(input);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rmsnorm_fp32_kernel<<<1, 256, 0, stream>>>(input.data_ptr<float>(),
        norm_weight.data_ptr<float>(), normalized.data_ptr<float>(), hidden_dim,
        static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int threads = static_cast<int>(outputs_per_block) * 32;
    const int fc1_blocks = (intermediate_dim
        + static_cast<int>(outputs_per_block) - 1)
        / static_cast<int>(outputs_per_block);
    const int fc2_blocks = (hidden_dim + static_cast<int>(outputs_per_block) - 1)
        / static_cast<int>(outputs_per_block);
    if (outputs_per_block == 2) {
        weight_only_linear_kernel<2, false, true><<<fc1_blocks, threads, 0, stream>>>(
            normalized.data_ptr<float>(),
            reinterpret_cast<const __half*>(fc1_weight.data_ptr<at::Half>()),
            fc1_bias_ptr, nullptr, activated.data_ptr<float>(), hidden_dim,
            intermediate_dim, fc1_bias_ptr != nullptr);
        weight_only_linear_kernel<2, true, false><<<fc2_blocks, threads, 0, stream>>>(
            activated.data_ptr<float>(),
            reinterpret_cast<const __half*>(fc2_weight.data_ptr<at::Half>()),
            fc2_bias_ptr, input.data_ptr<float>(), output.data_ptr<float>(),
            intermediate_dim, hidden_dim, fc2_bias_ptr != nullptr);
    } else if (outputs_per_block == 4) {
        weight_only_linear_kernel<4, false, true><<<fc1_blocks, threads, 0, stream>>>(
            normalized.data_ptr<float>(),
            reinterpret_cast<const __half*>(fc1_weight.data_ptr<at::Half>()),
            fc1_bias_ptr, nullptr, activated.data_ptr<float>(), hidden_dim,
            intermediate_dim, fc1_bias_ptr != nullptr);
        weight_only_linear_kernel<4, true, false><<<fc2_blocks, threads, 0, stream>>>(
            activated.data_ptr<float>(),
            reinterpret_cast<const __half*>(fc2_weight.data_ptr<at::Half>()),
            fc2_bias_ptr, input.data_ptr<float>(), output.data_ptr<float>(),
            intermediate_dim, hidden_dim, fc2_bias_ptr != nullptr);
    } else {
        weight_only_linear_kernel<8, false, true><<<fc1_blocks, threads, 0, stream>>>(
            normalized.data_ptr<float>(),
            reinterpret_cast<const __half*>(fc1_weight.data_ptr<at::Half>()),
            fc1_bias_ptr, nullptr, activated.data_ptr<float>(), hidden_dim,
            intermediate_dim, fc1_bias_ptr != nullptr);
        weight_only_linear_kernel<8, true, false><<<fc2_blocks, threads, 0, stream>>>(
            activated.data_ptr<float>(),
            reinterpret_cast<const __half*>(fc2_weight.data_ptr<at::Half>()),
            fc2_bias_ptr, input.data_ptr<float>(), output.data_ptr<float>(),
            intermediate_dim, hidden_dim, fc2_bias_ptr != nullptr);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""


def _load_weight_only_extension():
    global _WEIGHT_ONLY_EXTENSION
    if _WEIGHT_ONLY_EXTENSION is None:
        _WEIGHT_ONLY_EXTENSION = _load_inline(
            name="mapperatorinator_weight_only_fp16_scout",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=[
                "weight_only_linear",
                "weight_only_rmsnorm_linear",
                "weight_only_linear_residual",
                "weight_only_mlp_residual",
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _WEIGHT_ONLY_EXTENSION


def preload_weight_only_extension():
    return _load_weight_only_extension()


def _require_fp32_activation(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must retain FP32 storage")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _require_fp16_weight(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype != torch.float16:
        raise TypeError(f"{name} must use FP16 storage")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


@dataclass(frozen=True)
class PackedLinear:
    weight: torch.Tensor
    bias: torch.Tensor | None
    source_weight_bytes: int

    @classmethod
    def from_module(cls, module: Any) -> "PackedLinear":
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        if not isinstance(weight, torch.Tensor):
            raise TypeError("linear module must expose a tensor weight")
        _require_fp32_activation(weight, name="source weight")
        if bias is not None:
            _require_fp32_activation(bias, name="source bias")
            if bias.device != weight.device:
                raise ValueError("linear bias/weight device mismatch")
        packed = weight.to(dtype=torch.float16).contiguous()
        return cls(
            weight=packed,
            bias=bias,
            source_weight_bytes=weight.numel() * weight.element_size(),
        )

    @property
    def packed_weight_bytes(self) -> int:
        return self.weight.numel() * self.weight.element_size()


@dataclass(frozen=True)
class DecoderWeightPack:
    self_qkv: PackedLinear
    self_out: PackedLinear
    cross_q: PackedLinear
    cross_out: PackedLinear
    fc1: PackedLinear
    fc2: PackedLinear

    @classmethod
    def from_layer(cls, layer: Any) -> "DecoderWeightPack":
        return cls(
            self_qkv=PackedLinear.from_module(layer.self_attn.Wqkv),
            self_out=PackedLinear.from_module(layer.self_attn.Wo),
            cross_q=PackedLinear.from_module(layer.cross_attn.Wq),
            cross_out=PackedLinear.from_module(layer.cross_attn.Wo),
            fc1=PackedLinear.from_module(layer.fc1),
            fc2=PackedLinear.from_module(layer.fc2),
        )

    def linears(self) -> tuple[PackedLinear, ...]:
        return (
            self.self_qkv,
            self.self_out,
            self.cross_q,
            self.cross_out,
            self.fc1,
            self.fc2,
        )

    def memory_report(self) -> dict[str, int]:
        source = sum(value.source_weight_bytes for value in self.linears())
        packed = sum(value.packed_weight_bytes for value in self.linears())
        return {
            "source_weight_bytes": source,
            "packed_weight_bytes": packed,
            "projected_replacement_saving_bytes": source - packed,
            "scout_resident_increment_bytes": packed,
        }


def weight_only_linear(
    input: torch.Tensor,
    packed: PackedLinear,
    *,
    outputs_per_block: int = 8,
) -> torch.Tensor:
    _require_fp32_activation(input, name="input")
    _require_fp16_weight(packed.weight, name="weight")
    return _load_weight_only_extension().weight_only_linear(
        input, packed.weight, packed.bias, int(outputs_per_block)
    )


def weight_only_rmsnorm_linear(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    packed: PackedLinear,
    *,
    eps: float,
    outputs_per_block: int = 8,
) -> torch.Tensor:
    _require_fp32_activation(input, name="input")
    _require_fp32_activation(norm_weight, name="norm weight")
    _require_fp16_weight(packed.weight, name="weight")
    return _load_weight_only_extension().weight_only_rmsnorm_linear(
        input,
        norm_weight,
        packed.weight,
        packed.bias,
        float(eps),
        int(outputs_per_block),
    )


def weight_only_linear_residual(
    input: torch.Tensor,
    residual: torch.Tensor,
    packed: PackedLinear,
    *,
    outputs_per_block: int = 8,
) -> torch.Tensor:
    _require_fp32_activation(input, name="input")
    _require_fp32_activation(residual, name="residual")
    _require_fp16_weight(packed.weight, name="weight")
    return _load_weight_only_extension().weight_only_linear_residual(
        input,
        residual,
        packed.weight,
        packed.bias,
        int(outputs_per_block),
    )


def weight_only_mlp_residual(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    fc1: PackedLinear,
    fc2: PackedLinear,
    *,
    eps: float,
    outputs_per_block: int = 8,
) -> torch.Tensor:
    _require_fp32_activation(input, name="input")
    _require_fp32_activation(norm_weight, name="norm weight")
    _require_fp16_weight(fc1.weight, name="fc1 weight")
    _require_fp16_weight(fc2.weight, name="fc2 weight")
    return _load_weight_only_extension().weight_only_mlp_residual(
        input,
        norm_weight,
        fc1.weight,
        fc1.bias,
        fc2.weight,
        fc2.bias,
        float(eps),
        int(outputs_per_block),
    )


__all__ = [
    "DecoderWeightPack",
    "PackedLinear",
    "preload_weight_only_extension",
    "weight_only_linear",
    "weight_only_linear_residual",
    "weight_only_mlp_residual",
    "weight_only_rmsnorm_linear",
]
