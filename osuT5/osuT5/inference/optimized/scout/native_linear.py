"""Verifier-only fused one-token linear CUDA scouts.

These are fp16/fp32 adaptations of the historical decoder-layer verifier kernels.
All arithmetic is accumulated in FP32 and results are cast back to the input dtype.
Nothing in production inference imports or dispatches to this module.
"""

from __future__ import annotations

import math
from typing import Any

import torch


_SCOUT_NATIVE_LINEAR: Any | None = None


def _load_inline(**kwargs):
    # Keep compiler discovery and extension setup cold until explicitly requested.
    from torch.utils.cpp_extension import load_inline

    return load_inline(**kwargs)


_CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor scout_one_token_rmsnorm_linear(
    torch::Tensor input,
    torch::Tensor norm_weight,
    torch::Tensor linear_weight,
    c10::optional<torch::Tensor> linear_bias,
    double eps,
    int64_t outputs_per_block);
torch::Tensor scout_one_token_linear_residual(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t outputs_per_block);
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

template<typename scalar_t, int OUTPUTS_PER_BLOCK>
__global__ void scout_rmsnorm_linear_kernel(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ norm_weight,
        const scalar_t* __restrict__ linear_weight,
        const scalar_t* __restrict__ linear_bias,
        scalar_t* __restrict__ output,
        int hidden_dim,
        int output_dim,
        int has_bias,
        float eps) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int out_idx = blockIdx.x * OUTPUTS_PER_BLOCK + warp_id;
    if (warp_id >= OUTPUTS_PER_BLOCK || out_idx >= output_dim) {
        return;
    }

    const scalar_t* row = linear_weight + out_idx * hidden_dim;
    float local_sq = 0.0f;
    float local_dot = 0.0f;
    for (int idx = lane; idx < hidden_dim; idx += 32) {
        const float x = static_cast<float>(input[idx]);
        local_sq += x * x;
        local_dot += x
            * static_cast<float>(norm_weight[idx])
            * static_cast<float>(row[idx]);
    }

    constexpr unsigned full_warp_mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sq += __shfl_down_sync(full_warp_mask, local_sq, offset);
        local_dot += __shfl_down_sync(full_warp_mask, local_dot, offset);
    }
    if (lane == 0) {
        const float inv_rms = rsqrtf(
            local_sq / static_cast<float>(hidden_dim) + eps);
        float result = local_dot * inv_rms;
        if (has_bias) {
            result += static_cast<float>(linear_bias[out_idx]);
        }
        output[out_idx] = static_cast<scalar_t>(result);
    }
}

template<typename scalar_t, int OUTPUTS_PER_BLOCK>
__global__ void scout_linear_residual_kernel(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ residual,
        const scalar_t* __restrict__ weight,
        const scalar_t* __restrict__ bias,
        scalar_t* __restrict__ output,
        int hidden_dim,
        int has_bias) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int out_idx = blockIdx.x * OUTPUTS_PER_BLOCK + warp_id;
    if (warp_id >= OUTPUTS_PER_BLOCK || out_idx >= hidden_dim) {
        return;
    }

    const scalar_t* row = weight + out_idx * hidden_dim;
    float sum = 0.0f;
    for (int idx = lane; idx < hidden_dim; idx += 32) {
        sum += static_cast<float>(input[idx]) * static_cast<float>(row[idx]);
    }

    constexpr unsigned full_warp_mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(full_warp_mask, sum, offset);
    }
    if (lane == 0) {
        if (has_bias) {
            sum += static_cast<float>(bias[out_idx]);
        }
        const float result = static_cast<float>(residual[out_idx]) + sum;
        output[out_idx] = static_cast<scalar_t>(result);
    }
}

template<typename scalar_t>
void launch_rmsnorm_linear(
        const torch::Tensor& input,
        const torch::Tensor& norm_weight,
        const torch::Tensor& linear_weight,
        const scalar_t* bias,
        torch::Tensor& output,
        int hidden_dim,
        int output_dim,
        int has_bias,
        float eps,
        int64_t outputs_per_block,
        cudaStream_t stream) {
    if (outputs_per_block == 2) {
        scout_rmsnorm_linear_kernel<scalar_t, 2><<<(output_dim + 1) / 2, 64, 0, stream>>>(
            input.data_ptr<scalar_t>(), norm_weight.data_ptr<scalar_t>(),
            linear_weight.data_ptr<scalar_t>(), bias, output.data_ptr<scalar_t>(),
            hidden_dim, output_dim, has_bias, eps);
    } else if (outputs_per_block == 4) {
        scout_rmsnorm_linear_kernel<scalar_t, 4><<<(output_dim + 3) / 4, 128, 0, stream>>>(
            input.data_ptr<scalar_t>(), norm_weight.data_ptr<scalar_t>(),
            linear_weight.data_ptr<scalar_t>(), bias, output.data_ptr<scalar_t>(),
            hidden_dim, output_dim, has_bias, eps);
    } else {
        scout_rmsnorm_linear_kernel<scalar_t, 8><<<(output_dim + 7) / 8, 256, 0, stream>>>(
            input.data_ptr<scalar_t>(), norm_weight.data_ptr<scalar_t>(),
            linear_weight.data_ptr<scalar_t>(), bias, output.data_ptr<scalar_t>(),
            hidden_dim, output_dim, has_bias, eps);
    }
}

template<typename scalar_t>
void launch_linear_residual(
        const torch::Tensor& input,
        const torch::Tensor& residual,
        const torch::Tensor& weight,
        const scalar_t* bias,
        torch::Tensor& output,
        int hidden_dim,
        int has_bias,
        int64_t outputs_per_block,
        cudaStream_t stream) {
    if (outputs_per_block == 2) {
        scout_linear_residual_kernel<scalar_t, 2><<<(hidden_dim + 1) / 2, 64, 0, stream>>>(
            input.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(), bias, output.data_ptr<scalar_t>(),
            hidden_dim, has_bias);
    } else if (outputs_per_block == 4) {
        scout_linear_residual_kernel<scalar_t, 4><<<(hidden_dim + 3) / 4, 128, 0, stream>>>(
            input.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(), bias, output.data_ptr<scalar_t>(),
            hidden_dim, has_bias);
    } else {
        scout_linear_residual_kernel<scalar_t, 8><<<(hidden_dim + 7) / 8, 256, 0, stream>>>(
            input.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(), bias, output.data_ptr<scalar_t>(),
            hidden_dim, has_bias);
    }
}

torch::Tensor scout_one_token_rmsnorm_linear(
        torch::Tensor input,
        torch::Tensor norm_weight,
        torch::Tensor linear_weight,
        c10::optional<torch::Tensor> linear_bias,
        double eps,
        int64_t outputs_per_block) {
    TORCH_CHECK(input.is_cuda() && norm_weight.is_cuda() && linear_weight.is_cuda(),
                "input and weights must be CUDA tensors");
    TORCH_CHECK(input.device() == norm_weight.device() && input.device() == linear_weight.device(),
                "input and weights must use one CUDA device");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16 || input.scalar_type() == torch::kFloat32,
                "input and weights must be fp16 or fp32");
    TORCH_CHECK(norm_weight.scalar_type() == input.scalar_type()
                && linear_weight.scalar_type() == input.scalar_type(),
                "input and weight dtypes must match");
    TORCH_CHECK(input.dim() == 3 && input.size(0) == 1 && input.size(1) == 1,
                "input must have shape [1, 1, hidden_dim]");
    TORCH_CHECK(norm_weight.dim() == 1, "norm_weight must have shape [hidden_dim]");
    TORCH_CHECK(linear_weight.dim() == 2,
                "linear_weight must have shape [output_dim, hidden_dim]");
    const int hidden_dim = static_cast<int>(input.size(2));
    const int output_dim = static_cast<int>(linear_weight.size(0));
    TORCH_CHECK(hidden_dim > 0 && output_dim > 0, "hidden_dim and output_dim must be positive");
    TORCH_CHECK(norm_weight.numel() == hidden_dim, "norm_weight/hidden_dim mismatch");
    TORCH_CHECK(linear_weight.size(1) == hidden_dim, "linear_weight hidden_dim mismatch");
    TORCH_CHECK(std::isfinite(eps) && eps >= 0.0, "eps must be finite and non-negative");
    TORCH_CHECK(outputs_per_block == 2 || outputs_per_block == 4 || outputs_per_block == 8,
                "outputs_per_block must be 2, 4, or 8");

    auto input_contiguous = input.contiguous();
    auto norm_weight_contiguous = norm_weight.contiguous();
    auto linear_weight_contiguous = linear_weight.contiguous();
    torch::Tensor bias_contiguous;
    const void* bias_untyped = nullptr;
    int has_bias = 0;
    if (linear_bias.has_value()) {
        bias_contiguous = linear_bias.value().contiguous();
        TORCH_CHECK(bias_contiguous.is_cuda() && bias_contiguous.device() == input.device(),
                    "linear_bias must use the input CUDA device");
        TORCH_CHECK(bias_contiguous.scalar_type() == input.scalar_type(),
                    "linear_bias dtype must match input");
        TORCH_CHECK(bias_contiguous.numel() == output_dim, "linear_bias/output_dim mismatch");
        bias_untyped = bias_contiguous.data_ptr();
        has_bias = 1;
    }

    auto output = torch::empty({1, 1, output_dim}, input.options());
    c10::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "scout_rmsnorm_linear", [&] {
        launch_rmsnorm_linear<scalar_t>(
            input_contiguous, norm_weight_contiguous, linear_weight_contiguous,
            static_cast<const scalar_t*>(bias_untyped), output, hidden_dim, output_dim,
            has_bias, static_cast<float>(eps), outputs_per_block, stream);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor scout_one_token_linear_residual(
        torch::Tensor input,
        torch::Tensor residual,
        torch::Tensor weight,
        c10::optional<torch::Tensor> bias,
        int64_t outputs_per_block) {
    TORCH_CHECK(input.is_cuda() && residual.is_cuda() && weight.is_cuda(),
                "input/residual/weight must be CUDA tensors");
    TORCH_CHECK(input.device() == residual.device() && input.device() == weight.device(),
                "input/residual/weight must use one CUDA device");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16 || input.scalar_type() == torch::kFloat32,
                "input/residual/weight must be fp16 or fp32");
    TORCH_CHECK(residual.scalar_type() == input.scalar_type()
                && weight.scalar_type() == input.scalar_type(),
                "input/residual/weight dtypes must match");
    TORCH_CHECK(input.dim() == 3 && input.size(0) == 1 && input.size(1) == 1,
                "input must have shape [1, 1, hidden_dim]");
    TORCH_CHECK(residual.sizes() == input.sizes(), "residual shape must match input");
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [hidden_dim, hidden_dim]");
    const int hidden_dim = static_cast<int>(input.size(2));
    TORCH_CHECK(hidden_dim > 0, "hidden_dim must be positive");
    TORCH_CHECK(weight.size(0) == hidden_dim && weight.size(1) == hidden_dim,
                "weight hidden_dim mismatch");
    TORCH_CHECK(outputs_per_block == 2 || outputs_per_block == 4 || outputs_per_block == 8,
                "outputs_per_block must be 2, 4, or 8");

    auto input_contiguous = input.contiguous();
    auto residual_contiguous = residual.contiguous();
    auto weight_contiguous = weight.contiguous();
    torch::Tensor bias_contiguous;
    const void* bias_untyped = nullptr;
    int has_bias = 0;
    if (bias.has_value()) {
        bias_contiguous = bias.value().contiguous();
        TORCH_CHECK(bias_contiguous.is_cuda() && bias_contiguous.device() == input.device(),
                    "bias must use the input CUDA device");
        TORCH_CHECK(bias_contiguous.scalar_type() == input.scalar_type(),
                    "bias dtype must match input");
        TORCH_CHECK(bias_contiguous.numel() == hidden_dim, "bias/hidden_dim mismatch");
        bias_untyped = bias_contiguous.data_ptr();
        has_bias = 1;
    }

    auto output = torch::empty({1, 1, hidden_dim}, input.options());
    c10::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "scout_linear_residual", [&] {
        launch_linear_residual<scalar_t>(
            input_contiguous, residual_contiguous, weight_contiguous,
            static_cast<const scalar_t*>(bias_untyped), output, hidden_dim,
            has_bias, outputs_per_block, stream);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""


def _load_scout_native_linear():
    global _SCOUT_NATIVE_LINEAR
    if _SCOUT_NATIVE_LINEAR is None:
        _SCOUT_NATIVE_LINEAR = _load_inline(
            name="mapperatorinator_scout_native_linear_fp32_accum",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=[
                "scout_one_token_rmsnorm_linear",
                "scout_one_token_linear_residual",
            ],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _SCOUT_NATIVE_LINEAR


def preload_scout_native_linear():
    """Explicitly compile and cache the verifier-only fused linear extension."""

    return _load_scout_native_linear()


def _validate_outputs_per_block(outputs_per_block: int) -> None:
    if outputs_per_block not in (2, 4, 8):
        raise ValueError("outputs_per_block must be 2, 4, or 8")


def _validate_tensors(tensors: tuple[tuple[str, torch.Tensor], ...]) -> None:
    input_dtype = tensors[0][1].dtype
    if input_dtype not in (torch.float16, torch.float32):
        raise TypeError(f"tensors must be fp16 or fp32, got {input_dtype}")
    for name, tensor in tensors[1:]:
        if tensor.dtype != input_dtype:
            raise TypeError(f"{name} dtype must match input")
    if not all(tensor.is_cuda for _, tensor in tensors):
        raise ValueError("all tensors must be CUDA tensors")
    input_device = tensors[0][1].device
    if any(tensor.device != input_device for _, tensor in tensors[1:]):
        raise ValueError("all tensors must use one CUDA device")


def _validate_rmsnorm_linear(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor | None,
    eps: float,
    outputs_per_block: int,
) -> None:
    if input.ndim != 3 or input.shape[0] != 1 or input.shape[1] != 1:
        raise ValueError("input must have shape [1, 1, hidden_dim]")
    if norm_weight.ndim != 1:
        raise ValueError("norm_weight must have shape [hidden_dim]")
    if linear_weight.ndim != 2:
        raise ValueError("linear_weight must have shape [output_dim, hidden_dim]")
    hidden_dim = input.shape[2]
    output_dim = linear_weight.shape[0]
    if hidden_dim == 0 or output_dim == 0:
        raise ValueError("hidden_dim and output_dim must be positive")
    if norm_weight.numel() != hidden_dim or linear_weight.shape[1] != hidden_dim:
        raise ValueError("weight dimensions must match hidden_dim")
    if linear_bias is not None and linear_bias.numel() != output_dim:
        raise ValueError("linear_bias must have output_dim elements")
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and non-negative")
    _validate_outputs_per_block(outputs_per_block)
    tensors = [
        ("input", input),
        ("norm_weight", norm_weight),
        ("linear_weight", linear_weight),
    ]
    if linear_bias is not None:
        tensors.append(("linear_bias", linear_bias))
    _validate_tensors(tuple(tensors))


def _validate_linear_residual(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    outputs_per_block: int,
) -> None:
    if input.ndim != 3 or input.shape[0] != 1 or input.shape[1] != 1:
        raise ValueError("input must have shape [1, 1, hidden_dim]")
    if residual.shape != input.shape:
        raise ValueError("residual shape must match input")
    if weight.ndim != 2 or weight.shape != (input.shape[2], input.shape[2]):
        raise ValueError("weight must have shape [hidden_dim, hidden_dim]")
    if input.shape[2] == 0:
        raise ValueError("hidden_dim must be positive")
    if bias is not None and bias.numel() != input.shape[2]:
        raise ValueError("bias must have hidden_dim elements")
    _validate_outputs_per_block(outputs_per_block)
    tensors = [("input", input), ("residual", residual), ("weight", weight)]
    if bias is not None:
        tensors.append(("bias", bias))
    _validate_tensors(tuple(tensors))


def scout_one_token_rmsnorm_linear(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor | None = None,
    *,
    eps: float,
    outputs_per_block: int,
) -> torch.Tensor:
    """Run fused RMSNorm plus linear for one token, for verification only."""

    _validate_rmsnorm_linear(
        input,
        norm_weight,
        linear_weight,
        linear_bias,
        eps,
        outputs_per_block,
    )
    extension = _load_scout_native_linear()
    output = extension.scout_one_token_rmsnorm_linear(
        input,
        norm_weight,
        linear_weight,
        linear_bias,
        float(eps),
        int(outputs_per_block),
    )
    if output.dtype != input.dtype:
        raise RuntimeError("scout RMSNorm+Linear output dtype did not match input")
    return output


def scout_one_token_linear_residual(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    outputs_per_block: int,
) -> torch.Tensor:
    """Run fused linear plus residual for one token, for verification only."""

    _validate_linear_residual(input, residual, weight, bias, outputs_per_block)
    extension = _load_scout_native_linear()
    output = extension.scout_one_token_linear_residual(
        input,
        residual,
        weight,
        bias,
        int(outputs_per_block),
    )
    if output.dtype != input.dtype:
        raise RuntimeError("scout Linear+Residual output dtype did not match input")
    return output
