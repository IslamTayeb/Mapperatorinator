from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline


_NATIVE_DECODER_LAYER = None


def _load_native_decoder_layer():
    global _NATIVE_DECODER_LAYER
    if _NATIVE_DECODER_LAYER is not None:
        return _NATIVE_DECODER_LAYER

    cuda_source = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void rmsnorm_one_token_kernel(
        const float* __restrict__ input,
        const float* __restrict__ weight,
        float* __restrict__ normalized,
        int hidden_dim,
        float eps) {
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    for (int idx = tid; idx < hidden_dim; idx += BLOCK_SIZE) {
        float value = input[idx];
        local_sum += value * value;
    }

    __shared__ float scratch[BLOCK_SIZE];
    scratch[tid] = local_sum;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(scratch[0] / static_cast<float>(hidden_dim) + eps);
    for (int idx = tid; idx < hidden_dim; idx += BLOCK_SIZE) {
        normalized[idx] = input[idx] * inv_rms * weight[idx];
    }
}

__device__ __forceinline__ float exact_gelu(float value) {
    return 0.5f * value * (1.0f + erff(value * 0.7071067811865475244f));
}

template<int OUTPUTS_PER_BLOCK>
__global__ void fc1_gelu_warp_group_kernel(
        const float* __restrict__ input,
        const float* __restrict__ weight,
        const float* __restrict__ bias,
        float* __restrict__ output,
        int hidden_dim,
        int intermediate_dim,
        int has_bias) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int out_idx = blockIdx.x * OUTPUTS_PER_BLOCK + warp_id;
    if (warp_id >= OUTPUTS_PER_BLOCK || out_idx >= intermediate_dim) {
        return;
    }

    const float* w = weight + out_idx * hidden_dim;
    float sum = 0.0f;
    for (int idx = lane; idx < hidden_dim; idx += 32) {
        sum += input[idx] * w[idx];
    }

    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    if (lane == 0) {
        if (has_bias) {
            sum += bias[out_idx];
        }
        output[out_idx] = exact_gelu(sum);
    }
}

template<int OUTPUTS_PER_BLOCK>
__global__ void fc2_residual_warp_group_kernel(
        const float* __restrict__ input,
        const float* __restrict__ weight,
        const float* __restrict__ bias,
        const float* __restrict__ residual,
        float* __restrict__ output,
        int hidden_dim,
        int intermediate_dim,
        int has_bias) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int out_idx = blockIdx.x * OUTPUTS_PER_BLOCK + warp_id;
    if (warp_id >= OUTPUTS_PER_BLOCK || out_idx >= hidden_dim) {
        return;
    }

    const float* w = weight + out_idx * intermediate_dim;
    float sum = 0.0f;
    for (int idx = lane; idx < intermediate_dim; idx += 32) {
        sum += input[idx] * w[idx];
    }

    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    if (lane == 0) {
        if (has_bias) {
            sum += bias[out_idx];
        }
        output[out_idx] = residual[out_idx] + sum;
    }
}

torch::Tensor one_token_mlp_residual(
        torch::Tensor input,
        torch::Tensor norm_weight,
        torch::Tensor fc1_weight,
        c10::optional<torch::Tensor> fc1_bias,
        torch::Tensor fc2_weight,
        c10::optional<torch::Tensor> fc2_bias,
        double eps,
        int64_t outputs_per_block) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(norm_weight.is_cuda(), "norm_weight must be a CUDA tensor");
    TORCH_CHECK(fc1_weight.is_cuda() && fc2_weight.is_cuda(), "MLP weights must be CUDA tensors");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be fp32");
    TORCH_CHECK(norm_weight.scalar_type() == torch::kFloat32, "norm_weight must be fp32");
    TORCH_CHECK(fc1_weight.scalar_type() == torch::kFloat32, "fc1_weight must be fp32");
    TORCH_CHECK(fc2_weight.scalar_type() == torch::kFloat32, "fc2_weight must be fp32");
    TORCH_CHECK(input.dim() == 3 && input.size(0) == 1 && input.size(1) == 1,
            "input must have shape [1, 1, hidden_dim]");
    TORCH_CHECK(norm_weight.dim() == 1, "norm_weight must have shape [hidden_dim]");
    TORCH_CHECK(fc1_weight.dim() == 2, "fc1_weight must have shape [intermediate_dim, hidden_dim]");
    TORCH_CHECK(fc2_weight.dim() == 2, "fc2_weight must have shape [hidden_dim, intermediate_dim]");

    const int hidden_dim = static_cast<int>(input.size(2));
    const int intermediate_dim = static_cast<int>(fc1_weight.size(0));
    TORCH_CHECK(norm_weight.numel() == hidden_dim, "norm_weight/hidden_dim mismatch");
    TORCH_CHECK(fc1_weight.size(1) == hidden_dim, "fc1_weight hidden_dim mismatch");
    TORCH_CHECK(fc2_weight.size(0) == hidden_dim, "fc2_weight hidden_dim mismatch");
    TORCH_CHECK(fc2_weight.size(1) == intermediate_dim, "fc2_weight intermediate_dim mismatch");

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto input_contiguous = input.contiguous();
    auto norm_weight_contiguous = norm_weight.contiguous();
    auto fc1_weight_contiguous = fc1_weight.contiguous();
    auto fc2_weight_contiguous = fc2_weight.contiguous();

    torch::Tensor fc1_bias_contiguous;
    const float* fc1_bias_ptr = nullptr;
    int has_fc1_bias = 0;
    if (fc1_bias.has_value()) {
        fc1_bias_contiguous = fc1_bias.value().contiguous();
        TORCH_CHECK(fc1_bias_contiguous.is_cuda(), "fc1_bias must be CUDA");
        TORCH_CHECK(fc1_bias_contiguous.scalar_type() == torch::kFloat32, "fc1_bias must be fp32");
        TORCH_CHECK(fc1_bias_contiguous.numel() == intermediate_dim, "fc1_bias/intermediate_dim mismatch");
        fc1_bias_ptr = fc1_bias_contiguous.data_ptr<float>();
        has_fc1_bias = 1;
    }

    torch::Tensor fc2_bias_contiguous;
    const float* fc2_bias_ptr = nullptr;
    int has_fc2_bias = 0;
    if (fc2_bias.has_value()) {
        fc2_bias_contiguous = fc2_bias.value().contiguous();
        TORCH_CHECK(fc2_bias_contiguous.is_cuda(), "fc2_bias must be CUDA");
        TORCH_CHECK(fc2_bias_contiguous.scalar_type() == torch::kFloat32, "fc2_bias must be fp32");
        TORCH_CHECK(fc2_bias_contiguous.numel() == hidden_dim, "fc2_bias/hidden_dim mismatch");
        fc2_bias_ptr = fc2_bias_contiguous.data_ptr<float>();
        has_fc2_bias = 1;
    }

    auto normalized = torch::empty({hidden_dim}, input.options());
    auto activated = torch::empty({intermediate_dim}, input.options());
    auto output = torch::empty({1, 1, hidden_dim}, input.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rmsnorm_one_token_kernel<256><<<1, 256, 0, stream>>>(
        input_contiguous.data_ptr<float>(),
        norm_weight_contiguous.data_ptr<float>(),
        normalized.data_ptr<float>(),
        hidden_dim,
        static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (outputs_per_block == 2) {
        fc1_gelu_warp_group_kernel<2><<<(intermediate_dim + 1) / 2, 64, 0, stream>>>(
            normalized.data_ptr<float>(),
            fc1_weight_contiguous.data_ptr<float>(),
            fc1_bias_ptr,
            activated.data_ptr<float>(),
            hidden_dim,
            intermediate_dim,
            has_fc1_bias);
    } else if (outputs_per_block == 4) {
        fc1_gelu_warp_group_kernel<4><<<(intermediate_dim + 3) / 4, 128, 0, stream>>>(
            normalized.data_ptr<float>(),
            fc1_weight_contiguous.data_ptr<float>(),
            fc1_bias_ptr,
            activated.data_ptr<float>(),
            hidden_dim,
            intermediate_dim,
            has_fc1_bias);
    } else if (outputs_per_block == 8) {
        fc1_gelu_warp_group_kernel<8><<<(intermediate_dim + 7) / 8, 256, 0, stream>>>(
            normalized.data_ptr<float>(),
            fc1_weight_contiguous.data_ptr<float>(),
            fc1_bias_ptr,
            activated.data_ptr<float>(),
            hidden_dim,
            intermediate_dim,
            has_fc1_bias);
    } else {
        TORCH_CHECK(false, "unsupported outputs_per_block; expected 2, 4, or 8");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (outputs_per_block == 2) {
        fc2_residual_warp_group_kernel<2><<<(hidden_dim + 1) / 2, 64, 0, stream>>>(
            activated.data_ptr<float>(),
            fc2_weight_contiguous.data_ptr<float>(),
            fc2_bias_ptr,
            input_contiguous.data_ptr<float>(),
            output.data_ptr<float>(),
            hidden_dim,
            intermediate_dim,
            has_fc2_bias);
    } else if (outputs_per_block == 4) {
        fc2_residual_warp_group_kernel<4><<<(hidden_dim + 3) / 4, 128, 0, stream>>>(
            activated.data_ptr<float>(),
            fc2_weight_contiguous.data_ptr<float>(),
            fc2_bias_ptr,
            input_contiguous.data_ptr<float>(),
            output.data_ptr<float>(),
            hidden_dim,
            intermediate_dim,
            has_fc2_bias);
    } else {
        fc2_residual_warp_group_kernel<8><<<(hidden_dim + 7) / 8, 256, 0, stream>>>(
            activated.data_ptr<float>(),
            fc2_weight_contiguous.data_ptr<float>(),
            fc2_bias_ptr,
            input_contiguous.data_ptr<float>(),
            output.data_ptr<float>(),
            hidden_dim,
            intermediate_dim,
            has_fc2_bias);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

    _NATIVE_DECODER_LAYER = load_inline(
        name="mapperatorinator_native_decoder_layer",
        cpp_sources=(
            "#include <torch/extension.h>\n"
            "torch::Tensor one_token_mlp_residual("
            "torch::Tensor input, torch::Tensor norm_weight, torch::Tensor fc1_weight, "
            "c10::optional<torch::Tensor> fc1_bias, torch::Tensor fc2_weight, "
            "c10::optional<torch::Tensor> fc2_bias, double eps, int64_t outputs_per_block);\n"
        ),
        cuda_sources=cuda_source,
        functions=["one_token_mlp_residual"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return _NATIVE_DECODER_LAYER


def preload_native_decoder_layer():
    return _load_native_decoder_layer()


def native_one_token_mlp_residual(
        input: torch.Tensor,
        norm_weight: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_bias: torch.Tensor | None,
        fc2_weight: torch.Tensor,
        fc2_bias: torch.Tensor | None,
        *,
        eps: float,
        outputs_per_block: int,
) -> torch.Tensor:
    extension = _load_native_decoder_layer()
    return extension.one_token_mlp_residual(
        input,
        norm_weight,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
        float(eps),
        int(outputs_per_block),
    )
