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

template<typename scalar_t, int BLOCK_SIZE>
__global__ void rmsnorm_one_token_kernel(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ weight,
        scalar_t* __restrict__ normalized,
        int hidden_dim,
        float eps) {
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    for (int idx = tid; idx < hidden_dim; idx += BLOCK_SIZE) {
        float value = static_cast<float>(input[idx]);
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
        normalized[idx] = static_cast<scalar_t>(
            static_cast<float>(input[idx])
            * inv_rms
            * static_cast<float>(weight[idx]));
    }
}

__device__ __forceinline__ float exact_gelu(float value) {
    return 0.5f * value * (1.0f + erff(value * 0.7071067811865475244f));
}

template<typename scalar_t, int OUTPUTS_PER_BLOCK>
__global__ void rmsnorm_linear_warp_group_kernel(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ norm_weight,
        const scalar_t* __restrict__ linear_weight,
        const scalar_t* __restrict__ linear_bias,
        scalar_t* __restrict__ output,
        int hidden_dim,
        int output_dim,
        int has_bias,
        float eps) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int out_idx = blockIdx.x * OUTPUTS_PER_BLOCK + warp_id;
    if (warp_id >= OUTPUTS_PER_BLOCK || out_idx >= output_dim) {
        return;
    }

    const scalar_t* w = linear_weight + out_idx * hidden_dim;
    float local_sq = 0.0f;
    float local_dot = 0.0f;
    for (int idx = lane; idx < hidden_dim; idx += 32) {
        float x = static_cast<float>(input[idx]);
        local_sq += x * x;
        local_dot += x
            * static_cast<float>(norm_weight[idx])
            * static_cast<float>(w[idx]);
    }

    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sq += __shfl_down_sync(mask, local_sq, offset);
        local_dot += __shfl_down_sync(mask, local_dot, offset);
    }
    if (lane == 0) {
        float inv_rms = rsqrtf(local_sq / static_cast<float>(hidden_dim) + eps);
        float value = local_dot * inv_rms;
        if (has_bias) {
            value += static_cast<float>(linear_bias[out_idx]);
        }
        output[out_idx] = static_cast<scalar_t>(value);
    }
}

template<typename scalar_t, int OUTPUTS_PER_BLOCK>
__global__ void linear_rect_warp_group_kernel(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ weight,
        const scalar_t* __restrict__ bias,
        scalar_t* __restrict__ output,
        int hidden_dim,
        int output_dim,
        int has_bias) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int out_idx = blockIdx.x * OUTPUTS_PER_BLOCK + warp_id;
    if (warp_id >= OUTPUTS_PER_BLOCK || out_idx >= output_dim) {
        return;
    }

    const scalar_t* w = weight + out_idx * hidden_dim;
    float sum = 0.0f;
    for (int idx = lane; idx < hidden_dim; idx += 32) {
        sum += static_cast<float>(input[idx]) * static_cast<float>(w[idx]);
    }

    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    if (lane == 0) {
        if (has_bias) {
            sum += static_cast<float>(bias[out_idx]);
        }
        output[out_idx] = static_cast<scalar_t>(sum);
    }
}

template<typename scalar_t, int OUTPUTS_PER_BLOCK>
__global__ void linear_residual_warp_group_kernel(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ residual,
        const scalar_t* __restrict__ weight,
        const scalar_t* __restrict__ bias,
        scalar_t* __restrict__ output,
        int hidden_dim,
        int has_bias) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int out_idx = blockIdx.x * OUTPUTS_PER_BLOCK + warp_id;
    if (warp_id >= OUTPUTS_PER_BLOCK || out_idx >= hidden_dim) {
        return;
    }

    const scalar_t* w = weight + out_idx * hidden_dim;
    float sum = 0.0f;
    for (int idx = lane; idx < hidden_dim; idx += 32) {
        sum += static_cast<float>(input[idx]) * static_cast<float>(w[idx]);
    }

    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    if (lane == 0) {
        if (has_bias) {
            sum += static_cast<float>(bias[out_idx]);
        }
        output[out_idx] = static_cast<scalar_t>(
            static_cast<float>(residual[out_idx]) + sum);
    }
}

template<typename scalar_t, int OUTPUTS_PER_BLOCK>
__global__ void fc1_gelu_warp_group_kernel(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ weight,
        const scalar_t* __restrict__ bias,
        scalar_t* __restrict__ output,
        int hidden_dim,
        int intermediate_dim,
        int has_bias) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int out_idx = blockIdx.x * OUTPUTS_PER_BLOCK + warp_id;
    if (warp_id >= OUTPUTS_PER_BLOCK || out_idx >= intermediate_dim) {
        return;
    }

    const scalar_t* w = weight + out_idx * hidden_dim;
    float sum = 0.0f;
    for (int idx = lane; idx < hidden_dim; idx += 32) {
        sum += static_cast<float>(input[idx]) * static_cast<float>(w[idx]);
    }

    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    if (lane == 0) {
        if (has_bias) {
            sum += static_cast<float>(bias[out_idx]);
        }
        output[out_idx] = static_cast<scalar_t>(exact_gelu(sum));
    }
}

template<typename scalar_t, int OUTPUTS_PER_BLOCK>
__global__ void fc2_residual_warp_group_kernel(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ weight,
        const scalar_t* __restrict__ bias,
        const scalar_t* __restrict__ residual,
        scalar_t* __restrict__ output,
        int hidden_dim,
        int intermediate_dim,
        int has_bias) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int out_idx = blockIdx.x * OUTPUTS_PER_BLOCK + warp_id;
    if (warp_id >= OUTPUTS_PER_BLOCK || out_idx >= hidden_dim) {
        return;
    }

    const scalar_t* w = weight + out_idx * intermediate_dim;
    float sum = 0.0f;
    for (int idx = lane; idx < intermediate_dim; idx += 32) {
        sum += static_cast<float>(input[idx]) * static_cast<float>(w[idx]);
    }

    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    if (lane == 0) {
        if (has_bias) {
            sum += static_cast<float>(bias[out_idx]);
        }
        output[out_idx] = static_cast<scalar_t>(
            static_cast<float>(residual[out_idx]) + sum);
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
    TORCH_CHECK(norm_weight.device() == input.device(), "norm_weight/input device mismatch");
    TORCH_CHECK(fc1_weight.device() == input.device(), "fc1_weight/input device mismatch");
    TORCH_CHECK(fc2_weight.device() == input.device(), "fc2_weight/input device mismatch");
    TORCH_CHECK(
            input.scalar_type() == torch::kFloat32
            || input.scalar_type() == torch::kFloat16,
            "input must be fp32 or fp16");
    TORCH_CHECK(norm_weight.scalar_type() == input.scalar_type(),
            "norm_weight dtype must match input");
    TORCH_CHECK(fc1_weight.scalar_type() == input.scalar_type(),
            "fc1_weight dtype must match input");
    TORCH_CHECK(fc2_weight.scalar_type() == input.scalar_type(),
            "fc2_weight dtype must match input");
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
    TORCH_CHECK(outputs_per_block == 2 || outputs_per_block == 4 || outputs_per_block == 8,
            "unsupported outputs_per_block; expected 2, 4, or 8");

    c10::cuda::CUDAGuard device_guard(input.device());

    auto input_contiguous = input.contiguous();
    auto norm_weight_contiguous = norm_weight.contiguous();
    auto fc1_weight_contiguous = fc1_weight.contiguous();
    auto fc2_weight_contiguous = fc2_weight.contiguous();

    torch::Tensor fc1_bias_contiguous;
    const void* fc1_bias_ptr = nullptr;
    int has_fc1_bias = 0;
    if (fc1_bias.has_value()) {
        fc1_bias_contiguous = fc1_bias.value().contiguous();
        TORCH_CHECK(fc1_bias_contiguous.is_cuda(), "fc1_bias must be CUDA");
        TORCH_CHECK(fc1_bias_contiguous.device() == input.device(), "fc1_bias/input device mismatch");
        TORCH_CHECK(fc1_bias_contiguous.scalar_type() == input.scalar_type(),
                "fc1_bias dtype must match input");
        TORCH_CHECK(fc1_bias_contiguous.numel() == intermediate_dim, "fc1_bias/intermediate_dim mismatch");
        fc1_bias_ptr = fc1_bias_contiguous.data_ptr();
        has_fc1_bias = 1;
    }

    torch::Tensor fc2_bias_contiguous;
    const void* fc2_bias_ptr = nullptr;
    int has_fc2_bias = 0;
    if (fc2_bias.has_value()) {
        fc2_bias_contiguous = fc2_bias.value().contiguous();
        TORCH_CHECK(fc2_bias_contiguous.is_cuda(), "fc2_bias must be CUDA");
        TORCH_CHECK(fc2_bias_contiguous.device() == input.device(), "fc2_bias/input device mismatch");
        TORCH_CHECK(fc2_bias_contiguous.scalar_type() == input.scalar_type(),
                "fc2_bias dtype must match input");
        TORCH_CHECK(fc2_bias_contiguous.numel() == hidden_dim, "fc2_bias/hidden_dim mismatch");
        fc2_bias_ptr = fc2_bias_contiguous.data_ptr();
        has_fc2_bias = 1;
    }

    auto normalized = torch::empty({hidden_dim}, input.options());
    auto activated = torch::empty({intermediate_dim}, input.options());
    auto output = torch::empty({1, 1, hidden_dim}, input.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(), "one_token_mlp_residual", [&] {
        const auto* fc1_bias = static_cast<const scalar_t*>(fc1_bias_ptr);
        const auto* fc2_bias = static_cast<const scalar_t*>(fc2_bias_ptr);
        rmsnorm_one_token_kernel<scalar_t, 256><<<1, 256, 0, stream>>>(
            input_contiguous.data_ptr<scalar_t>(),
            norm_weight_contiguous.data_ptr<scalar_t>(),
            normalized.data_ptr<scalar_t>(),
            hidden_dim,
            static_cast<float>(eps));
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        if (outputs_per_block == 2) {
            fc1_gelu_warp_group_kernel<scalar_t, 2><<<(intermediate_dim + 1) / 2, 64, 0, stream>>>(
                normalized.data_ptr<scalar_t>(),
                fc1_weight_contiguous.data_ptr<scalar_t>(),
                fc1_bias,
                activated.data_ptr<scalar_t>(),
                hidden_dim,
                intermediate_dim,
                has_fc1_bias);
        } else if (outputs_per_block == 4) {
            fc1_gelu_warp_group_kernel<scalar_t, 4><<<(intermediate_dim + 3) / 4, 128, 0, stream>>>(
                normalized.data_ptr<scalar_t>(),
                fc1_weight_contiguous.data_ptr<scalar_t>(),
                fc1_bias,
                activated.data_ptr<scalar_t>(),
                hidden_dim,
                intermediate_dim,
                has_fc1_bias);
        } else {
            fc1_gelu_warp_group_kernel<scalar_t, 8><<<(intermediate_dim + 7) / 8, 256, 0, stream>>>(
                normalized.data_ptr<scalar_t>(),
                fc1_weight_contiguous.data_ptr<scalar_t>(),
                fc1_bias,
                activated.data_ptr<scalar_t>(),
                hidden_dim,
                intermediate_dim,
                has_fc1_bias);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        if (outputs_per_block == 2) {
            fc2_residual_warp_group_kernel<scalar_t, 2><<<(hidden_dim + 1) / 2, 64, 0, stream>>>(
                activated.data_ptr<scalar_t>(),
                fc2_weight_contiguous.data_ptr<scalar_t>(),
                fc2_bias,
                input_contiguous.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                hidden_dim,
                intermediate_dim,
                has_fc2_bias);
        } else if (outputs_per_block == 4) {
            fc2_residual_warp_group_kernel<scalar_t, 4><<<(hidden_dim + 3) / 4, 128, 0, stream>>>(
                activated.data_ptr<scalar_t>(),
                fc2_weight_contiguous.data_ptr<scalar_t>(),
                fc2_bias,
                input_contiguous.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                hidden_dim,
                intermediate_dim,
                has_fc2_bias);
        } else {
            fc2_residual_warp_group_kernel<scalar_t, 8><<<(hidden_dim + 7) / 8, 256, 0, stream>>>(
                activated.data_ptr<scalar_t>(),
                fc2_weight_contiguous.data_ptr<scalar_t>(),
                fc2_bias,
                input_contiguous.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                hidden_dim,
                intermediate_dim,
                has_fc2_bias);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });

    return output;
}

torch::Tensor one_token_rmsnorm_linear(
        torch::Tensor input,
        torch::Tensor norm_weight,
        torch::Tensor linear_weight,
        c10::optional<torch::Tensor> linear_bias,
        double eps,
        int64_t outputs_per_block) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(norm_weight.is_cuda(), "norm_weight must be a CUDA tensor");
    TORCH_CHECK(linear_weight.is_cuda(), "linear_weight must be a CUDA tensor");
    TORCH_CHECK(norm_weight.device() == input.device(), "norm_weight/input device mismatch");
    TORCH_CHECK(linear_weight.device() == input.device(), "linear_weight/input device mismatch");
    TORCH_CHECK(
            input.scalar_type() == torch::kFloat32
            || input.scalar_type() == torch::kFloat16,
            "input must be fp32 or fp16");
    TORCH_CHECK(norm_weight.scalar_type() == input.scalar_type(),
            "norm_weight dtype must match input");
    TORCH_CHECK(linear_weight.scalar_type() == input.scalar_type(),
            "linear_weight dtype must match input");
    TORCH_CHECK(input.dim() == 3 && input.size(0) == 1 && input.size(1) == 1,
            "input must have shape [1, 1, hidden_dim]");
    TORCH_CHECK(norm_weight.dim() == 1, "norm_weight must have shape [hidden_dim]");
    TORCH_CHECK(linear_weight.dim() == 2, "linear_weight must have shape [output_dim, hidden_dim]");

    const int hidden_dim = static_cast<int>(input.size(2));
    const int output_dim = static_cast<int>(linear_weight.size(0));
    TORCH_CHECK(norm_weight.numel() == hidden_dim, "norm_weight/hidden_dim mismatch");
    TORCH_CHECK(linear_weight.size(1) == hidden_dim, "linear_weight hidden_dim mismatch");
    TORCH_CHECK(outputs_per_block == 2 || outputs_per_block == 4 || outputs_per_block == 8,
            "unsupported outputs_per_block; expected 2, 4, or 8");

    c10::cuda::CUDAGuard device_guard(input.device());

    auto input_contiguous = input.contiguous();
    auto norm_weight_contiguous = norm_weight.contiguous();
    auto linear_weight_contiguous = linear_weight.contiguous();

    torch::Tensor linear_bias_contiguous;
    const void* linear_bias_ptr = nullptr;
    int has_linear_bias = 0;
    if (linear_bias.has_value()) {
        linear_bias_contiguous = linear_bias.value().contiguous();
        TORCH_CHECK(linear_bias_contiguous.is_cuda(), "linear_bias must be CUDA");
        TORCH_CHECK(linear_bias_contiguous.device() == input.device(), "linear_bias/input device mismatch");
        TORCH_CHECK(linear_bias_contiguous.scalar_type() == input.scalar_type(),
                "linear_bias dtype must match input");
        TORCH_CHECK(linear_bias_contiguous.numel() == output_dim, "linear_bias/output_dim mismatch");
        linear_bias_ptr = linear_bias_contiguous.data_ptr();
        has_linear_bias = 1;
    }

    auto output = torch::empty({1, 1, output_dim}, input.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(), "one_token_rmsnorm_linear", [&] {
        const auto* bias = static_cast<const scalar_t*>(linear_bias_ptr);
        if (outputs_per_block == 2) {
            rmsnorm_linear_warp_group_kernel<scalar_t, 2><<<(output_dim + 1) / 2, 64, 0, stream>>>(
                input_contiguous.data_ptr<scalar_t>(),
                norm_weight_contiguous.data_ptr<scalar_t>(),
                linear_weight_contiguous.data_ptr<scalar_t>(),
                bias,
                output.data_ptr<scalar_t>(),
                hidden_dim,
                output_dim,
                has_linear_bias,
                static_cast<float>(eps));
        } else if (outputs_per_block == 4) {
            rmsnorm_linear_warp_group_kernel<scalar_t, 4><<<(output_dim + 3) / 4, 128, 0, stream>>>(
                input_contiguous.data_ptr<scalar_t>(),
                norm_weight_contiguous.data_ptr<scalar_t>(),
                linear_weight_contiguous.data_ptr<scalar_t>(),
                bias,
                output.data_ptr<scalar_t>(),
                hidden_dim,
                output_dim,
                has_linear_bias,
                static_cast<float>(eps));
        } else {
            rmsnorm_linear_warp_group_kernel<scalar_t, 8><<<(output_dim + 7) / 8, 256, 0, stream>>>(
                input_contiguous.data_ptr<scalar_t>(),
                norm_weight_contiguous.data_ptr<scalar_t>(),
                linear_weight_contiguous.data_ptr<scalar_t>(),
                bias,
                output.data_ptr<scalar_t>(),
                hidden_dim,
                output_dim,
                has_linear_bias,
                static_cast<float>(eps));
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor one_token_linear_rect(
        torch::Tensor input,
        torch::Tensor weight,
        c10::optional<torch::Tensor> bias,
        int64_t outputs_per_block) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(weight.device() == input.device(), "weight/input device mismatch");
    TORCH_CHECK(
            input.scalar_type() == torch::kFloat32
            || input.scalar_type() == torch::kFloat16,
            "input must be fp32 or fp16");
    TORCH_CHECK(weight.scalar_type() == input.scalar_type(),
            "weight dtype must match input");
    TORCH_CHECK(input.dim() == 3 && input.size(0) == 1 && input.size(1) == 1,
            "input must have shape [1, 1, hidden_dim]");
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [output_dim, hidden_dim]");

    const int hidden_dim = static_cast<int>(input.size(2));
    const int output_dim = static_cast<int>(weight.size(0));
    TORCH_CHECK(weight.size(1) == hidden_dim, "weight hidden_dim mismatch");
    TORCH_CHECK(outputs_per_block == 2 || outputs_per_block == 4 || outputs_per_block == 8,
            "unsupported outputs_per_block; expected 2, 4, or 8");

    c10::cuda::CUDAGuard device_guard(input.device());

    auto input_contiguous = input.contiguous();
    auto weight_contiguous = weight.contiguous();

    torch::Tensor bias_contiguous;
    const void* bias_ptr = nullptr;
    int has_bias = 0;
    if (bias.has_value()) {
        bias_contiguous = bias.value().contiguous();
        TORCH_CHECK(bias_contiguous.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(bias_contiguous.device() == input.device(), "bias/input device mismatch");
        TORCH_CHECK(bias_contiguous.scalar_type() == input.scalar_type(),
                "bias dtype must match input");
        TORCH_CHECK(bias_contiguous.numel() == output_dim, "bias/output_dim mismatch");
        bias_ptr = bias_contiguous.data_ptr();
        has_bias = 1;
    }

    auto output = torch::empty({1, 1, output_dim}, input.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(), "one_token_linear_rect", [&] {
        const auto* bias = static_cast<const scalar_t*>(bias_ptr);
        if (outputs_per_block == 2) {
            linear_rect_warp_group_kernel<scalar_t, 2><<<(output_dim + 1) / 2, 64, 0, stream>>>(
                input_contiguous.data_ptr<scalar_t>(),
                weight_contiguous.data_ptr<scalar_t>(),
                bias,
                output.data_ptr<scalar_t>(),
                hidden_dim,
                output_dim,
                has_bias);
        } else if (outputs_per_block == 4) {
            linear_rect_warp_group_kernel<scalar_t, 4><<<(output_dim + 3) / 4, 128, 0, stream>>>(
                input_contiguous.data_ptr<scalar_t>(),
                weight_contiguous.data_ptr<scalar_t>(),
                bias,
                output.data_ptr<scalar_t>(),
                hidden_dim,
                output_dim,
                has_bias);
        } else {
            linear_rect_warp_group_kernel<scalar_t, 8><<<(output_dim + 7) / 8, 256, 0, stream>>>(
                input_contiguous.data_ptr<scalar_t>(),
                weight_contiguous.data_ptr<scalar_t>(),
                bias,
                output.data_ptr<scalar_t>(),
                hidden_dim,
                output_dim,
                has_bias);
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor one_token_linear_residual(
        torch::Tensor input,
        torch::Tensor residual,
        torch::Tensor weight,
        c10::optional<torch::Tensor> bias,
        int64_t outputs_per_block) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(residual.device() == input.device(), "residual/input device mismatch");
    TORCH_CHECK(weight.device() == input.device(), "weight/input device mismatch");
    TORCH_CHECK(
            input.scalar_type() == torch::kFloat32
            || input.scalar_type() == torch::kFloat16,
            "input must be fp32 or fp16");
    TORCH_CHECK(residual.scalar_type() == input.scalar_type(),
            "residual dtype must match input");
    TORCH_CHECK(weight.scalar_type() == input.scalar_type(),
            "weight dtype must match input");
    TORCH_CHECK(residual.dim() == 3 && residual.size(0) == 1 && residual.size(1) == 1,
            "residual must have shape [1, 1, hidden_dim]");
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [hidden_dim, hidden_dim]");

    const int hidden_dim = static_cast<int>(residual.size(2));
    TORCH_CHECK(input.numel() == hidden_dim, "input hidden_dim mismatch");
    TORCH_CHECK(weight.size(0) == hidden_dim && weight.size(1) == hidden_dim, "weight hidden_dim mismatch");
    TORCH_CHECK(outputs_per_block == 2 || outputs_per_block == 4 || outputs_per_block == 8,
            "unsupported outputs_per_block; expected 2, 4, or 8");

    c10::cuda::CUDAGuard device_guard(input.device());

    auto input_contiguous = input.contiguous();
    auto residual_contiguous = residual.contiguous();
    auto weight_contiguous = weight.contiguous();

    torch::Tensor bias_contiguous;
    const void* bias_ptr = nullptr;
    int has_bias = 0;
    if (bias.has_value()) {
        bias_contiguous = bias.value().contiguous();
        TORCH_CHECK(bias_contiguous.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(bias_contiguous.device() == input.device(), "bias/input device mismatch");
        TORCH_CHECK(bias_contiguous.scalar_type() == input.scalar_type(),
                "bias dtype must match input");
        TORCH_CHECK(bias_contiguous.numel() == hidden_dim, "bias/hidden_dim mismatch");
        bias_ptr = bias_contiguous.data_ptr();
        has_bias = 1;
    }

    auto output = torch::empty({1, 1, hidden_dim}, residual.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(), "one_token_linear_residual", [&] {
        const auto* bias = static_cast<const scalar_t*>(bias_ptr);
        if (outputs_per_block == 2) {
            linear_residual_warp_group_kernel<scalar_t, 2><<<(hidden_dim + 1) / 2, 64, 0, stream>>>(
                input_contiguous.data_ptr<scalar_t>(),
                residual_contiguous.data_ptr<scalar_t>(),
                weight_contiguous.data_ptr<scalar_t>(),
                bias,
                output.data_ptr<scalar_t>(),
                hidden_dim,
                has_bias);
        } else if (outputs_per_block == 4) {
            linear_residual_warp_group_kernel<scalar_t, 4><<<(hidden_dim + 3) / 4, 128, 0, stream>>>(
                input_contiguous.data_ptr<scalar_t>(),
                residual_contiguous.data_ptr<scalar_t>(),
                weight_contiguous.data_ptr<scalar_t>(),
                bias,
                output.data_ptr<scalar_t>(),
                hidden_dim,
                has_bias);
        } else {
            linear_residual_warp_group_kernel<scalar_t, 8><<<(hidden_dim + 7) / 8, 256, 0, stream>>>(
                input_contiguous.data_ptr<scalar_t>(),
                residual_contiguous.data_ptr<scalar_t>(),
                weight_contiguous.data_ptr<scalar_t>(),
                bias,
                output.data_ptr<scalar_t>(),
                hidden_dim,
                has_bias);
        }
    });
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
            "torch::Tensor one_token_rmsnorm_linear("
            "torch::Tensor input, torch::Tensor norm_weight, torch::Tensor linear_weight, "
            "c10::optional<torch::Tensor> linear_bias, double eps, int64_t outputs_per_block);\n"
            "torch::Tensor one_token_linear_rect("
            "torch::Tensor input, torch::Tensor weight, "
            "c10::optional<torch::Tensor> bias, int64_t outputs_per_block);\n"
            "torch::Tensor one_token_linear_residual("
            "torch::Tensor input, torch::Tensor residual, torch::Tensor weight, "
            "c10::optional<torch::Tensor> bias, int64_t outputs_per_block);\n"
        ),
        cuda_sources=cuda_source,
        functions=[
            "one_token_mlp_residual",
            "one_token_rmsnorm_linear",
            "one_token_linear_rect",
            "one_token_linear_residual",
        ],
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


def native_one_token_rmsnorm_linear(
        input: torch.Tensor,
        norm_weight: torch.Tensor,
        linear_weight: torch.Tensor,
        linear_bias: torch.Tensor | None,
        *,
        eps: float,
        outputs_per_block: int,
) -> torch.Tensor:
    extension = _load_native_decoder_layer()
    return extension.one_token_rmsnorm_linear(
        input,
        norm_weight,
        linear_weight,
        linear_bias,
        float(eps),
        int(outputs_per_block),
    )


def native_one_token_linear_rect(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        outputs_per_block: int,
) -> torch.Tensor:
    extension = _load_native_decoder_layer()
    return extension.one_token_linear_rect(
        input,
        weight,
        bias,
        int(outputs_per_block),
    )


def native_one_token_linear_residual(
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        outputs_per_block: int,
) -> torch.Tensor:
    extension = _load_native_decoder_layer()
    return extension.one_token_linear_residual(
        input,
        residual,
        weight,
        bias,
        int(outputs_per_block),
    )
