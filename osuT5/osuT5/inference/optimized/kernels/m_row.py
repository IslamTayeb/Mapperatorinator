"""§54 Stage A: m≤8-row extensions of owned warp-group decoder kernels.

Turbo verify path only. Keeps SDPA as the attention core; Stage B owns
fused m-row split-KV attention. Does not alter bit-exact Q=1 optimized
presets — callers must arm these from turbo teacher verify.
"""
from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline

MAX_M = 8
_NATIVE_M_ROW = None

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef MROW_MAX_M
#define MROW_MAX_M 8
#endif

__device__ __forceinline__ float exact_gelu(float value) {
    return 0.5f * value * (1.0f + erff(value * 0.7071067811865475244f));
}

template<typename scalar_t, int BLOCK_SIZE>
__global__ void rmsnorm_m_row_kernel(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ weight,
        scalar_t* __restrict__ normalized,
        int m,
        int hidden_dim,
        float eps) {
    int row = blockIdx.x;
    if (row >= m) return;
    int tid = threadIdx.x;
    const scalar_t* in = input + row * hidden_dim;
    scalar_t* out = normalized + row * hidden_dim;
    float local_sum = 0.0f;
    for (int idx = tid; idx < hidden_dim; idx += BLOCK_SIZE) {
        float value = static_cast<float>(in[idx]);
        local_sum += value * value;
    }
    __shared__ float scratch[BLOCK_SIZE];
    scratch[tid] = local_sum;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    float inv_rms = rsqrtf(scratch[0] / static_cast<float>(hidden_dim) + eps);
    for (int idx = tid; idx < hidden_dim; idx += BLOCK_SIZE) {
        out[idx] = static_cast<scalar_t>(
            static_cast<float>(in[idx]) * inv_rms * static_cast<float>(weight[idx]));
    }
}

template<typename scalar_t, int OUTPUTS_PER_BLOCK>
__global__ void fc1_gelu_m_row_warp_group_kernel(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ weight,
        const scalar_t* __restrict__ bias,
        scalar_t* __restrict__ output,
        int m,
        int hidden_dim,
        int intermediate_dim,
        int has_bias) {
    extern __shared__ unsigned char smem_raw[];
    scalar_t* input_smem = reinterpret_cast<scalar_t*>(smem_raw);
    int total = m * hidden_dim;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        input_smem[idx] = input[idx];
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int out_idx = blockIdx.x * OUTPUTS_PER_BLOCK + warp_id;
    if (warp_id >= OUTPUTS_PER_BLOCK || out_idx >= intermediate_dim) return;

    const scalar_t* w = weight + out_idx * hidden_dim;
    float sums[MROW_MAX_M];
#pragma unroll
    for (int r = 0; r < MROW_MAX_M; ++r) sums[r] = 0.0f;

    for (int idx = lane; idx < hidden_dim; idx += 32) {
        float wv = static_cast<float>(w[idx]);
#pragma unroll
        for (int r = 0; r < MROW_MAX_M; ++r) {
            if (r < m) {
                sums[r] += static_cast<float>(input_smem[r * hidden_dim + idx]) * wv;
            }
        }
    }
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int r = 0; r < MROW_MAX_M; ++r) {
            if (r < m) sums[r] += __shfl_down_sync(mask, sums[r], offset);
        }
    }
    if (lane == 0) {
        float b = has_bias ? static_cast<float>(bias[out_idx]) : 0.0f;
        for (int r = 0; r < m; ++r) {
            output[r * intermediate_dim + out_idx] =
                static_cast<scalar_t>(exact_gelu(sums[r] + b));
        }
    }
}

template<typename scalar_t, int OUTPUTS_PER_BLOCK>
__global__ void fc2_residual_m_row_warp_group_kernel(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ weight,
        const scalar_t* __restrict__ bias,
        const scalar_t* __restrict__ residual,
        scalar_t* __restrict__ output,
        int m,
        int hidden_dim,
        int intermediate_dim,
        int has_bias) {
    extern __shared__ unsigned char smem_raw[];
    scalar_t* input_smem = reinterpret_cast<scalar_t*>(smem_raw);
    int total = m * intermediate_dim;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        input_smem[idx] = input[idx];
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int out_idx = blockIdx.x * OUTPUTS_PER_BLOCK + warp_id;
    if (warp_id >= OUTPUTS_PER_BLOCK || out_idx >= hidden_dim) return;

    const scalar_t* w = weight + out_idx * intermediate_dim;
    float sums[MROW_MAX_M];
#pragma unroll
    for (int r = 0; r < MROW_MAX_M; ++r) sums[r] = 0.0f;

    for (int idx = lane; idx < intermediate_dim; idx += 32) {
        float wv = static_cast<float>(w[idx]);
#pragma unroll
        for (int r = 0; r < MROW_MAX_M; ++r) {
            if (r < m) {
                sums[r] += static_cast<float>(input_smem[r * intermediate_dim + idx]) * wv;
            }
        }
    }
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int r = 0; r < MROW_MAX_M; ++r) {
            if (r < m) sums[r] += __shfl_down_sync(mask, sums[r], offset);
        }
    }
    if (lane == 0) {
        float b = has_bias ? static_cast<float>(bias[out_idx]) : 0.0f;
        for (int r = 0; r < m; ++r) {
            float value = sums[r] + b + static_cast<float>(residual[r * hidden_dim + out_idx]);
            output[r * hidden_dim + out_idx] = static_cast<scalar_t>(value);
        }
    }
}

template<typename scalar_t, int OUTPUTS_PER_BLOCK>
__global__ void rmsnorm_linear_m_row_warp_group_kernel(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ norm_weight,
        const scalar_t* __restrict__ linear_weight,
        const scalar_t* __restrict__ linear_bias,
        scalar_t* __restrict__ output,
        int m,
        int hidden_dim,
        int output_dim,
        int has_bias,
        float eps) {
    extern __shared__ unsigned char smem_raw[];
    scalar_t* input_smem = reinterpret_cast<scalar_t*>(smem_raw);
    int total = m * hidden_dim;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        input_smem[idx] = input[idx];
    }
    __syncthreads();

    // Per-row RMS in shared (m floats).
    float* row_inv = reinterpret_cast<float*>(input_smem + total);
    if (threadIdx.x < m) {
        float sq = 0.0f;
        const scalar_t* row = input_smem + threadIdx.x * hidden_dim;
        for (int idx = 0; idx < hidden_dim; ++idx) {
            float x = static_cast<float>(row[idx]);
            sq += x * x;
        }
        row_inv[threadIdx.x] = rsqrtf(sq / static_cast<float>(hidden_dim) + eps);
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int out_idx = blockIdx.x * OUTPUTS_PER_BLOCK + warp_id;
    if (warp_id >= OUTPUTS_PER_BLOCK || out_idx >= output_dim) return;

    const scalar_t* w = linear_weight + out_idx * hidden_dim;
    float sums[MROW_MAX_M];
#pragma unroll
    for (int r = 0; r < MROW_MAX_M; ++r) sums[r] = 0.0f;

    for (int idx = lane; idx < hidden_dim; idx += 32) {
        float wv = static_cast<float>(w[idx]) * static_cast<float>(norm_weight[idx]);
#pragma unroll
        for (int r = 0; r < MROW_MAX_M; ++r) {
            if (r < m) {
                sums[r] += static_cast<float>(input_smem[r * hidden_dim + idx]) * wv;
            }
        }
    }
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int r = 0; r < MROW_MAX_M; ++r) {
            if (r < m) sums[r] += __shfl_down_sync(mask, sums[r], offset);
        }
    }
    if (lane == 0) {
        float b = has_bias ? static_cast<float>(linear_bias[out_idx]) : 0.0f;
        for (int r = 0; r < m; ++r) {
            output[r * output_dim + out_idx] =
                static_cast<scalar_t>(sums[r] * row_inv[r] + b);
        }
    }
}

template<typename scalar_t, int OUTPUTS_PER_BLOCK>
__global__ void linear_residual_m_row_warp_group_kernel(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ residual,
        const scalar_t* __restrict__ weight,
        const scalar_t* __restrict__ bias,
        scalar_t* __restrict__ output,
        int m,
        int in_dim,
        int hidden_dim,
        int has_bias) {
    extern __shared__ unsigned char smem_raw[];
    scalar_t* input_smem = reinterpret_cast<scalar_t*>(smem_raw);
    int total = m * in_dim;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        input_smem[idx] = input[idx];
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int out_idx = blockIdx.x * OUTPUTS_PER_BLOCK + warp_id;
    if (warp_id >= OUTPUTS_PER_BLOCK || out_idx >= hidden_dim) return;

    const scalar_t* w = weight + out_idx * in_dim;
    float sums[MROW_MAX_M];
#pragma unroll
    for (int r = 0; r < MROW_MAX_M; ++r) sums[r] = 0.0f;

    for (int idx = lane; idx < in_dim; idx += 32) {
        float wv = static_cast<float>(w[idx]);
#pragma unroll
        for (int r = 0; r < MROW_MAX_M; ++r) {
            if (r < m) {
                sums[r] += static_cast<float>(input_smem[r * in_dim + idx]) * wv;
            }
        }
    }
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int r = 0; r < MROW_MAX_M; ++r) {
            if (r < m) sums[r] += __shfl_down_sync(mask, sums[r], offset);
        }
    }
    if (lane == 0) {
        float b = has_bias ? static_cast<float>(bias[out_idx]) : 0.0f;
        for (int r = 0; r < m; ++r) {
            float value = sums[r] + b + static_cast<float>(residual[r * hidden_dim + out_idx]);
            output[r * hidden_dim + out_idx] = static_cast<scalar_t>(value);
        }
    }
}

template<typename scalar_t>
__global__ void m_row_rope_cache_write_kernel(
        const scalar_t* __restrict__ qkv,
        scalar_t* __restrict__ cache_keys,
        scalar_t* __restrict__ cache_values,
        const scalar_t* __restrict__ cos,
        const scalar_t* __restrict__ sin,
        const int64_t* __restrict__ cache_position,
        scalar_t* __restrict__ query_out,
        int m,
        int heads,
        int head_dim,
        int max_cache_len,
        long long qkv_token_stride,
        long long qkv_qkv_stride,
        long long qkv_head_stride,
        long long qkv_dim_stride,
        long long cache_head_stride,
        long long cache_len_stride,
        long long cache_dim_stride,
        long long cos_token_stride,
        long long cos_dim_stride,
        long long sin_token_stride,
        long long sin_dim_stride,
        long long q_out_head_stride,
        long long q_out_token_stride,
        long long q_out_dim_stride) {
    int head = blockIdx.x;
    int row = blockIdx.y;
    int tid = threadIdx.x;
    if (head >= heads || row >= m) return;

    int write_pos = static_cast<int>(cache_position[row]);
    if (write_pos < 0 || write_pos >= max_cache_len) {
        asm("trap;");
        return;
    }
    const int half_dim = head_dim / 2;
    const long long qkv_base =
        row * qkv_token_stride + head * qkv_head_stride;
    const long long cos_base = row * cos_token_stride;
    const long long sin_base = row * sin_token_stride;

    for (int dim = tid; dim < head_dim; dim += blockDim.x) {
        const int rotated_dim = dim < half_dim ? dim + half_dim : dim - half_dim;
        const float sign = dim < half_dim ? -1.0f : 1.0f;
        const float cos_v = static_cast<float>(cos[cos_base + dim * cos_dim_stride]);
        const float sin_v = static_cast<float>(sin[sin_base + dim * sin_dim_stride]);
        const float q_raw = static_cast<float>(
            qkv[qkv_base + 0 * qkv_qkv_stride + dim * qkv_dim_stride]);
        const float q_rot = static_cast<float>(
            qkv[qkv_base + 0 * qkv_qkv_stride + rotated_dim * qkv_dim_stride]);
        const float k_raw = static_cast<float>(
            qkv[qkv_base + 1 * qkv_qkv_stride + dim * qkv_dim_stride]);
        const float k_rot = static_cast<float>(
            qkv[qkv_base + 1 * qkv_qkv_stride + rotated_dim * qkv_dim_stride]);
        const float v_raw = static_cast<float>(
            qkv[qkv_base + 2 * qkv_qkv_stride + dim * qkv_dim_stride]);
        const float q_value = q_raw * cos_v + sign * q_rot * sin_v;
        const float k_value = k_raw * cos_v + sign * k_rot * sin_v;
        query_out[
            head * q_out_head_stride
            + row * q_out_token_stride
            + dim * q_out_dim_stride
        ] = static_cast<scalar_t>(q_value);
        cache_keys[
            head * cache_head_stride
            + write_pos * cache_len_stride
            + dim * cache_dim_stride
        ] = static_cast<scalar_t>(k_value);
        cache_values[
            head * cache_head_stride
            + write_pos * cache_len_stride
            + dim * cache_dim_stride
        ] = static_cast<scalar_t>(v_raw);
    }
}

torch::Tensor m_row_mlp_residual(
        torch::Tensor input,
        torch::Tensor norm_weight,
        torch::Tensor fc1_weight,
        c10::optional<torch::Tensor> fc1_bias,
        torch::Tensor fc2_weight,
        c10::optional<torch::Tensor> fc2_bias,
        double eps,
        int64_t outputs_per_block) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(
            input.scalar_type() == torch::kFloat32
            || input.scalar_type() == torch::kFloat16,
            "input must be fp32 or fp16");
    TORCH_CHECK(input.dim() == 3 && input.size(0) == 1, "input must be [1,m,H]");
    const int m = static_cast<int>(input.size(1));
    const int hidden_dim = static_cast<int>(input.size(2));
    TORCH_CHECK(m >= 1 && m <= MROW_MAX_M, "m must be in [1,8]");
    TORCH_CHECK(outputs_per_block == 2 || outputs_per_block == 4 || outputs_per_block == 8,
            "unsupported outputs_per_block");
    const int intermediate_dim = static_cast<int>(fc1_weight.size(0));
    TORCH_CHECK(fc1_weight.size(1) == hidden_dim, "fc1_weight mismatch");
    TORCH_CHECK(fc2_weight.size(0) == hidden_dim && fc2_weight.size(1) == intermediate_dim,
            "fc2_weight mismatch");

    c10::cuda::CUDAGuard device_guard(input.device());
    auto input_c = input.contiguous();
    auto norm_c = norm_weight.contiguous();
    auto fc1_c = fc1_weight.contiguous();
    auto fc2_c = fc2_weight.contiguous();

    torch::Tensor fc1_bias_c; const void* fc1_bias_ptr = nullptr; int has_fc1 = 0;
    if (fc1_bias.has_value()) {
        fc1_bias_c = fc1_bias.value().contiguous();
        fc1_bias_ptr = fc1_bias_c.data_ptr();
        has_fc1 = 1;
    }
    torch::Tensor fc2_bias_c; const void* fc2_bias_ptr = nullptr; int has_fc2 = 0;
    if (fc2_bias.has_value()) {
        fc2_bias_c = fc2_bias.value().contiguous();
        fc2_bias_ptr = fc2_bias_c.data_ptr();
        has_fc2 = 1;
    }

    auto normalized = torch::empty({m, hidden_dim}, input.options());
    auto activated = torch::empty({m, intermediate_dim}, input.options());
    auto output = torch::empty({1, m, hidden_dim}, input.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "m_row_mlp_residual", [&] {
        const auto* b1 = static_cast<const scalar_t*>(fc1_bias_ptr);
        const auto* b2 = static_cast<const scalar_t*>(fc2_bias_ptr);
        rmsnorm_m_row_kernel<scalar_t, 256><<<m, 256, 0, stream>>>(
            input_c.data_ptr<scalar_t>(),
            norm_c.data_ptr<scalar_t>(),
            normalized.data_ptr<scalar_t>(),
            m, hidden_dim, static_cast<float>(eps));
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        size_t smem1 = static_cast<size_t>(m) * hidden_dim * sizeof(scalar_t);
        size_t smem2 = static_cast<size_t>(m) * intermediate_dim * sizeof(scalar_t);
        if (outputs_per_block == 2) {
            fc1_gelu_m_row_warp_group_kernel<scalar_t, 2>
                <<<(intermediate_dim + 1) / 2, 64, smem1, stream>>>(
                    normalized.data_ptr<scalar_t>(), fc1_c.data_ptr<scalar_t>(), b1,
                    activated.data_ptr<scalar_t>(), m, hidden_dim, intermediate_dim, has_fc1);
            fc2_residual_m_row_warp_group_kernel<scalar_t, 2>
                <<<(hidden_dim + 1) / 2, 64, smem2, stream>>>(
                    activated.data_ptr<scalar_t>(), fc2_c.data_ptr<scalar_t>(), b2,
                    input_c.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    m, hidden_dim, intermediate_dim, has_fc2);
        } else if (outputs_per_block == 4) {
            fc1_gelu_m_row_warp_group_kernel<scalar_t, 4>
                <<<(intermediate_dim + 3) / 4, 128, smem1, stream>>>(
                    normalized.data_ptr<scalar_t>(), fc1_c.data_ptr<scalar_t>(), b1,
                    activated.data_ptr<scalar_t>(), m, hidden_dim, intermediate_dim, has_fc1);
            fc2_residual_m_row_warp_group_kernel<scalar_t, 4>
                <<<(hidden_dim + 3) / 4, 128, smem2, stream>>>(
                    activated.data_ptr<scalar_t>(), fc2_c.data_ptr<scalar_t>(), b2,
                    input_c.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    m, hidden_dim, intermediate_dim, has_fc2);
        } else {
            fc1_gelu_m_row_warp_group_kernel<scalar_t, 8>
                <<<(intermediate_dim + 7) / 8, 256, smem1, stream>>>(
                    normalized.data_ptr<scalar_t>(), fc1_c.data_ptr<scalar_t>(), b1,
                    activated.data_ptr<scalar_t>(), m, hidden_dim, intermediate_dim, has_fc1);
            fc2_residual_m_row_warp_group_kernel<scalar_t, 8>
                <<<(hidden_dim + 7) / 8, 256, smem2, stream>>>(
                    activated.data_ptr<scalar_t>(), fc2_c.data_ptr<scalar_t>(), b2,
                    input_c.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    m, hidden_dim, intermediate_dim, has_fc2);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return output;
}

torch::Tensor m_row_rmsnorm_linear(
        torch::Tensor input,
        torch::Tensor norm_weight,
        torch::Tensor linear_weight,
        c10::optional<torch::Tensor> linear_bias,
        double eps,
        int64_t outputs_per_block) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dim() == 3 && input.size(0) == 1, "input must be [1,m,H]");
    const int m = static_cast<int>(input.size(1));
    const int hidden_dim = static_cast<int>(input.size(2));
    const int output_dim = static_cast<int>(linear_weight.size(0));
    TORCH_CHECK(m >= 1 && m <= MROW_MAX_M, "m must be in [1,8]");
    TORCH_CHECK(linear_weight.size(1) == hidden_dim, "linear_weight mismatch");
    TORCH_CHECK(outputs_per_block == 2 || outputs_per_block == 4 || outputs_per_block == 8,
            "unsupported outputs_per_block");

    c10::cuda::CUDAGuard device_guard(input.device());
    auto input_c = input.contiguous();
    auto norm_c = norm_weight.contiguous();
    auto w_c = linear_weight.contiguous();
    torch::Tensor bias_c; const void* bias_ptr = nullptr; int has_bias = 0;
    if (linear_bias.has_value()) {
        bias_c = linear_bias.value().contiguous();
        bias_ptr = bias_c.data_ptr();
        has_bias = 1;
    }
    auto output = torch::empty({1, m, output_dim}, input.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "m_row_rmsnorm_linear", [&] {
        size_t smem = static_cast<size_t>(m) * hidden_dim * sizeof(scalar_t)
            + static_cast<size_t>(m) * sizeof(float);
        const auto* bias = static_cast<const scalar_t*>(bias_ptr);
        if (outputs_per_block == 2) {
            rmsnorm_linear_m_row_warp_group_kernel<scalar_t, 2>
                <<<(output_dim + 1) / 2, 64, smem, stream>>>(
                    input_c.data_ptr<scalar_t>(), norm_c.data_ptr<scalar_t>(),
                    w_c.data_ptr<scalar_t>(), bias, output.data_ptr<scalar_t>(),
                    m, hidden_dim, output_dim, has_bias, static_cast<float>(eps));
        } else if (outputs_per_block == 4) {
            rmsnorm_linear_m_row_warp_group_kernel<scalar_t, 4>
                <<<(output_dim + 3) / 4, 128, smem, stream>>>(
                    input_c.data_ptr<scalar_t>(), norm_c.data_ptr<scalar_t>(),
                    w_c.data_ptr<scalar_t>(), bias, output.data_ptr<scalar_t>(),
                    m, hidden_dim, output_dim, has_bias, static_cast<float>(eps));
        } else {
            rmsnorm_linear_m_row_warp_group_kernel<scalar_t, 8>
                <<<(output_dim + 7) / 8, 256, smem, stream>>>(
                    input_c.data_ptr<scalar_t>(), norm_c.data_ptr<scalar_t>(),
                    w_c.data_ptr<scalar_t>(), bias, output.data_ptr<scalar_t>(),
                    m, hidden_dim, output_dim, has_bias, static_cast<float>(eps));
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return output;
}

torch::Tensor m_row_linear_residual(
        torch::Tensor input,
        torch::Tensor residual,
        torch::Tensor weight,
        c10::optional<torch::Tensor> bias,
        int64_t outputs_per_block) {
    TORCH_CHECK(input.is_cuda() && residual.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(residual.dim() == 3 && residual.size(0) == 1, "residual must be [1,m,H]");
    const int m = static_cast<int>(residual.size(1));
    const int hidden_dim = static_cast<int>(residual.size(2));
    TORCH_CHECK(m >= 1 && m <= MROW_MAX_M, "m must be in [1,8]");
    TORCH_CHECK(weight.size(0) == hidden_dim, "weight out mismatch");
    const int in_dim = static_cast<int>(weight.size(1));
    TORCH_CHECK(input.numel() == m * in_dim, "input numel mismatch");
    TORCH_CHECK(outputs_per_block == 2 || outputs_per_block == 4 || outputs_per_block == 8,
            "unsupported outputs_per_block");

    c10::cuda::CUDAGuard device_guard(input.device());
    auto input_c = input.contiguous().view({m, in_dim});
    auto residual_c = residual.contiguous();
    auto w_c = weight.contiguous();
    torch::Tensor bias_c; const void* bias_ptr = nullptr; int has_bias = 0;
    if (bias.has_value()) {
        bias_c = bias.value().contiguous();
        bias_ptr = bias_c.data_ptr();
        has_bias = 1;
    }
    auto output = torch::empty({1, m, hidden_dim}, residual.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "m_row_linear_residual", [&] {
        size_t smem = static_cast<size_t>(m) * in_dim * sizeof(scalar_t);
        const auto* b = static_cast<const scalar_t*>(bias_ptr);
        if (outputs_per_block == 2) {
            linear_residual_m_row_warp_group_kernel<scalar_t, 2>
                <<<(hidden_dim + 1) / 2, 64, smem, stream>>>(
                    input_c.data_ptr<scalar_t>(), residual_c.data_ptr<scalar_t>(),
                    w_c.data_ptr<scalar_t>(), b, output.data_ptr<scalar_t>(),
                    m, in_dim, hidden_dim, has_bias);
        } else if (outputs_per_block == 4) {
            linear_residual_m_row_warp_group_kernel<scalar_t, 4>
                <<<(hidden_dim + 3) / 4, 128, smem, stream>>>(
                    input_c.data_ptr<scalar_t>(), residual_c.data_ptr<scalar_t>(),
                    w_c.data_ptr<scalar_t>(), b, output.data_ptr<scalar_t>(),
                    m, in_dim, hidden_dim, has_bias);
        } else {
            linear_residual_m_row_warp_group_kernel<scalar_t, 8>
                <<<(hidden_dim + 7) / 8, 256, smem, stream>>>(
                    input_c.data_ptr<scalar_t>(), residual_c.data_ptr<scalar_t>(),
                    w_c.data_ptr<scalar_t>(), b, output.data_ptr<scalar_t>(),
                    m, in_dim, hidden_dim, has_bias);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return output;
}

torch::Tensor m_row_rope_cache_write(
        torch::Tensor qkv,
        torch::Tensor cache_keys,
        torch::Tensor cache_values,
        torch::Tensor cos,
        torch::Tensor sin,
        torch::Tensor cache_position) {
    TORCH_CHECK(qkv.is_cuda(), "qkv must be CUDA");
    TORCH_CHECK(qkv.dim() == 5 && qkv.size(0) == 1 && qkv.size(2) == 3,
            "qkv must be [1,m,3,H,D]");
    const int m = static_cast<int>(qkv.size(1));
    const int heads = static_cast<int>(qkv.size(3));
    const int head_dim = static_cast<int>(qkv.size(4));
    TORCH_CHECK(m >= 1 && m <= MROW_MAX_M, "m must be in [1,8]");
    TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even");
    TORCH_CHECK(cache_position.numel() == m, "cache_position must have m elements");
    TORCH_CHECK(cache_position.scalar_type() == torch::kLong, "cache_position must be int64");
    TORCH_CHECK(cos.dim() == 3 && sin.dim() == 3, "cos/sin must be 3D");
    TORCH_CHECK(cos.size(0) == 1 && cos.size(1) == m && cos.size(2) == head_dim,
            "cos must be [1,m,D]");
    TORCH_CHECK(sin.size(0) == 1 && sin.size(1) == m && sin.size(2) == head_dim,
            "sin must be [1,m,D]");
    TORCH_CHECK(cache_keys.size(0) == 1 && cache_keys.size(1) == heads,
            "cache_keys shape mismatch");
    TORCH_CHECK(cache_values.sizes() == cache_keys.sizes(), "cache_values shape mismatch");

    c10::cuda::CUDAGuard device_guard(qkv.device());
    auto qkv_c = qkv.contiguous();
    auto cos_c = cos.contiguous();
    auto sin_c = sin.contiguous();
    auto pos_c = cache_position.contiguous();
    auto query = torch::empty({1, heads, m, head_dim}, qkv.options());
    const int max_cache_len = static_cast<int>(cache_keys.size(2));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(qkv.scalar_type(), "m_row_rope_cache_write", [&] {
        dim3 grid(heads, m);
        dim3 block(64);
        m_row_rope_cache_write_kernel<scalar_t><<<grid, block, 0, stream>>>(
            qkv_c.data_ptr<scalar_t>(),
            cache_keys.data_ptr<scalar_t>(),
            cache_values.data_ptr<scalar_t>(),
            cos_c.data_ptr<scalar_t>(),
            sin_c.data_ptr<scalar_t>(),
            pos_c.data_ptr<int64_t>(),
            query.data_ptr<scalar_t>(),
            m, heads, head_dim, max_cache_len,
            qkv_c.stride(1), qkv_c.stride(2), qkv_c.stride(3), qkv_c.stride(4),
            cache_keys.stride(1), cache_keys.stride(2), cache_keys.stride(3),
            cos_c.stride(1), cos_c.stride(2),
            sin_c.stride(1), sin_c.stride(2),
            query.stride(1), query.stride(2), query.stride(3));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return query;
}
"""


def _load_native_m_row():
    global _NATIVE_M_ROW
    if _NATIVE_M_ROW is not None:
        return _NATIVE_M_ROW
    _NATIVE_M_ROW = load_inline(
        name="mapperatorinator_native_m_row_verify",
        cpp_sources=(
            "#include <torch/extension.h>\n"
            "torch::Tensor m_row_mlp_residual("
            "torch::Tensor input, torch::Tensor norm_weight, torch::Tensor fc1_weight, "
            "c10::optional<torch::Tensor> fc1_bias, torch::Tensor fc2_weight, "
            "c10::optional<torch::Tensor> fc2_bias, double eps, int64_t outputs_per_block);\n"
            "torch::Tensor m_row_rmsnorm_linear("
            "torch::Tensor input, torch::Tensor norm_weight, torch::Tensor linear_weight, "
            "c10::optional<torch::Tensor> linear_bias, double eps, int64_t outputs_per_block);\n"
            "torch::Tensor m_row_linear_residual("
            "torch::Tensor input, torch::Tensor residual, torch::Tensor weight, "
            "c10::optional<torch::Tensor> bias, int64_t outputs_per_block);\n"
            "torch::Tensor m_row_rope_cache_write("
            "torch::Tensor qkv, torch::Tensor cache_keys, torch::Tensor cache_values, "
            "torch::Tensor cos, torch::Tensor sin, torch::Tensor cache_position);\n"
        ),
        cuda_sources=_CUDA,
        functions=[
            "m_row_mlp_residual",
            "m_row_rmsnorm_linear",
            "m_row_linear_residual",
            "m_row_rope_cache_write",
        ],
        extra_cuda_cflags=["-O3", f"-DMROW_MAX_M={MAX_M}"],
        verbose=False,
    )
    return _NATIVE_M_ROW


def preload_native_m_row():
    return _load_native_m_row()


def native_m_row_mlp_residual(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor | None,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor | None,
    *,
    eps: float,
    outputs_per_block: int = 8,
) -> torch.Tensor:
    return _load_native_m_row().m_row_mlp_residual(
        input,
        norm_weight,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
        float(eps),
        int(outputs_per_block),
    )


def native_m_row_rmsnorm_linear(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor | None,
    *,
    eps: float,
    outputs_per_block: int = 8,
) -> torch.Tensor:
    return _load_native_m_row().m_row_rmsnorm_linear(
        input,
        norm_weight,
        linear_weight,
        linear_bias,
        float(eps),
        int(outputs_per_block),
    )


def native_m_row_linear_residual(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    outputs_per_block: int = 8,
) -> torch.Tensor:
    return _load_native_m_row().m_row_linear_residual(
        input,
        residual,
        weight,
        bias,
        int(outputs_per_block),
    )


def native_m_row_rope_cache_write(
    qkv: torch.Tensor,
    cache_keys: torch.Tensor,
    cache_values: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_position: torch.Tensor,
) -> torch.Tensor:
    return _load_native_m_row().m_row_rope_cache_write(
        qkv,
        cache_keys,
        cache_values,
        cos,
        sin,
        cache_position,
    )


__all__ = [
    "MAX_M",
    "preload_native_m_row",
    "native_m_row_mlp_residual",
    "native_m_row_rmsnorm_linear",
    "native_m_row_linear_residual",
    "native_m_row_rope_cache_write",
]
