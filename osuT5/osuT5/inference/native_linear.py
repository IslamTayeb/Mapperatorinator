from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline


_NATIVE_LINEAR = None


def _load_native_linear():
    global _NATIVE_LINEAR
    if _NATIVE_LINEAR is not None:
        return _NATIVE_LINEAR

    cuda_source = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void one_token_linear_kernel(
        const float* __restrict__ input,
        const float* __restrict__ weight,
        const float* __restrict__ bias,
        float* __restrict__ output,
        int rows,
        int in_dim,
        int out_dim,
        int has_bias) {
    int out_idx = blockIdx.x;
    int row = blockIdx.y;
    int tid = threadIdx.x;
    if (out_idx >= out_dim || row >= rows) {
        return;
    }

    const float* x = input + row * in_dim;
    const float* w = weight + out_idx * in_dim;
    float sum = 0.0f;
    for (int idx = tid; idx < in_dim; idx += BLOCK_SIZE) {
        sum += x[idx] * w[idx];
    }

    __shared__ float scratch[BLOCK_SIZE];
    scratch[tid] = sum;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float value = scratch[0];
        if (has_bias) {
            value += bias[out_idx];
        }
        output[row * out_dim + out_idx] = value;
    }
}

torch::Tensor one_token_linear(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias, int64_t block_size) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "input/weight must be CUDA tensors");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be fp32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be fp32");
    TORCH_CHECK(input.dim() == 2, "input must have shape [rows, in_dim]");
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [out_dim, in_dim]");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(input.size(1) == weight.size(1), "input/weight in_dim mismatch");

    const int rows = static_cast<int>(input.size(0));
    const int in_dim = static_cast<int>(input.size(1));
    const int out_dim = static_cast<int>(weight.size(0));
    auto output = torch::empty({rows, out_dim}, input.options());

    const float* bias_ptr = nullptr;
    int has_bias = 0;
    torch::Tensor bias_contiguous;
    if (bias.has_value()) {
        bias_contiguous = bias.value().contiguous();
        TORCH_CHECK(bias_contiguous.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(bias_contiguous.scalar_type() == torch::kFloat32, "bias must be fp32");
        TORCH_CHECK(bias_contiguous.numel() == out_dim, "bias/out_dim mismatch");
        bias_ptr = bias_contiguous.data_ptr<float>();
        has_bias = 1;
    }

    dim3 grid(out_dim, rows);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (block_size == 128) {
        one_token_linear_kernel<128><<<grid, 128, 0, stream>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
            rows, in_dim, out_dim, has_bias);
    } else if (block_size == 256) {
        one_token_linear_kernel<256><<<grid, 256, 0, stream>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
            rows, in_dim, out_dim, has_bias);
    } else if (block_size == 512) {
        one_token_linear_kernel<512><<<grid, 512, 0, stream>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
            rows, in_dim, out_dim, has_bias);
    } else {
        TORCH_CHECK(false, "unsupported native linear block size; expected 128, 256, or 512");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

    _NATIVE_LINEAR = load_inline(
        name="mapperatorinator_native_linear",
        cpp_sources=(
            "#include <torch/extension.h>\n"
            "torch::Tensor one_token_linear(torch::Tensor input, torch::Tensor weight, "
            "c10::optional<torch::Tensor> bias, int64_t block_size);\n"
        ),
        cuda_sources=cuda_source,
        functions=["one_token_linear"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return _NATIVE_LINEAR


def preload_native_linear():
    return _load_native_linear()


def native_one_token_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        block_size: int,
) -> torch.Tensor:
    extension = _load_native_linear()
    output_shape = (*input.shape[:-1], weight.shape[0])
    input_2d = input.reshape(-1, input.shape[-1]).contiguous()
    weight_contiguous = weight.contiguous()
    output = extension.one_token_linear(input_2d, weight_contiguous, bias, int(block_size))
    return output.reshape(output_shape)
