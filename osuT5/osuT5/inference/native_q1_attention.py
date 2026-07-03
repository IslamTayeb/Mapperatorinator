from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline


_NATIVE_Q1_ATTENTION = None


def _load_native_q1_attention():
    global _NATIVE_Q1_ATTENTION
    if _NATIVE_Q1_ATTENTION is not None:
        return _NATIVE_Q1_ATTENTION

    cuda_source = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

template<int BLOCK_SIZE>
__global__ void q1_attention_kernel(
        const float* __restrict__ q,
        const float* __restrict__ k,
        const float* __restrict__ v,
        const float* __restrict__ mask,
        float* __restrict__ out,
        int heads,
        int kv_len,
        int head_dim,
        long long q_head_stride,
        long long q_dim_stride,
        long long k_head_stride,
        long long k_len_stride,
        long long k_dim_stride,
        long long v_head_stride,
        long long v_len_stride,
        long long v_dim_stride,
        int has_mask) {
    int head = blockIdx.x;
    int tid = threadIdx.x;
    if (head >= heads) {
        return;
    }

    const float* qh = q + head * q_head_stride;
    const float* kh = k + head * k_head_stride;
    const float* vh = v + head * v_head_stride;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    extern __shared__ float shared_mem[];
    float* scores = shared_mem;
    float* scratch = shared_mem + kv_len;

    float max_score = -FLT_MAX;
    for (int pos = tid; pos < kv_len; pos += BLOCK_SIZE) {
        float score = 0.0f;
        const float* kp = kh + pos * k_len_stride;
        for (int d = 0; d < head_dim; ++d) {
            score += qh[d * q_dim_stride] * kp[d * k_dim_stride];
        }
        score *= scale;
        if (has_mask) {
            score += mask[pos];
        }
        scores[pos] = score;
        max_score = fmaxf(max_score, score);
    }

    scratch[tid] = max_score;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }
    max_score = scratch[0];

    float denom = 0.0f;
    for (int pos = tid; pos < kv_len; pos += BLOCK_SIZE) {
        float weight = expf(scores[pos] - max_score);
        scores[pos] = weight;
        denom += weight;
    }

    scratch[tid] = denom;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    denom = scratch[0];

    for (int dim = tid; dim < head_dim; dim += BLOCK_SIZE) {
        float numer = 0.0f;
        for (int pos = 0; pos < kv_len; ++pos) {
            numer += scores[pos] * vh[pos * v_len_stride + dim * v_dim_stride];
        }
        out[head * head_dim + dim] = numer / denom;
    }
}

torch::Tensor q1_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, c10::optional<torch::Tensor> mask) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "q must be fp32");
    TORCH_CHECK(k.scalar_type() == torch::kFloat32, "k must be fp32");
    TORCH_CHECK(v.scalar_type() == torch::kFloat32, "v must be fp32");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be 4D");
    TORCH_CHECK(q.size(0) == 1 && q.size(2) == 1, "q must have shape [1, H, 1, D]");
    TORCH_CHECK(k.size(0) == 1 && v.size(0) == 1, "k/v batch must be 1");
    TORCH_CHECK(k.size(1) == q.size(1) && v.size(1) == q.size(1), "head count mismatch");
    TORCH_CHECK(k.size(3) == q.size(3) && v.size(3) == q.size(3), "head dim mismatch");
    TORCH_CHECK(k.size(2) == v.size(2), "kv length mismatch");
    int heads = static_cast<int>(q.size(1));
    int kv_len = static_cast<int>(k.size(2));
    int head_dim = static_cast<int>(q.size(3));
    auto output = torch::empty({1, heads, 1, head_dim}, q.options());

    const float* mask_ptr = nullptr;
    int has_mask = 0;
    torch::Tensor mask_contiguous;
    if (mask.has_value()) {
        mask_contiguous = mask.value().contiguous();
        TORCH_CHECK(mask_contiguous.is_cuda(), "mask must be CUDA");
        TORCH_CHECK(mask_contiguous.scalar_type() == torch::kFloat32, "mask must be fp32");
        TORCH_CHECK(mask_contiguous.numel() == kv_len, "mask must have kv_len elements");
        mask_ptr = mask_contiguous.data_ptr<float>();
        has_mask = 1;
    }

    constexpr int block_size = 128;
    dim3 grid(heads);
    dim3 block(block_size);
    size_t shared_bytes = static_cast<size_t>(kv_len + block_size) * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    q1_attention_kernel<block_size><<<grid, block, shared_bytes, stream>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        mask_ptr,
        output.data_ptr<float>(),
        heads,
        kv_len,
        head_dim,
        static_cast<long long>(q.stride(1)),
        static_cast<long long>(q.stride(3)),
        static_cast<long long>(k.stride(1)),
        static_cast<long long>(k.stride(2)),
        static_cast<long long>(k.stride(3)),
        static_cast<long long>(v.stride(1)),
        static_cast<long long>(v.stride(2)),
        static_cast<long long>(v.stride(3)),
        has_mask
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

    _NATIVE_Q1_ATTENTION = load_inline(
        name="mapperatorinator_q1_attention",
        cpp_sources=(
            "#include <torch/extension.h>\n"
            "torch::Tensor q1_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, "
            "c10::optional<torch::Tensor> mask);\n"
        ),
        cuda_sources=cuda_source,
        functions=["q1_attention"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return _NATIVE_Q1_ATTENTION


def preload_native_q1_attention():
    return _load_native_q1_attention()


def native_q1_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    extension = _load_native_q1_attention()
    mask = None
    if attention_mask is not None:
        mask = attention_mask.reshape(-1).to(dtype=torch.float32).contiguous()
    return extension.q1_attention(query, key, value, mask)
