"""§54 Stage B: row-tiled m-row split-KV attention for turbo verify.

Replaces Stage A SDPA core. Grid is (heads, splits) with an inner row loop so
KV is not re-read m× via a per-(head,row) launch. Per-row KV lengths encode
causal draft visibility (typically cache_position[r] + 1).
"""
from __future__ import annotations

import os

import torch
from torch.utils.cpp_extension import load_inline

from .m_row import MAX_M

_NATIVE_M_ROW_SPLIT_KV = None
_DEFAULT_SPLITS = 8

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

#ifndef MROW_MAX_M
#define MROW_MAX_M 8
#endif

template<typename scalar_t, int BLOCK_SIZE>
__global__ void m_row_split_kv_partial_kernel(
        const scalar_t* __restrict__ query,
        const scalar_t* __restrict__ cache_keys,
        const scalar_t* __restrict__ cache_values,
        const int* __restrict__ row_lengths,
        float* __restrict__ partial_max,
        float* __restrict__ partial_denom,
        float* __restrict__ partial_numer,
        int m,
        int heads,
        int head_dim,
        int max_prefix_len,
        int split_count,
        long long q_head_stride,
        long long q_row_stride,
        long long q_dim_stride,
        long long cache_head_stride,
        long long cache_len_stride,
        long long cache_dim_stride) {
    const int head = blockIdx.x;
    const int split = blockIdx.y;
    const int tid = threadIdx.x;
    if (head >= heads || split >= split_count) return;

    const int begin = max_prefix_len * split / split_count;
    const int end = max_prefix_len * (split + 1) / split_count;
    const int length = end - begin;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    extern __shared__ float shared[];
    // scores[m * length] then reduction[BLOCK_SIZE]
    float* scores = shared;
    float* reduction = shared + m * length;

    for (int idx = tid; idx < m * length; idx += BLOCK_SIZE) {
        scores[idx] = -FLT_MAX;
    }
    __syncthreads();

    for (int offset = tid; offset < length; offset += BLOCK_SIZE) {
        const int pos = begin + offset;
        for (int row = 0; row < m; ++row) {
            const int row_len = row_lengths[row];
            if (pos >= row_len) {
                continue;
            }
            float score = 0.0f;
            for (int dim = 0; dim < head_dim; ++dim) {
                const float qv = static_cast<float>(query[
                    head * q_head_stride + row * q_row_stride + dim * q_dim_stride]);
                const float kv = static_cast<float>(cache_keys[
                    head * cache_head_stride + pos * cache_len_stride
                    + dim * cache_dim_stride]);
                score += qv * kv;
            }
            scores[row * length + offset] = score * scale;
        }
    }
    __syncthreads();

    for (int row = 0; row < m; ++row) {
        const int row_len = row_lengths[row];
        const int row_end = row_len < end ? row_len : end;
        const int row_begin = begin;
        const int valid = row_end > row_begin ? (row_end - row_begin) : 0;
        const int partial_index = ((head * m) + row) * split_count + split;

        float local_max = -FLT_MAX;
        for (int offset = tid; offset < valid; offset += BLOCK_SIZE) {
            local_max = fmaxf(local_max, scores[row * length + offset]);
        }
        reduction[tid] = local_max;
        __syncthreads();
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
            }
            __syncthreads();
        }
        const float split_max = valid > 0 ? reduction[0] : -FLT_MAX;

        float local_denom = 0.0f;
        for (int offset = tid; offset < valid; offset += BLOCK_SIZE) {
            const float weight = expf(scores[row * length + offset] - split_max);
            scores[row * length + offset] = weight;
            local_denom += weight;
        }
        reduction[tid] = local_denom;
        __syncthreads();
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduction[tid] += reduction[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            partial_max[partial_index] = split_max;
            partial_denom[partial_index] = valid > 0 ? reduction[0] : 0.0f;
        }
        __syncthreads();

        for (int dim = tid; dim < head_dim; dim += BLOCK_SIZE) {
            float numerator = 0.0f;
            for (int offset = 0; offset < valid; ++offset) {
                const int pos = begin + offset;
                numerator += scores[row * length + offset] * static_cast<float>(
                    cache_values[
                        head * cache_head_stride + pos * cache_len_stride
                        + dim * cache_dim_stride]);
            }
            partial_numer[(partial_index * head_dim) + dim] = numerator;
        }
        __syncthreads();
    }
}

template<typename scalar_t, int BLOCK_SIZE>
__global__ void m_row_split_kv_merge_kernel(
        const float* __restrict__ partial_max,
        const float* __restrict__ partial_denom,
        const float* __restrict__ partial_numer,
        scalar_t* __restrict__ output,
        int m,
        int heads,
        int head_dim,
        int split_count,
        long long out_head_stride,
        long long out_row_stride,
        long long out_dim_stride) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    if (head >= heads || row >= m) return;

    __shared__ float global_max;
    __shared__ float global_denom;
    if (tid == 0) {
        float maximum = -FLT_MAX;
        for (int split = 0; split < split_count; ++split) {
            const int index = ((head * m) + row) * split_count + split;
            maximum = fmaxf(maximum, partial_max[index]);
        }
        float denominator = 0.0f;
        for (int split = 0; split < split_count; ++split) {
            const int index = ((head * m) + row) * split_count + split;
            denominator += expf(partial_max[index] - maximum) * partial_denom[index];
        }
        global_max = maximum;
        global_denom = denominator > 0.0f ? denominator : 1.0f;
    }
    __syncthreads();

    for (int dim = tid; dim < head_dim; dim += BLOCK_SIZE) {
        float numerator = 0.0f;
        for (int split = 0; split < split_count; ++split) {
            const int index = ((head * m) + row) * split_count + split;
            numerator += expf(partial_max[index] - global_max)
                * partial_numer[index * head_dim + dim];
        }
        output[
            head * out_head_stride + row * out_row_stride + dim * out_dim_stride
        ] = static_cast<scalar_t>(numerator / global_denom);
    }
}

torch::Tensor m_row_split_kv_attention(
        torch::Tensor query,
        torch::Tensor cache_keys,
        torch::Tensor cache_values,
        torch::Tensor row_lengths,
        int64_t max_prefix_len,
        int64_t split_count) {
    TORCH_CHECK(query.is_cuda() && cache_keys.is_cuda() && cache_values.is_cuda(),
            "m-row split-KV tensors must be CUDA");
    TORCH_CHECK(query.dim() == 4 && query.size(0) == 1,
            "query must be [1,H,m,D]");
    const int heads = static_cast<int>(query.size(1));
    const int m = static_cast<int>(query.size(2));
    const int head_dim = static_cast<int>(query.size(3));
    TORCH_CHECK(m >= 1 && m <= MROW_MAX_M, "m must be in [1,8]");
    TORCH_CHECK(head_dim > 0 && (head_dim % 2) == 0, "head_dim must be even");
    TORCH_CHECK(cache_keys.sizes() == cache_values.sizes(), "cache shape mismatch");
    TORCH_CHECK(cache_keys.size(0) == 1 && cache_keys.size(1) == heads
            && cache_keys.size(3) == head_dim,
            "cache must be [1,H,L,D]");
    TORCH_CHECK(row_lengths.is_cuda() && row_lengths.scalar_type() == torch::kInt,
            "row_lengths must be CUDA int32");
    TORCH_CHECK(row_lengths.numel() == m, "row_lengths must have m elements");
    TORCH_CHECK(split_count >= 1 && split_count <= 32, "split_count out of range");
    const int max_cache_len = static_cast<int>(cache_keys.size(2));
    TORCH_CHECK(max_prefix_len >= split_count && max_prefix_len <= max_cache_len,
            "max_prefix_len invalid");

    c10::cuda::CUDAGuard device_guard(query.device());
    auto query_c = query.contiguous();
    auto keys_c = cache_keys;
    auto values_c = cache_values;
    auto lengths_c = row_lengths.contiguous();

    auto fp32_options = query.options().dtype(torch::kFloat32);
    auto partial_max = torch::empty({heads, m, split_count}, fp32_options);
    auto partial_denom = torch::empty({heads, m, split_count}, fp32_options);
    auto partial_numer = torch::empty({heads, m, split_count, head_dim}, fp32_options);
    auto output = torch::empty_like(query_c);

    constexpr int block_size = 128;
    const int largest_split = static_cast<int>(
        (max_prefix_len + split_count - 1) / split_count);
    const size_t shared_bytes = static_cast<size_t>(
        m * largest_split + block_size) * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query.scalar_type(), "m_row_split_kv", [&] {
        m_row_split_kv_partial_kernel<scalar_t, block_size>
            <<<dim3(heads, static_cast<int>(split_count)), block_size, shared_bytes, stream>>>(
                query_c.data_ptr<scalar_t>(),
                keys_c.data_ptr<scalar_t>(),
                values_c.data_ptr<scalar_t>(),
                lengths_c.data_ptr<int>(),
                partial_max.data_ptr<float>(),
                partial_denom.data_ptr<float>(),
                partial_numer.data_ptr<float>(),
                m, heads, head_dim, static_cast<int>(max_prefix_len),
                static_cast<int>(split_count),
                query_c.stride(1), query_c.stride(2), query_c.stride(3),
                keys_c.stride(1), keys_c.stride(2), keys_c.stride(3));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        m_row_split_kv_merge_kernel<scalar_t, block_size>
            <<<dim3(heads, m), block_size, 0, stream>>>(
                partial_max.data_ptr<float>(),
                partial_denom.data_ptr<float>(),
                partial_numer.data_ptr<float>(),
                output.data_ptr<scalar_t>(),
                m, heads, head_dim, static_cast<int>(split_count),
                output.stride(1), output.stride(2), output.stride(3));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return output;
}
"""


def _split_count() -> int:
    raw = os.environ.get("MAPPERATORINATOR_TURBO_VERIFY_MROW_SPLITS", "").strip()
    if not raw:
        return _DEFAULT_SPLITS
    value = int(raw)
    if value < 1 or value > 32:
        raise ValueError(f"invalid m-row split count: {value}")
    return value


def _load_native_m_row_split_kv():
    global _NATIVE_M_ROW_SPLIT_KV
    if _NATIVE_M_ROW_SPLIT_KV is not None:
        return _NATIVE_M_ROW_SPLIT_KV
    _NATIVE_M_ROW_SPLIT_KV = load_inline(
        name="mapperatorinator_native_m_row_split_kv",
        cpp_sources=(
            "#include <torch/extension.h>\n"
            "torch::Tensor m_row_split_kv_attention("
            "torch::Tensor query, torch::Tensor cache_keys, torch::Tensor cache_values, "
            "torch::Tensor row_lengths, int64_t max_prefix_len, int64_t split_count);\n"
        ),
        cuda_sources=_CUDA,
        functions=["m_row_split_kv_attention"],
        extra_cuda_cflags=["-O3", f"-DMROW_MAX_M={MAX_M}"],
        verbose=False,
    )
    return _NATIVE_M_ROW_SPLIT_KV


def preload_native_m_row_split_kv():
    return _load_native_m_row_split_kv()


def native_m_row_split_kv_attention(
    query: torch.Tensor,
    cache_keys: torch.Tensor,
    cache_values: torch.Tensor,
    row_lengths: torch.Tensor,
    *,
    max_prefix_len: int,
    split_count: int | None = None,
) -> torch.Tensor:
    splits = int(split_count if split_count is not None else _split_count())
    return _load_native_m_row_split_kv().m_row_split_kv_attention(
        query,
        cache_keys,
        cache_values,
        row_lengths,
        int(max_prefix_len),
        splits,
    )


__all__ = [
    "preload_native_m_row_split_kv",
    "native_m_row_split_kv_attention",
]
