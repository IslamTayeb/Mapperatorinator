"""Opt-in SM75 split-KV scout for fused q1 RoPE/cache self-attention.

The accepted production kernel remains the source of truth.  This module is
loaded only by the split-KV component verifier and intentionally has no runtime
selector wiring.
"""

from __future__ import annotations

import torch

from ..kernels.q1_attention import (
    _native_mask,
    _validate_rope_cache_inputs,
)


SUPPORTED_SPLITS = (1, 2, 4, 8)
_SPLIT_KV_EXTENSION = None


def _load_inline(**kwargs):
    from torch.utils.cpp_extension import load_inline

    return load_inline(**kwargs)


_CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>

template<typename scalar_t>
struct SplitScalarTraits;

template<>
struct SplitScalarTraits<float> {
    __device__ __forceinline__ static float load(float value) { return value; }
    __device__ __forceinline__ static float store(float value) { return value; }
};

template<>
struct SplitScalarTraits<__half> {
    __device__ __forceinline__ static float load(__half value) {
        return __half2float(value);
    }
    __device__ __forceinline__ static __half store(float value) {
        return __float2half_rn(value);
    }
};

template<typename scalar_t, int BLOCK_SIZE>
__global__ void prepare_rope_cache_kernel(
        const scalar_t* __restrict__ qkv,
        scalar_t* __restrict__ cache_keys,
        scalar_t* __restrict__ cache_values,
        const scalar_t* __restrict__ cos,
        const scalar_t* __restrict__ sin,
        const int64_t* __restrict__ cache_position,
        float* __restrict__ query,
        int heads,
        int active_prefix_len,
        int max_cache_len,
        int head_dim,
        long long qkv_qkv_stride,
        long long qkv_head_stride,
        long long qkv_dim_stride,
        long long cache_head_stride,
        long long cache_len_stride,
        long long cache_dim_stride,
        long long cos_dim_stride,
        long long sin_dim_stride) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    if (head >= heads) return;
    const int write_pos = static_cast<int>(cache_position[0]);
    if (write_pos < 0 || write_pos >= active_prefix_len || write_pos >= max_cache_len) {
        asm("trap;");
        return;
    }
    using Traits = SplitScalarTraits<scalar_t>;
    const int half_dim = head_dim / 2;
    const long long q_base = head * qkv_head_stride;
    for (int dim = tid; dim < head_dim; dim += BLOCK_SIZE) {
        const int rotated_dim = dim < half_dim ? dim + half_dim : dim - half_dim;
        const float sign = dim < half_dim ? -1.0f : 1.0f;
        const float cos_v = Traits::load(cos[dim * cos_dim_stride]);
        const float sin_v = Traits::load(sin[dim * sin_dim_stride]);
        const float q_raw = Traits::load(
            qkv[q_base + dim * qkv_dim_stride]);
        const float q_rot_raw = Traits::load(
            qkv[q_base + rotated_dim * qkv_dim_stride]);
        const float k_raw = Traits::load(
            qkv[q_base + qkv_qkv_stride + dim * qkv_dim_stride]);
        const float k_rot_raw = Traits::load(
            qkv[q_base + qkv_qkv_stride + rotated_dim * qkv_dim_stride]);
        const float v_raw = Traits::load(
            qkv[q_base + 2 * qkv_qkv_stride + dim * qkv_dim_stride]);
        query[head * head_dim + dim] = q_raw * cos_v + sign * q_rot_raw * sin_v;
        cache_keys[
            head * cache_head_stride + write_pos * cache_len_stride
            + dim * cache_dim_stride
        ] = Traits::store(k_raw * cos_v + sign * k_rot_raw * sin_v);
        cache_values[
            head * cache_head_stride + write_pos * cache_len_stride
            + dim * cache_dim_stride
        ] = Traits::store(v_raw);
    }
}

template<typename scalar_t, int BLOCK_SIZE>
__global__ void split_partial_kernel(
        const float* __restrict__ query,
        const scalar_t* __restrict__ cache_keys,
        const scalar_t* __restrict__ cache_values,
        const float* __restrict__ mask,
        float* __restrict__ partial_max,
        float* __restrict__ partial_denom,
        float* __restrict__ partial_numer,
        int heads,
        int active_prefix_len,
        int head_dim,
        int split_count,
        long long cache_head_stride,
        long long cache_len_stride,
        long long cache_dim_stride,
        int has_mask) {
    const int head = blockIdx.x;
    const int split = blockIdx.y;
    const int tid = threadIdx.x;
    if (head >= heads || split >= split_count) return;
    using Traits = SplitScalarTraits<scalar_t>;
    const int begin = active_prefix_len * split / split_count;
    const int end = active_prefix_len * (split + 1) / split_count;
    const int length = end - begin;
    const int partial_index = head * split_count + split;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    extern __shared__ float shared[];
    float* scores = shared;
    float* reduction = shared + length;

    float local_max = -FLT_MAX;
    for (int offset = tid; offset < length; offset += BLOCK_SIZE) {
        const int pos = begin + offset;
        float score = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) {
            score += query[head * head_dim + dim] * Traits::load(cache_keys[
                head * cache_head_stride + pos * cache_len_stride
                + dim * cache_dim_stride
            ]);
        }
        score *= scale;
        if (has_mask) score += mask[pos];
        scores[offset] = score;
        local_max = fmaxf(local_max, score);
    }
    reduction[tid] = local_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
        __syncthreads();
    }
    const float split_max = reduction[0];

    float local_denom = 0.0f;
    for (int offset = tid; offset < length; offset += BLOCK_SIZE) {
        const float weight = expf(scores[offset] - split_max);
        scores[offset] = weight;
        local_denom += weight;
    }
    reduction[tid] = local_denom;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        partial_max[partial_index] = split_max;
        partial_denom[partial_index] = reduction[0];
    }
    for (int dim = tid; dim < head_dim; dim += BLOCK_SIZE) {
        float numerator = 0.0f;
        for (int offset = 0; offset < length; ++offset) {
            const int pos = begin + offset;
            numerator += scores[offset] * Traits::load(cache_values[
                head * cache_head_stride + pos * cache_len_stride
                + dim * cache_dim_stride
            ]);
        }
        partial_numer[(partial_index * head_dim) + dim] = numerator;
    }
}

template<typename scalar_t, int BLOCK_SIZE>
__global__ void merge_partials_kernel(
        const float* __restrict__ partial_max,
        const float* __restrict__ partial_denom,
        const float* __restrict__ partial_numer,
        scalar_t* __restrict__ output,
        int heads,
        int head_dim,
        int split_count) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    if (head >= heads) return;
    using Traits = SplitScalarTraits<scalar_t>;
    __shared__ float global_max;
    __shared__ float global_denom;
    if (tid == 0) {
        float maximum = -FLT_MAX;
        for (int split = 0; split < split_count; ++split) {
            maximum = fmaxf(maximum, partial_max[head * split_count + split]);
        }
        float denominator = 0.0f;
        for (int split = 0; split < split_count; ++split) {
            const int index = head * split_count + split;
            denominator += expf(partial_max[index] - maximum) * partial_denom[index];
        }
        global_max = maximum;
        global_denom = denominator;
    }
    __syncthreads();
    for (int dim = tid; dim < head_dim; dim += BLOCK_SIZE) {
        float numerator = 0.0f;
        for (int split = 0; split < split_count; ++split) {
            const int index = head * split_count + split;
            numerator += expf(partial_max[index] - global_max)
                * partial_numer[index * head_dim + dim];
        }
        output[head * head_dim + dim] = Traits::store(numerator / global_denom);
    }
}

torch::Tensor split_kv_q1_rope_cache_attention(
        torch::Tensor qkv,
        torch::Tensor cache_keys,
        torch::Tensor cache_values,
        torch::Tensor cos,
        torch::Tensor sin,
        torch::Tensor cache_position,
        c10::optional<torch::Tensor> mask,
        int64_t active_prefix_len,
        int64_t split_count) {
    TORCH_CHECK(qkv.is_cuda() && cache_keys.is_cuda() && cache_values.is_cuda(),
        "qkv/cache must be CUDA tensors");
    TORCH_CHECK(cos.is_cuda() && sin.is_cuda() && cache_position.is_cuda(),
        "RoPE/position must be CUDA tensors");
    TORCH_CHECK(qkv.scalar_type() == torch::kFloat32 || qkv.scalar_type() == torch::kFloat16,
        "qkv must be fp32 or fp16");
    TORCH_CHECK(qkv.scalar_type() == cache_keys.scalar_type()
        && qkv.scalar_type() == cache_values.scalar_type()
        && qkv.scalar_type() == cos.scalar_type()
        && qkv.scalar_type() == sin.scalar_type(), "storage dtype mismatch");
    TORCH_CHECK(split_count == 1 || split_count == 2 || split_count == 4 || split_count == 8,
        "split_count must be 1, 2, 4, or 8");
    TORCH_CHECK(active_prefix_len >= split_count, "active prefix is shorter than split count");
    TORCH_CHECK(qkv.dim() == 5 && qkv.size(0) == 1 && qkv.size(1) == 1
        && qkv.size(2) == 3, "qkv must have shape [1, 1, 3, H, D]");
    TORCH_CHECK(cache_keys.dim() == 4 && cache_values.dim() == 4,
        "cache must have shape [1, H, L, D]");
    TORCH_CHECK(cache_position.scalar_type() == torch::kLong
        && cache_position.numel() == 1, "cache_position must be scalar int64");
    const int heads = static_cast<int>(qkv.size(3));
    const int head_dim = static_cast<int>(qkv.size(4));
    const int max_cache_len = static_cast<int>(cache_keys.size(2));
    TORCH_CHECK(head_dim > 0 && head_dim % 2 == 0, "head_dim must be positive and even");
    TORCH_CHECK(active_prefix_len > 0 && active_prefix_len <= max_cache_len,
        "invalid active prefix");
    c10::cuda::CUDAGuard device_guard(qkv.device());
    cudaDeviceProp properties;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, qkv.get_device()));
    TORCH_CHECK(properties.major == 7 && properties.minor == 5,
        "split-KV scout requires SM75");

    const float* mask_ptr = nullptr;
    int has_mask = 0;
    torch::Tensor mask_contiguous;
    if (mask.has_value()) {
        mask_contiguous = mask.value().contiguous();
        TORCH_CHECK(mask_contiguous.is_cuda()
            && mask_contiguous.scalar_type() == torch::kFloat32,
            "mask must be CUDA fp32");
        TORCH_CHECK(mask_contiguous.numel() == active_prefix_len,
            "mask must match active prefix");
        mask_ptr = mask_contiguous.data_ptr<float>();
        has_mask = 1;
    }

    auto float_options = qkv.options().dtype(torch::kFloat32);
    auto query = torch::empty({heads, head_dim}, float_options);
    auto partial_max = torch::empty({heads, split_count}, float_options);
    auto partial_denom = torch::empty({heads, split_count}, float_options);
    auto partial_numer = torch::empty({heads, split_count, head_dim}, float_options);
    auto output = torch::empty({1, heads, 1, head_dim}, qkv.options());
    constexpr int block_size = 128;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int largest_split = static_cast<int>(
        (active_prefix_len + split_count - 1) / split_count);
    const size_t partial_shared = static_cast<size_t>(largest_split + block_size)
        * sizeof(float);

#define LAUNCH_SPLIT(TYPE, PTR) \
    prepare_rope_cache_kernel<TYPE, block_size><<<heads, block_size, 0, stream>>>( \
        PTR(qkv.data_ptr()), PTR(cache_keys.data_ptr()), PTR(cache_values.data_ptr()), \
        PTR(cos.data_ptr()), PTR(sin.data_ptr()), cache_position.data_ptr<int64_t>(), \
        query.data_ptr<float>(), heads, static_cast<int>(active_prefix_len), \
        max_cache_len, head_dim, qkv.stride(2), qkv.stride(3), qkv.stride(4), \
        cache_keys.stride(1), cache_keys.stride(2), cache_keys.stride(3), \
        cos.stride(2), sin.stride(2)); \
    split_partial_kernel<TYPE, block_size><<<dim3(heads, split_count), block_size, \
        partial_shared, stream>>>(query.data_ptr<float>(), PTR(cache_keys.data_ptr()), \
        PTR(cache_values.data_ptr()), mask_ptr, partial_max.data_ptr<float>(), \
        partial_denom.data_ptr<float>(), partial_numer.data_ptr<float>(), heads, \
        static_cast<int>(active_prefix_len), head_dim, static_cast<int>(split_count), \
        cache_keys.stride(1), cache_keys.stride(2), cache_keys.stride(3), has_mask); \
    merge_partials_kernel<TYPE, block_size><<<heads, block_size, 0, stream>>>( \
        partial_max.data_ptr<float>(), partial_denom.data_ptr<float>(), \
        partial_numer.data_ptr<float>(), PTR(output.data_ptr()), heads, head_dim, \
        static_cast<int>(split_count));

    if (qkv.scalar_type() == torch::kFloat32) {
#define FLOAT_PTR(value) reinterpret_cast<float*>(value)
        LAUNCH_SPLIT(float, FLOAT_PTR)
#undef FLOAT_PTR
    } else {
#define HALF_PTR(value) reinterpret_cast<__half*>(value)
        LAUNCH_SPLIT(__half, HALF_PTR)
#undef HALF_PTR
    }
#undef LAUNCH_SPLIT
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""


def _load_split_kv_extension():
    global _SPLIT_KV_EXTENSION
    if _SPLIT_KV_EXTENSION is None:
        _SPLIT_KV_EXTENSION = _load_inline(
            name="mapperatorinator_sm75_split_kv_q1_scout",
            cpp_sources=(
                "#include <torch/extension.h>\n"
                "torch::Tensor split_kv_q1_rope_cache_attention("
                "torch::Tensor qkv, torch::Tensor cache_keys, "
                "torch::Tensor cache_values, torch::Tensor cos, torch::Tensor sin, "
                "torch::Tensor cache_position, c10::optional<torch::Tensor> mask, "
                "int64_t active_prefix_len, int64_t split_count);\n"
            ),
            cuda_sources=_CUDA_SOURCE,
            functions=["split_kv_q1_rope_cache_attention"],
            extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_75,code=sm_75"],
            verbose=False,
        )
    return _SPLIT_KV_EXTENSION


def preload_split_kv_q1():
    return _load_split_kv_extension()


def split_kv_q1_rope_cache_attention(
    qkv: torch.Tensor,
    cache_keys: torch.Tensor,
    cache_values: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_position: torch.Tensor,
    attention_mask: torch.Tensor | None,
    active_prefix_length: int,
    split_count: int,
) -> torch.Tensor:
    _validate_rope_cache_inputs(
        qkv,
        cache_keys,
        cache_values,
        cos,
        sin,
        cache_position,
        active_prefix_length,
    )
    if isinstance(split_count, bool) or split_count not in SUPPORTED_SPLITS:
        raise ValueError(f"split_count must be one of {SUPPORTED_SPLITS}")
    if int(active_prefix_length) < split_count:
        raise ValueError("active_prefix_length must be at least split_count")
    capability = torch.cuda.get_device_capability(qkv.device)
    if capability != (7, 5):
        raise RuntimeError(f"split-KV q1 scout requires SM75, got sm_{capability[0]}{capability[1]}")
    mask = _native_mask(
        attention_mask,
        device=qkv.device,
        expected_numel=int(active_prefix_length),
    )
    return _load_split_kv_extension().split_kv_q1_rope_cache_attention(
        qkv,
        cache_keys,
        cache_values,
        cos,
        sin,
        cache_position,
        mask,
        int(active_prefix_length),
        split_count,
    )


__all__ = [
    "SUPPORTED_SPLITS",
    "preload_split_kv_q1",
    "split_kv_q1_rope_cache_attention",
]
