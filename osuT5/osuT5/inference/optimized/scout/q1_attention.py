"""Lazy, verifier-only q_len=1 attention kernels for FP32 and FP16.

Both storage dtypes accumulate attention scores, softmax reductions, and value
reductions in FP32.  The accepted production extension is intentionally not
imported or modified by this module.
"""

from __future__ import annotations

from typing import Any

import torch


_NATIVE_Q1_ATTENTION: Any | None = None
_SUPPORTED_DTYPES = (torch.float32, torch.float16)


def _load_inline(**kwargs):
    # Keep compiler imports cold until a scout explicitly preloads the kernel.
    from torch.utils.cpp_extension import load_inline

    return load_inline(**kwargs)


def _require_supported_tensor(
    name: str,
    value: torch.Tensor,
    *,
    ndim: int,
    dtype: torch.dtype | None = None,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if value.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must be float32 or float16, got {value.dtype}")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if value.dim() != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {tuple(value.shape)}")


def _mask_for_native(
    attention_mask: torch.Tensor | None,
    *,
    device: torch.device,
    expected_numel: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("attention_mask must be a tensor or None")
    if attention_mask.device != device:
        raise ValueError("attention_mask must be on the same device as q/k/v")
    if not attention_mask.is_floating_point():
        raise TypeError("attention_mask must be floating point")
    if attention_mask.numel() != expected_numel:
        raise ValueError(
            "attention_mask must contain exactly one value per key position; "
            f"expected {expected_numel}, got {attention_mask.numel()}"
        )
    return attention_mask.reshape(-1).to(dtype=torch.float32).contiguous()


def _validate_q1(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.dtype, int, int, int]:
    _require_supported_tensor("query", query, ndim=4)
    _require_supported_tensor("key", key, ndim=4, dtype=query.dtype)
    _require_supported_tensor("value", value, ndim=4, dtype=query.dtype)
    if key.device != query.device or value.device != query.device:
        raise ValueError("query/key/value devices must match")
    if query.shape[0] != 1 or query.shape[2] != 1:
        raise ValueError("query must have shape [1, heads, 1, head_dim]")
    if key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError("key/value batch size must be 1")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if key.shape[1] != query.shape[1] or key.shape[3] != query.shape[3]:
        raise ValueError("query/key/value head shapes must match")
    heads = int(query.shape[1])
    key_length = int(key.shape[2])
    head_dim = int(query.shape[3])
    if heads <= 0 or key_length <= 0 or head_dim <= 0:
        raise ValueError("q1 attention dimensions must be positive")
    return query.dtype, heads, key_length, head_dim


def _validate_rope_cache_q1(
    qkv: torch.Tensor,
    cache_keys: torch.Tensor,
    cache_values: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_position: torch.Tensor,
    active_prefix_length: int,
) -> tuple[torch.dtype, int, int, int]:
    _require_supported_tensor("qkv", qkv, ndim=5)
    _require_supported_tensor("cache_keys", cache_keys, ndim=4, dtype=qkv.dtype)
    _require_supported_tensor("cache_values", cache_values, ndim=4, dtype=qkv.dtype)
    _require_supported_tensor("cos", cos, ndim=3, dtype=qkv.dtype)
    _require_supported_tensor("sin", sin, ndim=3, dtype=qkv.dtype)
    for name, tensor in (
        ("cache_keys", cache_keys),
        ("cache_values", cache_values),
        ("cos", cos),
        ("sin", sin),
    ):
        if tensor.device != qkv.device:
            raise ValueError(f"{name} must be on the qkv device")
    if qkv.shape[:3] != (1, 1, 3):
        raise ValueError("qkv must have shape [1, 1, 3, heads, head_dim]")
    if cache_keys.shape != cache_values.shape:
        raise ValueError("cache key/value shapes must match")
    if cache_keys.shape[0] != 1:
        raise ValueError("cache batch size must be 1")
    heads = int(qkv.shape[3])
    head_dim = int(qkv.shape[4])
    max_cache_length = int(cache_keys.shape[2])
    if cache_keys.shape[1] != heads or cache_keys.shape[3] != head_dim:
        raise ValueError("qkv and cache head shapes must match")
    if cos.shape != (1, 1, head_dim) or sin.shape != (1, 1, head_dim):
        raise ValueError("cos/sin must have shape [1, 1, head_dim]")
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    if not isinstance(cache_position, torch.Tensor):
        raise TypeError("cache_position must be a tensor")
    if cache_position.device != qkv.device:
        raise ValueError("cache_position must be on the qkv device")
    if cache_position.dtype != torch.long or cache_position.numel() != 1:
        raise ValueError("cache_position must be a scalar int64 tensor")
    if not 0 < int(active_prefix_length) <= max_cache_length:
        raise ValueError("active_prefix_length must be within the cache")
    return qkv.dtype, heads, max_cache_length, head_dim


def _load_native_q1_attention():
    global _NATIVE_Q1_ATTENTION
    if _NATIVE_Q1_ATTENTION is not None:
        return _NATIVE_Q1_ATTENTION

    cuda_source = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>

template<typename scalar_t>
__device__ __forceinline__ float load_float(const scalar_t* ptr, long long offset);

template<>
__device__ __forceinline__ float load_float<float>(const float* ptr, long long offset) {
    return ptr[offset];
}

template<>
__device__ __forceinline__ float load_float<__half>(const __half* ptr, long long offset) {
    return __half2float(ptr[offset]);
}

template<typename scalar_t>
__device__ __forceinline__ void store_float(scalar_t* ptr, long long offset, float value);

template<>
__device__ __forceinline__ void store_float<float>(float* ptr, long long offset, float value) {
    ptr[offset] = value;
}

template<>
__device__ __forceinline__ void store_float<__half>(__half* ptr, long long offset, float value) {
    ptr[offset] = __float2half_rn(value);
}

template<typename scalar_t, int BLOCK_SIZE>
__global__ void q1_attention_kernel(
        const scalar_t* __restrict__ q,
        const scalar_t* __restrict__ k,
        const scalar_t* __restrict__ v,
        const float* __restrict__ mask,
        scalar_t* __restrict__ out,
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
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    if (head >= heads) return;

    const scalar_t* qh = q + head * q_head_stride;
    const scalar_t* kh = k + head * k_head_stride;
    const scalar_t* vh = v + head * v_head_stride;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    extern __shared__ float shared_mem[];
    float* scores = shared_mem;
    float* scratch = shared_mem + kv_len;

    float max_score = -FLT_MAX;
    for (int pos = tid; pos < kv_len; pos += BLOCK_SIZE) {
        float score = 0.0f;
        const scalar_t* kp = kh + pos * k_len_stride;
        for (int dim = 0; dim < head_dim; ++dim) {
            score += load_float(qh, dim * q_dim_stride)
                    * load_float(kp, dim * k_dim_stride);
        }
        score *= scale;
        if (has_mask) score += mask[pos];
        scores[pos] = score;
        max_score = fmaxf(max_score, score);
    }

    scratch[tid] = max_score;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        __syncthreads();
    }
    max_score = scratch[0];

    float denominator = 0.0f;
    for (int pos = tid; pos < kv_len; pos += BLOCK_SIZE) {
        const float weight = expf(scores[pos] - max_score);
        scores[pos] = weight;
        denominator += weight;
    }
    scratch[tid] = denominator;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    denominator = scratch[0];

    for (int dim = tid; dim < head_dim; dim += BLOCK_SIZE) {
        float numerator = 0.0f;
        for (int pos = 0; pos < kv_len; ++pos) {
            numerator += scores[pos]
                    * load_float(vh, pos * v_len_stride + dim * v_dim_stride);
        }
        store_float(out, head * head_dim + dim, numerator / denominator);
    }
}

template<typename scalar_t, int BLOCK_SIZE>
__global__ void q1_rope_cache_attention_kernel(
        const scalar_t* __restrict__ qkv,
        scalar_t* __restrict__ cache_keys,
        scalar_t* __restrict__ cache_values,
        const scalar_t* __restrict__ cos,
        const scalar_t* __restrict__ sin,
        const int64_t* __restrict__ cache_position,
        const float* __restrict__ mask,
        scalar_t* __restrict__ out,
        int heads,
        int active_prefix_len,
        int head_dim,
        long long qkv_qkv_stride,
        long long qkv_head_stride,
        long long qkv_dim_stride,
        long long cache_head_stride,
        long long cache_len_stride,
        long long cache_dim_stride,
        long long cos_dim_stride,
        long long sin_dim_stride,
        int has_mask) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    if (head >= heads) return;

    const int write_pos = static_cast<int>(cache_position[0]);
    const int half_dim = head_dim / 2;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    extern __shared__ float shared_mem[];
    float* scores = shared_mem;
    float* scratch = shared_mem + active_prefix_len;
    float* query_shared = scratch + BLOCK_SIZE;

    for (int dim = tid; dim < head_dim; dim += BLOCK_SIZE) {
        const int rotated_dim = dim < half_dim ? dim + half_dim : dim - half_dim;
        const float sign = dim < half_dim ? -1.0f : 1.0f;
        const float cos_value = load_float(cos, dim * cos_dim_stride);
        const float sin_value = load_float(sin, dim * sin_dim_stride);
        const long long base = head * qkv_head_stride;
        const float q_raw = load_float(qkv, base + dim * qkv_dim_stride);
        const float q_rot = load_float(qkv, base + rotated_dim * qkv_dim_stride);
        const float k_raw = load_float(qkv, base + qkv_qkv_stride + dim * qkv_dim_stride);
        const float k_rot = load_float(qkv, base + qkv_qkv_stride + rotated_dim * qkv_dim_stride);
        const float value = load_float(qkv, base + 2 * qkv_qkv_stride + dim * qkv_dim_stride);
        const float query = q_raw * cos_value + sign * q_rot * sin_value;
        const float key = k_raw * cos_value + sign * k_rot * sin_value;
        query_shared[dim] = query;
        store_float(cache_keys,
            head * cache_head_stride + write_pos * cache_len_stride + dim * cache_dim_stride,
            key);
        store_float(cache_values,
            head * cache_head_stride + write_pos * cache_len_stride + dim * cache_dim_stride,
            value);
    }
    __syncthreads();

    float max_score = -FLT_MAX;
    for (int pos = tid; pos < active_prefix_len; pos += BLOCK_SIZE) {
        float score = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) {
            score += query_shared[dim] * load_float(
                cache_keys,
                head * cache_head_stride + pos * cache_len_stride + dim * cache_dim_stride);
        }
        score *= scale;
        if (has_mask) score += mask[pos];
        scores[pos] = score;
        max_score = fmaxf(max_score, score);
    }
    scratch[tid] = max_score;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        __syncthreads();
    }
    max_score = scratch[0];

    float denominator = 0.0f;
    for (int pos = tid; pos < active_prefix_len; pos += BLOCK_SIZE) {
        const float weight = expf(scores[pos] - max_score);
        scores[pos] = weight;
        denominator += weight;
    }
    scratch[tid] = denominator;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    denominator = scratch[0];

    for (int dim = tid; dim < head_dim; dim += BLOCK_SIZE) {
        float numerator = 0.0f;
        for (int pos = 0; pos < active_prefix_len; ++pos) {
            numerator += scores[pos] * load_float(
                cache_values,
                head * cache_head_stride + pos * cache_len_stride + dim * cache_dim_stride);
        }
        store_float(out, head * head_dim + dim, numerator / denominator);
    }
}

template<typename scalar_t>
void launch_q1(
        torch::Tensor q, torch::Tensor k, torch::Tensor v,
        const float* mask_ptr, int has_mask, torch::Tensor output) {
    constexpr int block_size = 128;
    const int heads = static_cast<int>(q.size(1));
    const int kv_len = static_cast<int>(k.size(2));
    const int head_dim = static_cast<int>(q.size(3));
    const size_t shared_bytes = static_cast<size_t>(kv_len + block_size) * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    q1_attention_kernel<scalar_t, block_size><<<heads, block_size, shared_bytes, stream>>>(
        reinterpret_cast<const scalar_t*>(q.data_ptr()),
        reinterpret_cast<const scalar_t*>(k.data_ptr()),
        reinterpret_cast<const scalar_t*>(v.data_ptr()),
        mask_ptr,
        reinterpret_cast<scalar_t*>(output.data_ptr()),
        heads, kv_len, head_dim,
        q.stride(1), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        has_mask);
}

torch::Tensor q1_attention(
        torch::Tensor q, torch::Tensor k, torch::Tensor v,
        c10::optional<torch::Tensor> mask) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "q/k/v dtype mismatch");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32 || q.scalar_type() == torch::kFloat16,
        "q/k/v must be fp32 or fp16");
    auto output = torch::empty({1, q.size(1), 1, q.size(3)}, q.options());
    torch::Tensor mask_contiguous;
    const float* mask_ptr = nullptr;
    int has_mask = 0;
    if (mask.has_value()) {
        mask_contiguous = mask.value().contiguous();
        TORCH_CHECK(mask_contiguous.is_cuda() && mask_contiguous.scalar_type() == torch::kFloat32,
            "mask must be CUDA fp32");
        mask_ptr = mask_contiguous.data_ptr<float>();
        has_mask = 1;
    }
    if (q.scalar_type() == torch::kFloat32) {
        launch_q1<float>(q, k, v, mask_ptr, has_mask, output);
    } else {
        launch_q1<__half>(q, k, v, mask_ptr, has_mask, output);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template<typename scalar_t>
void launch_rope_cache(
        torch::Tensor qkv, torch::Tensor cache_keys, torch::Tensor cache_values,
        torch::Tensor cos, torch::Tensor sin, torch::Tensor cache_position,
        const float* mask_ptr, int has_mask, int active_prefix_len,
        torch::Tensor output) {
    constexpr int block_size = 128;
    const int heads = static_cast<int>(qkv.size(3));
    const int head_dim = static_cast<int>(qkv.size(4));
    const size_t shared_bytes = static_cast<size_t>(active_prefix_len + block_size + head_dim) * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    q1_rope_cache_attention_kernel<scalar_t, block_size><<<heads, block_size, shared_bytes, stream>>>(
        reinterpret_cast<const scalar_t*>(qkv.data_ptr()),
        reinterpret_cast<scalar_t*>(cache_keys.data_ptr()),
        reinterpret_cast<scalar_t*>(cache_values.data_ptr()),
        reinterpret_cast<const scalar_t*>(cos.data_ptr()),
        reinterpret_cast<const scalar_t*>(sin.data_ptr()),
        cache_position.data_ptr<int64_t>(),
        mask_ptr,
        reinterpret_cast<scalar_t*>(output.data_ptr()),
        heads, active_prefix_len, head_dim,
        qkv.stride(2), qkv.stride(3), qkv.stride(4),
        cache_keys.stride(1), cache_keys.stride(2), cache_keys.stride(3),
        cos.stride(2), sin.stride(2), has_mask);
}

torch::Tensor q1_rope_cache_attention(
        torch::Tensor qkv, torch::Tensor cache_keys, torch::Tensor cache_values,
        torch::Tensor cos, torch::Tensor sin, torch::Tensor cache_position,
        c10::optional<torch::Tensor> mask, int64_t active_prefix_len) {
    TORCH_CHECK(qkv.is_cuda() && cache_keys.is_cuda() && cache_values.is_cuda(), "qkv/cache must be CUDA");
    TORCH_CHECK(qkv.scalar_type() == cache_keys.scalar_type()
            && qkv.scalar_type() == cache_values.scalar_type()
            && qkv.scalar_type() == cos.scalar_type()
            && qkv.scalar_type() == sin.scalar_type(), "qkv/cache/cos/sin dtype mismatch");
    TORCH_CHECK(qkv.scalar_type() == torch::kFloat32 || qkv.scalar_type() == torch::kFloat16,
        "qkv/cache must be fp32 or fp16");
    auto output = torch::empty({1, qkv.size(3), 1, qkv.size(4)}, qkv.options());
    torch::Tensor mask_contiguous;
    const float* mask_ptr = nullptr;
    int has_mask = 0;
    if (mask.has_value()) {
        mask_contiguous = mask.value().contiguous();
        TORCH_CHECK(mask_contiguous.is_cuda() && mask_contiguous.scalar_type() == torch::kFloat32,
            "mask must be CUDA fp32");
        mask_ptr = mask_contiguous.data_ptr<float>();
        has_mask = 1;
    }
    if (qkv.scalar_type() == torch::kFloat32) {
        launch_rope_cache<float>(qkv, cache_keys, cache_values, cos, sin, cache_position,
            mask_ptr, has_mask, active_prefix_len, output);
    } else {
        launch_rope_cache<__half>(qkv, cache_keys, cache_values, cos, sin, cache_position,
            mask_ptr, has_mask, active_prefix_len, output);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

    _NATIVE_Q1_ATTENTION = _load_inline(
        name="mapperatorinator_native_prefix_q1_attention_scout",
        cpp_sources=(
            "#include <torch/extension.h>\n"
            "torch::Tensor q1_attention(torch::Tensor, torch::Tensor, torch::Tensor, "
            "c10::optional<torch::Tensor>);\n"
            "torch::Tensor q1_rope_cache_attention(torch::Tensor, torch::Tensor, "
            "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
            "c10::optional<torch::Tensor>, int64_t);\n"
        ),
        cuda_sources=cuda_source,
        functions=["q1_attention", "q1_rope_cache_attention"],
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
    _, _, key_length, _ = _validate_q1(query, key, value)
    mask = _mask_for_native(
        attention_mask,
        device=query.device,
        expected_numel=key_length,
    )
    return _load_native_q1_attention().q1_attention(query, key, value, mask)


def native_q1_rope_cache_attention(
    qkv: torch.Tensor,
    cache_keys: torch.Tensor,
    cache_values: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_position: torch.Tensor,
    attention_mask: torch.Tensor | None,
    active_prefix_length: int,
) -> torch.Tensor:
    _validate_rope_cache_q1(
        qkv,
        cache_keys,
        cache_values,
        cos,
        sin,
        cache_position,
        active_prefix_length,
    )
    mask = _mask_for_native(
        attention_mask,
        device=qkv.device,
        expected_numel=int(active_prefix_length),
    )
    return _load_native_q1_attention().q1_rope_cache_attention(
        qkv,
        cache_keys,
        cache_values,
        cos,
        sin,
        cache_position,
        mask,
        int(active_prefix_length),
    )


__all__ = [
    "native_q1_attention",
    "native_q1_rope_cache_attention",
    "preload_native_q1_attention",
]
