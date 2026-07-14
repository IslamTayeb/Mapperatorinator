from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline


_NATIVE_Q1_ATTENTION = None
_SUPPORTED_DTYPES = (torch.float32, torch.float16)
_SPLIT_KV_Q1_DTYPES = frozenset(_SUPPORTED_DTYPES)
_SPLIT_KV_Q1_PREFIXES = frozenset(range(192, 833, 64))
_SPLIT_KV_Q1_SPLITS = 8
_ACCEPTED_ROPE_CACHE_VARIANT = "accepted"
_SPLIT_KV_ROPE_CACHE_VARIANT = "split_kv_8"


def _device_capability(device: torch.device) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def _split_kv_q1_eligible(
    *,
    dtype: torch.dtype,
    device_type: str,
    active_prefix_length: int,
    capability: tuple[int, int],
) -> bool:
    return (
        dtype in _SPLIT_KV_Q1_DTYPES
        and device_type == "cuda"
        and active_prefix_length in _SPLIT_KV_Q1_PREFIXES
        and capability == (7, 5)
    )


def native_q1_rope_cache_attention_variant(
    qkv: torch.Tensor,
    active_prefix_length: int,
) -> str:
    """Select the dtype-generic SM75 split-KV path without changing fallbacks."""

    if not isinstance(qkv, torch.Tensor) or not qkv.is_cuda:
        return _ACCEPTED_ROPE_CACHE_VARIANT
    if not _split_kv_q1_eligible(
        dtype=qkv.dtype,
        device_type=qkv.device.type,
        active_prefix_length=active_prefix_length,
        capability=_device_capability(qkv.device),
    ):
        return _ACCEPTED_ROPE_CACHE_VARIANT
    return _SPLIT_KV_ROPE_CACHE_VARIANT


def _require_supported_tensor(
    name: str,
    value: torch.Tensor,
    *,
    ndim: int,
    dtype: torch.dtype | None = None,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must be float32 or float16, got {value.dtype}")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got {value.ndim}D")


def _validate_q1_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> int:
    _require_supported_tensor("query", query, ndim=4)
    _require_supported_tensor("key", key, ndim=4, dtype=query.dtype)
    _require_supported_tensor("value", value, ndim=4, dtype=query.dtype)
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise RuntimeError("query/key/value must be CUDA tensors")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query/key/value must use one CUDA device")
    if query.shape[0] != 1 or query.shape[2] != 1:
        raise ValueError("query must have shape [1, H, 1, D]")
    if key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError("key/value batch size must be 1")
    if key.shape[1] != query.shape[1] or value.shape[1] != query.shape[1]:
        raise ValueError("query/key/value head counts must match")
    if key.shape[2] != value.shape[2]:
        raise ValueError("key/value sequence lengths must match")
    if key.shape[3] != query.shape[3] or value.shape[3] != query.shape[3]:
        raise ValueError("query/key/value head dimensions must match")
    if key.shape[2] <= 0 or query.shape[1] <= 0 or query.shape[3] <= 0:
        raise ValueError("q1 attention dimensions must be positive")
    return int(key.shape[2])


def _validate_rope_cache_inputs(
    qkv: torch.Tensor,
    cache_keys: torch.Tensor,
    cache_values: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_position: torch.Tensor,
    active_prefix_length: int,
) -> None:
    _require_supported_tensor("qkv", qkv, ndim=5)
    _require_supported_tensor("cache_keys", cache_keys, ndim=4, dtype=qkv.dtype)
    _require_supported_tensor("cache_values", cache_values, ndim=4, dtype=qkv.dtype)
    _require_supported_tensor("cos", cos, ndim=3, dtype=qkv.dtype)
    _require_supported_tensor("sin", sin, ndim=3, dtype=qkv.dtype)
    if not all(tensor.is_cuda for tensor in (qkv, cache_keys, cache_values, cos, sin)):
        raise RuntimeError("qkv/cache/cos/sin must be CUDA tensors")
    tensors = (cache_keys, cache_values, cos, sin)
    if any(tensor.device != qkv.device for tensor in tensors):
        raise ValueError("qkv/cache/cos/sin must use one CUDA device")
    if qkv.shape[:3] != (1, 1, 3):
        raise ValueError("qkv must have shape [1, 1, 3, H, D]")
    heads = int(qkv.shape[3])
    head_dim = int(qkv.shape[4])
    if heads <= 0 or head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("qkv head count must be positive and head_dim must be positive/even")
    if cache_keys.shape[0] != 1 or cache_values.shape[0] != 1:
        raise ValueError("cache batch size must be 1")
    if cache_keys.shape[1] != heads or cache_values.shape[1] != heads:
        raise ValueError("qkv/cache head counts must match")
    if cache_keys.shape[2] != cache_values.shape[2]:
        raise ValueError("cache sequence lengths must match")
    if cache_keys.shape[3] != head_dim or cache_values.shape[3] != head_dim:
        raise ValueError("qkv/cache head dimensions must match")
    if cos.shape != (1, 1, head_dim) or sin.shape != (1, 1, head_dim):
        raise ValueError("cos/sin must have shape [1, 1, D]")
    if not isinstance(cache_position, torch.Tensor):
        raise TypeError("cache_position must be a tensor")
    if not cache_position.is_cuda:
        raise RuntimeError("cache_position must be a CUDA tensor")
    if cache_position.device != qkv.device:
        raise ValueError("cache_position must use the qkv CUDA device")
    if cache_position.dtype != torch.long or cache_position.numel() != 1:
        raise ValueError("cache_position must be a scalar int64 tensor")
    max_cache_length = int(cache_keys.shape[2])
    if active_prefix_length <= 0 or active_prefix_length > max_cache_length:
        raise ValueError("active_prefix_length must be within the cache capacity")


def _native_mask(
    attention_mask: torch.Tensor | None,
    *,
    device: torch.device,
    expected_numel: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("attention_mask must be a tensor")
    if not attention_mask.is_floating_point():
        raise TypeError("attention_mask must be floating-point")
    if attention_mask.device != device:
        raise ValueError("attention_mask must use the attention CUDA device")
    if attention_mask.numel() != expected_numel:
        raise ValueError(
            f"attention_mask must contain {expected_numel} elements, "
            f"got {attention_mask.numel()}"
        )
    return attention_mask.reshape(-1).to(dtype=torch.float32).contiguous()


def _load_native_q1_attention():
    global _NATIVE_Q1_ATTENTION
    if _NATIVE_Q1_ATTENTION is not None:
        return _NATIVE_Q1_ATTENTION

    cuda_source = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>

template<typename scalar_t>
struct Q1ScalarTraits;

template<>
struct Q1ScalarTraits<float> {
    __device__ __forceinline__ static float load(float value) {
        return value;
    }

    __device__ __forceinline__ static float store(float value) {
        return value;
    }
};

template<>
struct Q1ScalarTraits<__half> {
    __device__ __forceinline__ static float load(__half value) {
        return __half2float(value);
    }

    __device__ __forceinline__ static __half store(float value) {
        return __float2half_rn(value);
    }
};

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
    int head = blockIdx.x;
    int tid = threadIdx.x;
    if (head >= heads) {
        return;
    }

    using Traits = Q1ScalarTraits<scalar_t>;
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
        for (int d = 0; d < head_dim; ++d) {
            score += Traits::load(qh[d * q_dim_stride])
                    * Traits::load(kp[d * k_dim_stride]);
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
            numer += scores[pos]
                    * Traits::load(vh[pos * v_len_stride + dim * v_dim_stride]);
        }
        out[head * head_dim + dim] = Traits::store(numer / denom);
    }
}

torch::Tensor q1_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, c10::optional<torch::Tensor> mask) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
        "q/k/v dtype mismatch");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32 || q.scalar_type() == torch::kFloat16,
        "q/k/v must be fp32 or fp16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be 4D");
    TORCH_CHECK(q.size(0) == 1 && q.size(2) == 1, "q must have shape [1, H, 1, D]");
    TORCH_CHECK(k.size(0) == 1 && v.size(0) == 1, "k/v batch must be 1");
    TORCH_CHECK(k.size(1) == q.size(1) && v.size(1) == q.size(1), "head count mismatch");
    TORCH_CHECK(k.size(3) == q.size(3) && v.size(3) == q.size(3), "head dim mismatch");
    TORCH_CHECK(k.size(2) == v.size(2), "kv length mismatch");
    int heads = static_cast<int>(q.size(1));
    int kv_len = static_cast<int>(k.size(2));
    int head_dim = static_cast<int>(q.size(3));
    c10::cuda::CUDAGuard device_guard(q.device());
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
    if (q.scalar_type() == torch::kFloat32) {
        q1_attention_kernel<float, block_size><<<grid, block, shared_bytes, stream>>>(
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
    } else {
        q1_attention_kernel<__half, block_size><<<grid, block, shared_bytes, stream>>>(
            reinterpret_cast<const __half*>(q.data_ptr()),
            reinterpret_cast<const __half*>(k.data_ptr()),
            reinterpret_cast<const __half*>(v.data_ptr()),
            mask_ptr,
            reinterpret_cast<__half*>(output.data_ptr()),
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
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
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
        int max_cache_len,
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
    int head = blockIdx.x;
    int tid = threadIdx.x;
    if (head >= heads) {
        return;
    }

    using Traits = Q1ScalarTraits<scalar_t>;
    int write_pos = static_cast<int>(cache_position[0]);
    if (write_pos < 0 || write_pos >= active_prefix_len || write_pos >= max_cache_len) {
        asm("trap;");
        return;
    }
    const int half_dim = head_dim / 2;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    extern __shared__ float shared_mem[];
    float* scores = shared_mem;
    float* scratch = shared_mem + active_prefix_len;
    float* query_shared = scratch + BLOCK_SIZE;

    for (int dim = tid; dim < head_dim; dim += BLOCK_SIZE) {
        const int rotated_dim = dim < half_dim ? dim + half_dim : dim - half_dim;
        const float sign = dim < half_dim ? -1.0f : 1.0f;
        const float cos_v = Traits::load(cos[dim * cos_dim_stride]);
        const float sin_v = Traits::load(sin[dim * sin_dim_stride]);
        const long long q_base = head * qkv_head_stride;
        const float q_raw = Traits::load(
            qkv[q_base + 0 * qkv_qkv_stride + dim * qkv_dim_stride]);
        const float q_rot_raw = Traits::load(
            qkv[q_base + 0 * qkv_qkv_stride + rotated_dim * qkv_dim_stride]);
        const float k_raw = Traits::load(
            qkv[q_base + 1 * qkv_qkv_stride + dim * qkv_dim_stride]);
        const float k_rot_raw = Traits::load(
            qkv[q_base + 1 * qkv_qkv_stride + rotated_dim * qkv_dim_stride]);
        const float v_raw = Traits::load(
            qkv[q_base + 2 * qkv_qkv_stride + dim * qkv_dim_stride]);
        const float q_value = q_raw * cos_v + sign * q_rot_raw * sin_v;
        const float k_value = k_raw * cos_v + sign * k_rot_raw * sin_v;
        query_shared[dim] = q_value;
        cache_keys[
            head * cache_head_stride
            + write_pos * cache_len_stride
            + dim * cache_dim_stride
        ] = Traits::store(k_value);
        cache_values[
            head * cache_head_stride
            + write_pos * cache_len_stride
            + dim * cache_dim_stride
        ] = Traits::store(v_raw);
    }
    __syncthreads();

    float max_score = -FLT_MAX;
    for (int pos = tid; pos < active_prefix_len; pos += BLOCK_SIZE) {
        float score = 0.0f;
        const scalar_t* kp = cache_keys + head * cache_head_stride + pos * cache_len_stride;
        for (int dim = 0; dim < head_dim; ++dim) {
            score += query_shared[dim]
                    * Traits::load(kp[dim * cache_dim_stride]);
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
    for (int pos = tid; pos < active_prefix_len; pos += BLOCK_SIZE) {
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
        for (int pos = 0; pos < active_prefix_len; ++pos) {
            numer += scores[pos] * Traits::load(cache_values[
                head * cache_head_stride + pos * cache_len_stride + dim * cache_dim_stride
            ]);
        }
        out[head * head_dim + dim] = Traits::store(numer / denom);
    }
}

template<typename scalar_t, int BLOCK_SIZE>
__global__ void split_kv_prepare_rope_cache_kernel(
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
    const int half_dim = head_dim / 2;
    const long long q_base = head * qkv_head_stride;
    using Traits = Q1ScalarTraits<scalar_t>;
    for (int dim = tid; dim < head_dim; dim += BLOCK_SIZE) {
        const int rotated_dim = dim < half_dim ? dim + half_dim : dim - half_dim;
        const float sign = dim < half_dim ? -1.0f : 1.0f;
        const float cos_v = Traits::load(cos[dim * cos_dim_stride]);
        const float sin_v = Traits::load(sin[dim * sin_dim_stride]);
        const float q_raw = Traits::load(qkv[q_base + dim * qkv_dim_stride]);
        const float q_rot_raw = Traits::load(
            qkv[q_base + rotated_dim * qkv_dim_stride]);
        const float k_raw = Traits::load(qkv[
            q_base + qkv_qkv_stride + dim * qkv_dim_stride]);
        const float k_rot_raw = Traits::load(qkv[
            q_base + qkv_qkv_stride + rotated_dim * qkv_dim_stride]);
        const float v_raw = Traits::load(qkv[
            q_base + 2 * qkv_qkv_stride + dim * qkv_dim_stride]);
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
__global__ void split_kv_partial_kernel(
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
    const int begin = active_prefix_len * split / split_count;
    const int end = active_prefix_len * (split + 1) / split_count;
    const int length = end - begin;
    const int partial_index = head * split_count + split;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    extern __shared__ float shared[];
    float* scores = shared;
    float* reduction = shared + length;
    using Traits = Q1ScalarTraits<scalar_t>;

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
        if (tid < stride) reduction[tid] = fmaxf(
            reduction[tid], reduction[tid + stride]);
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
__global__ void split_kv_merge_kernel(
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
    using Traits = Q1ScalarTraits<scalar_t>;
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
            denominator += expf(partial_max[index] - maximum)
                * partial_denom[index];
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

torch::Tensor q1_rope_cache_attention(
        torch::Tensor qkv,
        torch::Tensor cache_keys,
        torch::Tensor cache_values,
        torch::Tensor cos,
        torch::Tensor sin,
        torch::Tensor cache_position,
        c10::optional<torch::Tensor> mask,
        int64_t active_prefix_len) {
    TORCH_CHECK(qkv.is_cuda() && cache_keys.is_cuda() && cache_values.is_cuda(), "qkv/cache tensors must be CUDA");
    TORCH_CHECK(cos.is_cuda() && sin.is_cuda() && cache_position.is_cuda(), "rope/cache_position tensors must be CUDA");
    TORCH_CHECK(qkv.scalar_type() == cache_keys.scalar_type()
            && qkv.scalar_type() == cache_values.scalar_type()
            && qkv.scalar_type() == cos.scalar_type()
            && qkv.scalar_type() == sin.scalar_type(),
        "qkv/cache/cos/sin dtype mismatch");
    TORCH_CHECK(qkv.scalar_type() == torch::kFloat32 || qkv.scalar_type() == torch::kFloat16,
        "qkv/cache/cos/sin must be fp32 or fp16");
    TORCH_CHECK(cache_position.scalar_type() == torch::kLong, "cache_position must be int64");
    TORCH_CHECK(qkv.dim() == 5, "qkv must have shape [1, 1, 3, H, D]");
    TORCH_CHECK(qkv.size(0) == 1 && qkv.size(1) == 1 && qkv.size(2) == 3, "qkv must have shape [1, 1, 3, H, D]");
    TORCH_CHECK(cache_keys.dim() == 4 && cache_values.dim() == 4, "cache tensors must have shape [1, H, L, D]");
    TORCH_CHECK(cache_keys.size(0) == 1 && cache_values.size(0) == 1, "cache batch must be 1");
    TORCH_CHECK(cache_keys.size(1) == qkv.size(3) && cache_values.size(1) == qkv.size(3), "cache head count mismatch");
    TORCH_CHECK(cache_keys.size(2) == cache_values.size(2), "cache length mismatch");
    TORCH_CHECK(cache_keys.size(3) == qkv.size(4) && cache_values.size(3) == qkv.size(4), "cache head dim mismatch");
    TORCH_CHECK(cos.dim() == 3 && sin.dim() == 3, "cos/sin must have shape [1, 1, D]");
    TORCH_CHECK(cos.size(0) == 1 && cos.size(1) == 1 && cos.size(2) == qkv.size(4), "cos shape mismatch");
    TORCH_CHECK(sin.size(0) == 1 && sin.size(1) == 1 && sin.size(2) == qkv.size(4), "sin shape mismatch");
    TORCH_CHECK(cache_position.numel() == 1, "cache_position must contain one element");

    int heads = static_cast<int>(qkv.size(3));
    int head_dim = static_cast<int>(qkv.size(4));
    int max_cache_len = static_cast<int>(cache_keys.size(2));
    TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even for RoPE");
    TORCH_CHECK(active_prefix_len > 0 && active_prefix_len <= max_cache_len, "invalid active_prefix_len");
    c10::cuda::CUDAGuard device_guard(qkv.device());
    auto output = torch::empty({1, heads, 1, head_dim}, qkv.options());

    const float* mask_ptr = nullptr;
    int has_mask = 0;
    torch::Tensor mask_contiguous;
    if (mask.has_value()) {
        mask_contiguous = mask.value().contiguous();
        TORCH_CHECK(mask_contiguous.is_cuda(), "mask must be CUDA");
        TORCH_CHECK(mask_contiguous.scalar_type() == torch::kFloat32, "mask must be fp32");
        TORCH_CHECK(mask_contiguous.numel() == active_prefix_len, "mask must have active_prefix_len elements");
        mask_ptr = mask_contiguous.data_ptr<float>();
        has_mask = 1;
    }

    constexpr int block_size = 128;
    dim3 grid(heads);
    dim3 block(block_size);
    size_t shared_bytes = static_cast<size_t>(active_prefix_len + block_size + head_dim) * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (qkv.scalar_type() == torch::kFloat32) {
        q1_rope_cache_attention_kernel<float, block_size><<<grid, block, shared_bytes, stream>>>(
            qkv.data_ptr<float>(),
            cache_keys.data_ptr<float>(),
            cache_values.data_ptr<float>(),
            cos.data_ptr<float>(),
            sin.data_ptr<float>(),
            cache_position.data_ptr<int64_t>(),
            mask_ptr,
            output.data_ptr<float>(),
            heads,
            static_cast<int>(active_prefix_len),
            max_cache_len,
            head_dim,
            static_cast<long long>(qkv.stride(2)),
            static_cast<long long>(qkv.stride(3)),
            static_cast<long long>(qkv.stride(4)),
            static_cast<long long>(cache_keys.stride(1)),
            static_cast<long long>(cache_keys.stride(2)),
            static_cast<long long>(cache_keys.stride(3)),
            static_cast<long long>(cos.stride(2)),
            static_cast<long long>(sin.stride(2)),
            has_mask
        );
    } else {
        q1_rope_cache_attention_kernel<__half, block_size><<<grid, block, shared_bytes, stream>>>(
            reinterpret_cast<const __half*>(qkv.data_ptr()),
            reinterpret_cast<__half*>(cache_keys.data_ptr()),
            reinterpret_cast<__half*>(cache_values.data_ptr()),
            reinterpret_cast<const __half*>(cos.data_ptr()),
            reinterpret_cast<const __half*>(sin.data_ptr()),
            cache_position.data_ptr<int64_t>(),
            mask_ptr,
            reinterpret_cast<__half*>(output.data_ptr()),
            heads,
            static_cast<int>(active_prefix_len),
            max_cache_len,
            head_dim,
            static_cast<long long>(qkv.stride(2)),
            static_cast<long long>(qkv.stride(3)),
            static_cast<long long>(qkv.stride(4)),
            static_cast<long long>(cache_keys.stride(1)),
            static_cast<long long>(cache_keys.stride(2)),
            static_cast<long long>(cache_keys.stride(3)),
            static_cast<long long>(cos.stride(2)),
            static_cast<long long>(sin.stride(2)),
            has_mask
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor split_kv_q1_rope_cache_attention(
        torch::Tensor qkv,
        torch::Tensor cache_keys,
        torch::Tensor cache_values,
        torch::Tensor cos,
        torch::Tensor sin,
        torch::Tensor cache_position,
        c10::optional<torch::Tensor> mask,
        int64_t active_prefix_len) {
    TORCH_CHECK(qkv.is_cuda() && cache_keys.is_cuda() && cache_values.is_cuda(),
        "split-KV qkv/cache tensors must be CUDA");
    TORCH_CHECK(cos.is_cuda() && sin.is_cuda() && cache_position.is_cuda(),
        "split-KV rope/cache_position tensors must be CUDA");
    TORCH_CHECK(qkv.scalar_type() == cache_keys.scalar_type()
            && qkv.scalar_type() == cache_values.scalar_type()
            && qkv.scalar_type() == cos.scalar_type()
            && qkv.scalar_type() == sin.scalar_type(),
        "split-KV qkv/cache/cos/sin dtype mismatch");
    TORCH_CHECK(qkv.scalar_type() == torch::kFloat32
            || qkv.scalar_type() == torch::kFloat16,
        "split-KV q1 requires fp32 or fp16 storage");
    TORCH_CHECK(cache_position.scalar_type() == torch::kLong,
        "split-KV cache_position must be int64");
    TORCH_CHECK(qkv.dim() == 5 && qkv.size(0) == 1 && qkv.size(1) == 1
            && qkv.size(2) == 3,
        "split-KV qkv must have shape [1, 1, 3, H, D]");
    TORCH_CHECK(cache_keys.dim() == 4 && cache_values.dim() == 4,
        "split-KV cache tensors must have shape [1, H, L, D]");
    TORCH_CHECK(cache_keys.size(0) == 1 && cache_values.size(0) == 1,
        "split-KV cache batch must be 1");
    TORCH_CHECK(cache_keys.size(1) == qkv.size(3)
            && cache_values.size(1) == qkv.size(3),
        "split-KV cache head count mismatch");
    TORCH_CHECK(cache_keys.size(2) == cache_values.size(2),
        "split-KV cache length mismatch");
    TORCH_CHECK(cache_keys.size(3) == qkv.size(4)
            && cache_values.size(3) == qkv.size(4),
        "split-KV cache head dim mismatch");
    TORCH_CHECK(cos.dim() == 3 && sin.dim() == 3
            && cos.size(0) == 1 && cos.size(1) == 1
            && sin.size(0) == 1 && sin.size(1) == 1
            && cos.size(2) == qkv.size(4) && sin.size(2) == qkv.size(4),
        "split-KV cos/sin must have shape [1, 1, D]");
    TORCH_CHECK(cache_position.numel() == 1,
        "split-KV cache_position must contain one element");

    const int heads = static_cast<int>(qkv.size(3));
    const int head_dim = static_cast<int>(qkv.size(4));
    const int max_cache_len = static_cast<int>(cache_keys.size(2));
    constexpr int split_count = 8;
    TORCH_CHECK(head_dim > 0 && head_dim % 2 == 0,
        "split-KV head_dim must be positive and even");
    TORCH_CHECK(active_prefix_len >= split_count
            && active_prefix_len <= max_cache_len,
        "split-KV active prefix is invalid");
    c10::cuda::CUDAGuard device_guard(qkv.device());
    cudaDeviceProp properties;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, qkv.get_device()));
    TORCH_CHECK(properties.major == 7 && properties.minor == 5,
        "split-KV q1 requires SM75");

    const float* mask_ptr = nullptr;
    int has_mask = 0;
    torch::Tensor mask_contiguous;
    if (mask.has_value()) {
        mask_contiguous = mask.value().contiguous();
        TORCH_CHECK(mask_contiguous.is_cuda()
                && mask_contiguous.scalar_type() == torch::kFloat32,
            "split-KV mask must be CUDA fp32");
        TORCH_CHECK(mask_contiguous.numel() == active_prefix_len,
            "split-KV mask must match active prefix");
        mask_ptr = mask_contiguous.data_ptr<float>();
        has_mask = 1;
    }

    auto fp32_options = qkv.options().dtype(torch::kFloat32);
    auto query = torch::empty({heads, head_dim}, fp32_options);
    auto partial_max = torch::empty({heads, split_count}, fp32_options);
    auto partial_denom = torch::empty({heads, split_count}, fp32_options);
    auto partial_numer = torch::empty(
        {heads, split_count, head_dim}, fp32_options);
    auto output = torch::empty({1, heads, 1, head_dim}, qkv.options());
    constexpr int block_size = 128;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int largest_split = static_cast<int>(
        (active_prefix_len + split_count - 1) / split_count);
    const size_t partial_shared = static_cast<size_t>(largest_split + block_size)
        * sizeof(float);

#define LAUNCH_SPLIT_KV(scalar_t, PTR) \
    split_kv_prepare_rope_cache_kernel<scalar_t, block_size><<< \
        heads, block_size, 0, stream>>>( \
            PTR(qkv.data_ptr()), PTR(cache_keys.data_ptr()), \
            PTR(cache_values.data_ptr()), PTR(cos.data_ptr()), \
            PTR(sin.data_ptr()), cache_position.data_ptr<int64_t>(), \
            query.data_ptr<float>(), heads, static_cast<int>(active_prefix_len), \
            max_cache_len, head_dim, static_cast<long long>(qkv.stride(2)), \
            static_cast<long long>(qkv.stride(3)), \
            static_cast<long long>(qkv.stride(4)), \
            static_cast<long long>(cache_keys.stride(1)), \
            static_cast<long long>(cache_keys.stride(2)), \
            static_cast<long long>(cache_keys.stride(3)), \
            static_cast<long long>(cos.stride(2)), \
            static_cast<long long>(sin.stride(2))); \
    split_kv_partial_kernel<scalar_t, block_size><<< \
        dim3(heads, split_count), block_size, partial_shared, stream>>>( \
            query.data_ptr<float>(), PTR(cache_keys.data_ptr()), \
            PTR(cache_values.data_ptr()), mask_ptr, \
            partial_max.data_ptr<float>(), partial_denom.data_ptr<float>(), \
            partial_numer.data_ptr<float>(), heads, \
            static_cast<int>(active_prefix_len), head_dim, split_count, \
            static_cast<long long>(cache_keys.stride(1)), \
            static_cast<long long>(cache_keys.stride(2)), \
            static_cast<long long>(cache_keys.stride(3)), has_mask); \
    split_kv_merge_kernel<scalar_t, block_size><<< \
        heads, block_size, 0, stream>>>( \
            partial_max.data_ptr<float>(), partial_denom.data_ptr<float>(), \
            partial_numer.data_ptr<float>(), PTR(output.data_ptr()), heads, \
            head_dim, split_count)

    if (qkv.scalar_type() == torch::kFloat32) {
#define FLOAT_PTR(value) reinterpret_cast<float*>(value)
        LAUNCH_SPLIT_KV(float, FLOAT_PTR);
#undef FLOAT_PTR
    } else {
#define HALF_PTR(value) reinterpret_cast<__half*>(value)
        LAUNCH_SPLIT_KV(__half, HALF_PTR);
#undef HALF_PTR
    }
#undef LAUNCH_SPLIT_KV
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
            "torch::Tensor q1_rope_cache_attention(torch::Tensor qkv, torch::Tensor cache_keys, "
            "torch::Tensor cache_values, torch::Tensor cos, torch::Tensor sin, "
            "torch::Tensor cache_position, c10::optional<torch::Tensor> mask, int64_t active_prefix_len);\n"
            "torch::Tensor split_kv_q1_rope_cache_attention(torch::Tensor qkv, "
            "torch::Tensor cache_keys, torch::Tensor cache_values, torch::Tensor cos, "
            "torch::Tensor sin, torch::Tensor cache_position, "
            "c10::optional<torch::Tensor> mask, int64_t active_prefix_len);\n"
        ),
        cuda_sources=cuda_source,
        functions=[
            "q1_attention",
            "q1_rope_cache_attention",
            "split_kv_q1_rope_cache_attention",
        ],
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
    key_length = _validate_q1_inputs(query, key, value)
    mask = _native_mask(
        attention_mask,
        device=query.device,
        expected_numel=key_length,
    )
    extension = _load_native_q1_attention()
    return extension.q1_attention(query, key, value, mask)


def _native_q1_rope_cache_attention_with_variant(
        qkv: torch.Tensor,
        cache_keys: torch.Tensor,
        cache_values: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache_position: torch.Tensor,
        attention_mask: torch.Tensor | None,
        active_prefix_length: int,
        *,
        variant: str,
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
    mask = _native_mask(
        attention_mask,
        device=qkv.device,
        expected_numel=int(active_prefix_length),
    )
    if variant not in {
        _ACCEPTED_ROPE_CACHE_VARIANT,
        _SPLIT_KV_ROPE_CACHE_VARIANT,
    }:
        raise ValueError(f"unsupported q1 RoPE/cache variant: {variant!r}")
    if variant == _SPLIT_KV_ROPE_CACHE_VARIANT and not _split_kv_q1_eligible(
        dtype=qkv.dtype,
        device_type=qkv.device.type,
        active_prefix_length=active_prefix_length,
        capability=_device_capability(qkv.device),
    ):
        raise RuntimeError(
            "split-KV q1 RoPE/cache requires FP32 or FP16 SM75 storage "
            "at a measured live prefix"
        )
    extension = _load_native_q1_attention()
    if variant == _SPLIT_KV_ROPE_CACHE_VARIANT:
        return extension.split_kv_q1_rope_cache_attention(
            qkv,
            cache_keys,
            cache_values,
            cos,
            sin,
            cache_position,
            mask,
            int(active_prefix_length),
        )
    return extension.q1_rope_cache_attention(
        qkv,
        cache_keys,
        cache_values,
        cos,
        sin,
        cache_position,
        mask,
        int(active_prefix_length),
    )


def accepted_q1_rope_cache_attention(
        qkv: torch.Tensor,
        cache_keys: torch.Tensor,
        cache_values: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache_position: torch.Tensor,
        attention_mask: torch.Tensor | None,
        active_prefix_length: int,
) -> torch.Tensor:
    """Run the accepted unsplit control for component verification."""

    return _native_q1_rope_cache_attention_with_variant(
        qkv,
        cache_keys,
        cache_values,
        cos,
        sin,
        cache_position,
        attention_mask,
        active_prefix_length,
        variant=_ACCEPTED_ROPE_CACHE_VARIANT,
    )


def split_kv_q1_rope_cache_attention(
        qkv: torch.Tensor,
        cache_keys: torch.Tensor,
        cache_values: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache_position: torch.Tensor,
        attention_mask: torch.Tensor | None,
        active_prefix_length: int,
) -> torch.Tensor:
    """Run the measured split-8 candidate for component verification."""

    return _native_q1_rope_cache_attention_with_variant(
        qkv,
        cache_keys,
        cache_values,
        cos,
        sin,
        cache_position,
        attention_mask,
        active_prefix_length,
        variant=_SPLIT_KV_ROPE_CACHE_VARIANT,
    )


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
    return _native_q1_rope_cache_attention_with_variant(
        qkv,
        cache_keys,
        cache_values,
        cos,
        sin,
        cache_position,
        attention_mask,
        active_prefix_length,
        variant=native_q1_rope_cache_attention_variant(
            qkv,
            active_prefix_length,
        ),
    )
