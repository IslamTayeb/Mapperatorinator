from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline


_NATIVE_Q1_ATTENTION = None
_SUPPORTED_DTYPES = (torch.float32, torch.float16)


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
    from .q1_mask_workspace import active_q1_mask_workspace

    workspace = active_q1_mask_workspace()
    if workspace is not None:
        return workspace.materialize(
            attention_mask,
            expected_numel=expected_numel,
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
    key_length = _validate_q1_inputs(query, key, value)
    mask = _native_mask(
        attention_mask,
        device=query.device,
        expected_numel=key_length,
    )
    extension = _load_native_q1_attention()
    return extension.q1_attention(query, key, value, mask)


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
    extension = _load_native_q1_attention()
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
