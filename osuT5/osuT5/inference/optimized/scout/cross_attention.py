"""Verifier-only fixed-shape q1 cross-attention kernels for SM75.

The production q1 BMM path remains untouched.  This module is deliberately
import-cold and is loaded only by the bounded component profiler.
"""

from __future__ import annotations

from typing import Any

import torch


_CROSS_ATTENTION_EXTENSION: Any | None = None
_SUPPORTED_SPLITS = frozenset({2, 4, 8})
_HEADS = 12
_KV_LENGTH = 1024
_HEAD_DIM = 64


def _load_inline(**kwargs):
    from torch.utils.cpp_extension import load_inline

    return load_inline(**kwargs)


_CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>

namespace {

constexpr int kHeads = 12;
constexpr int kKvLength = 1024;
constexpr int kHeadDim = 64;
constexpr int kThreads = 128;

template<typename scalar_t>
__device__ __forceinline__ float load_kv(const scalar_t* pointer, int offset);

template<>
__device__ __forceinline__ float load_kv<float>(const float* pointer, int offset) {
    return pointer[offset];
}

template<>
__device__ __forceinline__ float load_kv<__half>(const __half* pointer, int offset) {
    return __half2float(pointer[offset]);
}

__device__ __forceinline__ float reduce_max(float value, float* scratch) {
    const int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        __syncthreads();
    }
    return scratch[0];
}

__device__ __forceinline__ float reduce_sum(float value, float* scratch) {
    const int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    return scratch[0];
}

template<typename kv_t>
__global__ void one_pass_online_softmax(
        const float* __restrict__ query,
        const kv_t* __restrict__ keys,
        const kv_t* __restrict__ values,
        float* __restrict__ output) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    if (head >= kHeads) return;
    const float* q = query + head * kHeadDim;
    const kv_t* k = keys + head * kKvLength * kHeadDim;
    const kv_t* v = values + head * kKvLength * kHeadDim;
    extern __shared__ float shared[];
    float* scores = shared;
    float* scratch = shared + kKvLength;

    float thread_max = -FLT_MAX;
    for (int position = tid; position < kKvLength; position += kThreads) {
        float score = 0.0f;
#pragma unroll
        for (int dimension = 0; dimension < kHeadDim; ++dimension) {
            score = fmaf(q[dimension], load_kv(k, position * kHeadDim + dimension), score);
        }
        score *= 0.125f;
        scores[position] = score;
        thread_max = fmaxf(thread_max, score);
    }
    const float maximum = reduce_max(thread_max, scratch);

    float thread_denominator = 0.0f;
    for (int position = tid; position < kKvLength; position += kThreads) {
        const float weight = expf(scores[position] - maximum);
        scores[position] = weight;
        thread_denominator += weight;
    }
    const float denominator = reduce_sum(thread_denominator, scratch);

    if (tid < kHeadDim) {
        float numerator = 0.0f;
        for (int position = 0; position < kKvLength; ++position) {
            numerator = fmaf(
                scores[position],
                load_kv(v, position * kHeadDim + tid),
                numerator);
        }
        output[head * kHeadDim + tid] = numerator / denominator;
    }
}

template<typename kv_t>
__global__ void split_online_softmax_partials(
        const float* __restrict__ query,
        const kv_t* __restrict__ keys,
        const kv_t* __restrict__ values,
        float* __restrict__ partial_outputs,
        float* __restrict__ partial_maxima,
        float* __restrict__ partial_denominators,
        int splits) {
    const int head = blockIdx.x / splits;
    const int split = blockIdx.x - head * splits;
    const int tid = threadIdx.x;
    if (head >= kHeads) return;
    const int segment_length = kKvLength / splits;
    const int segment_begin = split * segment_length;
    const float* q = query + head * kHeadDim;
    const kv_t* k = keys + head * kKvLength * kHeadDim;
    const kv_t* v = values + head * kKvLength * kHeadDim;
    extern __shared__ float shared[];
    float* scores = shared;
    float* scratch = shared + segment_length;

    float thread_max = -FLT_MAX;
    for (int local_position = tid; local_position < segment_length; local_position += kThreads) {
        const int position = segment_begin + local_position;
        float score = 0.0f;
#pragma unroll
        for (int dimension = 0; dimension < kHeadDim; ++dimension) {
            score = fmaf(q[dimension], load_kv(k, position * kHeadDim + dimension), score);
        }
        score *= 0.125f;
        scores[local_position] = score;
        thread_max = fmaxf(thread_max, score);
    }
    const float maximum = reduce_max(thread_max, scratch);

    float thread_denominator = 0.0f;
    for (int local_position = tid; local_position < segment_length; local_position += kThreads) {
        const float weight = expf(scores[local_position] - maximum);
        scores[local_position] = weight;
        thread_denominator += weight;
    }
    const float denominator = reduce_sum(thread_denominator, scratch);
    const int partial_index = head * splits + split;
    if (tid == 0) {
        partial_maxima[partial_index] = maximum;
        partial_denominators[partial_index] = denominator;
    }

    if (tid < kHeadDim) {
        float numerator = 0.0f;
        for (int local_position = 0; local_position < segment_length; ++local_position) {
            const int position = segment_begin + local_position;
            numerator = fmaf(
                scores[local_position],
                load_kv(v, position * kHeadDim + tid),
                numerator);
        }
        partial_outputs[(partial_index * kHeadDim) + tid] = numerator;
    }
}

__global__ void merge_online_softmax_partials(
        const float* __restrict__ partial_outputs,
        const float* __restrict__ partial_maxima,
        const float* __restrict__ partial_denominators,
        float* __restrict__ output,
        int splits) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    if (head >= kHeads) return;
    __shared__ float global_maximum;
    __shared__ float global_denominator;
    if (tid == 0) {
        float maximum = -FLT_MAX;
        for (int split = 0; split < splits; ++split) {
            maximum = fmaxf(maximum, partial_maxima[head * splits + split]);
        }
        float denominator = 0.0f;
        for (int split = 0; split < splits; ++split) {
            const int index = head * splits + split;
            denominator += expf(partial_maxima[index] - maximum)
                * partial_denominators[index];
        }
        global_maximum = maximum;
        global_denominator = denominator;
    }
    __syncthreads();
    if (tid < kHeadDim) {
        float numerator = 0.0f;
        for (int split = 0; split < splits; ++split) {
            const int index = head * splits + split;
            numerator += expf(partial_maxima[index] - global_maximum)
                * partial_outputs[index * kHeadDim + tid];
        }
        output[head * kHeadDim + tid] = numerator / global_denominator;
    }
}

void validate_inputs(const torch::Tensor& query, const torch::Tensor& keys, const torch::Tensor& values) {
    TORCH_CHECK(query.is_cuda() && keys.is_cuda() && values.is_cuda(), "q/k/v must be CUDA tensors");
    TORCH_CHECK(query.scalar_type() == torch::kFloat32, "query must be float32");
    TORCH_CHECK(
        keys.scalar_type() == torch::kFloat32 || keys.scalar_type() == torch::kFloat16,
        "cross KV storage must be float32 or float16");
    TORCH_CHECK(keys.scalar_type() == values.scalar_type(), "cross key/value dtypes must match");
    TORCH_CHECK(query.device() == keys.device() && query.device() == values.device(), "q/k/v devices must match");
    TORCH_CHECK(query.is_contiguous() && keys.is_contiguous() && values.is_contiguous(), "q/k/v must be contiguous");
    TORCH_CHECK(
        query.dim() == 4 && query.size(0) == 1 && query.size(1) == kHeads
        && query.size(2) == 1 && query.size(3) == kHeadDim,
        "query must have shape [1,12,1,64]");
    TORCH_CHECK(
        keys.dim() == 4 && keys.size(0) == 1 && keys.size(1) == kHeads
        && keys.size(2) == kKvLength && keys.size(3) == kHeadDim,
        "keys must have shape [1,12,1024,64]");
    TORCH_CHECK(values.sizes() == keys.sizes(), "values must match key shape");
}

template<typename kv_t>
void launch_one_pass(
        const torch::Tensor& query,
        const torch::Tensor& keys,
        const torch::Tensor& values,
        torch::Tensor& output) {
    const int shared_bytes = (kKvLength + kThreads) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream(query.get_device());
    one_pass_online_softmax<kv_t><<<kHeads, kThreads, shared_bytes, stream>>>(
        query.data_ptr<float>(),
        reinterpret_cast<const kv_t*>(keys.data_ptr()),
        reinterpret_cast<const kv_t*>(values.data_ptr()),
        output.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename kv_t>
void launch_split(
        const torch::Tensor& query,
        const torch::Tensor& keys,
        const torch::Tensor& values,
        torch::Tensor& partial_outputs,
        torch::Tensor& partial_maxima,
        torch::Tensor& partial_denominators,
        torch::Tensor& output,
        int splits) {
    const int segment_length = kKvLength / splits;
    const int shared_bytes = (segment_length + kThreads) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream(query.get_device());
    split_online_softmax_partials<kv_t><<<kHeads * splits, kThreads, shared_bytes, stream>>>(
        query.data_ptr<float>(),
        reinterpret_cast<const kv_t*>(keys.data_ptr()),
        reinterpret_cast<const kv_t*>(values.data_ptr()),
        partial_outputs.data_ptr<float>(),
        partial_maxima.data_ptr<float>(),
        partial_denominators.data_ptr<float>(),
        splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    merge_online_softmax_partials<<<kHeads, 64, 0, stream>>>(
        partial_outputs.data_ptr<float>(),
        partial_maxima.data_ptr<float>(),
        partial_denominators.data_ptr<float>(),
        output.data_ptr<float>(),
        splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

torch::Tensor cross_attention_one_pass(
        torch::Tensor query,
        torch::Tensor keys,
        torch::Tensor values) {
    validate_inputs(query, keys, values);
    c10::cuda::CUDAGuard guard(query.device());
    auto output = torch::empty({1, kHeads, 1, kHeadDim}, query.options());
    if (keys.scalar_type() == torch::kFloat32) {
        launch_one_pass<float>(query, keys, values, output);
    } else {
        launch_one_pass<__half>(query, keys, values, output);
    }
    return output;
}

torch::Tensor cross_attention_split(
        torch::Tensor query,
        torch::Tensor keys,
        torch::Tensor values,
        int64_t splits) {
    validate_inputs(query, keys, values);
    TORCH_CHECK(splits == 2 || splits == 4 || splits == 8, "splits must be one of 2, 4, or 8");
    c10::cuda::CUDAGuard guard(query.device());
    auto output = torch::empty({1, kHeads, 1, kHeadDim}, query.options());
    auto partial_outputs = torch::empty({kHeads, splits, kHeadDim}, query.options());
    auto partial_maxima = torch::empty({kHeads, splits}, query.options());
    auto partial_denominators = torch::empty({kHeads, splits}, query.options());
    if (keys.scalar_type() == torch::kFloat32) {
        launch_split<float>(
            query, keys, values, partial_outputs, partial_maxima,
            partial_denominators, output, static_cast<int>(splits));
    } else {
        launch_split<__half>(
            query, keys, values, partial_outputs, partial_maxima,
            partial_denominators, output, static_cast<int>(splits));
    }
    return output;
}
"""


def _load_cross_attention_extension():
    global _CROSS_ATTENTION_EXTENSION
    if _CROSS_ATTENTION_EXTENSION is None:
        _CROSS_ATTENTION_EXTENSION = _load_inline(
            name="mapperatorinator_cross_attention_split_kv_scout",
            cpp_sources="",
            cuda_sources=_CUDA_SOURCE,
            functions=["cross_attention_one_pass", "cross_attention_split"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            with_cuda=True,
            verbose=False,
        )
    return _CROSS_ATTENTION_EXTENSION


def _validate_inputs(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> None:
    for name, tensor in (("query", query), ("keys", keys), ("values", values)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if query.dtype != torch.float32:
        raise TypeError("query must use FP32 storage")
    if keys.dtype not in (torch.float32, torch.float16):
        raise TypeError("cross KV storage must be FP32 or FP16")
    if values.dtype != keys.dtype:
        raise TypeError("cross key/value storage dtypes must match")
    if query.device != keys.device or query.device != values.device:
        raise ValueError("query, keys, and values must use one CUDA device")
    if tuple(query.shape) != (1, _HEADS, 1, _HEAD_DIM):
        raise ValueError("query must have fixed shape [1,12,1,64]")
    if tuple(keys.shape) != (1, _HEADS, _KV_LENGTH, _HEAD_DIM):
        raise ValueError("keys must have fixed shape [1,12,1024,64]")
    if tuple(values.shape) != tuple(keys.shape):
        raise ValueError("values must match the fixed key shape")
    for name, tensor in (("query", query), ("keys", keys), ("values", values)):
        if tensor.device.type != "cuda":
            raise RuntimeError(f"{name} must be a CUDA tensor")


def preload_cross_attention_scout():
    return _load_cross_attention_extension()


def cross_attention_one_pass(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    _validate_inputs(query, keys, values)
    return _load_cross_attention_extension().cross_attention_one_pass(
        query,
        keys,
        values,
    )


def cross_attention_split(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    splits: int,
) -> torch.Tensor:
    _validate_inputs(query, keys, values)
    if isinstance(splits, bool) or not isinstance(splits, int):
        raise TypeError("splits must be an integer")
    if splits not in _SUPPORTED_SPLITS:
        raise ValueError("splits must be one of 2, 4, or 8")
    return _load_cross_attention_extension().cross_attention_split(
        query,
        keys,
        values,
        splits,
    )


__all__ = [
    "cross_attention_one_pass",
    "cross_attention_split",
    "preload_cross_attention_scout",
]
