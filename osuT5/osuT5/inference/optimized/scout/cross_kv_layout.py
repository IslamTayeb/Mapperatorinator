"""TRT-LLM ContiguousKv / XQA-shape cross-KV layout scout (component only).

Idea borrow only — not IFB, not paged-KV batch/server, not production wiring.

Selected-stack cross today stores separate FP32 K/V as ``[1, H, L, D]`` and
runs accepted/compiles ``_q1_bmm_cross_attention`` (per-step ``k.transpose`` +
two BMMs + softmax).  TensorRT-LLM XQA/cross paths emphasize:

1. static encoder length for generation-phase cross (no KV write)
2. contiguous / fused KV layout for decode streaming
3. single-request q_len=1 online-softmax decode shape

This scout stages those layout/kernel shapes on SM75 for fixed Whisper-style
``H=12, L=1024, D=64``.  Exactness is documented per candidate:

- ``pretransposed_k_bmm``: exact vs accepted BMM (layout-only)
- ``fused_online_*``: relaxed / documented drift (online softmax order)

Cold on import.  Do not present component timings as production TPS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


_CROSS_KV_LAYOUT_EXTENSION: Any | None = None
_HEADS = 12
_KV_LENGTH = 1024
_HEAD_DIM = 64
_SCALE = _HEAD_DIM**-0.5


@dataclass(frozen=True)
class PackedCrossKv:
    """Static decode-optimal cross-KV packs for one layer."""

    keys: torch.Tensor  # [1, H, L, D] source view (unchanged)
    values: torch.Tensor
    key_pretransposed: torch.Tensor  # [H, D, L] contiguous FP32
    fused_kv: torch.Tensor  # [H, L, 2, D] contiguous
    fused_kv_fp16: torch.Tensor  # [H, L, 2, D] contiguous FP16
    layout: str = "trtllm_contiguous_kv_hl2d"


def _load_inline(**kwargs):
    from torch.utils.cpp_extension import load_inline

    return load_inline(**kwargs)


_CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor fused_online_cross_attention(
    torch::Tensor query,
    torch::Tensor fused_kv);
"""


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
constexpr float kScale = 0.125f;  // 1/sqrt(64)

template <typename kv_t>
__device__ __forceinline__ float load_kv(const kv_t* pointer, int offset);

template <>
__device__ __forceinline__ float load_kv<float>(const float* pointer, int offset) {
    return pointer[offset];
}

template <>
__device__ __forceinline__ float load_kv<__half>(const __half* pointer, int offset) {
    return __half2float(pointer[offset]);
}

// XQA-shaped single-pass online softmax over fused ContiguousKv [H,L,2,D].
// One CTA per head; threads cooperatively stream KV positions once.
template <typename kv_t>
__global__ void fused_online_cross_kernel(
        const float* __restrict__ query,
        const kv_t* __restrict__ fused_kv,
        float* __restrict__ output) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    if (head >= kHeads) {
        return;
    }

    const float* q = query + head * kHeadDim;
    const kv_t* kv_base = fused_kv + head * kKvLength * 2 * kHeadDim;

    // Per-thread online stats over a strided subset of positions, then merge.
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float o_i[kHeadDim];
#pragma unroll
    for (int d = 0; d < kHeadDim; ++d) {
        o_i[d] = 0.0f;
    }

    for (int position = tid; position < kKvLength; position += kThreads) {
        const kv_t* k_ptr = kv_base + position * 2 * kHeadDim;
        const kv_t* v_ptr = k_ptr + kHeadDim;
        float score = 0.0f;
#pragma unroll
        for (int d = 0; d < kHeadDim; ++d) {
            score = fmaf(q[d], load_kv(k_ptr, d), score);
        }
        score *= kScale;
        const float m_new = fmaxf(m_i, score);
        const float alpha = (m_i == -FLT_MAX) ? 0.0f : expf(m_i - m_new);
        const float beta = expf(score - m_new);
#pragma unroll
        for (int d = 0; d < kHeadDim; ++d) {
            o_i[d] = fmaf(alpha, o_i[d], beta * load_kv(v_ptr, d));
        }
        l_i = fmaf(alpha, l_i, beta);
        m_i = m_new;
    }

    extern __shared__ float shared[];
    float* shared_m = shared;
    float* shared_l = shared + kThreads;
    float* shared_o = shared + 2 * kThreads;

    shared_m[tid] = m_i;
    shared_l[tid] = l_i;
#pragma unroll
    for (int d = 0; d < kHeadDim; ++d) {
        shared_o[tid * kHeadDim + d] = o_i[d];
    }
    __syncthreads();

    if (tid == 0) {
        float m = -FLT_MAX;
        float l = 0.0f;
        float o[kHeadDim];
#pragma unroll
        for (int d = 0; d < kHeadDim; ++d) {
            o[d] = 0.0f;
        }
        for (int lane = 0; lane < kThreads; ++lane) {
            const float m_lane = shared_m[lane];
            const float l_lane = shared_l[lane];
            const float m_new = fmaxf(m, m_lane);
            const float alpha = (m == -FLT_MAX) ? 0.0f : expf(m - m_new);
            const float beta = (m_lane == -FLT_MAX) ? 0.0f : expf(m_lane - m_new);
#pragma unroll
            for (int d = 0; d < kHeadDim; ++d) {
                o[d] = fmaf(alpha, o[d], beta * shared_o[lane * kHeadDim + d]);
            }
            l = fmaf(alpha, l, beta * l_lane);
            m = m_new;
        }
        const float inv = 1.0f / l;
#pragma unroll
        for (int d = 0; d < kHeadDim; ++d) {
            output[head * kHeadDim + d] = o[d] * inv;
        }
    }
}

void validate_inputs(const torch::Tensor& query, const torch::Tensor& fused_kv) {
    TORCH_CHECK(query.is_cuda() && fused_kv.is_cuda(), "query/fused_kv must be CUDA");
    TORCH_CHECK(query.scalar_type() == torch::kFloat32, "query must be FP32");
    TORCH_CHECK(
        fused_kv.scalar_type() == torch::kFloat32 || fused_kv.scalar_type() == torch::kFloat16,
        "fused_kv must be FP32 or FP16");
    TORCH_CHECK(query.is_contiguous() && fused_kv.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(
        query.dim() == 4 && query.size(0) == 1 && query.size(1) == kHeads
            && query.size(2) == 1 && query.size(3) == kHeadDim,
        "query must have shape [1,12,1,64]");
    TORCH_CHECK(
        fused_kv.dim() == 4 && fused_kv.size(0) == kHeads && fused_kv.size(1) == kKvLength
            && fused_kv.size(2) == 2 && fused_kv.size(3) == kHeadDim,
        "fused_kv must have shape [12,1024,2,64]");
}

template <typename kv_t>
void launch(
        const torch::Tensor& query,
        const torch::Tensor& fused_kv,
        torch::Tensor& output) {
    const int shared_bytes =
        (2 * kThreads + kThreads * kHeadDim) * static_cast<int>(sizeof(float));
    auto stream = at::cuda::getCurrentCUDAStream(query.get_device());
    fused_online_cross_kernel<kv_t><<<kHeads, kThreads, shared_bytes, stream>>>(
        query.data_ptr<float>(),
        reinterpret_cast<const kv_t*>(fused_kv.data_ptr()),
        output.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor fused_online_cross_attention(
        torch::Tensor query,
        torch::Tensor fused_kv) {
    validate_inputs(query, fused_kv);
    c10::cuda::CUDAGuard guard(query.device());
    auto output = torch::empty({1, kHeads, 1, kHeadDim}, query.options());
    if (fused_kv.scalar_type() == torch::kFloat32) {
        launch<float>(query, fused_kv, output);
    } else {
        launch<__half>(query, fused_kv, output);
    }
    return output;
}
"""


def _validate_source_kv(keys: torch.Tensor, values: torch.Tensor) -> None:
    for name, tensor in (("keys", keys), ("values", values)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must use FP32 storage for packing")
        if tensor.device.type != "cuda":
            raise RuntimeError(f"{name} must be a CUDA tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if tuple(keys.shape) != (1, _HEADS, _KV_LENGTH, _HEAD_DIM):
        raise ValueError("keys must have fixed shape [1,12,1024,64]")
    if tuple(values.shape) != tuple(keys.shape):
        raise ValueError("values must match the fixed key shape")
    if keys.device != values.device:
        raise ValueError("keys and values must share one device")


def pack_static_cross_kv(
    keys: torch.Tensor,
    values: torch.Tensor,
) -> PackedCrossKv:
    """Pack separate HF cross-KV into decode-optimal static layouts once."""

    _validate_source_kv(keys, values)
    # [H, L, D] -> [H, D, L] so QK^T is a contiguous GEMV/BMM without per-step transpose.
    key_pretransposed = (
        keys.reshape(_HEADS, _KV_LENGTH, _HEAD_DIM).transpose(1, 2).contiguous()
    )
    fused_kv = torch.stack(
        (
            keys.reshape(_HEADS, _KV_LENGTH, _HEAD_DIM),
            values.reshape(_HEADS, _KV_LENGTH, _HEAD_DIM),
        ),
        dim=2,
    ).contiguous()
    fused_kv_fp16 = fused_kv.to(dtype=torch.float16).contiguous()
    return PackedCrossKv(
        keys=keys,
        values=values,
        key_pretransposed=key_pretransposed,
        fused_kv=fused_kv,
        fused_kv_fp16=fused_kv_fp16,
    )


def pretransposed_k_bmm_cross_attention(
    query: torch.Tensor,
    pack: PackedCrossKv,
) -> torch.Tensor:
    """Exact accepted math with pre-transposed K (layout-only borrow)."""

    if query.dtype != torch.float32:
        raise TypeError("query must be FP32")
    if tuple(query.shape) != (1, _HEADS, 1, _HEAD_DIM):
        raise ValueError("query must have fixed shape [1,12,1,64]")
    q = query.reshape(_HEADS, 1, _HEAD_DIM)
    # key_pretransposed is [H, D, L]; bmm(q, k_t) with k_t already [H, D, L].
    scores = torch.bmm(q, pack.key_pretransposed) * _SCALE
    v = pack.values.reshape(_HEADS, _KV_LENGTH, _HEAD_DIM)
    output = torch.bmm(torch.softmax(scores, dim=-1), v)
    return output.view(1, _HEADS, 1, _HEAD_DIM)


def _load_extension():
    global _CROSS_KV_LAYOUT_EXTENSION
    if _CROSS_KV_LAYOUT_EXTENSION is None:
        _CROSS_KV_LAYOUT_EXTENSION = _load_inline(
            name="mapperatorinator_cross_kv_layout_scout",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["fused_online_cross_attention"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            with_cuda=True,
            verbose=False,
        )
    return _CROSS_KV_LAYOUT_EXTENSION


def preload_cross_kv_layout_scout():
    return _load_extension()


def fused_online_cross_attention(
    query: torch.Tensor,
    fused_kv: torch.Tensor,
) -> torch.Tensor:
    if query.dtype != torch.float32:
        raise TypeError("query must be FP32")
    if not query.is_contiguous() or not fused_kv.is_contiguous():
        raise ValueError("query and fused_kv must be contiguous")
    if tuple(query.shape) != (1, _HEADS, 1, _HEAD_DIM):
        raise ValueError("query must have fixed shape [1,12,1,64]")
    if tuple(fused_kv.shape) != (_HEADS, _KV_LENGTH, 2, _HEAD_DIM):
        raise ValueError("fused_kv must have fixed shape [12,1024,2,64]")
    if fused_kv.dtype not in (torch.float32, torch.float16):
        raise TypeError("fused_kv must be FP32 or FP16")
    return _load_extension().fused_online_cross_attention(query, fused_kv)


def scout_metadata() -> dict[str, Any]:
    return {
        "candidate": "trtllm_contiguous_kv_xqa_shape_cross",
        "borrow": (
            "TRT-LLM ContiguousKv + XQA-style single-request cross decode "
            "(static encoder length, fused KV, online softmax); not IFB/paged batch"
        ),
        "heads": _HEADS,
        "kv_length": _KV_LENGTH,
        "head_dim": _HEAD_DIM,
        "exactness": {
            "pretransposed_k_bmm": "exact-vs-accepted-bmm",
            "fused_online_fp32": "relaxed-documented-drift",
            "fused_online_fp16_kv": "relaxed-documented-drift",
        },
        "production_wiring": False,
    }


__all__ = [
    "PackedCrossKv",
    "fused_online_cross_attention",
    "pack_static_cross_kv",
    "preload_cross_kv_layout_scout",
    "pretransposed_k_bmm_cross_attention",
    "scout_metadata",
]
