"""Cold, opt-in bounded top-p sampler for the RTX 2080 Ti scout.

The extension deliberately accepts an already-generated counter threshold.  It
does not own RNG state and it is not installed in the inference runtime.  A
call returns an overflow flag whenever the bounded top-256 retained set cannot
faithfully represent the requested nucleus; callers must then use the existing
unbounded sampler.
"""

from __future__ import annotations

import math
from threading import Lock
from typing import Any

import torch
from torch.utils.cpp_extension import load_inline


TOP_K = 256
MAX_VOCAB = 4097
CHUNK_SIZE = 512
MAX_CHUNKS = (MAX_VOCAB + CHUNK_SIZE - 1) // CHUNK_SIZE

_EXTENSION: Any | None = None
_EXTENSION_LOCK = Lock()


_CPP_SOURCE = r"""
#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> top256_sample_cuda(
    torch::Tensor scores,
    torch::Tensor threshold,
    double top_p);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "top256_sample",
        &top256_sample_cuda,
        "Bounded stable top-256 top-p counter sampler (CUDA)");
}
"""


_CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kTopK = 256;
constexpr int kChunkSize = 512;
constexpr int kCandidatesPerChunk = kTopK + 1;
constexpr int kMaxVocab = 4097;
constexpr int kMaxChunks = (kMaxVocab + kChunkSize - 1) / kChunkSize;
constexpr float kBoundaryRelativeEpsilon = 1.0e-6f;

__device__ __forceinline__ bool comes_before(
        float left_score,
        int left_id,
        float right_score,
        int right_id) {
    return left_score > right_score ||
        (left_score == right_score && left_id < right_id);
}

__global__ void chunk_top257_kernel(
        const float* __restrict__ scores,
        int vocab_size,
        float* __restrict__ candidate_scores,
        int* __restrict__ candidate_ids,
        unsigned int* __restrict__ invalid_input) {
    __shared__ float shared_scores[kChunkSize];
    __shared__ int shared_ids[kChunkSize];

    const int lane = threadIdx.x;
    const int token_id = blockIdx.x * kChunkSize + lane;
    float score = -CUDART_INF_F;
    int stored_id = INT_MAX;
    if (token_id < vocab_size) {
        score = scores[token_id];
        if (isnan(score) || score == CUDART_INF_F) {
            atomicExch(invalid_input, 1u);
            score = -CUDART_INF_F;
        }
        stored_id = token_id;
    }
    shared_scores[lane] = score;
    shared_ids[lane] = stored_id;
    __syncthreads();

    // Stable score-descending, token-id-ascending bitonic ordering.  Padding
    // uses (-inf, INT_MAX), so it always sorts after real vocabulary entries.
    for (int width = 2; width <= kChunkSize; width <<= 1) {
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
            const int peer = lane ^ stride;
            if (peer > lane) {
                const bool descending = (lane & width) == 0;
                const float left_score = shared_scores[lane];
                const int left_id = shared_ids[lane];
                const float right_score = shared_scores[peer];
                const int right_id = shared_ids[peer];
                const bool swap = descending
                    ? comes_before(right_score, right_id, left_score, left_id)
                    : comes_before(left_score, left_id, right_score, right_id);
                if (swap) {
                    shared_scores[lane] = right_score;
                    shared_ids[lane] = right_id;
                    shared_scores[peer] = left_score;
                    shared_ids[peer] = left_id;
                }
            }
            __syncthreads();
        }
    }

    if (lane < kCandidatesPerChunk) {
        const int output = blockIdx.x * kCandidatesPerChunk + lane;
        candidate_scores[output] = shared_scores[lane];
        candidate_ids[output] = shared_ids[lane];
    }
}

__global__ void merge_nucleus_sample_kernel(
        const float* __restrict__ scores,
        int vocab_size,
        const float* __restrict__ candidate_scores,
        const int* __restrict__ candidate_ids,
        int chunks,
        const float* __restrict__ threshold,
        float top_p,
        const unsigned int* __restrict__ invalid_input,
        int64_t* __restrict__ token,
        int* __restrict__ kept_count_output,
        unsigned char* __restrict__ overflow_output) {
    __shared__ float retained_scores[kTopK + 1];
    __shared__ int retained_ids[kTopK + 1];
    __shared__ int cursors[kMaxChunks];
    __shared__ float scratch[kTopK];
    __shared__ int kept_count;
    __shared__ int selected_position;
    __shared__ float maximum;
    __shared__ float total_mass;
    __shared__ float retained_mass;
    __shared__ unsigned int overflow;

    const int lane = threadIdx.x;
    if (lane < kMaxChunks) {
        cursors[lane] = 0;
    }
    // Lane zero consumes every cursor in the serial k-way merge.  The
    // initialization is distributed across lanes, so the merge cannot begin
    // until all cursor writes are visible.
    __syncthreads();
    if (lane == 0) {
        overflow = *invalid_input != 0u ? 1u : 0u;
        kept_count = 0;
        selected_position = kTopK;
        for (int rank = 0; rank <= kTopK; ++rank) {
            int best_chunk = -1;
            float best_score = -CUDART_INF_F;
            int best_id = INT_MAX;
            for (int chunk = 0; chunk < chunks; ++chunk) {
                const int cursor = cursors[chunk];
                if (cursor >= kCandidatesPerChunk) {
                    continue;
                }
                const int offset = chunk * kCandidatesPerChunk + cursor;
                const float candidate_score = candidate_scores[offset];
                const int candidate_id = candidate_ids[offset];
                if (best_chunk < 0 || comes_before(
                        candidate_score,
                        candidate_id,
                        best_score,
                        best_id)) {
                    best_chunk = chunk;
                    best_score = candidate_score;
                    best_id = candidate_id;
                }
            }
            retained_scores[rank] = best_score;
            retained_ids[rank] = best_id;
            if (best_chunk >= 0) {
                ++cursors[best_chunk];
            }
        }
        maximum = retained_scores[0];
        if (!isfinite(maximum) || retained_ids[0] == INT_MAX) {
            overflow = 1u;
        }
        const float sample_threshold = threshold[0];
        if (!isfinite(sample_threshold) ||
                sample_threshold <= 0.0f || sample_threshold >= 1.0f) {
            overflow = 1u;
        }
    }
    __syncthreads();

    float local_total = 0.0f;
    if (!overflow) {
        for (int index = lane; index < vocab_size; index += kTopK) {
            local_total += expf(scores[index] - maximum);
        }
    }
    scratch[lane] = local_total;
    __syncthreads();
    for (int stride = kTopK / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            scratch[lane] += scratch[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        total_mass = scratch[0];
        if (!isfinite(total_mass) || total_mass <= 0.0f) {
            overflow = 1u;
        }
        float cumulative = 0.0f;
        const float target = top_p * total_mass;
        for (int rank = 0; rank < kTopK && !overflow; ++rank) {
            const float prior = cumulative;
            cumulative += expf(retained_scores[rank] - maximum);
            ++kept_count;
            const float tolerance = kBoundaryRelativeEpsilon * total_mass;
            if (fabsf(prior - target) <= tolerance ||
                    fabsf(cumulative - target) <= tolerance) {
                overflow = 1u;
                break;
            }
            if (cumulative >= target) {
                break;
            }
        }
        if (cumulative < target || kept_count <= 0 || kept_count > kTopK) {
            overflow = 1u;
        }
        // Transformers sorts the full row to choose a nucleus.  A finite tie
        // across the actual retained/removed cutoff can select different
        // token ids under a bounded stable ordering, so force exact fallback.
        if (!overflow && kept_count < vocab_size &&
                isfinite(retained_scores[kept_count - 1]) &&
                retained_scores[kept_count - 1] == retained_scores[kept_count]) {
            overflow = 1u;
        }
    }
    __syncthreads();

    // Sampling must use original vocabulary-id order, not probability order.
    // Sort only the retained bounded set by token id before constructing CDF.
    if (lane < kTopK) {
        if (lane >= kept_count) {
            retained_ids[lane] = INT_MAX;
            retained_scores[lane] = -CUDART_INF_F;
        }
    }
    __syncthreads();
    for (int width = 2; width <= kTopK; width <<= 1) {
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
            const int peer = lane ^ stride;
            if (peer > lane) {
                const bool ascending = (lane & width) == 0;
                const int left_id = retained_ids[lane];
                const int right_id = retained_ids[peer];
                const bool swap = ascending ? left_id > right_id : left_id < right_id;
                if (swap) {
                    const float left_score = retained_scores[lane];
                    retained_ids[lane] = right_id;
                    retained_scores[lane] = retained_scores[peer];
                    retained_ids[peer] = left_id;
                    retained_scores[peer] = left_score;
                }
            }
            __syncthreads();
        }
    }

    float weight = 0.0f;
    if (!overflow && lane < kept_count) {
        weight = expf(retained_scores[lane] - maximum);
    }
    scratch[lane] = weight;
    __syncthreads();
    for (int stride = kTopK / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            scratch[lane] += scratch[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        retained_mass = scratch[0];
        if (!isfinite(retained_mass) || retained_mass <= 0.0f) {
            overflow = 1u;
        }
    }
    __syncthreads();

    scratch[lane] = weight;
    __syncthreads();
    for (int offset = 1; offset < kTopK; offset <<= 1) {
        const float add = lane >= offset ? scratch[lane - offset] : 0.0f;
        __syncthreads();
        scratch[lane] += add;
        __syncthreads();
    }
    if (!overflow && lane < kept_count) {
        const float boundary = threshold[0] * retained_mass;
        const float tolerance = kBoundaryRelativeEpsilon * retained_mass;
        if (fabsf(scratch[lane] - boundary) <= tolerance) {
            atomicExch(&overflow, 1u);
        }
    }
    __syncthreads();
    if (!overflow && lane < kept_count) {
        if (scratch[lane] >= threshold[0] * retained_mass) {
            atomicMin(&selected_position, lane);
        }
    }
    __syncthreads();

    if (lane == 0) {
        if (selected_position >= kept_count && !overflow) {
            // Match searchsorted(...).clamp(max=vocab-1) roundoff behavior by
            // selecting the final retained token, not vocab_size-1.
            selected_position = kept_count - 1;
        }
        *kept_count_output = kept_count;
        *overflow_output = static_cast<unsigned char>(overflow != 0u);
        *token = overflow ? -1 : static_cast<int64_t>(retained_ids[selected_position]);
    }
}

}  // namespace

std::vector<torch::Tensor> top256_sample_cuda(
        torch::Tensor scores,
        torch::Tensor threshold,
        double top_p) {
    TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");
    TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must be FP32");
    TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");
    TORCH_CHECK(scores.dim() == 2 && scores.size(0) == 1,
        "scores must have shape [1, vocab]");
    TORCH_CHECK(scores.size(1) > 0 && scores.size(1) <= kMaxVocab,
        "vocabulary size must be inside [1, 4097]");
    TORCH_CHECK(threshold.is_cuda(), "threshold must be a CUDA tensor");
    TORCH_CHECK(threshold.scalar_type() == at::kFloat, "threshold must be FP32");
    TORCH_CHECK(threshold.is_contiguous() && threshold.numel() == 1,
        "threshold must be one contiguous scalar");
    TORCH_CHECK(threshold.device() == scores.device(),
        "scores and threshold must share one CUDA device");
    TORCH_CHECK(top_p > 0.0 && top_p < 1.0 && std::isfinite(top_p),
        "top_p must be finite and inside (0, 1)");

    c10::cuda::CUDAGuard device_guard(scores.device());
    const int vocab_size = static_cast<int>(scores.size(1));
    const int chunks = (vocab_size + kChunkSize - 1) / kChunkSize;
    auto float_options = scores.options();
    auto int_options = scores.options().dtype(torch::kInt32);
    auto byte_options = scores.options().dtype(torch::kUInt8);
    auto long_options = scores.options().dtype(torch::kInt64);
    auto candidate_scores = torch::empty(
        {chunks, kCandidatesPerChunk}, float_options);
    auto candidate_ids = torch::empty(
        {chunks, kCandidatesPerChunk}, int_options);
    auto invalid_input = torch::zeros({1}, int_options);
    auto token = torch::empty({1}, long_options);
    auto kept_count = torch::empty({1}, int_options);
    auto overflow = torch::empty({1}, byte_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    chunk_top257_kernel<<<chunks, kChunkSize, 0, stream>>>(
        scores.data_ptr<float>(),
        vocab_size,
        candidate_scores.data_ptr<float>(),
        candidate_ids.data_ptr<int>(),
        reinterpret_cast<unsigned int*>(invalid_input.data_ptr<int>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    merge_nucleus_sample_kernel<<<1, kTopK, 0, stream>>>(
        scores.data_ptr<float>(),
        vocab_size,
        candidate_scores.data_ptr<float>(),
        candidate_ids.data_ptr<int>(),
        chunks,
        threshold.data_ptr<float>(),
        static_cast<float>(top_p),
        reinterpret_cast<unsigned int*>(invalid_input.data_ptr<int>()),
        token.data_ptr<int64_t>(),
        kept_count.data_ptr<int>(),
        overflow.data_ptr<unsigned char>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {token, kept_count, overflow};
}
"""


def _validate_inputs(
    scores: torch.Tensor,
    threshold: torch.Tensor,
    top_p: float,
) -> None:
    if not isinstance(scores, torch.Tensor):
        raise TypeError("scores must be a tensor")
    if scores.dtype != torch.float32:
        raise TypeError("scores must use FP32 storage")
    if scores.ndim != 2 or scores.shape[0] != 1:
        raise ValueError("scores must have shape [1, vocab]")
    if not 0 < scores.shape[1] <= MAX_VOCAB:
        raise ValueError(f"vocabulary width must be inside [1, {MAX_VOCAB}]")
    if not scores.is_contiguous():
        raise ValueError("scores must be contiguous")
    if not isinstance(threshold, torch.Tensor):
        raise TypeError("threshold must be a tensor")
    if threshold.dtype != torch.float32 or threshold.numel() != 1:
        raise TypeError("threshold must be one FP32 scalar tensor")
    if not threshold.is_contiguous():
        raise ValueError("threshold must be contiguous")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise TypeError("top_p must be numeric")
    if not math.isfinite(float(top_p)) or not 0.0 < float(top_p) < 1.0:
        raise ValueError("top_p must be finite and inside (0, 1)")
    if scores.device.type != "cuda":
        raise RuntimeError("scores must be a CUDA tensor")
    if threshold.device != scores.device:
        raise RuntimeError("threshold must use the scores CUDA device")


def _load_top256_extension() -> Any:
    global _EXTENSION
    if _EXTENSION is None:
        with _EXTENSION_LOCK:
            if _EXTENSION is None:
                _EXTENSION = load_inline(
                    name="mapperatorinator_top256_sampler_scout_v1",
                    cpp_sources=_CPP_SOURCE,
                    cuda_sources=_CUDA_SOURCE,
                    functions=None,
                    extra_cuda_cflags=["-O3", "-lineinfo"],
                    with_cuda=True,
                    verbose=False,
                )
    return _EXTENSION


def preload_top256_sampler_extension() -> Any:
    """Build the opt-in extension before graph capture."""

    return _load_top256_extension()


def top256_sample(
    scores: torch.Tensor,
    threshold: torch.Tensor,
    *,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(token, kept_count, overflow)`` for one bounded sample."""

    _validate_inputs(scores, threshold, top_p)
    result = _load_top256_extension().top256_sample(
        scores,
        threshold,
        float(top_p),
    )
    if not isinstance(result, (list, tuple)) or len(result) != 3:
        raise RuntimeError("top-256 extension returned an invalid result")
    token, kept_count, overflow = result
    if (
        token.dtype != torch.long
        or token.shape != (1,)
        or kept_count.dtype != torch.int32
        or kept_count.shape != (1,)
        or overflow.dtype != torch.uint8
        or overflow.shape != (1,)
    ):
        raise RuntimeError("top-256 extension returned invalid output tensors")
    return token, kept_count, overflow


__all__ = [
    "CHUNK_SIZE",
    "MAX_CHUNKS",
    "MAX_VOCAB",
    "TOP_K",
    "preload_top256_sampler_extension",
    "top256_sample",
]
