"""Opt-in CUTLASS-seeking SM75 w8a8 projection GEMM scout.

Exactness class: relaxed / documented drift.  Cold on import; never wired into
production dispatch.  Prefer a CUTLASS SM75 w8a8 GEMM when headers are present
via MAPPERATORINATOR_CUTLASS_HOME or a standard CUTLASS install.  This DCC env
currently has no CUTLASS tree, so the scout stages a bounded SM75 signed-DP4A
w8a8 fallback (dynamic INT8 activations x INT8 weights, INT32 accumulate,
FP32 dequant/bias) and records cutlass_available=false with an explicit revisit
condition.  Do not present component timings as production throughput.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch

from .int8_mlp import Int8PackedLinear


_W8A8_PROJECTION_EXTENSION = None

# Candidate CUTLASS roots.  Header-only detection is intentional: we refuse to
# fabricate a CUTLASS win when only a vendor binary/shared library is present.
_CUTLASS_MARKER = Path("include/cutlass/gemm/device/gemm.h")


def _load_inline(**kwargs):
    from torch.utils.cpp_extension import load_inline

    return load_inline(**kwargs)


def probe_cutlass_availability(
    *,
    cutlass_home: str | None = None,
) -> dict[str, Any]:
    """Detect a usable CUTLASS source tree without building anything."""

    candidates: list[Path] = []
    env_home = cutlass_home or os.environ.get("MAPPERATORINATOR_CUTLASS_HOME")
    if env_home:
        candidates.append(Path(env_home).expanduser())
    candidates.extend(
        Path(value).expanduser()
        for value in (
            "/hpc/group/romerolab/imt11/software/cutlass",
            "/hpc/group/romerolab/imt11/cutlass",
            "~/cutlass",
            "/usr/local/cutlass",
            "/opt/cutlass",
        )
    )
    checked: list[str] = []
    for root in candidates:
        checked.append(str(root))
        marker = root / _CUTLASS_MARKER
        if marker.is_file():
            return {
                "cutlass_available": True,
                "cutlass_home": str(root.resolve()),
                "cutlass_marker": str(marker.resolve()),
                "backend": "cutlass_sm75_w8a8",
                "checked_roots": checked,
                "revisit_condition": None,
                "note": (
                    "CUTLASS headers found; SM75 w8a8 GEMM template wiring is still "
                    "required before any CUTLASS throughput claim."
                ),
            }
    return {
        "cutlass_available": False,
        "cutlass_home": None,
        "cutlass_marker": None,
        "backend": "sm75_signed_dp4a_w8a8_fallback",
        "checked_roots": checked,
        "revisit_condition": (
            "Install/vendor CUTLASS with include/cutlass/gemm/device/gemm.h "
            "(set MAPPERATORINATOR_CUTLASS_HOME) on SM75, then replace the "
            "signed-DP4A fallback with a true CUTLASS w8a8 GEMM and re-gate."
        ),
        "note": (
            "CUTLASS not available in this env; ranking uses the bounded SM75 "
            "signed-DP4A w8a8 fallback only. Not a CUTLASS win."
        ),
    }


_CPP_SOURCE = r"""
#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> dp4a_rmsnorm_linear(
    torch::Tensor input,
    torch::Tensor norm_weight,
    torch::Tensor weight,
    torch::Tensor weight_scale,
    c10::optional<torch::Tensor> bias,
    double eps,
    int64_t active_warps,
    int64_t persistent_ctas);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "dp4a_rmsnorm_linear",
        &dp4a_rmsnorm_linear,
        "Dynamic INT8 RMSNorm plus signed DP4A linear (CUDA)");
}
"""


_CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

constexpr int NORM_THREADS = 256;
constexpr int HIDDEN_DIM = 768;

template<int ACTIVE_WARPS>
__global__ void persistent_rmsnorm_signed_dp4a_kernel(
        const float* __restrict__ input,
        const float* __restrict__ norm_weight,
        const int8_t* __restrict__ weight,
        const float* __restrict__ weight_scale,
        const float* __restrict__ bias,
        float* __restrict__ output,
        int8_t* __restrict__ observed_quantized,
        float* __restrict__ observed_activation_scale,
        int output_dim,
        int has_bias,
        float eps) {
    const int tid = threadIdx.x;
    float local_square = 0.0f;
    float local_weighted_max = 0.0f;
    for (int index = tid; index < HIDDEN_DIM; index += blockDim.x) {
        const float value = input[index];
        local_square = fmaf(value, value, local_square);
        local_weighted_max = fmaxf(
            local_weighted_max,
            fabsf(value * norm_weight[index]));
    }

    __shared__ float square_scratch[NORM_THREADS];
    __shared__ float max_scratch[NORM_THREADS];
    __shared__ float inverse_rms;
    __shared__ float shared_scale;
    __shared__ __align__(4) int8_t shared_quantized[HIDDEN_DIM];
    square_scratch[tid] = local_square;
    max_scratch[tid] = local_weighted_max;
    __syncthreads();
    for (int stride = NORM_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            square_scratch[tid] += square_scratch[tid + stride];
            max_scratch[tid] = fmaxf(max_scratch[tid], max_scratch[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        inverse_rms = rsqrtf(
            square_scratch[0] / static_cast<float>(HIDDEN_DIM) + eps);
        const float normalized_max = max_scratch[0] * inverse_rms;
        shared_scale = normalized_max > 0.0f ? normalized_max / 127.0f : 1.0f;
        if (blockIdx.x == 0) {
            observed_activation_scale[0] = shared_scale;
        }
    }
    __syncthreads();

    for (int index = tid; index < HIDDEN_DIM; index += blockDim.x) {
        const float normalized =
            input[index] * norm_weight[index] * inverse_rms;
        int value = __float2int_rn(normalized / shared_scale);
        value = value < -127 ? -127 : (value > 127 ? 127 : value);
        shared_quantized[index] = static_cast<int8_t>(value);
        if (blockIdx.x == 0) {
            observed_quantized[index] = static_cast<int8_t>(value);
        }
    }
    __syncthreads();
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (warp < ACTIVE_WARPS) {
        const int packed_dim = HIDDEN_DIM / 4;
        const int* input4 = reinterpret_cast<const int*>(shared_quantized);
        for (int out = blockIdx.x * ACTIVE_WARPS + warp;
             out < output_dim;
             out += gridDim.x * ACTIVE_WARPS) {
            const int* weight4 = reinterpret_cast<const int*>(
                weight + static_cast<int64_t>(out) * HIDDEN_DIM);
            int accumulator = 0;
            for (int packed = lane; packed < packed_dim; packed += 32) {
                accumulator = __dp4a(
                    weight4[packed], input4[packed], accumulator);
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                accumulator += __shfl_down_sync(
                    0xffffffffu, accumulator, offset);
            }
            if (lane == 0) {
                float value = static_cast<float>(accumulator)
                    * weight_scale[out] * shared_scale;
                if (has_bias) {
                    value += bias[out];
                }
                output[out] = value;
            }
        }
    }
}

void check_launch_shape(int64_t active_warps, int64_t persistent_ctas,
        int output_dim) {
    TORCH_CHECK(active_warps == 4 || active_warps == 8,
        "active_warps must be 4 or 8");
    const int64_t maximum_ctas =
        (output_dim + active_warps - 1) / active_warps;
    TORCH_CHECK(persistent_ctas >= 1 && persistent_ctas <= maximum_ctas,
        "persistent_ctas must cover at least one and no more than the output rows");
    TORCH_CHECK(persistent_ctas <= 68,
        "persistent_ctas must not exceed the RTX 2080 Ti SM count");
}

std::vector<torch::Tensor> dp4a_rmsnorm_linear(
        torch::Tensor input,
        torch::Tensor norm_weight,
        torch::Tensor weight,
        torch::Tensor weight_scale,
        c10::optional<torch::Tensor> bias,
        double eps,
        int64_t active_warps,
        int64_t persistent_ctas) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32 && input.is_contiguous(),
        "input must be contiguous FP32");
    TORCH_CHECK(input.dim() == 3 && input.size(0) == 1 && input.size(1) == 1,
        "input must have shape [1, 1, hidden_dim]");
    TORCH_CHECK(norm_weight.is_cuda() && norm_weight.device() == input.device(),
        "norm weight must use the input CUDA device");
    TORCH_CHECK(norm_weight.scalar_type() == torch::kFloat32
        && norm_weight.is_contiguous() && norm_weight.dim() == 1,
        "norm weight must be contiguous FP32 [hidden_dim]");
    TORCH_CHECK(weight.is_cuda() && weight.device() == input.device(),
        "weight must use the input CUDA device");
    TORCH_CHECK(weight.scalar_type() == torch::kInt8 && weight.is_contiguous()
        && weight.dim() == 2,
        "weight must be contiguous INT8 [output_dim, hidden_dim]");
    TORCH_CHECK(weight_scale.is_cuda()
        && weight_scale.device() == input.device(),
        "weight scale must use the input CUDA device");
    TORCH_CHECK(weight_scale.scalar_type() == torch::kFloat32
        && weight_scale.is_contiguous() && weight_scale.dim() == 1,
        "weight scale must be contiguous FP32 [output_dim]");
    const int hidden_dim = static_cast<int>(input.size(2));
    const int output_dim = static_cast<int>(weight.size(0));
    TORCH_CHECK(hidden_dim == HIDDEN_DIM,
        "w8a8 projection scout requires hidden_dim=768");
    TORCH_CHECK(output_dim == 768 || output_dim == 2304,
        "w8a8 projection scout supports only cross-Q or self-QKV output shapes");
    TORCH_CHECK(norm_weight.numel() == hidden_dim,
        "norm weight/input dimension mismatch");
    TORCH_CHECK(weight.size(1) == hidden_dim,
        "weight/input dimension mismatch");
    TORCH_CHECK(weight_scale.numel() == output_dim,
        "weight scale/output dimension mismatch");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(weight.data_ptr<int8_t>()) % 4 == 0,
        "weight storage must be four-byte aligned for signed DP4A");
    check_launch_shape(active_warps, persistent_ctas, output_dim);

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        const auto value = bias.value();
        TORCH_CHECK(value.is_cuda() && value.device() == input.device(),
            "bias must use the input CUDA device");
        TORCH_CHECK(value.scalar_type() == torch::kFloat32
            && value.is_contiguous() && value.numel() == output_dim,
            "bias must be contiguous FP32 [output_dim]");
        bias_ptr = value.data_ptr<float>();
    }

    c10::cuda::CUDAGuard guard(input.device());
    auto quantized = torch::empty(
        {hidden_dim}, input.options().dtype(torch::kInt8));
    auto activation_scale = torch::empty({1}, input.options());
    auto output = torch::empty({1, 1, output_dim}, input.options());
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(quantized.data_ptr<int8_t>()) % 4 == 0,
        "quantized activation storage must be four-byte aligned for signed DP4A");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (active_warps == 4) {
        persistent_rmsnorm_signed_dp4a_kernel<4>
            <<<persistent_ctas, NORM_THREADS, 0, stream>>>(
                input.data_ptr<float>(), norm_weight.data_ptr<float>(),
                weight.data_ptr<int8_t>(), weight_scale.data_ptr<float>(),
                bias_ptr, output.data_ptr<float>(), quantized.data_ptr<int8_t>(),
                activation_scale.data_ptr<float>(), output_dim,
                bias_ptr != nullptr, static_cast<float>(eps));
    } else {
        persistent_rmsnorm_signed_dp4a_kernel<8>
            <<<persistent_ctas, NORM_THREADS, 0, stream>>>(
                input.data_ptr<float>(), norm_weight.data_ptr<float>(),
                weight.data_ptr<int8_t>(), weight_scale.data_ptr<float>(),
                bias_ptr, output.data_ptr<float>(), quantized.data_ptr<int8_t>(),
                activation_scale.data_ptr<float>(), output_dim,
                bias_ptr != nullptr, static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, quantized, activation_scale};
}
"""


def _load_w8a8_projection_extension():
    global _W8A8_PROJECTION_EXTENSION
    if _W8A8_PROJECTION_EXTENSION is None:
        _W8A8_PROJECTION_EXTENSION = _load_inline(
            name="mapperatorinator_sm75_w8a8_projection_scout_v1",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-gencode=arch=compute_75,code=sm_75",
            ],
            verbose=False,
        )
    return _W8A8_PROJECTION_EXTENSION


def preload_w8a8_projection_extension():
    """Build/load the verifier-only extension without changing dispatch."""

    return _load_w8a8_projection_extension()


def _require_fp32_cuda(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use FP32 storage")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return value


def _validate_inputs(
    input: Any,
    norm_weight: Any,
    pack: Any,
    *,
    active_warps: int,
    persistent_ctas: int,
) -> tuple[torch.Tensor, torch.Tensor, Int8PackedLinear]:
    input = _require_fp32_cuda(input, name="input")
    norm_weight = _require_fp32_cuda(norm_weight, name="norm weight")
    if input.dim() != 3 or tuple(input.shape[:2]) != (1, 1):
        raise ValueError("input must have shape [1, 1, hidden_dim]")
    hidden_dim = int(input.shape[2])
    if hidden_dim != 768:
        raise ValueError("w8a8 projection scout requires hidden_dim=768")
    if norm_weight.dim() != 1 or norm_weight.numel() != hidden_dim:
        raise ValueError("norm weight/input dimension mismatch")
    if norm_weight.device != input.device:
        raise ValueError("norm weight/input device mismatch")
    if not isinstance(pack, Int8PackedLinear):
        raise TypeError("pack must be an Int8PackedLinear")
    if pack.weight.dtype != torch.int8 or not pack.weight.is_contiguous():
        raise TypeError("pack weight must be contiguous INT8")
    if pack.weight.device != input.device:
        raise ValueError("pack weight/input device mismatch")
    if pack.weight.dim() != 2 or pack.weight.shape[1] != hidden_dim:
        raise ValueError("pack weight/input dimension mismatch")
    if pack.weight.shape[0] not in (768, 2304):
        raise ValueError("w8a8 pack must be cross-Q [768,768] or self-QKV [2304,768]")
    if pack.scale.dtype != torch.float32 or not pack.scale.is_contiguous():
        raise TypeError("pack scale must be contiguous FP32")
    if pack.scale.device != input.device or pack.scale.numel() != pack.weight.shape[0]:
        raise ValueError("pack scale/output dimension mismatch")
    if pack.bias is not None:
        _require_fp32_cuda(pack.bias, name="pack bias")
        if pack.bias.device != input.device or pack.bias.numel() != pack.weight.shape[0]:
            raise ValueError("pack bias/output dimension mismatch")
    if active_warps not in (4, 8):
        raise ValueError("active_warps must be 4 or 8")
    if pack.weight.data_ptr() % 4:
        raise ValueError("pack weight must be four-byte aligned for signed DP4A")
    maximum_ctas = (int(pack.weight.shape[0]) + active_warps - 1) // active_warps
    if persistent_ctas < 1 or persistent_ctas > min(68, maximum_ctas):
        raise ValueError("persistent_ctas must be within the SM75 output coverage bound")
    return input, norm_weight, pack


def validate_w8a8_pack(pack: Int8PackedLinear) -> dict[str, float | bool]:
    """Synchronously validate immutable pack contents before graph capture."""

    if not isinstance(pack, Int8PackedLinear):
        raise TypeError("pack must be an Int8PackedLinear")
    if pack.weight.dtype != torch.int8 or pack.weight.dim() != 2:
        raise TypeError("pack weight must be a two-dimensional INT8 tensor")
    if not pack.weight.is_contiguous() or pack.weight.shape[1] != 768:
        raise ValueError("pack weight must be contiguous with hidden_dim=768")
    if pack.weight.shape[0] not in (768, 2304):
        raise ValueError("pack weight must be cross-Q or self-QKV shaped")
    if pack.scale.dtype != torch.float32 or not pack.scale.is_contiguous():
        raise TypeError("pack scale must be contiguous FP32")
    if pack.scale.device != pack.weight.device or pack.scale.numel() != pack.weight.shape[0]:
        raise ValueError("pack scale/output ownership mismatch")
    if not bool(torch.isfinite(pack.scale).all().item()) or not bool(
        (pack.scale > 0).all().item()
    ):
        raise ValueError("pack scales must be finite and positive")
    contains_minus_128 = bool((pack.weight == -128).any().item())
    if contains_minus_128:
        raise ValueError("w8a8 pack must not contain -128")
    if pack.bias is not None:
        if pack.bias.dtype != torch.float32 or not pack.bias.is_contiguous():
            raise TypeError("pack bias must be contiguous FP32")
        if pack.bias.device != pack.weight.device or pack.bias.numel() != pack.weight.shape[0]:
            raise ValueError("pack bias/output ownership mismatch")
        if not bool(torch.isfinite(pack.bias).all().item()):
            raise ValueError("pack bias must be finite")
    return {
        "scale_min": float(pack.scale.min().item()),
        "scale_max": float(pack.scale.max().item()),
        "weight_contains_minus_128": contains_minus_128,
    }



def select_w8a8_backend(*, cutlass_home: str | None = None) -> dict[str, Any]:
    """Choose the staged scout backend and refuse false CUTLASS throughput claims."""

    probe = probe_cutlass_availability(cutlass_home=cutlass_home)
    if probe["cutlass_available"]:
        # Headers alone are insufficient: true CUTLASS GEMM templates are not wired yet.
        return {
            **probe,
            "backend": "sm75_signed_dp4a_w8a8_fallback",
            "cutlass_gemm_wired": False,
            "note": (
                "CUTLASS headers were found, but SM75 w8a8 GEMM templates are not "
                "wired in this scout; measuring the signed-DP4A w8a8 fallback only."
            ),
            "revisit_condition": (
                "Wire CUTLASS SM75 TensorOp/SIMT w8a8 GEMM for [out,768]x[768] "
                "one-token projections, prove exactness/drift gates, then compare "
                "against this DP4A fallback on a distinct 2080 Ti."
            ),
        }
    return {**probe, "cutlass_gemm_wired": False}


def w8a8_rmsnorm_linear_with_state(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    pack: Int8PackedLinear,
    *,
    eps: float,
    active_warps: int = 8,
    persistent_ctas: int = 68,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return FP32 output plus owned quantized activation and its FP32 scale.

    CUTLASS GEMM templates are not wired yet; this measures the SM75 signed-DP4A
    w8a8 fallback while recording cutlass availability via select_w8a8_backend.
    """

    if not math.isfinite(float(eps)) or float(eps) <= 0:
        raise ValueError("eps must be finite and positive")
    input, norm_weight, pack = _validate_inputs(
        input,
        norm_weight,
        pack,
        active_warps=active_warps,
        persistent_ctas=persistent_ctas,
    )
    capability = torch.cuda.get_device_capability(input.device)
    if capability != (7, 5):
        raise RuntimeError(f"w8a8 projection scout requires SM75, got sm_{capability[0]}{capability[1]}")
    values = _load_w8a8_projection_extension().dp4a_rmsnorm_linear(
        input,
        norm_weight,
        pack.weight,
        pack.scale,
        pack.bias,
        float(eps),
        int(active_warps),
        int(persistent_ctas),
    )
    if not isinstance(values, (tuple, list)) or len(values) != 3:
        raise RuntimeError("w8a8 extension returned an invalid output tuple")
    output, quantized, activation_scale = values
    expected_output_shape = (1, 1, int(pack.weight.shape[0]))
    if output.dtype != torch.float32 or tuple(output.shape) != expected_output_shape:
        raise RuntimeError("w8a8 output shape/dtype contract failed")
    if quantized.dtype != torch.int8 or tuple(quantized.shape) != (input.shape[2],):
        raise RuntimeError("w8a8 quantized activation shape/dtype contract failed")
    if activation_scale.dtype != torch.float32 or tuple(activation_scale.shape) != (1,):
        raise RuntimeError("w8a8 activation scale shape/dtype contract failed")
    if output.device != input.device or quantized.device != input.device:
        raise RuntimeError("w8a8 extension returned tensors on the wrong device")
    if activation_scale.device != input.device:
        raise RuntimeError("w8a8 activation scale uses the wrong device")
    return output, quantized, activation_scale


def w8a8_rmsnorm_linear(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    pack: Int8PackedLinear,
    *,
    eps: float,
    active_warps: int = 8,
    persistent_ctas: int = 68,
) -> torch.Tensor:
    """Return the FP32 projection output for the verifier-only candidate."""

    return w8a8_rmsnorm_linear_with_state(
        input,
        norm_weight,
        pack,
        eps=eps,
        active_warps=active_warps,
        persistent_ctas=persistent_ctas,
    )[0]


__all__ = [
    "probe_cutlass_availability",
    "select_w8a8_backend",
    "preload_w8a8_projection_extension",
    "validate_w8a8_pack",
    "w8a8_rmsnorm_linear",
    "w8a8_rmsnorm_linear_with_state",
]
