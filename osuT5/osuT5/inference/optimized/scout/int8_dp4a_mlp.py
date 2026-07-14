"""Opt-in SM75 dynamic-activation INT8 DP4A MLP scout.

Importing this module is cold.  The verifier extension is built only by the
explicit preload or candidate call.  The candidate retains FP32 input, norm,
bias, residual, GELU, dequantization, and output storage while using per-row
INT8 weights, per-token dynamic INT8 activations, and signed INT32 DP4A sums.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .int8_mlp import Int8MlpPack, Int8PackedLinear


_DP4A_MLP_EXTENSION = None


def _load_inline(**kwargs):
    from torch.utils.cpp_extension import load_inline

    return load_inline(**kwargs)


_CPP_SOURCE = r"""
#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> dp4a_mlp_residual(
    torch::Tensor input,
    torch::Tensor norm_weight,
    torch::Tensor fc1_weight,
    torch::Tensor fc1_scale,
    c10::optional<torch::Tensor> fc1_bias,
    torch::Tensor fc2_weight,
    torch::Tensor fc2_scale,
    c10::optional<torch::Tensor> fc2_bias,
    double eps,
    int64_t active_warps,
    int64_t persistent_ctas);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "dp4a_mlp_residual",
        &dp4a_mlp_residual,
        "Dynamic INT8 signed-DP4A one-token MLP residual (CUDA)");
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

constexpr int THREADS = 256;
constexpr int HIDDEN_DIM = 768;
constexpr int INTERMEDIATE_DIM = 3072;

template<int ACTIVE_WARPS>
__global__ void rmsnorm_quant_dp4a_fc1_gelu_kernel(
        const float* __restrict__ input,
        const float* __restrict__ norm_weight,
        const int8_t* __restrict__ weight,
        const float* __restrict__ weight_scale,
        const float* __restrict__ bias,
        float* __restrict__ activated,
        int8_t* __restrict__ observed_quantized,
        float* __restrict__ observed_activation_scale,
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

    __shared__ float square_scratch[THREADS];
    __shared__ float max_scratch[THREADS];
    __shared__ float inverse_rms;
    __shared__ float activation_scale;
    __shared__ __align__(4) int8_t quantized[HIDDEN_DIM];
    square_scratch[tid] = local_square;
    max_scratch[tid] = local_weighted_max;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            square_scratch[tid] += square_scratch[tid + stride];
            max_scratch[tid] = fmaxf(
                max_scratch[tid], max_scratch[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        inverse_rms = rsqrtf(
            square_scratch[0] / static_cast<float>(HIDDEN_DIM) + eps);
        const float maximum = max_scratch[0] * inverse_rms;
        activation_scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
        if (blockIdx.x == 0) {
            observed_activation_scale[0] = activation_scale;
        }
    }
    __syncthreads();

    for (int index = tid; index < HIDDEN_DIM; index += blockDim.x) {
        const float normalized =
            input[index] * norm_weight[index] * inverse_rms;
        int value = __float2int_rn(normalized / activation_scale);
        value = value < -127 ? -127 : (value > 127 ? 127 : value);
        quantized[index] = static_cast<int8_t>(value);
        if (blockIdx.x == 0) {
            observed_quantized[index] = static_cast<int8_t>(value);
        }
    }
    __syncthreads();

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (warp < ACTIVE_WARPS) {
        const int packed_dim = HIDDEN_DIM / 4;
        const int* input4 = reinterpret_cast<const int*>(quantized);
        for (int out = blockIdx.x * ACTIVE_WARPS + warp;
             out < INTERMEDIATE_DIM;
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
                    * weight_scale[out] * activation_scale;
                if (has_bias) {
                    value += bias[out];
                }
                value = 0.5f * value *
                    (1.0f + erff(value * 0.7071067811865475244f));
                activated[out] = value;
            }
        }
    }
}

template<int ACTIVE_WARPS>
__global__ void quant_dp4a_fc2_residual_kernel(
        const float* __restrict__ activated,
        const int8_t* __restrict__ weight,
        const float* __restrict__ weight_scale,
        const float* __restrict__ bias,
        const float* __restrict__ residual,
        float* __restrict__ output,
        int8_t* __restrict__ observed_quantized,
        float* __restrict__ observed_activation_scale,
        int has_bias) {
    const int tid = threadIdx.x;
    float local_max = 0.0f;
    for (int index = tid; index < INTERMEDIATE_DIM; index += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(activated[index]));
    }

    __shared__ float max_scratch[THREADS];
    __shared__ float activation_scale;
    __shared__ __align__(4) int8_t quantized[INTERMEDIATE_DIM];
    max_scratch[tid] = local_max;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            max_scratch[tid] = fmaxf(
                max_scratch[tid], max_scratch[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        activation_scale =
            max_scratch[0] > 0.0f ? max_scratch[0] / 127.0f : 1.0f;
        if (blockIdx.x == 0) {
            observed_activation_scale[0] = activation_scale;
        }
    }
    __syncthreads();

    for (int index = tid; index < INTERMEDIATE_DIM; index += blockDim.x) {
        int value = __float2int_rn(activated[index] / activation_scale);
        value = value < -127 ? -127 : (value > 127 ? 127 : value);
        quantized[index] = static_cast<int8_t>(value);
        if (blockIdx.x == 0) {
            observed_quantized[index] = static_cast<int8_t>(value);
        }
    }
    __syncthreads();

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (warp < ACTIVE_WARPS) {
        const int packed_dim = INTERMEDIATE_DIM / 4;
        const int* input4 = reinterpret_cast<const int*>(quantized);
        for (int out = blockIdx.x * ACTIVE_WARPS + warp;
             out < HIDDEN_DIM;
             out += gridDim.x * ACTIVE_WARPS) {
            const int* weight4 = reinterpret_cast<const int*>(
                weight + static_cast<int64_t>(out) * INTERMEDIATE_DIM);
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
                    * weight_scale[out] * activation_scale;
                if (has_bias) {
                    value += bias[out];
                }
                output[out] = value + residual[out];
            }
        }
    }
}

void check_launch_shape(int64_t active_warps, int64_t persistent_ctas) {
    TORCH_CHECK(active_warps == 4 || active_warps == 8,
        "active_warps must be 4 or 8");
    TORCH_CHECK(persistent_ctas >= 1 && persistent_ctas <= 68,
        "persistent_ctas must be between one and the RTX 2080 Ti SM count");
    TORCH_CHECK(
        persistent_ctas <= (HIDDEN_DIM + active_warps - 1) / active_warps,
        "persistent_ctas must not exceed FC2 output coverage");
}

const float* check_bias(
        const c10::optional<torch::Tensor>& bias,
        const torch::Tensor& input,
        int outputs,
        const char* name) {
    if (!bias.has_value()) {
        return nullptr;
    }
    const auto value = bias.value();
    TORCH_CHECK(value.is_cuda() && value.device() == input.device(),
        name, " bias must use the input CUDA device");
    TORCH_CHECK(value.scalar_type() == torch::kFloat32
        && value.is_contiguous() && value.numel() == outputs,
        name, " bias must be contiguous FP32 with one value per output");
    return value.data_ptr<float>();
}

void check_pack(
        const torch::Tensor& weight,
        const torch::Tensor& scale,
        const torch::Tensor& input,
        int rows,
        int columns,
        const char* name) {
    TORCH_CHECK(weight.is_cuda() && weight.device() == input.device(),
        name, " weight must use the input CUDA device");
    TORCH_CHECK(weight.scalar_type() == torch::kInt8 && weight.is_contiguous(),
        name, " weight must be contiguous INT8");
    TORCH_CHECK(weight.dim() == 2 && weight.size(0) == rows
        && weight.size(1) == columns,
        name, " weight has the wrong fixed MLP shape");
    TORCH_CHECK(scale.is_cuda() && scale.device() == input.device(),
        name, " scale must use the input CUDA device");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32
        && scale.is_contiguous() && scale.numel() == rows,
        name, " scale must be contiguous FP32 with one value per output");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(weight.data_ptr<int8_t>()) % 4 == 0,
        name, " weight must be four-byte aligned for signed DP4A");
}

std::vector<torch::Tensor> dp4a_mlp_residual(
        torch::Tensor input,
        torch::Tensor norm_weight,
        torch::Tensor fc1_weight,
        torch::Tensor fc1_scale,
        c10::optional<torch::Tensor> fc1_bias,
        torch::Tensor fc2_weight,
        torch::Tensor fc2_scale,
        c10::optional<torch::Tensor> fc2_bias,
        double eps,
        int64_t active_warps,
        int64_t persistent_ctas) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32 && input.is_contiguous(),
        "input must be contiguous FP32");
    TORCH_CHECK(input.dim() == 3 && input.size(0) == 1 && input.size(1) == 1
        && input.size(2) == HIDDEN_DIM,
        "input must have fixed shape [1, 1, 768]");
    TORCH_CHECK(norm_weight.is_cuda() && norm_weight.device() == input.device(),
        "norm weight must use the input CUDA device");
    TORCH_CHECK(norm_weight.scalar_type() == torch::kFloat32
        && norm_weight.is_contiguous() && norm_weight.numel() == HIDDEN_DIM,
        "norm weight must be contiguous FP32 [768]");
    check_pack(
        fc1_weight, fc1_scale, input,
        INTERMEDIATE_DIM, HIDDEN_DIM, "fc1");
    check_pack(
        fc2_weight, fc2_scale, input,
        HIDDEN_DIM, INTERMEDIATE_DIM, "fc2");
    check_launch_shape(active_warps, persistent_ctas);
    const float* fc1_bias_ptr = check_bias(
        fc1_bias, input, INTERMEDIATE_DIM, "fc1");
    const float* fc2_bias_ptr = check_bias(
        fc2_bias, input, HIDDEN_DIM, "fc2");

    c10::cuda::CUDAGuard guard(input.device());
    auto activated = torch::empty(
        {1, 1, INTERMEDIATE_DIM}, input.options());
    auto output = torch::empty_like(input);
    auto fc1_quantized = torch::empty(
        {HIDDEN_DIM}, input.options().dtype(torch::kInt8));
    auto fc1_activation_scale = torch::empty({1}, input.options());
    auto fc2_quantized = torch::empty(
        {INTERMEDIATE_DIM}, input.options().dtype(torch::kInt8));
    auto fc2_activation_scale = torch::empty({1}, input.options());
    TORCH_CHECK(reinterpret_cast<uintptr_t>(fc1_quantized.data_ptr<int8_t>()) % 4 == 0,
        "FC1 activation storage must be four-byte aligned for signed DP4A");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(fc2_quantized.data_ptr<int8_t>()) % 4 == 0,
        "FC2 activation storage must be four-byte aligned for signed DP4A");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (active_warps == 4) {
        rmsnorm_quant_dp4a_fc1_gelu_kernel<4>
            <<<persistent_ctas, THREADS, 0, stream>>>(
                input.data_ptr<float>(), norm_weight.data_ptr<float>(),
                fc1_weight.data_ptr<int8_t>(), fc1_scale.data_ptr<float>(),
                fc1_bias_ptr, activated.data_ptr<float>(),
                fc1_quantized.data_ptr<int8_t>(),
                fc1_activation_scale.data_ptr<float>(),
                fc1_bias_ptr != nullptr, static_cast<float>(eps));
    } else {
        rmsnorm_quant_dp4a_fc1_gelu_kernel<8>
            <<<persistent_ctas, THREADS, 0, stream>>>(
                input.data_ptr<float>(), norm_weight.data_ptr<float>(),
                fc1_weight.data_ptr<int8_t>(), fc1_scale.data_ptr<float>(),
                fc1_bias_ptr, activated.data_ptr<float>(),
                fc1_quantized.data_ptr<int8_t>(),
                fc1_activation_scale.data_ptr<float>(),
                fc1_bias_ptr != nullptr, static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (active_warps == 4) {
        quant_dp4a_fc2_residual_kernel<4>
            <<<persistent_ctas, THREADS, 0, stream>>>(
                activated.data_ptr<float>(), fc2_weight.data_ptr<int8_t>(),
                fc2_scale.data_ptr<float>(), fc2_bias_ptr,
                input.data_ptr<float>(), output.data_ptr<float>(),
                fc2_quantized.data_ptr<int8_t>(),
                fc2_activation_scale.data_ptr<float>(),
                fc2_bias_ptr != nullptr);
    } else {
        quant_dp4a_fc2_residual_kernel<8>
            <<<persistent_ctas, THREADS, 0, stream>>>(
                activated.data_ptr<float>(), fc2_weight.data_ptr<int8_t>(),
                fc2_scale.data_ptr<float>(), fc2_bias_ptr,
                input.data_ptr<float>(), output.data_ptr<float>(),
                fc2_quantized.data_ptr<int8_t>(),
                fc2_activation_scale.data_ptr<float>(),
                fc2_bias_ptr != nullptr);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        output,
        activated,
        fc1_quantized,
        fc1_activation_scale,
        fc2_quantized,
        fc2_activation_scale,
    };
}
"""


def _load_dp4a_mlp_extension():
    global _DP4A_MLP_EXTENSION
    if _DP4A_MLP_EXTENSION is None:
        _DP4A_MLP_EXTENSION = _load_inline(
            name="mapperatorinator_sm75_dp4a_mlp_scout_v1",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-gencode=arch=compute_75,code=sm_75",
            ],
            verbose=False,
        )
    return _DP4A_MLP_EXTENSION


def preload_dp4a_mlp_extension():
    """Build/load the verifier-only extension without changing dispatch."""

    return _load_dp4a_mlp_extension()


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


def _validate_linear(
    value: Int8PackedLinear,
    *,
    name: str,
    device: torch.device,
    shape: tuple[int, int],
) -> None:
    if not isinstance(value, Int8PackedLinear):
        raise TypeError(f"{name} must be an Int8PackedLinear")
    if value.weight.dtype != torch.int8 or not value.weight.is_contiguous():
        raise TypeError(f"{name} weight must be contiguous INT8")
    if tuple(value.weight.shape) != shape:
        raise ValueError(f"{name} weight must have shape {shape}")
    if value.weight.device != device:
        raise ValueError(f"{name} weight/input device mismatch")
    if value.weight.data_ptr() % 4:
        raise ValueError(f"{name} weight must be four-byte aligned for signed DP4A")
    if value.scale.dtype != torch.float32 or not value.scale.is_contiguous():
        raise TypeError(f"{name} scale must be contiguous FP32")
    if value.scale.device != device or tuple(value.scale.shape) != (shape[0],):
        raise ValueError(f"{name} scale/output ownership mismatch")
    if value.bias is not None:
        _require_fp32_cuda(value.bias, name=f"{name} bias")
        if value.bias.device != device or value.bias.numel() != shape[0]:
            raise ValueError(f"{name} bias/output ownership mismatch")


def _validate_inputs(
    input: Any,
    norm_weight: Any,
    pack: Any,
    *,
    active_warps: int,
    persistent_ctas: int,
) -> tuple[torch.Tensor, torch.Tensor, Int8MlpPack]:
    input = _require_fp32_cuda(input, name="input")
    norm_weight = _require_fp32_cuda(norm_weight, name="norm weight")
    if tuple(input.shape) != (1, 1, 768):
        raise ValueError("input must have fixed shape [1, 1, 768]")
    if tuple(norm_weight.shape) != (768,) or norm_weight.device != input.device:
        raise ValueError("norm weight must be input-owned [768]")
    if not isinstance(pack, Int8MlpPack):
        raise TypeError("pack must be an Int8MlpPack")
    _validate_linear(
        pack.fc1,
        name="fc1",
        device=input.device,
        shape=(3072, 768),
    )
    _validate_linear(
        pack.fc2,
        name="fc2",
        device=input.device,
        shape=(768, 3072),
    )
    if active_warps not in (4, 8):
        raise ValueError("active_warps must be 4 or 8")
    maximum_ctas = (768 + active_warps - 1) // active_warps
    if persistent_ctas < 1 or persistent_ctas > min(68, maximum_ctas):
        raise ValueError("persistent_ctas must be within the SM75 FC2 output bound")
    return input, norm_weight, pack


def validate_dp4a_mlp_pack(pack: Int8MlpPack) -> dict[str, Any]:
    """Synchronously validate immutable INT8 pack contents before capture."""

    if not isinstance(pack, Int8MlpPack):
        raise TypeError("pack must be an Int8MlpPack")
    device = pack.fc1.weight.device
    _validate_linear(pack.fc1, name="fc1", device=device, shape=(3072, 768))
    _validate_linear(pack.fc2, name="fc2", device=device, shape=(768, 3072))
    rows: dict[str, Any] = {}
    for name, value in (("fc1", pack.fc1), ("fc2", pack.fc2)):
        if not bool(torch.isfinite(value.scale).all().item()) or not bool(
            (value.scale > 0).all().item()
        ):
            raise ValueError(f"{name} scales must be finite and positive")
        contains_minus_128 = bool((value.weight == -128).any().item())
        if contains_minus_128:
            raise ValueError(f"{name} signed DP4A weights must not contain -128")
        if value.bias is not None and not bool(torch.isfinite(value.bias).all().item()):
            raise ValueError(f"{name} bias must be finite")
        rows[name] = {
            "scale_min": float(value.scale.min().item()),
            "scale_max": float(value.scale.max().item()),
            "weight_contains_minus_128": contains_minus_128,
        }
    return rows


def dp4a_mlp_residual_with_state(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    pack: Int8MlpPack,
    *,
    eps: float,
    active_warps: int = 8,
    persistent_ctas: int = 68,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return FP32 output plus the owned intermediate quantization state."""

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
        raise RuntimeError(
            f"signed DP4A MLP scout requires SM75, got sm_{capability[0]}{capability[1]}"
        )
    values = _load_dp4a_mlp_extension().dp4a_mlp_residual(
        input,
        norm_weight,
        pack.fc1.weight,
        pack.fc1.scale,
        pack.fc1.bias,
        pack.fc2.weight,
        pack.fc2.scale,
        pack.fc2.bias,
        float(eps),
        int(active_warps),
        int(persistent_ctas),
    )
    if not isinstance(values, (tuple, list)) or len(values) != 6:
        raise RuntimeError("DP4A MLP extension returned an invalid output tuple")
    output, activated, fc1_quantized, fc1_scale, fc2_quantized, fc2_scale = values
    contracts = (
        (output, torch.float32, (1, 1, 768), "output"),
        (activated, torch.float32, (1, 1, 3072), "activated"),
        (fc1_quantized, torch.int8, (768,), "FC1 quantized activation"),
        (fc1_scale, torch.float32, (1,), "FC1 activation scale"),
        (fc2_quantized, torch.int8, (3072,), "FC2 quantized activation"),
        (fc2_scale, torch.float32, (1,), "FC2 activation scale"),
    )
    for value, dtype, shape, name in contracts:
        if value.dtype != dtype or tuple(value.shape) != shape:
            raise RuntimeError(f"DP4A MLP {name} shape/dtype contract failed")
        if value.device != input.device:
            raise RuntimeError(f"DP4A MLP {name} uses the wrong device")
    return tuple(values)


def dp4a_mlp_residual(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    pack: Int8MlpPack,
    *,
    eps: float,
    active_warps: int = 8,
    persistent_ctas: int = 68,
) -> torch.Tensor:
    """Run the verifier-only two-stage DP4A MLP and return FP32 output."""

    return dp4a_mlp_residual_with_state(
        input,
        norm_weight,
        pack,
        eps=eps,
        active_warps=active_warps,
        persistent_ctas=persistent_ctas,
    )[0]


__all__ = [
    "dp4a_mlp_residual",
    "dp4a_mlp_residual_with_state",
    "preload_dp4a_mlp_extension",
    "validate_dp4a_mlp_pack",
]
