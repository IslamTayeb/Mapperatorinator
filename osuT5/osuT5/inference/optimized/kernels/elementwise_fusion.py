"""Opt-in exact elementwise/copy fusion (§30 / research §25).

Fuses elementwise-ONLY chains (never matmul / reduction epilogues):
1. direct_copy_cast / RoPE cast-copy — folded into RoPE epilogue kernel
2. float16_copy (q1 attn pack) — reshape when contiguous [B,H,1,D]
3. float scale (RoPE attention_scaling) — folded into RoPE epilogue
4. residual add — standalone owned add (not fused into GEMM)
5. RoPE cos/sin (+ cat) — single epilogue kernel from freqs

V32 / cold optimized default: off. Enable via
``elementwise_fusion_candidate_context``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Iterator

import torch
from torch.utils.cpp_extension import load_inline


_REQUESTED = False
_EXT = None


@dataclass
class ElementwiseFusionStats:
    rope_epilogue_hits: int = 0
    attn_pack_hits: int = 0
    residual_add_hits: int = 0
    modules_patched: int = 0
    member_names: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rope_epilogue_hits": int(self.rope_epilogue_hits),
            "attn_pack_hits": int(self.attn_pack_hits),
            "residual_add_hits": int(self.residual_add_hits),
            "modules_patched": int(self.modules_patched),
            "member_names": list(self.member_names),
        }


_STATS = ElementwiseFusionStats()


def elementwise_fusion_requested() -> bool:
    return _REQUESTED


def elementwise_fusion_stats() -> ElementwiseFusionStats:
    return _STATS


def reset_elementwise_fusion_stats() -> None:
    global _STATS
    _STATS = ElementwiseFusionStats()


def _load_extension():
    global _EXT
    if _EXT is not None:
        return _EXT

    cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

template <typename OutT>
__device__ __forceinline__ OutT store_float(float value);

template <>
__device__ __forceinline__ float store_float<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ __half store_float<__half>(float value) {
    return __float2half_rn(value);
}

// One thread per output dim element. freqs layout [N, half]; outs [N, 2*half].
template <typename OutT>
__global__ void rope_epilogue_kernel(
        const float* __restrict__ freqs,
        OutT* __restrict__ cos_out,
        OutT* __restrict__ sin_out,
        int64_t n_rows,
        int half,
        float scale) {
    const int64_t numel = n_rows * static_cast<int64_t>(half) * 2;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < numel;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t row = idx / (half * 2);
        const int dim = static_cast<int>(idx % (half * 2));
        const float freq = freqs[row * half + (dim % half)];
        const float c = cosf(freq) * scale;
        const float s = sinf(freq) * scale;
        cos_out[idx] = store_float<OutT>(c);
        sin_out[idx] = store_float<OutT>(s);
    }
}

std::vector<torch::Tensor> rope_epilogue(
        torch::Tensor freqs,
        double scale,
        bool out_is_half) {
    TORCH_CHECK(freqs.is_cuda(), "freqs must be CUDA");
    TORCH_CHECK(freqs.scalar_type() == at::kFloat, "freqs must be float32");
    TORCH_CHECK(freqs.dim() >= 1, "freqs rank");
    TORCH_CHECK(freqs.is_contiguous(), "freqs must be contiguous");
    const int half = static_cast<int>(freqs.size(-1));
    TORCH_CHECK(half > 0, "freqs last dim");
    const int64_t n_rows = freqs.numel() / half;
    auto out_sizes = freqs.sizes().vec();
    out_sizes.back() = static_cast<int64_t>(half) * 2;

    const auto out_dtype = out_is_half ? at::kHalf : at::kFloat;
    auto opts = freqs.options().dtype(out_dtype);
    auto cos = torch::empty(out_sizes, opts);
    auto sin = torch::empty(out_sizes, opts);

    const int threads = 256;
    const int64_t numel = n_rows * static_cast<int64_t>(half) * 2;
    const int blocks = static_cast<int>(
        std::min<int64_t>((numel + threads - 1) / threads, 2048));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float scale_f = static_cast<float>(scale);

    if (out_dtype == at::kFloat) {
        rope_epilogue_kernel<float><<<blocks, threads, 0, stream>>>(
            freqs.data_ptr<float>(),
            cos.data_ptr<float>(),
            sin.data_ptr<float>(),
            n_rows,
            half,
            scale_f);
    } else if (out_dtype == at::kHalf) {
        rope_epilogue_kernel<__half><<<blocks, threads, 0, stream>>>(
            freqs.data_ptr<float>(),
            reinterpret_cast<__half*>(cos.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(sin.data_ptr<at::Half>()),
            n_rows,
            half,
            scale_f);
    } else {
        TORCH_CHECK(false, "rope epilogue supports float16/float32 only");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {cos, sin};
}
"""

    _EXT = load_inline(
        name="mapperatorinator_elementwise_fusion",
        cpp_sources=(
            "#include <torch/extension.h>\n"
            "#include <vector>\n"
            "std::vector<torch::Tensor> rope_epilogue("
            "torch::Tensor freqs, double scale, bool out_is_half);\n"
        ),
        cuda_sources=cuda_source,
        functions=["rope_epilogue"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return _EXT


def reference_rope_epilogue(
    freqs: torch.Tensor,
    *,
    attention_scaling: float,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager RoPE emb epilogue (ATen reference)."""

    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    return cos.to(dtype=out_dtype), sin.to(dtype=out_dtype)


def fused_rope_epilogue(
    freqs: torch.Tensor,
    *,
    attention_scaling: float,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-kernel RoPE emb epilogue: cat + cos + sin + scale + cast."""

    if not isinstance(freqs, torch.Tensor) or freqs.dtype != torch.float32:
        raise TypeError("fused RoPE epilogue requires float32 freqs")
    if out_dtype not in (torch.float16, torch.float32):
        raise TypeError("fused RoPE epilogue supports only float16/float32 out")
    if not freqs.is_cuda:
        raise RuntimeError("fused RoPE epilogue requires CUDA freqs")
    freqs_c = freqs.contiguous()
    extension = _load_extension()
    cos, sin = extension.rope_epilogue(
        freqs_c,
        float(attention_scaling),
        out_dtype == torch.float16,
    )
    _STATS.rope_epilogue_hits += 1
    return cos, sin


def fused_pack_q1_attn_out(
    output: torch.Tensor,
    *,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    all_head_size: int,
) -> torch.Tensor:
    """Pack contiguous [B,H,1,D] → [B,1,H*D] without transpose+contiguous copy."""

    if (
        not isinstance(output, torch.Tensor)
        or output.ndim != 4
        or tuple(output.shape) != (batch_size, num_heads, 1, head_dim)
        or not output.is_contiguous()
    ):
        raise RuntimeError(
            "fused q1 pack requires contiguous [B, H, 1, D] native output"
        )
    if int(num_heads) * int(head_dim) != int(all_head_size):
        raise RuntimeError("fused q1 pack head geometry mismatch")
    packed = output.reshape(batch_size, -1, all_head_size)
    _STATS.attn_pack_hits += 1
    return packed


def fused_residual_add(
    residual: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Standalone residual add (elementwise-only; never fused into GEMM)."""

    if residual.dtype != hidden_states.dtype or residual.shape != hidden_states.shape:
        raise RuntimeError("fused residual add requires matching shape/dtype")
    if residual.device != hidden_states.device:
        raise RuntimeError("fused residual add requires one device")
    out = residual + hidden_states
    _STATS.residual_add_hits += 1
    return out


@dataclass
class _Patch:
    module: torch.nn.Module
    had_instance_forward: bool
    instance_forward: Any

    def restore(self) -> None:
        if self.had_instance_forward:
            self.module.forward = self.instance_forward
        else:
            delattr(self.module, "forward")


def _patch_forward(module: torch.nn.Module, replacement) -> _Patch:
    patch = _Patch(
        module=module,
        had_instance_forward="forward" in module.__dict__,
        instance_forward=module.__dict__.get("forward"),
    )
    module.forward = MethodType(replacement, module)
    return patch


def _rope_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    from osuT5.osuT5.model.custom_transformers.modeling_varwhisper import (
        VarWhisperDecoderLayer,
    )

    found: list[tuple[str, torch.nn.Module]] = []
    for name, layer in model.named_modules():
        if not isinstance(layer, VarWhisperDecoderLayer):
            continue
        rotary = getattr(getattr(layer, "self_attn", None), "rotary_emb", None)
        if not isinstance(rotary, torch.nn.Module):
            raise TypeError(f"decoder layer {name} has no RoPE module")
        found.append((f"{name}.self_attn.rotary_emb", rotary))
    if not found:
        raise RuntimeError("elementwise fusion found no decoder RoPE modules")
    return found


@contextmanager
def elementwise_fusion_candidate_context(
    model: torch.nn.Module | None = None,
) -> Iterator[ElementwiseFusionStats]:
    """Request elementwise/copy fusion; optionally patch RoPE modules."""

    global _REQUESTED
    previous = _REQUESTED
    _REQUESTED = True
    reset_elementwise_fusion_stats()
    patches: list[_Patch] = []
    try:
        if model is not None:
            _load_extension()
            for member_name, rotary in _rope_modules(model):
                original_forward = rotary.forward

                def rope_forward(
                    self,
                    x,
                    position_ids,
                    *,
                    _original=original_forward,
                ):
                    if "dynamic" in getattr(self, "rope_type", ""):
                        return _original(x, position_ids=position_ids)
                    inv_freq_expanded = (
                        self.inv_freq[None, :, None]
                        .float()
                        .expand(position_ids.shape[0], -1, 1)
                    )
                    position_ids_expanded = position_ids[:, None, :].float()
                    device_type = x.device.type
                    device_type = (
                        device_type
                        if isinstance(device_type, str) and device_type != "mps"
                        else "cpu"
                    )
                    with torch.autocast(device_type=device_type, enabled=False):
                        freqs = (
                            inv_freq_expanded.float() @ position_ids_expanded.float()
                        ).transpose(1, 2)
                    return fused_rope_epilogue(
                        freqs,
                        attention_scaling=float(self.attention_scaling),
                        out_dtype=x.dtype,
                    )

                patches.append(_patch_forward(rotary, rope_forward))
                _STATS.member_names.append(member_name)
            _STATS.modules_patched = len(patches)
        yield _STATS
    finally:
        for patch in reversed(patches):
            patch.restore()
        _REQUESTED = previous


__all__ = [
    "ElementwiseFusionStats",
    "elementwise_fusion_candidate_context",
    "elementwise_fusion_requested",
    "elementwise_fusion_stats",
    "fused_pack_q1_attn_out",
    "fused_residual_add",
    "fused_rope_epilogue",
    "reference_rope_epilogue",
    "reset_elementwise_fusion_stats",
]
