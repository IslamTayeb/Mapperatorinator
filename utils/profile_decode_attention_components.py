from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

from inference import compile_args, load_model_with_server, setup_inference_environment
from osuT5.osuT5.inference.cache_utils import get_cache
from osuT5.osuT5.inference.direct_decode import (
    decode_one_token_raw_logits,
    last_token_logits,
    prefill_static_cache,
)
from osuT5.osuT5.inference.server import get_eos_token_id
from osuT5.osuT5.model.custom_transformers import modeling_varwhisper
from osuT5.osuT5.inference.optimized.kernels.dispatch import (
    active_prefix_attention_inputs,
)
from osuT5.osuT5.runtime_profiling import generation_profile_context
from osuT5.osuT5.tokenizer import ContextType
from utils.profile_decode_linear_kernels import (
    _allclose,
    _bucketed_prefix_length,
    _cuda_event_time_ms,
    _max_abs,
    _safe_ratio,
)
from utils.verify_one_token_decode import (
    _build_generation_logits_processors,
    _build_probe_inputs,
    _condition_kwargs,
    _load_args,
    _move_kwargs_for_model,
)


_NATIVE_Q1_ATTENTION = None


def _load_native_q1_attention():
    global _NATIVE_Q1_ATTENTION
    if _NATIVE_Q1_ATTENTION is not None:
        return _NATIVE_Q1_ATTENTION

    cuda_source = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

template<int BLOCK_SIZE>
__global__ void q1_attention_kernel(
        const float* __restrict__ q,
        const float* __restrict__ k,
        const float* __restrict__ v,
        const float* __restrict__ mask,
        float* __restrict__ out,
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

    const float* qh = q + head * q_head_stride;
    const float* kh = k + head * k_head_stride;
    const float* vh = v + head * v_head_stride;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    extern __shared__ float shared_mem[];
    float* scores = shared_mem;
    float* scratch = shared_mem + kv_len;

    float max_score = -FLT_MAX;
    for (int pos = tid; pos < kv_len; pos += BLOCK_SIZE) {
        float score = 0.0f;
        const float* kp = kh + pos * k_len_stride;
        for (int d = 0; d < head_dim; ++d) {
            score += qh[d * q_dim_stride] * kp[d * k_dim_stride];
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
            numer += scores[pos] * vh[pos * v_len_stride + dim * v_dim_stride];
        }
        out[head * head_dim + dim] = numer / denom;
    }
}

template<int BLOCK_SIZE>
torch::Tensor q1_attention_impl(torch::Tensor q, torch::Tensor k, torch::Tensor v, c10::optional<torch::Tensor> mask) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "q must be fp32");
    TORCH_CHECK(k.scalar_type() == torch::kFloat32, "k must be fp32");
    TORCH_CHECK(v.scalar_type() == torch::kFloat32, "v must be fp32");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be 4D");
    TORCH_CHECK(q.size(0) == 1 && q.size(2) == 1, "q must have shape [1, H, 1, D]");
    TORCH_CHECK(k.size(0) == 1 && v.size(0) == 1, "k/v batch must be 1");
    TORCH_CHECK(k.size(1) == q.size(1) && v.size(1) == q.size(1), "head count mismatch");
    TORCH_CHECK(k.size(3) == q.size(3) && v.size(3) == q.size(3), "head dim mismatch");
    TORCH_CHECK(k.size(2) == v.size(2), "kv length mismatch");
    int heads = static_cast<int>(q.size(1));
    int kv_len = static_cast<int>(k.size(2));
    int head_dim = static_cast<int>(q.size(3));
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

    dim3 grid(heads);
    dim3 block(BLOCK_SIZE);
    size_t shared_bytes = static_cast<size_t>(kv_len + BLOCK_SIZE) * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    q1_attention_kernel<BLOCK_SIZE><<<grid, block, shared_bytes, stream>>>(
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
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor q1_attention_block(torch::Tensor q, torch::Tensor k, torch::Tensor v, c10::optional<torch::Tensor> mask, int block_size) {
    switch (block_size) {
        case 64:
            return q1_attention_impl<64>(q, k, v, mask);
        case 128:
            return q1_attention_impl<128>(q, k, v, mask);
        case 256:
            return q1_attention_impl<256>(q, k, v, mask);
        default:
            TORCH_CHECK(false, "unsupported q1 attention block size; expected 64, 128, or 256");
    }
}

torch::Tensor q1_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, c10::optional<torch::Tensor> mask) {
    return q1_attention_block(q, k, v, mask, 128);
}
"""

    _NATIVE_Q1_ATTENTION = load_inline(
        name="mapperatorinator_q1_attention_probe",
        cpp_sources=(
            "#include <torch/extension.h>\n"
            "torch::Tensor q1_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, "
            "c10::optional<torch::Tensor> mask);\n"
            "torch::Tensor q1_attention_block(torch::Tensor q, torch::Tensor k, torch::Tensor v, "
            "c10::optional<torch::Tensor> mask, int block_size);\n"
        ),
        cuda_sources=cuda_source,
        functions=["q1_attention", "q1_attention_block"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return _NATIVE_Q1_ATTENTION


@dataclass
class AttentionCapture:
    name: str
    kind: str
    layer_idx: int
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attention_mask: torch.Tensor | None
    output: torch.Tensor
    used_q1_bmm: bool
    bs: int
    dim: int


def _effective_attention_inputs(
        *,
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        sliding_window_mask: torch.Tensor | None,
        local_attention: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    effective_mask = (
        sliding_window_mask
        if local_attention != (-1, -1)
        else attention_mask
    )
    return active_prefix_attention_inputs(
        module=module,
        query=query,
        key=key,
        value=value,
        attention_mask=effective_mask,
    )


def _uses_q1_bmm(
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        enabled: bool,
) -> bool:
    return (
        enabled
        and module.is_cross_attention
        and not module.training
        and query.dtype == torch.float32
        and key.dtype == torch.float32
        and value.dtype == torch.float32
        and attention_mask is None
        and query.shape[0] == 1
        and query.shape[-2] == 1
        and key.shape[0] == 1
        and value.shape[0] == 1
    )


def _attention_signature(capture: AttentionCapture) -> str:
    mask_shape = "none" if capture.attention_mask is None else "x".join(str(dim) for dim in capture.attention_mask.shape)
    return (
        f"{capture.kind}_q{'x'.join(str(dim) for dim in capture.query.shape)}_"
        f"k{'x'.join(str(dim) for dim in capture.key.shape)}_"
        f"mask{mask_shape}_q1{int(capture.used_q1_bmm)}"
    )


def _repo_sdpa(capture: AttentionCapture) -> torch.Tensor:
    attn_output = F.scaled_dot_product_attention(
        capture.query,
        capture.key,
        capture.value,
        dropout_p=0.0,
        attn_mask=capture.attention_mask,
    )
    return attn_output.transpose(1, 2).contiguous().view(capture.bs, -1, capture.dim)


def _sdpa_attention_only(capture: AttentionCapture) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        capture.query,
        capture.key,
        capture.value,
        dropout_p=0.0,
        attn_mask=capture.attention_mask,
    )


def _reference_attention_only(capture: AttentionCapture) -> torch.Tensor:
    batch = capture.query.shape[0]
    heads = capture.query.shape[1]
    q_len = capture.query.shape[2]
    head_dim = capture.query.shape[3]
    return capture.output.view(batch, q_len, heads, head_dim).transpose(1, 2).contiguous()


def _output_transform_only(capture: AttentionCapture, attention_output: torch.Tensor) -> torch.Tensor:
    return attention_output.transpose(1, 2).contiguous().view(capture.bs, -1, capture.dim)


def _native_q1_attention_only(capture: AttentionCapture, *, block_size: int | None = None) -> torch.Tensor:
    batch, _heads, q_len, _head_dim = capture.query.shape
    if batch != 1 or q_len != 1:
        raise ValueError(f"native q1 attention only supports batch=1 and q_len=1, got {list(capture.query.shape)}")
    extension = _load_native_q1_attention()
    mask = None
    if capture.attention_mask is not None:
        mask = capture.attention_mask.reshape(-1).to(dtype=torch.float32).contiguous()
    if block_size is None:
        return extension.q1_attention(
            capture.query,
            capture.key,
            capture.value,
            mask,
        )
    return extension.q1_attention_block(
        capture.query,
        capture.key,
        capture.value,
        mask,
        int(block_size),
    )


def _native_q1_attention(capture: AttentionCapture, *, block_size: int | None = None) -> torch.Tensor:
    return _output_transform_only(capture, _native_q1_attention_only(capture, block_size=block_size))


def _q1_bmm(capture: AttentionCapture) -> torch.Tensor:
    batch, heads, q_len, head_dim = capture.query.shape
    if batch != 1 or q_len != 1:
        raise ValueError(f"q1_bmm variant only supports batch=1 and q_len=1, got {list(capture.query.shape)}")
    kv_len = capture.key.shape[-2]
    q = capture.query.reshape(heads, q_len, head_dim)
    k = capture.key.reshape(heads, kv_len, head_dim)
    v = capture.value.reshape(heads, kv_len, head_dim)
    scores = torch.bmm(q, k.transpose(1, 2)) * (head_dim ** -0.5)
    if capture.attention_mask is not None:
        mask = capture.attention_mask.to(dtype=scores.dtype)
        mask = mask.expand(batch, heads, q_len, kv_len).reshape(batch * heads, q_len, kv_len)
        scores = scores + mask
    attn_output = torch.bmm(torch.softmax(scores, dim=-1), v).view(batch, heads, q_len, head_dim)
    return attn_output.transpose(1, 2).contiguous().view(capture.bs, -1, capture.dim)


def _q1_bmm_attention_only(capture: AttentionCapture) -> torch.Tensor:
    batch, heads, q_len, head_dim = capture.query.shape
    if batch != 1 or q_len != 1:
        raise ValueError(f"q1_bmm variant only supports batch=1 and q_len=1, got {list(capture.query.shape)}")
    kv_len = capture.key.shape[-2]
    q = capture.query.reshape(heads, q_len, head_dim)
    k = capture.key.reshape(heads, kv_len, head_dim)
    v = capture.value.reshape(heads, kv_len, head_dim)
    scores = torch.bmm(q, k.transpose(1, 2)) * (head_dim ** -0.5)
    if capture.attention_mask is not None:
        mask = capture.attention_mask.to(dtype=scores.dtype)
        mask = mask.expand(batch, heads, q_len, kv_len).reshape(batch * heads, q_len, kv_len)
        scores = scores + mask
    attn_output = torch.bmm(torch.softmax(scores, dim=-1), v).view(batch, heads, q_len, head_dim)
    return attn_output


def _benchmark_attention_capture(
        capture: AttentionCapture,
        *,
        warmup: int,
        iters: int,
        atol: float,
        rtol: float,
        native_q1_attention: bool,
        native_q1_block_sizes: tuple[int, ...],
) -> dict[str, Any]:
    reference = capture.output
    attention_reference = _reference_attention_only(capture)
    variants: dict[str, tuple[Callable[[], torch.Tensor], torch.Tensor]] = {
        "repo_sdpa": (lambda: _repo_sdpa(capture), reference),
        "sdpa_attention_only": (lambda: _sdpa_attention_only(capture), attention_reference),
        "sdpa_output_transform_only": (
            lambda: _output_transform_only(capture, attention_reference),
            reference,
        ),
    }
    if capture.query.shape[0] == 1 and capture.query.shape[-2] == 1:
        variants["q1_bmm"] = (lambda: _q1_bmm(capture), reference)
        variants["q1_bmm_attention_only"] = (lambda: _q1_bmm_attention_only(capture), attention_reference)
        if native_q1_attention:
            variants["native_q1_attention"] = (lambda: _native_q1_attention(capture), reference)
            variants["native_q1_attention_only"] = (
                lambda: _native_q1_attention_only(capture),
                attention_reference,
            )
            for block_size in native_q1_block_sizes:
                variants[f"native_q1_attention_bs{block_size}"] = (
                    lambda block_size=block_size: _native_q1_attention(capture, block_size=block_size),
                    reference,
                )
                variants[f"native_q1_attention_bs{block_size}_only"] = (
                    lambda block_size=block_size: _native_q1_attention_only(capture, block_size=block_size),
                    attention_reference,
                )

    results: dict[str, dict[str, Any]] = {}
    for name, (fn, expected) in variants.items():
        ms, output = _cuda_event_time_ms(fn, warmup=warmup, iters=iters)
        results[name] = {
            "ms_per_call": ms,
            "allclose": _allclose(expected, output, atol=atol, rtol=rtol),
            "max_abs": _max_abs(expected, output),
            "output_shape": list(output.shape),
        }
    base_ms = results["repo_sdpa"]["ms_per_call"]
    for result in results.values():
        result["speedup_vs_repo_sdpa"] = _safe_ratio(base_ms, float(result["ms_per_call"]))
    return results


def _capture_decoder_attention(
        model,
        prepared_inputs: dict[str, Any],
        *,
        active_prefix_length: int,
        q1_bmm_cross_attention: bool,
        sdpa_backend: str | None,
) -> tuple[dict[str, AttentionCapture], torch.Tensor]:
    original = modeling_varwhisper.VARWHISPER_ATTENTION_FUNCTION["sdpa"]
    captures: dict[str, AttentionCapture] = {}

    def capturing_forward(
            module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            bs: int,
            dim: int,
            local_attention: tuple[int, int] = (-1, -1),
            attention_mask: torch.Tensor | None = None,
            sliding_window_mask: torch.Tensor | None = None,
            **kwargs,
    ):
        effective_query, effective_key, effective_value, effective_mask = _effective_attention_inputs(
            module=module,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            local_attention=local_attention,
        )
        outputs = original(
            module=module,
            query=query,
            key=key,
            value=value,
            bs=bs,
            dim=dim,
            local_attention=local_attention,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            **kwargs,
        )
        kind = "cross" if module.is_cross_attention else "self"
        name = f"attention.layer{module.layer_idx}.{kind}"
        if name not in captures:
            captures[name] = AttentionCapture(
                name=name,
                kind=kind,
                layer_idx=int(module.layer_idx),
                query=effective_query.detach(),
                key=effective_key.detach(),
                value=effective_value.detach(),
                attention_mask=effective_mask.detach() if isinstance(effective_mask, torch.Tensor) else None,
                output=outputs[0].detach(),
                used_q1_bmm=_uses_q1_bmm(
                    module,
                    effective_query,
                    effective_key,
                    effective_value,
                    effective_mask,
                    enabled=q1_bmm_cross_attention,
                ),
                bs=int(bs),
                dim=int(dim),
            )
        return outputs

    modeling_varwhisper.VARWHISPER_ATTENTION_FUNCTION["sdpa"] = capturing_forward
    try:
        with generation_profile_context(
                sdpa_backend=sdpa_backend,
                active_prefix_self_attention_length=active_prefix_length,
                q1_bmm_cross_attention=q1_bmm_cross_attention,
        ):
            outputs = model(**prepared_inputs)
            logits = last_token_logits(outputs.logits)
    finally:
        modeling_varwhisper.VARWHISPER_ATTENTION_FUNCTION["sdpa"] = original

    return captures, logits


@torch.no_grad()
def profile_decode_attention_components(
        args,
        *,
        sequence_index: int,
        active_prefix_bucket_size: int,
        active_prefix_decode_length: int | None,
        q1_bmm_cross_attention: bool,
        native_q1_attention: bool,
        native_q1_block_sizes: tuple[int, ...],
        warmup: int,
        iters: int,
        atol: float,
        rtol: float,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Decode attention component profiling requires CUDA")

    compile_args(args, verbose=False)
    setup_inference_environment(args.seed)
    model, tokenizer = load_model_with_server(
        args.model_path,
        args.train,
        args.device,
        max_batch_size=args.max_batch_size,
        use_server=False,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
        lora_path=args.lora_path,
        gamemode=args.gamemode,
        auto_select_gamemode_model=args.auto_select_gamemode_model,
        generation_compile=args.inference_generation_compile,
    )
    model.eval()
    model_inputs = _move_kwargs_for_model(
        model,
        _build_probe_inputs(args, model, tokenizer, sequence_index=sequence_index),
    )
    metadata = {
        "model_path": args.model_path,
        "precision": args.precision,
        "attn_implementation": args.attn_implementation,
        "profile_sdpa_backend": args.profile_sdpa_backend,
        "torch_version": torch.__version__,
        "sequence_index": model_inputs.pop("sequence_index"),
        "frame_time_ms": model_inputs.pop("frame_time_ms"),
        "context_type": model_inputs.pop("context_type"),
        "lookback_time": model_inputs.pop("lookback_time"),
        "lookahead_time": model_inputs.pop("lookahead_time"),
        "warmup": warmup,
        "iters": iters,
        "active_prefix_bucket_size": active_prefix_bucket_size,
        "active_prefix_decode_length_override": active_prefix_decode_length,
        "q1_bmm_cross_attention": bool(q1_bmm_cross_attention),
        "native_q1_attention": bool(native_q1_attention),
        "native_q1_block_sizes": list(native_q1_block_sizes),
    }
    prompt = model_inputs["decoder_input_ids"]
    prompt_mask = model_inputs["decoder_attention_mask"]
    prompt_len = int(prompt.shape[-1])
    condition_kwargs = _condition_kwargs(model_inputs)
    logits_processors = _build_generation_logits_processors(
        args,
        tokenizer,
        model.device,
        lookback_time=float(metadata["lookback_time"]),
    )
    context_type = ContextType(metadata["context_type"])
    eos_token_ids = get_eos_token_id(
        tokenizer,
        lookback_time=float(metadata["lookback_time"]),
        lookahead_time=float(metadata["lookahead_time"]),
        context_type=context_type,
    )

    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=args.precision == "amp"), \
            generation_profile_context(
                sdpa_backend=args.profile_sdpa_backend,
                q1_bmm_cross_attention=q1_bmm_cross_attention,
            ):
        hf_cache = get_cache(model, batch_size=1, num_beams=1, cfg_scale=1.0)
        hf_generate_outputs = model.generate(
            inputs=model_inputs["frames"],
            decoder_input_ids=prompt,
            decoder_attention_mask=prompt_mask,
            **condition_kwargs,
            do_sample=args.do_sample,
            num_beams=args.num_beams,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=1,
            use_cache=True,
            past_key_values=hf_cache,
            logits_processor=logits_processors,
            eos_token_id=eos_token_ids,
            return_dict_in_generate=True,
            output_logits=False,
        )
        probe_token = hf_generate_outputs.sequences[:, prompt_len:prompt_len + 1].to(torch.long)
        if probe_token.numel() != 1:
            raise RuntimeError(f"expected one generated probe token, got shape {list(probe_token.shape)}")
        full_prefix = torch.cat([prompt, probe_token], dim=-1)
        full_mask = torch.cat([prompt_mask, torch.ones_like(probe_token, dtype=prompt_mask.dtype)], dim=-1)

        state = prefill_static_cache(
            model,
            prompt=prompt,
            prompt_attention_mask=prompt_mask,
            frames=model_inputs["frames"],
            condition_kwargs=condition_kwargs,
            active_prefix_self_attention=False,
        )
        cache_position = torch.arange(state.prompt_length, state.prompt_length + 1, device=prompt.device)
        max_cache_len = int(state.cache.get_max_cache_shape())
        computed_active_prefix_length = _bucketed_prefix_length(
            int(full_prefix.shape[-1]),
            active_prefix_bucket_size,
            max_cache_len,
        )
        active_prefix_length = (
            int(active_prefix_decode_length)
            if active_prefix_decode_length is not None
            else computed_active_prefix_length
        )
        direct_result = decode_one_token_raw_logits(
            model,
            state,
            full_prefix=full_prefix,
            full_attention_mask=full_mask,
            condition_kwargs=condition_kwargs,
            cache_position=cache_position,
            active_prefix_self_attention=True,
            active_prefix_self_attention_length=active_prefix_length,
        )
        captures, replay_logits = _capture_decoder_attention(
            model,
            direct_result.prepared_inputs,
            active_prefix_length=active_prefix_length,
            q1_bmm_cross_attention=q1_bmm_cross_attention,
            sdpa_backend=args.profile_sdpa_backend,
        )

    if not captures:
        raise RuntimeError("did not capture any VarWhisper SDPA attention calls; check the model/backend")

    signature_members: dict[str, list[str]] = defaultdict(list)
    for name, capture in captures.items():
        signature_members[_attention_signature(capture)].append(name)

    signature_reports: dict[str, Any] = {}
    operation_counts: dict[str, int] = defaultdict(int)
    for capture in captures.values():
        operation_counts[capture.kind] += 1

    for signature, names in sorted(signature_members.items()):
        capture = captures[sorted(names)[0]]
        signature_reports[signature] = {
            "representative": capture.name,
            "members": sorted(names),
            "member_count": len(names),
            "kind": capture.kind,
            "layer_idx": capture.layer_idx,
            "query_shape": list(capture.query.shape),
            "key_shape": list(capture.key.shape),
            "value_shape": list(capture.value.shape),
            "attention_mask_shape": (
                list(capture.attention_mask.shape)
                if isinstance(capture.attention_mask, torch.Tensor)
                else None
            ),
            "attention_mask_dtype": (
                str(capture.attention_mask.dtype)
                if isinstance(capture.attention_mask, torch.Tensor)
                else None
            ),
            "attention_mask_min": (
                float(capture.attention_mask.detach().to(torch.float32).min().item())
                if isinstance(capture.attention_mask, torch.Tensor)
                else None
            ),
            "attention_mask_max": (
                float(capture.attention_mask.detach().to(torch.float32).max().item())
                if isinstance(capture.attention_mask, torch.Tensor)
                else None
            ),
            "used_q1_bmm": bool(capture.used_q1_bmm),
            "results": _benchmark_attention_capture(
                capture,
                warmup=warmup,
                iters=iters,
                atol=atol,
                rtol=rtol,
                native_q1_attention=native_q1_attention,
                native_q1_block_sizes=native_q1_block_sizes,
            ),
        }

    return {
        "pass": bool(_allclose(direct_result.logits, replay_logits, atol=atol, rtol=rtol)),
        "prompt_tokens": prompt_len,
        "probe_token_id": int(probe_token.item()),
        "full_prefix_tokens": int(full_prefix.shape[-1]),
        "cache_position": [int(item) for item in cache_position.detach().cpu().tolist()],
        "max_cache_len": max_cache_len,
        "computed_active_prefix_length": computed_active_prefix_length,
        "active_prefix_length": active_prefix_length,
        "captured_attention_count": len(captures),
        "operation_counts": dict(sorted(operation_counts.items())),
        "signature_reports": signature_reports,
        "logits_replay_allclose": bool(_allclose(direct_result.logits, replay_logits, atol=atol, rtol=rtol)),
        "logits_replay_max_abs": _max_abs(direct_result.logits, replay_logits),
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark real one-token decoder self/cross attention tensors captured from a "
            "Mapperatorinator direct decode step. Diagnostic only; not an inference speed claim."
        )
    )
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--sequence-index", type=int, default=9)
    parser.add_argument("--active-prefix-bucket-size", type=int, default=64)
    parser.add_argument("--active-prefix-decode-length", type=int, default=None)
    parser.add_argument("--q1-bmm-cross-attention", action="store_true")
    parser.add_argument(
        "--native-q1-attention",
        action="store_true",
        help="Compile and benchmark a diagnostic native CUDA q_len=1 attention kernel. Diagnostic only.",
    )
    parser.add_argument(
        "--native-q1-block-sizes",
        default="",
        help=(
            "Comma-separated diagnostic native q1 attention CUDA block sizes to benchmark, "
            "currently limited to 64,128,256. Requires --native-q1-attention."
        ),
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model_path=/path/to/model")
    cli_args = parser.parse_args()

    start = time.perf_counter()
    args = _load_args(cli_args.config_name, cli_args.overrides)
    native_q1_block_sizes = tuple(
        int(item)
        for item in cli_args.native_q1_block_sizes.split(",")
        if item.strip()
    )
    unsupported_block_sizes = sorted(set(native_q1_block_sizes) - {64, 128, 256})
    if unsupported_block_sizes:
        raise ValueError(f"unsupported native q1 block sizes: {unsupported_block_sizes}")
    if native_q1_block_sizes and not cli_args.native_q1_attention:
        raise ValueError("--native-q1-block-sizes requires --native-q1-attention")
    result = profile_decode_attention_components(
        args,
        sequence_index=cli_args.sequence_index,
        active_prefix_bucket_size=cli_args.active_prefix_bucket_size,
        active_prefix_decode_length=cli_args.active_prefix_decode_length,
        q1_bmm_cross_attention=cli_args.q1_bmm_cross_attention,
        native_q1_attention=cli_args.native_q1_attention,
        native_q1_block_sizes=native_q1_block_sizes,
        warmup=cli_args.warmup,
        iters=cli_args.iters,
        atol=cli_args.atol,
        rtol=cli_args.rtol,
    )
    result["metadata"]["config_name"] = cli_args.config_name
    result["wall_seconds"] = time.perf_counter() - start

    if cli_args.report_path is not None:
        cli_args.report_path.parent.mkdir(parents=True, exist_ok=True)
        cli_args.report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
