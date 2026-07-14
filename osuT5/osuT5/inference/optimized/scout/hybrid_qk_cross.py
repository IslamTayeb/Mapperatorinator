"""Opt-in mixed-storage q1 cross-attention component candidate.

This module is verifier-only.  Importing the optimized runtime does not import
it, build native code, convert caches, or change production dispatch.
"""

from __future__ import annotations

import torch


def _validate_hybrid_qk_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[int, int, int]:
    if not all(isinstance(tensor, torch.Tensor) for tensor in (query, key, value)):
        raise TypeError("hybrid q1 query, key, and value must be tensors")
    if query.dtype != torch.float32:
        raise TypeError("hybrid q1 query activation must use FP32 storage")
    if key.dtype != torch.float16:
        raise TypeError("hybrid q1 key cache must use FP16 storage")
    if value.dtype != torch.float32:
        raise TypeError("hybrid q1 value cache must use FP32 storage")
    if query.device != key.device or query.device != value.device:
        raise ValueError("hybrid q1 query, key, and value must use one device")
    if query.ndim != 4 or query.shape[0] != 1 or query.shape[-2] != 1:
        raise ValueError("hybrid q1 query must have shape [1, heads, 1, head_dim]")
    if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
        raise ValueError("hybrid q1 key and value must have matching 4D shapes")
    if (
        key.shape[0] != 1
        or key.shape[1] != query.shape[1]
        or key.shape[3] != query.shape[3]
        or key.shape[2] <= 0
    ):
        raise ValueError("hybrid q1 key/value shapes must match the query heads")
    if not query.is_contiguous() or not key.is_contiguous() or not value.is_contiguous():
        raise ValueError("hybrid q1 query, key, and value must be contiguous")
    return int(query.shape[1]), int(query.shape[-1]), int(key.shape[2])


def hybrid_qk_fp32_value_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Use FP16 QK traffic with FP32 score and value accumulation.

    The query projection remains an FP32 activation.  Only its q1 BMM input is
    converted to FP16.  Keys are prepacked once outside the decode graph;
    values, probabilities, reductions, and the returned activation stay FP32.
    """

    heads, head_dim, kv_length = _validate_hybrid_qk_inputs(query, key, value)
    q = query.to(dtype=torch.float16).reshape(heads, 1, head_dim)
    k = key.reshape(heads, kv_length, head_dim)
    v = value.reshape(heads, kv_length, head_dim)
    if query.device.type == "cuda":
        scores = torch.bmm(q, k.transpose(1, 2), out_dtype=torch.float32)
    else:
        # CPU reference path exists only for fail-loud unit coverage.
        scores = torch.bmm(q.float(), k.float().transpose(1, 2))
    scores.mul_(head_dim**-0.5)
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.bmm(probabilities, v)
    if output.dtype != torch.float32:
        raise RuntimeError("hybrid q1 value reduction did not return FP32 storage")
    return output.view(1, heads, 1, head_dim)


__all__ = ["hybrid_qk_fp32_value_attention"]
