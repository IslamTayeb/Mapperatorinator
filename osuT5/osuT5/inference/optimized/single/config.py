"""Frozen configuration for the accepted FP32 optimized single-song stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


OPTIMIZED_SINGLE_CONFIG_VERSION = "accepted-fp32-270.475-v1"


@dataclass(frozen=True, slots=True)
class OptimizedSingleConfig:
    version: str = OPTIMIZED_SINGLE_CONFIG_VERSION
    result_class: str = "exact-output"
    precision: str = "fp32"
    attn_implementation: str = "sdpa"
    generation_compile: bool = True
    batch_size: int = 1
    cfg_scale: float = 1.0
    num_beams: int = 1
    active_prefix_decode_loop: bool = True
    active_prefix_decode_bucket_size: int = 64
    active_prefix_decode_cuda_graph: bool = True
    active_prefix_decode_cuda_graph_warmup: int = 0
    active_prefix_decode_cuda_graph_min_decode_steps: int = 1
    stateful_monotonic_logits_processor: bool = True
    q1_bmm_cross_attention: bool = True
    decode_session_runtime: bool = True
    decode_session_cuda_graph: bool = True
    native_decode_kernels: bool = True
    native_q1_self_attention: bool = True
    native_q1_rope_cache_self_attention: bool = True

    def generation_overrides(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "cfg_scale": self.cfg_scale,
            "num_beams": self.num_beams,
            "active_prefix_decode_loop": self.active_prefix_decode_loop,
            "active_prefix_decode_bucket_size": self.active_prefix_decode_bucket_size,
            "active_prefix_decode_cuda_graph": self.active_prefix_decode_cuda_graph,
            "active_prefix_decode_cuda_graph_warmup": self.active_prefix_decode_cuda_graph_warmup,
            "active_prefix_decode_cuda_graph_min_decode_steps": (
                self.active_prefix_decode_cuda_graph_min_decode_steps
            ),
            "stateful_monotonic_logits_processor": (
                self.stateful_monotonic_logits_processor
            ),
            "q1_bmm_cross_attention": self.q1_bmm_cross_attention,
            "native_q1_self_attention": self.native_q1_self_attention,
            "native_q1_rope_cache_self_attention": (
                self.native_q1_rope_cache_self_attention
            ),
            "decode_session_cuda_graph": self.decode_session_cuda_graph,
        }

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


ACCEPTED_OPTIMIZED_SINGLE_CONFIG = OptimizedSingleConfig()
