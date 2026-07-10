"""Lazy compatibility exports for the optimized active-prefix decode loop."""

from __future__ import annotations

from importlib import import_module


__all__ = [
    "_bucketed_prefix_length",
    "_max_cache_shape",
    "_add_diagnostic_wall",
    "_record_diagnostic_cuda_start",
    "_maybe_record_diagnostic_cuda_start",
    "_add_diagnostic_cuda_event",
    "_finalize_active_prefix_diagnostics",
    "_diagnostic_range",
    "_record_decode_bucket",
    "_apply_logits_processors_with_diagnostics",
    "_clone_static_graph_inputs",
    "_copy_static_graph_inputs",
    "_record_static_input_copy_bytes",
    "_stable_encoder_outputs",
    "_capture_decode_cuda_graph",
    "_cuda_graph_signature",
    "_update_cuda_graph_diagnostics",
    "active_prefix_decode_generate",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    implementation = import_module(
        "osuT5.osuT5.inference.optimized.single.decode_loop"
    )
    value = getattr(implementation, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
