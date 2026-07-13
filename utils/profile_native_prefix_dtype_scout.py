"""Real-tensor FP32/FP16 native-prefix component scout.

This utility runs the accepted optimized SALVALAI path once, retains the main
``ProductionDecodeSession`` graph inputs, and benchmarks verifier-only variants.
It never installs runtime or configuration wiring.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402


BUCKET_COUNTS = {
    128: 22,
    192: 64,
    256: 126,
    320: 227,
    384: 136,
    448: 332,
    512: 682,
    576: 1_727,
    640: 2_907,
    704: 1_221,
    768: 108,
}
SENTINEL_BUCKETS = (128, 576, 640)
ALL_BUCKETS = tuple(BUCKET_COUNTS)
VARIANTS = (
    "fp32_accepted",
    "fp32_framework",
    "fp32_native_prefix",
    "fp16_framework",
    "fp16_native_prefix",
)
REQUIRED_CHECKS = (
    "finite_outputs",
    "layer_output_allclose",
    "cache_slot_allclose",
    "logits_allclose",
    "cache_shapes_valid",
    "active_slot_writes_valid",
    "future_slots_untouched",
    "cross_cache_unchanged",
    "storage_ownership_valid",
    "candidate_repeat_deterministic",
    "graph_repeat_deterministic",
    "memory_stable",
)


@dataclass
class LayerCapture:
    module: torch.nn.Module
    layer_idx: int
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor | None
    encoder_hidden_states: torch.Tensor
    past_key_value: Any
    cache_position: torch.Tensor
    position_ids: torch.Tensor
    active_prefix_length: int


@dataclass
class CapturedGraph:
    graph: torch.cuda.CUDAGraph
    outputs: Any
    setup_seconds: float
    peak_vram_bytes: int


@dataclass
class FullObservation:
    logits: torch.Tensor
    key_slots: list[torch.Tensor]
    value_slots: list[torch.Tensor]


@dataclass
class PrefixObservation:
    output: torch.Tensor
    key_slot: torch.Tensor
    value_slot: torch.Tensor


DRIFT_LIMIT = 1e-3


def _load_args(config_name: str, overrides: list[str]):
    import hydra
    from omegaconf import DictConfig, OmegaConf

    config_dir = REPO_ROOT / "configs"
    resolved_name = (
        config_name
        if config_name.startswith("inference/")
        else f"inference/{config_name}"
    )
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = hydra.compose(config_name=resolved_name, overrides=overrides)
    return OmegaConf.to_object(cfg) if isinstance(cfg, DictConfig) else cfg


def _sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    try:
        payload = value.numpy().tobytes()
    except TypeError:
        payload = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _rng_state() -> dict[str, Any]:
    state = {
        "cpu_sha256": _sha256_tensor(torch.random.get_rng_state()),
        "cuda_sha256": [],
    }
    if torch.cuda.is_available():
        state["cuda_sha256"] = [
            _sha256_tensor(value) for value in torch.cuda.get_rng_state_all()
        ]
    return state


def _is_finite(tensor: torch.Tensor) -> bool:
    if tensor.is_floating_point() or tensor.is_complex():
        return bool(torch.isfinite(tensor).all().item())
    return True


def _model_logits(outputs: Any) -> torch.Tensor:
    logits = getattr(outputs, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("captured full-model output does not expose tensor logits")
    return logits[:, -1, :].detach().to(dtype=torch.float32)


def _entry_prefix(entry: dict[str, Any]) -> int:
    prefix = entry.get("active_prefix_length")
    if not isinstance(prefix, int) or prefix <= 0:
        raise ValueError(f"graph entry has invalid active_prefix_length={prefix!r}")
    return prefix


def validate_accepted_graph_cache(
    graph_cache: dict[Any, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Require the accepted eleven signatures and exact production replay counts."""

    if not isinstance(graph_cache, dict):
        raise TypeError("ProductionDecodeSession.graph_cache must be a dict")
    by_prefix: dict[int, dict[str, Any]] = {}
    for entry in graph_cache.values():
        if not isinstance(entry, dict):
            raise TypeError("graph_cache entries must be dicts")
        prefix = _entry_prefix(entry)
        if prefix in by_prefix:
            raise RuntimeError(f"accepted graph cache repeats prefix {prefix}")
        required = ("graph", "outputs", "static_inputs", "decode_replays")
        missing = [name for name in required if name not in entry]
        if missing:
            raise RuntimeError(f"accepted prefix {prefix} is missing {missing}")
        by_prefix[prefix] = entry
    if tuple(sorted(by_prefix)) != ALL_BUCKETS:
        raise RuntimeError(
            f"accepted graph cache must contain eleven buckets {list(ALL_BUCKETS)}, "
            f"got {sorted(by_prefix)}"
        )
    counts = {prefix: int(entry["decode_replays"]) for prefix, entry in by_prefix.items()}
    if counts != BUCKET_COUNTS:
        raise RuntimeError(
            f"accepted graph replay counts changed: expected {BUCKET_COUNTS}, got {counts}"
        )
    return dict(sorted(by_prefix.items()))


def select_buckets(
    entries: dict[int, dict[str, Any]],
    mode: str,
) -> dict[int, dict[str, Any]]:
    if mode == "all":
        selected = ALL_BUCKETS
    elif mode == "sentinel":
        selected = SENTINEL_BUCKETS
    else:
        raise ValueError("bucket mode must be 'sentinel' or 'all'")
    return {prefix: entries[prefix] for prefix in selected}


def _cache_from_static_inputs(static_inputs: dict[str, Any]) -> Any:
    cache = static_inputs.get("past_key_values")
    if cache is None:
        raise RuntimeError("real static inputs do not contain past_key_values")
    return cache


def _cache_layer_count(cache: Any) -> int:
    self_layers = getattr(getattr(cache, "self_attention_cache", None), "layers", None)
    cross_layers = getattr(getattr(cache, "cross_attention_cache", None), "layers", None)
    if not isinstance(self_layers, (list, tuple)) or not isinstance(cross_layers, (list, tuple)):
        raise RuntimeError("expected self/cross cache layers")
    if not self_layers or len(self_layers) != len(cross_layers):
        raise RuntimeError("self/cross cache layer counts differ")
    return len(self_layers)


def _cache_tensor(cache: Any, kind: str, layer_idx: int, name: str) -> torch.Tensor:
    owner = getattr(cache, f"{kind}_attention_cache")
    tensor = getattr(owner.layers[layer_idx], name)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{kind} layer {layer_idx} {name} is not a tensor")
    return tensor


def _all_cache_snapshots(cache: Any, cache_position: torch.Tensor):
    from osuT5.osuT5.inference.optimized.scout.verifier import snapshot_cache_state

    return [
        snapshot_cache_state(
            cache,
            layer_idx=layer_idx,
            cache_position=cache_position,
        )
        for layer_idx in range(_cache_layer_count(cache))
    ]


def _restore_all_cache(cache: Any, snapshots: list[Any]) -> None:
    from osuT5.osuT5.inference.optimized.scout.verifier import restore_cache_state

    for snapshot in snapshots:
        restore_cache_state(cache, snapshot)


def _clone_tensor(value: Any) -> Any:
    return value.detach().clone() if isinstance(value, torch.Tensor) else value


def _convert_cache_dtype(cache: Any, dtype: torch.dtype) -> Any:
    for kind in ("self", "cross"):
        owner = getattr(cache, f"{kind}_attention_cache", None)
        layers = getattr(owner, "layers", None)
        if not isinstance(layers, (list, tuple)):
            raise RuntimeError(f"missing {kind}-attention cache layers")
        for layer in layers:
            for name in ("keys", "values"):
                tensor = getattr(layer, name, None)
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"cache {kind} {name} is not a tensor")
                if tensor.is_floating_point() and tensor.dtype != dtype:
                    setattr(layer, name, tensor.to(dtype=dtype))
    return cache


def convert_static_value_dtype(
    value: Any,
    dtype: torch.dtype,
    *,
    key: str = "",
) -> Any:
    """Convert floating graph state while preserving integer and mask semantics."""

    if dtype not in (torch.float32, torch.float16):
        raise TypeError(f"scout supports float32/float16, got {dtype}")
    if isinstance(value, torch.Tensor):
        if not value.is_floating_point() or "mask" in key.lower():
            return value
        return value.to(dtype=dtype)
    if hasattr(value, "self_attention_cache") and hasattr(value, "cross_attention_cache"):
        return _convert_cache_dtype(value, dtype)
    if isinstance(value, dict):
        converted = {
            child_key: convert_static_value_dtype(child, dtype, key=str(child_key))
            for child_key, child in value.items()
        }
        if value.__class__ is dict:
            return converted
        try:
            return value.__class__(**converted)
        except TypeError:
            return converted
    if isinstance(value, list):
        return [convert_static_value_dtype(child, dtype, key=key) for child in value]
    if isinstance(value, tuple):
        return tuple(convert_static_value_dtype(child, dtype, key=key) for child in value)
    return value


def convert_static_inputs_dtype(
    static_inputs: dict[str, Any],
    dtype: torch.dtype,
) -> dict[str, Any]:
    return {
        key: convert_static_value_dtype(value, dtype, key=key)
        for key, value in static_inputs.items()
    }


@contextmanager
def _full_model_variant_context(
    *,
    model: torch.nn.Module,
    prefix: int,
    variant: str,
) -> Iterator[None]:
    from osuT5.osuT5.inference.optimized.scout.native_prefix import (
        fp32_rms_norm,
        framework_prefix,
        native_prefix,
    )
    from osuT5.osuT5.inference.optimized.single.runtime_context import (
        active_prefix_self_attention_context,
    )
    from osuT5.osuT5.model.custom_transformers.modeling_varwhisper import (
        VarWhisperDecoderLayer,
    )

    if variant not in {"framework", "native"}:
        raise ValueError("full-model scout variant must be framework or native")
    candidate = native_prefix if variant == "native" else framework_prefix
    originals: list[tuple[torch.nn.Module, Any]] = []
    final_originals: list[tuple[torch.nn.Module, Any]] = []

    def replacement(layer):
        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            past_key_value=None,
            cache_position=None,
            position_ids=None,
            **kwargs,
        ):
            del kwargs
            if encoder_hidden_states is None or past_key_value is None:
                raise RuntimeError("native-prefix scout requires cross attention and cache")
            if cache_position is None or position_ids is None:
                raise RuntimeError("native-prefix scout requires cache/position tensors")
            return (
                candidate(
                    self,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    past_key_value=past_key_value,
                    cache_position=cache_position,
                    position_ids=position_ids,
                    active_prefix_length=prefix,
                ),
            )

        return MethodType(forward, layer)

    for module in model.modules():
        if isinstance(module, VarWhisperDecoderLayer):
            originals.append((module, module.forward))
            module.forward = replacement(module)
    if not originals:
        raise RuntimeError("native-prefix scout found no decoder layers to patch")
    decoder = getattr(getattr(model, "model", None), "decoder", None)
    decoder_norm = getattr(decoder, "layer_norm", None)
    projection = getattr(model, "proj_out", None)
    if not isinstance(decoder_norm, torch.nn.Module) or not isinstance(
        projection,
        torch.nn.Module,
    ):
        raise RuntimeError("native-prefix scout could not locate final norm/projection")
    final_originals.extend(
        ((decoder_norm, decoder_norm.forward), (projection, projection.forward))
    )

    def final_norm_forward(self, hidden_states):
        return fp32_rms_norm(hidden_states, self.weight, getattr(self, "eps", None))

    def projection_forward(self, hidden_states):
        bias = getattr(self, "bias", None)
        return torch.nn.functional.linear(
            hidden_states.float(),
            self.weight.float(),
            None if bias is None else bias.float(),
        )

    decoder_norm.forward = MethodType(final_norm_forward, decoder_norm)
    projection.forward = MethodType(projection_forward, projection)
    try:
        with active_prefix_self_attention_context(prefix):
            yield
    finally:
        for module, original in originals:
            module.forward = original
        for module, original in final_originals:
            module.forward = original


def _capture_cuda_graph(
    fn: Callable[[], Any],
    *,
    context: Callable[[], Any],
    warmup: int,
) -> CapturedGraph:
    if not torch.cuda.is_available():
        raise RuntimeError("component graph capture requires CUDA")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    outputs = None
    with torch.cuda.stream(capture_stream), context():
        for _ in range(warmup):
            outputs = fn()
    torch.cuda.current_stream().wait_stream(capture_stream)
    graph = torch.cuda.CUDAGraph()
    with context(), torch.cuda.graph(graph):
        outputs = fn()
    graph.replay()
    torch.cuda.synchronize()
    if outputs is None:
        raise RuntimeError("CUDA graph candidate did not execute")
    return CapturedGraph(
        graph=graph,
        outputs=outputs,
        setup_seconds=time.perf_counter() - started,
        peak_vram_bytes=int(torch.cuda.max_memory_allocated()),
    )


def _time_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    allocated_before = int(torch.cuda.memory_allocated())
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    allocated_after = int(torch.cuda.memory_allocated())
    return {
        "ms_per_call": float(start.elapsed_time(end)) / iters,
        "allocated_before_bytes": allocated_before,
        "allocated_after_bytes": allocated_after,
        "memory_stable": allocated_before == allocated_after,
    }


def _reciprocal_graph_rounds(
    graphs: dict[str, torch.cuda.CUDAGraph],
    *,
    restore: Callable[[], None],
    warmup: int,
    iters: int,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, bool]]:
    names = list(graphs)
    rounds: list[dict[str, Any]] = []
    memory_stable = {name: True for name in names}
    for order in (names, list(reversed(names))):
        for name in order:
            restore()
            result = _time_graph(graphs[name], warmup=warmup, iters=iters)
            memory_stable[name] &= bool(result["memory_stable"])
            rounds.append({"variant": name, "order": list(order), **result})
    medians = {
        name: max(
            float(row["ms_per_call"])
            for row in rounds
            if row["variant"] == name
        )
        for name in names
    }
    return medians, rounds, memory_stable


def _cache_shapes_valid(cache: Any) -> bool:
    try:
        for layer_idx in range(_cache_layer_count(cache)):
            self_keys = _cache_tensor(cache, "self", layer_idx, "keys")
            self_values = _cache_tensor(cache, "self", layer_idx, "values")
            cross_keys = _cache_tensor(cache, "cross", layer_idx, "keys")
            cross_values = _cache_tensor(cache, "cross", layer_idx, "values")
            if (
                self_keys.ndim != 4
                or cross_keys.ndim != 4
                or self_keys.shape != self_values.shape
                or cross_keys.shape != cross_values.shape
            ):
                return False
        return True
    except (TypeError, RuntimeError):
        return False


def _verify_full_graph(
    captured: CapturedGraph,
    *,
    cache: Any,
    snapshots: list[Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    from osuT5.osuT5.inference.optimized.scout.verifier import (
        cache_ownership_matches,
    )

    position = snapshots[0].cache_position
    repeat_logits: list[torch.Tensor] = []
    repeat_slots: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
    per_repeat: list[dict[str, Any]] = []
    memory_before = int(torch.cuda.memory_allocated())
    rng_before = _rng_state()
    try:
        for repeat in range(2):
            _restore_all_cache(cache, snapshots)
            for layer_idx in range(len(snapshots)):
                _cache_tensor(cache, "self", layer_idx, "keys")[..., position : position + 1, :].fill_(float("nan"))
                _cache_tensor(cache, "self", layer_idx, "values")[..., position : position + 1, :].fill_(float("nan"))
            captured.graph.replay()
            torch.cuda.synchronize()
            logits = _model_logits(captured.outputs).cpu()
            slots: list[tuple[torch.Tensor, torch.Tensor]] = []
            active_writes = True
            outside_untouched = True
            cross_unchanged = True
            ownership = True
            caches_finite = True
            for snapshot in snapshots:
                layer_idx = snapshot.layer_idx
                keys = _cache_tensor(cache, "self", layer_idx, "keys")
                values = _cache_tensor(cache, "self", layer_idx, "values")
                key_slot = keys[..., position : position + 1, :]
                value_slot = values[..., position : position + 1, :]
                active_writes &= _is_finite(key_slot) and _is_finite(value_slot)
                outside_untouched &= bool(
                    torch.equal(keys[..., :position, :], snapshot.tensors["self_keys"].value[..., :position, :])
                    and torch.equal(keys[..., position + 1 :, :], snapshot.tensors["self_keys"].value[..., position + 1 :, :])
                    and torch.equal(values[..., :position, :], snapshot.tensors["self_values"].value[..., :position, :])
                    and torch.equal(values[..., position + 1 :, :], snapshot.tensors["self_values"].value[..., position + 1 :, :])
                )
                cross_keys = _cache_tensor(cache, "cross", layer_idx, "keys")
                cross_values = _cache_tensor(cache, "cross", layer_idx, "values")
                cross_unchanged &= bool(
                    torch.equal(cross_keys, snapshot.tensors["cross_keys"].value)
                    and torch.equal(cross_values, snapshot.tensors["cross_values"].value)
                )
                ownership &= all(cache_ownership_matches(cache, snapshot).values())
                caches_finite &= all(
                    _is_finite(tensor)
                    for tensor in (keys, values, cross_keys, cross_values)
                )
                slots.append((key_slot.detach().cpu().clone(), value_slot.detach().cpu().clone()))
            repeat_logits.append(logits)
            repeat_slots.append(slots)
            per_repeat.append(
                {
                    "repeat": repeat,
                    "logits_finite": _is_finite(logits),
                    "active_slot_writes_valid": bool(active_writes),
                    "future_slots_untouched": bool(outside_untouched),
                    "cross_cache_unchanged": bool(cross_unchanged),
                    "storage_ownership_valid": bool(ownership),
                    "caches_finite": bool(caches_finite),
                    "logits_sha256": _sha256_tensor(logits),
                }
            )
    finally:
        _restore_all_cache(cache, snapshots)
    rng_after = _rng_state()
    rng_unchanged = rng_before == rng_after
    memory_after = int(torch.cuda.memory_allocated())
    logits_repeat = bool(torch.equal(repeat_logits[0], repeat_logits[1]))
    slots_repeat = all(
        torch.equal(first_key, second_key) and torch.equal(first_value, second_value)
        for (first_key, first_value), (second_key, second_value) in zip(
            repeat_slots[0], repeat_slots[1]
        )
    )
    checks = {
        "finite_outputs": all(
            row["logits_finite"] and row["caches_finite"] for row in per_repeat
        ),
        "cache_shapes_valid": _cache_shapes_valid(cache),
        "active_slot_writes_valid": all(
            row["active_slot_writes_valid"] for row in per_repeat
        ),
        "future_slots_untouched": all(
            row["future_slots_untouched"] for row in per_repeat
        ),
        "cross_cache_unchanged": all(
            row["cross_cache_unchanged"] for row in per_repeat
        ),
        "storage_ownership_valid": all(
            row["storage_ownership_valid"] for row in per_repeat
        ),
        "candidate_repeat_deterministic": bool(
            logits_repeat and slots_repeat and rng_unchanged
        ),
        "graph_repeat_deterministic": bool(logits_repeat and slots_repeat),
        "memory_stable": memory_before == memory_after,
    }
    return checks, {
        "repeats": per_repeat,
        "rng_before": rng_before,
        "rng_after": rng_after,
        "rng_unchanged": rng_unchanged,
        "memory_before_bytes": memory_before,
        "memory_after_bytes": memory_after,
        "self_slots_repeat_exact": slots_repeat,
        "logits_repeat_exact": logits_repeat,
    }


def _observe_full_graph(
    captured: CapturedGraph,
    *,
    cache: Any,
    snapshots: list[Any],
) -> FullObservation:
    position = snapshots[0].cache_position
    try:
        _restore_all_cache(cache, snapshots)
        for snapshot in snapshots:
            layer_idx = snapshot.layer_idx
            _cache_tensor(cache, "self", layer_idx, "keys")[
                ..., position : position + 1, :
            ].fill_(float("nan"))
            _cache_tensor(cache, "self", layer_idx, "values")[
                ..., position : position + 1, :
            ].fill_(float("nan"))
        captured.graph.replay()
        torch.cuda.synchronize()
        return FullObservation(
            logits=_model_logits(captured.outputs).cpu().clone(),
            key_slots=[
                _cache_tensor(cache, "self", snapshot.layer_idx, "keys")[
                    ..., position : position + 1, :
                ]
                .float()
                .cpu()
                .clone()
                for snapshot in snapshots
            ],
            value_slots=[
                _cache_tensor(cache, "self", snapshot.layer_idx, "values")[
                    ..., position : position + 1, :
                ]
                .float()
                .cpu()
                .clone()
                for snapshot in snapshots
            ],
        )
    finally:
        _restore_all_cache(cache, snapshots)


def _observe_prefix_graph(
    captured: CapturedGraph,
    *,
    capture: LayerCapture,
) -> PrefixObservation:
    from osuT5.osuT5.inference.optimized.scout.verifier import (
        restore_cache_state,
        snapshot_cache_state,
    )

    snapshot = snapshot_cache_state(
        capture.past_key_value,
        layer_idx=capture.layer_idx,
        cache_position=capture.cache_position,
    )
    position = snapshot.cache_position
    try:
        restore_cache_state(capture.past_key_value, snapshot)
        keys = _cache_tensor(capture.past_key_value, "self", capture.layer_idx, "keys")
        values = _cache_tensor(
            capture.past_key_value,
            "self",
            capture.layer_idx,
            "values",
        )
        keys[..., position : position + 1, :].fill_(float("nan"))
        values[..., position : position + 1, :].fill_(float("nan"))
        captured.graph.replay()
        torch.cuda.synchronize()
        if not isinstance(captured.outputs, torch.Tensor):
            raise TypeError("prefix graph output must be a tensor")
        return PrefixObservation(
            output=captured.outputs.detach().float().cpu().clone(),
            key_slot=keys[..., position : position + 1, :].float().cpu().clone(),
            value_slot=values[..., position : position + 1, :].float().cpu().clone(),
        )
    finally:
        restore_cache_state(capture.past_key_value, snapshot)


def _max_abs(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    if reference.shape != candidate.shape:
        return math.inf
    return float((reference.float() - candidate.float()).abs().max().item())


def _observation_drift(
    reference_full: FullObservation,
    candidate_full: FullObservation,
    reference_prefix: PrefixObservation,
    candidate_prefix: PrefixObservation,
) -> tuple[dict[str, bool], dict[str, float]]:
    key_max = max(
        _max_abs(reference, candidate)
        for reference, candidate in zip(
            reference_full.key_slots,
            candidate_full.key_slots,
            strict=True,
        )
    )
    value_max = max(
        _max_abs(reference, candidate)
        for reference, candidate in zip(
            reference_full.value_slots,
            candidate_full.value_slots,
            strict=True,
        )
    )
    layer_max = _max_abs(reference_prefix.output, candidate_prefix.output)
    logits_max = _max_abs(reference_full.logits, candidate_full.logits)
    drift = {
        "layer_output_max_abs": layer_max,
        "cache_key_slot_max_abs": key_max,
        "cache_value_slot_max_abs": value_max,
        "logits_max_abs": logits_max,
    }
    return (
        {
            "layer_output_allclose": layer_max <= DRIFT_LIMIT,
            "cache_slot_allclose": max(key_max, value_max) <= DRIFT_LIMIT,
            "logits_allclose": logits_max <= DRIFT_LIMIT,
        },
        drift,
    )


def _capture_representative_layer(
    model: torch.nn.Module,
    static_inputs: dict[str, Any],
    *,
    prefix: int,
) -> LayerCapture:
    from osuT5.osuT5.inference.optimized.single.runtime_context import (
        active_prefix_self_attention_context,
    )
    from osuT5.osuT5.model.custom_transformers.modeling_varwhisper import (
        VarWhisperDecoderLayer,
    )

    found: dict[str, LayerCapture] = {}

    def hook(module, inputs, kwargs, output):
        del output
        if found:
            return
        hidden = kwargs.get("hidden_states", inputs[0] if inputs else None)
        encoder = kwargs.get("encoder_hidden_states")
        cache_position = kwargs.get("cache_position")
        position_ids = kwargs.get("position_ids")
        cache = kwargs.get("past_key_value")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (hidden, encoder, cache_position, position_ids)
        ) or cache is None:
            return
        layer_idx = int(module.self_attn.layer_idx)
        found["capture"] = LayerCapture(
            module=module,
            layer_idx=layer_idx,
            hidden_states=hidden.detach().clone(),
            attention_mask=_clone_tensor(kwargs.get("attention_mask")),
            encoder_hidden_states=encoder.detach().clone(),
            past_key_value=cache,
            cache_position=cache_position.detach().clone(),
            position_ids=position_ids.detach().clone(),
            active_prefix_length=prefix,
        )

    handle = None
    for module in model.modules():
        if isinstance(module, VarWhisperDecoderLayer):
            handle = module.register_forward_hook(hook, with_kwargs=True)
            break
    if handle is None:
        raise RuntimeError("could not find a VarWhisperDecoderLayer")
    try:
        with active_prefix_self_attention_context(prefix):
            model(**static_inputs, return_dict=True)
        torch.cuda.synchronize()
    finally:
        handle.remove()
    if "capture" not in found:
        raise RuntimeError(f"did not capture real decoder-layer inputs for prefix {prefix}")
    return found["capture"]


def _prefix_callable(capture: LayerCapture, *, variant: str) -> Callable[[], torch.Tensor]:
    from osuT5.osuT5.inference.optimized.scout.native_prefix import (
        accepted_fp32_prefix,
        framework_prefix,
        native_prefix,
    )

    functions = {
        "accepted": accepted_fp32_prefix,
        "framework": framework_prefix,
        "native": native_prefix,
    }
    try:
        fn = functions[variant]
    except KeyError as exc:
        raise ValueError(f"unknown prefix variant {variant!r}") from exc

    def run() -> torch.Tensor:
        return fn(
            capture.module,
            hidden_states=capture.hidden_states,
            attention_mask=capture.attention_mask,
            encoder_hidden_states=capture.encoder_hidden_states,
            past_key_value=capture.past_key_value,
            cache_position=capture.cache_position,
            position_ids=capture.position_ids,
            active_prefix_length=capture.active_prefix_length,
        )

    return run


def _prefix_graphs(
    capture: LayerCapture,
    *,
    include_accepted: bool,
    warmup: int,
) -> tuple[dict[str, CapturedGraph], dict[str, Any]]:
    from osuT5.osuT5.inference.optimized.scout.verifier import (
        verify_candidate_cache_behavior,
    )

    cache = capture.past_key_value
    names = ("accepted", "framework", "native") if include_accepted else (
        "framework",
        "native",
    )
    graphs: dict[str, CapturedGraph] = {}
    verifiers: dict[str, Any] = {}
    for name in names:
        fn = _prefix_callable(capture, variant=name)
        verifiers[name] = verify_candidate_cache_behavior(
            cache,
            layer_idx=capture.layer_idx,
            cache_position=capture.cache_position,
            candidate=fn,
            repeats=2,
        )
        graphs[name] = _capture_cuda_graph(fn, context=nullcontext, warmup=0)
    return graphs, verifiers


def _accepted_main_session_run(args, *, output_path: Path):
    from inference import (
        compile_args,
        generate,
        get_config,
        load_model_with_engine,
        setup_inference_environment,
    )
    from osuT5.osuT5.inference import Processor

    compile_args(args, verbose=False)
    setup_inference_environment(args.seed)
    binding, tokenizer = load_model_with_engine(
        args.model_path,
        args.train,
        args.device,
        max_batch_size=args.max_batch_size,
        use_server=args.use_server,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
        lora_path=args.lora_path,
        gamemode=args.gamemode,
        auto_select_gamemode_model=args.auto_select_gamemode_model,
        inference_engine=args.inference_engine,
    )
    generation_config, beatmap_config = get_config(args)
    captured: dict[str, Any] = {}
    original_generate = Processor.generate

    def wrapped_generate(processor, *positional, **kwargs):
        result = original_generate(processor, *positional, **kwargs)
        if kwargs.get("profile_label") == "main_generation":
            if "processor" in captured:
                raise RuntimeError("captured more than one main Processor")
            captured["processor"] = processor
            captured["session"] = processor.decode_session_state
        return result

    Processor.generate = wrapped_generate
    try:
        generated, result_path = generate(
            args,
            output_path=str(output_path),
            generation_config=generation_config,
            beatmap_config=beatmap_config,
            model=binding,
            tokenizer=tokenizer,
            timing_model=binding,
            timing_tokenizer=tokenizer,
            verbose=False,
        )
    finally:
        Processor.generate = original_generate
    if "session" not in captured or captured["session"] is None:
        raise RuntimeError("full SALVALAI run did not expose a main ProductionDecodeSession")
    processor = captured["processor"]
    return {
        "model": processor.model,
        "tokenizer": tokenizer,
        "processor": processor,
        "session": captured["session"],
        "generated": generated,
        "result_path": str(result_path),
    }


def _assert_scout_args(args) -> None:
    requirements = {
        "inference_engine": "optimized",
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "device": "cuda",
        "use_server": False,
        "parallel": False,
        "cfg_scale": 1.0,
        "num_beams": 1,
    }
    failures = {
        name: (getattr(args, name), expected)
        for name, expected in requirements.items()
        if getattr(args, name) != expected
    }
    if failures:
        raise ValueError(f"native-prefix dtype scout requires {requirements}; got {failures}")


def _bucket_entry(
    *,
    full_ms: float,
    setup_seconds: float,
    prefix_ms: float,
    checks: dict[str, bool],
    drift: dict[str, float],
    details: dict[str, Any],
) -> dict[str, Any]:
    missing = [name for name in REQUIRED_CHECKS if name not in checks]
    if missing:
        raise RuntimeError(f"bucket checks missing {missing}")
    values = (full_ms, setup_seconds, prefix_ms)
    if not all(math.isfinite(float(value)) for value in values):
        raise RuntimeError("bucket timing contains non-finite values")
    if full_ms <= 0 or setup_seconds < 0 or prefix_ms <= 0:
        raise RuntimeError("bucket timing violates positive/non-negative schema")
    required_drift = {
        "layer_output_max_abs",
        "cache_key_slot_max_abs",
        "cache_value_slot_max_abs",
        "logits_max_abs",
    }
    if set(drift) != required_drift or not all(
        math.isfinite(float(value)) and float(value) >= 0 for value in drift.values()
    ):
        raise RuntimeError("bucket drift fields are missing, negative, or non-finite")
    return {
        "full_model_replay_ms_per_call": float(full_ms),
        "capture_setup_seconds": float(setup_seconds),
        "prefix_replay_ms_per_layer": float(prefix_ms),
        "checks": {name: bool(checks[name]) for name in REQUIRED_CHECKS},
        "drift": {name: float(value) for name, value in drift.items()},
        "details": details,
    }


@torch.no_grad()
def profile_native_prefix_dtype_scout(
    args,
    *,
    output_path: Path,
    bucket_mode: str,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("native-prefix dtype scout requires CUDA")
    if warmup < 1 or iters < 1:
        raise ValueError("warmup and iters must be positive")
    _assert_scout_args(args)
    output_path.mkdir(parents=True, exist_ok=True)
    run = _accepted_main_session_run(args, output_path=output_path)
    model = run["model"]
    model.eval()
    accepted_entries = validate_accepted_graph_cache(run["session"].graph_cache)
    selected = select_buckets(accepted_entries, bucket_mode)
    measured_prefixes = tuple(selected)
    static_inputs = {prefix: entry["static_inputs"] for prefix, entry in selected.items()}
    variants = {name: {"buckets": {}} for name in VARIANTS}
    raw_buckets: dict[str, Any] = {}
    fp32_references: dict[int, tuple[FullObservation, PrefixObservation]] = {}

    # Extension build/load is a scout preflight, not a per-signature graph cost.
    from osuT5.osuT5.inference.optimized.scout.q1_attention import (
        preload_native_q1_attention,
    )
    torch.cuda.synchronize()
    extension_started = time.perf_counter()
    preload_native_q1_attention()
    torch.cuda.synchronize()
    extension_preload_seconds = time.perf_counter() - extension_started

    # FP32 accepted/framework/native graphs are timed together before dtype conversion.
    for prefix, accepted in selected.items():
        inputs = static_inputs[prefix]
        cache = _cache_from_static_inputs(inputs)
        cache_position = inputs.get("cache_position")
        if not isinstance(cache_position, torch.Tensor) or cache_position.numel() != 1:
            raise RuntimeError(f"prefix {prefix} has invalid cache_position")
        snapshots = _all_cache_snapshots(cache, cache_position)
        capture = _capture_representative_layer(model, inputs, prefix=prefix)
        _restore_all_cache(cache, snapshots)
        framework = _capture_cuda_graph(
            lambda inputs=inputs: model(**inputs, return_dict=True),
            context=lambda prefix=prefix: _full_model_variant_context(
                model=model, prefix=prefix, variant="framework"
            ),
            warmup=0,
        )
        _restore_all_cache(cache, snapshots)
        native = _capture_cuda_graph(
            lambda inputs=inputs: model(**inputs, return_dict=True),
            context=lambda prefix=prefix: _full_model_variant_context(
                model=model, prefix=prefix, variant="native"
            ),
            warmup=0,
        )
        accepted_graph = CapturedGraph(
            graph=accepted["graph"],
            outputs=accepted["outputs"],
            setup_seconds=float(accepted.get("capture_seconds", 0.0)),
            peak_vram_bytes=0,
        )
        full_graphs = {
            "fp32_accepted": accepted_graph,
            "fp32_framework": framework,
            "fp32_native_prefix": native,
        }
        full_checks: dict[str, dict[str, bool]] = {}
        full_details: dict[str, Any] = {}
        for name, graph in full_graphs.items():
            full_checks[name], full_details[name] = _verify_full_graph(
                graph, cache=cache, snapshots=snapshots
            )
        medians, rounds, memory = _reciprocal_graph_rounds(
            {name: graph.graph for name, graph in full_graphs.items()},
            restore=lambda: _restore_all_cache(cache, snapshots),
            warmup=warmup,
            iters=iters,
        )
        prefix_graphs, prefix_verifiers = _prefix_graphs(
            capture,
            include_accepted=True,
            warmup=warmup,
        )
        prefix_snapshot = _all_cache_snapshots(cache, cache_position)
        prefix_medians, prefix_rounds, prefix_memory = _reciprocal_graph_rounds(
            {name: graph.graph for name, graph in prefix_graphs.items()},
            restore=lambda: _restore_all_cache(cache, prefix_snapshot),
            warmup=warmup,
            iters=iters,
        )
        full_observations = {
            name: _observe_full_graph(graph, cache=cache, snapshots=snapshots)
            for name, graph in full_graphs.items()
        }
        prefix_observations = {
            name: _observe_prefix_graph(graph, capture=capture)
            for name, graph in prefix_graphs.items()
        }
        fp32_references[prefix] = (
            full_observations["fp32_accepted"],
            prefix_observations["accepted"],
        )
        prefix_by_variant = {
            "fp32_accepted": prefix_medians["accepted"],
            "fp32_framework": prefix_medians["framework"],
            "fp32_native_prefix": prefix_medians["native"],
        }
        verifier_by_variant = {
            "fp32_accepted": prefix_verifiers["accepted"],
            "fp32_framework": prefix_verifiers["framework"],
            "fp32_native_prefix": prefix_verifiers["native"],
        }
        for name, graph in full_graphs.items():
            checks = dict(full_checks[name])
            prefix_name = (
                "accepted"
                if name == "fp32_accepted"
                else "native"
                if name.endswith("native_prefix")
                else "framework"
            )
            drift_checks, drift = _observation_drift(
                full_observations["fp32_accepted"],
                full_observations[name],
                prefix_observations["accepted"],
                prefix_observations[prefix_name],
            )
            checks.update(drift_checks)
            checks["candidate_repeat_deterministic"] &= bool(
                verifier_by_variant[name]["pass"]
            )
            checks["memory_stable"] &= bool(memory[name])
            variants[name]["buckets"][str(prefix)] = _bucket_entry(
                full_ms=medians[name],
                setup_seconds=graph.setup_seconds,
                prefix_ms=prefix_by_variant[name],
                checks=checks,
                drift=drift,
                details={
                    "dtype": "torch.float32",
                    "decode_replays": int(accepted["decode_replays"]),
                    "cache_position": int(cache_position.item()),
                    "full_graph": full_details[name],
                    "prefix_verifier": verifier_by_variant[name],
                    "capture_peak_vram_bytes": graph.peak_vram_bytes,
                    "timing_memory_stable": memory[name],
                    "prefix_timing_memory_stable": prefix_memory[
                        prefix_name
                    ],
                },
            )
        raw_buckets[str(prefix)] = {
            "fp32_full_rounds": rounds,
            "fp32_prefix_rounds": prefix_rounds,
            "static_inputs": _static_input_metadata(inputs),
        }

    # Accepted graphs have been fully measured; release them before changing dtype.
    # CUDA graphs retain private pools and static tensor references. Release all
    # FP32 graph/snapshot containers before converting model/cache storage.
    run["session"].graph_cache.clear()
    accepted_entries.clear()
    selected.clear()
    del (
        accepted,
        accepted_graph,
        full_graphs,
        framework,
        native,
        prefix_graphs,
        prefix_snapshot,
        snapshots,
        graph,
    )
    gc.collect()
    torch.cuda.empty_cache()
    model.to(dtype=torch.float16)
    static_inputs = {
        prefix: convert_static_inputs_dtype(inputs, torch.float16)
        for prefix, inputs in static_inputs.items()
    }

    for prefix, inputs in static_inputs.items():
        cache = _cache_from_static_inputs(inputs)
        cache_position = inputs["cache_position"]
        snapshots = _all_cache_snapshots(cache, cache_position)
        capture = _capture_representative_layer(model, inputs, prefix=prefix)
        _restore_all_cache(cache, snapshots)
        framework = _capture_cuda_graph(
            lambda inputs=inputs: model(**inputs, return_dict=True),
            context=lambda prefix=prefix: _full_model_variant_context(
                model=model, prefix=prefix, variant="framework"
            ),
            warmup=0,
        )
        _restore_all_cache(cache, snapshots)
        native = _capture_cuda_graph(
            lambda inputs=inputs: model(**inputs, return_dict=True),
            context=lambda prefix=prefix: _full_model_variant_context(
                model=model, prefix=prefix, variant="native"
            ),
            warmup=0,
        )
        full_graphs = {
            "fp16_framework": framework,
            "fp16_native_prefix": native,
        }
        full_checks: dict[str, dict[str, bool]] = {}
        full_details: dict[str, Any] = {}
        for name, graph in full_graphs.items():
            full_checks[name], full_details[name] = _verify_full_graph(
                graph, cache=cache, snapshots=snapshots
            )
        medians, rounds, memory = _reciprocal_graph_rounds(
            {name: graph.graph for name, graph in full_graphs.items()},
            restore=lambda: _restore_all_cache(cache, snapshots),
            warmup=warmup,
            iters=iters,
        )
        prefix_graphs, prefix_verifiers = _prefix_graphs(
            capture,
            include_accepted=False,
            warmup=warmup,
        )
        prefix_snapshot = _all_cache_snapshots(cache, cache_position)
        prefix_medians, prefix_rounds, prefix_memory = _reciprocal_graph_rounds(
            {name: graph.graph for name, graph in prefix_graphs.items()},
            restore=lambda: _restore_all_cache(cache, prefix_snapshot),
            warmup=warmup,
            iters=iters,
        )
        full_observations = {
            name: _observe_full_graph(graph, cache=cache, snapshots=snapshots)
            for name, graph in full_graphs.items()
        }
        prefix_observations = {
            name: _observe_prefix_graph(graph, capture=capture)
            for name, graph in prefix_graphs.items()
        }
        reference_full, reference_prefix = fp32_references[prefix]
        for name, graph in full_graphs.items():
            prefix_name = "native" if name.endswith("native_prefix") else "framework"
            checks = dict(full_checks[name])
            drift_checks, drift = _observation_drift(
                reference_full,
                full_observations[name],
                reference_prefix,
                prefix_observations[prefix_name],
            )
            checks.update(drift_checks)
            checks["candidate_repeat_deterministic"] &= bool(
                prefix_verifiers[prefix_name]["pass"]
            )
            checks["memory_stable"] &= bool(memory[name])
            variants[name]["buckets"][str(prefix)] = _bucket_entry(
                full_ms=medians[name],
                setup_seconds=graph.setup_seconds,
                prefix_ms=prefix_medians[prefix_name],
                checks=checks,
                drift=drift,
                details={
                    "dtype": "torch.float16",
                    "decode_replays": BUCKET_COUNTS[prefix],
                    "cache_position": int(cache_position.item()),
                    "logits_cast_outside_graph": "torch.float32",
                    "full_graph": full_details[name],
                    "prefix_verifier": prefix_verifiers[prefix_name],
                    "capture_peak_vram_bytes": graph.peak_vram_bytes,
                    "timing_memory_stable": memory[name],
                    "prefix_timing_memory_stable": prefix_memory[prefix_name],
                },
            )
        raw_buckets[str(prefix)]["fp16_full_rounds"] = rounds
        raw_buckets[str(prefix)]["fp16_prefix_rounds"] = prefix_rounds
        raw_buckets[str(prefix)]["fp16_static_inputs"] = _static_input_metadata(inputs)

    return {
        "schema_version": 1,
        "variants": variants,
        "metadata": {
            "result_class": "documented-drift-component-scout",
            "source_engine": "optimized",
            "source_precision": "fp32",
            "source_attn_implementation": "sdpa",
            "bucket_mode": bucket_mode,
            "measured_buckets": list(measured_prefixes),
            "accepted_bucket_replay_counts": BUCKET_COUNTS,
            "warmup": warmup,
            "iters": iters,
            "full_salvalai_runs": 1,
            "full_model_metric": "full one-token model CUDA graph replay",
            "extension_preload_seconds": extension_preload_seconds,
            "result_path": run["result_path"],
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(),
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "raw": {"buckets": raw_buckets},
    }


def _static_input_metadata(static_inputs: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in static_inputs.items():
        if isinstance(value, torch.Tensor):
            metadata[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
        else:
            metadata[key] = {"type": type(value).__name__}
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--bucket-mode", choices=("sentinel", "all"), default="sentinel")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()
    overrides = list(cli.overrides)
    if not any(value.startswith("output_path=") for value in overrides):
        overrides.append(f"output_path={cli.output_path}")
    args = _load_args(cli.config_name, overrides)
    result = profile_native_prefix_dtype_scout(
        args,
        output_path=cli.output_path,
        bucket_mode=cli.bucket_mode,
        warmup=cli.warmup,
        iters=cli.iters,
    )
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["metadata"], indent=2))
    raise SystemExit(0 if all(
        entry["checks"][check]
        for variant in result["variants"].values()
        for entry in variant["buckets"].values()
        for check in REQUIRED_CHECKS
    ) else 1)


if __name__ == "__main__":
    main()
