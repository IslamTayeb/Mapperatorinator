"""Size exact-shape and bucket-64 decoder-prefill CUDA graphs on real inputs.

This is an opt-in component scout.  It never changes the optimized selector or
the production decoder.  A clean accepted full-song run establishes the fixed
work control; a second run captures the 87 main-generation prefill inputs.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from transformers.modeling_outputs import BaseModelOutput  # noqa: E402

from utils.profile_native_prefix_dtype_scout import (  # noqa: E402
    _assert_scout_args,
    _load_args,
)


SCHEMA_VERSION = 1
EXPECTED_PREFILLS = 87
BUCKET_SIZE = 64
DEFAULT_WARMUP = 3
DEFAULT_ITERATIONS = 5


@dataclass(slots=True)
class PrefillRecord:
    index: int
    decoder_input_ids: torch.Tensor
    decoder_attention_mask: torch.Tensor | None
    attention_mask: torch.Tensor | None
    encoder_hidden_state: torch.Tensor
    use_cache: bool

    @property
    def prompt_length(self) -> int:
        return int(self.decoder_input_ids.shape[1])

    @property
    def source_length(self) -> int:
        return int(self.encoder_hidden_state.shape[1])


@dataclass(slots=True)
class Observation:
    logits: torch.Tensor
    self_keys: list[torch.Tensor]
    self_values: list[torch.Tensor]
    cross_keys: list[torch.Tensor]
    cross_values: list[torch.Tensor]


@dataclass(slots=True)
class CapturedPrefillGraph:
    graph: torch.cuda.CUDAGraph
    outputs: Any
    static_inputs: dict[str, Any]
    cache: Any
    capture_seconds: float
    allocated_delta_bytes: int
    reserved_delta_bytes: int


@dataclass(slots=True)
class RngSnapshot:
    cpu: torch.Tensor
    cuda: tuple[torch.Tensor, ...]


def snapshot_rng_state() -> RngSnapshot:
    """Capture every generator consumed by the accepted sampling path."""

    return RngSnapshot(
        cpu=torch.random.get_rng_state().clone(),
        cuda=(
            tuple(state.clone() for state in torch.cuda.get_rng_state_all())
            if torch.cuda.is_available()
            else ()
        ),
    )


def restore_rng_state(snapshot: RngSnapshot) -> None:
    """Restore a reciprocal song run to the exact CPU and CUDA RNG state."""

    if not isinstance(snapshot, RngSnapshot):
        raise TypeError("prefill reciprocal RNG state must be an RngSnapshot")
    torch.random.set_rng_state(snapshot.cpu.clone())
    if snapshot.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("captured CUDA RNG state cannot be restored without CUDA")
        device_count = torch.cuda.device_count()
        if len(snapshot.cuda) != device_count:
            raise RuntimeError(
                "captured CUDA RNG state count changed: "
                f"captured={len(snapshot.cuda)} current={device_count}"
            )
        torch.cuda.set_rng_state_all([state.clone() for state in snapshot.cuda])


def load_timing_pair(
    args: Any,
    *,
    loader,
    needs_separate,
    main_binding: Any,
    main_tokenizer: Any,
) -> tuple[Any | None, Any | None]:
    """Mirror production timing-model selection without aliasing the main model."""

    if not needs_separate(args):
        return None, None
    timing_model, timing_tokenizer = loader(
        args.model_path,
        args.train,
        args.device,
        max_batch_size=args.max_batch_size,
        use_server=args.use_server,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
        gamemode=args.gamemode,
        auto_select_gamemode_model=False,
        inference_engine=args.inference_engine,
    )
    if timing_model is None or timing_tokenizer is None:
        raise RuntimeError("separate timing loader returned an incomplete model/tokenizer pair")
    if timing_model is main_binding:
        raise RuntimeError("separate timing model loader aliased the main model binding")
    if timing_tokenizer is main_tokenizer:
        raise RuntimeError("separate timing tokenizer loader aliased the main tokenizer")
    return timing_model, timing_tokenizer


def bucket_prompt_length(length: int, *, bucket_size: int = BUCKET_SIZE) -> int:
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("prompt length must be a positive integer")
    if isinstance(bucket_size, bool) or not isinstance(bucket_size, int) or bucket_size <= 0:
        raise ValueError("bucket size must be a positive integer")
    return ((length + bucket_size - 1) // bucket_size) * bucket_size


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _snapshot_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").clone(memory_format=torch.contiguous_format)


def _snapshot_encoder(value: Any) -> torch.Tensor:
    if not isinstance(value, BaseModelOutput):
        raise TypeError(f"prefill encoder_outputs must be BaseModelOutput, got {type(value).__name__}")
    hidden = value.last_hidden_state
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise TypeError("prefill encoder last_hidden_state must be a rank-3 tensor")
    return _snapshot_tensor(hidden)


def snapshot_prefill_record(
    *,
    index: int,
    decoder_input_ids: torch.Tensor,
    kwargs: dict[str, Any],
) -> PrefillRecord:
    if decoder_input_ids.ndim != 2 or decoder_input_ids.shape[0] != 1:
        raise ValueError("prefill decoder_input_ids must have shape [1, prompt]")
    decoder_mask = kwargs.get("decoder_attention_mask")
    attention_mask = kwargs.get("attention_mask")
    for name, value in (
        ("decoder_attention_mask", decoder_mask),
        ("attention_mask", attention_mask),
    ):
        if value is not None and not isinstance(value, torch.Tensor):
            raise TypeError(f"prefill {name} must be a tensor or None")
    return PrefillRecord(
        index=index,
        decoder_input_ids=_snapshot_tensor(decoder_input_ids),
        decoder_attention_mask=(None if decoder_mask is None else _snapshot_tensor(decoder_mask)),
        attention_mask=(None if attention_mask is None else _snapshot_tensor(attention_mask)),
        encoder_hidden_state=_snapshot_encoder(kwargs.get("encoder_outputs")),
        use_cache=bool(kwargs.get("use_cache", True)),
    )


@contextmanager
def capture_main_prefills(processor: Any, records: list[PrefillRecord]) -> Iterator[None]:
    """Temporarily observe prepared main prefills without changing decoder forward."""

    model = processor.model
    original = model.prepare_inputs_for_generation

    def wrapped(self, decoder_input_ids, *args, **kwargs):
        prepared = original(decoder_input_ids, *args, **kwargs)
        prepared_ids = prepared.get("decoder_input_ids")
        if isinstance(prepared_ids, torch.Tensor) and prepared_ids.shape[1] > 1:
            records.append(
                snapshot_prefill_record(
                    index=len(records),
                    decoder_input_ids=decoder_input_ids,
                    kwargs=kwargs,
                )
            )
        return prepared

    model.prepare_inputs_for_generation = MethodType(wrapped, model)
    try:
        yield
    finally:
        model.prepare_inputs_for_generation = original


def _to_device_tensor(value: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    target_dtype = dtype if value.is_floating_point() else value.dtype
    return value.to(device=device, dtype=target_dtype).contiguous()


def _fresh_cache(model: torch.nn.Module):
    from osuT5.osuT5.inference.cache_utils import get_cache

    return get_cache(model, batch_size=1, num_beams=1, cfg_scale=1.0)


def prepare_record_inputs(
    model: torch.nn.Module,
    record: PrefillRecord,
    *,
    padded_length: int,
    pad_token_id: int,
) -> tuple[dict[str, Any], int]:
    if padded_length < record.prompt_length:
        raise ValueError("padded length cannot be smaller than prompt length")
    device = torch.device(model.device)
    dtype = model.dtype
    pad = padded_length - record.prompt_length
    decoder_ids = _to_device_tensor(record.decoder_input_ids, device=device, dtype=dtype)
    decoder_mask = (
        torch.ones_like(decoder_ids, dtype=torch.long)
        if record.decoder_attention_mask is None
        else _to_device_tensor(record.decoder_attention_mask, device=device, dtype=dtype)
    )
    if decoder_mask.ndim != 2 or decoder_mask.shape != decoder_ids.shape:
        raise ValueError("raw decoder attention mask must match decoder_input_ids")
    if pad:
        decoder_ids = torch.nn.functional.pad(decoder_ids, (pad, 0), value=pad_token_id)
        decoder_mask = torch.nn.functional.pad(decoder_mask, (pad, 0), value=0)
    attention_mask = (
        None
        if record.attention_mask is None
        else _to_device_tensor(record.attention_mask, device=device, dtype=dtype)
    )
    encoder_outputs = BaseModelOutput(
        last_hidden_state=_to_device_tensor(
            record.encoder_hidden_state,
            device=device,
            dtype=dtype,
        )
    )
    cache = _fresh_cache(model)
    cache_position = torch.arange(padded_length, device=device, dtype=torch.long)
    prepared = model.prepare_inputs_for_generation(
        decoder_ids,
        past_key_values=cache,
        use_cache=record.use_cache,
        encoder_outputs=encoder_outputs,
        attention_mask=attention_mask,
        decoder_attention_mask=decoder_mask,
        cache_position=cache_position,
    )
    if prepared.get("past_key_values") is not cache:
        raise RuntimeError("prepared prefill did not retain the fresh static cache")
    return prepared, pad


def prefill_signature(record: PrefillRecord, *, mode: str) -> tuple[Any, ...]:
    if mode == "exact":
        prompt = record.prompt_length
    elif mode == "bucket64":
        prompt = bucket_prompt_length(record.prompt_length)
    else:
        raise ValueError(f"unsupported prefill bucket mode {mode!r}")
    return (
        prompt,
        tuple(record.encoder_hidden_state.shape),
        None if record.attention_mask is None else tuple(record.attention_mask.shape),
    )


def _clone_graph_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone(memory_format=torch.contiguous_format)
    if isinstance(value, BaseModelOutput):
        return BaseModelOutput(last_hidden_state=_clone_graph_value(value.last_hidden_state))
    return value


def clone_graph_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {key: _clone_graph_value(value) for key, value in inputs.items()}


def _copy_graph_value(target: Any, source: Any, *, name: str) -> tuple[int, int]:
    if isinstance(target, torch.Tensor):
        if not isinstance(source, torch.Tensor):
            raise TypeError(f"graph input {name} changed from tensor to {type(source).__name__}")
        if target.shape != source.shape or target.dtype != source.dtype or target.device != source.device:
            raise ValueError(f"graph input {name} shape/dtype/device changed")
        target.copy_(source)
        return target.numel() * target.element_size(), 1
    if isinstance(target, BaseModelOutput):
        if not isinstance(source, BaseModelOutput):
            raise TypeError(f"graph input {name} changed BaseModelOutput type")
        return _copy_graph_value(
            target.last_hidden_state,
            source.last_hidden_state,
            name=f"{name}.last_hidden_state",
        )
    return 0, 0


def copy_graph_inputs(target: dict[str, Any], source: dict[str, Any]) -> tuple[int, int]:
    if set(target) != set(source):
        raise ValueError("graph input keys changed")
    copied_bytes = 0
    copied_tensors = 0
    for key, target_value in target.items():
        if key == "past_key_values":
            continue
        byte_count, tensor_count = _copy_graph_value(target_value, source[key], name=key)
        copied_bytes += byte_count
        copied_tensors += tensor_count
    return copied_bytes, copied_tensors


def _cache_layer_tensors(cache: Any, kind: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    cache_object = (
        cache.self_attention_cache if kind == "self" else cache.cross_attention_cache
    )
    layers = getattr(cache_object, "layers", None)
    if not isinstance(layers, list) or not layers:
        raise RuntimeError(f"prefill {kind} cache exposes no layer storage")
    result = []
    for layer in layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise TypeError(f"prefill {kind} cache layer lacks tensor keys/values")
        result.append((keys, values))
    return result


def observe_prefill(
    outputs: Any,
    cache: Any,
    *,
    prompt_start: int,
    prompt_length: int,
    source_length: int,
) -> Observation:
    logits = getattr(outputs, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise TypeError("prefill outputs must expose rank-3 logits")
    if prompt_start < 0 or prompt_length <= 0 or prompt_start + prompt_length > logits.shape[1]:
        raise ValueError("prefill observation prompt slice is invalid")
    self_layers = _cache_layer_tensors(cache, "self")
    cross_layers = _cache_layer_tensors(cache, "cross")
    return Observation(
        logits=logits[:, prompt_start + prompt_length - 1, :].detach().clone(),
        self_keys=[keys[..., prompt_start : prompt_start + prompt_length, :].detach().clone() for keys, _ in self_layers],
        self_values=[values[..., prompt_start : prompt_start + prompt_length, :].detach().clone() for _, values in self_layers],
        cross_keys=[keys[..., :source_length, :].detach().clone() for keys, _ in cross_layers],
        cross_values=[values[..., :source_length, :].detach().clone() for _, values in cross_layers],
    )


def _observation_tensors(observation: Observation) -> list[torch.Tensor]:
    return [
        observation.logits,
        *observation.self_keys,
        *observation.self_values,
        *observation.cross_keys,
        *observation.cross_values,
    ]


def compare_observations(reference: Observation, candidate: Observation) -> dict[str, Any]:
    reference_tensors = _observation_tensors(reference)
    candidate_tensors = _observation_tensors(candidate)
    if len(reference_tensors) != len(candidate_tensors):
        return {"exact": False, "finite": False, "max_abs_drift": math.inf, "shape_valid": False}
    exact = True
    finite = True
    shape_valid = True
    max_abs = 0.0
    for expected, actual in zip(reference_tensors, candidate_tensors, strict=True):
        if expected.shape != actual.shape:
            shape_valid = False
            exact = False
            max_abs = math.inf
            continue
        finite &= bool(torch.isfinite(actual).all().item())
        exact &= bool(torch.equal(expected, actual))
        drift = float((expected.float() - actual.float()).abs().max().item())
        max_abs = max(max_abs, drift)
    return {"exact": exact, "finite": finite, "max_abs_drift": max_abs, "shape_valid": shape_valid}


def _cuda_elapsed_ms(fn, *, iterations: int) -> float:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    values: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return sorted(values)[len(values) // 2]


def _run_eager(model: torch.nn.Module, inputs: dict[str, Any]) -> Any:
    return model(**inputs, return_dict=True)


def capture_prefill_graph(
    model: torch.nn.Module,
    inputs: dict[str, Any],
    *,
    warmup: int,
) -> CapturedPrefillGraph:
    if warmup < 1:
        raise ValueError("graph warmup must be positive")
    for _ in range(warmup):
        _run_eager(model, inputs)
    torch.cuda.synchronize()
    static_inputs = clone_graph_inputs(inputs)
    static_cache = static_inputs.get("past_key_values")
    if static_cache is inputs.get("past_key_values"):
        # Cache objects require their own storage; tensor-only cloning is insufficient.
        static_cache = _fresh_cache(model)
        static_inputs["past_key_values"] = static_cache
    before_allocated = torch.cuda.memory_allocated()
    before_reserved = torch.cuda.memory_reserved()
    graph = torch.cuda.CUDAGraph()
    capture_start = time.perf_counter()
    with torch.cuda.graph(graph):
        outputs = _run_eager(model, static_inputs)
    torch.cuda.synchronize()
    capture_seconds = time.perf_counter() - capture_start
    return CapturedPrefillGraph(
        graph=graph,
        outputs=outputs,
        static_inputs=static_inputs,
        cache=static_cache,
        capture_seconds=capture_seconds,
        allocated_delta_bytes=max(0, torch.cuda.memory_allocated() - before_allocated),
        reserved_delta_bytes=max(0, torch.cuda.memory_reserved() - before_reserved),
    )


def _benchmark_eager(
    model: torch.nn.Module,
    record: PrefillRecord,
    *,
    pad_token_id: int,
    iterations: int,
) -> tuple[float, Observation]:
    samples: list[float] = []
    observation = None
    for _ in range(iterations):
        inputs, _ = prepare_record_inputs(
            model,
            record,
            padded_length=record.prompt_length,
            pad_token_id=pad_token_id,
        )
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs = _run_eager(model, inputs)
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
        observation = observe_prefill(
            outputs,
            inputs["past_key_values"],
            prompt_start=0,
            prompt_length=record.prompt_length,
            source_length=record.source_length,
        )
    assert observation is not None
    return sorted(samples)[len(samples) // 2], observation


def _profile_graph_group(
    model: torch.nn.Module,
    records: list[PrefillRecord],
    *,
    mode: str,
    pad_token_id: int,
    warmup: int,
    iterations: int,
    fp32_model: torch.nn.Module,
    fp32_eager_ms: dict[int, float],
) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
    representative = records[0]
    padded_length = representative.prompt_length if mode == "exact" else bucket_prompt_length(representative.prompt_length)
    representative_inputs, _ = prepare_record_inputs(
        model,
        representative,
        padded_length=padded_length,
        pad_token_id=pad_token_id,
    )
    captured = capture_prefill_graph(model, representative_inputs, warmup=warmup)
    rows: list[dict[str, Any]] = []
    for record in records:
        record_padded = record.prompt_length if mode == "exact" else bucket_prompt_length(record.prompt_length)
        inputs, pad = prepare_record_inputs(
            model,
            record,
            padded_length=record_padded,
            pad_token_id=pad_token_id,
        )
        copy_bytes = 0
        copy_tensors = 0

        def copy_inputs() -> None:
            nonlocal copy_bytes, copy_tensors
            copy_bytes, copy_tensors = copy_graph_inputs(captured.static_inputs, inputs)

        copy_ms = _cuda_elapsed_ms(copy_inputs, iterations=iterations)
        copy_inputs()
        replay_ms = _cuda_elapsed_ms(captured.graph.replay, iterations=iterations)
        captured.graph.replay()
        torch.cuda.synchronize()
        first = observe_prefill(
            captured.outputs,
            captured.cache,
            prompt_start=pad,
            prompt_length=record.prompt_length,
            source_length=record.source_length,
        )
        captured.graph.replay()
        torch.cuda.synchronize()
        second = observe_prefill(
            captured.outputs,
            captured.cache,
            prompt_start=pad,
            prompt_length=record.prompt_length,
            source_length=record.source_length,
        )
        repeat = compare_observations(first, second)
        _, same_precision_eager = _benchmark_eager(
            model,
            record,
            pad_token_id=pad_token_id,
            iterations=1,
        )
        same_precision = compare_observations(same_precision_eager, first)
        _, fp32_reference = _benchmark_eager(
            fp32_model,
            record,
            pad_token_id=pad_token_id,
            iterations=1,
        )
        fp32_comparison = compare_observations(fp32_reference, first)
        rows.append(
            {
                "index": record.index,
                "prompt_length": record.prompt_length,
                "padded_length": record_padded,
                "padding_tokens": pad,
                "source_length": record.source_length,
                "eager_fp32_ms": fp32_eager_ms[record.index],
                "copy_ms": copy_ms,
                "replay_ms": replay_ms,
                "copy_bytes": copy_bytes,
                "copy_tensors": copy_tensors,
                "repeat_deterministic": repeat["exact"],
                "same_precision_graph_exact": same_precision["exact"],
                "fp32_reference_exact": fp32_comparison["exact"],
                "fp32_max_abs_drift": fp32_comparison["max_abs_drift"],
                "finite": repeat["finite"] and same_precision["finite"],
                "cache_valid": repeat["shape_valid"] and fp32_comparison["shape_valid"],
            }
        )
    setup = {
        "capture_seconds": captured.capture_seconds,
        "allocated_delta_bytes": captured.allocated_delta_bytes,
        "reserved_delta_bytes": captured.reserved_delta_bytes,
    }
    del captured
    gc.collect()
    torch.cuda.empty_cache()
    return rows, setup


def _profile_variant(
    model: torch.nn.Module,
    records: list[PrefillRecord],
    *,
    mode: str,
    pad_token_id: int,
    warmup: int,
    iterations: int,
    fp32_model: torch.nn.Module,
    fp32_eager_ms: dict[int, float],
    dtype_setup_seconds: float,
) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[PrefillRecord]] = defaultdict(list)
    for record in records:
        grouped[prefill_signature(record, mode=mode)].append(record)
    all_rows: list[dict[str, Any]] = []
    captures: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    for signature, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        rows, setup = _profile_graph_group(
            model,
            group,
            mode=mode,
            pad_token_id=pad_token_id,
            warmup=warmup,
            iterations=iterations,
            fp32_model=fp32_model,
            fp32_eager_ms=fp32_eager_ms,
        )
        all_rows.extend(rows)
        captures.append({"signature": repr(signature), "record_count": len(group), **setup})
    return {
        "bucket_mode": mode,
        "graph_count": len(grouped),
        "capture_setup_seconds": sum(float(row["capture_seconds"]) for row in captures),
        "dtype_setup_seconds": dtype_setup_seconds,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "estimated_retained_graph_bytes": sum(int(row["allocated_delta_bytes"]) for row in captures),
        "captures": captures,
        "records": sorted(all_rows, key=lambda row: int(row["index"])),
    }


def _run_song(
    args: Any,
    *,
    binding: Any,
    tokenizer: Any,
    timing_model: Any,
    timing_tokenizer: Any,
    generation_config: Any,
    beatmap_config: Any,
    output_path: Path,
    capture_records: list[PrefillRecord] | None,
) -> dict[str, Any]:
    from inference import generate
    from osuT5.osuT5.inference import Processor

    captured: dict[str, Any] = {}
    original_generate = Processor.generate
    output_path.mkdir(parents=True, exist_ok=False)

    def wrapped_generate(processor, *positional, **kwargs):
        is_main = kwargs.get("profile_label") == "main_generation"
        if is_main and capture_records is not None:
            with capture_main_prefills(processor, capture_records):
                result = original_generate(processor, *positional, **kwargs)
        else:
            result = original_generate(processor, *positional, **kwargs)
        if is_main:
            if "processor" in captured:
                raise RuntimeError("captured more than one main Processor")
            captured["processor"] = processor
        return result

    Processor.generate = wrapped_generate
    started = time.perf_counter()
    try:
        generated, result_path = generate(
            args,
            output_path=str(output_path),
            generation_config=generation_config,
            beatmap_config=beatmap_config,
            model=binding,
            tokenizer=tokenizer,
            timing_model=timing_model,
            timing_tokenizer=timing_tokenizer,
            verbose=False,
        )
        torch.cuda.synchronize()
    finally:
        Processor.generate = original_generate
    request_seconds = time.perf_counter() - started
    processor = captured.get("processor")
    if processor is None or not isinstance(processor.last_generation_stats, dict):
        raise RuntimeError("full song did not expose main generation stats")
    return {
        "model": processor.model,
        "generated": generated,
        "result_path": str(result_path),
        "main_stats": dict(processor.last_generation_stats),
        "request_seconds": request_seconds,
    }


@torch.no_grad()
def run(args: Any, *, output_path: Path, warmup: int, iterations: int) -> dict[str, Any]:
    from inference import (
        compile_args,
        get_config,
        load_model_with_engine,
        setup_inference_environment,
        should_load_separate_timing_model,
    )

    _assert_scout_args(args)
    if not bool(getattr(args, "profile_inference", False)):
        raise ValueError(
            "prefill graph scout requires profile_inference=true for synchronized controls"
        )
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
    timing_model, timing_tokenizer = load_timing_pair(
        args,
        loader=load_model_with_engine,
        needs_separate=should_load_separate_timing_model,
        main_binding=binding,
        main_tokenizer=tokenizer,
    )
    generation_config, beatmap_config = get_config(args)
    reciprocal_rng = snapshot_rng_state()
    baseline = _run_song(
        args,
        binding=binding,
        tokenizer=tokenizer,
        timing_model=timing_model,
        timing_tokenizer=timing_tokenizer,
        generation_config=generation_config,
        beatmap_config=beatmap_config,
        output_path=output_path / "baseline",
        capture_records=None,
    )
    restore_rng_state(reciprocal_rng)
    records: list[PrefillRecord] = []
    captured = _run_song(
        args,
        binding=binding,
        tokenizer=tokenizer,
        timing_model=timing_model,
        timing_tokenizer=timing_tokenizer,
        generation_config=generation_config,
        beatmap_config=beatmap_config,
        output_path=output_path / "capture",
        capture_records=records,
    )
    if len(records) != EXPECTED_PREFILLS:
        raise RuntimeError(f"expected {EXPECTED_PREFILLS} live main prefills, captured {len(records)}")
    if baseline["main_stats"]["generated_tokens"] != captured["main_stats"]["generated_tokens"]:
        raise RuntimeError("prefill capture changed the accepted main token count")
    baseline_bytes = Path(baseline["result_path"]).read_bytes()
    captured_bytes = Path(captured["result_path"]).read_bytes()
    if baseline_bytes != captured_bytes:
        raise RuntimeError("prefill capture changed the accepted final .osu bytes")
    gc.collect()
    torch.cuda.empty_cache()
    fp32_model = captured["model"]
    if fp32_model.dtype != torch.float32:
        raise TypeError(f"accepted source model must be FP32, got {fp32_model.dtype}")
    pad_token_id = int(tokenizer.pad_id)
    eager_rows: list[dict[str, Any]] = []
    eager_ms: dict[int, float] = {}
    for record in records:
        elapsed_ms, _ = _benchmark_eager(
            fp32_model,
            record,
            pad_token_id=pad_token_id,
            iterations=iterations,
        )
        eager_ms[record.index] = elapsed_ms
        eager_rows.append(
            {
                "index": record.index,
                "prompt_length": record.prompt_length,
                "source_length": record.source_length,
                "execution_ms": elapsed_ms,
            }
        )

    fp16_started = time.perf_counter()
    fp16_model = copy.deepcopy(fp32_model).half().eval()
    torch.cuda.synchronize()
    fp16_setup_seconds = time.perf_counter() - fp16_started
    variants = {
        "exact_fp32_graph": _profile_variant(
            fp32_model,
            records,
            mode="exact",
            pad_token_id=pad_token_id,
            warmup=warmup,
            iterations=iterations,
            fp32_model=fp32_model,
            fp32_eager_ms=eager_ms,
            dtype_setup_seconds=0.0,
        ),
        "bucket64_fp32_graph": _profile_variant(
            fp32_model,
            records,
            mode="bucket64",
            pad_token_id=pad_token_id,
            warmup=warmup,
            iterations=iterations,
            fp32_model=fp32_model,
            fp32_eager_ms=eager_ms,
            dtype_setup_seconds=0.0,
        ),
        "exact_fp16_graph": _profile_variant(
            fp16_model,
            records,
            mode="exact",
            pad_token_id=pad_token_id,
            warmup=warmup,
            iterations=iterations,
            fp32_model=fp32_model,
            fp32_eager_ms=eager_ms,
            dtype_setup_seconds=fp16_setup_seconds,
        ),
        "bucket64_fp16_graph": _profile_variant(
            fp16_model,
            records,
            mode="bucket64",
            pad_token_id=pad_token_id,
            warmup=warmup,
            iterations=iterations,
            fp32_model=fp32_model,
            fp32_eager_ms=eager_ms,
            dtype_setup_seconds=fp16_setup_seconds,
        ),
    }
    baseline_stats = baseline["main_stats"]
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "commit": _git_head(),
            "hardware": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "baseline_main_seconds": float(baseline_stats["elapsed_seconds"]),
            "baseline_main_tokens": int(baseline_stats["generated_tokens"]),
            "baseline_main_tps": float(baseline_stats["tokens_per_second"]),
            "baseline_request_seconds": float(baseline["request_seconds"]),
            "captured_prefills": len(records),
            "capture_token_count_exact": True,
            "capture_final_osu_exact": True,
            "baseline_osu_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
            "capture_osu_sha256": hashlib.sha256(captured_bytes).hexdigest(),
            "unique_exact_signatures": len({prefill_signature(record, mode="exact") for record in records}),
            "unique_bucket64_signatures": len({prefill_signature(record, mode="bucket64") for record in records}),
            "prompt_tokens": sum(record.prompt_length for record in records),
            "padded_prompt_tokens_bucket64": sum(bucket_prompt_length(record.prompt_length) for record in records),
            "warmup": warmup,
            "iterations": iterations,
        },
        "eager_fp32": {"records": eager_rows},
        "variants": variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()
    if cli.warmup < 1 or cli.iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    cli.output_path.mkdir(parents=True, exist_ok=False)
    args = _load_args(cli.config_name, cli.overrides)
    payload = run(
        args,
        output_path=cli.output_path,
        warmup=cli.warmup,
        iterations=cli.iterations,
    )
    raw_path = cli.output_path / "prefill-graph-raw.json"
    raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(raw_path)


if __name__ == "__main__":
    main()
