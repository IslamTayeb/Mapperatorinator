"""Measure batched all-window encoder precompute on accepted FP32 tensors.

This is a verifier-only ceiling harness.  It reconstructs the exact audio-window
tensors and conditioning used by the current sequential Processor, then compares
B1 encoder precompute with B2/B4/B8/B16 over those same tensors.  It does not
install an encoder store in the runtime, change generation, or touch the server.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402


DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16)
SCHEMA_VERSION = 1


def _load_args(config_name: str, overrides: list[str]):
    import hydra
    from omegaconf import DictConfig, OmegaConf

    __import__("config")
    config_dir = REPO_ROOT / "configs" / "inference"
    resolved_name = config_name.removeprefix("inference/")
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = hydra.compose(config_name=resolved_name, overrides=overrides)
    return OmegaConf.to_object(cfg) if isinstance(cfg, DictConfig) else cfg


def parse_batch_sizes(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"invalid comma-separated batch sizes: {value!r}") from exc
    if not parsed:
        raise ValueError("at least one encoder batch size is required")
    if any(item <= 0 for item in parsed):
        raise ValueError("encoder batch sizes must be positive")
    if len(set(parsed)) != len(parsed) or tuple(sorted(parsed)) != parsed:
        raise ValueError("encoder batch sizes must be unique and increasing")
    if parsed[0] != 1:
        raise ValueError("encoder ceiling requires B1 as the reference")
    return parsed


def validate_batch_sizes(
    batch_sizes: Sequence[int],
    *,
    max_batch_size: int,
) -> tuple[int, ...]:
    if max_batch_size <= 0:
        raise ValueError("args.max_batch_size must be positive")
    parsed = parse_batch_sizes(",".join(str(value) for value in batch_sizes))
    overflow = [value for value in parsed if value > max_batch_size]
    if overflow:
        raise ValueError(
            f"encoder batch sizes {overflow} exceed args.max_batch_size="
            f"{max_batch_size}; increase max_batch_size or pass a capped list"
        )
    return parsed


def _assert_accepted_args(args) -> None:
    required = {
        "inference_engine": "optimized",
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "device": "cuda",
        "use_server": False,
        "parallel": False,
        "cfg_scale": 1.0,
        "num_beams": 1,
    }
    mismatches = {
        key: {"actual": getattr(args, key), "required": expected}
        for key, expected in required.items()
        if getattr(args, key) != expected
    }
    if mismatches:
        raise ValueError(
            "batched encoder ceiling requires the accepted FP32 runtime: "
            f"{mismatches}"
        )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    try:
        payload = value.numpy().tobytes()
    except TypeError:
        payload = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _window_input_manifest(
    frames: torch.Tensor,
    window_kwargs: Sequence[dict[str, torch.Tensor]],
) -> dict[str, Any]:
    if frames.ndim < 2 or len(frames) != len(window_kwargs):
        raise ValueError("frames and conditioning must have the same live window count")
    if len(frames) == 0:
        raise ValueError("encoder ceiling requires at least one live audio window")
    frame_hashes = [_tensor_sha256(frame) for frame in frames]
    conditioning_hashes: list[str] = []
    for index, kwargs in enumerate(window_kwargs):
        digest = hashlib.sha256()
        for key in sorted(kwargs):
            value = kwargs[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"window {index} conditioning {key!r} is not a tensor")
            digest.update(key.encode("utf-8"))
            digest.update(_tensor_sha256(value).encode("ascii"))
        conditioning_hashes.append(digest.hexdigest())
    combined = hashlib.sha256()
    for frame_hash, conditioning_hash in zip(
        frame_hashes,
        conditioning_hashes,
        strict=True,
    ):
        combined.update(frame_hash.encode("ascii"))
        combined.update(conditioning_hash.encode("ascii"))
    return {
        "live_window_count": len(frames),
        "frames_shape": list(frames.shape),
        "frames_dtype": str(frames.dtype),
        "per_window_frame_sha256": frame_hashes,
        "per_window_conditioning_sha256": conditioning_hashes,
        "combined_sha256": combined.hexdigest(),
    }


def _build_window_kwargs(processor, generation_config, frame_times, song_length):
    static_kwargs = processor._get_model_cond_kwargs(generation_config)
    result: list[dict[str, torch.Tensor]] = []
    for frame_time_tensor in frame_times:
        frame_time = float(frame_time_tensor.item())
        kwargs = {key: value.clone() for key, value in static_kwargs.items()}
        if processor.do_song_position_embed:
            kwargs["song_position"] = torch.tensor(
                [[
                    frame_time / song_length,
                    (frame_time + processor.miliseconds_per_sequence) / song_length,
                ]],
                dtype=torch.float32,
            )
        result.append(kwargs)
    return result


def _stack_window_kwargs(
    window_kwargs: Sequence[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not window_kwargs:
        raise ValueError("cannot stack empty window conditioning")
    keys = tuple(sorted(window_kwargs[0]))
    if any(tuple(sorted(item)) != keys for item in window_kwargs):
        raise ValueError("window conditioning keys changed inside one encoder batch")
    return {
        key: torch.cat([item[key] for item in window_kwargs], dim=0)
        for key in keys
    }


def encoder_drift(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"encoder output shape changed: {tuple(reference.shape)} vs "
            f"{tuple(candidate.shape)}"
        )
    if reference.dtype != candidate.dtype:
        raise TypeError(
            f"encoder output dtype changed: {reference.dtype} vs {candidate.dtype}"
        )
    per_window: list[float] = []
    exact_windows = 0
    for reference_row, candidate_row in zip(reference, candidate, strict=True):
        if torch.equal(reference_row, candidate_row):
            exact_windows += 1
        per_window.append(
            float((reference_row.float() - candidate_row.float()).abs().max().item())
        )
    return {
        "max_abs": max(per_window, default=0.0),
        "mean_window_max_abs": (
            sum(per_window) / len(per_window) if per_window else 0.0
        ),
        "exact_window_count": exact_windows,
        "window_count": len(per_window),
        "per_window_max_abs": per_window,
    }


def _cuda_sync() -> None:
    torch.cuda.synchronize()


def _cuda_elapsed(fn) -> tuple[Any, float]:
    _cuda_sync()
    started = time.perf_counter()
    value = fn()
    _cuda_sync()
    return value, time.perf_counter() - started


def _encoder_hidden(outputs: Any) -> torch.Tensor:
    hidden = getattr(outputs, "last_hidden_state", None)
    if not isinstance(hidden, torch.Tensor):
        raise TypeError("encoder output does not expose last_hidden_state")
    if hidden.dtype != torch.float32:
        raise TypeError(f"accepted encoder output must be FP32, got {hidden.dtype}")
    if not bool(torch.isfinite(hidden).all().item()):
        raise RuntimeError("encoder output contains non-finite values")
    return hidden


@torch.no_grad()
def _profile_batch_size(
    encoder,
    *,
    frames: torch.Tensor,
    window_kwargs: Sequence[dict[str, torch.Tensor]],
    batch_size: int,
    warmup: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    if frames.device.type != "cpu":
        raise ValueError("source window frames must remain on CPU for each variant")
    if len(frames) != len(window_kwargs):
        raise ValueError("frames and conditioning counts differ")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    warmup_seconds = 0.0
    if warmup:
        warmup_count = min(batch_size, len(frames))
        warmup_kwargs = _stack_window_kwargs(window_kwargs[:warmup_count])
        warmup_frames = frames[:warmup_count].to("cuda")
        warmup_kwargs = {key: value.to("cuda") for key, value in warmup_kwargs.items()}
        _cuda_sync()
        warmup_started = time.perf_counter()
        for _ in range(warmup):
            warmup_output = encoder(
                frames=warmup_frames,
                **warmup_kwargs,
                return_dict=True,
            )
            del warmup_output
        _cuda_sync()
        warmup_seconds = time.perf_counter() - warmup_started
        del warmup_frames, warmup_kwargs
        gc.collect()
        torch.cuda.empty_cache()
        _cuda_sync()

    baseline_allocated = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    setup_seconds = 0.0
    input_copy_seconds = 0.0
    encoder_seconds = 0.0
    storage_allocation_seconds = 0.0
    output_store_copy_seconds = 0.0
    store: torch.Tensor | None = None
    output_shape: tuple[int, ...] | None = None
    batch_rows: list[dict[str, Any]] = []

    _cuda_sync()
    complete_started = time.perf_counter()
    for start in range(0, len(frames), batch_size):
        end = min(start + batch_size, len(frames))
        setup_started = time.perf_counter()
        frame_chunk = frames[start:end]
        kwargs_chunk = _stack_window_kwargs(window_kwargs[start:end])
        setup_seconds += time.perf_counter() - setup_started

        def copy_inputs():
            return (
                frame_chunk.to("cuda"),
                {key: value.to("cuda") for key, value in kwargs_chunk.items()},
            )

        (device_frames, device_kwargs), copied = _cuda_elapsed(copy_inputs)
        input_copy_seconds += copied

        outputs, encoded = _cuda_elapsed(
            lambda: encoder(
                frames=device_frames,
                **device_kwargs,
                return_dict=True,
            )
        )
        encoder_seconds += encoded
        hidden = _encoder_hidden(outputs)
        current_shape = tuple(hidden.shape[1:])
        if output_shape is None:
            output_shape = current_shape

            def allocate_store():
                return torch.empty(
                    (len(frames), *current_shape),
                    dtype=hidden.dtype,
                    device=hidden.device,
                )

            store, allocated = _cuda_elapsed(allocate_store)
            storage_allocation_seconds += allocated
        elif current_shape != output_shape:
            raise RuntimeError(
                f"encoder output shape changed across chunks: {output_shape} vs "
                f"{current_shape}"
            )
        assert store is not None
        _, stored = _cuda_elapsed(lambda: store[start:end].copy_(hidden))
        output_store_copy_seconds += stored
        batch_rows.append(
            {
                "start": start,
                "end": end,
                "rows": end - start,
                "encoder_seconds": encoded,
                "input_copy_seconds": copied,
                "output_store_copy_seconds": stored,
            }
        )
        del outputs, hidden, device_frames, device_kwargs

    _cuda_sync()
    complete_seconds = time.perf_counter() - complete_started
    if store is None:
        raise RuntimeError("encoder ceiling received zero live windows")
    peak_allocated = int(torch.cuda.max_memory_allocated())
    storage_bytes = store.numel() * store.element_size()
    cpu_store, verification_copy_seconds = _cuda_elapsed(lambda: store.cpu())
    result = {
        "batch_size": batch_size,
        "live_window_count": len(frames),
        "batch_count": len(batch_rows),
        "warmup_iterations": warmup,
        "warmup_seconds_excluded": warmup_seconds,
        "batch_setup_seconds": setup_seconds,
        "input_copy_seconds": input_copy_seconds,
        "encoder_synchronized_seconds": encoder_seconds,
        "storage_allocation_seconds": storage_allocation_seconds,
        "output_store_copy_seconds": output_store_copy_seconds,
        "complete_precompute_seconds": complete_seconds,
        "verification_device_to_cpu_seconds_excluded": verification_copy_seconds,
        "windows_per_encoder_second": len(frames) / encoder_seconds,
        "windows_per_complete_second": len(frames) / complete_seconds,
        "output_store_bytes": storage_bytes,
        "baseline_allocated_vram_bytes": baseline_allocated,
        "peak_allocated_vram_bytes": peak_allocated,
        "incremental_peak_vram_bytes": peak_allocated - baseline_allocated,
        "output_shape": list(cpu_store.shape),
        "output_dtype": str(cpu_store.dtype),
        "batches": batch_rows,
    }
    del store
    gc.collect()
    torch.cuda.empty_cache()
    _cuda_sync()
    return result, cpu_store


def _validate_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("encoder ceiling report has the wrong schema version")
    metadata = report.get("metadata")
    variants = report.get("variants")
    if not isinstance(metadata, dict) or not isinstance(variants, dict):
        raise ValueError("encoder ceiling report is missing metadata or variants")
    live_windows = metadata.get("live_window_count")
    if not isinstance(live_windows, int) or live_windows <= 0:
        raise ValueError("encoder ceiling report has invalid live_window_count")
    expected = [str(value) for value in metadata.get("batch_sizes", [])]
    if list(variants) != expected:
        raise ValueError("encoder ceiling variants do not match declared batch sizes")
    for name, variant in variants.items():
        if variant.get("live_window_count") != live_windows:
            raise ValueError(f"B{name} does not cover every live window")
        for field in (
            "batch_setup_seconds",
            "input_copy_seconds",
            "encoder_synchronized_seconds",
            "storage_allocation_seconds",
            "output_store_copy_seconds",
            "complete_precompute_seconds",
        ):
            value = variant.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"B{name}.{field} is missing or non-finite")
            if field == "encoder_synchronized_seconds" and value <= 0:
                raise ValueError(f"B{name}.{field} must be positive")
            if value < 0:
                raise ValueError(f"B{name}.{field} must be non-negative")
        drift = variant.get("encoder_drift_vs_b1")
        if not isinstance(drift, dict) or drift.get("window_count") != live_windows:
            raise ValueError(f"B{name} has invalid encoder drift evidence")


@torch.no_grad()
def profile_batched_encoder_precompute_ceiling(
    args,
    *,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    warmup: int = 1,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("batched encoder precompute ceiling requires CUDA")
    _assert_accepted_args(args)
    batch_sizes = validate_batch_sizes(
        batch_sizes,
        max_batch_size=int(args.max_batch_size),
    )

    from inference import (
        compile_args,
        get_config,
        load_model_with_engine,
        setup_inference_environment,
    )
    from osuT5.osuT5.inference import Preprocessor, Processor
    from osuT5.osuT5.inference.engine_binding import unwrap_engine_binding

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
    model, runtime = unwrap_engine_binding(binding)
    if runtime is None:
        raise RuntimeError("encoder ceiling did not load the accepted optimized runtime")
    model.eval()
    model_dtype = next(model.parameters()).dtype
    if model_dtype != torch.float32:
        raise TypeError(f"accepted model parameters must be FP32, got {model_dtype}")

    preprocessor = Preprocessor(args, parallel=False)
    processor = Processor(args, binding, tokenizer)
    generation_config, _ = get_config(args)
    audio = preprocessor.load(args.audio_path)
    frames, frame_times, song_length = preprocessor.segment(audio)
    frames = processor.prepare_frames(frames).detach().contiguous().cpu()
    if frames.dtype != torch.float32:
        raise TypeError(f"accepted source frames must be FP32, got {frames.dtype}")
    window_kwargs = _build_window_kwargs(
        processor,
        generation_config,
        frame_times,
        song_length,
    )
    input_manifest = _window_input_manifest(frames, window_kwargs)
    encoder = model.get_encoder()
    encoder.eval()

    variants: dict[str, Any] = {}
    reference: torch.Tensor | None = None
    for batch_size in batch_sizes:
        row, outputs = _profile_batch_size(
            encoder,
            frames=frames,
            window_kwargs=window_kwargs,
            batch_size=batch_size,
            warmup=warmup,
        )
        if reference is None:
            reference = outputs
        drift = encoder_drift(reference, outputs)
        row["encoder_drift_vs_b1"] = drift
        variants[str(batch_size)] = row
        if batch_size != 1:
            del outputs
            gc.collect()
    if reference is None:
        raise RuntimeError("B1 encoder reference was not produced")

    b1 = variants["1"]
    for row in variants.values():
        row["encoder_speedup_vs_b1_pct"] = (
            (b1["encoder_synchronized_seconds"] - row["encoder_synchronized_seconds"])
            / b1["encoder_synchronized_seconds"]
            * 100.0
        )
        row["complete_speedup_vs_b1_pct"] = (
            (b1["complete_precompute_seconds"] - row["complete_precompute_seconds"])
            / b1["complete_precompute_seconds"]
            * 100.0
        )
        row["complete_seconds_saved_vs_b1"] = (
            b1["complete_precompute_seconds"] - row["complete_precompute_seconds"]
        )

    runtime_metadata = runtime.profile_metadata()
    report = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "result_class": "documented-drift-component-ceiling",
            "scope": "all live audio windows, encoder only",
            "production_wiring": False,
            "server_changes": False,
            "source_engine": args.inference_engine,
            "source_precision": args.precision,
            "source_attn_implementation": args.attn_implementation,
            "max_batch_size": int(args.max_batch_size),
            "batch_sizes": list(batch_sizes),
            "live_window_count": len(frames),
            "song_length_ms": float(song_length),
            "audio_path": str(args.audio_path),
            "model_path": str(args.model_path),
            "seed": int(args.seed),
            "warmup_iterations_per_batch_size": warmup,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(),
            "runtime": runtime_metadata,
        },
        "input_manifest": input_manifest,
        "variants": variants,
    }
    _validate_report(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--batch-sizes", default=",".join(map(str, DEFAULT_BATCH_SIZES)))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()
    args = _load_args(cli.config_name, list(cli.overrides))
    report = profile_batched_encoder_precompute_ceiling(
        args,
        batch_sizes=parse_batch_sizes(cli.batch_sizes),
        warmup=cli.warmup,
    )
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metadata"], indent=2))
    print(json.dumps({name: {
        "complete_precompute_seconds": row["complete_precompute_seconds"],
        "encoder_speedup_vs_b1_pct": row["encoder_speedup_vs_b1_pct"],
        "complete_speedup_vs_b1_pct": row["complete_speedup_vs_b1_pct"],
        "encoder_max_abs_drift": row["encoder_drift_vs_b1"]["max_abs"],
        "incremental_peak_vram_bytes": row["incremental_peak_vram_bytes"],
    } for name, row in report["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
