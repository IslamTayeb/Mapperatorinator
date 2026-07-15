"""Opt-in batched all-window encoder store for reciprocal experiments."""

from __future__ import annotations

import hashlib
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import torch
from transformers.modeling_outputs import BaseModelOutput


STORE_VERSION = "shared-precision-batched-all-window-encoder-store-v3"
SUPPORTED_LABELS = ("timing_context", "main_generation")
SUPPORTED_PRECISIONS = ("fp32", "fp16")
ENCODER_CONDITIONING_KEYS = (
    "beatmap_idx",
    "difficulty",
    "mapper_idx",
    "song_position",
)


class EncoderStoreError(RuntimeError):
    pass


def _sync_cuda() -> None:
    torch.cuda.synchronize()


def _elapsed_cuda(fn):
    _sync_cuda()
    started = time.perf_counter()
    value = fn()
    _sync_cuda()
    return value, time.perf_counter() - started


def _stack_conditioning(
    rows: Sequence[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not rows:
        raise EncoderStoreError("cannot stack empty encoder conditioning")
    keys = tuple(sorted(rows[0]))
    if any(tuple(sorted(row)) != keys for row in rows):
        raise EncoderStoreError("encoder conditioning keys changed between windows")
    return {
        key: torch.cat([row[key] for row in rows], dim=0)
        for key in keys
    }


def _conditioning_digest(rows: Sequence[dict[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for index, row in enumerate(rows):
        digest.update(str(index).encode("ascii"))
        for key in sorted(row):
            value = row[key].detach().contiguous().cpu()
            digest.update(key.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _window_conditioning(
    processor,
    generation_config,
    frame_times: torch.Tensor,
    song_length_ms: float,
) -> list[dict[str, torch.Tensor]]:
    static = processor._get_model_cond_kwargs(generation_config)
    unsupported = sorted(set(static) - set(ENCODER_CONDITIONING_KEYS))
    if unsupported:
        raise EncoderStoreError(
            f"encoder store does not own conditioning keys {unsupported}"
        )
    rows: list[dict[str, torch.Tensor]] = []
    for raw_time in frame_times:
        frame_time = float(raw_time.item())
        row = {key: value.detach().clone() for key, value in static.items()}
        if processor.do_song_position_embed:
            row["song_position"] = torch.tensor(
                [[
                    frame_time / song_length_ms,
                    (frame_time + processor.miliseconds_per_sequence)
                    / song_length_ms,
                ]],
                dtype=torch.float32,
            )
        rows.append(row)
    return rows


def _tensor_contract(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


@dataclass
class _GenerationUse:
    label: str
    processor_id: int
    entry: "_StoreEntry"
    model_generate_calls: int = 0


@dataclass
class _StoreEntry:
    index: int
    model: Any
    tokenizer: Any
    source_frames: torch.Tensor
    source_frame_times: torch.Tensor
    conditioning: list[dict[str, torch.Tensor]]
    store: torch.Tensor
    evidence: dict[str, Any]
    labels: list[str] = field(default_factory=list)


@dataclass
class BatchedAllWindowEncoderStore:
    batch_size: int = 16
    precision: str = "fp32"
    _entries: list[_StoreEntry] = field(default_factory=list, init=False, repr=False)
    _active: dict[int, _GenerationUse] = field(default_factory=dict, init=False)
    _labels: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError("encoder store batch_size must be an integer")
        if self.batch_size <= 0:
            raise ValueError("encoder store batch_size must be positive")
        if self.precision not in SUPPORTED_PRECISIONS:
            raise ValueError(
                f"encoder store precision must be one of {SUPPORTED_PRECISIONS}"
            )

    @property
    def evidence(self) -> dict[str, Any]:
        entries = [
            {
                **entry.evidence,
                "entry_index": entry.index,
                "labels": list(entry.labels),
            }
            for entry in self._entries
        ]
        return {
            "version": STORE_VERSION,
            "result_class": (
                "same-precision-exactness-candidate"
                if self.batch_size == 1
                else "same-precision-gate-with-documented-encoder-drift"
            ),
            "production_default": False,
            "selector_changes": False,
            "decoder_changes": False,
            "model_forward_changes": False,
            "server_changes": False,
            "precision": self.precision,
            "strict_fp32": self.precision == "fp32",
            "strict_exactness_required_for_promotion": True,
            "batch_size": self.batch_size,
            "store_count": len(entries),
            "shared_across_timing_main": (
                len(entries) == 1 and tuple(entries[0]["labels"]) == SUPPORTED_LABELS
            ),
            "total_complete_precompute_seconds": sum(
                float(entry["complete_precompute_seconds"]) for entry in entries
            ),
            "total_output_store_bytes": sum(
                int(entry["output_store_bytes"]) for entry in entries
            ),
            "entries": entries,
            "labels_completed": list(self._labels),
            "active_processor_count": len(self._active),
        }

    def _assert_processor(self, processor) -> None:
        if processor.parallel:
            raise EncoderStoreError("encoder store requires sequential generation")
        if processor.precision != self.precision:
            raise EncoderStoreError(
                f"encoder store expected precision={self.precision}, got "
                f"{processor.precision}"
            )
        if processor.cfg_scale != 1.0:
            raise EncoderStoreError("encoder store requires cfg_scale=1")
        if processor.inference_runtime is None:
            raise EncoderStoreError("encoder store requires the optimized runtime")
        if processor.device != "cuda":
            raise EncoderStoreError("encoder store requires device=cuda")
        if not torch.cuda.is_available():
            raise EncoderStoreError("encoder store requires an allocated CUDA device")
        if processor.model.device.type != "cuda":
            raise EncoderStoreError("encoder store model must be resident on CUDA")
        expected_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
        }[self.precision]
        if processor.model.dtype != expected_dtype:
            raise EncoderStoreError(
                f"encoder store {self.precision} model must use {expected_dtype}, "
                f"got {processor.model.dtype}"
            )
        if self.precision == "fp32" and torch.get_float32_matmul_precision() != "highest":
            raise EncoderStoreError(
                "encoder store requires float32_matmul_precision='highest'"
            )
        if torch.backends.cuda.matmul.allow_tf32:
            raise EncoderStoreError("encoder store requires CUDA matmul TF32 disabled")
        if torch.backends.cudnn.allow_tf32:
            raise EncoderStoreError("encoder store requires cuDNN TF32 disabled")
        if self.batch_size > int(processor.max_batch_size):
            raise EncoderStoreError(
                f"encoder store B{self.batch_size} exceeds max_batch_size="
                f"{processor.max_batch_size}"
            )

    @staticmethod
    def _conditioning_equal(
        reference: list[dict[str, torch.Tensor]],
        candidate: list[dict[str, torch.Tensor]],
    ) -> bool:
        if len(reference) != len(candidate):
            return False
        for index, (reference, candidate) in enumerate(
            zip(reference, candidate, strict=True)
        ):
            if tuple(sorted(reference)) != tuple(sorted(candidate)):
                return False
            for key in reference:
                if not torch.equal(reference[key], candidate[key]):
                    return False
        return True

    def _compatible_entry(
        self,
        processor,
        sequences,
        conditioning: list[dict[str, torch.Tensor]],
    ) -> _StoreEntry | None:
        for entry in self._entries:
            if (
                processor.model is entry.model
                and sequences[0] is entry.source_frames
                and sequences[1] is entry.source_frame_times
                and self._conditioning_equal(entry.conditioning, conditioning)
            ):
                return entry
        return None

    @torch.no_grad()
    def _precompute(
        self,
        processor,
        frames: torch.Tensor,
        conditioning: list[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if frames.device.type != "cpu" or frames.dtype != torch.float32:
            raise EncoderStoreError(
                "encoder store source frames must remain contiguous FP32 CPU tensors"
            )
        frames = frames.contiguous()
        if len(frames) == 0 or len(frames) != len(conditioning):
            raise EncoderStoreError("encoder store requires matched nonempty windows")

        encoder = processor.model.get_encoder()
        encoder.eval()
        _sync_cuda()
        baseline_allocated = int(torch.cuda.memory_allocated())
        baseline_reserved = int(torch.cuda.memory_reserved())
        torch.cuda.reset_peak_memory_stats()
        complete_started = time.perf_counter()
        setup_seconds = 0.0
        input_copy_seconds = 0.0
        encoder_seconds = 0.0
        storage_allocation_seconds = 0.0
        output_store_copy_seconds = 0.0
        batches: list[dict[str, Any]] = []
        store: torch.Tensor | None = None
        output_shape: tuple[int, ...] | None = None

        for start in range(0, len(frames), self.batch_size):
            end = min(start + self.batch_size, len(frames))
            setup_started = time.perf_counter()
            frame_chunk = frames[start:end]
            kwargs_chunk = _stack_conditioning(conditioning[start:end])
            setup_seconds += time.perf_counter() - setup_started

            def copy_inputs():
                return (
                    frame_chunk.to(processor.model.device),
                    {
                        key: value.to(processor.model.device)
                        for key, value in kwargs_chunk.items()
                    },
                )

            (device_frames, device_kwargs), copy_seconds = _elapsed_cuda(copy_inputs)
            outputs, encoded_seconds = _elapsed_cuda(
                lambda: encoder(
                    frames=device_frames,
                    **device_kwargs,
                    return_dict=True,
                )
            )
            hidden = getattr(outputs, "last_hidden_state", None)
            if not isinstance(hidden, torch.Tensor):
                raise EncoderStoreError("encoder output lacks last_hidden_state")
            expected_dtype = {
                "fp32": torch.float32,
                "fp16": torch.float16,
            }[self.precision]
            if hidden.dtype != expected_dtype or hidden.device.type != "cuda":
                raise EncoderStoreError(
                    f"encoder store output must use {expected_dtype} CUDA storage"
                )
            if not bool(torch.isfinite(hidden).all().item()):
                raise EncoderStoreError("encoder store output is not finite")
            current_shape = tuple(hidden.shape[1:])
            if output_shape is None:
                output_shape = current_shape
                store, allocation_seconds = _elapsed_cuda(
                    lambda: torch.empty(
                        (len(frames), *current_shape),
                        dtype=hidden.dtype,
                        device=hidden.device,
                    )
                )
                storage_allocation_seconds += allocation_seconds
            elif output_shape != current_shape:
                raise EncoderStoreError(
                    "encoder output shape changed between precompute batches"
                )
            assert store is not None
            _, store_seconds = _elapsed_cuda(
                lambda: store[start:end].copy_(hidden)
            )
            input_copy_seconds += copy_seconds
            encoder_seconds += encoded_seconds
            output_store_copy_seconds += store_seconds
            batches.append(
                {
                    "start": start,
                    "end": end,
                    "rows": end - start,
                    "input_copy_seconds": copy_seconds,
                    "encoder_seconds": encoded_seconds,
                    "output_store_copy_seconds": store_seconds,
                }
            )
            del outputs, hidden, device_frames, device_kwargs

        _sync_cuda()
        complete_seconds = time.perf_counter() - complete_started
        if store is None:
            raise EncoderStoreError("encoder store produced no output")
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        evidence = {
            "version": STORE_VERSION,
            "result_class": (
                "same-precision-exactness-candidate"
                if self.batch_size == 1
                else "same-precision-gate-with-documented-encoder-drift"
            ),
            "production_default": False,
            "selector_changes": False,
            "decoder_changes": False,
            "model_forward_changes": False,
            "server_changes": False,
            "precision": self.precision,
            "strict_fp32": self.precision == "fp32",
            "strict_exactness_required_for_promotion": True,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "batch_size": self.batch_size,
            "live_window_count": len(frames),
            "batch_count": len(batches),
            "frame_contract": _tensor_contract(frames),
            "conditioning_sha256": _conditioning_digest(conditioning),
            "output_shape": list(store.shape),
            "output_dtype": str(store.dtype),
            "output_store_bytes": store.numel() * store.element_size(),
            "batch_setup_seconds": setup_seconds,
            "input_copy_seconds": input_copy_seconds,
            "encoder_synchronized_seconds": encoder_seconds,
            "storage_allocation_seconds": storage_allocation_seconds,
            "output_store_copy_seconds": output_store_copy_seconds,
            "complete_precompute_seconds": complete_seconds,
            "baseline_allocated_vram_bytes": baseline_allocated,
            "baseline_reserved_vram_bytes": baseline_reserved,
            "peak_allocated_vram_bytes": peak_allocated,
            "peak_reserved_vram_bytes": peak_reserved,
            "incremental_peak_allocated_vram_bytes": (
                peak_allocated - baseline_allocated
            ),
            "incremental_peak_reserved_vram_bytes": peak_reserved - baseline_reserved,
            "drift_reference": "external_b1_same-input audit",
            "drift_charged_to_request_wall": False,
            "batches": batches,
        }
        return store, evidence

    def begin_generation(
        self,
        processor,
        *,
        sequences,
        generation_config,
        profile_label: str,
    ) -> None:
        if profile_label not in SUPPORTED_LABELS:
            raise EncoderStoreError(
                f"encoder store requires one of {SUPPORTED_LABELS}, got "
                f"{profile_label!r}"
            )
        if profile_label in self._labels:
            raise EncoderStoreError(
                f"encoder store generation label {profile_label!r} repeated"
            )
        expected_next = SUPPORTED_LABELS[len(self._labels)]
        if profile_label != expected_next:
            raise EncoderStoreError(
                f"encoder store expected {expected_next!r}, got {profile_label!r}"
            )
        self._assert_processor(processor)
        frames, frame_times, song_length_ms = sequences
        if not isinstance(frames, torch.Tensor) or not isinstance(
            frame_times, torch.Tensor
        ):
            raise TypeError("encoder store sequences must be tensors")
        conditioning = _window_conditioning(
            processor,
            generation_config,
            frame_times,
            float(song_length_ms),
        )
        if self._entries:
            first = self._entries[0]
            if frames is not first.source_frames or frame_times is not first.source_frame_times:
                raise EncoderStoreError(
                    "timing and main stages must share the request's source tensors"
                )
        entry = self._compatible_entry(processor, sequences, conditioning)
        if entry is None:
            store, evidence = self._precompute(
                processor,
                processor.prepare_frames(frames),
                conditioning,
            )
            entry = _StoreEntry(
                index=len(self._entries),
                model=processor.model,
                tokenizer=processor.tokenizer,
                source_frames=frames,
                source_frame_times=frame_times,
                conditioning=conditioning,
                store=store,
                evidence={
                    **evidence,
                    "model_class": type(processor.model).__name__,
                    "tokenizer_class": type(processor.tokenizer).__name__,
                },
            )
            self._entries.append(entry)
        processor_id = id(processor)
        if processor_id in self._active:
            raise EncoderStoreError("encoder store processor is already active")
        self._active[processor_id] = _GenerationUse(
            profile_label,
            processor_id,
            entry,
        )
        self._publish(processor)

    def inject(self, processor, model_kwargs: dict[str, Any]) -> dict[str, Any]:
        use = self._active.get(id(processor))
        if use is None:
            raise EncoderStoreError("model_generate called outside encoder-store stage")
        entry = use.entry
        if processor.model is not entry.model:
            raise EncoderStoreError(
                "encoder store refused accidental cross-model output reuse"
            )
        window_count = len(entry.source_frames)
        index = use.model_generate_calls % window_count
        source = model_kwargs.get("inputs")
        if not isinstance(source, torch.Tensor):
            raise EncoderStoreError("encoder store model call is missing input frames")
        expected_source = entry.source_frames[index : index + 1]
        if (
            source.data_ptr() != expected_source.data_ptr()
            or source.storage_offset() != expected_source.storage_offset()
            or source.shape != expected_source.shape
            or source.dtype != expected_source.dtype
            or source.device != expected_source.device
        ):
            raise EncoderStoreError(
                f"encoder source window {index} changed before reuse"
            )
        expected_conditioning = entry.conditioning[index]
        actual_conditioning = {
            key: value
            for key, value in model_kwargs.items()
            if key in ENCODER_CONDITIONING_KEYS
        }
        if tuple(sorted(actual_conditioning)) != tuple(
            sorted(expected_conditioning)
        ):
            raise EncoderStoreError(
                f"encoder conditioning keys changed at reuse window {index}"
            )
        for key, reference in expected_conditioning.items():
            value = actual_conditioning[key]
            if not isinstance(value, torch.Tensor) or not torch.equal(value, reference):
                raise EncoderStoreError(
                    f"encoder conditioning {key!r} changed at reuse window {index}"
                )
        if model_kwargs.get("encoder_outputs") is not None:
            raise EncoderStoreError("model call already contains encoder_outputs")
        use.model_generate_calls += 1
        return {
            **model_kwargs,
            "encoder_outputs": BaseModelOutput(
                last_hidden_state=entry.store[index : index + 1]
            ),
        }

    def end_generation(self, processor) -> None:
        use = self._active.pop(id(processor), None)
        if use is None:
            raise EncoderStoreError("encoder store stage ended without begin")
        window_count = len(use.entry.source_frames)
        if use.model_generate_calls <= 0 or use.model_generate_calls % window_count:
            raise EncoderStoreError(
                f"{use.label} used {use.model_generate_calls} encoder rows for "
                f"{window_count} windows"
            )
        self._labels.append(use.label)
        use.entry.labels.append(use.label)
        use.entry.evidence.setdefault("uses", {})[use.label] = {
            "model_generate_calls": use.model_generate_calls,
            "complete_window_passes": use.model_generate_calls // window_count,
            "rows_reused": use.model_generate_calls,
            "per_window_device_copy": False,
        }
        self._publish(processor)

    def _publish(self, processor) -> None:
        profiler = getattr(processor, "profiler", None)
        set_metadata = getattr(profiler, "set_metadata", None)
        if callable(set_metadata):
            set_metadata(optimized_batched_encoder_store=self.evidence)

    def finalize(self) -> dict[str, Any]:
        if self._active:
            raise EncoderStoreError("encoder store finalized with active processors")
        if tuple(self._labels) != SUPPORTED_LABELS:
            raise EncoderStoreError(
                f"encoder store requires timing then main reuse; got {self._labels}"
            )
        evidence = self.evidence
        if len(self._entries) not in {1, 2}:
            raise EncoderStoreError(
                f"encoder store expected one shared or two independent stores, got "
                f"{len(self._entries)}"
            )
        required_times = (
            "batch_setup_seconds",
            "input_copy_seconds",
            "encoder_synchronized_seconds",
            "storage_allocation_seconds",
            "output_store_copy_seconds",
            "complete_precompute_seconds",
        )
        for entry_index, entry in enumerate(evidence["entries"]):
            for key in required_times:
                value = entry.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise EncoderStoreError(
                        f"encoder store entry {entry_index} evidence {key} is invalid"
                    )
        return evidence


@contextmanager
def install_batched_encoder_store_candidate(
    processor_class,
    *,
    batch_size: int = 16,
    precision: str = "fp32",
) -> Iterator[BatchedAllWindowEncoderStore]:
    """Temporarily install reuse without touching selectors or default imports."""

    manager = BatchedAllWindowEncoderStore(
        batch_size=batch_size,
        precision=precision,
    )
    original_generate = processor_class.generate
    original_model_generate = processor_class.model_generate

    def generate_with_store(self, *args, **kwargs):
        sequences = kwargs.get("sequences")
        generation_config = kwargs.get("generation_config")
        profile_label = kwargs.get("profile_label")
        if sequences is None or generation_config is None:
            raise EncoderStoreError(
                "encoder store requires keyword sequences and generation_config"
            )
        manager.begin_generation(
            self,
            sequences=sequences,
            generation_config=generation_config,
            profile_label=profile_label,
        )
        try:
            return original_generate(self, *args, **kwargs)
        finally:
            manager.end_generation(self)

    def model_generate_with_store(self, model_kwargs, **generate_kwargs):
        return original_model_generate(
            self,
            manager.inject(self, model_kwargs),
            **generate_kwargs,
        )

    processor_class.generate = generate_with_store
    processor_class.model_generate = model_generate_with_store
    try:
        yield manager
    finally:
        processor_class.generate = original_generate
        processor_class.model_generate = original_model_generate


__all__ = [
    "BatchedAllWindowEncoderStore",
    "ENCODER_CONDITIONING_KEYS",
    "EncoderStoreError",
    "STORE_VERSION",
    "SUPPORTED_LABELS",
    "SUPPORTED_PRECISIONS",
    "install_batched_encoder_store_candidate",
]
