"""W3 scout: Tiger-style 16-wide batched encoder prefill (opt-in only).

Port of upstream PR #120 ``precompute_encoder_outputs`` into the optimized
ownership boundary.  Nothing here is imported by the accepted runtime.

Design notes
------------
- Encoder is a pure function of audio windows + static conditioning, so all
  live windows can be encoded in B16 chunks before sequential decode.
- Default storage is CPU (Tiger): long songs must not pin several GB of
  encoder state in VRAM; one window is moved back at inject time.
- Batched matmuls are quality-equivalent but not bit-identical to B1
  (documented drift, comparable to bf16-vs-fp32).  Exactness gate for
  promotion is therefore greedy token/``.osu`` drift policy + wall, not
  hidden-state equality.
- Prior STOP_ENCODER_PRECOMPUTE (``49903861``) was a B1 wall regress; W3
  revisits only after W1 harvest seals tiger ≥ optimized and measures
  complete-request wall with B16 savings credited.
"""

from __future__ import annotations

import hashlib
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Sequence

import torch
from transformers.modeling_outputs import BaseModelOutput


PREFILL_VERSION = "w3-batched-encoder-prefill-v1"
DEFAULT_ENCODER_BATCH_SIZE = 16
SUPPORTED_LABELS = ("timing_context", "main_generation")
SUPPORTED_PRECISIONS = ("fp16", "fp32")
ENCODER_CONDITIONING_KEYS = (
    "beatmap_idx",
    "difficulty",
    "mapper_idx",
    "song_position",
)
StorageDevice = Literal["cpu", "cuda"]


class BatchedEncoderPrefillError(RuntimeError):
    pass


def chunk_ranges(total: int, batch_size: int) -> list[tuple[int, int]]:
    """Return ``[start, end)`` ranges for Tiger-style B16 encoder chunks."""

    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("total window count must be a non-negative integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("encoder batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("encoder batch_size must be positive")
    return [(start, min(start + batch_size, total)) for start in range(0, total, batch_size)]


def stack_conditioning(
    rows: Sequence[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not rows:
        raise BatchedEncoderPrefillError("cannot stack empty encoder conditioning")
    keys = tuple(sorted(rows[0]))
    if any(tuple(sorted(row)) != keys for row in rows):
        raise BatchedEncoderPrefillError(
            "encoder conditioning keys changed between windows"
        )
    return {key: torch.cat([row[key] for row in rows], dim=0) for key in keys}


def conditioning_digest(rows: Sequence[dict[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for index, row in enumerate(rows):
        digest.update(str(index).encode("ascii"))
        for key in sorted(row):
            value = row[key].detach().contiguous().cpu()
            digest.update(key.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            try:
                payload = value.numpy().tobytes()
            except TypeError:
                payload = value.view(torch.uint8).numpy().tobytes()
            digest.update(payload)
    return digest.hexdigest()


def window_conditioning(
    processor,
    generation_config,
    frame_times: torch.Tensor,
    song_length_ms: float,
) -> list[dict[str, torch.Tensor]]:
    static = processor._get_model_cond_kwargs(generation_config)
    unsupported = sorted(set(static) - set(ENCODER_CONDITIONING_KEYS))
    if unsupported:
        raise BatchedEncoderPrefillError(
            f"batched encoder prefill does not own conditioning keys {unsupported}"
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


def encoder_hidden_drift(
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


def _sync_if_cuda(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed(fn, *, sync_device: torch.device | str | None = None):
    if sync_device is not None:
        _sync_if_cuda(sync_device)
    started = time.perf_counter()
    value = fn()
    if sync_device is not None:
        _sync_if_cuda(sync_device)
    return value, time.perf_counter() - started


def _tensor_contract(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


@torch.no_grad()
def precompute_encoder_hidden_states(
    encoder,
    frames: torch.Tensor,
    conditioning: Sequence[dict[str, torch.Tensor]],
    *,
    batch_size: int = DEFAULT_ENCODER_BATCH_SIZE,
    compute_device: torch.device | str,
    storage: StorageDevice = "cpu",
    expected_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Batched all-window encoder forward (Tiger 16-wide prefill).

    ``encoder`` is typically ``model.get_encoder()`` so spectrogram +
    conditioning match the sequential path.  Returns a store tensor on
    ``storage`` plus timing/VRAM evidence suitable for reciprocal gates.
    """

    if frames.ndim < 2:
        raise BatchedEncoderPrefillError("frames must be (N, L_raw[, ...])")
    if frames.device.type != "cpu" or not frames.is_floating_point():
        raise BatchedEncoderPrefillError(
            "source frames must remain floating-point CPU tensors"
        )
    frames = frames.contiguous()
    if len(frames) == 0 or len(frames) != len(conditioning):
        raise BatchedEncoderPrefillError("requires matched nonempty windows")
    if storage not in {"cpu", "cuda"}:
        raise ValueError("storage must be 'cpu' or 'cuda'")

    compute_device = torch.device(compute_device)
    encoder.eval()
    ranges = chunk_ranges(len(frames), batch_size)
    baseline_allocated = 0
    baseline_reserved = 0
    if compute_device.type == "cuda":
        baseline_allocated = int(torch.cuda.memory_allocated(compute_device))
        baseline_reserved = int(torch.cuda.memory_reserved(compute_device))
        torch.cuda.reset_peak_memory_stats(compute_device)

    setup_seconds = 0.0
    input_copy_seconds = 0.0
    encoder_seconds = 0.0
    storage_allocation_seconds = 0.0
    output_store_copy_seconds = 0.0
    batches: list[dict[str, Any]] = []
    store: torch.Tensor | None = None
    output_shape: tuple[int, ...] | None = None

    complete_started = time.perf_counter()
    for start, end in ranges:
        setup_started = time.perf_counter()
        frame_chunk = frames[start:end]
        kwargs_chunk = stack_conditioning(conditioning[start:end])
        setup_seconds += time.perf_counter() - setup_started

        def copy_inputs():
            return (
                frame_chunk.to(compute_device),
                {
                    key: value.to(compute_device)
                    for key, value in kwargs_chunk.items()
                },
            )

        (device_frames, device_kwargs), copy_seconds = _elapsed(
            copy_inputs,
            sync_device=compute_device,
        )
        outputs, encoded_seconds = _elapsed(
            lambda: encoder(
                frames=device_frames,
                **device_kwargs,
                return_dict=True,
            ),
            sync_device=compute_device,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if not isinstance(hidden, torch.Tensor):
            raise BatchedEncoderPrefillError("encoder output lacks last_hidden_state")
        if expected_dtype is not None and hidden.dtype != expected_dtype:
            raise BatchedEncoderPrefillError(
                f"encoder dtype {hidden.dtype} != expected {expected_dtype}"
            )
        if not bool(torch.isfinite(hidden).all().item()):
            raise BatchedEncoderPrefillError("encoder store output is not finite")
        current_shape = tuple(hidden.shape[1:])
        if output_shape is None:
            output_shape = current_shape
            store_device = (
                torch.device("cpu")
                if storage == "cpu"
                else compute_device
            )

            def allocate_store():
                return torch.empty(
                    (len(frames), *current_shape),
                    dtype=hidden.dtype,
                    device=store_device,
                )

            store, allocation_seconds = _elapsed(
                allocate_store,
                sync_device=compute_device if store_device.type == "cuda" else None,
            )
            storage_allocation_seconds += allocation_seconds
        elif output_shape != current_shape:
            raise BatchedEncoderPrefillError(
                "encoder output shape changed between precompute batches"
            )
        assert store is not None
        source = hidden if storage == "cuda" else hidden.cpu()
        _, store_seconds = _elapsed(
            lambda: store[start:end].copy_(source),
            sync_device=compute_device if storage == "cuda" else None,
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
        del outputs, hidden, device_frames, device_kwargs, source

    if compute_device.type == "cuda":
        _sync_if_cuda(compute_device)
    complete_seconds = time.perf_counter() - complete_started
    if store is None:
        raise BatchedEncoderPrefillError("encoder prefill produced no output")

    peak_allocated = baseline_allocated
    peak_reserved = baseline_reserved
    if compute_device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(compute_device))
        peak_reserved = int(torch.cuda.max_memory_reserved(compute_device))

    evidence = {
        "version": PREFILL_VERSION,
        "result_class": "documented-drift",
        "production_default": False,
        "tiger_pr": 120,
        "batch_size": batch_size,
        "storage": storage,
        "live_window_count": len(frames),
        "batch_count": len(batches),
        "frame_contract": _tensor_contract(frames),
        "conditioning_sha256": conditioning_digest(list(conditioning)),
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
        "incremental_peak_allocated_vram_bytes": peak_allocated - baseline_allocated,
        "incremental_peak_reserved_vram_bytes": peak_reserved - baseline_reserved,
        "drift_reference": "external_b1_same-input audit",
        "drift_charged_to_request_wall": False,
        "batches": batches,
    }
    return store, evidence


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
class BatchedEncoderPrefillStore:
    """Opt-in manager: precompute once, inject per window during generate."""

    batch_size: int = DEFAULT_ENCODER_BATCH_SIZE
    storage: StorageDevice = "cpu"
    _entries: list[_StoreEntry] = field(default_factory=list, init=False, repr=False)
    _active: dict[int, _GenerationUse] = field(default_factory=dict, init=False)
    _labels: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError("encoder prefill batch_size must be an integer")
        if self.batch_size <= 0:
            raise ValueError("encoder prefill batch_size must be positive")
        if self.storage not in {"cpu", "cuda"}:
            raise ValueError("storage must be 'cpu' or 'cuda'")

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
            "version": PREFILL_VERSION,
            "result_class": "documented-drift",
            "production_default": False,
            "tiger_pr": 120,
            "batch_size": self.batch_size,
            "storage": self.storage,
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
            raise BatchedEncoderPrefillError(
                "batched encoder prefill requires sequential generation"
            )
        if processor.precision not in SUPPORTED_PRECISIONS:
            raise BatchedEncoderPrefillError(
                f"batched encoder prefill requires precision in "
                f"{SUPPORTED_PRECISIONS}, got {processor.precision!r}"
            )
        if processor.cfg_scale != 1.0:
            raise BatchedEncoderPrefillError(
                "batched encoder prefill requires cfg_scale=1"
            )
        if processor.inference_runtime is None:
            raise BatchedEncoderPrefillError(
                "batched encoder prefill requires the optimized runtime"
            )
        if processor.device != "cuda":
            raise BatchedEncoderPrefillError(
                "batched encoder prefill requires device=cuda"
            )
        if not torch.cuda.is_available():
            raise BatchedEncoderPrefillError(
                "batched encoder prefill requires an allocated CUDA device"
            )
        if processor.model.device.type != "cuda":
            raise BatchedEncoderPrefillError(
                "batched encoder prefill model must be resident on CUDA"
            )
        expected_dtype = (
            torch.float16 if processor.precision == "fp16" else torch.float32
        )
        if processor.model.dtype != expected_dtype:
            raise BatchedEncoderPrefillError(
                f"model dtype {processor.model.dtype} != {expected_dtype} for "
                f"precision={processor.precision}"
            )
        if self.batch_size > int(processor.max_batch_size):
            raise BatchedEncoderPrefillError(
                f"encoder prefill B{self.batch_size} exceeds max_batch_size="
                f"{processor.max_batch_size}"
            )
        if self.storage == "cuda" and processor.model.device.type != "cuda":
            raise BatchedEncoderPrefillError("cuda storage requires a CUDA model")

    @staticmethod
    def _conditioning_equal(
        reference: list[dict[str, torch.Tensor]],
        candidate: list[dict[str, torch.Tensor]],
    ) -> bool:
        if len(reference) != len(candidate):
            return False
        for reference_row, candidate_row in zip(reference, candidate, strict=True):
            if tuple(sorted(reference_row)) != tuple(sorted(candidate_row)):
                return False
            for key in reference_row:
                if not torch.equal(reference_row[key], candidate_row[key]):
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

    def begin_generation(
        self,
        processor,
        *,
        sequences,
        generation_config,
        profile_label: str,
    ) -> None:
        if profile_label not in SUPPORTED_LABELS:
            raise BatchedEncoderPrefillError(
                f"batched encoder prefill requires one of {SUPPORTED_LABELS}, "
                f"got {profile_label!r}"
            )
        if profile_label in self._labels:
            raise BatchedEncoderPrefillError(
                f"generation label {profile_label!r} repeated"
            )
        expected_next = SUPPORTED_LABELS[len(self._labels)]
        if profile_label != expected_next:
            raise BatchedEncoderPrefillError(
                f"expected {expected_next!r}, got {profile_label!r}"
            )
        self._assert_processor(processor)
        frames, frame_times, song_length_ms = sequences
        if not isinstance(frames, torch.Tensor) or not isinstance(
            frame_times, torch.Tensor
        ):
            raise TypeError("sequences must be tensors")
        conditioning = window_conditioning(
            processor,
            generation_config,
            frame_times,
            float(song_length_ms),
        )
        if self._entries:
            first = self._entries[0]
            if (
                frames is not first.source_frames
                or frame_times is not first.source_frame_times
            ):
                raise BatchedEncoderPrefillError(
                    "timing and main stages must share the request's source tensors"
                )
        entry = self._compatible_entry(processor, sequences, conditioning)
        if entry is None:
            expected_dtype = (
                torch.float16 if processor.precision == "fp16" else torch.float32
            )
            store, evidence = precompute_encoder_hidden_states(
                processor.model.get_encoder(),
                processor.prepare_frames(frames),
                conditioning,
                batch_size=self.batch_size,
                compute_device=processor.model.device,
                storage=self.storage,
                expected_dtype=expected_dtype,
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
                    "precision": processor.precision,
                    "model_class": type(processor.model).__name__,
                    "tokenizer_class": type(processor.tokenizer).__name__,
                },
            )
            self._entries.append(entry)
        processor_id = id(processor)
        if processor_id in self._active:
            raise BatchedEncoderPrefillError("processor is already active")
        self._active[processor_id] = _GenerationUse(
            profile_label,
            processor_id,
            entry,
        )
        self._publish(processor)

    def inject(self, processor, model_kwargs: dict[str, Any]) -> dict[str, Any]:
        use = self._active.get(id(processor))
        if use is None:
            raise BatchedEncoderPrefillError(
                "model_generate called outside batched-encoder-prefill stage"
            )
        entry = use.entry
        if processor.model is not entry.model:
            raise BatchedEncoderPrefillError(
                "refused accidental cross-model encoder output reuse"
            )
        window_count = len(entry.source_frames)
        index = use.model_generate_calls % window_count
        source = model_kwargs.get("inputs")
        if not isinstance(source, torch.Tensor):
            raise BatchedEncoderPrefillError("model call is missing input frames")
        expected_source = entry.source_frames[index : index + 1]
        if (
            source.data_ptr() != expected_source.data_ptr()
            or source.storage_offset() != expected_source.storage_offset()
            or source.shape != expected_source.shape
            or source.dtype != expected_source.dtype
            or source.device != expected_source.device
        ):
            raise BatchedEncoderPrefillError(
                f"encoder source window {index} changed before reuse"
            )
        expected_conditioning = entry.conditioning[index]
        actual_conditioning = {
            key: value
            for key, value in model_kwargs.items()
            if key in ENCODER_CONDITIONING_KEYS
        }
        if tuple(sorted(actual_conditioning)) != tuple(sorted(expected_conditioning)):
            raise BatchedEncoderPrefillError(
                f"encoder conditioning keys changed at reuse window {index}"
            )
        for key, reference in expected_conditioning.items():
            value = actual_conditioning[key]
            if not isinstance(value, torch.Tensor) or not torch.equal(value, reference):
                raise BatchedEncoderPrefillError(
                    f"encoder conditioning {key!r} changed at reuse window {index}"
                )
        if model_kwargs.get("encoder_outputs") is not None:
            raise BatchedEncoderPrefillError(
                "model call already contains encoder_outputs"
            )
        use.model_generate_calls += 1
        hidden = entry.store[index : index + 1]
        if hidden.device.type == "cpu":
            hidden = hidden.to(
                device=processor.model.device,
                dtype=processor.model.dtype,
                non_blocking=False,
            )
        # Keep ``inputs`` so optimized ``_generate_window`` can still read
        # batch_size; the model skips encoder when encoder_outputs is set.
        return {
            **model_kwargs,
            "encoder_outputs": BaseModelOutput(last_hidden_state=hidden),
        }

    def end_generation(self, processor) -> None:
        use = self._active.pop(id(processor), None)
        if use is None:
            raise BatchedEncoderPrefillError("stage ended without begin")
        window_count = len(use.entry.source_frames)
        if use.model_generate_calls <= 0 or use.model_generate_calls % window_count:
            raise BatchedEncoderPrefillError(
                f"{use.label} used {use.model_generate_calls} encoder rows for "
                f"{window_count} windows"
            )
        self._labels.append(use.label)
        use.entry.labels.append(use.label)
        use.entry.evidence.setdefault("uses", {})[use.label] = {
            "model_generate_calls": use.model_generate_calls,
            "complete_window_passes": use.model_generate_calls // window_count,
            "rows_reused": use.model_generate_calls,
            "storage": self.storage,
            "per_window_host_to_device": self.storage == "cpu",
        }
        self._publish(processor)

    def _publish(self, processor) -> None:
        profiler = getattr(processor, "profiler", None)
        set_metadata = getattr(profiler, "set_metadata", None)
        if callable(set_metadata):
            set_metadata(optimized_batched_encoder_prefill=self.evidence)

    def finalize(self) -> dict[str, Any]:
        if self._active:
            raise BatchedEncoderPrefillError("finalized with active processors")
        if tuple(self._labels) != SUPPORTED_LABELS:
            raise BatchedEncoderPrefillError(
                f"requires timing then main reuse; got {self._labels}"
            )
        evidence = self.evidence
        if len(self._entries) not in {1, 2}:
            raise BatchedEncoderPrefillError(
                f"expected one shared or two independent stores, got "
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
                    raise BatchedEncoderPrefillError(
                        f"entry {entry_index} evidence {key} is invalid"
                    )
        return evidence


@contextmanager
def install_batched_encoder_prefill_candidate(
    processor_class,
    *,
    batch_size: int = DEFAULT_ENCODER_BATCH_SIZE,
    storage: StorageDevice = "cpu",
) -> Iterator[BatchedEncoderPrefillStore]:
    """Temporarily install W3 reuse without touching selectors or defaults."""

    manager = BatchedEncoderPrefillStore(batch_size=batch_size, storage=storage)
    original_generate = processor_class.generate
    original_model_generate = processor_class.model_generate

    def generate_with_prefill(self, *args, **kwargs):
        sequences = kwargs.get("sequences")
        generation_config = kwargs.get("generation_config")
        profile_label = kwargs.get("profile_label")
        if sequences is None or generation_config is None:
            raise BatchedEncoderPrefillError(
                "requires keyword sequences and generation_config"
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

    def model_generate_with_prefill(self, model_kwargs, **generate_kwargs):
        return original_model_generate(
            self,
            manager.inject(self, model_kwargs),
            **generate_kwargs,
        )

    processor_class.generate = generate_with_prefill
    processor_class.model_generate = model_generate_with_prefill
    try:
        yield manager
    finally:
        processor_class.generate = original_generate
        processor_class.model_generate = original_model_generate


__all__ = [
    "DEFAULT_ENCODER_BATCH_SIZE",
    "ENCODER_CONDITIONING_KEYS",
    "PREFILL_VERSION",
    "SUPPORTED_LABELS",
    "SUPPORTED_PRECISIONS",
    "BatchedEncoderPrefillError",
    "BatchedEncoderPrefillStore",
    "chunk_ranges",
    "conditioning_digest",
    "encoder_hidden_drift",
    "install_batched_encoder_prefill_candidate",
    "precompute_encoder_hidden_states",
    "stack_conditioning",
    "window_conditioning",
]
