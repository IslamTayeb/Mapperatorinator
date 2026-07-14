"""Opt-in exact B1 main-encoder overlap experiment."""

from __future__ import annotations

import hashlib
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import torch
from transformers.modeling_outputs import BaseModelOutput

OVERLAP_VERSION = "exact-b1-main-encoder-overlap-v1"
MATERIAL_AUDIT_VERSION = "encoder-cross-kv-material-audit-v1"
SUPPORTED_LABELS = ("timing_context", "main_generation")
ENCODER_CONDITIONING_KEYS = (
    "beatmap_idx",
    "difficulty",
    "mapper_idx",
    "song_position",
)


class EncoderOverlapError(RuntimeError):
    pass


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
    processor: Any,
    generation_config: Any,
    frame_times: torch.Tensor,
    song_length_ms: float,
) -> list[dict[str, torch.Tensor]]:
    static = processor._get_model_cond_kwargs(generation_config)
    unsupported = sorted(set(static) - set(ENCODER_CONDITIONING_KEYS))
    if unsupported:
        raise EncoderOverlapError(
            f"encoder overlap does not own conditioning keys {unsupported}"
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


def _conditioning_equal(
    reference: list[dict[str, torch.Tensor]],
    candidate: list[dict[str, torch.Tensor]],
) -> bool:
    if len(reference) != len(candidate):
        return False
    for reference_row, candidate_row in zip(reference, candidate, strict=True):
        if tuple(sorted(reference_row)) != tuple(sorted(candidate_row)):
            return False
        if any(
            not torch.equal(reference_row[key], candidate_row[key])
            for key in reference_row
        ):
            return False
    return True


def graph_manifest(processor: Any) -> dict[str, Any]:
    state = getattr(processor, "decode_session_state", None)
    summarize = getattr(state, "graph_profile_summary", None)
    if not callable(summarize):
        raise EncoderOverlapError("timing decoder lacks accepted CUDA graph state")
    summary = summarize()
    buckets = summary.get("buckets")
    graph_count = summary.get("graph_count")
    if not isinstance(buckets, dict) or not isinstance(graph_count, int):
        raise EncoderOverlapError("timing CUDA graph manifest is malformed")
    canonical: dict[str, int] = {}
    for raw_prefix, row in buckets.items():
        if not isinstance(row, dict):
            raise EncoderOverlapError("timing CUDA graph bucket is malformed")
        count = row.get("graph_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise EncoderOverlapError("timing CUDA graph count must be positive")
        prefix = str(int(raw_prefix))
        if prefix != str(raw_prefix):
            raise EncoderOverlapError("timing CUDA graph prefix is not canonical")
        canonical[prefix] = count
    if graph_count <= 0 or graph_count != sum(canonical.values()):
        raise EncoderOverlapError("timing CUDA graph total does not match buckets")
    return {
        "graph_count": graph_count,
        "buckets": dict(sorted(canonical.items(), key=lambda item: int(item[0]))),
    }


@dataclass
class _OverlapRow:
    hidden: torch.Tensor
    ready: torch.cuda.Event


@dataclass
class ExactMainEncoderOverlap:
    stable_observations_required: int = 2
    _processors: list[Any] = field(default_factory=list, init=False, repr=False)
    _main_processor: Any | None = field(default=None, init=False, repr=False)
    _timing_processor: Any | None = field(default=None, init=False, repr=False)
    _source_frames: torch.Tensor | None = field(default=None, init=False, repr=False)
    _source_times: torch.Tensor | None = field(default=None, init=False, repr=False)
    _conditioning: list[dict[str, torch.Tensor]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _pinned_frames: torch.Tensor | None = field(default=None, init=False, repr=False)
    _pinned_conditioning: list[dict[str, torch.Tensor]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _rows: list[_OverlapRow] = field(default_factory=list, init=False, repr=False)
    _stream: torch.cuda.Stream | None = field(default=None, init=False, repr=False)
    _encoder_start: torch.cuda.Event | None = field(default=None, init=False, repr=False)
    _encoder_end: torch.cuda.Event | None = field(default=None, init=False, repr=False)
    _timing_end: torch.cuda.Event | None = field(default=None, init=False, repr=False)
    _join_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _active_label: str | None = field(default=None, init=False)
    _labels: list[str] = field(default_factory=list, init=False, repr=False)
    _timing_calls: int = field(default=0, init=False)
    _main_calls: int = field(default=0, init=False)
    _previous_manifest: dict[str, Any] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _stable_observations: int = field(default=0, init=False)
    _launch_manifest: dict[str, Any] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _launch_after_timing_window: int | None = field(default=None, init=False)
    _manifest_changed_after_launch: bool = field(default=False, init=False)
    _pinned_setup_seconds: float = field(default=0.0, init=False)
    _launch_host_seconds: float = field(default=0.0, init=False)
    _baseline_allocated: int = field(default=0, init=False)
    _baseline_reserved: int = field(default=0, init=False)
    _least_stream_priority: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.stable_observations_required, bool)
            or not isinstance(self.stable_observations_required, int)
            or self.stable_observations_required < 2
        ):
            raise ValueError("encoder overlap requires at least two stable observations")

    def register_processor(self, processor: Any) -> None:
        if processor in self._processors:
            raise EncoderOverlapError("processor registered twice")
        if len(self._processors) >= 2:
            raise EncoderOverlapError(
                "exact overlap scout supports only main and timing processors"
            )
        self._processors.append(processor)
        if len(self._processors) == 1:
            self._main_processor = processor
        else:
            self._timing_processor = processor

    @staticmethod
    def _assert_processor(processor: Any, *, role: str) -> None:
        failures = []
        if processor.parallel:
            failures.append("parallel")
        if processor.precision != "fp32":
            failures.append(f"precision={processor.precision}")
        if processor.cfg_scale != 1.0:
            failures.append(f"cfg_scale={processor.cfg_scale}")
        if processor.inference_runtime is None:
            failures.append("runtime=v32")
        if processor.device != "cuda":
            failures.append(f"device={processor.device}")
        if not torch.cuda.is_available():
            failures.append("cuda_unavailable")
        if getattr(processor.model, "device", None) is None:
            failures.append("model_device_missing")
        elif processor.model.device.type != "cuda":
            failures.append(f"model_device={processor.model.device}")
        if getattr(processor.model, "dtype", None) != torch.float32:
            failures.append(f"model_dtype={getattr(processor.model, 'dtype', None)}")
        if failures:
            raise EncoderOverlapError(f"{role} processor is unsupported: {failures}")

    def _prepare_pinned_inputs(self, frames: torch.Tensor) -> None:
        if frames.device.type != "cpu" or frames.dtype != torch.float32:
            raise EncoderOverlapError(
                "overlap source frames must be contiguous FP32 CPU storage"
            )
        started = time.perf_counter()
        self._pinned_frames = frames.contiguous().pin_memory()
        self._pinned_conditioning = [
            {
                key: value.detach().contiguous().pin_memory()
                for key, value in row.items()
            }
            for row in self._conditioning
        ]
        self._pinned_setup_seconds = time.perf_counter() - started

    def begin_generation(
        self,
        processor: Any,
        *,
        sequences: tuple[torch.Tensor, torch.Tensor, float],
        generation_config: Any,
        profile_label: str,
    ) -> None:
        if self._active_label is not None:
            raise EncoderOverlapError("overlap scout does not support nested generation")
        expected = SUPPORTED_LABELS[len(self._labels)] if len(self._labels) < 2 else None
        if profile_label != expected:
            raise EncoderOverlapError(
                f"overlap scout expected {expected!r}, got {profile_label!r}"
            )
        if profile_label == "timing_context":
            if len(self._processors) != 2 or processor is not self._timing_processor:
                raise EncoderOverlapError(
                    "timing generation began before distinct main/timing processors existed"
                )
            assert self._main_processor is not None
            self._assert_processor(processor, role="timing")
            self._assert_processor(self._main_processor, role="main")
            if processor.model is self._main_processor.model:
                raise EncoderOverlapError(
                    "exact overlap requires standard SALVALAI's distinct timing/main models"
                )
            frames, frame_times, song_length_ms = sequences
            if not isinstance(frames, torch.Tensor) or not isinstance(
                frame_times,
                torch.Tensor,
            ):
                raise TypeError("overlap sequences must contain tensors")
            conditioning = _window_conditioning(
                self._main_processor,
                generation_config,
                frame_times,
                float(song_length_ms),
            )
            if len(frames) == 0 or len(frames) != len(conditioning):
                raise EncoderOverlapError("overlap requires matched nonempty windows")
            self._source_frames = frames
            self._source_times = frame_times
            self._conditioning = conditioning
            self._prepare_pinned_inputs(frames)
            device = self._main_processor.model.device
            self._baseline_allocated = int(torch.cuda.memory_allocated(device))
            self._baseline_reserved = int(torch.cuda.memory_reserved(device))
            torch.cuda.reset_peak_memory_stats(device)
        else:
            if processor is not self._main_processor:
                raise EncoderOverlapError("main generation used an unexpected processor")
            if self._launch_manifest is None or not self._rows:
                raise EncoderOverlapError(
                    "timing graph manifest never stabilized; main overlap is unavailable"
                )
            frames, frame_times, _ = sequences
            if frames is not self._source_frames or frame_times is not self._source_times:
                raise EncoderOverlapError(
                    "timing and main stages must share the request source tensors"
                )
            candidate = _window_conditioning(
                processor,
                generation_config,
                frame_times,
                float(sequences[2]),
            )
            if not _conditioning_equal(self._conditioning, candidate):
                raise EncoderOverlapError(
                    "main encoder conditioning changed after overlap launch"
                )
        self._active_label = profile_label

    @torch.no_grad()
    def _launch(self) -> None:
        if self._main_processor is None or self._pinned_frames is None:
            raise EncoderOverlapError("overlap launch lacks main inputs")
        if self._rows or self._stream is not None:
            raise EncoderOverlapError("overlap encoder launched more than once")
        device = self._main_processor.model.device
        least_priority, _greatest_priority = torch.cuda.get_stream_priority_range()
        self._least_stream_priority = int(least_priority)
        stream = torch.cuda.Stream(device=device, priority=least_priority)
        # Own the stream before enqueueing anything so the context's failure
        # path can always synchronize work submitted by a partially built row.
        self._stream = stream
        current = torch.cuda.current_stream(device)
        stream.wait_stream(current)
        encoder = self._main_processor.model.get_encoder()
        encoder.eval()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        rows: list[_OverlapRow] = []
        started = time.perf_counter()
        try:
            with torch.cuda.stream(stream):
                start_event.record(stream)
                for index in range(len(self._pinned_frames)):
                    frame = self._pinned_frames[index : index + 1].to(
                        device,
                        non_blocking=True,
                    )
                    kwargs = {
                        key: value.to(device, non_blocking=True)
                        for key, value in self._pinned_conditioning[index].items()
                    }
                    outputs = encoder(frames=frame, **kwargs, return_dict=True)
                    hidden = getattr(outputs, "last_hidden_state", None)
                    if not isinstance(hidden, torch.Tensor):
                        raise EncoderOverlapError(
                            f"main encoder row {index} lacks last_hidden_state"
                        )
                    if hidden.dtype != torch.float32 or hidden.device != device:
                        raise EncoderOverlapError(
                            f"main encoder row {index} changed storage dtype/device"
                        )
                    ready = torch.cuda.Event(enable_timing=True)
                    ready.record(stream)
                    rows.append(_OverlapRow(hidden=hidden, ready=ready))
                end_event.record(stream)
        except BaseException:
            stream.synchronize()
            raise
        self._launch_host_seconds = time.perf_counter() - started
        self._encoder_start = start_event
        self._encoder_end = end_event
        self._rows = rows

    def after_timing_model_generate(self, processor: Any) -> None:
        if self._active_label != "timing_context":
            raise EncoderOverlapError("timing hook ran outside timing generation")
        manifest = graph_manifest(processor)
        self._timing_calls += 1
        if self._launch_manifest is not None:
            if manifest != self._launch_manifest:
                self._manifest_changed_after_launch = True
                self.abort()
                raise EncoderOverlapError(
                    "timing CUDA graph manifest changed after overlap launch"
                )
            return
        if manifest == self._previous_manifest:
            self._stable_observations += 1
        else:
            self._previous_manifest = manifest
            self._stable_observations = 1
        if self._stable_observations >= self.stable_observations_required:
            self._launch_manifest = manifest
            self._launch_after_timing_window = self._timing_calls
            self._launch()
            self._publish(processor)

    def inject_main(self, processor: Any, model_kwargs: dict[str, Any]) -> dict[str, Any]:
        if self._active_label != "main_generation":
            raise EncoderOverlapError("main encoder injection ran outside main generation")
        if processor is not self._main_processor or self._source_frames is None:
            raise EncoderOverlapError("main encoder injection used the wrong processor")
        index = self._main_calls
        if index >= len(self._rows):
            raise EncoderOverlapError("main generation requested too many encoder rows")
        source = model_kwargs.get("inputs")
        expected = self._source_frames[index : index + 1]
        if not isinstance(source, torch.Tensor) or (
            source.data_ptr() != expected.data_ptr()
            or source.storage_offset() != expected.storage_offset()
            or source.shape != expected.shape
            or source.dtype != expected.dtype
            or source.device != expected.device
        ):
            raise EncoderOverlapError(f"main source row {index} changed before reuse")
        actual_conditioning = {
            key: value
            for key, value in model_kwargs.items()
            if key in ENCODER_CONDITIONING_KEYS
        }
        expected_conditioning = self._conditioning[index]
        if tuple(sorted(actual_conditioning)) != tuple(sorted(expected_conditioning)):
            raise EncoderOverlapError(
                f"main conditioning keys changed at row {index}"
            )
        for key, reference in expected_conditioning.items():
            value = actual_conditioning[key]
            if not isinstance(value, torch.Tensor) or not torch.equal(value, reference):
                raise EncoderOverlapError(
                    f"main conditioning {key!r} changed at row {index}"
                )
        if model_kwargs.get("encoder_outputs") is not None:
            raise EncoderOverlapError("main call already contains encoder_outputs")
        current = torch.cuda.current_stream(processor.model.device)
        join_start = torch.cuda.Event(enable_timing=True)
        join_end = torch.cuda.Event(enable_timing=True)
        join_start.record(current)
        current.wait_event(self._rows[index].ready)
        join_end.record(current)
        self._rows[index].hidden.record_stream(current)
        self._join_events.append((join_start, join_end))
        self._main_calls += 1
        return {
            **model_kwargs,
            "encoder_outputs": BaseModelOutput(
                last_hidden_state=self._rows[index].hidden
            ),
        }

    def end_generation(self, processor: Any) -> None:
        label = self._active_label
        if label is None:
            raise EncoderOverlapError("overlap stage ended without begin")
        if label == "timing_context":
            if self._launch_manifest is None:
                raise EncoderOverlapError(
                    "timing graph manifest did not stabilize before timing ended"
                )
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(processor.model.device))
            self._timing_end = event
        else:
            expected_rows = len(self._rows)
            if self._main_calls != expected_rows:
                raise EncoderOverlapError(
                    f"main consumed {self._main_calls}/{expected_rows} encoder rows"
                )
        self._labels.append(label)
        self._active_label = None
        self._publish(processor)

    def _publish(self, processor: Any) -> None:
        profiler = getattr(processor, "profiler", None)
        set_metadata = getattr(profiler, "set_metadata", None)
        if callable(set_metadata):
            set_metadata(optimized_exact_main_encoder_overlap=self.evidence())

    def evidence(self, *, synchronize: bool = False) -> dict[str, Any]:
        if synchronize and self._encoder_end is not None:
            self._encoder_end.synchronize()
            torch.cuda.synchronize()
        encoder_seconds = None
        overlap_seconds = None
        join_seconds = None
        per_row_join = None
        if (
            synchronize
            and self._encoder_start is not None
            and self._encoder_end is not None
            and self._timing_end is not None
        ):
            encoder_seconds = self._encoder_start.elapsed_time(self._encoder_end) / 1000.0
            until_timing_end = max(
                0.0,
                self._encoder_start.elapsed_time(self._timing_end) / 1000.0,
            )
            overlap_seconds = min(encoder_seconds, until_timing_end)
            per_row_join = [
                start.elapsed_time(end) / 1000.0
                for start, end in self._join_events
            ]
            join_seconds = sum(per_row_join)
        output_bytes = sum(
            row.hidden.numel() * row.hidden.element_size() for row in self._rows
        )
        pinned_input_bytes = (
            (
                self._pinned_frames.numel() * self._pinned_frames.element_size()
                if self._pinned_frames is not None
                else 0
            )
            + sum(
                value.numel() * value.element_size()
                for row in self._pinned_conditioning
                for value in row.values()
            )
        )
        result = {
            "version": OVERLAP_VERSION,
            "result_class": "exact-output-candidate",
            "production_default": False,
            "batch_size": 1,
            "separate_timing_main_models": (
                self._main_processor is not None
                and self._timing_processor is not None
                and self._main_processor.model is not self._timing_processor.model
            ),
            "labels_completed": list(self._labels),
            "live_window_count": (
                len(self._source_frames) if self._source_frames is not None else 0
            ),
            "timing_model_generate_calls": self._timing_calls,
            "main_model_generate_calls": self._main_calls,
            "stable_observations_required": self.stable_observations_required,
            "stable_observations_at_launch": self._stable_observations,
            "launch_after_timing_window": (
                self._launch_after_timing_window
            ),
            "launch_manifest": self._launch_manifest,
            "manifest_changed_after_launch": self._manifest_changed_after_launch,
            "low_priority_stream_priority": self._least_stream_priority,
            "pinned_input_setup_seconds": self._pinned_setup_seconds,
            "launch_host_seconds": self._launch_host_seconds,
            "encoder_elapsed_seconds": encoder_seconds,
            "encoder_overlap_with_timing_seconds": overlap_seconds,
            "encoder_after_timing_seconds": (
                None
                if encoder_seconds is None or overlap_seconds is None
                else max(0.0, encoder_seconds - overlap_seconds)
            ),
            "main_join_wait_seconds": join_seconds,
            "per_row_main_join_wait_seconds": per_row_join,
            "pinned_input_bytes": pinned_input_bytes,
            "device_input_copy_bytes": pinned_input_bytes,
            "output_store_bytes": output_bytes,
            "baseline_allocated_vram_bytes": self._baseline_allocated,
            "baseline_reserved_vram_bytes": self._baseline_reserved,
            "peak_allocated_vram_bytes": (
                int(torch.cuda.max_memory_allocated()) if synchronize and self._rows else None
            ),
            "peak_reserved_vram_bytes": (
                int(torch.cuda.max_memory_reserved()) if synchronize and self._rows else None
            ),
            "source_contract": (
                _tensor_contract(self._source_frames)
                if self._source_frames is not None
                else None
            ),
            "conditioning_sha256": (
                _conditioning_digest(self._conditioning)
                if self._conditioning
                else None
            ),
            "all_encoder_outputs_finite": (
                all(bool(torch.isfinite(row.hidden).all().item()) for row in self._rows)
                if synchronize
                else None
            ),
        }
        if synchronize and result["peak_allocated_vram_bytes"] is not None:
            result["incremental_peak_allocated_vram_bytes"] = (
                result["peak_allocated_vram_bytes"] - self._baseline_allocated
            )
            result["incremental_peak_reserved_vram_bytes"] = (
                result["peak_reserved_vram_bytes"] - self._baseline_reserved
            )
        else:
            result["incremental_peak_allocated_vram_bytes"] = None
            result["incremental_peak_reserved_vram_bytes"] = None
        return result

    def finalize(self) -> dict[str, Any]:
        if self._active_label is not None:
            raise EncoderOverlapError("overlap finalized with an active stage")
        if tuple(self._labels) != SUPPORTED_LABELS:
            raise EncoderOverlapError(
                f"overlap requires timing then main; got {self._labels}"
            )
        result = self.evidence(synchronize=True)
        if result["manifest_changed_after_launch"]:
            raise EncoderOverlapError("timing graph manifest changed after launch")
        if not result["all_encoder_outputs_finite"]:
            raise EncoderOverlapError("main encoder overlap produced non-finite output")
        required = (
            "encoder_elapsed_seconds",
            "encoder_overlap_with_timing_seconds",
            "encoder_after_timing_seconds",
            "main_join_wait_seconds",
        )
        for key in required:
            value = result[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise EncoderOverlapError(f"overlap evidence {key} is invalid")
        return result

    def abort(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()


def _tensor_material(value: torch.Tensor, *, name: str) -> tuple[dict[str, Any], int]:
    if not isinstance(value, torch.Tensor):
        raise EncoderOverlapError(f"{name} must be a tensor")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all().item()):
        raise EncoderOverlapError(f"{name} must be finite floating-point material")
    contiguous = value.detach().contiguous().cpu()
    raw = contiguous.view(torch.uint8).numpy().tobytes()
    return (
        {
            "name": name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "contract": _tensor_contract(value),
            "bytes": len(raw),
        },
        len(raw),
    )


def _rng_material() -> dict[str, str]:
    result = {
        "cpu": hashlib.sha256(
            torch.get_rng_state().contiguous().numpy().tobytes()
        ).hexdigest()
    }
    if torch.cuda.is_available():
        result["cuda"] = hashlib.sha256(
            torch.cuda.get_rng_state().contiguous().cpu().numpy().tobytes()
        ).hexdigest()
    return result


@dataclass
class EncoderMaterialAudit:
    """Capture exact last-window encoder and cross-KV bytes for both arms."""

    _processors: list[Any] = field(default_factory=list, init=False, repr=False)
    _labels: list[str] = field(default_factory=list, init=False, repr=False)
    _stages: dict[str, dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def register_processor(self, processor: Any) -> None:
        if processor in self._processors:
            raise EncoderOverlapError("material-audit processor registered twice")
        if len(self._processors) >= 2:
            raise EncoderOverlapError("material audit supports exactly two processors")
        self._processors.append(processor)

    def capture(self, processor: Any, *, profile_label: str) -> None:
        expected = SUPPORTED_LABELS[len(self._labels)] if len(self._labels) < 2 else None
        if profile_label != expected:
            raise EncoderOverlapError(
                f"material audit expected {expected!r}, got {profile_label!r}"
            )
        if processor not in self._processors:
            raise EncoderOverlapError("material audit saw an unregistered processor")
        if profile_label == "timing_context":
            self._labels.append(profile_label)
            self._stages[profile_label] = {
                "profile_label": profile_label,
                "captured": False,
                "reason": "avoid_synchronizing_main_encoder_overlap_stream",
                "copied_bytes": 0,
                "copy_and_hash_seconds": 0.0,
            }
            profiler = getattr(processor, "profiler", None)
            set_metadata = getattr(profiler, "set_metadata", None)
            if callable(set_metadata):
                set_metadata(optimized_encoder_material_audit=self.evidence())
            return
        state = getattr(processor, "decode_session_state", None)
        signature = getattr(state, "active_state_signature", None)
        if signature is None:
            raise EncoderOverlapError("material audit lacks an active decode signature")
        holders = getattr(state, "stable_encoder_holders", None)
        caches = getattr(state, "caches", None)
        if not isinstance(holders, dict) or not isinstance(caches, dict):
            raise EncoderOverlapError("material audit lacks stable encoder/cache state")
        holder = holders.get(signature)
        cache = caches.get(signature)
        encoder_outputs = holder.get("encoder_outputs") if isinstance(holder, dict) else None
        encoder_hidden = getattr(encoder_outputs, "last_hidden_state", None)
        cross = getattr(cache, "cross_attention_cache", None)
        layers = getattr(cross, "layers", None)
        if not isinstance(encoder_hidden, torch.Tensor):
            raise EncoderOverlapError("material audit lacks raw encoder output")
        if not isinstance(layers, (list, tuple)) or not layers:
            raise EncoderOverlapError("material audit lacks cross-attention cache layers")
        if not all(bool(getattr(layer, "is_initialized", False)) for layer in layers):
            raise EncoderOverlapError("material audit found uninitialized cross cache")
        updated = getattr(cache, "is_updated", None)
        if not isinstance(updated, dict) or not all(
            bool(updated.get(index)) for index in range(len(layers))
        ):
            raise EncoderOverlapError("material audit found incomplete cross-cache ownership")

        torch.cuda.synchronize(processor.model.device)
        started = time.perf_counter()
        encoder, copied_bytes = _tensor_material(
            encoder_hidden,
            name="encoder.last_hidden_state",
        )
        cross_rows: list[dict[str, Any]] = []
        for index, layer in enumerate(layers):
            keys, key_bytes = _tensor_material(
                getattr(layer, "keys", None),
                name=f"cross.layers.{index}.keys",
            )
            values, value_bytes = _tensor_material(
                getattr(layer, "values", None),
                name=f"cross.layers.{index}.values",
            )
            copied_bytes += key_bytes + value_bytes
            cross_rows.append({"layer": index, "keys": keys, "values": values})
        torch.cuda.synchronize(processor.model.device)
        copy_seconds = time.perf_counter() - started
        aggregate = hashlib.sha256()
        aggregate.update(encoder["sha256"].encode("ascii"))
        for row in cross_rows:
            aggregate.update(row["keys"]["sha256"].encode("ascii"))
            aggregate.update(row["values"]["sha256"].encode("ascii"))
        stage = {
            "profile_label": profile_label,
            "captured": True,
            "encoder": encoder,
            "cross_kv": cross_rows,
            "aggregate_sha256": aggregate.hexdigest(),
            "copied_bytes": copied_bytes,
            "copy_and_hash_seconds": copy_seconds,
            "rng_after_stage": _rng_material(),
            "cross_layer_count": len(cross_rows),
        }
        self._labels.append(profile_label)
        self._stages[profile_label] = stage
        profiler = getattr(processor, "profiler", None)
        set_metadata = getattr(profiler, "set_metadata", None)
        if callable(set_metadata):
            set_metadata(optimized_encoder_material_audit=self.evidence())

    def evidence(self) -> dict[str, Any]:
        return {
            "version": MATERIAL_AUDIT_VERSION,
            "production_default": False,
            "labels_completed": list(self._labels),
            "processor_count": len(self._processors),
            "stages": self._stages,
            "total_copied_bytes": sum(
                int(stage["copied_bytes"]) for stage in self._stages.values()
            ),
            "total_copy_and_hash_seconds": sum(
                float(stage["copy_and_hash_seconds"])
                for stage in self._stages.values()
            ),
        }

    def finalize(self) -> dict[str, Any]:
        if tuple(self._labels) != SUPPORTED_LABELS or len(self._processors) != 2:
            raise EncoderOverlapError(
                "material audit requires two processors and timing/main completion"
            )
        result = self.evidence()
        for label in SUPPORTED_LABELS:
            stage = result["stages"].get(label)
            if not isinstance(stage, dict):
                raise EncoderOverlapError(f"material audit lacks {label} evidence")
        if result["stages"]["timing_context"] != {
            "profile_label": "timing_context",
            "captured": False,
            "reason": "avoid_synchronizing_main_encoder_overlap_stream",
            "copied_bytes": 0,
            "copy_and_hash_seconds": 0.0,
        }:
            raise EncoderOverlapError("timing material audit unexpectedly synchronized")
        if result["stages"]["main_generation"].get("cross_layer_count", 0) <= 0:
            raise EncoderOverlapError("material audit lacks main raw material")
        return result


@contextmanager
def install_encoder_material_audit(
    processor_class: type,
) -> Iterator[EncoderMaterialAudit]:
    audit = EncoderMaterialAudit()
    original_init = processor_class.__init__
    original_generate = processor_class.generate

    def init_with_audit(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        audit.register_processor(self)

    def generate_with_audit(self, *args, **kwargs):
        profile_label = kwargs.get("profile_label")
        result = original_generate(self, *args, **kwargs)
        audit.capture(self, profile_label=profile_label)
        return result

    processor_class.__init__ = init_with_audit
    processor_class.generate = generate_with_audit
    try:
        yield audit
    finally:
        processor_class.__init__ = original_init
        processor_class.generate = original_generate


@contextmanager
def install_exact_main_encoder_overlap(
    processor_class: type,
    *,
    stable_observations_required: int = 2,
) -> Iterator[ExactMainEncoderOverlap]:
    """Temporarily install exact overlap without changing runtime selectors."""

    manager = ExactMainEncoderOverlap(
        stable_observations_required=stable_observations_required
    )
    original_init = processor_class.__init__
    original_generate = processor_class.generate
    original_model_generate = processor_class.model_generate

    def init_with_registration(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        manager.register_processor(self)

    def generate_with_overlap(self, *args, **kwargs):
        sequences = kwargs.get("sequences")
        generation_config = kwargs.get("generation_config")
        profile_label = kwargs.get("profile_label")
        if not isinstance(sequences, tuple) or generation_config is None:
            raise EncoderOverlapError(
                "overlap requires keyword sequences and generation_config"
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

    def model_generate_with_overlap(self, model_kwargs, **generate_kwargs):
        if manager._active_label == "main_generation":
            model_kwargs = manager.inject_main(self, model_kwargs)
        result = original_model_generate(self, model_kwargs, **generate_kwargs)
        if manager._active_label == "timing_context":
            manager.after_timing_model_generate(self)
        return result

    processor_class.__init__ = init_with_registration
    processor_class.generate = generate_with_overlap
    processor_class.model_generate = model_generate_with_overlap
    try:
        yield manager
    finally:
        manager.abort()
        processor_class.__init__ = original_init
        processor_class.generate = original_generate
        processor_class.model_generate = original_model_generate


__all__ = [
    "EncoderMaterialAudit",
    "EncoderOverlapError",
    "ExactMainEncoderOverlap",
    "MATERIAL_AUDIT_VERSION",
    "OVERLAP_VERSION",
    "graph_manifest",
    "install_encoder_material_audit",
    "install_exact_main_encoder_overlap",
]
