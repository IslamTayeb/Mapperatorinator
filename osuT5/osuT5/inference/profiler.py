from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.profiler import ProfilerActivity, profile, record_function


class InferenceProfiler:
    def __init__(
            self,
            *,
            enabled: bool = False,
            output_path: str | None = None,
            sync_cuda: bool = True,
            torch_generation: bool = False,
            torch_output_dir: str | None = None,
            torch_generation_limit: int = 3,
            torch_generation_label_filter: str | None = None,
            torch_record_shapes: bool = True,
            torch_profile_memory: bool = True,
            torch_with_stack: bool = False,
            torch_event_limit: int = 50,
            nvtx_generation_ranges: bool = False,
    ):
        self.enabled = enabled
        self.output_path = output_path
        self.sync_cuda = sync_cuda
        self.torch_generation = torch_generation
        self.torch_output_dir = torch_output_dir
        self.torch_generation_limit = max(0, torch_generation_limit)
        self.torch_generation_label_filter = torch_generation_label_filter
        self.torch_record_shapes = torch_record_shapes
        self.torch_profile_memory = torch_profile_memory
        self.torch_with_stack = torch_with_stack
        self.torch_event_limit = max(0, int(torch_event_limit))
        self.nvtx_generation_ranges = nvtx_generation_ranges
        self.started_at = time.time()
        self.metadata: dict[str, Any] = {}
        self.stages: list[dict[str, Any]] = []
        self.generation: list[dict[str, Any]] = []
        self.torch_profiles: list[dict[str, Any]] = []
        self._pending_torch_generation_traces: list[dict[str, Any]] = []
        self._torch_generation_count = 0

    @classmethod
    def from_args(cls, args) -> "InferenceProfiler":
        return cls(
            enabled=bool(getattr(args, "profile_inference", False)),
            output_path=getattr(args, "profile_output_path", None),
            sync_cuda=bool(getattr(args, "profile_sync_cuda", True)),
            torch_generation=bool(getattr(args, "profile_torch_generation", False)),
            torch_output_dir=getattr(args, "profile_torch_output_dir", None),
            torch_generation_limit=int(getattr(args, "profile_torch_generation_limit", 3)),
            torch_generation_label_filter=getattr(args, "profile_torch_generation_label_filter", None),
            torch_record_shapes=bool(getattr(args, "profile_torch_record_shapes", True)),
            torch_profile_memory=bool(getattr(args, "profile_torch_profile_memory", True)),
            torch_with_stack=bool(getattr(args, "profile_torch_with_stack", False)),
            torch_event_limit=int(getattr(args, "profile_torch_event_limit", 50)),
            nvtx_generation_ranges=bool(getattr(args, "profile_nvtx_generation_ranges", False)),
        )

    def set_metadata(self, **metadata: Any) -> None:
        if not self.enabled:
            return
        self.metadata.update(metadata)

    @contextmanager
    def stage(self, name: str, **metadata: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        self._sync_cuda()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync_cuda()
            record = {
                "name": name,
                "wall_seconds": time.perf_counter() - start,
            }
            record.update(metadata)
            self._add_cuda_memory(record)
            self.stages.append(self._to_jsonable(record))

    def record_generation(self, **record: Any) -> None:
        if not self.enabled:
            return
        self._add_cuda_memory(record)
        self.generation.append(self._to_jsonable(record))

    def pop_torch_generation_trace(self) -> dict[str, Any] | None:
        if not self._pending_torch_generation_traces:
            return None
        return self._pending_torch_generation_traces.pop(0)

    @contextmanager
    def torch_generation_trace(self, name: str, **metadata: Any) -> Iterator[None]:
        if not self._should_trace_torch_generation(name):
            if self._should_emit_nvtx_generation(name):
                with self._nvtx_range(name):
                    yield
            else:
                yield
            return

        trace_index = self._torch_generation_count
        self._torch_generation_count += 1
        output_dir = self._torch_trace_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        label = self._safe_label(name)
        trace_path = output_dir / f"{trace_index:03d}_{label}.trace.json"

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        profile_kwargs = {
            "activities": activities,
            "record_shapes": self.torch_record_shapes,
            "profile_memory": self.torch_profile_memory,
            "with_stack": self.torch_with_stack,
        }
        try:
            profiler_context = profile(**profile_kwargs, acc_events=True)
        except TypeError:
            profiler_context = profile(**profile_kwargs)

        self._sync_cuda()
        wall_start = time.perf_counter()
        error: str | None = None
        try:
            with profiler_context as prof:
                with self._nvtx_range(name):
                    yield
                self._sync_cuda()
        except Exception as exc:
            error = repr(exc)
            raise
        finally:
            wall_seconds = time.perf_counter() - wall_start
            record = {
                "name": name,
                "trace_index": trace_index,
                "trace_path": trace_path,
                "wall_seconds": wall_seconds,
                "error": error,
            }
            record.update(metadata)
            if "prof" in locals():
                prof.export_chrome_trace(str(trace_path))
                record["events"] = self._torch_profile_events(prof, limit=self.torch_event_limit)
            self._add_cuda_memory(record)
            record = self._to_jsonable(record)
            self.torch_profiles.append(record)
            self._pending_torch_generation_traces.append(record)

    def sync(self) -> None:
        if self.enabled:
            self._sync_cuda()

    def default_output_path(self, result_path: str | Path) -> Path:
        result_path = Path(result_path)
        return result_path.with_suffix(result_path.suffix + ".profile.json")

    def write(self, path: str | Path | None = None) -> Path | None:
        if not self.enabled:
            return None

        output_path = Path(path or self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "started_at_unix": self.started_at,
            "metadata": self._to_jsonable(self.metadata),
            "summary": self._summary(),
            "stages": self.stages,
            "generation": self.generation,
            "torch_profiles": self.torch_profiles,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path

    def _sync_cuda(self) -> None:
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _should_trace_torch_generation(self, name: str) -> bool:
        if self.torch_generation_label_filter and self.torch_generation_label_filter not in name:
            return False
        return (
            self.enabled
            and self.torch_generation
            and self._torch_generation_count < self.torch_generation_limit
        )

    def _should_emit_nvtx_generation(self, name: str) -> bool:
        if self.torch_generation_label_filter and self.torch_generation_label_filter not in name:
            return False
        return self.enabled and self.nvtx_generation_ranges

    def _torch_trace_output_dir(self) -> Path:
        if self.torch_output_dir:
            return Path(self.torch_output_dir)
        if self.output_path:
            output_path = Path(self.output_path)
            base = output_path.parent if output_path.suffix else output_path
            return base / "torch_profiles"
        return Path("torch_profiles")

    @contextmanager
    def _nvtx_range(self, name: str) -> Iterator[None]:
        pushed = False
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(name)
            pushed = True
        with record_function(name):
            try:
                yield
            finally:
                if pushed:
                    torch.cuda.nvtx.range_pop()

    @classmethod
    def _torch_profile_events(cls, prof, *, limit: int = 50) -> list[dict[str, Any]]:
        key_averages = prof.key_averages(group_by_input_shape=True)
        events: list[dict[str, Any]] = []
        for event in key_averages:
            cuda_total = cls._event_time(event, "cuda_time_total", "device_time_total")
            self_cuda_total = cls._event_time(event, "self_cuda_time_total", "self_device_time_total")
            cpu_total = cls._event_time(event, "cpu_time_total")
            self_cpu_total = cls._event_time(event, "self_cpu_time_total")
            if cuda_total <= 0 and self_cuda_total <= 0 and cpu_total <= 0:
                continue
            events.append({
                "key": event.key,
                "count": int(getattr(event, "count", 0)),
                "cuda_time_total_us": cuda_total,
                "self_cuda_time_total_us": self_cuda_total,
                "cpu_time_total_us": cpu_total,
                "self_cpu_time_total_us": self_cpu_total,
                "input_shapes": getattr(event, "input_shapes", None),
            })
        events.sort(
            key=lambda item: max(
                float(item["self_cuda_time_total_us"]),
                float(item["cuda_time_total_us"]),
                float(item["self_cpu_time_total_us"]),
            ),
            reverse=True,
        )
        return events if limit <= 0 else events[:limit]

    @staticmethod
    def _event_time(event: object, *names: str) -> float:
        for name in names:
            if hasattr(event, name):
                value = getattr(event, name)
                if value is not None:
                    return float(value)
        return 0.0

    @staticmethod
    def _add_cuda_memory(record: dict[str, Any]) -> None:
        if torch.cuda.is_available():
            record["cuda_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            record["cuda_max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024

    @staticmethod
    def _safe_label(label: str) -> str:
        return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in label)[:160]

    def _summary(self) -> dict[str, Any]:
        generation_by_label: dict[str, dict[str, Any]] = {}
        generation_by_context: dict[str, dict[str, Any]] = {}

        for record in self.generation:
            label = str(record.get("profile_label", "unknown"))
            context_type = str(record.get("context_type", "unknown"))
            self._add_generation_summary(generation_by_label, label, record)
            self._add_generation_summary(generation_by_context, context_type, record)

        return {
            "stage_wall_seconds": self._sum_stages(),
            "generation_by_label": generation_by_label,
            "generation_by_context": generation_by_context,
        }

    def _sum_stages(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for record in self.stages:
            name = str(record.get("name", "unknown"))
            totals[name] = totals.get(name, 0.0) + float(record.get("wall_seconds", 0.0) or 0.0)
        return totals

    @staticmethod
    def _add_generation_summary(target: dict[str, dict[str, Any]], key: str, record: dict[str, Any]) -> None:
        summary = target.setdefault(key, {
            "records": 0,
            "wall_seconds": 0.0,
            "model_elapsed_seconds": 0.0,
            "generated_tokens": 0,
            "model_generate_cpu_elapsed_seconds": 0.0,
            "model_generate_cuda_event_seconds": 0.0,
            "model_generate_host_gap_seconds": 0.0,
            "model_generate_records_with_cuda_event": 0,
        })
        summary["records"] += 1
        summary["wall_seconds"] += float(record.get("wall_seconds", 0.0) or 0.0)
        summary["model_elapsed_seconds"] += float(record.get("model_elapsed_seconds", 0.0) or 0.0)
        summary["generated_tokens"] += int(record.get("generated_tokens", 0) or 0)
        if isinstance(record.get("model_generate_cpu_elapsed_seconds"), (int, float)):
            summary["model_generate_cpu_elapsed_seconds"] += float(record["model_generate_cpu_elapsed_seconds"])
        if isinstance(record.get("model_generate_cuda_event_seconds"), (int, float)):
            summary["model_generate_cuda_event_seconds"] += float(record["model_generate_cuda_event_seconds"])
            summary["model_generate_records_with_cuda_event"] += 1
        if isinstance(record.get("model_generate_host_gap_seconds"), (int, float)):
            summary["model_generate_host_gap_seconds"] += float(record["model_generate_host_gap_seconds"])
        model_elapsed = summary["model_elapsed_seconds"]
        summary["tokens_per_second"] = summary["generated_tokens"] / model_elapsed if model_elapsed > 0 else 0.0
        summary["model_generate_cuda_event_fraction"] = (
            summary["model_generate_cuda_event_seconds"] / model_elapsed
            if model_elapsed > 0 and summary["model_generate_records_with_cuda_event"] > 0
            else None
        )
        summary["model_generate_host_gap_fraction"] = (
            summary["model_generate_host_gap_seconds"] / model_elapsed
            if model_elapsed > 0 and summary["model_generate_records_with_cuda_event"] > 0
            else None
        )

    @classmethod
    def _to_jsonable(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(cls._to_jsonable(k)): cls._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_jsonable(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "value"):
            return cls._to_jsonable(value.value)
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return value
