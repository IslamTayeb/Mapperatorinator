from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

from ..runtime_profiling import detail_ranges_context, profile_range


PROFILE_PASS_KINDS = frozenset(
    {
        "unspecified",
        "untraced_control",
        "untraced_budget",
        "nsys_graph",
        "nsys_node",
        "ncu_kernel",
    }
)


class InferenceProfiler:
    """Small synchronized JSON profiler enabled by ``profile_inference``.

    Diagnostic traces and backend controls deliberately live outside the runtime
    profile. A profile run always uses synchronized wall timing, records token
    identity through the processor, and writes one stable JSON shape.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        detail_ranges: bool = False,
        cuda_capture: bool = False,
        pass_kind: str = "unspecified",
    ):
        if pass_kind not in PROFILE_PASS_KINDS:
            valid = ", ".join(sorted(PROFILE_PASS_KINDS))
            raise ValueError(f"profile_pass_kind must be one of: {valid}")
        if detail_ranges and not enabled:
            raise ValueError("profile_detail_ranges requires profile_inference=true")
        if cuda_capture and not detail_ranges:
            raise ValueError(
                "profile_cuda_capture requires profile_detail_ranges=true"
            )
        self.enabled = enabled
        self.detail_ranges = detail_ranges
        self.cuda_capture = cuda_capture
        self.pass_kind = pass_kind
        self._generation_roi_active = False
        self.started_at = time.time()
        self.metadata: dict[str, Any] = {
            "profile_pass_kind": pass_kind,
            "authoritative_performance": pass_kind == "untraced_control",
            "untraced_budget_enabled": pass_kind == "untraced_budget",
            "profile_detail_ranges": detail_ranges,
            "profile_cuda_capture": cuda_capture,
        } if enabled else {}
        self.stages: list[dict[str, Any]] = []
        self.generation: list[dict[str, Any]] = []

    @classmethod
    def from_args(cls, args) -> "InferenceProfiler":
        return cls(
            enabled=bool(getattr(args, "profile_inference", False)),
            detail_ranges=bool(getattr(args, "profile_detail_ranges", False)),
            cuda_capture=bool(getattr(args, "profile_cuda_capture", False)),
            pass_kind=str(getattr(args, "profile_pass_kind", "unspecified")),
        )

    def start_generation_capture(self) -> None:
        if not self.detail_ranges:
            return
        if self._generation_roi_active:
            raise RuntimeError("generation profiler capture is already active")
        if self.cuda_capture and not torch.cuda.is_available():
            raise RuntimeError("profile_cuda_capture requires CUDA")
        if torch.cuda.is_available():
            self._sync_cuda()
            if self.cuda_capture:
                torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push("mapperatorinator.roi.inference_generation")
        self._generation_roi_active = True

    def stop_generation_capture(self) -> None:
        if not self.detail_ranges:
            return
        if not self._generation_roi_active:
            raise RuntimeError("generation profiler capture is not active")
        if torch.cuda.is_available():
            self._sync_cuda()
            torch.cuda.nvtx.range_pop()
            if self.cuda_capture:
                torch.cuda.cudart().cudaProfilerStop()
        self._generation_roi_active = False

    def set_metadata(self, **metadata: Any) -> None:
        if self.enabled:
            self.metadata.update(metadata)

    @contextmanager
    def stage(self, name: str, **metadata: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        with detail_ranges_context(self.detail_ranges):
            self._sync_cuda()
            started = time.perf_counter()
            try:
                with profile_range(f"stage.{name}"):
                    yield
            finally:
                self._sync_cuda()
                finished = time.perf_counter()
                record = {
                    "name": name,
                    "wall_seconds": finished - started,
                    "started_at_perf_counter_seconds": started,
                    "finished_at_perf_counter_seconds": finished,
                    **metadata,
                }
                self._add_cuda_memory(record)
                self.stages.append(self._to_jsonable(record))

    def record_generation(self, **record: Any) -> None:
        if self.enabled:
            self._add_cuda_memory(record)
            self.generation.append(self._to_jsonable(record))

    def sync(self) -> None:
        if self.enabled:
            self._sync_cuda()

    @staticmethod
    def default_output_path(result_path: str | Path) -> Path:
        result_path = Path(result_path)
        return result_path.with_suffix(result_path.suffix + ".profile.json")

    def write(self, path: str | Path) -> Path | None:
        if not self.enabled:
            return None

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "started_at_unix": self.started_at,
            "metadata": self._to_jsonable(self.metadata),
            "summary": self._summary(),
            "stages": self.stages,
            "generation": self.generation,
        }
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return output_path

    @staticmethod
    def _sync_cuda() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _add_cuda_memory(record: dict[str, Any]) -> None:
        if torch.cuda.is_available():
            record["cuda_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            record["cuda_max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024

    def _summary(self) -> dict[str, Any]:
        by_label: dict[str, dict[str, Any]] = {}
        by_context: dict[str, dict[str, Any]] = {}
        for record in self.generation:
            self._add_generation_summary(by_label, str(record.get("profile_label", "unknown")), record)
            self._add_generation_summary(by_context, str(record.get("context_type", "unknown")), record)
        return {
            "stage_wall_seconds": self._sum_stages(),
            "generation_by_label": by_label,
            "generation_by_context": by_context,
        }

    def _sum_stages(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for record in self.stages:
            name = str(record.get("name", "unknown"))
            totals[name] = totals.get(name, 0.0) + float(record.get("wall_seconds", 0.0) or 0.0)
        return totals

    @staticmethod
    def _add_generation_summary(
        target: dict[str, dict[str, Any]],
        key: str,
        record: dict[str, Any],
    ) -> None:
        summary = target.setdefault(
            key,
            {"records": 0, "wall_seconds": 0.0, "model_elapsed_seconds": 0.0, "generated_tokens": 0},
        )
        summary["records"] += 1
        summary["wall_seconds"] += float(record.get("wall_seconds", 0.0) or 0.0)
        summary["model_elapsed_seconds"] += float(record.get("model_elapsed_seconds", 0.0) or 0.0)
        summary["generated_tokens"] += int(record.get("generated_tokens", 0) or 0)
        model_seconds = summary["model_elapsed_seconds"]
        summary["tokens_per_second"] = summary["generated_tokens"] / model_seconds if model_seconds > 0 else 0.0

    @classmethod
    def _to_jsonable(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(cls._to_jsonable(key)): cls._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_jsonable(item) for item in value]
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
