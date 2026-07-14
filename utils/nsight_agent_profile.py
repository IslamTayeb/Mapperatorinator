"""Build a deterministic, agent-friendly summary from native Nsight exports.

The DCC wrapper owns collection.  This module deliberately consumes only an
explicit manifest plus native CSV/SQLite artifacts; it never guesses a GUI
layout and never treats traced throughput as authoritative.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "mapperatorinator.nsight-summary.v1"
MANIFEST_SCHEMA_VERSION = "mapperatorinator.nsight-manifest.v1"
STAGES = ("timing_generation", "main_generation")
DECODER_REGIONS = (
    "self_norm_qkv",
    "fused_self_attention",
    "self_out_residual",
    "cross_norm_q",
    "q1_bmm_cross",
    "cross_out_residual",
    "mlp",
    "final_norm_logits",
    "unattributed",
)
TRACED_PASS_KINDS = {"nsys_graph", "nsys_node", "ncu_kernel"}
PASS_KINDS = TRACED_PASS_KINDS | {"untraced_control", "ncu_permission_probe"}
STAGE_REPORT_ROLES = frozenset(
    {
        "cuda_kernel_summary",
        "cuda_kernel_trace",
        "cuda_api_summary",
        "cuda_memory_time_summary",
        "cuda_memory_size_summary",
        "nvtx_summary",
        "cuda_api_sync",
        "gpu_gaps",
        "gpu_time_util",
    }
)
ANALYSIS_ROLES = ("cuda_api_sync", "gpu_gaps", "gpu_time_util")
STAGE_NVTX_RANGE_NAMES = {
    "timing_generation": "mapperatorinator.stage.timing_context_generation",
    "main_generation": "mapperatorinator.stage.main_generation",
}
PROFILE_LABELS = ("timing_context", "main_generation")
TIMING_POSTPROCESS_STAGES = (
    "timing_context_postprocess",
    "super_timing_postprocess",
)
MAIN_POSTPROCESS_STAGES = (
    "merge_generated_events",
    "derive_timing_from_generated_events",
    "resnap_events",
    "postprocess_generate_osu",
    "merge_with_reference_beatmap",
    "write_osu",
    "write_osz",
)
PROFILE_WORKLOAD_KEYS = (
    "model_path",
    "audio_path",
    "beatmap_path",
    "seed",
    "precision",
    "attn_implementation",
    "inference_engine",
    "use_server",
    "parallel",
    "max_batch_size",
    "temperature",
    "timing_temperature",
    "mania_column_temperature",
    "taiko_hit_temperature",
    "timeshift_bias",
    "top_p",
    "top_k",
    "do_sample",
    "num_beams",
    "cfg_scale",
    "lookback",
    "lookahead",
    "start_time",
    "end_time",
    "in_context",
    "output_type",
    "sequence_count",
    "song_length_ms",
)
PROFILE_PRESET_KEYS = (
    "optimized_effective_config_version",
    "optimized_effective_config",
    "optimized_runtime_owner",
    "optimized_result_class",
)
PROFILE_GRAPH_CACHE_RECORD_KEYS = (
    "decoder_loop_backend",
    "torch_compile_enabled",
    "generation_compile_enabled",
    "optimized_effective_config_version",
    "optimized_dispatch_capture_hits",
    "optimized_cuda_graphs",
    "native_cross_mlp_tail_requested",
    "native_cross_mlp_tail_enabled",
    "native_cross_mlp_tail_disabled_reason",
    "graph_cache_entries",
    "graph_capture_count",
    "graph_replay_count",
    "decode_replays",
    "cache_summary",
    "graph_cache_summary",
    "cuda_graph_summary",
)


class NsightProfileError(RuntimeError):
    """Base class for fail-loud analyzer errors."""


class ArtifactError(NsightProfileError):
    pass


class ColumnError(NsightProfileError):
    pass


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_osu_structure(path: Path) -> dict[str, Any]:
    """Return deterministic final-map structure without requiring an osu parser."""

    section = ""
    timing_points = 0
    inherited_timing_points = 0
    uninherited_timing_points = 0
    hit_objects = 0
    malformed_by_section = {"TimingPoints": 0, "HitObjects": 0}
    nonfinite_values = 0
    hit_object_types = {
        "circles": 0,
        "sliders": 0,
        "spinners": 0,
        "holds": 0,
        "unknown": 0,
    }

    def parse_number(field: str) -> tuple[bool, float | None]:
        nonlocal nonfinite_values
        try:
            value = float(field)
        except ValueError:
            return False, None
        if not math.isfinite(value):
            nonfinite_values += 1
            return False, value
        return True, value

    def parse_integer(field: str) -> tuple[bool, int | None]:
        valid, value = parse_number(field)
        if not valid or value is None or not value.is_integer():
            return False, None
        return True, int(value)

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section == "TimingPoints":
            timing_points += 1
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 8:
                malformed_by_section[section] += 1
                continue
            numeric_valid = all(parse_number(field)[0] for field in fields[:2])
            integer_values = [parse_integer(field) for field in fields[2:]]
            numeric_valid &= all(valid for valid, _ in integer_values)
            uninherited = integer_values[4][1]
            if not numeric_valid or uninherited not in {0, 1}:
                malformed_by_section[section] += 1
                continue
            if uninherited == 1:
                uninherited_timing_points += 1
            else:
                inherited_timing_points += 1
        elif section == "HitObjects":
            hit_objects += 1
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 6:
                malformed_by_section[section] += 1
                continue
            core_values = [parse_integer(field) for field in fields[:5]]
            if not all(valid for valid, _ in core_values):
                malformed_by_section[section] += 1
                continue
            object_type = core_values[3][1]
            if object_type is None:
                malformed_by_section[section] += 1
                continue
            base_type = object_type & (1 | 2 | 8 | 128)
            if base_type not in {1, 2, 8, 128}:
                malformed_by_section[section] += 1
                continue
            type_specific_valid = True
            if base_type == 2:
                repeat_valid, repeat_count = (
                    parse_integer(fields[6]) if len(fields) >= 7 else (False, None)
                )
                length_valid, pixel_length = (
                    parse_number(fields[7]) if len(fields) >= 8 else (False, None)
                )
                type_specific_valid = (
                    len(fields) >= 8
                    and bool(fields[5])
                    and repeat_valid
                    and repeat_count is not None
                    and repeat_count >= 1
                    and length_valid
                    and pixel_length is not None
                    and pixel_length >= 0
                )
            elif base_type == 8:
                end_valid, end_time = (
                    parse_number(fields[5]) if len(fields) >= 6 else (False, None)
                )
                type_specific_valid = (
                    len(fields) >= 7
                    and end_valid
                    and end_time is not None
                    and core_values[2][1] is not None
                    and end_time >= core_values[2][1]
                )
            elif base_type == 128:
                end_time = fields[5].split(":", 1)[0]
                end_valid, parsed_end_time = (
                    parse_number(end_time) if end_time else (False, None)
                )
                type_specific_valid = (
                    end_valid
                    and parsed_end_time is not None
                    and core_values[2][1] is not None
                    and parsed_end_time >= core_values[2][1]
                )
            if not type_specific_valid:
                malformed_by_section[section] += 1
                continue
            if base_type == 128:
                hit_object_types["holds"] += 1
            elif base_type == 8:
                hit_object_types["spinners"] += 1
            elif base_type == 2:
                hit_object_types["sliders"] += 1
            elif base_type == 1:
                hit_object_types["circles"] += 1
            else:
                hit_object_types["unknown"] += 1

    malformed_lines = sum(malformed_by_section.values())
    return {
        "timing_points": timing_points,
        "uninherited_timing_points": uninherited_timing_points,
        "inherited_timing_points": inherited_timing_points,
        "hit_objects": hit_objects,
        "hit_object_types": hit_object_types,
        "malformed_lines": malformed_lines,
        "malformed_by_section": malformed_by_section,
        "nonfinite_values": nonfinite_values,
        "numeric_validation_scope": (
            "timing_all_fields_and_hitobject_core_plus_type_specific_fields"
        ),
        "finite_and_well_formed": malformed_lines == 0 and nonfinite_values == 0,
    }


def _finite_number(value: Any, *, field: str) -> float:
    try:
        parsed = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} must be finite and non-negative: {value!r}")
    return parsed


def _normalize_header(value: str) -> str:
    value = value.strip().lstrip("\ufeff").lower()
    value = value.replace("μ", "u").replace("µ", "u")
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def _unit_multiplier_ns(header: str) -> int:
    normalized = _normalize_header(header)
    if normalized.endswith("_ns") or "nanosecond" in normalized:
        return 1
    if normalized.endswith("_us") or "microsecond" in normalized:
        return 1_000
    if normalized.endswith("_ms") or "millisecond" in normalized:
        return 1_000_000
    if normalized.endswith("_s") or normalized.endswith("_sec") or "second" in normalized:
        return 1_000_000_000
    # Native nsys reports use ns for an unqualified Start/Duration in several
    # versions.  Preserve that documented export convention in one place.
    return 1


def _duration_to_ns(value: Any, header: str, *, field: str) -> int:
    return int(round(_finite_number(value, field=field) * _unit_multiplier_ns(header)))


def _byte_multiplier(header: str) -> int:
    match = re.search(r"\((gib|mib|kib|gb|mb|kb|bytes?|b)\)", header, flags=re.IGNORECASE)
    if not match:
        if _normalize_header(header).endswith("_bytes"):
            return 1
        raise ColumnError(f"byte column has no recognized unit: {header!r}")
    unit = match.group(1).lower()
    return {
        "b": 1,
        "byte": 1,
        "bytes": 1,
        "kb": 1_000,
        "mb": 1_000_000,
        "gb": 1_000_000_000,
        "kib": 1 << 10,
        "mib": 1 << 20,
        "gib": 1 << 30,
    }[unit]


def _size_to_bytes(value: Any, header: str, *, field: str) -> int:
    return int(round(_finite_number(value, field=field) * _byte_multiplier(header)))


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read an nsys CSV, tolerating report-title/preamble rows."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ArtifactError(f"empty CSV report: {path}")

    header_index = None
    for index, row in enumerate(rows):
        normalized = {_normalize_header(cell) for cell in row}
        has_name = bool(normalized & {"name", "kernel_name", "operation", "range"})
        has_metric = any(
            token in column
            for column in normalized
            for token in (
                "duration",
                "total_time",
                "total",
                "start",
                "instances",
                "calls",
                "operations",
            )
        )
        if len(row) >= 2 and has_name and has_metric:
            header_index = index
            break
    if header_index is None:
        raise ColumnError(f"could not locate an nsys CSV header in {path}")

    headers = [cell.strip().lstrip("\ufeff") for cell in rows[header_index]]
    records: list[dict[str, str]] = []
    for row in rows[header_index + 1 :]:
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) < len(headers):
            row = row + [""] * (len(headers) - len(row))
        records.append(dict(zip(headers, row[: len(headers)])))
    return headers, records


def _read_structured_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read arbitrary native analysis CSV without inventing semantic columns."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ArtifactError(f"empty CSV report: {path}")

    header_index = None
    for index, row in enumerate(rows):
        normalized = [_normalize_header(cell) for cell in row]
        if (
            len(row) >= 2
            and all(normalized)
        ):
            header_index = index
            break
    if header_index is None:
        raise ColumnError(f"could not locate a structured CSV header in {path}")

    headers = [cell.strip().lstrip("\ufeff") for cell in rows[header_index]]
    records: list[dict[str, str]] = []
    for row in rows[header_index + 1 :]:
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) < len(headers):
            row = row + [""] * (len(headers) - len(row))
        records.append(dict(zip(headers, row[: len(headers)])))
    return headers, records


def _resolve_columns(
    headers: Sequence[str],
    aliases: Mapping[str, Sequence[str]],
    *,
    required: Iterable[str],
    field_units: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    field_units = field_units or {}
    normalized_to_headers: dict[str, list[str]] = defaultdict(list)
    for header in headers:
        normalized_to_headers[_normalize_header(header)].append(header)

    resolved: dict[str, str] = {}
    canonical: dict[str, Any] = {}
    for field, field_aliases in aliases.items():
        candidates: list[str] = []
        for alias in field_aliases:
            candidates.extend(normalized_to_headers.get(_normalize_header(alias), []))
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) > 1:
            raise ColumnError(
                f"ambiguous columns for {field}: {candidates}; observed={list(headers)}"
            )
        if candidates:
            source = candidates[0]
            resolved[field] = source
            unit_kind = field_units.get(
                field,
                "duration" if field.endswith("_ns") else "none",
            )
            canonical[field] = {
                "source_column": source,
                "source_unit": {"duration": "ns", "bytes": "bytes"}.get(unit_kind),
                "multiplier": (
                    _unit_multiplier_ns(source)
                    if unit_kind == "duration"
                    else _byte_multiplier(source)
                    if unit_kind == "bytes"
                    else None
                ),
            }

    required_set = set(required)
    missing_required = sorted(required_set - resolved.keys())
    missing_optional = sorted(set(aliases) - required_set - resolved.keys())
    used = set(resolved.values())
    provenance = {
        "observed_columns": list(headers),
        "canonical_fields": canonical,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "unknown_columns": [header for header in headers if header not in used],
    }
    if missing_required:
        raise ColumnError(
            f"missing required columns {missing_required}; observed={list(headers)}"
        )
    return resolved, provenance


_SUMMARY_ALIASES = {
    "name": ("Name", "Kernel Name"),
    "total_ns": ("Total Time (ns)", "Total Time (us)", "Total Time (ms)", "Total Time"),
    "calls": ("Instances", "Calls", "Count", "Launches"),
    "average_ns": ("Avg (ns)", "Average (ns)", "Avg (us)", "Average (us)", "Avg"),
    "minimum_ns": ("Min (ns)", "Minimum (ns)", "Min (us)", "Min"),
    "maximum_ns": ("Max (ns)", "Maximum (ns)", "Max (us)", "Max"),
}

_TRACE_ALIASES = {
    "name": ("Name", "Kernel Name"),
    "start_ns": ("Start (ns)", "Start (us)", "Start (ms)", "Start"),
    "duration_ns": ("Duration (ns)", "Duration (us)", "Duration (ms)", "Duration"),
    "grid": ("GridXYZ", "Grid XYZ", "Grid"),
    "block": ("BlockXYZ", "Block XYZ", "Block"),
    "correlation_id": ("CorrId", "Correlation ID", "CorrelationId"),
    "graph_node_id": ("GraphNodeId", "Graph Node ID"),
    "graph_id": ("GraphId", "Graph ID"),
}

_CUDA_API_ALIASES = {
    "name": ("Name", "API Name"),
    "total_ns": ("Total Time (ns)", "Total Time (us)", "Total Time (ms)", "Total Time"),
    "calls": ("Num Calls", "Calls", "Instances", "Count"),
    "average_ns": ("Avg (ns)", "Average (ns)", "Avg (us)", "Average (us)", "Avg"),
    "minimum_ns": ("Min (ns)", "Minimum (ns)", "Min (us)", "Min"),
    "maximum_ns": ("Max (ns)", "Maximum (ns)", "Max (us)", "Max"),
}

_MEMORY_TIME_ALIASES = {
    "operation": ("Operation", "Name"),
    "total_ns": ("Total Time (ns)", "Total Time (us)", "Total Time (ms)", "Total Time"),
    "calls": (
        "Num Calls",
        "Operations",
        "Num Operations",
        "Calls",
        "Instances",
        "Count",
    ),
    "average_ns": ("Avg (ns)", "Average (ns)", "Avg (us)", "Average (us)", "Avg"),
    "minimum_ns": ("Min (ns)", "Minimum (ns)", "Min (us)", "Min"),
    "maximum_ns": ("Max (ns)", "Maximum (ns)", "Max (us)", "Max"),
}

_MEMORY_SIZE_ALIASES = {
    "operation": ("Operation", "Name"),
    "calls": ("Operations", "Num Operations", "Calls", "Instances", "Count"),
    "total_bytes": (
        "Total (B)",
        "Total (KB)",
        "Total (KiB)",
        "Total (MB)",
        "Total (MiB)",
        "Total (GB)",
        "Total (GiB)",
        "Total Bytes",
    ),
    "average_bytes": (
        "Avg (B)", "Avg (KB)", "Avg (KiB)", "Avg (MB)", "Avg (MiB)",
        "Avg (GB)", "Avg (GiB)", "Avg",
    ),
    "minimum_bytes": (
        "Min (B)", "Min (KB)", "Min (KiB)", "Min (MB)", "Min (MiB)",
        "Min (GB)", "Min (GiB)", "Min",
    ),
    "maximum_bytes": (
        "Max (B)", "Max (KB)", "Max (KiB)", "Max (MB)", "Max (MiB)",
        "Max (GB)", "Max (GiB)", "Max",
    ),
}

_NVTX_ALIASES = {
    "name": ("Range", "Name", "NVTX Range"),
    "total_ns": ("Total Time (ns)", "Total Time (us)", "Total Time (ms)", "Total Time"),
    "calls": ("Instances", "Num Calls", "Calls", "Count"),
    "average_ns": ("Avg (ns)", "Average (ns)", "Avg (us)", "Average (us)", "Avg"),
    "minimum_ns": ("Min (ns)", "Minimum (ns)", "Min (us)", "Min"),
    "maximum_ns": ("Max (ns)", "Maximum (ns)", "Max (us)", "Max"),
}


def classify_kernel_family(name: str) -> tuple[str, str]:
    """Classify for roll-up while retaining the exact symbol as identity."""

    lowered = name.lower()
    rules = (
        (
            "native_q1_self_rope_cache",
            r"q1_rope_cache_attention|rope_cache.*q1|q1_attention_kernel|"
            r"native_q1_(?:rope_cache_)?attention|q1_self_attention|self.*q1",
        ),
        (
            "fused_fc1_gelu",
            r"fc1[_:]*gelu|gelu[_:]*fc1|one_token_fc1_gelu",
        ),
        (
            "fused_fc2_residual",
            r"fc2[_:]*residual|residual[_:]*fc2|one_token_fc2_residual",
        ),
        (
            "fmha_cross_attention",
            r"fmha|flash[_:]*attention|flash_attn|scaled_dot|sdpa|"
            r"cross[_:]*(?:attention|attn|bmm)|q1_bmm|bmm_cross",
        ),
        (
            "gemm_gemv_projection",
            r"gemm|gemv|matmul|(?:^|[^a-z0-9])bmm(?:[^a-z0-9]|$)|"
            r"cublas|cutlass|linear|projection|out_proj|proj_",
        ),
        (
            "sampling_radix_sort",
            r"softmax|multinomial|topk|top_k|radix[_:]*sort|radixsort|sort|"
            r"categorical|distribution|argmax|arg_min|curand|philox",
        ),
        (
            "memory",
            r"memcpy|memset|copy|catarray|gather|scatter|index_|fillfunctor",
        ),
        (
            "elementwise",
            r"gelu|silu|relu|rmsnorm|layernorm|layer_norm|reduce|"
            r"elementwise|vectorized|unrolled|masked_fill|clamp|where_kernel|compare",
        ),
    )
    for family, pattern in rules:
        if re.search(pattern, lowered):
            return family, pattern
    return "other", "fallback"


def _kernel_identity(name: str) -> dict[str, str]:
    family, rule = classify_kernel_family(name)
    return {
        "kernel_id": _sha256_bytes(name.encode("utf-8")),
        "raw_name": name,
        "demangled_name": name,
        "display_name": name,
        "family": family,
        "family_rule": rule,
    }


def parse_kernel_summary(path: Path) -> dict[str, Any]:
    headers, records = _read_csv(path)
    columns, provenance = _resolve_columns(
        headers, _SUMMARY_ALIASES, required=("name", "total_ns", "calls")
    )
    kernels: list[dict[str, Any]] = []
    for row in records:
        name = row[columns["name"]].strip()
        if not name:
            continue
        total_ns = _duration_to_ns(row[columns["total_ns"]], columns["total_ns"], field="total_ns")
        calls = int(round(_finite_number(row[columns["calls"]], field="calls")))
        entry: dict[str, Any] = {
            **_kernel_identity(name),
            "calls": calls,
            "total_ns": total_ns,
            "average_ns": int(round(total_ns / calls)) if calls else None,
            "minimum_ns": None,
            "maximum_ns": None,
            "launch_shapes": [],
        }
        for canonical in ("average_ns", "minimum_ns", "maximum_ns"):
            if canonical in columns and row[columns[canonical]].strip():
                entry[canonical] = _duration_to_ns(
                    row[columns[canonical]], columns[canonical], field=canonical
                )
        kernels.append(entry)
    if not kernels:
        raise ArtifactError(f"kernel summary contains no kernel rows: {path}")
    kernels.sort(key=lambda item: (-item["total_ns"], item["raw_name"]))
    return {
        "status": "available",
        "source_path": str(path),
        "column_resolution": provenance,
        "kernels": kernels,
        "accumulated_kernel_ns": sum(item["total_ns"] for item in kernels),
    }


def _count(value: Any, *, field: str) -> int:
    parsed = _finite_number(value, field=field)
    if not parsed.is_integer():
        raise ValueError(f"{field} must be an integer count: {value!r}")
    return int(parsed)


def _parse_timing_rows(
    path: Path,
    aliases: Mapping[str, Sequence[str]],
    *,
    name_field: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    headers, records = _read_csv(path)
    columns, provenance = _resolve_columns(
        headers,
        aliases,
        required=(name_field, "total_ns", "calls", "average_ns", "minimum_ns", "maximum_ns"),
    )
    entries: list[dict[str, Any]] = []
    for row_index, row in enumerate(records):
        name = row[columns[name_field]].strip()
        if not name:
            continue
        calls = _count(row[columns["calls"]], field=f"row[{row_index}].calls")
        entry = {
            name_field: name,
            "calls": calls,
            "total_ns": _duration_to_ns(
                row[columns["total_ns"]],
                columns["total_ns"],
                field=f"row[{row_index}].total_ns",
            ),
            "average_ns": _duration_to_ns(
                row[columns["average_ns"]],
                columns["average_ns"],
                field=f"row[{row_index}].average_ns",
            ),
            "minimum_ns": _duration_to_ns(
                row[columns["minimum_ns"]],
                columns["minimum_ns"],
                field=f"row[{row_index}].minimum_ns",
            ),
            "maximum_ns": _duration_to_ns(
                row[columns["maximum_ns"]],
                columns["maximum_ns"],
                field=f"row[{row_index}].maximum_ns",
            ),
        }
        entries.append(entry)
    if not entries:
        raise ArtifactError(f"timing summary contains no data rows: {path}")
    entries.sort(key=lambda item: (-item["total_ns"], item[name_field]))
    return entries, provenance


def _is_cuda_synchronization_call(name: str) -> bool:
    base = name.split("(", 1)[0].strip()
    return bool(
        re.search(r"(?:Synchronize|WaitExternalSemaphores(?:Async)?)$", base)
    )


def parse_cuda_api_summary(path: Path) -> dict[str, Any]:
    apis, provenance = _parse_timing_rows(path, _CUDA_API_ALIASES, name_field="name")
    for api in apis:
        api["is_synchronization_call"] = _is_cuda_synchronization_call(api["name"])
    synchronization_apis = [api for api in apis if api["is_synchronization_call"]]
    return {
        "status": "available",
        "source_path": str(path),
        "column_resolution": provenance,
        "apis": apis,
        "total_calls": sum(api["calls"] for api in apis),
        "accumulated_api_ns": sum(api["total_ns"] for api in apis),
        "synchronization": {
            "api_names": [api["name"] for api in synchronization_apis],
            "calls": sum(api["calls"] for api in synchronization_apis),
            "total_ns": sum(api["total_ns"] for api in synchronization_apis),
        },
    }


def parse_cuda_memory_time_summary(path: Path) -> dict[str, Any]:
    operations, provenance = _parse_timing_rows(
        path,
        _MEMORY_TIME_ALIASES,
        name_field="operation",
    )
    return {
        "status": "available",
        "source_path": str(path),
        "column_resolution": provenance,
        "operations": operations,
        "total_calls": sum(operation["calls"] for operation in operations),
        "accumulated_memory_operation_ns": sum(
            operation["total_ns"] for operation in operations
        ),
    }


def parse_cuda_memory_size_summary(path: Path) -> dict[str, Any]:
    headers, records = _read_csv(path)
    columns, provenance = _resolve_columns(
        headers,
        _MEMORY_SIZE_ALIASES,
        required=(
            "operation",
            "calls",
            "total_bytes",
            "average_bytes",
            "minimum_bytes",
            "maximum_bytes",
        ),
        field_units={
            "total_bytes": "bytes",
            "average_bytes": "bytes",
            "minimum_bytes": "bytes",
            "maximum_bytes": "bytes",
        },
    )
    operations: list[dict[str, Any]] = []
    for row_index, row in enumerate(records):
        operation = row[columns["operation"]].strip()
        if not operation:
            continue
        entry: dict[str, Any] = {
            "operation": operation,
            "calls": _count(row[columns["calls"]], field=f"row[{row_index}].calls"),
        }
        for field in ("total_bytes", "average_bytes", "minimum_bytes", "maximum_bytes"):
            entry[field] = _size_to_bytes(
                row[columns[field]],
                columns[field],
                field=f"row[{row_index}].{field}",
            )
        operations.append(entry)
    if not operations:
        raise ArtifactError(f"memory size summary contains no operation rows: {path}")
    operations.sort(key=lambda item: (-item["total_bytes"], item["operation"]))
    return {
        "status": "available",
        "source_path": str(path),
        "column_resolution": provenance,
        "operations": operations,
        "total_calls": sum(operation["calls"] for operation in operations),
        "total_bytes": sum(operation["total_bytes"] for operation in operations),
    }


def parse_nvtx_summary(path: Path) -> dict[str, Any]:
    ranges, provenance = _parse_timing_rows(path, _NVTX_ALIASES, name_field="name")
    return {
        "status": "available",
        "source_path": str(path),
        "column_resolution": provenance,
        "ranges": ranges,
        "total_calls": sum(entry["calls"] for entry in ranges),
        "accumulated_range_ns": sum(entry["total_ns"] for entry in ranges),
    }


def _semantic_nvtx_views(report: Mapping[str, Any]) -> dict[str, Any]:
    """Expose semantic CPU ranges without claiming CUDA-graph replay attribution."""

    if report.get("status") != "available":
        return {
            "status": "not_available",
            "metric_kind": "nvtx_cpu_range_elapsed",
            "replay_attribution": False,
            "generation_regions": {},
            "decoder_regions": {region: [] for region in DECODER_REGIONS[:-1]},
        }

    generation_names = {
        "encoder_prefill": "encoder_prefill",
        "decode_graph_capture_setup": "decode_graph_capture_setup",
        "decode_graph_replay": "decode_graph_replay",
        "cache_update": "cache_update",
        "logits_processors": "logits_processors",
        "sampling": "sampling",
        "token_append_and_stopping": "token_append_cache_stopping",
        "final_device_to_host": "final_device_to_host",
    }
    decoder_patterns = (
        ("self_norm_qkv", r"\.self_attn_norm$|\.self\.qkv_proj$"),
        ("fused_self_attention", r"\.self_attn$|\.self\.rope_cache_native_q1$"),
        (
            "self_out_residual",
            r"\.self\.out_proj$|\.self\.residual$|\.self_attn_out_residual$",
        ),
        ("cross_norm_q", r"\.cross_attn_norm$|\.cross\.q_proj$|\.cross_q$"),
        ("q1_bmm_cross", r"\.cross\.sdpa$|\.cross_bmm$"),
        (
            "cross_out_residual",
            r"\.cross\.out_proj$|\.cross\.residual$|\.cross_attn_out_residual$|\.cross_out$",
        ),
        ("mlp", r"\.mlp(?:\.|$)|\.mlp_norm$|\.native_cross_mlp_tail\.mlp$"),
        ("final_norm_logits", r"decoder\.final_norm$|decoder\.output_projection$"),
    )
    generation_regions: dict[str, dict[str, Any]] = {}
    decoder_regions: dict[str, list[dict[str, Any]]] = {
        region: [] for region in DECODER_REGIONS[:-1]
    }
    for entry in report["ranges"]:
        name = str(entry["name"])
        generation_prefix = "mapperatorinator.generation."
        if name.startswith(generation_prefix):
            suffix = name.removeprefix(generation_prefix)
            canonical = generation_names.get(suffix)
            if canonical is not None:
                generation_regions[canonical] = {
                    "range_name": name,
                    "calls": int(entry["calls"]),
                    "total_ns": int(entry["total_ns"]),
                    "average_ns": int(entry["average_ns"]),
                    "minimum_ns": int(entry["minimum_ns"]),
                    "maximum_ns": int(entry["maximum_ns"]),
                }
        for region, pattern in decoder_patterns:
            if re.search(pattern, name):
                decoder_regions[region].append(
                    {
                        "range_name": name,
                        "calls": int(entry["calls"]),
                        "total_ns": int(entry["total_ns"]),
                    }
                )
                break

    for entries in decoder_regions.values():
        entries.sort(key=lambda item: (-item["total_ns"], item["range_name"]))
    return {
        "status": "available",
        "metric_kind": "nvtx_cpu_range_elapsed",
        "authoritative_performance": False,
        "replay_attribution": False,
        "nested_ranges_may_overlap": True,
        "generation_regions": generation_regions,
        "decoder_regions": decoder_regions,
    }


def parse_analysis_csv(path: Path, *, analysis_kind: str) -> dict[str, Any]:
    if analysis_kind not in ANALYSIS_ROLES:
        raise ValueError(f"unsupported analysis kind: {analysis_kind}")
    headers, records = _read_structured_csv(path)
    normalized = [_normalize_header(header) for header in headers]
    if len(set(normalized)) != len(normalized):
        raise ColumnError(
            f"analysis CSV has colliding normalized columns: {headers}"
        )
    if not records:
        raise ArtifactError(f"analysis CSV contains no rows: {path}")
    column_provenance = [
        {
            "index": index,
            "source_column": header,
            "normalized_field": normalized[index],
        }
        for index, header in enumerate(headers)
    ]
    structured_rows = [
        {
            "source_row_index": row_index,
            "values": {
                normalized[index]: row[header]
                for index, header in enumerate(headers)
            },
        }
        for row_index, row in enumerate(records, start=1)
    ]
    return {
        "status": "available",
        "analysis_kind": analysis_kind,
        "source_path": str(path),
        "column_provenance": column_provenance,
        "row_count": len(structured_rows),
        "rows": structured_rows,
    }


def interval_union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    valid = sorted((start, end) for start, end in intervals if end > start)
    if not valid:
        return 0
    total = 0
    current_start, current_end = valid[0]
    for start, end in valid[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def parse_kernel_trace(path: Path) -> dict[str, Any]:
    headers, records = _read_csv(path)
    columns, provenance = _resolve_columns(
        headers, _TRACE_ALIASES, required=("name", "start_ns", "duration_ns")
    )
    launches: list[dict[str, Any]] = []
    for row in records:
        name = row[columns["name"]].strip()
        if not name:
            continue
        start_ns = _duration_to_ns(row[columns["start_ns"]], columns["start_ns"], field="start_ns")
        duration_ns = _duration_to_ns(
            row[columns["duration_ns"]], columns["duration_ns"], field="duration_ns"
        )
        launch = {
            **_kernel_identity(name),
            "start_ns": start_ns,
            "duration_ns": duration_ns,
            "end_ns": start_ns + duration_ns,
            "grid": row[columns["grid"]].strip() if "grid" in columns else None,
            "block": row[columns["block"]].strip() if "block" in columns else None,
            "correlation_id": (
                row[columns["correlation_id"]].strip()
                if "correlation_id" in columns
                else None
            ),
            "graph_node_id": (
                row[columns["graph_node_id"]].strip()
                if "graph_node_id" in columns
                else None
            ),
            "graph_id": (
                row[columns["graph_id"]].strip()
                if "graph_id" in columns
                else None
            ),
        }
        launches.append(launch)
    if not launches:
        raise ArtifactError(f"kernel trace contains no kernel rows: {path}")
    launches.sort(key=lambda item: (item["start_ns"], item["raw_name"]))
    accumulated = sum(item["duration_ns"] for item in launches)
    busy_union = interval_union_ns((item["start_ns"], item["end_ns"]) for item in launches)
    return {
        "status": "available",
        "source_path": str(path),
        "column_resolution": provenance,
        "launches": launches,
        "start_ns": min(item["start_ns"] for item in launches),
        "end_ns": max(item["end_ns"] for item in launches),
        "accumulated_kernel_ns": accumulated,
        "gpu_busy_union_ns": busy_union,
    }


def inspect_sqlite(path: Path) -> dict[str, Any]:
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            if not str(row[0]).startswith("sqlite_")
        ]
        if not integrity or integrity[0] != "ok":
            raise ArtifactError(f"SQLite integrity check failed for {path}: {integrity}")
        if not tables:
            raise ArtifactError(f"SQLite contains no user tables: {path}")
        table_rows: dict[str, int] = {}
        for table in tables:
            escaped = table.replace('"', '""')
            table_rows[table] = int(
                connection.execute(f'SELECT COUNT(*) FROM "{escaped}"').fetchone()[0]
            )
        return {"status": "available", "integrity_check": "ok", "table_rows": table_rows}
    except sqlite3.DatabaseError as exc:
        raise ArtifactError(f"invalid SQLite artifact {path}: {exc}") from exc
    finally:
        try:
            connection.close()
        except UnboundLocalError:
            pass


_SQLITE_REQUIRED_COLUMNS = {
    "StringIds": {"id", "value"},
    "NVTX_EVENTS": {"start", "end", "text", "textId"},
    "CUPTI_ACTIVITY_KIND_KERNEL": {
        "start",
        "end",
        "demangledName",
        "gridX",
        "gridY",
        "gridZ",
        "blockX",
        "blockY",
        "blockZ",
    },
    "CUPTI_ACTIVITY_KIND_RUNTIME": {"start", "end", "nameId"},
    "CUPTI_ACTIVITY_KIND_MEMCPY": {"start", "end", "bytes", "copyKind"},
    "CUPTI_ACTIVITY_KIND_MEMSET": {"start", "end", "bytes"},
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION": {
        "start",
        "end",
        "syncType",
        "eventId",
        "correlationId",
    },
    "ENUM_CUDA_MEMCPY_OPER": {"id", "label"},
    "ENUM_CUPTI_SYNC_TYPE": {"id", "label"},
}


def _sqlite_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    escaped = table.replace('"', '""')
    return {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{escaped}")')
    }


def _require_sqlite_schema(
    connection: sqlite3.Connection,
    *,
    path: Path,
    graph_level: bool = False,
) -> None:
    integrity = connection.execute("PRAGMA integrity_check").fetchone()
    if not integrity or integrity[0] != "ok":
        raise ArtifactError(f"SQLite integrity check failed for {path}: {integrity}")
    for table, required in _SQLITE_REQUIRED_COLUMNS.items():
        observed = _sqlite_columns(connection, table)
        if not observed:
            if graph_level and table == "CUPTI_ACTIVITY_KIND_KERNEL":
                continue
            raise ArtifactError(f"SQLite artifact is missing required table {table}: {path}")
        missing = sorted(required - observed)
        if missing:
            raise ColumnError(
                f"SQLite table {table} is missing required columns {missing}; "
                f"observed={sorted(observed)}"
            )


def _stage_bounds(
    connection: sqlite3.Connection,
    *,
    stage: str,
) -> dict[str, Any]:
    range_name = STAGE_NVTX_RANGE_NAMES[stage]
    rows = connection.execute(
        """
        SELECT n.start, n.end, n.textId, COALESCE(n.text, s.value) AS value
        FROM NVTX_EVENTS AS n
        LEFT JOIN StringIds AS s ON s.id = n.textId
        WHERE COALESCE(n.text, s.value) = ?
        ORDER BY n.start, n.end
        """,
        (range_name,),
    ).fetchall()
    if len(rows) != 1:
        raise ArtifactError(
            f"expected exactly one NVTX range named {range_name!r}, got {len(rows)}"
        )
    row = rows[0]
    if row[1] is None:
        raise ArtifactError(f"NVTX stage range {range_name!r} is not closed")
    start_ns, end_ns = int(row[0]), int(row[1])
    if end_ns <= start_ns:
        raise ArtifactError(
            f"NVTX stage range {range_name!r} has invalid bounds {start_ns}..{end_ns}"
        )
    return {
        "stage": stage,
        "range_name": str(row[3]),
        "text_id": int(row[2]) if row[2] is not None else None,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_ns": end_ns - start_ns,
    }


def _contained_rows(
    connection: sqlite3.Connection,
    query: str,
    *,
    start_ns: int,
    end_ns: int,
) -> list[sqlite3.Row]:
    return list(connection.execute(query, (start_ns, end_ns)).fetchall())


def _median_ns(values: Sequence[int]) -> int:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return int(round((ordered[midpoint - 1] + ordered[midpoint]) / 2))


def _summarize_durations(
    rows: Sequence[Mapping[str, Any]],
    *,
    name_key: str,
) -> list[list[Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[str(row[name_key])].append(int(row["end_ns"]) - int(row["start_ns"]))
    total_all = sum(sum(values) for values in grouped.values())
    summary: list[list[Any]] = []
    for name, values in grouped.items():
        total = sum(values)
        summary.append(
            [
                (total / total_all * 100.0) if total_all else 0.0,
                total,
                len(values),
                int(round(total / len(values))),
                _median_ns(values),
                min(values),
                max(values),
                name,
            ]
        )
    return sorted(summary, key=lambda row: (-int(row[1]), str(row[-1])))


def _summarize_sizes(
    rows: Sequence[Mapping[str, Any]],
    *,
    name_key: str,
) -> list[list[Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[str(row[name_key])].append(int(row["bytes"]))
    summary: list[list[Any]] = []
    for name, values in grouped.items():
        total = sum(values)
        summary.append(
            [
                total,
                len(values),
                int(round(total / len(values))),
                _median_ns(values),
                min(values),
                max(values),
                name,
            ]
        )
    return sorted(summary, key=lambda row: (-int(row[0]), str(row[-1])))


def _write_csv_report(
    path: Path,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def _extracted_artifact(
    path: Path,
    *,
    role: str,
    stage: str,
    row_count: int,
    allow_empty_rows: bool = False,
) -> dict[str, Any]:
    return {
        "role": role,
        "stage": stage,
        "path": str(path),
        "required": True,
        "allow_empty_rows": allow_empty_rows,
        "row_count": row_count,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def extract_sqlite_reports(
    sqlite_path: Path,
    output_dir: Path,
    *,
    graph_level: bool = False,
) -> dict[str, Any]:
    """Extract exact stage-contained native-like CSVs from an nsys SQLite export."""

    sqlite_path = sqlite_path.resolve()
    output_dir = output_dir.resolve()
    if not sqlite_path.is_file():
        raise ArtifactError(f"SQLite artifact is missing: {sqlite_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        _require_sqlite_schema(
            connection,
            path=sqlite_path,
            graph_level=graph_level,
        )
        bounds_by_stage = {
            stage: _stage_bounds(connection, stage=stage)
            for stage in STAGES
        }
        timing = bounds_by_stage["timing_generation"]
        main = bounds_by_stage["main_generation"]
        if timing["start_ns"] < main["end_ns"] and main["start_ns"] < timing["end_ns"]:
            raise ArtifactError("timing and main NVTX stage ranges overlap")

        extracted_stages: dict[str, Any] = {}
        all_artifacts: list[dict[str, Any]] = []
        for stage in STAGES:
            bounds = bounds_by_stage[stage]
            start_ns, end_ns = bounds["start_ns"], bounds["end_ns"]
            kernel_columns = _sqlite_columns(
                connection,
                "CUPTI_ACTIVITY_KIND_KERNEL",
            )
            optional_kernel_selects = ", ".join(
                (
                    f"k.{column} AS {column}"
                    if column in kernel_columns
                    else f"NULL AS {column}"
                )
                for column in ("correlationId", "graphNodeId", "graphId")
            )
            kernels = (
                _contained_rows(
                    connection,
                    f"""
                    SELECT k.start AS start_ns, k.end AS end_ns,
                           s.value AS name,
                           k.gridX, k.gridY, k.gridZ,
                           k.blockX, k.blockY, k.blockZ,
                           {optional_kernel_selects}
                    FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
                    JOIN StringIds AS s ON s.id = k.demangledName
                    WHERE k.start >= ? AND k.end <= ?
                    ORDER BY k.start, k.end, s.value
                    """,
                    start_ns=start_ns,
                    end_ns=end_ns,
                )
                if kernel_columns
                else []
            )
            runtimes = _contained_rows(
                connection,
                """
                SELECT r.start AS start_ns, r.end AS end_ns, s.value AS name
                FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
                JOIN StringIds AS s ON s.id = r.nameId
                WHERE r.start >= ? AND r.end <= ?
                ORDER BY r.start, r.end, s.value
                """,
                start_ns=start_ns,
                end_ns=end_ns,
            )
            memcopies = _contained_rows(
                connection,
                """
                SELECT m.start AS start_ns, m.end AS end_ns, m.bytes,
                       e.label AS operation
                FROM CUPTI_ACTIVITY_KIND_MEMCPY AS m
                JOIN ENUM_CUDA_MEMCPY_OPER AS e ON e.id = m.copyKind
                WHERE m.start >= ? AND m.end <= ?
                ORDER BY m.start, m.end, e.label
                """,
                start_ns=start_ns,
                end_ns=end_ns,
            )
            memsets = _contained_rows(
                connection,
                """
                SELECT m.start AS start_ns, m.end AS end_ns, m.bytes,
                       'Memset' AS operation
                FROM CUPTI_ACTIVITY_KIND_MEMSET AS m
                WHERE m.start >= ? AND m.end <= ?
                ORDER BY m.start, m.end
                """,
                start_ns=start_ns,
                end_ns=end_ns,
            )
            nvtx = _contained_rows(
                connection,
                """
                SELECT n.start AS start_ns, n.end AS end_ns,
                       COALESCE(n.text, s.value) AS name
                FROM NVTX_EVENTS AS n
                LEFT JOIN StringIds AS s ON s.id = n.textId
                WHERE n.end IS NOT NULL AND n.start >= ? AND n.end <= ?
                      AND COALESCE(n.text, s.value) IS NOT NULL
                ORDER BY n.start, n.end, name
                """,
                start_ns=start_ns,
                end_ns=end_ns,
            )
            synchronizations = _contained_rows(
                connection,
                """
                SELECT s.start AS start_ns, s.end AS end_ns,
                       e.label AS sync_type, s.eventId AS event_id,
                       s.correlationId AS correlation_id
                FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION AS s
                JOIN ENUM_CUPTI_SYNC_TYPE AS e ON e.id = s.syncType
                WHERE s.start >= ? AND s.end <= ?
                ORDER BY s.start, s.end, e.label
                """,
                start_ns=start_ns,
                end_ns=end_ns,
            )
            memory_rows = list(memcopies) + list(memsets)
            if not graph_level and not kernels:
                raise ArtifactError(
                    f"node-level extraction found no kernel rows inside {stage}"
                )

            prefix = output_dir / stage
            kernel_trace_path = prefix.with_suffix(".cuda_kernel_trace.csv")
            kernel_summary_path = prefix.with_suffix(".cuda_kernel_summary.csv")
            api_path = prefix.with_suffix(".cuda_api_summary.csv")
            memory_time_path = prefix.with_suffix(".cuda_memory_time_summary.csv")
            memory_size_path = prefix.with_suffix(".cuda_memory_size_summary.csv")
            nvtx_path = prefix.with_suffix(".nvtx_summary.csv")
            sync_path = prefix.with_suffix(".cuda_api_sync.csv")

            _write_csv_report(
                kernel_trace_path,
                (
                    "Start (ns)",
                    "Duration (ns)",
                    "CorrId",
                    "GraphNodeId",
                    "GraphId",
                    "GridXYZ",
                    "BlockXYZ",
                    "Name",
                ),
                [
                    (
                        row["start_ns"],
                        int(row["end_ns"]) - int(row["start_ns"]),
                        row["correlationId"],
                        row["graphNodeId"],
                        row["graphId"],
                        f"{row['gridX']} {row['gridY']} {row['gridZ']}",
                        f"{row['blockX']} {row['blockY']} {row['blockZ']}",
                        row["name"],
                    )
                    for row in kernels
                ],
            )
            _write_csv_report(
                kernel_summary_path,
                (
                    "Time (%)",
                    "Total Time (ns)",
                    "Instances",
                    "Avg (ns)",
                    "Med (ns)",
                    "Min (ns)",
                    "Max (ns)",
                    "Name",
                ),
                _summarize_durations(kernels, name_key="name"),
            )
            timing_headers = (
                "Time (%)",
                "Total Time (ns)",
                "Num Calls",
                "Avg (ns)",
                "Med (ns)",
                "Min (ns)",
                "Max (ns)",
                "Name",
            )
            _write_csv_report(
                api_path,
                timing_headers,
                _summarize_durations(runtimes, name_key="name"),
            )
            memory_timing_rows = _summarize_durations(memory_rows, name_key="operation")
            _write_csv_report(
                memory_time_path,
                timing_headers[:-1] + ("Operation",),
                memory_timing_rows,
            )
            _write_csv_report(
                memory_size_path,
                (
                    "Total (B)",
                    "Operations",
                    "Avg (B)",
                    "Med (B)",
                    "Min (B)",
                    "Max (B)",
                    "Operation",
                ),
                _summarize_sizes(memory_rows, name_key="operation"),
            )
            _write_csv_report(
                nvtx_path,
                timing_headers[:-1] + ("Range",),
                _summarize_durations(nvtx, name_key="name"),
            )
            _write_csv_report(
                sync_path,
                (
                    "Start (ns)",
                    "Duration (ns)",
                    "Sync Type",
                    "Event ID",
                    "Correlation ID",
                ),
                [
                    (
                        row["start_ns"],
                        int(row["end_ns"]) - int(row["start_ns"]),
                        row["sync_type"],
                        row["event_id"],
                        row["correlation_id"],
                    )
                    for row in synchronizations
                ],
            )

            artifact_specs = [
                _extracted_artifact(
                    kernel_summary_path,
                    role="cuda_kernel_summary",
                    stage=stage,
                    row_count=len(kernels),
                    allow_empty_rows=graph_level,
                ),
                _extracted_artifact(
                    kernel_trace_path,
                    role="cuda_kernel_trace",
                    stage=stage,
                    row_count=len(kernels),
                    allow_empty_rows=graph_level,
                ),
                _extracted_artifact(api_path, role="cuda_api_summary", stage=stage, row_count=len(runtimes)),
                _extracted_artifact(
                    memory_time_path,
                    role="cuda_memory_time_summary",
                    stage=stage,
                    row_count=len(memory_rows),
                    allow_empty_rows=True,
                ),
                _extracted_artifact(
                    memory_size_path,
                    role="cuda_memory_size_summary",
                    stage=stage,
                    row_count=len(memory_rows),
                    allow_empty_rows=True,
                ),
                _extracted_artifact(nvtx_path, role="nvtx_summary", stage=stage, row_count=len(nvtx)),
                _extracted_artifact(
                    sync_path,
                    role="cuda_api_sync",
                    stage=stage,
                    row_count=len(synchronizations),
                    allow_empty_rows=True,
                ),
            ]
            all_artifacts.extend(artifact_specs)
            graph_launches = None
            if _sqlite_columns(connection, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
                graph_trace_columns = _sqlite_columns(
                    connection,
                    "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
                )
                missing_graph_bounds = {"start", "end"} - graph_trace_columns
                if missing_graph_bounds:
                    raise ColumnError(
                        "SQLite table CUPTI_ACTIVITY_KIND_GRAPH_TRACE is missing "
                        f"required columns {sorted(missing_graph_bounds)}"
                    )
                graph_launches = int(
                    connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE
                        WHERE start >= ? AND end <= ?
                        """,
                        (start_ns, end_ns),
                    ).fetchone()[0]
                )
            extracted_stages[stage] = {
                **bounds,
                "counts": {
                    "kernels": len(kernels),
                    "kernel_launches": len(kernels),
                    "cuda_graph_launches": graph_launches,
                    "cuda_api_calls": len(runtimes),
                    "memory_operations": len(memory_rows),
                    "nvtx_ranges": len(nvtx),
                    "synchronizations": len(synchronizations),
                    "sync_rows": len(synchronizations),
                },
                "artifacts": artifact_specs,
            }
    except sqlite3.DatabaseError as exc:
        raise ArtifactError(f"invalid SQLite artifact {sqlite_path}: {exc}") from exc
    finally:
        connection.close()

    extraction: dict[str, Any] = {
        "schema_version": "mapperatorinator.nsight-sqlite-extraction.v1",
        "source_sqlite": str(sqlite_path),
        "source_sqlite_sha256": _sha256_file(sqlite_path),
        "graph_level": graph_level,
        "stage_assignment": "exact_nvtx_text_resolution_and_timestamp_containment",
        "semantic_graph_attribution_inferred": False,
        "stages": extracted_stages,
        "artifacts": all_artifacts,
    }
    extraction["extraction_sha256"] = _sha256_bytes(_canonical_json(extraction))
    extraction_path = output_dir / "extraction.json"
    extraction_path.write_text(
        json.dumps(extraction, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return extraction


def _percentage(numerator_ns: int, denominator_id: str, denominator_ns: int) -> dict[str, Any]:
    return {
        "numerator_ns": numerator_ns,
        "denominator_id": denominator_id,
        "denominator_ns": denominator_ns,
        "percent": (numerator_ns / denominator_ns * 100.0) if denominator_ns else None,
        "status": "available" if denominator_ns else "undefined_zero_denominator",
    }


def _duration_from_stage(stage: Mapping[str, Any], canonical: str) -> int | None:
    aliases = {
        "outer_wall_ns": ("outer_wall_ns", "wall_ns"),
        "synchronized_model_ns": ("synchronized_model_ns", "model_ns"),
    }[canonical]
    for key in aliases:
        if stage.get(key) is not None:
            return int(round(_finite_number(stage[key], field=key)))
    second_aliases = {
        "outer_wall_ns": ("outer_wall_seconds", "wall_seconds"),
        "synchronized_model_ns": ("synchronized_model_seconds", "model_elapsed_seconds"),
    }[canonical]
    for key in second_aliases:
        if stage.get(key) is not None:
            return int(round(_finite_number(stage[key], field=key) * 1_000_000_000))
    return None


def _resolve_path(path: str, base: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate


def _validate_artifact(
    spec: Mapping[str, Any],
    base: Path,
    *,
    allow_empty: bool = False,
) -> tuple[Path, dict[str, Any]]:
    if not spec.get("role") or not spec.get("path"):
        raise ArtifactError(f"artifact must contain role and path: {spec}")
    path = _resolve_path(str(spec["path"]), base)
    required = bool(spec.get("required", True))
    if not path.exists():
        if required:
            raise ArtifactError(f"required artifact is missing: {path}")
        return path, {
            "role": spec["role"],
            "stage": spec.get("stage"),
            "path": str(path),
            "required": False,
            "status": "missing",
            "size_bytes": None,
            "sha256": None,
        }
    if not path.is_file():
        raise ArtifactError(f"artifact is not a regular file: {path}")
    size = path.stat().st_size
    if size == 0 and not (allow_empty or spec.get("allow_empty", False)):
        raise ArtifactError(f"artifact is empty: {path}")
    digest = _sha256_file(path)
    expected = spec.get("sha256") or spec.get("expected_sha256")
    if expected is not None and str(expected).lower() != digest:
        raise ArtifactError(
            f"artifact SHA-256 mismatch for {path}: expected {expected}, got {digest}"
        )
    return path, {
        "role": spec["role"],
        "stage": spec.get("stage"),
        "path": str(path),
        "required": required,
        "status": "available" if size else "empty",
        "size_bytes": size,
        "sha256": digest,
    }


def _artifact_by_role(
    artifacts: Sequence[tuple[Mapping[str, Any], Path, dict[str, Any]]],
    role: str,
    stage: str | None = None,
) -> tuple[Mapping[str, Any], Path, dict[str, Any]] | None:
    matches = [
        item
        for item in artifacts
        if item[0]["role"] == role and (stage is None or item[0].get("stage") == stage)
    ]
    if len(matches) > 1:
        raise ArtifactError(f"multiple {role} artifacts declared for stage {stage}")
    return matches[0] if matches else None


def _parse_stage_report(
    artifacts: Sequence[tuple[Mapping[str, Any], Path, dict[str, Any]]],
    *,
    role: str,
    stage: str,
    parser: Any,
) -> dict[str, Any]:
    item = _artifact_by_role(artifacts, role, stage)
    if item is None:
        return {
            "status": "not_available",
            "role": role,
            "stage": stage,
            "reason": "no stage-scoped artifact was declared",
        }
    spec, path, record = item
    if record["status"] in {"missing", "empty"}:
        return {
            "status": record["status"],
            "role": role,
            "stage": stage,
            "source_path": str(path),
            "required": bool(spec.get("required", True)),
        }
    if spec.get("allow_empty_rows", False):
        declared_row_count = int(spec.get("row_count", -1))
        row_reader = _read_structured_csv if role in ANALYSIS_ROLES else _read_csv
        if declared_row_count == 0 or (
            declared_row_count < 0 and not row_reader(path)[1]
        ):
            return {
                "status": "empty",
                "role": role,
                "stage": stage,
                "source_path": str(path),
                "required": bool(spec.get("required", True)),
                "row_count": 0,
            }
    return parser(path)


def _duration_report_denominator(
    report: dict[str, Any],
    *,
    entries_key: str,
    total_key: str,
    denominator_id: str,
    denominators: dict[str, dict[str, Any]],
    kind: str,
) -> None:
    if report.get("status") != "available":
        return
    total = int(report[total_key])
    denominators[denominator_id] = {
        "value_ns": total,
        "kind": kind,
        "authoritative": False,
        "may_exceed_window_wall": True,
    }
    for entry in report[entries_key]:
        entry["share_of_report_accumulated_time"] = _percentage(
            int(entry["total_ns"]), denominator_id, total
        )


def _aggregate_trace_kernels(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    shapes: dict[str, dict[tuple[str | None, str | None], int]] = defaultdict(lambda: defaultdict(int))
    for launch in trace["launches"]:
        name = launch["raw_name"]
        entry = grouped.setdefault(
            name,
            {
                **_kernel_identity(name),
                "calls": 0,
                "total_ns": 0,
                "average_ns": None,
                "minimum_ns": None,
                "maximum_ns": None,
                "launch_shapes": [],
            },
        )
        duration = launch["duration_ns"]
        entry["calls"] += 1
        entry["total_ns"] += duration
        entry["minimum_ns"] = duration if entry["minimum_ns"] is None else min(entry["minimum_ns"], duration)
        entry["maximum_ns"] = duration if entry["maximum_ns"] is None else max(entry["maximum_ns"], duration)
        shapes[name][(launch["grid"], launch["block"])] += 1
    for name, entry in grouped.items():
        entry["average_ns"] = int(round(entry["total_ns"] / entry["calls"]))
        entry["launch_shapes"] = [
            {"grid": grid, "block": block, "calls": calls}
            for (grid, block), calls in sorted(shapes[name].items(), key=lambda item: str(item[0]))
        ]
    return sorted(grouped.values(), key=lambda item: (-item["total_ns"], item["raw_name"]))


def _kernel_family_summary(
    kernels: Sequence[Mapping[str, Any]],
    *,
    denominator_id: str,
    denominator_ns: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for kernel in kernels:
        family = str(kernel["family"])
        entry = grouped.setdefault(
            family,
            {
                "family": family,
                "kernel_count": 0,
                "calls": 0,
                "total_ns": 0,
            },
        )
        entry["kernel_count"] += 1
        entry["calls"] += int(kernel["calls"])
        entry["total_ns"] += int(kernel["total_ns"])
    families = sorted(
        grouped.values(),
        key=lambda entry: (-entry["total_ns"], entry["family"]),
    )
    for entry in families:
        entry["share_of_accumulated_kernel_time"] = _percentage(
            entry["total_ns"], denominator_id, denominator_ns
        )
    return families


def _attach_trace_launch_shapes(
    kernels: Sequence[Mapping[str, Any]],
    trace_kernels: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    trace_by_name = {str(kernel["raw_name"]): kernel for kernel in trace_kernels}
    enriched: list[dict[str, Any]] = []
    for kernel in kernels:
        entry = dict(kernel)
        trace_entry = trace_by_name.get(str(entry["raw_name"]))
        if trace_entry is not None:
            entry["launch_shapes"] = list(trace_entry["launch_shapes"])
            entry["trace_calls"] = int(trace_entry["calls"])
        enriched.append(entry)
    return enriched


def _semantic_regions(
    kernels: Sequence[Mapping[str, Any]],
    attribution: Mapping[str, Any] | None,
    *,
    denominator_id: str,
    denominator_ns: int,
    graph_pass: bool,
) -> dict[str, Any]:
    attribution = attribution or {}
    method = str(attribution.get("method", "none"))
    verified = bool(attribution.get("verified_on_replay", False))
    region_map = attribution.get("kernel_regions", {})
    if not isinstance(region_map, dict):
        raise NsightProfileError("graph_attribution.kernel_regions must be an object")
    if graph_pass and not verified:
        can_assign = False
        reason = "graph replay was not correlated to capture-time semantic ranges"
    elif method in {"none", "projection"}:
        can_assign = False
        reason = "attribution is unavailable or projection-only"
    else:
        can_assign = True
        reason = None

    totals = {region: 0 for region in DECODER_REGIONS}
    for kernel in kernels:
        region = region_map.get(kernel["raw_name"]) if can_assign else None
        if region not in DECODER_REGIONS[:-1]:
            region = "unattributed"
        totals[region] += int(kernel["total_ns"])
    attributed = sum(totals[region] for region in DECODER_REGIONS[:-1])
    return {
        "attribution": {
            "method": method,
            "confidence": "exact" if can_assign else "none",
            "verified_on_replay": verified,
            "reason": reason,
            "matched_kernel_ns": attributed,
            "unmatched_kernel_ns": totals["unattributed"],
        },
        "coverage": _percentage(attributed, denominator_id, denominator_ns),
        "regions": {
            region: {
                "kernel_accumulated_ns": totals[region],
                "share": _percentage(totals[region], denominator_id, denominator_ns),
            }
            for region in DECODER_REGIONS
        },
    }


def _analyze_stage(
    run: Mapping[str, Any],
    stage_name: str,
    artifacts: Sequence[tuple[Mapping[str, Any], Path, dict[str, Any]]],
) -> dict[str, Any]:
    stage = run.get("stages", {}).get(stage_name)
    if stage is None:
        return {"status": "not_present"}
    if not isinstance(stage, dict):
        raise NsightProfileError(f"{run['run_id']} {stage_name} must be an object")

    run_id = str(run["run_id"])
    outer_ns = _duration_from_stage(stage, "outer_wall_ns")
    model_ns = _duration_from_stage(stage, "synchronized_model_ns")
    tokens = int(stage.get("generated_tokens", 0) or 0)
    performance = {
        "outer_wall_ns": outer_ns,
        "synchronized_model_ns": model_ns,
        "generated_tokens": tokens,
        "tps": {
            "value": (tokens / (model_ns / 1_000_000_000)) if model_ns else None,
            "authoritative": bool(run.get("authoritative_performance", False)),
            "source_run_id": run_id,
        },
    }
    denominators: dict[str, dict[str, Any]] = {}
    if outer_ns is not None:
        denominators[f"{run_id}.{stage_name}.outer_wall_ns"] = {
            "value_ns": outer_ns,
            "kind": "outer_stage_wall",
            "authoritative": bool(run.get("authoritative_performance", False)),
        }
    if model_ns is not None:
        denominators[f"{run_id}.{stage_name}.synchronized_model_ns"] = {
            "value_ns": model_ns,
            "kind": "synchronized_model_time",
            "authoritative": bool(run.get("authoritative_performance", False)),
        }

    parsed_summary = _parse_stage_report(
        artifacts,
        role="cuda_kernel_summary",
        stage=stage_name,
        parser=parse_kernel_summary,
    )
    parsed_trace = _parse_stage_report(
        artifacts,
        role="cuda_kernel_trace",
        stage=stage_name,
        parser=parse_kernel_trace,
    )
    trace_kernels = (
        _aggregate_trace_kernels(parsed_trace)
        if parsed_trace["status"] == "available"
        else []
    )
    if parsed_summary["status"] == "available":
        kernels = _attach_trace_launch_shapes(
            parsed_summary["kernels"],
            trace_kernels,
        )
        accumulated = parsed_summary["accumulated_kernel_ns"]
        source = "cuda_kernel_summary"
    elif parsed_trace["status"] == "available":
        kernels = trace_kernels
        accumulated = parsed_trace["accumulated_kernel_ns"]
        source = "cuda_kernel_trace"
    else:
        kernels = []
        accumulated = 0
        source = None

    kernel_denom_id = f"{run_id}.{stage_name}.accumulated_kernel_ns"
    if source:
        denominators[kernel_denom_id] = {
            "value_ns": accumulated,
            "kind": "accumulated_kernel_duration",
            "source": source,
            "may_exceed_window_wall": True,
        }
    top_kernels = []
    for kernel in kernels:
        entry = dict(kernel)
        entry["share_of_accumulated_kernel_time"] = _percentage(
            int(entry["total_ns"]), kernel_denom_id, accumulated
        )
        top_kernels.append(entry)
    kernel_families = _kernel_family_summary(
        kernels,
        denominator_id=kernel_denom_id,
        denominator_ns=accumulated,
    )

    trace_window: dict[str, Any] = {"status": "not_available"}
    if parsed_trace["status"] == "available":
        trace_start = int(stage.get("trace_window_start_ns", parsed_trace["start_ns"]))
        trace_end = int(stage.get("trace_window_end_ns", parsed_trace["end_ns"]))
        if trace_end < trace_start:
            raise NsightProfileError(f"{run_id} {stage_name} has a negative trace window")
        wall = trace_end - trace_start
        busy = parsed_trace["gpu_busy_union_ns"]
        if busy > wall:
            raise NsightProfileError(
                f"{run_id} {stage_name} GPU busy union exceeds the trace window"
            )
        window_id = f"{run_id}.{stage_name}.trace_window_wall_ns"
        busy_id = f"{run_id}.{stage_name}.gpu_busy_union_ns"
        denominators[window_id] = {
            "value_ns": wall,
            "kind": "traced_capture_window_wall",
            "authoritative": False,
        }
        denominators[busy_id] = {
            "value_ns": busy,
            "kind": "gpu_busy_interval_union",
            "authoritative": False,
        }
        trace_window = {
            "status": "available",
            "start_ns": trace_start,
            "end_ns": trace_end,
            "wall_ns": wall,
            "kernel_accumulated_ns": parsed_trace["accumulated_kernel_ns"],
            "gpu_busy_union_ns": busy,
            "gpu_idle_gap_union_ns": wall - busy,
            "gpu_busy_share_of_window": _percentage(busy, window_id, wall),
            "kernel_accumulated_share_of_window": _percentage(
                parsed_trace["accumulated_kernel_ns"], window_id, wall
            ) | {"may_exceed_100": True},
        }

    graph_pass = run["pass_kind"] in {"nsys_graph", "nsys_node"}
    semantic = _semantic_regions(
        kernels,
        stage.get("graph_attribution"),
        denominator_id=kernel_denom_id,
        denominator_ns=accumulated,
        graph_pass=graph_pass,
    )

    cuda_api_report = _parse_stage_report(
        artifacts,
        role="cuda_api_summary",
        stage=stage_name,
        parser=parse_cuda_api_summary,
    )
    api_denominator_id = f"{run_id}.{stage_name}.accumulated_cuda_api_ns"
    _duration_report_denominator(
        cuda_api_report,
        entries_key="apis",
        total_key="accumulated_api_ns",
        denominator_id=api_denominator_id,
        denominators=denominators,
        kind="accumulated_cuda_api_duration",
    )
    if cuda_api_report.get("status") == "available":
        sync = cuda_api_report["synchronization"]
        sync["share_of_report_accumulated_time"] = _percentage(
            int(sync["total_ns"]),
            api_denominator_id,
            int(cuda_api_report["accumulated_api_ns"]),
        )

    memory_time_report = _parse_stage_report(
        artifacts,
        role="cuda_memory_time_summary",
        stage=stage_name,
        parser=parse_cuda_memory_time_summary,
    )
    _duration_report_denominator(
        memory_time_report,
        entries_key="operations",
        total_key="accumulated_memory_operation_ns",
        denominator_id=f"{run_id}.{stage_name}.accumulated_cuda_memory_operation_ns",
        denominators=denominators,
        kind="accumulated_cuda_memory_operation_duration",
    )

    memory_size_report = _parse_stage_report(
        artifacts,
        role="cuda_memory_size_summary",
        stage=stage_name,
        parser=parse_cuda_memory_size_summary,
    )
    if memory_size_report.get("status") == "available":
        size_denominator_id = f"{run_id}.{stage_name}.total_cuda_memory_bytes"
        total_bytes = int(memory_size_report["total_bytes"])
        denominators[size_denominator_id] = {
            "value_bytes": total_bytes,
            "unit": "bytes",
            "kind": "total_cuda_memory_transfer_size",
            "authoritative": False,
        }
        for operation in memory_size_report["operations"]:
            operation["share_of_report_total_bytes"] = {
                "numerator_bytes": int(operation["total_bytes"]),
                "denominator_id": size_denominator_id,
                "denominator_bytes": total_bytes,
                "percent": (
                    int(operation["total_bytes"]) / total_bytes * 100.0
                    if total_bytes
                    else None
                ),
                "status": "available" if total_bytes else "undefined_zero_denominator",
            }

    nvtx_report = _parse_stage_report(
        artifacts,
        role="nvtx_summary",
        stage=stage_name,
        parser=parse_nvtx_summary,
    )
    _duration_report_denominator(
        nvtx_report,
        entries_key="ranges",
        total_key="accumulated_range_ns",
        denominator_id=f"{run_id}.{stage_name}.accumulated_nvtx_range_ns",
        denominators=denominators,
        kind="accumulated_nvtx_range_duration",
    )

    analysis_reports = {
        role: _parse_stage_report(
            artifacts,
            role=role,
            stage=stage_name,
            parser=lambda path, role=role: parse_analysis_csv(
                path,
                analysis_kind=role,
            ),
        )
        for role in ANALYSIS_ROLES
    }
    diagnostic_counts = stage.get("diagnostic_counts", {})
    if not isinstance(diagnostic_counts, dict):
        raise NsightProfileError(
            f"{run_id} {stage_name} diagnostic_counts must be an object"
        )
    diagnostic_counts = {
        str(key): (None if value is None else int(value))
        for key, value in diagnostic_counts.items()
    }
    return {
        "status": "available",
        "performance": performance,
        "transparency": {
            key: stage.get(key)
            for key in ("token_ids_sha256", "stopping_sha256", "cache_behavior_sha256")
        },
        "denominators": denominators,
        "trace_window": trace_window,
        "kernel_report": parsed_summary,
        "kernel_trace": (
            {
                key: value
                for key, value in parsed_trace.items()
                if key != "launches"
            }
            if parsed_trace["status"] == "available"
            else None
        ),
        "top_kernels": top_kernels,
        "kernel_families": kernel_families,
        "decoder_semantic_regions": semantic,
        "cuda_api_report": cuda_api_report,
        "cuda_memory_time_report": memory_time_report,
        "cuda_memory_size_report": memory_size_report,
        "nvtx_report": nvtx_report,
        "semantic_nvtx_ranges": _semantic_nvtx_views(nvtx_report),
        "diagnostic_counts": diagnostic_counts,
        "analysis_reports": analysis_reports,
    }


def classify_ncu_probe(probe: Mapping[str, Any] | None, base: Path) -> dict[str, Any]:
    if not probe or not probe.get("attempted", False):
        return {"status": "not_run", "attempted": False, "usable": False}
    attempts = int(probe.get("attempt_count", 1))
    log_text = str(probe.get("output", ""))
    log_path = probe.get("log_path")
    log_sha = None
    if log_path:
        path = _resolve_path(str(log_path), base)
        if not path.exists() or not path.is_file():
            raise ArtifactError(f"NCU probe log is missing: {path}")
        log_text += "\n" + path.read_text(encoding="utf-8", errors="replace")
        log_sha = _sha256_file(path)
    exit_code = int(probe.get("exit_code", 0))
    targeted = list(probe.get("targeted_collections", []))
    if "ERR_NVGPUCTRPERM" in log_text:
        if attempts != 1:
            raise NsightProfileError("ERR_NVGPUCTRPERM probe must not be retried")
        if targeted:
            raise NsightProfileError(
                "targeted NCU collection was attempted after ERR_NVGPUCTRPERM"
            )
        return {
            "status": "permission_denied",
            "attempted": True,
            "attempt_count": attempts,
            "usable": False,
            "exit_code": exit_code,
            "error_code": "ERR_NVGPUCTRPERM",
            "log_sha256": log_sha,
            "stop_enforced": True,
            "targeted_collections": [],
        }
    if exit_code == 0:
        status, usable = "available", True
    elif re.search(r"not found|no such file", log_text, flags=re.IGNORECASE):
        status, usable = "missing_executable", False
    else:
        status, usable = "probe_failed", False
    return {
        "status": status,
        "attempted": True,
        "attempt_count": attempts,
        "usable": usable,
        "exit_code": exit_code,
        "error_code": None,
        "log_sha256": log_sha,
        "stop_enforced": False,
        "targeted_collections": targeted,
    }


def _load_inference_profile(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise NsightProfileError(f"inference profile must be an object: {path}")
    if not isinstance(payload.get("metadata"), dict):
        raise NsightProfileError(f"inference profile metadata must be an object: {path}")
    if not isinstance(payload.get("generation"), list):
        raise NsightProfileError(f"inference profile generation must be a list: {path}")
    return payload


def _profile_precision(profile: Mapping[str, Any], *, path: Path) -> str:
    values = {str(profile["metadata"].get("precision", ""))}
    values.update(
        str(record.get("precision", ""))
        for record in profile["generation"]
        if isinstance(record, dict) and record.get("precision") is not None
    )
    values.discard("")
    if len(values) != 1:
        raise NsightProfileError(
            f"profile must expose exactly one consistent precision, got {sorted(values)}: {path}"
        )
    return next(iter(values))


def _profile_record_key(record: Mapping[str, Any], index: int) -> list[Any]:
    return [
        record.get("context_type"),
        record.get("mode"),
        record.get("sequence_index", record.get("batch_start_index", index)),
    ]


def _record_token_groups(record: Mapping[str, Any]) -> list[list[int]] | None:
    per_sample = record.get("generated_token_ids_per_sample")
    if per_sample is not None:
        if not isinstance(per_sample, list) or any(not isinstance(group, list) for group in per_sample):
            return None
        return [[int(token) for token in group] for group in per_sample]
    tokens = record.get("generated_token_ids")
    if not isinstance(tokens, list):
        return None
    return [[int(token) for token in tokens]]


def _profile_label_signature(profile: Mapping[str, Any], label: str) -> dict[str, Any]:
    records = [
        record
        for record in profile["generation"]
        if isinstance(record, dict) and record.get("profile_label") == label
    ]
    if not records:
        return {
            "status": "not_checked",
            "reason": f"profile contains no {label} records",
            "records": [],
            "token_stream_sha256": None,
            "stopping_sha256": None,
            "generated_tokens": None,
            "self_consistent": False,
        }

    canonical_records: list[dict[str, Any]] = []
    self_consistent = True
    for index, record in enumerate(records):
        groups = _record_token_groups(record)
        if groups is None:
            return {
                "status": "not_checked",
                "reason": f"{label} record {index} does not contain token IDs",
                "records": [],
                "token_stream_sha256": None,
                "stopping_sha256": None,
                "generated_tokens": None,
                "self_consistent": False,
            }
        generated = int(record.get("generated_tokens", 0) or 0)
        group_counts = [len(group) for group in groups]
        per_sample = record.get("generated_tokens_per_sample")
        if per_sample is not None:
            declared_counts = [int(value) for value in per_sample]
            self_consistent &= declared_counts == group_counts
        else:
            declared_counts = None
        self_consistent &= generated == sum(group_counts)
        canonical_records.append(
            {
                "record_key": _profile_record_key(record, index),
                "generated_tokens": generated,
                "generated_tokens_per_sample": declared_counts,
                "token_counts_per_sample": group_counts,
                "token_ids_sha256": _sha256_bytes(_canonical_json(groups)),
                "final_token_ids": [group[-1] if group else None for group in groups],
                "token_groups": groups,
            }
        )

    token_material = [record["token_groups"] for record in canonical_records]
    stopping_material = [
        {
            "record_key": record["record_key"],
            "generated_tokens": record["generated_tokens"],
            "token_counts_per_sample": record["token_counts_per_sample"],
            "final_token_ids": record["final_token_ids"],
        }
        for record in canonical_records
    ]
    public_records = [
        {key: value for key, value in record.items() if key != "token_groups"}
        for record in canonical_records
    ]
    return {
        "status": "available",
        "records": public_records,
        "token_stream_sha256": _sha256_bytes(_canonical_json(token_material)),
        "stopping_sha256": _sha256_bytes(_canonical_json(stopping_material)),
        "generated_tokens": sum(record["generated_tokens"] for record in canonical_records),
        "self_consistent": self_consistent,
    }


def _canonicalize_graph_cache_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _canonicalize_graph_cache_value(child)
            for key, child in value.items()
            if key != "capture_seconds"
        }
    if isinstance(value, list):
        return [_canonicalize_graph_cache_value(child) for child in value]
    return value


def _profile_graph_cache_signature(profile: Mapping[str, Any], labels: Sequence[str]) -> dict[str, Any]:
    metadata = profile["metadata"]
    material: dict[str, Any] = {
        key: metadata[key]
        for key in PROFILE_PRESET_KEYS
        if key in metadata
    }
    record_material: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        entries = []
        for index, record in enumerate(profile["generation"]):
            if not isinstance(record, dict) or record.get("profile_label") != label:
                continue
            values = {
                key: (
                    _canonicalize_graph_cache_value(record[key])
                    if key == "optimized_cuda_graphs"
                    else record[key]
                )
                for key in PROFILE_GRAPH_CACHE_RECORD_KEYS
                if key in record
            }
            if values:
                entries.append({"record_key": _profile_record_key(record, index), **values})
        if entries:
            record_material[label] = entries
    material["records"] = record_material
    if not any(key in metadata for key in PROFILE_PRESET_KEYS) and not record_material:
        return {"status": "not_available", "sha256": None, "material": None}
    return {
        "status": "available",
        "sha256": _sha256_bytes(_canonical_json(material)),
        "material": material,
    }


def _exact_check(control: Any, traced: Any, *, required: bool = True) -> dict[str, Any]:
    if control is None or traced is None:
        return {
            "control": control,
            "traced": traced,
            "status": "not_checked" if required else "not_available",
            "pass": not required and control is None and traced is None,
        }
    return {
        "control": control,
        "traced": traced,
        "status": "PASS" if control == traced else "FAIL",
        "pass": control == traced,
    }


def _mapping_check(
    control: Mapping[str, Any],
    traced: Mapping[str, Any],
    key: str,
) -> dict[str, Any]:
    if key not in control or key not in traced:
        return {
            "control": control.get(key),
            "traced": traced.get(key),
            "control_present": key in control,
            "traced_present": key in traced,
            "status": "not_checked",
            "pass": False,
        }
    return {
        "control": control[key],
        "traced": traced[key],
        "control_present": True,
        "traced_present": True,
        "status": "PASS" if control[key] == traced[key] else "FAIL",
        "pass": control[key] == traced[key],
    }


def compare_profile_transparency(
    control_path: Path,
    traced_path: Path,
    *,
    labels: Sequence[str] = PROFILE_LABELS,
) -> dict[str, Any]:
    """Compare profiler transparency without using any performance metric."""

    control_path = control_path.resolve()
    traced_path = traced_path.resolve()
    control = _load_inference_profile(control_path)
    traced = _load_inference_profile(traced_path)
    control_precision = _profile_precision(control, path=control_path)
    traced_precision = _profile_precision(traced, path=traced_path)
    if control_precision != traced_precision:
        raise NsightProfileError(
            f"profiler transparency rejects cross-precision comparison: "
            f"{control_precision} != {traced_precision}"
        )

    control_metadata = control["metadata"]
    traced_metadata = traced["metadata"]
    workload_checks = {
        key: _mapping_check(control_metadata, traced_metadata, key)
        for key in PROFILE_WORKLOAD_KEYS
    }
    preset_checks = {
        key: _mapping_check(control_metadata, traced_metadata, key)
        for key in PROFILE_PRESET_KEYS
    }
    artifact_checks = {
        "result_file_sha256": _exact_check(
            control_metadata.get("result_file_sha256"),
            traced_metadata.get("result_file_sha256"),
        ),
        "result_file_size_bytes": _exact_check(
            control_metadata.get("result_file_size_bytes"),
            traced_metadata.get("result_file_size_bytes"),
        ),
    }

    label_checks: dict[str, Any] = {}
    for label in labels:
        control_signature = _profile_label_signature(control, label)
        traced_signature = _profile_label_signature(traced, label)
        checks = {
            "token_stream_sha256": _exact_check(
                control_signature["token_stream_sha256"],
                traced_signature["token_stream_sha256"],
            ),
            "generated_tokens": _exact_check(
                control_signature["generated_tokens"],
                traced_signature["generated_tokens"],
            ),
            "stopping_sha256": _exact_check(
                control_signature["stopping_sha256"],
                traced_signature["stopping_sha256"],
            ),
            "record_shape": _exact_check(
                [record["record_key"] for record in control_signature["records"]],
                [record["record_key"] for record in traced_signature["records"]],
            ),
        }
        checks["control_self_consistent"] = {
            "status": "PASS" if control_signature["self_consistent"] else "FAIL",
            "pass": bool(control_signature["self_consistent"]),
        }
        checks["traced_self_consistent"] = {
            "status": "PASS" if traced_signature["self_consistent"] else "FAIL",
            "pass": bool(traced_signature["self_consistent"]),
        }
        label_checks[label] = {
            "control": control_signature,
            "traced": traced_signature,
            "checks": checks,
            "pass": all(check["pass"] for check in checks.values()),
        }

    control_graph = _profile_graph_cache_signature(control, labels)
    traced_graph = _profile_graph_cache_signature(traced, labels)
    graph_required = control_graph["status"] == "available" or traced_graph["status"] == "available"
    graph_check = _exact_check(
        control_graph["sha256"], traced_graph["sha256"], required=graph_required
    )
    graph_check["control_status"] = control_graph["status"]
    graph_check["traced_status"] = traced_graph["status"]

    all_checks = [
        *workload_checks.values(),
        *preset_checks.values(),
        *artifact_checks.values(),
        graph_check,
    ]
    passed = (
        all(check["pass"] for check in all_checks)
        and all(check["pass"] for check in label_checks.values())
    )
    report: dict[str, Any] = {
        "schema_version": "mapperatorinator.profiler-transparency.v1",
        "comparison_type": "profiler_transparency",
        "control_path": str(control_path),
        "traced_path": str(traced_path),
        "precision": control_precision,
        "labels": list(labels),
        "performance_ignored_for_pass_fail": True,
        "workload_checks": workload_checks,
        "preset_checks": preset_checks,
        "artifact_checks": artifact_checks,
        "label_checks": label_checks,
        "graph_cache_check": graph_check,
        "pass": passed,
    }
    report["analysis_sha256"] = _sha256_bytes(_canonical_json(report))
    return report


def _hash_or_value(run: Mapping[str, Any], key: str) -> Any:
    if run.get(key) is not None:
        return run[key]
    if key == "workload_contract_sha256" and run.get("workload_contract") is not None:
        return _sha256_bytes(_canonical_json(run["workload_contract"]))
    return None


def _transparency_comparison(control: Mapping[str, Any], traced: Mapping[str, Any]) -> dict[str, Any]:
    if control["precision"] != traced["precision"]:
        raise NsightProfileError("profiler transparency cannot compare different precisions")
    checks: dict[str, Any] = {}
    for key in ("workload_contract_sha256", "output_sha256", "output_size_bytes"):
        left, right = _hash_or_value(control, key), _hash_or_value(traced, key)
        checks[key] = {
            "control": left,
            "traced": right,
            "status": "not_checked" if left is None or right is None else ("PASS" if left == right else "FAIL"),
            "pass": left is not None and right is not None and left == right,
        }
    stages: dict[str, Any] = {}
    for stage_name in STAGES:
        left_stage = control.get("stages", {}).get(stage_name)
        right_stage = traced.get("stages", {}).get(stage_name)
        if left_stage is None and right_stage is None:
            continue
        stage_checks = {}
        for key in ("token_ids_sha256", "stopping_sha256", "cache_behavior_sha256"):
            left = left_stage.get(key) if isinstance(left_stage, dict) else None
            right = right_stage.get(key) if isinstance(right_stage, dict) else None
            stage_checks[key] = {
                "control": left,
                "traced": right,
                "status": "not_checked" if left is None or right is None else ("PASS" if left == right else "FAIL"),
                "pass": left is not None and right is not None and left == right,
            }
        control_model = _duration_from_stage(left_stage or {}, "synchronized_model_ns")
        traced_model = _duration_from_stage(right_stage or {}, "synchronized_model_ns")
        control_outer = _duration_from_stage(left_stage or {}, "outer_wall_ns")
        traced_outer = _duration_from_stage(right_stage or {}, "outer_wall_ns")
        stages[stage_name] = {
            "checks": stage_checks,
            "pass": all(item["pass"] for item in stage_checks.values()),
            "profiler_overhead": {
                "control_model_ns": control_model,
                "traced_model_ns": traced_model,
                "model_delta_percent": (
                    (traced_model - control_model) / control_model * 100.0
                    if control_model and traced_model is not None
                    else None
                ),
                "control_outer_ns": control_outer,
                "traced_outer_ns": traced_outer,
                "outer_delta_percent": (
                    (traced_outer - control_outer) / control_outer * 100.0
                    if control_outer and traced_outer is not None
                    else None
                ),
                "authoritative_tps_run_id": control["run_id"],
                "traced_tps_authoritative": False,
            },
        }
    passed = all(item["pass"] for item in checks.values()) and all(
        item["pass"] for item in stages.values()
    )
    return {
        "comparison_type": "profiler_transparency",
        "control_run_id": control["run_id"],
        "traced_run_id": traced["run_id"],
        "precision": control["precision"],
        "checks": checks,
        "stages": stages,
        "pass": passed,
    }


def _pipeline_stage_summary(run: Mapping[str, Any]) -> dict[str, Any]:
    raw = run.get("pipeline_stage_wall_ns")
    if raw is None:
        return {"status": "not_available", "stage_wall_ns": {}}
    if not isinstance(raw, dict):
        raise NsightProfileError("pipeline_stage_wall_ns must be an object")
    stages = {
        str(name): int(round(_finite_number(value, field=f"pipeline_stage_wall_ns.{name}")))
        for name, value in raw.items()
    }

    def group(names: Sequence[str]) -> dict[str, Any]:
        included = [name for name in names if name in stages]
        return {
            "stage_names": included,
            "total_ns": sum(stages[name] for name in included),
        }

    return {
        "status": "available",
        "stage_wall_ns": dict(sorted(stages.items())),
        "timing_postprocessing": group(TIMING_POSTPROCESS_STAGES),
        "main_postprocessing": group(MAIN_POSTPROCESS_STAGES),
    }


def _validated_output_structure(run: Mapping[str, Any]) -> dict[str, Any]:
    raw = run.get("output_structure")
    if raw is None:
        return {"status": "not_available"}
    if not isinstance(raw, dict):
        raise NsightProfileError("output_structure must be an object")
    count_fields = (
        "timing_points",
        "uninherited_timing_points",
        "inherited_timing_points",
        "hit_objects",
        "malformed_lines",
        "nonfinite_values",
    )
    missing = [field for field in count_fields if field not in raw]
    if missing:
        raise NsightProfileError(f"output_structure is missing required fields: {missing}")

    def count(value: Any, *, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise NsightProfileError(f"{field} must be a non-negative integer")
        return value

    result: dict[str, Any] = {
        "status": "available",
        **{
            field: count(raw[field], field=f"output_structure.{field}")
            for field in count_fields
        },
    }
    for nested_field in ("hit_object_types", "malformed_by_section"):
        nested = raw.get(nested_field)
        if not isinstance(nested, dict):
            raise NsightProfileError(f"output_structure.{nested_field} must be an object")
        result[nested_field] = {
            str(key): count(
                value,
                field=f"output_structure.{nested_field}.{key}",
            )
            for key, value in sorted(nested.items())
        }
    scope = raw.get("numeric_validation_scope")
    if not isinstance(scope, str) or not scope:
        raise NsightProfileError(
            "output_structure.numeric_validation_scope must be a non-empty string"
        )
    well_formed = raw.get("finite_and_well_formed")
    if not isinstance(well_formed, bool):
        raise NsightProfileError(
            "output_structure.finite_and_well_formed must be a boolean"
        )
    timing_classified = (
        result["uninherited_timing_points"]
        + result["inherited_timing_points"]
        + result["malformed_by_section"].get("TimingPoints", 0)
    )
    hit_objects_classified = (
        sum(result["hit_object_types"].values())
        + result["malformed_by_section"].get("HitObjects", 0)
    )
    if timing_classified != result["timing_points"]:
        raise NsightProfileError("output_structure timing-point counts are inconsistent")
    if hit_objects_classified != result["hit_objects"]:
        raise NsightProfileError("output_structure hit-object counts are inconsistent")
    expected_well_formed = (
        result["malformed_lines"] == 0 and result["nonfinite_values"] == 0
    )
    if well_formed != expected_well_formed:
        raise NsightProfileError("output_structure well-formed status is inconsistent")
    result["numeric_validation_scope"] = scope
    result["finite_and_well_formed"] = well_formed
    return result


def _cross_precision_comparison(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    if left["precision"] == right["precision"]:
        raise NsightProfileError("cross_precision_divergence requires different precisions")
    stage_results: dict[str, Any] = {}
    for stage_name in STAGES:
        lhs = left.get("stages", {}).get(stage_name, {})
        rhs = right.get("stages", {}).get(stage_name, {})
        if not lhs and not rhs:
            continue
        stage_results[stage_name] = {
            key: {
                "left": lhs.get(key),
                "right": rhs.get(key),
                "equal": lhs.get(key) is not None and lhs.get(key) == rhs.get(key),
            }
            for key in ("token_ids_sha256", "stopping_sha256", "cache_behavior_sha256")
        }
    left_structure = _validated_output_structure(left)
    right_structure = _validated_output_structure(right)
    structure_fields = sorted(
        (set(left_structure) | set(right_structure)) - {"status"}
    )
    structure = {
        "left_status": left_structure["status"],
        "right_status": right_structure["status"],
        "fields": {
            field: {
                "left": left_structure.get(field),
                "right": right_structure.get(field),
                "equal": (
                    left_structure.get(field) is not None
                    and left_structure.get(field) == right_structure.get(field)
                ),
            }
            for field in structure_fields
        },
    }
    return {
        "comparison_type": "cross_precision_divergence",
        "left_run_id": left["run_id"],
        "right_run_id": right["run_id"],
        "left_precision": left["precision"],
        "right_precision": right["precision"],
        "stages": stage_results,
        "output_equal": (
            left.get("output_sha256") is not None
            and left.get("output_sha256") == right.get("output_sha256")
        ),
        "output_structure": structure,
        "is_transparency_gate": False,
    }


def analyze_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise NsightProfileError("manifest must be a JSON object")
    version = payload.get("schema_version")
    if version not in (None, MANIFEST_SCHEMA_VERSION):
        raise NsightProfileError(f"unsupported manifest schema: {version}")
    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise NsightProfileError("manifest must contain a non-empty runs list")

    runs_by_id: dict[str, Mapping[str, Any]] = {}
    analyzed_runs: dict[str, Any] = {}
    groups: dict[str, dict[str, Any]] = {}
    flattened_artifacts: list[dict[str, Any]] = []
    base = manifest_path.parent

    for raw_run in raw_runs:
        if not isinstance(raw_run, dict):
            raise NsightProfileError("every run must be an object")
        run_id = str(raw_run.get("run_id", ""))
        precision = str(raw_run.get("precision", ""))
        engine_variant = str(raw_run.get("engine_variant", ""))
        pass_kind = str(raw_run.get("pass_kind", ""))
        if not run_id or not precision or not engine_variant or pass_kind not in PASS_KINDS:
            raise NsightProfileError(f"invalid run identity: {raw_run}")
        if run_id in runs_by_id:
            raise NsightProfileError(f"duplicate run_id: {run_id}")
        authoritative = bool(raw_run.get("authoritative_performance", False))
        if authoritative and pass_kind != "untraced_control":
            raise NsightProfileError(
                f"traced/non-control run {run_id} cannot have authoritative TPS"
            )
        runs_by_id[run_id] = raw_run

        group_id = str(raw_run.get("run_group_id", f"{precision}:{engine_variant}"))
        group = groups.setdefault(
            group_id,
            {"precision": precision, "engine_variant": engine_variant, "run_ids": []},
        )
        if group["precision"] != precision or group["engine_variant"] != engine_variant:
            raise NsightProfileError(
                f"run group {group_id} mixes precision or engine variants"
            )
        group["run_ids"].append(run_id)

        artifact_specs = raw_run.get("artifacts", [])
        if not isinstance(artifact_specs, list):
            raise ArtifactError(f"{run_id} artifacts must be a list")
        validated: list[tuple[Mapping[str, Any], Path, dict[str, Any]]] = []
        for spec in artifact_specs:
            if not isinstance(spec, dict):
                raise ArtifactError(f"{run_id} artifact entry must be an object")
            role = str(spec.get("role", ""))
            if role in STAGE_REPORT_ROLES and spec.get("stage") not in STAGES:
                raise ArtifactError(
                    f"{run_id} {role} artifact must declare one stage from {list(STAGES)}"
                )
            graph_kernel_empty = (
                pass_kind == "nsys_graph"
                and role in {"cuda_kernel_summary", "cuda_kernel_trace"}
            )
            effective_spec: Mapping[str, Any] = spec
            if graph_kernel_empty:
                effective_spec = {**spec, "allow_empty_rows": True}
            path, record = _validate_artifact(
                effective_spec,
                base,
                allow_empty=graph_kernel_empty,
            )
            record["run_id"] = run_id
            flattened_artifacts.append(record)
            validated.append((effective_spec, path, record))

        sqlite_item = _artifact_by_role(validated, "sqlite")
        sqlite_metadata = inspect_sqlite(sqlite_item[1]) if sqlite_item else None
        stages = {
            stage_name: _analyze_stage(raw_run, stage_name, validated)
            for stage_name in STAGES
        }
        analyzed_runs[run_id] = {
            "run_id": run_id,
            "run_group_id": group_id,
            "precision": precision,
            "engine_variant": engine_variant,
            "engine_preset_version": raw_run.get("engine_preset_version"),
            "pass_kind": pass_kind,
            "authoritative_performance": authoritative,
            "paired_control_run_id": raw_run.get("paired_control_run_id"),
            "workload_contract_sha256": _hash_or_value(raw_run, "workload_contract_sha256"),
            "output_sha256": raw_run.get("output_sha256"),
            "output_size_bytes": raw_run.get("output_size_bytes"),
            "output_structure": _validated_output_structure(raw_run),
            "pipeline": _pipeline_stage_summary(raw_run),
            "identity": raw_run.get("identity", {}),
            "sqlite": sqlite_metadata,
            "stages": stages,
        }

    for group in groups.values():
        group["run_ids"].sort()

    comparisons: list[dict[str, Any]] = []
    for run_id, run in runs_by_id.items():
        control_id = run.get("paired_control_run_id")
        if control_id is None:
            continue
        if control_id not in runs_by_id:
            raise NsightProfileError(f"{run_id} references missing control {control_id}")
        comparisons.append(_transparency_comparison(runs_by_id[control_id], run))
    for request in payload.get("comparisons", []):
        if request.get("comparison_type") != "cross_precision_divergence":
            raise NsightProfileError(f"unsupported requested comparison: {request}")
        left_id, right_id = request.get("left_run_id"), request.get("right_run_id")
        if left_id not in runs_by_id or right_id not in runs_by_id:
            raise NsightProfileError(f"comparison references unknown run: {request}")
        comparisons.append(_cross_precision_comparison(runs_by_id[left_id], runs_by_id[right_id]))

    ncu = classify_ncu_probe(payload.get("ncu_probe"), base)
    status = "complete_with_ncu_unavailable" if ncu["status"] == "permission_denied" else "complete"
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "provenance": payload.get("provenance", {}),
        "tool_capabilities": payload.get("tool_capabilities", {}),
        "run_groups": dict(sorted(groups.items())),
        "runs": dict(sorted(analyzed_runs.items())),
        "comparisons": sorted(
            comparisons,
            key=lambda item: (
                item["comparison_type"],
                item.get("control_run_id", item.get("left_run_id", "")),
                item.get("traced_run_id", item.get("right_run_id", "")),
            ),
        ),
        "ncu": ncu,
        "artifact_manifest": sorted(
            flattened_artifacts,
            key=lambda item: (item["run_id"], str(item.get("stage")), item["role"]),
        ),
        "validation": {
            "traced_tps_authoritative": False,
            "precision_groups_isolated": True,
            "artifact_hashes_verified": True,
        },
    }
    summary["analysis_sha256"] = _sha256_bytes(_canonical_json(summary))
    return summary


def render_text_summary(summary: Mapping[str, Any]) -> str:
    lines = [f"status: {summary['status']}"]
    for group_id, group in summary["run_groups"].items():
        lines.append(
            f"group {group_id}: precision={group['precision']} "
            f"engine={group['engine_variant']} runs={','.join(group['run_ids'])}"
        )
    for run_id, run in summary["runs"].items():
        for stage_name in STAGES:
            stage = run["stages"][stage_name]
            if stage.get("status") != "available":
                continue
            perf = stage["performance"]
            tps = perf["tps"]
            authority = "authoritative" if tps["authoritative"] else "diagnostic"
            lines.append(
                f"{run_id} {stage_name}: tokens={perf['generated_tokens']} "
                f"tps={tps['value']} ({authority})"
            )
            for kernel in stage["top_kernels"][:5]:
                lines.append(
                    f"  kernel {kernel['raw_name']}: total_ns={kernel['total_ns']} "
                    f"calls={kernel['calls']} family={kernel['family']}"
                )
            for family in stage["kernel_families"]:
                lines.append(
                    f"  family {family['family']}: total_ns={family['total_ns']} "
                    f"calls={family['calls']} kernels={family['kernel_count']}"
                )
        pipeline = run.get("pipeline", {})
        if pipeline.get("status") == "available":
            timing_post = pipeline["timing_postprocessing"]
            main_post = pipeline["main_postprocessing"]
            lines.append(
                f"{run_id} postprocessing: timing_ns={timing_post['total_ns']} "
                f"main_ns={main_post['total_ns']}"
            )
    lines.append(f"ncu: {summary['ncu']['status']}")
    return "\n".join(lines) + "\n"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="analyze a run manifest")
    analyze.add_argument("--manifest", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--text-output", type=Path)
    transparency = subparsers.add_parser(
        "transparency",
        help="gate exact same-precision profiler transparency",
    )
    transparency.add_argument("--control", type=Path, required=True)
    transparency.add_argument("--traced", type=Path, required=True)
    transparency.add_argument("--output", type=Path, required=True)
    transparency.add_argument(
        "--labels",
        default=",".join(PROFILE_LABELS),
        help="comma-separated profile labels (default: timing_context,main_generation)",
    )
    extract = subparsers.add_parser(
        "extract-sqlite",
        help="extract exact stage-scoped CSV reports from an nsys SQLite export",
    )
    extract.add_argument("--sqlite", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument(
        "--graph-level",
        action="store_true",
        help="allow graph-level traces to contain zero kernel activity rows",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "analyze":
            summary = analyze_manifest(args.manifest)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            if args.text_output:
                args.text_output.parent.mkdir(parents=True, exist_ok=True)
                args.text_output.write_text(render_text_summary(summary), encoding="utf-8")
            return 0
        if args.command == "transparency":
            labels = [label.strip() for label in args.labels.split(",") if label.strip()]
            if not labels:
                raise NsightProfileError("transparency requires at least one label")
            report = compare_profile_transparency(
                args.control,
                args.traced,
                labels=labels,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            return 0 if report["pass"] else 1
        if args.command == "extract-sqlite":
            extract_sqlite_reports(
                args.sqlite,
                args.output_dir,
                graph_level=args.graph_level,
            )
            return 0
        raise AssertionError(args.command)
    except (NsightProfileError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"nsight-agent-profile: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
