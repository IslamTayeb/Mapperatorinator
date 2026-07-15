"""Strict reciprocal full-song candidate analysis.

The input order is fixed to baseline, candidate, candidate, baseline so drift
from process order is visible.  Performance is accepted only from untraced,
authoritative profiles.  Exact modes are deliberately hard gates; relaxed mode
reports precision/output divergence without calling it exact.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from utils.nsight_agent_profile import (
        PROFILE_GRAPH_CACHE_RECORD_KEYS,
        PROFILE_LABELS,
        PROFILE_PRESET_KEYS,
        PROFILE_WORKLOAD_KEYS,
        _profile_graph_cache_signature,
        _profile_label_signature,
        _profile_precision,
        summarize_osu_structure,
    )
except ModuleNotFoundError:  # Direct ``python utils/...py`` execution.
    from nsight_agent_profile import (  # type: ignore[no-redef]
        PROFILE_GRAPH_CACHE_RECORD_KEYS,
        PROFILE_LABELS,
        PROFILE_PRESET_KEYS,
        PROFILE_WORKLOAD_KEYS,
        _profile_graph_cache_signature,
        _profile_label_signature,
        _profile_precision,
        summarize_osu_structure,
    )


SCHEMA_VERSION = "mapperatorinator.reciprocal-full-song-candidate.v1"
PROFILE_SCHEMA_VERSION = 1
FINAL_WRITE_STAGES = frozenset({"write_osu", "write_osz"})
GROUP_STAGE_NAMES = {
    "pre_request_setup_load": frozenset(
        {
            "compile_args",
            "setup_inference_environment",
            "load_main_model",
            "load_timing_model",
            "load_diffusion_model",
            "build_generation_config",
        }
    ),
    "request_setup": frozenset({"validate_inputs", "setup_processors"}),
    "audio_preparation": frozenset({"audio_load", "audio_segment"}),
    "timing_generation": frozenset(
        {
            "super_timing_generation",
            "timing_context_generation",
            "load_reference_timing",
        }
    ),
    "timing_postprocess": frozenset(
        {"super_timing_postprocess", "timing_context_postprocess"}
    ),
    "main_generation": frozenset({"main_generation"}),
    "merge_resnap_postprocess_write": frozenset(
        {
            "merge_generated_events",
            "derive_timing_from_generated_events",
            "resnap_events",
            "diffusion_position_generation",
            "postprocess_generate_osu",
            "merge_with_reference_beatmap",
            "write_osu",
            "write_osz",
        }
    ),
}
ALL_SUPPORTED_STAGE_NAMES = frozenset().union(*GROUP_STAGE_NAMES.values())
RUN_ORDER = (
    "baseline_first",
    "candidate_first",
    "candidate_second",
    "baseline_second",
)
BASELINE_ROLES = ("baseline_first", "baseline_second")
CANDIDATE_ROLES = ("candidate_first", "candidate_second")
MODES = ("exact-fp32", "exact-same-precision", "relaxed")
TIMING_GENERATION_STAGES = frozenset(
    {"timing_context_generation", "super_timing_generation"}
)
TIMING_POSTPROCESS_STAGES = frozenset(
    {"timing_context_postprocess", "super_timing_postprocess"}
)
FINAL_POSTPROCESS_STAGES = GROUP_STAGE_NAMES["merge_resnap_postprocess_write"]
SETUP_STAGES = (
    GROUP_STAGE_NAMES["pre_request_setup_load"]
    | GROUP_STAGE_NAMES["request_setup"]
)
INVARIANT_METADATA = {
    "inference_engine": "optimized",
    "use_server": False,
    "parallel": False,
    "optimized_runtime_owner": "osuT5.osuT5.inference.optimized.single.engine",
}
INVARIANT_EFFECTIVE_CONFIG = {
    "batch_size": 1,
    "decoder_loop_backend": "active_prefix_cuda_graph",
    "q1_bmm_cross_attention": True,
    "native_decode_kernels": True,
    "native_q1_self_attention": True,
    "native_q1_rope_cache_self_attention": True,
}


class CandidateAnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class Stage:
    name: str
    wall_seconds: float
    started: float
    finished: float
    raw: dict[str, Any]


@dataclass(frozen=True)
class ParsedRun:
    role: str
    profile_path: Path
    osu_path: Path
    profile: dict[str, Any]
    metadata: dict[str, Any]
    precision: str
    stages: tuple[Stage, ...]
    stage_sequence: tuple[str, ...]
    label_signatures: dict[str, dict[str, Any]]
    graph_signature: dict[str, Any]
    workload: dict[str, Any]
    osu_sha256: str
    osu_size_bytes: int
    osu_structure: dict[str, Any]
    metrics: dict[str, float]


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CandidateAnalysisError(f"{name} must be an object")
    return value


def _list(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise CandidateAnalysisError(f"{name} must be a list")
    return value


def _number(
    value: Any,
    *,
    name: str,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateAnalysisError(f"{name} must be numeric")
    parsed = float(value)
    invalid = not math.isfinite(parsed) or parsed < 0.0 or (positive and parsed <= 0.0)
    if invalid:
        qualifier = "positive finite" if positive else "finite non-negative"
        raise CandidateAnalysisError(f"{name} must be {qualifier}")
    return parsed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _load_profile(path: Path, *, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise CandidateAnalysisError(f"{role} profile does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateAnalysisError(f"cannot read {role} profile {path}: {exc}") from exc
    root = _object(payload, name=f"{role}.profile")
    if root.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise CandidateAnalysisError(
            f"{role}.profile.schema_version must be {PROFILE_SCHEMA_VERSION}"
        )
    return root


def _resolve_osu_path(
    profile_path: Path,
    metadata: Mapping[str, Any],
    override: Path | None,
    *,
    role: str,
) -> Path:
    value: str | Path | None = override
    if value is None:
        value = metadata.get("result_file_path", metadata.get("result_path"))
    if not isinstance(value, (str, Path)) or not str(value):
        raise CandidateAnalysisError(
            f"{role} profile must reference result_file_path/result_path or receive an OSU override"
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = profile_path.parent / path
    path = path.resolve()
    if path.suffix.lower() != ".osu" or not path.is_file():
        raise CandidateAnalysisError(f"{role} OSU output is missing or not .osu: {path}")
    return path


def _parse_stages(profile: Mapping[str, Any], *, role: str) -> tuple[Stage, ...]:
    values = _list(profile.get("stages"), name=f"{role}.profile.stages")
    if not values:
        raise CandidateAnalysisError(f"{role}.profile.stages must be non-empty")
    stages: list[Stage] = []
    for index, value in enumerate(values):
        raw = _object(value, name=f"{role}.profile.stages[{index}]")
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise CandidateAnalysisError(f"{role}.profile.stages[{index}].name is invalid")
        if name not in ALL_SUPPORTED_STAGE_NAMES:
            raise CandidateAnalysisError(f"{role} contains unsupported stage {name!r}")
        started = _number(
            raw.get("started_at_perf_counter_seconds"),
            name=f"{role}.{name}.started_at_perf_counter_seconds",
        )
        finished = _number(
            raw.get("finished_at_perf_counter_seconds"),
            name=f"{role}.{name}.finished_at_perf_counter_seconds",
        )
        wall = _number(raw.get("wall_seconds"), name=f"{role}.{name}.wall_seconds")
        if finished < started:
            raise CandidateAnalysisError(f"{role} stage {name!r} finishes before it starts")
        if not math.isclose(wall, finished - started, rel_tol=1e-8, abs_tol=1e-8):
            raise CandidateAnalysisError(f"{role} stage {name!r} wall disagrees with timestamps")
        stages.append(Stage(name, wall, started, finished, dict(raw)))
    for previous, current in zip(stages, stages[1:], strict=False):
        if current.started < previous.started:
            raise CandidateAnalysisError(f"{role} stages are not chronological")
        if current.started < previous.finished - 1e-8:
            raise CandidateAnalysisError(
                f"{role} stages overlap: {previous.name!r} and {current.name!r}"
            )
    required_counts = {
        "compile_args": 1,
        "validate_inputs": 1,
        "main_generation": 1,
    }
    for name, expected in required_counts.items():
        actual = sum(stage.name == name for stage in stages)
        if actual != expected:
            raise CandidateAnalysisError(
                f"{role} profile must contain exactly {expected} {name} stage"
            )
    timing_count = sum(stage.name in TIMING_GENERATION_STAGES for stage in stages)
    if timing_count != 1:
        raise CandidateAnalysisError(
            f"{role} profile must contain exactly one timing generation stage"
        )
    final_count = sum(stage.name in FINAL_WRITE_STAGES for stage in stages)
    if final_count != 1:
        raise CandidateAnalysisError(f"{role} profile must contain exactly one final write")
    if stages[0].name != "compile_args" or stages[-1].name not in FINAL_WRITE_STAGES:
        raise CandidateAnalysisError(
            f"{role} stages must start at compile_args and finish at write_osu/write_osz"
        )
    return tuple(stages)


def _unique_stage(stages: Sequence[Stage], names: set[str] | frozenset[str]) -> Stage:
    matches = [stage for stage in stages if stage.name in names]
    if len(matches) != 1:
        raise CandidateAnalysisError(f"expected one stage from {sorted(names)}, got {len(matches)}")
    return matches[0]


def _stage_sum(stages: Sequence[Stage], names: set[str] | frozenset[str]) -> float:
    return sum(stage.wall_seconds for stage in stages if stage.name in names)


def _generation_records(
    profile: Mapping[str, Any], label: str, *, role: str
) -> list[dict[str, Any]]:
    generation = _list(profile.get("generation"), name=f"{role}.profile.generation")
    records = [
        _object(value, name=f"{role}.profile.generation[{index}]")
        for index, value in enumerate(generation)
        if isinstance(value, dict) and value.get("profile_label") == label
    ]
    if not records:
        raise CandidateAnalysisError(f"{role} contains no {label} generation records")
    return records


def _label_metrics(
    profile: Mapping[str, Any], label: str, *, role: str
) -> tuple[float, int, float, float]:
    records = _generation_records(profile, label, role=role)
    model_seconds = sum(
        _number(
            record.get("model_elapsed_seconds"),
            name=f"{role}.{label}.model_elapsed_seconds",
            positive=True,
        )
        for record in records
    )
    tokens = 0
    capture = 0.0
    for record in records:
        value = record.get("generated_tokens")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CandidateAnalysisError(
                f"{role}.{label}.generated_tokens must be a positive integer"
            )
        tokens += value
        capture += _number(
            record.get("decode_graph_capture_seconds_delta"),
            name=f"{role}.{label}.decode_graph_capture_seconds_delta",
        )
    return model_seconds, tokens, tokens / model_seconds, capture


def _validate_invariants(metadata: Mapping[str, Any], *, role: str) -> None:
    for key, expected in INVARIANT_METADATA.items():
        if metadata.get(key) != expected:
            raise CandidateAnalysisError(
                f"{role}.metadata.{key} must remain {expected!r}, got {metadata.get(key)!r}"
            )
    config = _object(
        metadata.get("optimized_effective_config"),
        name=f"{role}.metadata.optimized_effective_config",
    )
    for key, expected in INVARIANT_EFFECTIVE_CONFIG.items():
        if config.get(key) != expected:
            raise CandidateAnalysisError(
                f"{role}.optimized_effective_config.{key} must remain {expected!r}, "
                f"got {config.get(key)!r}"
            )


def _parse_run(
    role: str,
    profile_path: Path,
    osu_override: Path | None,
) -> ParsedRun:
    profile_path = profile_path.expanduser().resolve()
    profile = _load_profile(profile_path, role=role)
    metadata = _object(profile.get("metadata"), name=f"{role}.profile.metadata")
    if metadata.get("profile_pass_kind") != "untraced_control":
        raise CandidateAnalysisError(
            f"{role} must use profile_pass_kind=untraced_control"
        )
    if metadata.get("authoritative_performance") is not True:
        raise CandidateAnalysisError(f"{role} must set authoritative_performance=true")
    _validate_invariants(metadata, role=role)
    try:
        precision = _profile_precision(profile, path=profile_path)
    except Exception as exc:
        raise CandidateAnalysisError(str(exc)) from exc
    stages = _parse_stages(profile, role=role)
    osu_path = _resolve_osu_path(
        profile_path, metadata, osu_override, role=role
    )
    structure = summarize_osu_structure(osu_path)
    if not structure.get("finite_and_well_formed"):
        raise CandidateAnalysisError(f"{role} OSU output is malformed or non-finite")

    signatures: dict[str, dict[str, Any]] = {}
    label_values: dict[str, tuple[float, int, float, float]] = {}
    for label in PROFILE_LABELS:
        signature = _profile_label_signature(profile, label)
        if signature.get("status") != "available" or not signature.get("self_consistent"):
            raise CandidateAnalysisError(
                f"{role} {label} token/stopping signature is unavailable or inconsistent"
            )
        signatures[label] = signature
        label_values[label] = _label_metrics(profile, label, role=role)
        if signature["generated_tokens"] != label_values[label][1]:
            raise CandidateAnalysisError(
                f"{role} {label} signature token count disagrees with timing records"
            )

    graph_signature = _profile_graph_cache_signature(profile, PROFILE_LABELS)
    if graph_signature.get("status") != "available":
        raise CandidateAnalysisError(f"{role} dispatch/cache metadata is unavailable")

    timing_stage = _unique_stage(stages, TIMING_GENERATION_STAGES)
    main_stage = _unique_stage(stages, frozenset({"main_generation"}))
    compile_stage = _unique_stage(stages, frozenset({"compile_args"}))
    validate_stage = _unique_stage(stages, frozenset({"validate_inputs"}))
    final_stage = _unique_stage(stages, FINAL_WRITE_STAGES)
    cold_stage_sum = sum(stage.wall_seconds for stage in stages)
    complete_wall = final_stage.finished - validate_stage.started
    cold_outer_wall = final_stage.finished - compile_stage.started
    if complete_wall <= 0.0 or cold_outer_wall <= 0.0:
        raise CandidateAnalysisError(f"{role} request/cold wall must be positive")
    setup_stage_sum = _stage_sum(stages, SETUP_STAGES)
    capture_sum = sum(values[3] for values in label_values.values())
    timing_postprocess = _stage_sum(stages, TIMING_POSTPROCESS_STAGES)
    final_postprocess = _stage_sum(stages, FINAL_POSTPROCESS_STAGES)

    memory_records = [stage.raw for stage in stages]
    memory_records.extend(
        value
        for value in _list(profile.get("generation"), name=f"{role}.generation")
        if isinstance(value, dict)
    )
    peak_values = [
        _number(
            value["cuda_max_memory_allocated_mb"],
            name=f"{role}.cuda_max_memory_allocated_mb",
        )
        for value in memory_records
        if "cuda_max_memory_allocated_mb" in value
    ]
    current_values = [
        _number(
            value["cuda_memory_allocated_mb"],
            name=f"{role}.cuda_memory_allocated_mb",
        )
        for value in memory_records
        if "cuda_memory_allocated_mb" in value
    ]
    if not peak_values or not current_values:
        raise CandidateAnalysisError(f"{role} contains no VRAM measurements")

    timing_model, timing_tokens, timing_tps, _ = label_values["timing_context"]
    main_model, main_tokens, main_tps, _ = label_values["main_generation"]
    metrics = {
        "timing_model_seconds": timing_model,
        "timing_tokens": float(timing_tokens),
        "timing_tps": timing_tps,
        "timing_outer_stage_wall_seconds": timing_stage.wall_seconds,
        "timing_postprocess_seconds": timing_postprocess,
        "main_model_seconds": main_model,
        "main_tokens": float(main_tokens),
        "main_tps": main_tps,
        "main_outer_stage_wall_seconds": main_stage.wall_seconds,
        "final_postprocess_write_seconds": final_postprocess,
        "complete_request_wall_seconds": complete_wall,
        "cold_process_stage_sum_seconds": cold_stage_sum,
        "cold_process_outer_wall_seconds": cold_outer_wall,
        "setup_stage_sum_seconds": setup_stage_sum,
        "graph_capture_seconds": capture_sum,
        "setup_plus_capture_seconds": setup_stage_sum + capture_sum,
        "total_postprocess_seconds": timing_postprocess + final_postprocess,
        "peak_cuda_memory_allocated_mb": max(peak_values),
        "max_current_cuda_memory_allocated_mb": max(current_values),
    }
    if any(not math.isfinite(value) or value < 0.0 for value in metrics.values()):
        raise CandidateAnalysisError(f"{role} contains invalid computed metrics")

    workload = {
        key: metadata[key]
        for key in PROFILE_WORKLOAD_KEYS
        if key in metadata
    }
    missing_workload = [
        key
        for key in ("model_path", "audio_path", "seed", "precision", "inference_engine")
        if key not in workload
    ]
    if missing_workload:
        raise CandidateAnalysisError(
            f"{role} is missing workload metadata: {missing_workload}"
        )
    return ParsedRun(
        role=role,
        profile_path=profile_path,
        osu_path=osu_path,
        profile=profile,
        metadata=metadata,
        precision=precision,
        stages=stages,
        stage_sequence=tuple(stage.name for stage in stages),
        label_signatures=signatures,
        graph_signature=graph_signature,
        workload=workload,
        osu_sha256=_sha256_file(osu_path),
        osu_size_bytes=osu_path.stat().st_size,
        osu_structure=structure,
        metrics=metrics,
    )


def _flatten(value: Any, *, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        if not value:
            return {prefix: {}}
        flattened: dict[str, Any] = {}
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(value[key], prefix=child_prefix))
        return flattened
    if isinstance(value, list):
        if not value:
            return {prefix: []}
        flattened = {}
        for index, child in enumerate(value):
            flattened.update(_flatten(child, prefix=f"{prefix}[{index}]"))
        return flattened
    return {prefix: value}


def _material_differences(reference: Any, candidate: Any) -> list[dict[str, Any]]:
    left = _flatten(reference)
    right = _flatten(candidate)
    paths = sorted(set(left) | set(right))
    return [
        {
            "path": path,
            "baseline": left.get(path, {"missing": True}),
            "candidate": right.get(path, {"missing": True}),
        }
        for path in paths
        if path not in left or path not in right or left[path] != right[path]
    ]


def _validate_allowed_differences(
    differences: Sequence[dict[str, Any]], patterns: Sequence[str]
) -> dict[str, Any]:
    clean_patterns = tuple(dict.fromkeys(pattern.strip() for pattern in patterns if pattern.strip()))
    matched: dict[str, list[str]] = {pattern: [] for pattern in clean_patterns}
    undeclared: list[str] = []
    for difference in differences:
        path = str(difference["path"])
        matches = [
            pattern for pattern in clean_patterns if fnmatch.fnmatchcase(path, pattern)
        ]
        if not matches:
            undeclared.append(path)
        for pattern in matches:
            matched[pattern].append(path)
    unused = [pattern for pattern, paths in matched.items() if not paths]
    return {
        "declared_patterns": list(clean_patterns),
        "differences": list(differences),
        "matched_paths_by_pattern": matched,
        "undeclared_paths": undeclared,
        "unused_patterns": unused,
        "pass": not undeclared and not unused,
    }


def _require_equal(values: Mapping[str, Any], *, name: str) -> None:
    iterator = iter(values.items())
    first_role, first = next(iterator)
    for role, value in iterator:
        if value != first:
            raise CandidateAnalysisError(
                f"{name} differs between {first_role} and {role}"
            )


def _validate_workloads(runs: Mapping[str, ParsedRun], *, mode: str) -> dict[str, Any]:
    baseline = {role: runs[role].workload for role in BASELINE_ROLES}
    candidate = {role: runs[role].workload for role in CANDIDATE_ROLES}
    _require_equal(baseline, name="baseline workload")
    _require_equal(candidate, name="candidate workload")
    cross_differences = _material_differences(
        runs["baseline_first"].workload,
        runs["candidate_first"].workload,
    )
    allowed_paths = {"precision"} if mode == "relaxed" else set()
    undeclared = [
        difference["path"]
        for difference in cross_differences
        if difference["path"] not in allowed_paths
    ]
    if undeclared:
        raise CandidateAnalysisError(
            f"baseline/candidate workload differs outside precision: {undeclared}"
        )
    precisions = {run.precision for run in runs.values()}
    if mode == "exact-fp32" and precisions != {"fp32"}:
        raise CandidateAnalysisError("exact-fp32 mode requires all four profiles to be fp32")
    if mode == "exact-same-precision" and len(precisions) != 1:
        raise CandidateAnalysisError(
            "exact-same-precision mode requires all four profiles to use one precision"
        )
    return {
        "baseline_sha256": _sha256_json(runs["baseline_first"].workload),
        "candidate_sha256": _sha256_json(runs["candidate_first"].workload),
        "cross_differences": cross_differences,
        "precision_change_only": not undeclared,
    }


def _metric_direction(metric: str) -> str:
    return "higher_is_better" if metric.endswith("_tps") else "lower_is_better"


def _metric_summary(metric: str, runs: Mapping[str, ParsedRun]) -> dict[str, Any]:
    values = {role: runs[role].metrics[metric] for role in RUN_ORDER}
    baseline_values = [values[role] for role in BASELINE_ROLES]
    candidate_values = [values[role] for role in CANDIDATE_ROLES]
    baseline_median = statistics.median(baseline_values)
    candidate_median = statistics.median(candidate_values)
    delta = candidate_median - baseline_median
    direction = _metric_direction(metric)
    improvement = delta if direction == "higher_is_better" else -delta
    pair_roles = (
        ("baseline_then_candidate", "baseline_first", "candidate_first"),
        ("candidate_then_baseline", "baseline_second", "candidate_second"),
    )
    pairs = []
    for order, baseline_role, candidate_role in pair_roles:
        pair_delta = values[candidate_role] - values[baseline_role]
        pair_improvement = pair_delta if direction == "higher_is_better" else -pair_delta
        pairs.append(
            {
                "order": order,
                "baseline": values[baseline_role],
                "candidate": values[candidate_role],
                "candidate_minus_baseline": pair_delta,
                "improvement": pair_improvement,
                "improvement_fraction_of_baseline": (
                    pair_improvement / values[baseline_role]
                    if values[baseline_role] > 0.0
                    else None
                ),
            }
        )
    pair_improvements = [pair["improvement"] for pair in pairs]
    return {
        "direction": direction,
        "run_values": values,
        "baseline_median": baseline_median,
        "candidate_median": candidate_median,
        "candidate_minus_baseline": delta,
        "candidate_minus_baseline_fraction": (
            delta / baseline_median if baseline_median > 0.0 else None
        ),
        "improvement": improvement,
        "improvement_fraction_of_baseline": (
            improvement / baseline_median if baseline_median > 0.0 else None
        ),
        "reciprocal_pairs": pairs,
        "reciprocal_improvement_median": statistics.median(pair_improvements),
        "reciprocal_improvement_range": max(pair_improvements) - min(pair_improvements),
    }


def _token_groups(profile: Mapping[str, Any], label: str) -> list[list[int]]:
    groups: list[list[int]] = []
    for record in profile["generation"]:
        if not isinstance(record, dict) or record.get("profile_label") != label:
            continue
        per_sample = record.get("generated_token_ids_per_sample")
        if per_sample is not None:
            groups.extend([[int(token) for token in group] for group in per_sample])
        else:
            groups.append([int(token) for token in record["generated_token_ids"]])
    return groups


def _token_divergence(reference: ParsedRun, candidate: ParsedRun, label: str) -> dict[str, Any]:
    reference_groups = _token_groups(reference.profile, label)
    candidate_groups = _token_groups(candidate.profile, label)
    reference_flat = [token for group in reference_groups for token in group]
    candidate_flat = [token for group in candidate_groups for token in group]
    aligned = min(len(reference_flat), len(candidate_flat))
    mismatches = sum(
        left != right
        for left, right in zip(reference_flat[:aligned], candidate_flat[:aligned], strict=True)
    )
    return {
        "baseline_tokens": len(reference_flat),
        "candidate_tokens": len(candidate_flat),
        "token_count_delta": len(candidate_flat) - len(reference_flat),
        "aligned_tokens": aligned,
        "aligned_mismatches": mismatches,
        "aligned_mismatch_fraction": mismatches / aligned if aligned else 0.0,
        "record_count_delta": len(candidate_groups) - len(reference_groups),
        "token_stream_equal": reference_flat == candidate_flat,
        "stopping_equal": (
            reference.label_signatures[label]["stopping_sha256"]
            == candidate.label_signatures[label]["stopping_sha256"]
        ),
    }


def _structure_divergence(reference: ParsedRun, candidate: ParsedRun) -> dict[str, Any]:
    scalar_keys = (
        "timing_points",
        "uninherited_timing_points",
        "inherited_timing_points",
        "hit_objects",
    )
    scalar_deltas = {
        key: candidate.osu_structure[key] - reference.osu_structure[key]
        for key in scalar_keys
    }
    object_types = sorted(
        set(reference.osu_structure["hit_object_types"])
        | set(candidate.osu_structure["hit_object_types"])
    )
    return {
        "final_map_equal": reference.osu_sha256 == candidate.osu_sha256,
        "baseline_sha256": reference.osu_sha256,
        "candidate_sha256": candidate.osu_sha256,
        "scalar_deltas": scalar_deltas,
        "hit_object_type_deltas": {
            key: candidate.osu_structure["hit_object_types"].get(key, 0)
            - reference.osu_structure["hit_object_types"].get(key, 0)
            for key in object_types
        },
        "baseline_structure": reference.osu_structure,
        "candidate_structure": candidate.osu_structure,
    }


def _parity_and_divergence(
    runs: Mapping[str, ParsedRun],
    *,
    mode: str,
    allowed_dispatch_deltas: Sequence[str],
) -> dict[str, Any]:
    for label in PROFILE_LABELS:
        _require_equal(
            {
                role: runs[role].label_signatures[label]["token_stream_sha256"]
                for role in BASELINE_ROLES
            },
            name=f"baseline {label} token stream",
        )
        _require_equal(
            {
                role: runs[role].label_signatures[label]["token_stream_sha256"]
                for role in CANDIDATE_ROLES
            },
            name=f"candidate {label} token stream",
        )
        _require_equal(
            {
                role: runs[role].label_signatures[label]["stopping_sha256"]
                for role in BASELINE_ROLES
            },
            name=f"baseline {label} stopping",
        )
        _require_equal(
            {
                role: runs[role].label_signatures[label]["stopping_sha256"]
                for role in CANDIDATE_ROLES
            },
            name=f"candidate {label} stopping",
        )
    _require_equal(
        {role: runs[role].osu_sha256 for role in BASELINE_ROLES},
        name="baseline final OSU",
    )
    _require_equal(
        {role: runs[role].osu_sha256 for role in CANDIDATE_ROLES},
        name="candidate final OSU",
    )
    _require_equal(
        {role: runs[role].graph_signature["sha256"] for role in BASELINE_ROLES},
        name="baseline dispatch/cache metadata",
    )
    _require_equal(
        {role: runs[role].graph_signature["sha256"] for role in CANDIDATE_ROLES},
        name="candidate dispatch/cache metadata",
    )

    graph_differences = _material_differences(
        runs["baseline_first"].graph_signature["material"],
        runs["candidate_first"].graph_signature["material"],
    )
    dispatch_declaration = _validate_allowed_differences(
        graph_differences, allowed_dispatch_deltas
    )
    exact_mode = mode in {"exact-fp32", "exact-same-precision"}
    if exact_mode and not dispatch_declaration["pass"]:
        raise CandidateAnalysisError(
            "dispatch/cache metadata contains undeclared or unused expected deltas: "
            f"undeclared={dispatch_declaration['undeclared_paths']}, "
            f"unused={dispatch_declaration['unused_patterns']}"
        )

    token_divergence = {
        label: _token_divergence(
            runs["baseline_first"], runs["candidate_first"], label
        )
        for label in PROFILE_LABELS
    }
    output_divergence = _structure_divergence(
        runs["baseline_first"], runs["candidate_first"]
    )
    exact_cross_candidate = all(
        divergence["token_stream_equal"] and divergence["stopping_equal"]
        for divergence in token_divergence.values()
    ) and output_divergence["final_map_equal"]
    if exact_mode and not exact_cross_candidate:
        raise CandidateAnalysisError(
            f"{mode} candidate differs in tokens, stopping, or final OSU bytes"
        )
    return {
        "claim": mode if exact_mode else "relaxed-nonexact",
        "baseline_repeat_stable": True,
        "candidate_repeat_stable": True,
        "cross_candidate_exact": exact_cross_candidate,
        "dispatch_cache_topology": dispatch_declaration,
        "token_and_stopping_divergence": token_divergence,
        "output_divergence": output_divergence,
    }


def analyze(
    profile_paths: Mapping[str, Path],
    *,
    mode: str = "exact-fp32",
    osu_overrides: Mapping[str, Path | None] | None = None,
    allowed_dispatch_deltas: Sequence[str] = (),
) -> dict[str, Any]:
    if mode not in MODES:
        raise CandidateAnalysisError(f"mode must be one of {MODES}")
    if set(profile_paths) != set(RUN_ORDER):
        raise CandidateAnalysisError(f"profile roles must be exactly {list(RUN_ORDER)}")
    overrides = osu_overrides or {}
    runs = {
        role: _parse_run(role, profile_paths[role], overrides.get(role))
        for role in RUN_ORDER
    }
    sequences = {role: runs[role].stage_sequence for role in RUN_ORDER}
    _require_equal(sequences, name="stage sequence")
    workload = _validate_workloads(runs, mode=mode)
    parity = _parity_and_divergence(
        runs,
        mode=mode,
        allowed_dispatch_deltas=allowed_dispatch_deltas,
    )
    metric_names = tuple(runs[RUN_ORDER[0]].metrics)
    for role in RUN_ORDER[1:]:
        if tuple(runs[role].metrics) != metric_names:
            raise CandidateAnalysisError(f"{role} metric schema differs")
    metrics = {metric: _metric_summary(metric, runs) for metric in metric_names}
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "run_order": list(RUN_ORDER),
        "measurement_scope": {
            "authoritative_profiles_only": True,
            "timing_and_main_model_seconds": "sum of synchronized untraced model_elapsed_seconds",
            "outer_stage_wall": "CUDA-synchronized profiler stage wall",
            "complete_request_wall": "validate_inputs start through write_osu/write_osz finish",
            "cold_process_stage_sum": "sum of measured stage walls from compile_args through final write",
            "cold_process_outer_wall": "compile_args start through final write finish",
            "reciprocal_order": "baseline,candidate,candidate,baseline",
        },
        "workload": workload,
        "parity": parity,
        "runs": {
            role: {
                "profile_path": str(run.profile_path),
                "osu_path": str(run.osu_path),
                "precision": run.precision,
                "profile_sha256": _sha256_file(run.profile_path),
                "osu_sha256": run.osu_sha256,
                "osu_size_bytes": run.osu_size_bytes,
                "stage_sequence": list(run.stage_sequence),
                "metrics": run.metrics,
                "token_signatures": run.label_signatures,
                "dispatch_cache_signature_sha256": run.graph_signature["sha256"],
                "osu_structure": run.osu_structure,
            }
            for role, run in runs.items()
        },
        "metrics": metrics,
    }


def text_report(report: Mapping[str, Any]) -> str:
    parity = report["parity"]
    lines = [
        f"mode={report['mode']}",
        "run_order=baseline,candidate,candidate,baseline",
        f"parity.claim={parity['claim']}",
        f"parity.cross_candidate_exact={str(parity['cross_candidate_exact']).lower()}",
        "parity.baseline_repeat_stable=true",
        "parity.candidate_repeat_stable=true",
    ]
    topology = parity["dispatch_cache_topology"]
    lines.append(
        "dispatch.declared_patterns="
        + (",".join(topology["declared_patterns"]) or "none")
    )
    lines.append(f"dispatch.changed_paths={len(topology['differences'])}")
    lines.append(f"dispatch.declaration_pass={str(topology['pass']).lower()}")
    for metric, values in report["metrics"].items():
        lines.append(
            f"metric.{metric}=baseline_median:{values['baseline_median']:.9f},"
            f"candidate_median:{values['candidate_median']:.9f},"
            f"candidate_minus_baseline:{values['candidate_minus_baseline']:.9f},"
            f"improvement:{values['improvement']:.9f},"
            f"reciprocal_improvement_median:{values['reciprocal_improvement_median']:.9f},"
            f"reciprocal_improvement_range:{values['reciprocal_improvement_range']:.9f}"
        )
    for label, divergence in parity["token_and_stopping_divergence"].items():
        lines.append(
            f"divergence.{label}=baseline_tokens:{divergence['baseline_tokens']},"
            f"candidate_tokens:{divergence['candidate_tokens']},"
            f"aligned_mismatches:{divergence['aligned_mismatches']},"
            f"aligned_mismatch_fraction:{divergence['aligned_mismatch_fraction']:.9f},"
            f"stopping_equal:{str(divergence['stopping_equal']).lower()}"
        )
    output = parity["output_divergence"]
    lines.append(f"divergence.final_map_equal={str(output['final_map_equal']).lower()}")
    for key, delta in output["scalar_deltas"].items():
        lines.append(f"divergence.osu.{key}_delta={delta}")
    return "\n".join(lines) + "\n"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for role in RUN_ORDER:
        flag = role.replace("_", "-")
        parser.add_argument(f"--{flag}-profile", type=Path, required=True)
        parser.add_argument(f"--{flag}-osu", type=Path)
    parser.add_argument("--mode", choices=MODES, default="exact-fp32")
    parser.add_argument(
        "--allow-dispatch-delta",
        action="append",
        default=[],
        metavar="GLOB",
        help="repeatable fnmatch path for an intentional dispatch/cache topology delta",
    )
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    profile_paths = {
        role: getattr(args, f"{role}_profile") for role in RUN_ORDER
    }
    osu_overrides = {role: getattr(args, f"{role}_osu") for role in RUN_ORDER}
    report = analyze(
        profile_paths,
        mode=args.mode,
        osu_overrides=osu_overrides,
        allowed_dispatch_deltas=args.allow_dispatch_delta,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.text_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.text_output.write_text(text_report(report), encoding="utf-8")


if __name__ == "__main__":
    main()
