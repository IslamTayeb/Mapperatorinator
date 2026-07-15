"""Fail-loud analysis for five fresh-process strict-FP32 controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import nsight_agent_profile as nsight
except ImportError:  # Direct ``python utils/...py`` execution.
    import nsight_agent_profile as nsight


SCHEMA_VERSION = "mapperatorinator.fp32-fresh-baseline.v1"
RUN_NAMES = tuple(f"run-{index:02d}" for index in range(1, 6))
LABELS = ("timing_context", "main_generation")
STRICT_RUNTIME_METADATA = {
    "precision": "fp32",
    "inference_engine": "optimized",
    "attn_implementation": "sdpa",
    "profile_pass_kind": "untraced_control",
    "authoritative_performance": True,
    "strict_exactness_evidence": False,
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "nvidia_tf32_override": "0",
}
EXACTNESS_RUNTIME_METADATA = {
    **STRICT_RUNTIME_METADATA,
    "profile_pass_kind": "exactness_audit",
    "authoritative_performance": False,
    "strict_exactness_evidence": True,
}
RECORD_CONTRACT_KEYS = (
    "profile_label",
    "context_type",
    "mode",
    "sequence_index",
    "batch_start_index",
    "precision",
    "decoder_loop_backend",
    "torch_compile_enabled",
    "generation_compile_enabled",
    "optimized_effective_config_version",
    "optimized_batched_super_timing",
    "optimized_dispatch_mode",
    "optimized_dispatch_policy",
    "native_cross_mlp_tail_requested",
    "native_cross_mlp_tail_enabled",
    "native_cross_mlp_tail_disabled_reason",
    "optimized_dispatch_capture_hits",
    "decode_graph_count_before",
    "decode_graph_count_after",
    "decode_graph_count_delta",
    "decode_graph_replays_delta",
    "optimized_cuda_graphs",
    "graph_cache_entries",
    "graph_capture_count",
    "graph_replay_count",
    "decode_replays",
    "cache_summary",
    "graph_cache_summary",
    "cuda_graph_summary",
)


class BaselineError(RuntimeError):
    """The fresh-process evidence does not satisfy the baseline contract."""


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


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BaselineError(f"{name} must be an object")
    return value


def _finite_number(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        qualifier = "positive " if positive else "non-negative "
        raise BaselineError(f"{name} must be a {qualifier}finite number")
    return result


def _required_file(path: Path, *, allow_empty: bool = False) -> Path:
    if not path.is_file():
        raise BaselineError(f"required artifact is missing: {path}")
    if not allow_empty and path.stat().st_size <= 0:
        raise BaselineError(f"required artifact is empty: {path}")
    return path


def _path_from_pointer(run_dir: Path, name: str) -> Path:
    pointer = _required_file(run_dir / name)
    raw = pointer.read_text(encoding="utf-8").strip()
    if not raw:
        raise BaselineError(f"artifact pointer is empty: {pointer}")
    path = Path(raw).expanduser().resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise BaselineError(
            f"artifact pointer escapes its run directory: {pointer} -> {path}"
        ) from exc
    return _required_file(path)


def _workload_contract(profile: Mapping[str, Any]) -> dict[str, Any]:
    metadata = profile["metadata"]
    return {
        key: metadata[key]
        for key in nsight.PROFILE_WORKLOAD_KEYS
        if key in metadata
    }


def _preset_contract(profile: Mapping[str, Any]) -> dict[str, Any]:
    metadata = profile["metadata"]
    return {
        key: metadata[key]
        for key in nsight.PROFILE_PRESET_KEYS
        if key in metadata
    }


def _record_contract(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(profile["generation"]):
        record = _object(raw, name=f"generation[{index}]")
        if record.get("profile_label") not in LABELS:
            continue
        material = {
            key: (
                nsight._canonicalize_graph_cache_value(record[key])
                if key == "optimized_cuda_graphs"
                else record[key]
            )
            for key in RECORD_CONTRACT_KEYS
            if key in record
        }
        records.append(material)
    if not records:
        raise BaselineError("profile contains no timing/main generation records")
    return records


def _label_metrics(profile: Mapping[str, Any], label: str) -> dict[str, Any]:
    summary = _object(profile.get("summary"), name="profile.summary")
    by_label = _object(
        summary.get("generation_by_label"),
        name="profile.summary.generation_by_label",
    )
    values = _object(by_label.get(label), name=f"generation_by_label.{label}")
    model_seconds = _finite_number(
        values.get("model_elapsed_seconds"),
        name=f"{label}.model_elapsed_seconds",
        positive=True,
    )
    outer_seconds = _finite_number(
        values.get("wall_seconds"),
        name=f"{label}.wall_seconds",
        positive=True,
    )
    generated_tokens = values.get("generated_tokens")
    if isinstance(generated_tokens, bool) or not isinstance(generated_tokens, int):
        raise BaselineError(f"{label}.generated_tokens must be an integer")
    if generated_tokens <= 0:
        raise BaselineError(f"{label}.generated_tokens must be positive")
    return {
        "records": int(values.get("records", 0)),
        "generated_tokens": generated_tokens,
        "synchronized_model_seconds": model_seconds,
        "outer_wall_seconds": outer_seconds,
        "tokens_per_second": generated_tokens / model_seconds,
    }


def _peak_memory_mb(profile: Mapping[str, Any]) -> float:
    values = []
    for collection in (profile.get("stages", []), profile.get("generation", [])):
        if not isinstance(collection, list):
            raise BaselineError("profile stages/generation must be arrays")
        for raw in collection:
            if not isinstance(raw, dict):
                continue
            value = raw.get("cuda_max_memory_allocated_mb")
            if value is not None:
                values.append(
                    _finite_number(value, name="cuda_max_memory_allocated_mb")
                )
    if not values:
        raise BaselineError("profile does not expose peak CUDA memory")
    return max(values)


def _request_wall_seconds(profile: Mapping[str, Any]) -> float:
    stages = profile.get("stages")
    if not isinstance(stages, list) or not stages:
        raise BaselineError("profile contains no stage timing bounds")
    request_starts: list[float] = []
    output_finishes: list[float] = []
    for index, raw in enumerate(stages):
        stage = _object(raw, name=f"profile.stages[{index}]")
        started = _finite_number(
            stage.get("started_at_perf_counter_seconds"),
            name=f"profile.stages[{index}].started_at_perf_counter_seconds",
        )
        finished = _finite_number(
            stage.get("finished_at_perf_counter_seconds"),
            name=f"profile.stages[{index}].finished_at_perf_counter_seconds",
        )
        if finished < started:
            raise BaselineError(f"profile stage {index} finishes before it starts")
        if stage.get("name") == "validate_inputs":
            request_starts.append(started)
        if stage.get("name") in {"write_osu", "write_osz"}:
            output_finishes.append(finished)
    if len(request_starts) != 1:
        raise BaselineError(
            "request wall requires exactly one validate_inputs stage, got "
            f"{len(request_starts)}"
        )
    if not output_finishes:
        raise BaselineError("request wall requires write_osu or write_osz")
    wall = max(output_finishes) - request_starts[0]
    if wall <= 0:
        raise BaselineError("request-to-final-osu wall must be positive")
    return wall


def _wall_artifact(run_dir: Path) -> dict[str, Any]:
    path = _required_file(run_dir / "process-wall.json")
    payload = _object(json.loads(path.read_text(encoding="utf-8")), name=str(path))
    required = {"started_unix_ns", "finished_unix_ns", "elapsed_ns", "exit_code"}
    if set(payload) != required:
        raise BaselineError(
            f"{path} must contain exactly {sorted(required)}, got {sorted(payload)}"
        )
    for key in ("started_unix_ns", "finished_unix_ns", "elapsed_ns", "exit_code"):
        if isinstance(payload[key], bool) or not isinstance(payload[key], int):
            raise BaselineError(f"{path}:{key} must be an integer")
    if payload["exit_code"] != 0:
        raise BaselineError(f"fresh process failed for {run_dir.name}")
    expected = payload["finished_unix_ns"] - payload["started_unix_ns"]
    if payload["elapsed_ns"] != expected or expected <= 0:
        raise BaselineError(f"invalid process wall interval for {run_dir.name}")
    return {**payload, "elapsed_seconds": expected / 1_000_000_000}


def _runtime_contract(
    profile: Mapping[str, Any],
    *,
    expected_commit: str,
    expected_branch: str,
    exactness_audit: bool = False,
) -> dict[str, Any]:
    metadata = profile["metadata"]
    expected = {
        **(
            EXACTNESS_RUNTIME_METADATA
            if exactness_audit
            else STRICT_RUNTIME_METADATA
        ),
        "git_commit": expected_commit,
        "git_branch": expected_branch,
    }
    failures = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if failures:
        raise BaselineError(f"strict FP32 runtime metadata mismatch: {failures}")
    if "2080 Ti" not in str(metadata.get("cuda_device_name", "")):
        raise BaselineError(
            f"expected RTX 2080 Ti metadata, got {metadata.get('cuda_device_name')!r}"
        )
    if metadata.get("cuda_device_capability") != [7, 5]:
        raise BaselineError(
            "strict baseline requires SM75 capability [7, 5], got "
            f"{metadata.get('cuda_device_capability')!r}"
        )
    effective = _object(
        metadata.get("optimized_effective_config"),
        name="metadata.optimized_effective_config",
    )
    if effective.get("precision") != "fp32":
        raise BaselineError("optimized effective config is not FP32")
    generation_precisions = {
        record.get("precision")
        for record in profile["generation"]
        if isinstance(record, dict)
    }
    if generation_precisions != {"fp32"}:
        raise BaselineError(
            f"generation records are not uniformly FP32: {generation_precisions}"
        )
    return expected


def _hex_digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise BaselineError(f"{name} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise BaselineError(f"{name} must be hexadecimal") from exc
    return value


def _strict_exactness_contract(profile: Mapping[str, Any]) -> dict[str, Any]:
    by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
    for index, raw in enumerate(profile["generation"]):
        record = _object(raw, name=f"generation[{index}]")
        label = record.get("profile_label")
        if label not in LABELS:
            continue
        evidence = _object(
            record.get("strict_exactness"),
            name=f"generation[{index}].strict_exactness",
        )
        if evidence.get("schema_version") != 1:
            raise BaselineError(f"generation[{index}] exactness schema changed")
        if evidence.get("timing_class") != "non_authoritative_outside_model_elapsed":
            raise BaselineError(f"generation[{index}] exactness timing class changed")
        rng_material = {}
        for boundary in ("rng_before", "rng_after"):
            rng = _object(
                evidence.get(boundary),
                name=f"generation[{index}].strict_exactness.{boundary}",
            )
            if rng.get("cuda_device") != "cuda:0":
                raise BaselineError(
                    f"generation[{index}] {boundary} did not audit CUDA device 0"
                )
            rng_material[boundary] = {
                "cpu_sha256": _hex_digest(
                    rng.get("cpu_sha256"),
                    name=f"generation[{index}].{boundary}.cpu_sha256",
                ),
                "cuda_device": rng["cuda_device"],
                "cuda_sha256": _hex_digest(
                    rng.get("cuda_sha256"),
                    name=f"generation[{index}].{boundary}.cuda_sha256",
                ),
            }
        cache = _object(
            evidence.get("cache_writes"),
            name=f"generation[{index}].strict_exactness.cache_writes",
        )
        if cache.get("schema_version") != 1:
            raise BaselineError(f"generation[{index}] cache-write schema changed")
        cross_sequence_length = cache.get("cross_sequence_length")
        if (
            isinstance(cross_sequence_length, bool)
            or not isinstance(cross_sequence_length, int)
            or cross_sequence_length <= 0
        ):
            raise BaselineError(
                f"generation[{index}] cross cache sequence length must be positive"
            )
        layer_count = cache.get("layer_count")
        layers = cache.get("layers")
        if layer_count != 12 or not isinstance(layers, list) or len(layers) != layer_count:
            raise BaselineError(
                f"generation[{index}] cache evidence must cover all 12 layers"
            )
        output_counts = record.get("output_tokens_per_sample")
        expected_sequence_length = (
            output_counts[0] - 1
            if isinstance(output_counts, list)
            and len(output_counts) == 1
            and isinstance(output_counts[0], int)
            and not isinstance(output_counts[0], bool)
            else None
        )
        if (
            expected_sequence_length is None
            or cache.get("self_sequence_length") != expected_sequence_length
        ):
            raise BaselineError(
                f"generation[{index}] cache sequence length does not match output tokens"
            )
        cache_layers = []
        for layer_index, raw_layer in enumerate(layers):
            layer = _object(
                raw_layer,
                name=f"generation[{index}].cache_writes.layers[{layer_index}]",
            )
            if layer.get("layer_idx") != layer_index:
                raise BaselineError(
                    f"generation[{index}] cache layer ordering changed"
                )
            cache_layers.append(
                {
                    "layer_idx": layer_index,
                    **{
                        key: _hex_digest(
                            layer.get(key),
                            name=f"generation[{index}].cache.{layer_index}.{key}",
                        )
                        for key in (
                            "self_keys",
                            "self_values",
                            "cross_keys",
                            "cross_values",
                        )
                    },
                }
            )
        by_label[str(label)].append(
            {
                "record_key": nsight._profile_record_key(record, index),
                **rng_material,
                "cache_writes": {
                    "self_sequence_length": cache["self_sequence_length"],
                    "cross_sequence_length": cross_sequence_length,
                    "layer_count": layer_count,
                    "aggregate_sha256": _hex_digest(
                        cache.get("aggregate_sha256"),
                        name=f"generation[{index}].cache.aggregate_sha256",
                    ),
                    "layers": cache_layers,
                },
            }
        )
    if any(not records for records in by_label.values()):
        raise BaselineError("exactness audit is missing timing or main evidence")
    return by_label


def _same(name: str, values: Sequence[Any]) -> dict[str, Any]:
    hashes = [_sha256_bytes(_canonical_json(value)) for value in values]
    passed = len(set(hashes)) == 1
    return {
        "name": name,
        "pass": passed,
        "sha256_by_run": dict(zip(RUN_NAMES, hashes, strict=True)),
    }


def _aggregate(values: Sequence[float]) -> dict[str, float]:
    if len(values) != len(RUN_NAMES):
        raise BaselineError("aggregate requires exactly five values")
    return {
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
        "range": float(max(values) - min(values)),
    }


def analyze(
    run_root: Path,
    *,
    expected_commit: str,
    expected_branch: str,
    expected_main_tokens: int = 8294,
    expected_timing_tokens: int = 821,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    if len(expected_commit) != 40:
        raise BaselineError("expected commit must be a full 40-character SHA")
    if expected_main_tokens <= 0 or expected_timing_tokens <= 0:
        raise BaselineError("expected token counts must be positive")

    runs: list[dict[str, Any]] = []
    output_hashes: list[str] = []
    output_structures: list[dict[str, Any]] = []
    workload_contracts: list[dict[str, Any]] = []
    preset_contracts: list[dict[str, Any]] = []
    record_contracts: list[list[dict[str, Any]]] = []
    label_signatures: dict[str, list[dict[str, Any]]] = {
        label: [] for label in LABELS
    }

    for run_name in RUN_NAMES:
        run_dir = run_root / run_name
        if not run_dir.is_dir():
            raise BaselineError(f"missing fresh-process run directory: {run_dir}")
        for artifact, allow_empty in (
            ("command.txt", False),
            ("stdout.txt", False),
            ("stderr.txt", True),
            ("profile-path.txt", False),
            ("result-path.txt", False),
        ):
            _required_file(run_dir / artifact, allow_empty=allow_empty)
        profile_path = _path_from_pointer(run_dir, "profile-path.txt")
        result_path = _path_from_pointer(run_dir, "result-path.txt")
        profile = nsight._load_inference_profile(profile_path)
        _runtime_contract(
            profile,
            expected_commit=expected_commit,
            expected_branch=expected_branch,
        )
        wall = _wall_artifact(run_dir)
        output_hash = _sha256_file(result_path)
        metadata = profile["metadata"]
        if metadata.get("result_file_sha256") != output_hash:
            raise BaselineError(f"{run_name} profile/output SHA-256 mismatch")
        if metadata.get("result_file_size_bytes") != result_path.stat().st_size:
            raise BaselineError(f"{run_name} profile/output size mismatch")
        structure = nsight.summarize_osu_structure(result_path)
        if structure["malformed_by_section"] != {"TimingPoints": 0, "HitObjects": 0}:
            raise BaselineError(f"{run_name} result has malformed map rows")
        if structure["nonfinite_values"] != 0:
            raise BaselineError(f"{run_name} result contains non-finite values")

        metrics = {label: _label_metrics(profile, label) for label in LABELS}
        if metrics["main_generation"]["generated_tokens"] != expected_main_tokens:
            raise BaselineError(
                f"{run_name} main token count changed: "
                f"{metrics['main_generation']['generated_tokens']} != {expected_main_tokens}"
            )
        if metrics["timing_context"]["generated_tokens"] != expected_timing_tokens:
            raise BaselineError(
                f"{run_name} timing token count changed: "
                f"{metrics['timing_context']['generated_tokens']} != {expected_timing_tokens}"
            )
        signatures = {
            label: nsight._profile_label_signature(profile, label)
            for label in LABELS
        }
        for label, signature in signatures.items():
            if signature["status"] != "available" or not signature["self_consistent"]:
                raise BaselineError(
                    f"{run_name} {label} token/stopping signature is unavailable or inconsistent"
                )
            label_signatures[label].append(signature)
        graph_signature = nsight._profile_graph_cache_signature(profile, LABELS)
        if graph_signature["status"] != "available":
            raise BaselineError(f"{run_name} graph signature is unavailable")

        workload = _workload_contract(profile)
        preset = _preset_contract(profile)
        records = _record_contract(profile)
        workload_contracts.append(workload)
        preset_contracts.append(preset)
        record_contracts.append(records)
        output_hashes.append(output_hash)
        output_structures.append(structure)
        stage_wall = _object(
            _object(profile.get("summary"), name="profile.summary").get(
                "stage_wall_seconds"
            ),
            name="profile.summary.stage_wall_seconds",
        )
        runs.append(
            {
                "run_id": run_name,
                "profile_path": str(profile_path),
                "result_path": str(result_path),
                "process_wall_seconds": wall["elapsed_seconds"],
                "request_to_final_osu_wall_seconds": _request_wall_seconds(profile),
                "stages": {key: float(value) for key, value in stage_wall.items()},
                "generation": metrics,
                "peak_cuda_memory_mb": _peak_memory_mb(profile),
                "output_sha256": output_hash,
                "output_size_bytes": result_path.stat().st_size,
                "output_structure": structure,
                "graph_signature_sha256": graph_signature["sha256"],
                "dispatch_and_graph_contract_sha256": _sha256_bytes(
                    _canonical_json(records)
                ),
                "token_signatures": {
                    label: {
                        "token_stream_sha256": signature["token_stream_sha256"],
                        "stopping_sha256": signature["stopping_sha256"],
                        "generated_tokens": signature["generated_tokens"],
                    }
                    for label, signature in signatures.items()
                },
            }
        )

    exactness_dir = run_root / "exactness-audit"
    if not exactness_dir.is_dir():
        raise BaselineError(f"missing separate exactness audit: {exactness_dir}")
    for artifact, allow_empty in (
        ("command.txt", False),
        ("stdout.txt", False),
        ("stderr.txt", True),
        ("profile-path.txt", False),
        ("result-path.txt", False),
    ):
        _required_file(exactness_dir / artifact, allow_empty=allow_empty)
    exactness_profile_path = _path_from_pointer(
        exactness_dir, "profile-path.txt"
    )
    exactness_result_path = _path_from_pointer(
        exactness_dir, "result-path.txt"
    )
    exactness_profile = nsight._load_inference_profile(exactness_profile_path)
    _runtime_contract(
        exactness_profile,
        expected_commit=expected_commit,
        expected_branch=expected_branch,
        exactness_audit=True,
    )
    exactness_process_wall = _wall_artifact(exactness_dir)
    exactness_output_hash = _sha256_file(exactness_result_path)
    exactness_metadata = exactness_profile["metadata"]
    if exactness_metadata.get("result_file_sha256") != exactness_output_hash:
        raise BaselineError("exactness-audit profile/output SHA-256 mismatch")
    exactness_structure = nsight.summarize_osu_structure(exactness_result_path)
    exactness_label_signatures = {
        label: nsight._profile_label_signature(exactness_profile, label)
        for label in LABELS
    }
    for label, signature in exactness_label_signatures.items():
        if signature["status"] != "available" or not signature["self_consistent"]:
            raise BaselineError(
                f"exactness audit {label} token/stopping signature is unavailable"
            )
    exactness_graph = nsight._profile_graph_cache_signature(
        exactness_profile, LABELS
    )
    if exactness_graph["status"] != "available":
        raise BaselineError("exactness audit graph signature is unavailable")
    exactness_record_contract = _record_contract(exactness_profile)
    exactness_evidence = _strict_exactness_contract(exactness_profile)

    checks = {
        "workload_metadata": _same("workload_metadata", workload_contracts),
        "preset_metadata": _same("preset_metadata", preset_contracts),
        "dispatch_and_graph_contract": _same(
            "dispatch_and_graph_contract", record_contracts
        ),
        "output_bytes": {
            "name": "output_bytes",
            "pass": len(set(output_hashes)) == 1,
            "sha256_by_run": dict(zip(RUN_NAMES, output_hashes, strict=True)),
        },
        "output_structure": _same("output_structure", output_structures),
        "exactness_audit_workload": {
            "name": "exactness_audit_workload",
            "pass": _workload_contract(exactness_profile) == workload_contracts[0],
        },
        "exactness_audit_preset": {
            "name": "exactness_audit_preset",
            "pass": _preset_contract(exactness_profile) == preset_contracts[0],
        },
        "exactness_audit_dispatch_and_graph": {
            "name": "exactness_audit_dispatch_and_graph",
            "pass": exactness_record_contract == record_contracts[0],
        },
        "exactness_audit_output_bytes": {
            "name": "exactness_audit_output_bytes",
            "pass": exactness_output_hash == output_hashes[0],
        },
        "exactness_audit_output_structure": {
            "name": "exactness_audit_output_structure",
            "pass": exactness_structure == output_structures[0],
        },
    }
    for label in LABELS:
        checks[f"{label}_tokens"] = _same(
            f"{label}_tokens",
            [signature["token_stream_sha256"] for signature in label_signatures[label]],
        )
        checks[f"{label}_stopping"] = _same(
            f"{label}_stopping",
            [signature["stopping_sha256"] for signature in label_signatures[label]],
        )
        checks[f"exactness_audit_{label}_tokens"] = {
            "name": f"exactness_audit_{label}_tokens",
            "pass": (
                exactness_label_signatures[label]["token_stream_sha256"]
                == label_signatures[label][0]["token_stream_sha256"]
            ),
        }
        checks[f"exactness_audit_{label}_stopping"] = {
            "name": f"exactness_audit_{label}_stopping",
            "pass": (
                exactness_label_signatures[label]["stopping_sha256"]
                == label_signatures[label][0]["stopping_sha256"]
            ),
        }
    failed = [name for name, check in checks.items() if not check["pass"]]
    if failed:
        raise BaselineError(f"fresh FP32 exactness checks failed: {failed}")

    aggregates = {
        "process_wall_seconds": _aggregate(
            [run["process_wall_seconds"] for run in runs]
        ),
        "request_to_final_osu_wall_seconds": _aggregate(
            [run["request_to_final_osu_wall_seconds"] for run in runs]
        ),
        "peak_cuda_memory_mb": _aggregate(
            [run["peak_cuda_memory_mb"] for run in runs]
        ),
        "timing_synchronized_model_seconds": _aggregate(
            [
                run["generation"]["timing_context"]["synchronized_model_seconds"]
                for run in runs
            ]
        ),
        "timing_outer_wall_seconds": _aggregate(
            [
                run["generation"]["timing_context"]["outer_wall_seconds"]
                for run in runs
            ]
        ),
        "timing_tokens_per_second": _aggregate(
            [
                run["generation"]["timing_context"]["tokens_per_second"]
                for run in runs
            ]
        ),
        "main_synchronized_model_seconds": _aggregate(
            [
                run["generation"]["main_generation"]["synchronized_model_seconds"]
                for run in runs
            ]
        ),
        "main_outer_wall_seconds": _aggregate(
            [
                run["generation"]["main_generation"]["outer_wall_seconds"]
                for run in runs
            ]
        ),
        "main_tokens_per_second": _aggregate(
            [
                run["generation"]["main_generation"]["tokens_per_second"]
                for run in runs
            ]
        ),
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "authoritative_performance": True,
        "run_count": len(runs),
        "fresh_processes": True,
        "precision_contract": {
            **STRICT_RUNTIME_METADATA,
            "cuda_device_capability": [7, 5],
            "cuda_device_name_contains": "2080 Ti",
            "expected_commit": expected_commit,
            "expected_branch": expected_branch,
        },
        "workload_contract": {
            "expected_main_tokens": expected_main_tokens,
            "expected_timing_tokens": expected_timing_tokens,
            "seed": 12345,
            "config_name": "profile_salvalai",
        },
        "checks": checks,
        "runs": runs,
        "exactness_audit": {
            "authoritative_performance": False,
            "profile_path": str(exactness_profile_path),
            "result_path": str(exactness_result_path),
            "process_wall_seconds": exactness_process_wall["elapsed_seconds"],
            "request_to_final_osu_wall_seconds": _request_wall_seconds(
                exactness_profile
            ),
            "output_sha256": exactness_output_hash,
            "output_structure": exactness_structure,
            "graph_signature_sha256": exactness_graph["sha256"],
            "strict_exactness": exactness_evidence,
        },
        "aggregates": aggregates,
    }
    result["analysis_sha256"] = _sha256_bytes(_canonical_json(result))
    return result


def _text_report(result: Mapping[str, Any]) -> str:
    aggregates = result["aggregates"]
    lines = [
        f"status={result['status']}",
        f"run_count={result['run_count']}",
        "precision=fp32",
        "tf32=disabled",
        f"process_wall_median_seconds={aggregates['process_wall_seconds']['median']:.6f}",
        "request_to_final_osu_wall_median_seconds="
        f"{aggregates['request_to_final_osu_wall_seconds']['median']:.6f}",
        "timing_model_median_seconds="
        f"{aggregates['timing_synchronized_model_seconds']['median']:.6f}",
        "timing_tps_median="
        f"{aggregates['timing_tokens_per_second']['median']:.6f}",
        "main_model_median_seconds="
        f"{aggregates['main_synchronized_model_seconds']['median']:.6f}",
        f"main_tps_median={aggregates['main_tokens_per_second']['median']:.6f}",
        f"peak_cuda_memory_median_mb={aggregates['peak_cuda_memory_mb']['median']:.3f}",
    ]
    for run in result["runs"]:
        lines.append(
            f"{run['run_id']} process_wall={run['process_wall_seconds']:.6f} "
            f"timing_model={run['generation']['timing_context']['synchronized_model_seconds']:.6f} "
            f"main_model={run['generation']['main_generation']['synchronized_model_seconds']:.6f} "
            f"main_tps={run['generation']['main_generation']['tokens_per_second']:.6f}"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-main-tokens", type=int, default=8294)
    parser.add_argument("--expected-timing-tokens", type=int, default=821)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path)
    cli = parser.parse_args(argv)
    result = analyze(
        cli.run_root,
        expected_commit=cli.expected_commit,
        expected_branch=cli.expected_branch,
        expected_main_tokens=cli.expected_main_tokens,
        expected_timing_tokens=cli.expected_timing_tokens,
    )
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if cli.text_output is not None:
        cli.text_output.parent.mkdir(parents=True, exist_ok=True)
        cli.text_output.write_text(_text_report(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
