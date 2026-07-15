"""Verify the exact incremental FP32 timing-native-self candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from utils import nsight_agent_profile as nsight
from utils.analyze_fp32_fresh_baseline import _strict_exactness_contract


LABELS = ("timing_context", "main_generation")
VERSION = "strict-fp32-timing-native-self-v1"


class TimingNativeSelfError(RuntimeError):
    """Candidate artifacts do not satisfy the exact strict-FP32 contract."""


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TimingNativeSelfError(f"{name} must be an object")
    return value


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), name=name)
    except (OSError, json.JSONDecodeError) as exc:
        raise TimingNativeSelfError(f"{name} is missing or invalid: {path}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise TimingNativeSelfError(f"result artifact is missing: {path}") from exc
    return digest.hexdigest()


def _strict_runtime(profile: Mapping[str, Any], *, name: str) -> None:
    metadata = _object(profile.get("metadata"), name=f"{name}.metadata")
    expected = {
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "inference_engine": "optimized",
        "use_server": False,
        "parallel": False,
        "num_beams": 1,
        "cfg_scale": 1.0,
        "profile_pass_kind": "exactness_audit",
        "authoritative_performance": False,
        "strict_exactness_evidence": True,
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "nvidia_tf32_override": "0",
        "cuda_device_capability": [7, 5],
    }
    failures = {
        key: {"expected": expected_value, "actual": metadata.get(key)}
        for key, expected_value in expected.items()
        if metadata.get(key) != expected_value
    }
    if "2080 Ti" not in str(metadata.get("cuda_device_name", "")):
        failures["cuda_device_name"] = {
            "expected": "contains 2080 Ti",
            "actual": metadata.get("cuda_device_name"),
        }
    effective = _object(
        metadata.get("optimized_effective_config"),
        name=f"{name}.metadata.optimized_effective_config",
    )
    if effective.get("precision") != "fp32":
        failures["optimized_effective_config.precision"] = {
            "expected": "fp32",
            "actual": effective.get("precision"),
        }
    if failures:
        raise TimingNativeSelfError(f"{name} strict runtime mismatch: {failures}")


def _records(profile: Mapping[str, Any], label: str, *, name: str) -> list[dict[str, Any]]:
    generation = profile.get("generation")
    if not isinstance(generation, list):
        raise TimingNativeSelfError(f"{name}.generation must be a list")
    records = [
        _object(row, name=f"{name}.generation[{index}]")
        for index, row in enumerate(generation)
        if isinstance(row, dict) and row.get("profile_label") == label
    ]
    if not records:
        raise TimingNativeSelfError(f"{name} contains no {label} records")
    return records


def _count(record: Mapping[str, Any], key: str, *, name: str) -> int:
    hits = _object(
        record.get("optimized_dispatch_capture_hits"),
        name=f"{name}.optimized_dispatch_capture_hits",
    )
    value = hits.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TimingNativeSelfError(f"{name}.{key} must be a non-negative integer")
    return value


def _candidate_metadata(value: Any, *, name: str) -> dict[str, Any]:
    metadata = _object(value, name=name)
    expected = {
        "version": VERSION,
        "scope": "standalone-timing-model-batch1-only",
        "precision": "fp32",
        "torch_dtype": "torch.float32",
        "tf32_disabled_required": True,
        "result_class": "exact-incremental-candidate",
        "exactness_claim": True,
        "native_q1_self_attention": True,
        "native_q1_rope_cache_self_attention": True,
        "q1_bmm_cross_attention_retained": True,
        "native_cross_mlp_tail_retained_disabled": True,
        "original_decoder_forward_retained": True,
        "accepted_kernel_implementation_reused": True,
        "owns_model_weights": False,
        "reduced_precision_weights": False,
        "reduced_precision_activations": False,
        "counter_rng": False,
        "production_selector_unchanged": True,
    }
    failures = {
        key: {"expected": expected_value, "actual": metadata.get(key)}
        for key, expected_value in expected.items()
        if metadata.get(key) != expected_value
    }
    if failures:
        raise TimingNativeSelfError(f"{name} mismatch: {failures}")
    return metadata


def _initialization(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        raise TimingNativeSelfError("candidate initialization schema changed")
    if payload.get("runner_version") != "strict-fp32-timing-native-self-runner-v1":
        raise TimingNativeSelfError("candidate initialization runner changed")
    loads = _object(payload.get("model_loads"), name="initialization.model_loads")
    expected_loads = {
        "count": 2,
        "main_auto_select_gamemode_model": True,
        "timing_auto_select_gamemode_model": False,
        "owners_distinct": True,
    }
    if loads != expected_loads:
        raise TimingNativeSelfError(
            f"initialization model ownership mismatch: {loads}"
        )
    strict = _object(payload.get("strict_fp32"), name="initialization.strict_fp32")
    expected_strict = {
        "NVIDIA_TF32_OVERRIDE": "0",
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    if strict != expected_strict:
        raise TimingNativeSelfError(
            f"initialization strict FP32 contract mismatch: {strict}"
        )
    return _candidate_metadata(
        payload.get("timing_native_self"),
        name="initialization.timing_native_self",
    )


def _dispatch_contract(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    totals: dict[str, dict[str, int]] = {}
    for label in LABELS:
        control_records = _records(control, label, name="control")
        candidate_records = _records(candidate, label, name="candidate")
        if len(control_records) != len(candidate_records):
            raise TimingNativeSelfError(f"{label} record count changed")
        totals[label] = {
            "control_native_self": 0,
            "candidate_native_self": 0,
            "control_q1_bmm_cross": 0,
            "candidate_q1_bmm_cross": 0,
            "control_native_cross_mlp": 0,
            "candidate_native_cross_mlp": 0,
        }
        for index, (control_row, candidate_row) in enumerate(
            zip(control_records, candidate_records, strict=True)
        ):
            row_name = f"{label}[{index}]"
            for row_role, row in (("control", control_row), ("candidate", candidate_row)):
                if row.get("precision") != "fp32":
                    raise TimingNativeSelfError(f"{row_role}.{row_name} is not FP32")
            if (
                control_row.get("optimized_effective_config_version")
                != candidate_row.get("optimized_effective_config_version")
            ):
                raise TimingNativeSelfError(f"{row_name} accepted preset changed")
            for key, total_key in (
                ("native_q1_rope_cache_self_attention", "native_self"),
                ("q1_bmm_cross_attention", "q1_bmm_cross"),
                ("native_cross_mlp_tail", "native_cross_mlp"),
            ):
                totals[label][f"control_{total_key}"] += _count(
                    control_row, key, name=f"control.{row_name}"
                )
                totals[label][f"candidate_{total_key}"] += _count(
                    candidate_row, key, name=f"candidate.{row_name}"
                )

            control_policy = _object(
                control_row.get("optimized_dispatch_policy"),
                name=f"control.{row_name}.optimized_dispatch_policy",
            )
            candidate_policy = _object(
                candidate_row.get("optimized_dispatch_policy"),
                name=f"candidate.{row_name}.optimized_dispatch_policy",
            )
            if label == "timing_context":
                if control_row.get("optimized_dispatch_mode") != "accepted_batch1":
                    raise TimingNativeSelfError("control timing dispatch mode changed")
                if (
                    candidate_row.get("optimized_dispatch_mode")
                    != "strict_fp32_timing_native_self_batch1"
                ):
                    raise TimingNativeSelfError("candidate timing dispatch mode missing")
                if control_policy.get("q1_bmm_cross_attention", {}).get("enabled") is not True:
                    raise TimingNativeSelfError("control timing lost q1 BMM cross")
                if candidate_policy.get("q1_bmm_cross_attention", {}).get("enabled") is not True:
                    raise TimingNativeSelfError("candidate timing lost q1 BMM cross")
                for key in (
                    "native_q1_self_attention",
                    "native_q1_rope_cache_self_attention",
                ):
                    if control_policy.get(key, {}).get("enabled") is not False:
                        raise TimingNativeSelfError(f"control timing unexpectedly enabled {key}")
                    if candidate_policy.get(key, {}).get("enabled") is not True:
                        raise TimingNativeSelfError(f"candidate timing failed to enable {key}")
                for policy in (control_policy, candidate_policy):
                    if policy.get("native_cross_mlp_tail", {}).get("enabled") is not False:
                        raise TimingNativeSelfError("timing enabled native cross/MLP")
                expected_delta = {
                    "requested": True,
                    "enabled": True,
                    "disabled_reason": None,
                    "result_class": "exact-incremental-candidate",
                    "exactness_claim": True,
                }
                if candidate_policy.get("strict_fp32_timing_native_self") != expected_delta:
                    raise TimingNativeSelfError("candidate timing policy evidence changed")
                _candidate_metadata(
                    candidate_row.get("optimized_strict_fp32_timing_native_self"),
                    name=f"candidate.{row_name}.metadata",
                )
                if (
                    "strict_fp32_timing_native_self" in control_policy
                    or "optimized_strict_fp32_timing_native_self" in control_row
                ):
                    raise TimingNativeSelfError("control contains candidate evidence")
            else:
                if control_row.get("optimized_dispatch_mode") != candidate_row.get(
                    "optimized_dispatch_mode"
                ):
                    raise TimingNativeSelfError("main dispatch mode changed")
                if control_policy != candidate_policy:
                    raise TimingNativeSelfError("main dispatch policy changed")
                if control_row.get("optimized_dispatch_capture_hits") != candidate_row.get(
                    "optimized_dispatch_capture_hits"
                ):
                    raise TimingNativeSelfError("main dispatch hits changed")
                if (
                    "strict_fp32_timing_native_self" in candidate_policy
                    or "optimized_strict_fp32_timing_native_self" in candidate_row
                ):
                    raise TimingNativeSelfError("candidate policy leaked into main")

    timing = totals["timing_context"]
    if timing["control_native_self"] != 0 or timing["candidate_native_self"] <= 0:
        raise TimingNativeSelfError("timing native-self dispatch delta was not observed")
    if timing["control_q1_bmm_cross"] <= 0 or timing["candidate_q1_bmm_cross"] <= 0:
        raise TimingNativeSelfError("timing q1 BMM cross was not retained")
    if timing["control_native_cross_mlp"] or timing["candidate_native_cross_mlp"]:
        raise TimingNativeSelfError("timing native cross/MLP unexpectedly executed")
    return totals


def _label_model_seconds(profile: Mapping[str, Any], label: str) -> float:
    return sum(
        float(record.get("model_elapsed_seconds", 0.0) or 0.0)
        for record in _records(profile, label, name="profile")
    )


def verify(
    control_profile: Mapping[str, Any],
    candidate_profile: Mapping[str, Any],
    initialization: Mapping[str, Any],
    *,
    control_result: Path,
    candidate_result: Path,
) -> dict[str, Any]:
    _strict_runtime(control_profile, name="control")
    _strict_runtime(candidate_profile, name="candidate")
    metadata = _initialization(initialization)
    control_config = control_profile["metadata"].get("optimized_effective_config")
    candidate_config = candidate_profile["metadata"].get("optimized_effective_config")
    if control_config != candidate_config:
        raise TimingNativeSelfError("accepted optimized preset metadata changed")

    token_checks = {}
    exactness_checks = {}
    control_exactness = _strict_exactness_contract(control_profile)
    candidate_exactness = _strict_exactness_contract(candidate_profile)
    for label in LABELS:
        control_signature = nsight._profile_label_signature(control_profile, label)
        candidate_signature = nsight._profile_label_signature(candidate_profile, label)
        if control_signature.get("status") != "available" or not control_signature.get(
            "self_consistent"
        ):
            raise TimingNativeSelfError(f"control {label} token signature unavailable")
        if candidate_signature.get("status") != "available" or not candidate_signature.get(
            "self_consistent"
        ):
            raise TimingNativeSelfError(f"candidate {label} token signature unavailable")
        token_checks[label] = {
            "token_stream_sha256": control_signature["token_stream_sha256"],
            "stopping_sha256": control_signature["stopping_sha256"],
            "generated_tokens": control_signature["generated_tokens"],
            "pass": control_signature == candidate_signature,
        }
        if not token_checks[label]["pass"]:
            raise TimingNativeSelfError(f"{label} tokens or stopping changed")
        exactness_checks[label] = {
            "rng_and_cache_sha256": hashlib.sha256(
                json.dumps(
                    control_exactness[label],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "pass": control_exactness[label] == candidate_exactness[label],
        }
        if not exactness_checks[label]["pass"]:
            raise TimingNativeSelfError(f"{label} RNG progression or cache writes changed")

    control_hash = _sha256(control_result)
    candidate_hash = _sha256(candidate_result)
    if control_hash != candidate_hash:
        raise TimingNativeSelfError("final .osu is not byte-identical")
    dispatch = _dispatch_contract(control_profile, candidate_profile)
    control_timing = _label_model_seconds(control_profile, "timing_context")
    candidate_timing = _label_model_seconds(candidate_profile, "timing_context")
    return {
        "schema_version": 1,
        "candidate": metadata,
        "strict_fp32": True,
        "all_model_storage_fp32": True,
        "tf32_disabled": True,
        "counter_rng": False,
        "reduced_precision": False,
        "token_and_stopping": token_checks,
        "rng_and_cache": exactness_checks,
        "dispatch": dispatch,
        "final_osu_sha256": control_hash,
        "timing_model_elapsed_seconds": {
            "control": control_timing,
            "candidate": candidate_timing,
            "speedup_pct": (
                (control_timing - candidate_timing) / control_timing * 100.0
                if control_timing > 0
                else None
            ),
            "authoritative": False,
        },
        "pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-profile", type=Path, required=True)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--control-result", type=Path, required=True)
    parser.add_argument("--candidate-result", type=Path, required=True)
    parser.add_argument("--candidate-initialization", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(
        _read_json(args.control_profile, name="control profile"),
        _read_json(args.candidate_profile, name="candidate profile"),
        _read_json(args.candidate_initialization, name="candidate initialization"),
        control_result=args.control_result,
        candidate_result=args.candidate_result,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
