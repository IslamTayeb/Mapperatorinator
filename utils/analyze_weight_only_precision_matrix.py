"""Compare the mixed-weight candidate with FP32 reciprocal and accepted FP16 runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

try:
    from utils.analyze_reciprocal_full_song_candidate import (
        PROFILE_LABELS,
        SCHEMA_VERSION as RECIPROCAL_SCHEMA_VERSION,
        CandidateAnalysisError,
        ParsedRun,
        _material_differences,
        _parse_run,
        _sha256_file,
        _structure_divergence,
        _token_divergence,
    )
    from utils.fixed_seed_inference import SEED_POLICY_VERSION
except ModuleNotFoundError:  # Direct ``python utils/...py`` execution.
    from analyze_reciprocal_full_song_candidate import (  # type: ignore[no-redef]
        PROFILE_LABELS,
        SCHEMA_VERSION as RECIPROCAL_SCHEMA_VERSION,
        CandidateAnalysisError,
        ParsedRun,
        _material_differences,
        _parse_run,
        _sha256_file,
        _structure_divergence,
        _token_divergence,
    )
    from fixed_seed_inference import SEED_POLICY_VERSION  # type: ignore[no-redef]


SCHEMA_VERSION = "mapperatorinator.weight-only-precision-matrix.v1"
CANDIDATE_ROLES = ("candidate_first", "candidate_second")
FP16_ROLES = ("accepted_fp16_first", "accepted_fp16_second")
ALL_ROLES = CANDIDATE_ROLES + FP16_ROLES
FIXED_WORK_TOKENS = 8_294
FULL_SONG_SAVING_TARGET_SECONDS = 3.5


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CandidateAnalysisError(f"{name} must be an object")
    return value


def _load_reciprocal(value: Mapping[str, Any] | Path) -> dict[str, Any]:
    if isinstance(value, Path):
        try:
            payload = json.loads(value.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CandidateAnalysisError(
                f"cannot read reciprocal report {value}: {exc}"
            ) from exc
    else:
        payload = dict(value)
    report = _object(payload, name="reciprocal_report")
    if report.get("schema_version") != RECIPROCAL_SCHEMA_VERSION:
        raise CandidateAnalysisError("reciprocal report schema is unsupported")
    if report.get("mode") != "relaxed":
        raise CandidateAnalysisError("weight-only matrix requires a relaxed reciprocal report")
    parity = _object(report.get("parity"), name="reciprocal_report.parity")
    if parity.get("baseline_repeat_stable") is not True:
        raise CandidateAnalysisError("FP32 baseline reciprocal pair is not stable")
    if parity.get("candidate_repeat_stable") is not True:
        raise CandidateAnalysisError("mixed-weight reciprocal pair is not stable")
    return report


def _require_pair_stable(
    runs: Mapping[str, ParsedRun], roles: tuple[str, str], *, name: str
) -> dict[str, Any]:
    first, second = (runs[role] for role in roles)
    if first.precision != second.precision:
        raise CandidateAnalysisError(f"{name} precision differs between repetitions")
    if first.stage_sequence != second.stage_sequence:
        raise CandidateAnalysisError(f"{name} stage sequence differs between repetitions")
    token_equal: dict[str, bool] = {}
    stopping_equal: dict[str, bool] = {}
    for label in PROFILE_LABELS:
        token_equal[label] = (
            first.label_signatures[label]["token_stream_sha256"]
            == second.label_signatures[label]["token_stream_sha256"]
        )
        stopping_equal[label] = (
            first.label_signatures[label]["stopping_sha256"]
            == second.label_signatures[label]["stopping_sha256"]
        )
    if not all(token_equal.values()):
        raise CandidateAnalysisError(f"{name} token stream is not repeat-stable")
    if not all(stopping_equal.values()):
        raise CandidateAnalysisError(f"{name} stopping is not repeat-stable")
    if first.osu_sha256 != second.osu_sha256:
        raise CandidateAnalysisError(f"{name} final OSU is not repeat-stable")
    if first.graph_signature["sha256"] != second.graph_signature["sha256"]:
        raise CandidateAnalysisError(f"{name} dispatch/cache signature is not repeat-stable")
    rng_equal: dict[str, bool] = {}
    for label in PROFILE_LABELS:
        for device in ("cpu", "cuda"):
            key = f"reciprocal_rng_{device}_sha256_{label}"
            left = first.metadata.get(key)
            right = second.metadata.get(key)
            valid = (
                isinstance(left, str)
                and len(left) == 64
                and left == right
            )
            rng_equal[f"{device}.{label}"] = valid
    if not all(rng_equal.values()):
        raise CandidateAnalysisError(f"{name} RNG fingerprints are not repeat-stable")
    for run in (first, second):
        if run.metadata.get("reciprocal_seed_policy") != SEED_POLICY_VERSION:
            raise CandidateAnalysisError(f"{name} seed policy is missing or wrong")
    return {
        "token_stream_equal": token_equal,
        "stopping_equal": stopping_equal,
        "final_osu_equal": True,
        "dispatch_cache_equal": True,
        "rng_fingerprints_equal": rng_equal,
    }


def _median_metrics(
    runs: Mapping[str, ParsedRun], roles: tuple[str, str]
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in runs[roles[0]].metrics:
        values = [runs[role].metrics[key] for role in roles]
        metrics[key] = float(statistics.median(values))
    main_tps = metrics["main_tps"]
    if not math.isfinite(main_tps) or main_tps <= 0.0:
        raise CandidateAnalysisError("main TPS must be positive and finite")
    metrics["fixed_work_tokens"] = float(FIXED_WORK_TOKENS)
    metrics["fixed_work_main_seconds"] = FIXED_WORK_TOKENS / main_tps
    return metrics


def _reciprocal_group_metrics(report: Mapping[str, Any], name: str) -> dict[str, float]:
    group_key = "baseline_median" if name == "fp32_reference" else "candidate_median"
    metrics = _object(report.get("metrics"), name="reciprocal_report.metrics")
    result = {
        "main_model_seconds": float(
            _object(metrics.get("main_model_seconds"), name="main_model_seconds")[group_key]
        ),
        "main_tokens": float(
            _object(metrics.get("main_tokens"), name="main_tokens")[group_key]
        ),
        "main_tps": float(_object(metrics.get("main_tps"), name="main_tps")[group_key]),
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in result.values()):
        raise CandidateAnalysisError(f"{name} reciprocal metrics are invalid")
    result["fixed_work_tokens"] = float(FIXED_WORK_TOKENS)
    result["fixed_work_main_seconds"] = FIXED_WORK_TOKENS / result["main_tps"]
    return result


def analyze(
    profile_paths: Mapping[str, Path],
    reciprocal_report: Mapping[str, Any] | Path,
) -> dict[str, Any]:
    if set(profile_paths) != set(ALL_ROLES):
        raise CandidateAnalysisError(f"profile roles must be exactly {list(ALL_ROLES)}")
    reciprocal = _load_reciprocal(reciprocal_report)
    runs = {
        role: _parse_run(role, profile_paths[role], None)
        for role in ALL_ROLES
    }
    if any(runs[role].precision != "fp32" for role in CANDIDATE_ROLES):
        raise CandidateAnalysisError("mixed-weight candidate profiles must report fp32 state")
    if any(runs[role].precision != "fp16" for role in FP16_ROLES):
        raise CandidateAnalysisError("accepted FP16 profiles must report fp16")

    reciprocal_runs = _object(reciprocal.get("runs"), name="reciprocal_report.runs")
    for role in CANDIDATE_ROLES:
        reciprocal_run = _object(reciprocal_runs.get(role), name=f"runs.{role}")
        if reciprocal_run.get("profile_sha256") != _sha256_file(runs[role].profile_path):
            raise CandidateAnalysisError(
                f"{role} profile does not match the reciprocal report"
            )

    repeatability = {
        "mixed_weight": _require_pair_stable(
            runs, CANDIDATE_ROLES, name="mixed-weight candidate"
        ),
        "accepted_fp16": _require_pair_stable(
            runs, FP16_ROLES, name="accepted FP16"
        ),
    }
    candidate = runs[CANDIDATE_ROLES[0]]
    accepted_fp16 = runs[FP16_ROLES[0]]
    cross_workload_differences = _material_differences(
        candidate.workload, accepted_fp16.workload
    )
    unexpected_workload = [
        difference["path"]
        for difference in cross_workload_differences
        if difference["path"] != "precision"
    ]
    if unexpected_workload:
        raise CandidateAnalysisError(
            "mixed-weight/accepted-FP16 workload differs outside precision: "
            f"{unexpected_workload}"
        )

    groups = {
        "fp32_reference": _reciprocal_group_metrics(reciprocal, "fp32_reference"),
        "mixed_weight": _reciprocal_group_metrics(reciprocal, "mixed_weight"),
        "accepted_fp16": _median_metrics(runs, FP16_ROLES),
    }
    fp32_fixed = groups["fp32_reference"]["fixed_work_main_seconds"]
    mixed_fixed = groups["mixed_weight"]["fixed_work_main_seconds"]
    fixed_saving = fp32_fixed - mixed_fixed
    performance = {
        "comparison_policy": (
            "FP32 reference versus mixed-weight uses reciprocal medians; accepted FP16 "
            "is descriptive because its two repetitions run after the reciprocal ladder"
        ),
        "groups": groups,
        "mixed_weight_fixed_work_saving_vs_fp32_seconds": fixed_saving,
        "mixed_weight_fixed_work_saving_target_seconds": FULL_SONG_SAVING_TARGET_SECONDS,
        "mixed_weight_meets_full_song_saving_target": (
            fixed_saving >= FULL_SONG_SAVING_TARGET_SECONDS
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "result_class": "documented-drift",
        "exactness_claim": False,
        "fixed_work_tokens": FIXED_WORK_TOKENS,
        "repeatability": repeatability,
        "performance": performance,
        "mixed_weight_vs_accepted_fp16": {
            "workload_differences": cross_workload_differences,
            "token_and_stopping_divergence": {
                label: _token_divergence(candidate, accepted_fp16, label)
                for label in PROFILE_LABELS
            },
            "output_divergence": _structure_divergence(candidate, accepted_fp16),
            "dispatch_cache_differences": _material_differences(
                candidate.graph_signature["material"],
                accepted_fp16.graph_signature["material"],
            ),
        },
        "runs": {
            role: {
                "profile_path": str(run.profile_path),
                "profile_sha256": _sha256_file(run.profile_path),
                "osu_path": str(run.osu_path),
                "osu_sha256": run.osu_sha256,
                "precision": run.precision,
                "metrics": run.metrics,
            }
            for role, run in runs.items()
        },
    }


def text_report(report: Mapping[str, Any]) -> str:
    performance = report["performance"]
    groups = performance["groups"]
    lines = [
        f"result_class={report['result_class']}",
        f"exactness_claim={str(report['exactness_claim']).lower()}",
        f"fixed_work_tokens={report['fixed_work_tokens']}",
    ]
    for name in ("fp32_reference", "mixed_weight", "accepted_fp16"):
        metrics = groups[name]
        lines.append(
            f"group.{name}=main_seconds:{metrics['main_model_seconds']:.9f},"
            f"main_tokens:{metrics['main_tokens']:.0f},"
            f"main_tps:{metrics['main_tps']:.9f},"
            f"fixed_work_main_seconds:{metrics['fixed_work_main_seconds']:.9f}"
        )
    lines.append(
        "mixed_weight.fixed_work_saving_vs_fp32_seconds="
        f"{performance['mixed_weight_fixed_work_saving_vs_fp32_seconds']:.9f}"
    )
    lines.append(
        "mixed_weight.meets_3_5s_target="
        f"{str(performance['mixed_weight_meets_full_song_saving_target']).lower()}"
    )
    divergence = report["mixed_weight_vs_accepted_fp16"]
    for label, values in divergence["token_and_stopping_divergence"].items():
        lines.append(
            f"mixed_weight_vs_fp16.{label}=candidate_tokens:{values['baseline_tokens']},"
            f"fp16_tokens:{values['candidate_tokens']},"
            f"aligned_mismatch_fraction:{values['aligned_mismatch_fraction']:.9f},"
            f"stopping_equal:{str(values['stopping_equal']).lower()}"
        )
    lines.append(
        "mixed_weight_vs_fp16.final_map_equal="
        f"{str(divergence['output_divergence']['final_map_equal']).lower()}"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    for role in ALL_ROLES:
        parser.add_argument(
            f"--{role.replace('_', '-')}-profile", type=Path, required=True
        )
    parser.add_argument("--reciprocal-report", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        {role: getattr(args, f"{role}_profile") for role in ALL_ROLES},
        args.reciprocal_report,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.text_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.text_output.write_text(text_report(report), encoding="utf-8")


if __name__ == "__main__":
    main()
