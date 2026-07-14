"""Fail-loud decision classifier for the serial 500-TPS component ladder."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PROMOTION = "promotion"
VALID_NEGATIVE = "valid_negative"
RETAIN_FOR_IMPLEMENTATION = "retain_for_implementation"
REQUIRED_MAIN_SAVING_SECONDS = 1.503
CONDITIONAL_VARIANTS = ("k1", "k4", "k8", "while")
CONDITIONAL_ACCEPTED_DISPATCH = {
    "precision": "fp32",
    "q1_bmm_cross_attention": True,
    "native_q1_self_attention": True,
    "native_q1_rope_cache_self_attention": True,
}


class GateClassificationError(ValueError):
    pass


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GateClassificationError(f"{name} must be a JSON object")
    return value


def _require_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise GateClassificationError(f"{name} must be a JSON boolean")
    return value


def _number(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateClassificationError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0 or (positive and parsed <= 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise GateClassificationError(f"{name} must be finite and {qualifier}")
    return parsed


def _schema(report: dict[str, Any], expected: Any, *, name: str) -> None:
    if report.get("schema_version") != expected:
        raise GateClassificationError(
            f"{name} schema_version must be {expected!r}"
        )


def _exit_pair(
    status: int,
    promoted: bool,
    *,
    name: str,
) -> str:
    if status == 0 and promoted:
        return PROMOTION
    if status == 3 and not promoted:
        return VALID_NEGATIVE
    raise GateClassificationError(
        f"{name} has unexpected exit/promotion pair "
        f"status={status}, promoted={promoted}"
    )


def _conditional_case(case: Any, *, name: str) -> bool:
    parsed = _object(case, name=name)
    if not _require_bool(parsed.get("pass"), name=f"{name}.pass"):
        raise GateClassificationError(f"{name} invariants failed")
    for field in (
        "visible_tokens_exact",
        "logical_cache_exact",
        "logical_steps_exact",
        "physical_steps_exact",
        "while_no_post_stop_waste",
        "k1_no_post_stop_waste",
        "while_k1_full_cache_exact",
        "memory_stable",
    ):
        if not _require_bool(parsed.get(field), name=f"{name}.{field}"):
            raise GateClassificationError(f"{name}.{field} failed")
    repeatable = _object(parsed.get("repeatable"), name=f"{name}.repeatable")
    if set(repeatable) != set(CONDITIONAL_VARIANTS) or not all(
        _require_bool(repeatable[variant], name=f"{name}.repeatable.{variant}")
        for variant in CONDITIONAL_VARIANTS
    ):
        raise GateClassificationError(f"{name} repeatability failed or is incomplete")
    timings = _object(
        parsed.get("reciprocal_cuda_ms"), name=f"{name}.reciprocal_cuda_ms"
    )
    if set(timings) != set(CONDITIONAL_VARIANTS):
        raise GateClassificationError(f"{name} reciprocal timings are incomplete")
    parsed_timings = {
        variant: _number(
            timings[variant],
            name=f"{name}.reciprocal_cuda_ms.{variant}",
            positive=True,
        )
        for variant in CONDITIONAL_VARIANTS
    }
    # K4 is the runtime this scout would replace. A real-prefix promotion must
    # win both the full block and every forced-stop case, not merely prove that
    # CUDA's conditional-node API exists.
    return parsed_timings["while"] < parsed_timings["k4"]


def classify_gate(name: str, status: int, report: dict[str, Any]) -> str:
    if status < 0:
        raise GateClassificationError("child exit status must be non-negative")

    if name == "cross":
        _schema(report, 1, name="cross")
        variants = _object(report.get("variants"), name="cross.variants")
        fp32_checks = []
        for variant, raw_entry in variants.items():
            entry = _object(raw_entry, name=f"cross.variants.{variant}")
            if entry.get("kv_storage_dtype") == "torch.float32":
                fp32_checks.append(
                    _require_bool(
                        entry.get("checks_pass"),
                        name=f"cross.variants.{variant}.checks_pass",
                    )
                )
        if not fp32_checks or not all(fp32_checks):
            raise GateClassificationError("FP32 correctness checks failed or are missing")
        summary = _object(report.get("summary"), name="cross.summary")
        candidates = _object(
            summary.get("candidates"), name="cross.summary.candidates"
        )
        candidate_promotions = []
        for variant, raw_entry in candidates.items():
            entry = _object(raw_entry, name=f"cross.summary.candidates.{variant}")
            eligible = _require_bool(
                entry.get("promotion_eligible"),
                name=f"cross.summary.candidates.{variant}.promotion_eligible",
            )
            correct = _require_bool(
                entry.get("correctness_pass"),
                name=f"cross.summary.candidates.{variant}.correctness_pass",
            )
            sizing = _require_bool(
                entry.get("sizing_pass"),
                name=f"cross.summary.candidates.{variant}.sizing_pass",
            )
            promoted = _require_bool(
                entry.get("promotion_pass"),
                name=f"cross.summary.candidates.{variant}.promotion_pass",
            )
            if promoted is not (eligible and correct and sizing):
                raise GateClassificationError(
                    f"cross candidate {variant} has an inconsistent promotion decision"
                )
            candidate_promotions.append(promoted)
        promoted = _require_bool(
            summary.get("any_fp32_promotion_pass"),
            name="cross.summary.any_fp32_promotion_pass",
        )
        if promoted is not any(candidate_promotions):
            raise GateClassificationError("cross aggregate promotion is inconsistent")
        return _exit_pair(status, promoted, name="cross")

    if name == "conditional":
        _schema(report, 1, name="conditional real-prefix")
        metadata = _object(report.get("metadata"), name="conditional.metadata")
        if metadata.get("result_class") != "real-prefix-conditional-while-component-gate":
            raise GateClassificationError(
                "conditional report is not the real-prefix performance gate"
            )
        if metadata.get("production_wiring") is not False:
            raise GateClassificationError("conditional production_wiring must be false")
        if metadata.get("accepted_dispatch") != CONDITIONAL_ACCEPTED_DISPATCH:
            raise GateClassificationError("conditional accepted dispatch changed")
        decision = _object(report.get("decision"), name="conditional.decision")
        if not _require_bool(decision.get("pass"), name="conditional.decision.pass"):
            raise GateClassificationError("conditional real-prefix decision failed")
        if status != 0:
            raise GateClassificationError(
                f"conditional valid report requires exit status 0, got {status}"
            )
        performance = [
            _conditional_case(report.get("fixed_work"), name="conditional.fixed_work")
        ]
        forced = _object(report.get("forced_stop"), name="conditional.forced_stop")
        if set(forced) != {"1", "3", "7"}:
            raise GateClassificationError(
                "conditional forced_stop evidence must contain exactly 1, 3, and 7"
            )
        performance.extend(
            _conditional_case(forced[stop], name=f"conditional.forced_stop.{stop}")
            for stop in ("1", "3", "7")
        )
        return PROMOTION if all(performance) else VALID_NEGATIVE

    if name == "fp16_split":
        _schema(report, 3, name="fp16_split")
        summary = _object(report.get("summary"), name="fp16_split.summary")
        invariants = _require_bool(
            summary.get("invariants_pass"),
            name="fp16_split.summary.invariants_pass",
        )
        correctness = _require_bool(
            summary.get("correctness_pass"),
            name="fp16_split.summary.correctness_pass",
        )
        sizing = _require_bool(
            summary.get("sizing_pass"), name="fp16_split.summary.sizing_pass"
        )
        promoted = _require_bool(
            summary.get("promotion_pass"),
            name="fp16_split.summary.promotion_pass",
        )
        if not invariants or correctness is not invariants:
            raise GateClassificationError("fp16_split invariants/correctness failed")
        if promoted is not (invariants and sizing):
            raise GateClassificationError("fp16_split promotion decision is inconsistent")
        return _exit_pair(status, promoted, name="fp16_split")

    if name == "int8_mlp":
        _schema(report, 1, name="int8_mlp")
        summary = _object(report.get("summary"), name="int8_mlp.summary")
        invariants = _require_bool(
            summary.get("invariants_pass"), name="int8_mlp.summary.invariants_pass"
        )
        child_sizing = _require_bool(
            summary.get("sizing_pass"), name="int8_mlp.summary.sizing_pass"
        )
        child_promotion = _require_bool(
            summary.get("promotion_pass"), name="int8_mlp.summary.promotion_pass"
        )
        if not invariants:
            raise GateClassificationError("int8_mlp invariants failed")
        if child_promotion is not (invariants and child_sizing):
            raise GateClassificationError("int8_mlp child promotion is inconsistent")
        if not ((status == 0 and child_sizing) or (status == 3 and not child_sizing)):
            raise GateClassificationError(
                f"int8_mlp has unexpected exit/sizing pair "
                f"status={status}, sizing={child_sizing}"
            )
        saving = _number(
            summary.get("fixed_work_main_saving_seconds"),
            name="int8_mlp.summary.fixed_work_main_saving_seconds",
        )
        # The child also exposes a weaker 1.4x local-retention gate. The ladder
        # promotes only evidence that clears the common whole-main threshold.
        promoted = child_promotion and saving >= REQUIRED_MAIN_SAVING_SECONDS
        return PROMOTION if promoted else VALID_NEGATIVE

    if name == "vocab":
        _schema(report, 1, name="vocab")
        if status != 0:
            raise GateClassificationError(f"vocab unexpected exit status {status}")
        if report.get("result_class") != "verifier-only-component-sizing":
            raise GateClassificationError("vocab result_class is invalid")
        if report.get("production_wiring_changed") is not False:
            raise GateClassificationError("vocab production wiring changed")
        if report.get("authoritative_generation_timing") is not False:
            raise GateClassificationError("vocab timing is mislabeled authoritative")
        if report.get("rng_exact_with_k4_control") is not True:
            raise GateClassificationError("vocab K4 RNG control is not exact")
        component = _object(report.get("component_timing"), name="vocab.component_timing")
        rows = component.get("representatives")
        if not isinstance(rows, list) or not rows:
            raise GateClassificationError("vocab representatives are missing")
        for index, raw_row in enumerate(rows):
            row = _object(raw_row, name=f"vocab.representatives[{index}]")
            for field in ("selected_token_exact", "top_p_mask_exact"):
                if not _require_bool(
                    row.get(field), name=f"vocab.representatives[{index}].{field}"
                ):
                    raise GateClassificationError(
                        f"vocab representatives[{index}].{field} failed"
                    )
        ceiling = _object(
            report.get("fixed_work_ceiling"), name="vocab.fixed_work_ceiling"
        )
        ideal = _number(
            ceiling.get("ideal_main_total_seconds"),
            name="vocab.fixed_work_ceiling.ideal_main_total_seconds",
        )
        threshold = _number(
            ceiling.get("promotion_threshold_seconds"),
            name="vocab.fixed_work_ceiling.promotion_threshold_seconds",
        )
        clears = _require_bool(
            ceiling.get("main_ceiling_clears_threshold"),
            name="vocab.fixed_work_ceiling.main_ceiling_clears_threshold",
        )
        if clears is not (ideal >= threshold):
            raise GateClassificationError("vocab ceiling decision is inconsistent")
        expected = (
            "retain_for_candidate_kernel" if clears else "stop_below_main_component_gate"
        )
        if report.get("decision") != expected:
            raise GateClassificationError(
                "vocab decision is inconsistent with its fixed-work ceiling"
            )
        return RETAIN_FOR_IMPLEMENTATION if clears else VALID_NEGATIVE

    if name == "shared_rope":
        _schema(report, "mapperatorinator.shared-rope-scout.v1", name="shared_rope")
        summary = _object(report.get("summary"), name="shared_rope.summary")
        exact = _require_bool(
            summary.get("exact_pass"), name="shared_rope.summary.exact_pass"
        )
        accounting = _require_bool(
            summary.get("rope_call_accounting_pass"),
            name="shared_rope.summary.rope_call_accounting_pass",
        )
        performance = _require_bool(
            summary.get("performance_pass"),
            name="shared_rope.summary.performance_pass",
        )
        saving = _number(
            summary.get("projected_main_saving_seconds"),
            name="shared_rope.summary.projected_main_saving_seconds",
        )
        required = _number(
            summary.get("required_main_saving_seconds"),
            name="shared_rope.summary.required_main_saving_seconds",
        )
        if performance is not (saving >= required):
            raise GateClassificationError("shared_rope performance decision is inconsistent")
        promoted = _require_bool(
            summary.get("promotion_pass"), name="shared_rope.summary.promotion_pass"
        )
        if promoted is not (exact and accounting and performance):
            raise GateClassificationError("shared_rope promotion decision is inconsistent")
        if not exact or not accounting:
            raise GateClassificationError("shared_rope exactness or accounting failed")
        return _exit_pair(status, promoted, name="shared_rope")

    if name == "prefill":
        _schema(report, 1, name="prefill")
        variants = _object(report.get("variants"), name="prefill.variants")
        expected_variants = {
            "exact_fp16_graph",
            "exact_fp32_graph",
            "bucket64_fp16_graph",
            "bucket64_fp32_graph",
        }
        if set(variants) != expected_variants:
            raise GateClassificationError("prefill variants are incomplete")
        variant_promotions = []
        for variant, raw_entry in variants.items():
            entry = _object(raw_entry, name=f"prefill.variants.{variant}")
            correctness = _require_bool(
                entry.get("correctness_pass"),
                name=f"prefill.variants.{variant}.correctness_pass",
            )
            performance = _require_bool(
                entry.get("performance_pass"),
                name=f"prefill.variants.{variant}.performance_pass",
            )
            promoted = _require_bool(
                entry.get("promotion_pass"),
                name=f"prefill.variants.{variant}.promotion_pass",
            )
            if promoted is not (correctness and performance):
                raise GateClassificationError(
                    f"prefill variant {variant} promotion is inconsistent"
                )
            variant_promotions.append(promoted)
        promoted = _require_bool(
            report.get("promotion_pass"), name="prefill.promotion_pass"
        )
        if promoted is not any(variant_promotions):
            raise GateClassificationError("prefill aggregate promotion is inconsistent")
        expected_decision = "PROMOTE_PREFILL_SCOUT" if promoted else "STOP_PREFILL_SCOUT"
        if report.get("decision") != expected_decision:
            raise GateClassificationError("prefill decision is inconsistent")
        return _exit_pair(status, promoted, name="prefill")

    raise GateClassificationError(f"unknown gate {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--status", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise GateClassificationError("gate report must be a JSON object")
    outcome = classify_gate(args.name, args.status, report)
    decision = {
        "schema_version": "mapperatorinator.500tps-kernel-ladder-decision.v1",
        "gate": args.name,
        "child_exit_status": args.status,
        "outcome": outcome,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(outcome)


if __name__ == "__main__":
    main()
