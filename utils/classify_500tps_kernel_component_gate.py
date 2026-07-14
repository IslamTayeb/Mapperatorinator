"""Fail-loud decision classifier for the serial 500-TPS component ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class GateClassificationError(ValueError):
    pass


def _require_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise GateClassificationError(f"{name} must be a JSON boolean")
    return value


def classify_gate(name: str, status: int, report: dict[str, Any]) -> None:
    if name == "cross":
        variants = report.get("variants")
        if not isinstance(variants, dict):
            raise GateClassificationError("cross variants are missing")
        fp32_checks = []
        for variant, entry in variants.items():
            if not isinstance(entry, dict):
                raise GateClassificationError(f"cross variant {variant} is not an object")
            if entry.get("kv_storage_dtype") == "torch.float32":
                fp32_checks.append(
                    _require_bool(
                        entry.get("checks_pass"),
                        name=f"cross variant {variant}.checks_pass",
                    )
                )
        if not fp32_checks or not all(fp32_checks):
            raise GateClassificationError("FP32 correctness checks failed or are missing")
        summary = report.get("summary")
        if not isinstance(summary, dict):
            raise GateClassificationError("cross summary is missing")
        promoted = _require_bool(
            summary.get("any_fp32_promotion_pass"),
            name="cross summary.any_fp32_promotion_pass",
        )
        if not ((status == 0 and promoted) or (status == 3 and not promoted)):
            raise GateClassificationError(
                f"unexpected exit/promotion pair status={status}, promoted={promoted}"
            )
        return

    if name == "conditional":
        decision = report.get("decision")
        if not isinstance(decision, dict):
            raise GateClassificationError("conditional decision is missing")
        passed = _require_bool(decision.get("pass"), name="conditional decision.pass")
        if status != 0 or not passed:
            raise GateClassificationError(
                f"conditional capability did not pass with exit status {status}"
            )
        return

    if name in {"fp16_split", "int8_mlp"}:
        summary = report.get("summary")
        if not isinstance(summary, dict):
            raise GateClassificationError(f"{name} summary is missing")
        invariants = _require_bool(
            summary.get("invariants_pass"), name=f"{name} summary.invariants_pass"
        )
        sizing = _require_bool(
            summary.get("sizing_pass"), name=f"{name} summary.sizing_pass"
        )
        if not invariants:
            raise GateClassificationError(f"{name} invariants failed")
        if not ((status == 0 and sizing) or (status == 3 and not sizing)):
            raise GateClassificationError(
                f"unexpected exit/sizing pair status={status}, sizing={sizing}"
            )
        return

    if name == "vocab":
        if status != 0:
            raise GateClassificationError(f"unexpected exit status {status}")
        ceiling = report.get("fixed_work_ceiling")
        if not isinstance(ceiling, dict):
            raise GateClassificationError("vocabulary fixed-work ceiling is missing")
        clears = _require_bool(
            ceiling.get("main_ceiling_clears_threshold"),
            name="vocabulary main_ceiling_clears_threshold",
        )
        expected = (
            "retain_for_candidate_kernel" if clears else "stop_below_main_component_gate"
        )
        if report.get("decision") != expected:
            raise GateClassificationError(
                "vocabulary decision is inconsistent with its fixed-work ceiling"
            )
        return

    if name == "shared_rope":
        summary = report.get("summary")
        if not isinstance(summary, dict):
            raise GateClassificationError("shared RoPE summary is missing")
        exact = _require_bool(summary.get("exact_pass"), name="shared RoPE exact_pass")
        accounting = _require_bool(
            summary.get("rope_call_accounting_pass"),
            name="shared RoPE rope_call_accounting_pass",
        )
        promoted = _require_bool(
            summary.get("promotion_pass"), name="shared RoPE promotion_pass"
        )
        if not exact or not accounting:
            raise GateClassificationError("shared RoPE exactness or call accounting failed")
        if not ((status == 0 and promoted) or (status == 3 and not promoted)):
            raise GateClassificationError(
                f"unexpected exit/promotion pair status={status}, promoted={promoted}"
            )
        return

    raise GateClassificationError(f"unknown gate {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--status", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise GateClassificationError("gate report must be a JSON object")
    classify_gate(args.name, args.status, report)


if __name__ == "__main__":
    main()
